//! Turns a generated character strip into a uniform sprite sheet.
//!
//! Figures are sliced at fully transparent gaps (rows first, then columns),
//! keep their vertical position against the row's common baseline, share one
//! scale factor, and are packed left to right into `frames` cells.

use crate::error::{GenVizError, Result};
use ::image::{imageops, Rgba, RgbaImage};
use std::collections::VecDeque;

/// Alpha at or below this counts as empty when looking for gaps.
const ALPHA_EMPTY: u8 = 16;
/// Max per-channel distance from a corner colour that `Key::Auto` treats as background.
const KEY_TOLERANCE: u8 = 40;
/// Islands with less ink than this fraction of the median figure merge into a neighbour.
const TINY_ISLAND: f64 = 0.2;

/// Background handling for inputs without transparency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// Flood-fill a flat background from the corners when the input has no transparent pixels.
    Auto,
    /// Use the alpha channel as is.
    None,
}

/// Sprite packing options.
#[derive(Debug, Clone, Copy)]
pub struct SpriteOptions {
    /// Number of figures expected in the strip.
    pub frames: usize,
    /// Number of rows the figures are laid out in (row-major).
    pub rows: usize,
    /// Output cell width in pixels.
    pub cell_width: u32,
    /// Output cell height in pixels.
    pub cell_height: u32,
    /// Background keying mode.
    pub key: Key,
}

/// A packed sprite strip and its quality signals.
#[derive(Debug)]
pub struct Sprite {
    /// `frames * cell_width` by `cell_height` RGBA strip.
    pub image: RgbaImage,
    /// Difference between first and last frame, 0 (identical) to 1.
    pub loop_delta: f64,
    /// Relative variation of figure heights, 0 (all equal) to 1.
    pub height_spread: f64,
}

#[derive(Debug, Clone, Copy)]
struct Run {
    start: u32,
    end: u32,
    mass: u64,
}

struct Figure {
    x0: u32,
    x1: u32,
    top: u32,
    bottom: u32,
    centre: f64,
    row: usize,
}

/// Slices, aligns, scales and packs the figures in `strip`.
pub fn pack(strip: &RgbaImage, opts: SpriteOptions) -> Result<Sprite> {
    if opts.frames == 0 || opts.rows == 0 || opts.cell_width == 0 || opts.cell_height == 0 {
        return Err(GenVizError::InvalidRequest(
            "frames, rows and cell must be positive".into(),
        ));
    }
    let mut img = strip.clone();
    if opts.key == Key::Auto && img.pixels().all(|p| p.0[3] == 255) {
        key_corners(&mut img);
    }
    let ink = |x: u32, y: u32| img.get_pixel(x, y).0[3] > ALPHA_EMPTY;

    let row_profile: Vec<u64> = (0..img.height())
        .map(|y| (0..img.width()).filter(|&x| ink(x, y)).count() as u64)
        .collect();
    let mut bands = runs(&row_profile);
    while bands.len() > opts.rows {
        let smallest = (0..bands.len()).min_by_key(|&i| bands[i].mass).unwrap_or(0);
        merge_into_nearest(&mut bands, smallest);
    }
    if bands.len() != opts.rows {
        return Err(GenVizError::InvalidRequest(format!(
            "found {} row(s) of figures, expected {}",
            bands.len(),
            opts.rows
        )));
    }

    let mut figures = Vec::new();
    for (row, band) in bands.iter().enumerate() {
        let col_profile: Vec<u64> = (0..img.width())
            .map(|x| (band.start..band.end).filter(|&y| ink(x, y)).count() as u64)
            .collect();
        let mut cols = runs(&col_profile);
        loop {
            let mut masses: Vec<u64> = cols.iter().map(|r| r.mass).collect();
            masses.sort_unstable();
            let median = masses.get(masses.len() / 2).copied().unwrap_or(0) as f64;
            let tiny = (0..cols.len())
                .filter(|&i| (cols[i].mass as f64) < median * TINY_ISLAND)
                .min_by_key(|&i| cols[i].mass);
            match tiny {
                Some(i) if cols.len() > 1 => merge_into_nearest(&mut cols, i),
                _ => break,
            }
        }
        for c in cols {
            let has_ink = |y: &u32| (c.start..c.end).any(|x| ink(x, *y));
            let top = (band.start..band.end).find(has_ink).unwrap_or(band.start);
            let bottom = (band.start..band.end)
                .rfind(has_ink)
                .map_or(band.end, |y| y + 1);
            let (mut sum, mut weight) = (0.0, 0.0);
            for x in c.start..c.end {
                for y in band.start..band.end {
                    let a = img.get_pixel(x, y).0[3] as f64;
                    sum += a * (x as f64 + 0.5);
                    weight += a;
                }
            }
            figures.push(Figure {
                x0: c.start,
                x1: c.end,
                top,
                bottom,
                centre: if weight > 0.0 {
                    sum / weight
                } else {
                    (c.start + c.end) as f64 / 2.0
                },
                row,
            });
        }
    }
    if figures.len() != opts.frames {
        return Err(GenVizError::InvalidRequest(format!(
            "found {} figure(s), expected {}",
            figures.len(),
            opts.frames
        )));
    }

    let row_top: Vec<u32> = (0..opts.rows)
        .map(|r| {
            figures
                .iter()
                .filter(|f| f.row == r)
                .map(|f| f.top)
                .min()
                .unwrap_or(0)
        })
        .collect();
    let row_bottom: Vec<u32> = (0..opts.rows)
        .map(|r| {
            figures
                .iter()
                .filter(|f| f.row == r)
                .map(|f| f.bottom)
                .max()
                .unwrap_or(0)
        })
        .collect();
    let need_w = figures
        .iter()
        .map(|f| 2.0 * (f.centre - f.x0 as f64).max(f.x1 as f64 - f.centre))
        .fold(1.0, f64::max);
    let need_h = (0..opts.rows)
        .map(|r| (row_bottom[r] - row_top[r]) as f64)
        .fold(1.0, f64::max);
    let scale = (opts.cell_width as f64 / need_w).min(opts.cell_height as f64 / need_h);

    premultiply(&mut img);
    let mut out = RgbaImage::new(opts.cell_width * opts.frames as u32, opts.cell_height);
    for (i, f) in figures.iter().enumerate() {
        let (top, bottom) = (row_top[f.row], row_bottom[f.row]);
        let crop = imageops::crop_imm(&img, f.x0, top, f.x1 - f.x0, bottom - top).to_image();
        let w = ((crop.width() as f64 * scale).round() as u32).max(1);
        let h = ((crop.height() as f64 * scale).round() as u32).max(1);
        let scaled = imageops::resize(&crop, w, h, imageops::FilterType::Lanczos3);
        let cell_x = (i as u32 * opts.cell_width) as f64;
        let x = cell_x + opts.cell_width as f64 / 2.0 - (f.centre - f.x0 as f64) * scale;
        let y = opts.cell_height as i64 - h as i64;
        let mut cell = RgbaImage::new(opts.cell_width, opts.cell_height);
        imageops::replace(&mut cell, &scaled, (x - cell_x).round() as i64, y);
        imageops::replace(&mut out, &cell, cell_x as i64, 0);
    }
    unpremultiply(&mut out);

    let heights: Vec<f64> = figures.iter().map(|f| (f.bottom - f.top) as f64).collect();
    let max_h = heights.iter().cloned().fold(0.0, f64::max);
    let min_h = heights.iter().cloned().fold(f64::INFINITY, f64::min);
    Ok(Sprite {
        loop_delta: frame_delta(&out, opts.cell_width, 0, opts.frames - 1),
        height_spread: if max_h > 0.0 {
            (max_h - min_h) / max_h
        } else {
            0.0
        },
        image: out,
    })
}

fn runs(profile: &[u64]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for (i, &v) in profile.iter().enumerate() {
        let i = i as u32;
        match out.last_mut() {
            Some(r) if v > 0 && r.end == i => {
                r.end = i + 1;
                r.mass += v;
            }
            _ if v > 0 => out.push(Run {
                start: i,
                end: i + 1,
                mass: v,
            }),
            _ => {}
        }
    }
    out
}

fn merge_into_nearest(runs: &mut Vec<Run>, i: usize) {
    let gap_left = (i > 0).then(|| runs[i].start - runs[i - 1].end);
    let gap_right = runs.get(i + 1).map(|r| r.start - runs[i].end);
    let j = match (gap_left, gap_right) {
        (Some(l), Some(r)) if r < l => i + 1,
        (Some(_), _) => i - 1,
        (None, Some(_)) => i + 1,
        (None, None) => return,
    };
    let (a, b) = (i.min(j), i.max(j));
    runs[a] = Run {
        start: runs[a].start,
        end: runs[b].end,
        mass: runs[a].mass + runs[b].mass,
    };
    runs.remove(b);
}

fn key_corners(img: &mut RgbaImage) {
    let (w, h) = img.dimensions();
    let mut seen = vec![false; (w * h) as usize];
    for (cx, cy) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
        let seed = *img.get_pixel(cx, cy);
        let mut queue = VecDeque::from([(cx, cy)]);
        while let Some((x, y)) = queue.pop_front() {
            let idx = (y * w + x) as usize;
            if seen[idx] || !close(img.get_pixel(x, y), &seed) {
                continue;
            }
            seen[idx] = true;
            img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            if x > 0 {
                queue.push_back((x - 1, y));
            }
            if y > 0 {
                queue.push_back((x, y - 1));
            }
            if x + 1 < w {
                queue.push_back((x + 1, y));
            }
            if y + 1 < h {
                queue.push_back((x, y + 1));
            }
        }
    }
}

fn close(p: &Rgba<u8>, seed: &Rgba<u8>) -> bool {
    p.0[3] != 0 && (0..3).all(|c| p.0[c].abs_diff(seed.0[c]) <= KEY_TOLERANCE)
}

fn premultiply(img: &mut RgbaImage) {
    for p in img.pixels_mut() {
        let a = p.0[3] as u32;
        for c in 0..3 {
            p.0[c] = ((p.0[c] as u32 * a + 127) / 255) as u8;
        }
    }
}

fn unpremultiply(img: &mut RgbaImage) {
    for p in img.pixels_mut() {
        let a = p.0[3] as u32;
        for c in 0..3 {
            p.0[c] = (p.0[c] as u32 * 255 + a / 2)
                .checked_div(a)
                .map_or(0, |v| v.min(255) as u8);
        }
    }
}

fn frame_delta(strip: &RgbaImage, cell_w: u32, a: usize, b: usize) -> f64 {
    let (mut diff, mut count) = (0u64, 0u64);
    for y in 0..strip.height() {
        for x in 0..cell_w {
            let p = strip.get_pixel(a as u32 * cell_w + x, y).0;
            let q = strip.get_pixel(b as u32 * cell_w + x, y).0;
            if p[3] > 0 || q[3] > 0 {
                diff += (0..4).map(|c| p[c].abs_diff(q[c]) as u64).sum::<u64>();
                count += 4;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        diff as f64 / (count as f64 * 255.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(img: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, colour: [u8; 4]) {
        for yy in y..y + h {
            for xx in x..x + w {
                img.put_pixel(xx, yy, Rgba(colour));
            }
        }
    }

    #[test]
    fn packs_synthetic_strip_keeping_the_jump() {
        let red = [200, 30, 30, 255];
        let mut strip = RgbaImage::new(200, 60);
        rect(&mut strip, 10, 20, 20, 40, red);
        rect(&mut strip, 60, 10, 20, 40, red);
        rect(&mut strip, 110, 20, 20, 40, red);
        rect(&mut strip, 160, 20, 20, 40, red);
        rect(&mut strip, 133, 15, 2, 2, red);
        let opts = SpriteOptions {
            frames: 4,
            rows: 1,
            cell_width: 50,
            cell_height: 50,
            key: Key::None,
        };
        let sprite = pack(&strip, opts).unwrap();

        assert_eq!(sprite.image.dimensions(), (200, 50));
        let bottom_ink = |frame: u32| {
            (0..50)
                .rev()
                .find(|&y| (0..50).any(|x| sprite.image.get_pixel(frame * 50 + x, y).0[3] > 128))
                .unwrap()
        };
        assert!(
            bottom_ink(1) + 5 < bottom_ink(0),
            "jumping frame stays lifted"
        );
        assert_eq!(bottom_ink(0), 49);
        assert!(sprite.loop_delta < 0.01);
        assert!(sprite.height_spread > 0.0 && sprite.height_spread < 0.2);

        let three = SpriteOptions { frames: 3, ..opts };
        assert!(pack(&strip, three).is_err());

        let mut opaque = RgbaImage::from_pixel(200, 60, Rgba([250, 250, 250, 255]));
        rect(&mut opaque, 10, 20, 20, 40, red);
        rect(&mut opaque, 60, 10, 20, 40, red);
        rect(&mut opaque, 110, 20, 20, 40, red);
        rect(&mut opaque, 160, 20, 20, 40, red);
        assert!(pack(
            &opaque,
            SpriteOptions {
                key: Key::Auto,
                ..opts
            }
        )
        .is_ok());
        assert!(pack(&opaque, opts).is_err());
    }
}
