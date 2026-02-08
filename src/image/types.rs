//! Core types for image generation.

use crate::error::{GenVizError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Supported image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    /// PNG format (lossless).
    #[default]
    Png,
    /// JPEG format (lossy).
    Jpeg,
    /// WebP format (modern, efficient).
    WebP,
}

impl ImageFormat {
    /// Returns the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::WebP => "webp",
        }
    }

    /// Returns the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::WebP => "image/webp",
        }
    }

    /// Attempts to detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "webp" => Some(Self::WebP),
            _ => None,
        }
    }

    /// Detects image format from magic bytes.
    pub fn from_magic_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return Some(Self::Png);
        }

        // JPEG: FF D8 FF
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return Some(Self::Jpeg);
        }

        // WebP: RIFF....WEBP
        if data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
            return Some(Self::WebP);
        }

        None
    }

    /// Checks if the given data matches this format's magic bytes.
    pub fn matches_bytes(&self, data: &[u8]) -> bool {
        Self::from_magic_bytes(data) == Some(*self)
    }
}

/// Image provider kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageProviderKind {
    /// Black Forest Labs Flux models.
    Flux,
    /// Google Gemini image models.
    Gemini,
    /// xAI Grok Imagine models.
    Grok,
    /// OpenAI image models (GPT Image, DALL-E).
    OpenAI,
}

impl std::fmt::Display for ImageProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flux => write!(f, "flux"),
            Self::Gemini => write!(f, "gemini"),
            Self::Grok => write!(f, "grok"),
            Self::OpenAI => write!(f, "openai"),
        }
    }
}

/// Common aspect ratios for image generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectRatio {
    /// 1:1 square aspect ratio.
    #[serde(rename = "1:1")]
    Square,
    /// 16:9 landscape (widescreen) aspect ratio.
    #[serde(rename = "16:9")]
    Landscape,
    /// 9:16 portrait (tall) aspect ratio.
    #[serde(rename = "9:16")]
    Portrait,
    /// 4:3 standard landscape aspect ratio.
    #[serde(rename = "4:3")]
    Standard,
    /// 3:4 standard portrait aspect ratio.
    #[serde(rename = "3:4")]
    StandardPortrait,
    /// 21:9 ultrawide aspect ratio.
    #[serde(rename = "21:9")]
    Ultrawide,
}

impl AspectRatio {
    /// Returns the aspect ratio as a string (e.g., "16:9").
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Square => "1:1",
            Self::Landscape => "16:9",
            Self::Portrait => "9:16",
            Self::Standard => "4:3",
            Self::StandardPortrait => "3:4",
            Self::Ultrawide => "21:9",
        }
    }
}

impl std::fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Metadata about the generation process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Model used for generation.
    pub model: Option<String>,
    /// Seed used (if deterministic).
    pub seed: Option<u64>,
    /// Generation duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Whether safety filters were applied.
    pub safety_filtered: bool,
}

/// A request to generate an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// The text prompt describing the desired image.
    pub prompt: String,
    /// Desired width in pixels.
    pub width: Option<u32>,
    /// Desired height in pixels.
    pub height: Option<u32>,
    /// Seed for deterministic generation.
    pub seed: Option<u64>,
    /// Aspect ratio (alternative to width/height).
    pub aspect_ratio: Option<AspectRatio>,
    /// Desired output format.
    pub format: Option<ImageFormat>,
    /// Input image for editing/inpainting (raw bytes).
    /// Supported by Gemini and Flux, not Grok.
    #[serde(skip)]
    pub input_image: Option<Vec<u8>>,
}

impl GenerationRequest {
    /// Creates a new request with the given prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            width: None,
            height: None,
            seed: None,
            aspect_ratio: None,
            format: None,
            input_image: None,
        }
    }

    /// Sets the desired dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Sets the seed for deterministic generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the aspect ratio.
    pub fn with_aspect_ratio(mut self, ratio: AspectRatio) -> Self {
        self.aspect_ratio = Some(ratio);
        self
    }

    /// Sets the desired output format.
    pub fn with_format(mut self, format: ImageFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets an input image for editing/inpainting.
    /// Supported by Gemini and Flux providers.
    pub fn with_input_image(mut self, image: Vec<u8>) -> Self {
        self.input_image = Some(image);
        self
    }

    /// Returns true if this is an image editing request (has input image).
    pub fn is_edit(&self) -> bool {
        self.input_image.is_some()
    }
}

/// A generated image with its data and metadata.
#[derive(Debug, Clone)]
#[must_use = "generated image should be saved or processed"]
pub struct GeneratedImage {
    /// Raw image bytes.
    pub data: Vec<u8>,
    /// Image format.
    pub format: ImageFormat,
    /// Provider that generated this image.
    pub provider: ImageProviderKind,
    /// Generation metadata.
    pub metadata: GenerationMetadata,
}

impl GeneratedImage {
    /// Creates a new generated image.
    pub fn new(
        data: Vec<u8>,
        format: ImageFormat,
        provider: ImageProviderKind,
        metadata: GenerationMetadata,
    ) -> Self {
        Self {
            data,
            format,
            provider,
            metadata,
        }
    }

    /// Creates a new generated image, detecting format from magic bytes.
    pub fn from_bytes(
        data: Vec<u8>,
        provider: ImageProviderKind,
        metadata: GenerationMetadata,
    ) -> Result<Self> {
        let format = ImageFormat::from_magic_bytes(&data)
            .ok_or_else(|| GenVizError::Decode("Unknown image format".into()))?;
        Ok(Self::new(data, format, provider, metadata))
    }

    /// Validates that the image data matches the claimed format.
    pub fn validate_format(&self) -> bool {
        self.format.matches_bytes(&self.data)
    }

    /// Returns the actual format detected from magic bytes.
    pub fn detected_format(&self) -> Option<ImageFormat> {
        ImageFormat::from_magic_bytes(&self.data)
    }

    /// Returns the size of the image data in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Saves the image to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, &self.data)?;
        Ok(())
    }

    /// Encodes the image data as base64.
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(&self.data)
    }

    /// Returns the image as a data URL.
    pub fn to_data_url(&self) -> String {
        format!(
            "data:{};base64,{}",
            self.format.mime_type(),
            self.to_base64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG_MAGIC: [u8; 12] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
    const JPEG_MAGIC: [u8; 12] = [0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0];
    const WEBP_MAGIC: [u8; 12] = *b"RIFF\x00\x00\x00\x00WEBP";

    #[test]
    fn test_format_from_magic_bytes() {
        assert_eq!(
            ImageFormat::from_magic_bytes(&PNG_MAGIC),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            ImageFormat::from_magic_bytes(&JPEG_MAGIC),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            ImageFormat::from_magic_bytes(&WEBP_MAGIC),
            Some(ImageFormat::WebP)
        );
    }

    #[test]
    fn test_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::from_extension("jpg"), Some(ImageFormat::Jpeg));
        assert_eq!(ImageFormat::from_extension("webp"), Some(ImageFormat::WebP));
    }

    #[test]
    fn test_aspect_ratio_as_str() {
        assert_eq!(AspectRatio::Square.as_str(), "1:1");
        assert_eq!(AspectRatio::Landscape.as_str(), "16:9");
    }

    #[test]
    fn test_provider_kind_display() {
        assert_eq!(ImageProviderKind::Flux.to_string(), "flux");
        assert_eq!(ImageProviderKind::Gemini.to_string(), "gemini");
        assert_eq!(ImageProviderKind::Grok.to_string(), "grok");
    }
}
