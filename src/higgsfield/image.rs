//! Higgsfield image provider (CLI-backed).

use crate::error::{GenVizError, Result};
use crate::higgsfield::cli::{download, HiggsfieldCli, HiggsfieldMode, HiggsfieldSurface};
use crate::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProvider,
    ImageProviderKind,
};
use async_trait::async_trait;
use std::time::Instant;

/// Higgsfield image models — wraps `job_set_type` strings exposed by the CLI.
///
/// Use `Custom(String)` for models we don't list explicitly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum HiggsfieldImageModel {
    /// GPT Image 2 (`gpt_image_2`) — Higgsfield's flagship image model. Default.
    #[default]
    GptImage2,
    /// Nano Banana Pro (`nano_banana_2`) — Gemini 3 Pro Image.
    NanoBananaPro,
    /// Nano Banana 2 (`nano_banana_flash`) — fast variant.
    NanoBanana2,
    /// FLUX.2 (`flux_2`).
    Flux2,
    /// Flux Kontext (`flux_kontext`).
    FluxKontext,
    /// Higgsfield Soul V2 (`text2image_soul_v2`) — personalized identity model.
    SoulV2,
    /// Soul Cinematic (`soul_cinematic`).
    SoulCinematic,
    /// Soul Location (`soul_location`).
    SoulLocation,
    /// Seedream 4.5 (`seedream_v4_5`).
    Seedream45,
    /// Seedream V5 Lite (`seedream_v5_lite`).
    Seedream5Lite,
    /// Kling O1 Image (`kling_omni_image`).
    KlingOmni,
    /// Cinematic Studio 2.5 (`cinematic_studio_2_5`).
    CinematicStudio25,
    /// Grok Image (`grok_image`).
    GrokImage,
    /// Z Image (`z_image`).
    ZImage,
    /// Custom `job_set_type` for any model not enumerated above.
    Custom(String),
}

impl HiggsfieldImageModel {
    /// Returns the Higgsfield `job_set_type` string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::GptImage2 => "gpt_image_2",
            Self::NanoBananaPro => "nano_banana_2",
            Self::NanoBanana2 => "nano_banana_flash",
            Self::Flux2 => "flux_2",
            Self::FluxKontext => "flux_kontext",
            Self::SoulV2 => "text2image_soul_v2",
            Self::SoulCinematic => "soul_cinematic",
            Self::SoulLocation => "soul_location",
            Self::Seedream45 => "seedream_v4_5",
            Self::Seedream5Lite => "seedream_v5_lite",
            Self::KlingOmni => "kling_omni_image",
            Self::CinematicStudio25 => "cinematic_studio_2_5",
            Self::GrokImage => "grok_image",
            Self::ZImage => "z_image",
            Self::Custom(s) => s,
        }
    }
}

/// Builder for [`HiggsfieldImageProvider`].
#[derive(Debug, Clone, Default)]
pub struct HiggsfieldImageProviderBuilder {
    cli: HiggsfieldCli,
    model: HiggsfieldImageModel,
    surface: Option<HiggsfieldSurface>,
    mode: Option<HiggsfieldMode>,
    soul_id: Option<String>,
    aspect_ratio_override: Option<String>,
    count: Option<u32>,
}

impl HiggsfieldImageProviderBuilder {
    /// New builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the model (default: `gpt_image_2`).
    pub fn model(mut self, model: HiggsfieldImageModel) -> Self {
        self.model = model;
        self
    }

    /// Pick the Higgsfield surface (defaults to `Generate`).
    pub fn surface(mut self, surface: HiggsfieldSurface) -> Self {
        self.surface = Some(surface);
        self
    }

    /// Set the product-photoshoot mode (used only when `surface` is `ProductPhotoshoot`).
    pub fn mode(mut self, mode: HiggsfieldMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Reference a trained Soul Character (Soul-aware models like `text2image_soul_v2`).
    pub fn soul_id(mut self, soul_id: impl Into<String>) -> Self {
        self.soul_id = Some(soul_id.into());
        self
    }

    /// Override how many images to request in a single job.
    pub fn count(mut self, n: u32) -> Self {
        self.count = Some(n);
        self
    }

    /// Override the CLI binary location / wait timeout.
    pub fn with_cli(mut self, cli: HiggsfieldCli) -> Self {
        self.cli = cli;
        self
    }

    /// Build.
    pub fn build(self) -> Result<HiggsfieldImageProvider> {
        Ok(HiggsfieldImageProvider {
            cli: self.cli,
            model: self.model,
            surface: self.surface.unwrap_or(HiggsfieldSurface::Generate),
            mode: self.mode,
            soul_id: self.soul_id,
            aspect_ratio_override: self.aspect_ratio_override,
            count: self.count,
        })
    }
}

/// Higgsfield image provider.
///
/// Wraps the local `higgsfield` CLI for image generation across all
/// Higgsfield-exposed image models, with optional product-photoshoot and
/// marketplace-cards surface modes.
pub struct HiggsfieldImageProvider {
    cli: HiggsfieldCli,
    model: HiggsfieldImageModel,
    surface: HiggsfieldSurface,
    mode: Option<HiggsfieldMode>,
    soul_id: Option<String>,
    aspect_ratio_override: Option<String>,
    count: Option<u32>,
}

impl HiggsfieldImageProvider {
    /// Builder.
    pub fn builder() -> HiggsfieldImageProviderBuilder {
        HiggsfieldImageProviderBuilder::new()
    }

    fn build_extra_args(
        &self,
        request: &GenerationRequest,
    ) -> Result<(Vec<String>, Option<tempfile::NamedTempFile>)> {
        let mut args: Vec<String> = Vec::new();

        // Aspect ratio — most Higgsfield image models accept --aspect-ratio NxM
        let ar = self.aspect_ratio_override.clone().or_else(|| {
            request
                .aspect_ratio
                .as_ref()
                .map(|a| a.as_str().to_string())
        });
        if let Some(ar) = ar {
            args.push("--aspect-ratio".into());
            args.push(ar);
        }

        if let Some(seed) = request.seed {
            args.push("--seed".into());
            args.push(seed.to_string());
        }

        if let Some(soul) = &self.soul_id {
            args.push("--soul-id".into());
            args.push(soul.clone());
        }

        if let Some(count) = self.count {
            args.push("--count".into());
            args.push(count.to_string());
        }

        // Input image for edits — write bytes to a temp file so the CLI can auto-upload it.
        let mut tmp_holder: Option<tempfile::NamedTempFile> = None;
        if let Some(bytes) = &request.input_image {
            let mut tmp = tempfile::Builder::new()
                .prefix("genviz-higgsfield-")
                .suffix(".png")
                .tempfile()
                .map_err(GenVizError::Io)?;
            use std::io::Write;
            tmp.write_all(bytes).map_err(GenVizError::Io)?;
            args.push("--image".into());
            args.push(tmp.path().display().to_string());
            tmp_holder = Some(tmp);
        }

        Ok((args, tmp_holder))
    }
}

#[async_trait]
impl ImageProvider for HiggsfieldImageProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();
        let (extra_args, _tmp) = self.build_extra_args(request)?;

        let job = match self.surface {
            HiggsfieldSurface::Generate => {
                self.cli
                    .generate(self.model.as_str(), &request.prompt, &extra_args)
                    .await?
            }
            HiggsfieldSurface::ProductPhotoshoot => {
                let mode = self.mode.ok_or_else(|| {
                    GenVizError::InvalidRequest(
                        "product-photoshoot surface requires a mode (HiggsfieldMode::*)".into(),
                    )
                })?;
                self.cli
                    .product_photoshoot(mode, &request.prompt, &extra_args)
                    .await?
            }
            HiggsfieldSurface::MarketplaceCards => {
                self.cli
                    .marketplace_cards(&request.prompt, &extra_args)
                    .await?
            }
        };

        let url = job.result_url.as_deref().ok_or_else(|| {
            GenVizError::UnexpectedResponse(format!("Higgsfield job {} has no result_url", job.id))
        })?;
        let data = download(url).await?;
        let duration_ms = start.elapsed().as_millis() as u64;
        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Higgsfield,
            GenerationMetadata {
                model: Some(
                    job.job_set_type
                        .unwrap_or_else(|| self.model.as_str().to_string()),
                ),
                seed: request.seed,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Higgsfield
    }

    async fn health_check(&self) -> Result<()> {
        self.cli.version().await.map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_as_str() {
        assert_eq!(HiggsfieldImageModel::GptImage2.as_str(), "gpt_image_2");
        assert_eq!(
            HiggsfieldImageModel::NanoBananaPro.as_str(),
            "nano_banana_2"
        );
        assert_eq!(HiggsfieldImageModel::SoulV2.as_str(), "text2image_soul_v2");
        assert_eq!(
            HiggsfieldImageModel::Custom("future_model".into()).as_str(),
            "future_model"
        );
    }

    #[test]
    fn test_builder_defaults() {
        let p = HiggsfieldImageProvider::builder().build().unwrap();
        assert_eq!(p.model, HiggsfieldImageModel::GptImage2);
        assert_eq!(p.surface, HiggsfieldSurface::Generate);
    }

    #[test]
    fn test_builder_product_photoshoot() {
        let p = HiggsfieldImageProvider::builder()
            .surface(HiggsfieldSurface::ProductPhotoshoot)
            .mode(HiggsfieldMode::LifestyleScene)
            .build()
            .unwrap();
        assert_eq!(p.surface, HiggsfieldSurface::ProductPhotoshoot);
        assert_eq!(p.mode, Some(HiggsfieldMode::LifestyleScene));
    }

    #[test]
    fn test_extra_args_aspect_ratio_and_seed() {
        let p = HiggsfieldImageProvider::builder().build().unwrap();
        let req = GenerationRequest::new("test")
            .with_aspect_ratio(crate::image::AspectRatio::Landscape)
            .with_seed(42);
        let (args, _tmp) = p.build_extra_args(&req).unwrap();
        assert!(args.windows(2).any(|w| w == ["--aspect-ratio", "16:9"]));
        assert!(args.windows(2).any(|w| w == ["--seed", "42"]));
    }

    #[test]
    fn test_extra_args_soul_id() {
        let p = HiggsfieldImageProvider::builder()
            .soul_id("soul-abc")
            .build()
            .unwrap();
        let req = GenerationRequest::new("test");
        let (args, _tmp) = p.build_extra_args(&req).unwrap();
        assert!(args.windows(2).any(|w| w == ["--soul-id", "soul-abc"]));
    }

    #[test]
    fn test_extra_args_writes_temp_image_file() {
        let p = HiggsfieldImageProvider::builder().build().unwrap();
        let png_magic = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let req = GenerationRequest::new("edit").with_input_image(png_magic.clone());
        let (args, tmp) = p.build_extra_args(&req).unwrap();
        let tmp = tmp.expect("temp file should exist");
        // --image <path> pair should be present and the path on disk should contain the bytes
        let pos = args
            .iter()
            .position(|a| a == "--image")
            .expect("--image arg");
        let path = &args[pos + 1];
        assert_eq!(path, &tmp.path().display().to_string());
        let written = std::fs::read(tmp.path()).unwrap();
        assert_eq!(written, png_magic);
    }
}
