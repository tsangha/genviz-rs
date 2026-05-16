//! Higgsfield video provider (CLI-backed).

use crate::error::{GenVizError, Result};
use crate::higgsfield::cli::{download, HiggsfieldCli};
use crate::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProvider, VideoProviderKind,
};
use async_trait::async_trait;
use std::time::Instant;

/// Higgsfield video models — wraps the CLI's `job_set_type` strings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum HiggsfieldVideoModel {
    /// ByteDance Seedance 2.0 (`seedance_2_0`). Default.
    #[default]
    Seedance20,
    /// ByteDance Seedance 1.5 Pro (`seedance1_5`).
    Seedance15Pro,
    /// Google Veo 3.1 (`veo3_1`).
    Veo31,
    /// Google Veo 3.1 Lite (`veo3_1_lite`).
    Veo31Lite,
    /// Google Veo 3 (`veo3`).
    Veo3,
    /// Kling v3.0 (`kling3_0`).
    Kling30,
    /// Kling 2.6 Video (`kling2_6`).
    Kling26,
    /// Alibaba Wan 2.7 (`wan2_7`).
    Wan27,
    /// Alibaba Wan 2.6 Video (`wan2_6`).
    Wan26,
    /// MiniMax Hailuo (`minimax_hailuo`).
    Hailuo,
    /// Grok Video (`grok_video`).
    GrokVideo,
    /// Higgsfield Soul Cast (`soul_cast`) — identity-aware character video.
    SoulCast,
    /// Cinematic Studio 3.0 video (`cinematic_studio_3_0`).
    CinematicStudio3,
    /// Cinematic Studio Video V2 (`cinematic_studio_video_v2`).
    CinematicStudioVideoV2,
    /// Marketing Studio Video (`marketing_studio_video`).
    MarketingStudioVideo,
    /// Custom `job_set_type`.
    Custom(String),
}

impl HiggsfieldVideoModel {
    /// Returns the Higgsfield `job_set_type` string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Seedance20 => "seedance_2_0",
            Self::Seedance15Pro => "seedance1_5",
            Self::Veo31 => "veo3_1",
            Self::Veo31Lite => "veo3_1_lite",
            Self::Veo3 => "veo3",
            Self::Kling30 => "kling3_0",
            Self::Kling26 => "kling2_6",
            Self::Wan27 => "wan2_7",
            Self::Wan26 => "wan2_6",
            Self::Hailuo => "minimax_hailuo",
            Self::GrokVideo => "grok_video",
            Self::SoulCast => "soul_cast",
            Self::CinematicStudio3 => "cinematic_studio_3_0",
            Self::CinematicStudioVideoV2 => "cinematic_studio_video_v2",
            Self::MarketingStudioVideo => "marketing_studio_video",
            Self::Custom(s) => s,
        }
    }
}

/// Builder for [`HiggsfieldVideoProvider`].
#[derive(Debug, Clone, Default)]
pub struct HiggsfieldVideoProviderBuilder {
    cli: HiggsfieldCli,
    model: HiggsfieldVideoModel,
    soul_id: Option<String>,
}

impl HiggsfieldVideoProviderBuilder {
    /// New builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the model (default: `seedance_2_0`).
    pub fn model(mut self, model: HiggsfieldVideoModel) -> Self {
        self.model = model;
        self
    }

    /// Reference a trained Soul Character (Soul Cast).
    pub fn soul_id(mut self, soul_id: impl Into<String>) -> Self {
        self.soul_id = Some(soul_id.into());
        self
    }

    /// Override the CLI binary location / wait timeout.
    pub fn with_cli(mut self, cli: HiggsfieldCli) -> Self {
        self.cli = cli;
        self
    }

    /// Build.
    pub fn build(self) -> Result<HiggsfieldVideoProvider> {
        Ok(HiggsfieldVideoProvider {
            cli: self.cli,
            model: self.model,
            soul_id: self.soul_id,
        })
    }
}

/// Higgsfield video provider (CLI-backed).
pub struct HiggsfieldVideoProvider {
    cli: HiggsfieldCli,
    model: HiggsfieldVideoModel,
    soul_id: Option<String>,
}

impl HiggsfieldVideoProvider {
    /// Builder.
    pub fn builder() -> HiggsfieldVideoProviderBuilder {
        HiggsfieldVideoProviderBuilder::new()
    }

    fn build_extra_args(&self, request: &VideoGenerationRequest) -> Vec<String> {
        let mut args: Vec<String> = Vec::new();

        if let Some(ar) = &request.aspect_ratio {
            args.push("--aspect-ratio".into());
            args.push(ar.clone());
        }
        if let Some(d) = request.duration_secs {
            args.push("--duration".into());
            args.push(d.to_string());
        }
        if let Some(res) = &request.resolution {
            args.push("--resolution".into());
            args.push(res.clone());
        }
        if let Some(neg) = &request.negative_prompt {
            args.push("--negative-prompt".into());
            args.push(neg.clone());
        }
        if let Some(seed) = request.seed {
            args.push("--seed".into());
            args.push(seed.to_string());
        }
        // Higgsfield accepts an upload-id or URL for `--image` and `--video`.
        if let Some(src) = &request.source_image_url {
            args.push("--image".into());
            args.push(src.clone());
        }
        if let Some(last) = &request.last_frame_url {
            args.push("--end-image".into());
            args.push(last.clone());
        }
        if let Some(soul) = &self.soul_id {
            args.push("--soul-id".into());
            args.push(soul.clone());
        }

        args
    }
}

#[async_trait]
impl VideoProvider for HiggsfieldVideoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();
        let extra_args = self.build_extra_args(request);

        let job = self
            .cli
            .generate(self.model.as_str(), &request.prompt, &extra_args)
            .await?;

        let url = job.result_url.as_deref().ok_or_else(|| {
            GenVizError::UnexpectedResponse(format!("Higgsfield job {} has no result_url", job.id))
        })?;
        let data = download(url).await?;
        let duration_ms = start.elapsed().as_millis() as u64;

        let mime = if url.ends_with(".webm") {
            "video/webm"
        } else if url.ends_with(".mov") {
            "video/quicktime"
        } else {
            "video/mp4"
        };

        Ok(GeneratedVideo::new(
            data,
            mime,
            VideoProviderKind::Higgsfield,
            VideoMetadata {
                model: Some(
                    job.job_set_type
                        .unwrap_or_else(|| self.model.as_str().to_string()),
                ),
                duration_ms: Some(duration_ms),
                video_duration_secs: request.duration_secs,
                resolution: request.resolution.clone(),
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::Higgsfield
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
        assert_eq!(HiggsfieldVideoModel::Seedance20.as_str(), "seedance_2_0");
        assert_eq!(HiggsfieldVideoModel::Veo31Lite.as_str(), "veo3_1_lite");
        assert_eq!(HiggsfieldVideoModel::Kling30.as_str(), "kling3_0");
        assert_eq!(HiggsfieldVideoModel::SoulCast.as_str(), "soul_cast");
        assert_eq!(HiggsfieldVideoModel::Wan27.as_str(), "wan2_7");
        assert_eq!(
            HiggsfieldVideoModel::Custom("future_video".into()).as_str(),
            "future_video"
        );
    }

    #[test]
    fn test_model_default() {
        assert_eq!(
            HiggsfieldVideoModel::default(),
            HiggsfieldVideoModel::Seedance20
        );
    }

    #[test]
    fn test_extra_args_basic() {
        let p = HiggsfieldVideoProvider::builder().build().unwrap();
        let req = VideoGenerationRequest::new("test")
            .with_aspect_ratio("16:9")
            .with_duration(5)
            .with_resolution("1080p");
        let args = p.build_extra_args(&req);
        assert!(args.windows(2).any(|w| w == ["--aspect-ratio", "16:9"]));
        assert!(args.windows(2).any(|w| w == ["--duration", "5"]));
        assert!(args.windows(2).any(|w| w == ["--resolution", "1080p"]));
    }

    #[test]
    fn test_extra_args_source_and_last_frame() {
        let p = HiggsfieldVideoProvider::builder().build().unwrap();
        let req = VideoGenerationRequest::new("test")
            .with_source_image("https://x/a.png")
            .with_last_frame_url("https://x/b.png");
        let args = p.build_extra_args(&req);
        assert!(args.windows(2).any(|w| w == ["--image", "https://x/a.png"]));
        assert!(args
            .windows(2)
            .any(|w| w == ["--end-image", "https://x/b.png"]));
    }

    #[test]
    fn test_extra_args_soul_id() {
        let p = HiggsfieldVideoProvider::builder()
            .soul_id("soul-xyz")
            .build()
            .unwrap();
        let req = VideoGenerationRequest::new("test");
        let args = p.build_extra_args(&req);
        assert!(args.windows(2).any(|w| w == ["--soul-id", "soul-xyz"]));
    }
}
