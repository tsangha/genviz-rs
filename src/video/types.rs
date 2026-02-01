//! Core types for video generation.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Video provider kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoProviderKind {
    Grok,
    Veo,
}

impl std::fmt::Display for VideoProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grok => write!(f, "grok"),
            Self::Veo => write!(f, "veo"),
        }
    }
}

/// Metadata about the video generation process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Model used for generation.
    pub model: Option<String>,
    /// Generation duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Video duration in seconds.
    pub video_duration_secs: Option<u32>,
    /// Video resolution.
    pub resolution: Option<String>,
}

/// A request to generate a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationRequest {
    /// The text prompt describing the desired video.
    pub prompt: String,
    /// Desired video duration in seconds (1-15 for Grok).
    pub duration_secs: Option<u32>,
    /// Aspect ratio (e.g., "16:9", "9:16").
    pub aspect_ratio: Option<String>,
    /// Resolution (e.g., "720p").
    pub resolution: Option<String>,
    /// Source image URL (for image-to-video).
    pub source_image_url: Option<String>,
}

impl VideoGenerationRequest {
    /// Creates a new request with the given prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            duration_secs: None,
            aspect_ratio: None,
            resolution: None,
            source_image_url: None,
        }
    }

    /// Sets the desired video duration in seconds.
    pub fn with_duration(mut self, secs: u32) -> Self {
        self.duration_secs = Some(secs);
        self
    }

    /// Sets the aspect ratio.
    pub fn with_aspect_ratio(mut self, ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(ratio.into());
        self
    }

    /// Sets the resolution.
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Sets a source image for image-to-video generation.
    pub fn with_source_image(mut self, url: impl Into<String>) -> Self {
        self.source_image_url = Some(url.into());
        self
    }
}

/// A generated video with its data and metadata.
#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    /// Raw video bytes.
    pub data: Vec<u8>,
    /// MIME type (e.g., "video/mp4").
    pub mime_type: String,
    /// Provider that generated this video.
    pub provider: VideoProviderKind,
    /// Generation metadata.
    pub metadata: VideoMetadata,
}

impl GeneratedVideo {
    /// Creates a new generated video.
    pub fn new(
        data: Vec<u8>,
        mime_type: impl Into<String>,
        provider: VideoProviderKind,
        metadata: VideoMetadata,
    ) -> Self {
        Self {
            data,
            mime_type: mime_type.into(),
            provider,
            metadata,
        }
    }

    /// Returns the size of the video data in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Saves the video to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, &self.data)?;
        Ok(())
    }

    /// Encodes the video data as base64.
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(&self.data)
    }

    /// Returns the video as a data URL.
    pub fn to_data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.to_base64())
    }
}
