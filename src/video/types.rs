//! Core types for video generation.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Video provider kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoProviderKind {
    /// xAI Grok video generation.
    Grok,
    /// Google Veo video generation.
    Veo,
    /// OpenAI Sora video generation.
    OpenAI,
    /// Kuaishou Kling AI video generation.
    Kling,
    /// fal.ai video models.
    Fal,
    /// MiniMax Hailuo video generation (direct API).
    MiniMax,
}

impl std::fmt::Display for VideoProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grok => write!(f, "grok"),
            Self::Veo => write!(f, "veo"),
            Self::OpenAI => write!(f, "openai"),
            Self::Kling => write!(f, "kling"),
            Self::Fal => write!(f, "fal"),
            Self::MiniMax => write!(f, "minimax"),
        }
    }
}

/// A subject reference for character consistency (MiniMax direct API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectReference {
    /// Reference type (e.g., "character").
    #[serde(rename = "type")]
    pub ref_type: String,
    /// Image URL or base64 data.
    pub image: String,
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
    /// Source image URL for image-to-video (Grok, Kling, fal.ai, Sora).
    pub source_image_url: Option<String>,
    /// URL to last frame image for video interpolation (fal.ai Wan FLF2V, Kling-via-fal).
    pub last_frame_url: Option<String>,
    /// Base64-encoded first frame image (Veo only).
    pub image: Option<String>,
    /// Base64-encoded last frame image (Veo only).
    pub last_frame: Option<String>,
    /// Base64-encoded video for extension/continuation (Veo only).
    pub video: Option<String>,
    /// Base64-encoded reference images for style/asset guidance (Veo only, max 3).
    pub reference_images: Option<Vec<String>>,
    /// Negative prompt — describes what to avoid (Veo, fal.ai).
    pub negative_prompt: Option<String>,
    /// Person generation policy: "allow_all" or "allow_adult" (Veo only).
    pub person_generation: Option<String>,
    /// Number of videos to generate (Veo only).
    pub number_of_videos: Option<u32>,
    /// GCS bucket URI for video output (Vertex AI only).
    pub storage_uri: Option<String>,
    /// Enable prompt enhancement (Vertex AI only).
    pub enhance_prompt: Option<bool>,
    /// Enable audio generation (Vertex AI only).
    pub generate_audio: Option<bool>,
    /// Seed for deterministic generation (Seedance).
    pub seed: Option<i64>,
    /// Lock camera position (Seedance).
    pub camera_fixed: Option<bool>,
    /// Enable MiniMax prompt enhancement.
    pub prompt_optimizer: Option<bool>,
    /// Subject references for character consistency (MiniMax direct API).
    pub subject_reference: Option<Vec<SubjectReference>>,
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
            last_frame_url: None,
            image: None,
            last_frame: None,
            video: None,
            reference_images: None,
            negative_prompt: None,
            person_generation: None,
            number_of_videos: None,
            storage_uri: None,
            enhance_prompt: None,
            generate_audio: None,
            seed: None,
            camera_fixed: None,
            prompt_optimizer: None,
            subject_reference: None,
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

    /// Sets a last frame URL for video interpolation (fal.ai Wan FLF2V).
    pub fn with_last_frame_url(mut self, url: impl Into<String>) -> Self {
        self.last_frame_url = Some(url.into());
        self
    }

    /// Sets a base64-encoded first frame image (Veo only).
    pub fn with_image(mut self, base64: impl Into<String>) -> Self {
        self.image = Some(base64.into());
        self
    }

    /// Sets a base64-encoded last frame image (Veo only).
    pub fn with_last_frame(mut self, base64: impl Into<String>) -> Self {
        self.last_frame = Some(base64.into());
        self
    }

    /// Sets a base64-encoded video for extension/continuation (Veo only).
    pub fn with_video(mut self, base64: impl Into<String>) -> Self {
        self.video = Some(base64.into());
        self
    }

    /// Adds a base64-encoded reference image (Veo only, max 3).
    pub fn with_reference_image(mut self, base64: impl Into<String>) -> Self {
        self.reference_images
            .get_or_insert_with(Vec::new)
            .push(base64.into());
        self
    }

    /// Sets the negative prompt (Veo, fal.ai).
    pub fn with_negative_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(prompt.into());
        self
    }

    /// Sets the person generation policy (Veo only).
    pub fn with_person_generation(mut self, policy: impl Into<String>) -> Self {
        self.person_generation = Some(policy.into());
        self
    }

    /// Sets the number of videos to generate (Veo only).
    pub fn with_number_of_videos(mut self, n: u32) -> Self {
        self.number_of_videos = Some(n);
        self
    }

    /// Sets the GCS bucket URI for video output (Vertex AI only).
    pub fn with_storage_uri(mut self, uri: impl Into<String>) -> Self {
        self.storage_uri = Some(uri.into());
        self
    }

    /// Enables or disables prompt enhancement (Vertex AI only).
    pub fn with_enhance_prompt(mut self, enhance: bool) -> Self {
        self.enhance_prompt = Some(enhance);
        self
    }

    /// Enables or disables audio generation (Vertex AI only).
    pub fn with_generate_audio(mut self, audio: bool) -> Self {
        self.generate_audio = Some(audio);
        self
    }

    /// Sets the seed for deterministic generation (Seedance).
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Locks the camera position (Seedance).
    pub fn with_camera_fixed(mut self, fixed: bool) -> Self {
        self.camera_fixed = Some(fixed);
        self
    }

    /// Enables or disables prompt optimization (MiniMax).
    pub fn with_prompt_optimizer(mut self, optimize: bool) -> Self {
        self.prompt_optimizer = Some(optimize);
        self
    }

    /// Adds a subject reference for character consistency (MiniMax direct API).
    pub fn with_subject_reference(
        mut self,
        ref_type: impl Into<String>,
        image: impl Into<String>,
    ) -> Self {
        self.subject_reference
            .get_or_insert_with(Vec::new)
            .push(SubjectReference {
                ref_type: ref_type.into(),
                image: image.into(),
            });
        self
    }
}

/// A generated video with its data and metadata.
#[derive(Debug, Clone)]
#[must_use = "generated video should be saved or processed"]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_provider_kind_display() {
        assert_eq!(VideoProviderKind::Grok.to_string(), "grok");
        assert_eq!(VideoProviderKind::Veo.to_string(), "veo");
        assert_eq!(VideoProviderKind::OpenAI.to_string(), "openai");
        assert_eq!(VideoProviderKind::Kling.to_string(), "kling");
        assert_eq!(VideoProviderKind::Fal.to_string(), "fal");
        assert_eq!(VideoProviderKind::MiniMax.to_string(), "minimax");
    }

    #[test]
    fn test_video_request_builder_chain() {
        let req = VideoGenerationRequest::new("Ocean waves")
            .with_duration(10)
            .with_aspect_ratio("16:9")
            .with_resolution("720p")
            .with_source_image("https://example.com/photo.jpg");

        assert_eq!(req.prompt, "Ocean waves");
        assert_eq!(req.duration_secs, Some(10));
        assert_eq!(req.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(req.resolution.as_deref(), Some("720p"));
        assert_eq!(
            req.source_image_url.as_deref(),
            Some("https://example.com/photo.jpg")
        );
        assert!(req.image.is_none());
        assert!(req.last_frame.is_none());
        assert!(req.last_frame_url.is_none());
    }

    #[test]
    fn test_video_request_with_frame_images() {
        let req = VideoGenerationRequest::new("Interpolate between frames")
            .with_image("first_frame_b64")
            .with_last_frame("last_frame_b64");

        assert_eq!(req.image.as_deref(), Some("first_frame_b64"));
        assert_eq!(req.last_frame.as_deref(), Some("last_frame_b64"));
    }

    #[test]
    fn test_generated_video_size() {
        let video = GeneratedVideo::new(
            vec![0; 1024],
            "video/mp4",
            VideoProviderKind::Grok,
            VideoMetadata::default(),
        );
        assert_eq!(video.size(), 1024);
    }

    #[test]
    fn test_generated_video_to_base64() {
        let video = GeneratedVideo::new(
            vec![1, 2, 3],
            "video/mp4",
            VideoProviderKind::Grok,
            VideoMetadata::default(),
        );
        assert_eq!(video.to_base64(), "AQID");
    }

    #[test]
    fn test_generated_video_to_data_url() {
        let video = GeneratedVideo::new(
            vec![1, 2, 3],
            "video/mp4",
            VideoProviderKind::Grok,
            VideoMetadata::default(),
        );
        assert_eq!(video.to_data_url(), "data:video/mp4;base64,AQID");
    }

    #[test]
    fn test_subject_reference_serde_rename() {
        let sr = SubjectReference {
            ref_type: "character".into(),
            image: "https://example.com/face.jpg".into(),
        };
        let json = serde_json::to_value(&sr).unwrap();
        // The field should serialize as "type", not "ref_type"
        assert_eq!(json["type"], "character");
        assert!(json.get("ref_type").is_none());
        assert_eq!(json["image"], "https://example.com/face.jpg");

        // Round-trip: deserialize back
        let deserialized: SubjectReference = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.ref_type, "character");
        assert_eq!(deserialized.image, "https://example.com/face.jpg");
    }

    #[test]
    fn test_builder_with_seed() {
        let req = VideoGenerationRequest::new("Test").with_seed(42);
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn test_builder_with_camera_fixed() {
        let req = VideoGenerationRequest::new("Test").with_camera_fixed(true);
        assert_eq!(req.camera_fixed, Some(true));
    }

    #[test]
    fn test_builder_with_prompt_optimizer() {
        let req = VideoGenerationRequest::new("Test").with_prompt_optimizer(true);
        assert_eq!(req.prompt_optimizer, Some(true));
    }

    #[test]
    fn test_builder_with_subject_reference() {
        let req = VideoGenerationRequest::new("Test")
            .with_subject_reference("character", "https://example.com/face.jpg");
        let refs = req.subject_reference.as_ref().unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].ref_type, "character");
        assert_eq!(refs[0].image, "https://example.com/face.jpg");
    }

    #[test]
    fn test_video_provider_kind_serde_roundtrip() {
        let variants = [
            (VideoProviderKind::Grok, "grok"),
            (VideoProviderKind::Veo, "veo"),
            (VideoProviderKind::OpenAI, "openai"),
            (VideoProviderKind::Kling, "kling"),
            (VideoProviderKind::Fal, "fal"),
            (VideoProviderKind::MiniMax, "minimax"),
        ];
        for (variant, expected_str) in &variants {
            let json = serde_json::to_value(variant).unwrap();
            assert_eq!(json.as_str().unwrap(), *expected_str);
            let deserialized: VideoProviderKind = serde_json::from_value(json).unwrap();
            assert_eq!(&deserialized, variant);
        }
    }
}
