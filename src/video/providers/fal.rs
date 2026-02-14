//! fal.ai video generation provider.

use crate::error::{sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// fal.ai video model variants.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FalVideoModel {
    /// Wan 2.1 - text-to-video generation (default).
    #[default]
    Wan21,
    /// Wan 2.1 - image-to-video generation.
    Wan21I2V,
    /// MiniMax Hailuo 2.3 Standard - text-to-video generation.
    Hailuo23Std,
    /// MiniMax Hailuo 2.3 Pro - text-to-video generation.
    Hailuo23Pro,
    /// MiniMax Hailuo 2.3 Fast - image-to-video generation.
    Hailuo23Fast,
    /// ByteDance Seedance V1 Pro - text-to-video generation.
    SeedancePro,
    /// ByteDance Seedance V1 Lite - text-to-video generation.
    SeedanceLite,
    /// ByteDance Seedance V1.5 Pro - text-to-video generation.
    Seedance15Pro,
    /// LTX Video 2 - text-to-video generation.
    LtxVideo,
    /// Wan 2.1 - first-last-frame-to-video (requires first_frame + last_frame URLs).
    Wan21FLF2V,
    /// Kling V1.6 Standard via fal.ai.
    KlingStd,
    /// Kling V1.6 Pro via fal.ai.
    KlingPro,
    /// Custom fal.ai video model ID.
    Custom(String),
}

impl FalVideoModel {
    /// Returns the fal.ai model identifier string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Wan21 => "fal-ai/wan-t2v",
            Self::Wan21I2V => "fal-ai/wan-i2v",
            Self::Hailuo23Std => "fal-ai/minimax/hailuo-2.3/standard/text-to-video",
            Self::Hailuo23Pro => "fal-ai/minimax/hailuo-2.3/pro/text-to-video",
            Self::Hailuo23Fast => "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video",
            Self::SeedancePro => "fal-ai/bytedance/seedance/v1/pro/text-to-video",
            Self::SeedanceLite => "fal-ai/bytedance/seedance/v1/lite/text-to-video",
            Self::Seedance15Pro => "fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
            Self::LtxVideo => "fal-ai/ltx-2/text-to-video",
            Self::Wan21FLF2V => "fal-ai/wan-flf2v",
            Self::KlingStd => "fal-ai/kling-video/v1.6/standard/text-to-video",
            Self::KlingPro => "fal-ai/kling-video/v1.6/pro/text-to-video",
            Self::Custom(id) => id,
        }
    }
}

impl FalVideoModel {
    /// Returns the image-to-video variant if applicable.
    fn i2v_variant(&self) -> Option<Self> {
        match self {
            Self::Hailuo23Std => Some(Self::Custom(
                "fal-ai/minimax/hailuo-2.3/standard/image-to-video".to_string(),
            )),
            Self::Hailuo23Pro => Some(Self::Custom(
                "fal-ai/minimax/hailuo-2.3/pro/image-to-video".to_string(),
            )),
            Self::Hailuo23Fast => None, // already an i2v endpoint
            Self::SeedancePro => Some(Self::Custom(
                "fal-ai/bytedance/seedance/v1/pro/image-to-video".to_string(),
            )),
            Self::SeedanceLite => Some(Self::Custom(
                "fal-ai/bytedance/seedance/v1/lite/image-to-video".to_string(),
            )),
            Self::Seedance15Pro => Some(Self::Custom(
                "fal-ai/bytedance/seedance/v1.5/pro/image-to-video".to_string(),
            )),
            Self::KlingStd => Some(Self::Custom(
                "fal-ai/kling-video/v1.6/standard/image-to-video".to_string(),
            )),
            Self::KlingPro => Some(Self::Custom(
                "fal-ai/kling-video/v1.6/pro/image-to-video".to_string(),
            )),
            _ => None,
        }
    }
}

/// Builder for [`FalVideoProvider`].
#[derive(Debug, Clone)]
pub struct FalVideoProviderBuilder {
    api_key: Option<String>,
    model: FalVideoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for FalVideoProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: FalVideoModel::default(),
            poll_interval: Duration::from_secs(3),
            timeout: Duration::from_secs(600), // 10 minutes for video
        }
    }
}

impl FalVideoProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `FAL_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the fal.ai video model variant.
    pub fn model(mut self, model: FalVideoModel) -> Self {
        self.model = model;
        self
    }

    /// Sets the polling interval for async generation.
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the maximum time to wait for generation.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builds the provider, resolving the API key.
    pub fn build(self) -> Result<FalVideoProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("FAL_KEY").ok())
            .ok_or_else(|| GenVizError::Auth("FAL_KEY not set and no API key provided".into()))?;

        Ok(FalVideoProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// fal.ai video generation provider.
///
/// Supports text-to-video (Wan 2.1, LTX Video, Hailuo 2.3, Seedance) and
/// image-to-video (Wan 2.1 I2V, Hailuo 2.3, Seedance, Kling) generation
/// through fal.ai's queue API.
#[derive(Debug)]
pub struct FalVideoProvider {
    client: reqwest::Client,
    api_key: String,
    model: FalVideoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl FalVideoProvider {
    /// Creates a new [`FalVideoProviderBuilder`].
    pub fn builder() -> FalVideoProviderBuilder {
        FalVideoProviderBuilder::new()
    }

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
        let text = sanitize_error_message(text);

        // Try to extract fal.ai error detail from JSON response.
        if let Ok(error_resp) = serde_json::from_str::<FalErrorResponse>(&text) {
            let detail = error_resp.detail;
            let lower = detail.to_lowercase();
            if lower.contains("unauthorized") || lower.contains("invalid key") {
                return GenVizError::Auth(detail);
            }
            if lower.contains("rate") && lower.contains("limit") {
                return GenVizError::RateLimited { retry_after: None };
            }
            return GenVizError::Api {
                status,
                message: detail,
            };
        }

        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        if status == 429 {
            return GenVizError::RateLimited { retry_after: None };
        }

        GenVizError::Api {
            status,
            message: text,
        }
    }

    /// Submit a video generation request to the fal.ai queue.
    ///
    /// Returns the submit response with request_id, status_url, and response_url.
    /// fal.ai's queue system uses different URL paths than the submit endpoint
    /// for models with nested paths, so we use their returned URLs for polling
    /// and apply a fallback strategy for result fetching.
    async fn submit(
        &self,
        request: &VideoGenerationRequest,
        model_id: &str,
    ) -> Result<FalSubmitResponse> {
        let url = format!("https://queue.fal.run/{}", model_id);

        let body = FalVideoRequest {
            prompt: request.prompt.clone(),
            image_url: request.source_image_url.clone(),
            duration: request.duration_secs,
            aspect_ratio: request.aspect_ratio.clone(),
            negative_prompt: request.negative_prompt.clone(),
            start_image_url: request.source_image_url.clone(),
            end_image_url: request.last_frame_url.clone(),
            resolution: request.resolution.clone(),
            camera_fixed: request.camera_fixed,
            seed: request.seed,
            generate_audio: request.generate_audio,
            prompt_optimizer: request.prompt_optimizer,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Key {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let submit_response: FalSubmitResponse = response.json().await?;

        Ok(submit_response)
    }

    /// Poll the fal.ai queue until the video generation is complete.
    async fn poll_until_ready(&self, request_id: &str, status_url: &str) -> Result<()> {
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(status_url)
                .header("Authorization", format!("Key {}", self.api_key))
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text));
            }

            let status_response: FalStatusResponse = response.json().await?;

            match status_response.status.as_str() {
                "COMPLETED" => return Ok(()),
                "IN_QUEUE" | "IN_PROGRESS" => {
                    tracing::debug!(
                        request_id = %request_id,
                        status = %status_response.status,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling fal.ai video generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                }
                "FAILED" => {
                    return Err(GenVizError::VideoGeneration(
                        "fal.ai video generation failed".into(),
                    ));
                }
                other => {
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "fal.ai returned unexpected status: {}",
                        other
                    )));
                }
            }
        }
    }

    /// Fetch the completed result from the fal.ai queue.
    ///
    /// Tries the fal.ai-provided `response_url` first. If that returns a 404
    /// (which happens for video models with nested paths like `wan/v2.1/1080p`),
    /// falls back to constructing the URL from the full model_id.
    async fn fetch_result(
        &self,
        response_url: &str,
        model_id: &str,
        request_id: &str,
    ) -> Result<String> {
        let response = self
            .client
            .get(response_url)
            .header("Authorization", format!("Key {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if status.as_u16() == 404 || status.as_u16() == 405 {
            // Fallback: construct URL from full model_id
            tracing::debug!(
                response_url = %response_url,
                "fal.ai response_url returned {}, falling back to model_id-based URL",
                status.as_u16()
            );
            let fallback_url =
                format!("https://queue.fal.run/{}/requests/{}", model_id, request_id);
            let fallback = self
                .client
                .get(&fallback_url)
                .header("Authorization", format!("Key {}", self.api_key))
                .send()
                .await?;

            let fb_status = fallback.status();
            if !fb_status.is_success() {
                let text = fallback.text().await.unwrap_or_default();
                return Err(self.parse_error(fb_status.as_u16(), &text));
            }

            let result: FalVideoResult = fallback.json().await?;
            return Ok(result.video.url);
        }

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let result: FalVideoResult = response.json().await?;

        Ok(result.video.url)
    }

    /// Download the video from the given URL.
    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(GenVizError::Api {
                status: response.status().as_u16(),
                message: "Failed to download video".into(),
            });
        }

        Ok(response.bytes().await?.to_vec())
    }
}

#[async_trait]
impl VideoProvider for FalVideoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();

        // Validate i2v-only models require a source image
        if matches!(self.model, FalVideoModel::Hailuo23Fast) && request.source_image_url.is_none() {
            return Err(GenVizError::InvalidRequest(
                "Hailuo 2.3 Fast is image-to-video only — provide source_image_url".into(),
            ));
        }

        // Determine the effective model for this request (handles auto-selection).
        let mut effective_model = if request.source_image_url.is_some() {
            match &self.model {
                FalVideoModel::Wan21 => {
                    if request.last_frame_url.is_some() {
                        FalVideoModel::Wan21FLF2V // first+last frame → FLF2V
                    } else {
                        FalVideoModel::Wan21I2V // first frame only → I2V
                    }
                }
                other => other.clone(),
            }
        } else {
            self.model.clone()
        };

        // If Kling model with source image, switch to i2v variant
        if request.source_image_url.is_some() {
            if let Some(i2v) = effective_model.i2v_variant() {
                effective_model = i2v;
            }
        }

        let model_id = effective_model.as_str().to_string();

        let submit = self.submit(request, &model_id).await?;
        tracing::debug!(request_id = %submit.request_id, model = %model_id, "submitted fal.ai video generation request");

        self.poll_until_ready(&submit.request_id, &submit.status_url)
            .await?;
        tracing::debug!(request_id = %submit.request_id, "fal.ai video generation complete");

        let video_url = self
            .fetch_result(&submit.response_url, &model_id, &submit.request_id)
            .await?;
        tracing::debug!(url = %video_url, "fetched fal.ai video result");

        let data = self.download(&video_url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::Fal,
            VideoMetadata {
                model: Some(model_id),
                duration_ms: Some(duration_ms),
                video_duration_secs: request.duration_secs,
                resolution: None,
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::Fal
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(GenVizError::Auth("FAL_KEY is empty".into()));
        }
        Ok(())
    }
}

// Request types

#[derive(Debug, Serialize)]
struct FalVideoRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    /// Start image URL for FLF2V (first-last-frame-to-video).
    #[serde(skip_serializing_if = "Option::is_none")]
    start_image_url: Option<String>,
    /// End image URL for FLF2V (first-last-frame-to-video).
    #[serde(skip_serializing_if = "Option::is_none")]
    end_image_url: Option<String>,
    /// Video resolution for Seedance models ("480p", "720p", "1080p").
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    /// Lock camera position (Seedance).
    #[serde(skip_serializing_if = "Option::is_none")]
    camera_fixed: Option<bool>,
    /// Seed for deterministic generation (Seedance).
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    /// Enable audio generation (Seedance 1.5).
    #[serde(skip_serializing_if = "Option::is_none")]
    generate_audio: Option<bool>,
    /// Enable MiniMax Hailuo prompt enhancement.
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_optimizer: Option<bool>,
}

// Response types

#[derive(Debug, Deserialize)]
struct FalSubmitResponse {
    request_id: String,
    /// URL to poll for status (provided by fal.ai).
    status_url: String,
    /// URL to fetch completed result (provided by fal.ai).
    response_url: String,
}

#[derive(Debug, Deserialize)]
struct FalStatusResponse {
    status: String,
}

#[derive(Debug, Deserialize)]
struct FalVideoResult {
    video: FalVideoData,
}

#[derive(Debug, Deserialize)]
struct FalVideoData {
    url: String,
}

#[derive(Debug, Deserialize)]
struct FalErrorResponse {
    detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fal_video_model_as_str() {
        assert_eq!(FalVideoModel::Wan21.as_str(), "fal-ai/wan-t2v");
        assert_eq!(FalVideoModel::Wan21I2V.as_str(), "fal-ai/wan-i2v");
        assert_eq!(
            FalVideoModel::Hailuo23Std.as_str(),
            "fal-ai/minimax/hailuo-2.3/standard/text-to-video"
        );
        assert_eq!(
            FalVideoModel::LtxVideo.as_str(),
            "fal-ai/ltx-2/text-to-video"
        );
    }

    #[test]
    fn test_fal_video_model_default() {
        assert_eq!(FalVideoModel::default(), FalVideoModel::Wan21);
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = FalVideoProviderBuilder::new()
            .api_key("test-key")
            .model(FalVideoModel::LtxVideo)
            .build();
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.model, FalVideoModel::LtxVideo);
    }

    #[test]
    fn test_builder_missing_key() {
        std::env::remove_var("FAL_KEY");

        let result = FalVideoProviderBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = FalVideoProviderBuilder::new()
            .api_key("test-key")
            .poll_interval(Duration::from_secs(5))
            .timeout(Duration::from_secs(900))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(5));
        assert_eq!(provider.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_request_serialization() {
        let req = FalVideoRequest {
            prompt: "A cat playing piano".to_string(),
            image_url: Some("https://example.com/cat.jpg".to_string()),
            duration: Some(5),
            aspect_ratio: Some("16:9".to_string()),
            negative_prompt: None,
            start_image_url: None,
            end_image_url: None,
            resolution: None,
            camera_fixed: None,
            seed: None,
            generate_audio: None,
            prompt_optimizer: None,
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["prompt"], "A cat playing piano");
        assert_eq!(json["image_url"], "https://example.com/cat.jpg");
        assert_eq!(json["duration"], 5);
        assert_eq!(json["aspect_ratio"], "16:9");
    }

    #[test]
    fn test_request_serialization_no_optional_fields() {
        let req = FalVideoRequest {
            prompt: "Ocean waves".to_string(),
            image_url: None,
            duration: None,
            aspect_ratio: None,
            negative_prompt: None,
            start_image_url: None,
            end_image_url: None,
            resolution: None,
            camera_fixed: None,
            seed: None,
            generate_audio: None,
            prompt_optimizer: None,
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["prompt"], "Ocean waves");
        assert!(json.get("image_url").is_none());
        assert!(json.get("duration").is_none());
        assert!(json.get("aspect_ratio").is_none());
        assert!(json.get("negative_prompt").is_none());
        assert!(json.get("start_image_url").is_none());
        assert!(json.get("end_image_url").is_none());
        assert!(json.get("resolution").is_none());
        assert!(json.get("camera_fixed").is_none());
        assert!(json.get("seed").is_none());
        assert!(json.get("generate_audio").is_none());
        assert!(json.get("prompt_optimizer").is_none());
    }

    #[test]
    fn test_submit_response_deserialization() {
        let json = r#"{
            "request_id": "req-abc-123",
            "status_url": "https://queue.fal.run/fal-ai/wan/requests/req-abc-123/status",
            "response_url": "https://queue.fal.run/fal-ai/wan/requests/req-abc-123"
        }"#;
        let resp: FalSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.request_id, "req-abc-123");
        assert!(resp.status_url.contains("status"));
        assert!(resp.response_url.contains("req-abc-123"));
    }

    #[test]
    fn test_video_result_deserialization() {
        let json = r#"{
            "video": {
                "url": "https://fal.media/files/video.mp4",
                "content_type": "video/mp4"
            }
        }"#;
        let resp: FalVideoResult = serde_json::from_str(json).unwrap();
        assert_eq!(resp.video.url, "https://fal.media/files/video.mp4");
    }

    #[test]
    fn test_status_response_deserialization() {
        let json = r#"{"status": "COMPLETED"}"#;
        let resp: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "COMPLETED");

        let json = r#"{"status": "IN_QUEUE"}"#;
        let resp: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "IN_QUEUE");

        let json = r#"{"status": "IN_PROGRESS"}"#;
        let resp: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "IN_PROGRESS");
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"detail": "Invalid API key provided"}"#;
        let resp: FalErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.detail, "Invalid API key provided");
    }

    #[test]
    fn test_parse_error_auth() {
        let provider = FalVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();

        let err = provider.parse_error(401, "Unauthorized");
        assert!(matches!(err, GenVizError::Auth(_)));

        let err = provider.parse_error(403, "Forbidden");
        assert!(matches!(err, GenVizError::Auth(_)));

        // Also test JSON-based auth error
        let err = provider.parse_error(401, r#"{"detail": "Unauthorized: invalid key format"}"#);
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_parse_error_rate_limited() {
        let provider = FalVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();

        let err = provider.parse_error(429, "Too many requests");
        assert!(matches!(err, GenVizError::RateLimited { .. }));

        // Also test JSON-based rate limit error
        let err = provider.parse_error(429, r#"{"detail": "Rate limit exceeded"}"#);
        assert!(matches!(err, GenVizError::RateLimited { .. }));
    }

    #[test]
    fn test_custom_model() {
        let custom = FalVideoModel::Custom("fal-ai/custom-video-model".to_string());
        assert_eq!(custom.as_str(), "fal-ai/custom-video-model");

        let provider = FalVideoProviderBuilder::new()
            .api_key("test-key")
            .model(custom.clone())
            .build()
            .unwrap();
        assert_eq!(provider.model, custom);
    }

    #[test]
    fn test_auto_model_selection() {
        // When source image is provided and model is Wan21 (text-to-video),
        // it should auto-switch to Wan21I2V (image-to-video).
        let request_with_image = VideoGenerationRequest::new("Animate this")
            .with_source_image("https://example.com/photo.jpg");

        // Wan21 should switch to Wan21I2V
        let effective = if request_with_image.source_image_url.is_some() {
            match &FalVideoModel::Wan21 {
                FalVideoModel::Wan21 => FalVideoModel::Wan21I2V,
                other => other.clone(),
            }
        } else {
            FalVideoModel::Wan21
        };
        assert_eq!(effective, FalVideoModel::Wan21I2V);

        // LtxVideo should NOT switch (it's not Wan21)
        let effective = if request_with_image.source_image_url.is_some() {
            match &FalVideoModel::LtxVideo {
                FalVideoModel::Wan21 => FalVideoModel::Wan21I2V,
                other => other.clone(),
            }
        } else {
            FalVideoModel::LtxVideo
        };
        assert_eq!(effective, FalVideoModel::LtxVideo);

        // Without source image, Wan21 should stay Wan21
        let request_no_image = VideoGenerationRequest::new("A sunset");
        let effective = if request_no_image.source_image_url.is_some() {
            match &FalVideoModel::Wan21 {
                FalVideoModel::Wan21 => FalVideoModel::Wan21I2V,
                other => other.clone(),
            }
        } else {
            FalVideoModel::Wan21
        };
        assert_eq!(effective, FalVideoModel::Wan21);
    }

    #[test]
    fn test_fal_video_model_wan21_flf2v_as_str() {
        assert_eq!(FalVideoModel::Wan21FLF2V.as_str(), "fal-ai/wan-flf2v");
    }

    #[test]
    fn test_fal_video_model_kling_as_str() {
        assert_eq!(
            FalVideoModel::KlingStd.as_str(),
            "fal-ai/kling-video/v1.6/standard/text-to-video"
        );
        assert_eq!(
            FalVideoModel::KlingPro.as_str(),
            "fal-ai/kling-video/v1.6/pro/text-to-video"
        );
    }

    #[test]
    fn test_request_serialization_with_negative_prompt() {
        let req = FalVideoRequest {
            prompt: "A sunset".to_string(),
            image_url: None,
            duration: None,
            aspect_ratio: None,
            negative_prompt: Some("people, buildings".to_string()),
            start_image_url: None,
            end_image_url: None,
            resolution: None,
            camera_fixed: None,
            seed: None,
            generate_audio: None,
            prompt_optimizer: None,
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["negative_prompt"], "people, buildings");
    }

    #[test]
    fn test_request_serialization_with_frame_urls() {
        let req = FalVideoRequest {
            prompt: "Interpolate".to_string(),
            image_url: None,
            duration: None,
            aspect_ratio: None,
            negative_prompt: None,
            start_image_url: Some("https://example.com/first.jpg".to_string()),
            end_image_url: Some("https://example.com/last.jpg".to_string()),
            resolution: None,
            camera_fixed: None,
            seed: None,
            generate_audio: None,
            prompt_optimizer: None,
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["start_image_url"], "https://example.com/first.jpg");
        assert_eq!(json["end_image_url"], "https://example.com/last.jpg");
    }

    #[test]
    fn test_auto_model_selection_flf2v() {
        // When source image + last frame URL provided and model is Wan21,
        // it should auto-switch to Wan21FLF2V.
        let request = VideoGenerationRequest::new("Interpolate")
            .with_source_image("https://example.com/first.jpg")
            .with_last_frame_url("https://example.com/last.jpg");

        let effective = if request.source_image_url.is_some() {
            match &FalVideoModel::Wan21 {
                FalVideoModel::Wan21 => {
                    if request.last_frame_url.is_some() {
                        FalVideoModel::Wan21FLF2V
                    } else {
                        FalVideoModel::Wan21I2V
                    }
                }
                other => other.clone(),
            }
        } else {
            FalVideoModel::Wan21
        };
        assert_eq!(effective, FalVideoModel::Wan21FLF2V);
    }

    #[test]
    fn test_auto_model_selection_kling_i2v() {
        // When Kling model has source image, i2v_variant should return I2V model ID.
        let i2v = FalVideoModel::KlingStd.i2v_variant();
        assert!(i2v.is_some());
        assert_eq!(
            i2v.unwrap().as_str(),
            "fal-ai/kling-video/v1.6/standard/image-to-video"
        );

        let i2v_pro = FalVideoModel::KlingPro.i2v_variant();
        assert!(i2v_pro.is_some());
        assert_eq!(
            i2v_pro.unwrap().as_str(),
            "fal-ai/kling-video/v1.6/pro/image-to-video"
        );

        // Models without i2v variant should return None
        assert!(FalVideoModel::Wan21.i2v_variant().is_none());
        assert!(FalVideoModel::LtxVideo.i2v_variant().is_none());
    }

    #[test]
    fn test_hailuo_model_as_str() {
        assert_eq!(
            FalVideoModel::Hailuo23Std.as_str(),
            "fal-ai/minimax/hailuo-2.3/standard/text-to-video"
        );
        assert_eq!(
            FalVideoModel::Hailuo23Pro.as_str(),
            "fal-ai/minimax/hailuo-2.3/pro/text-to-video"
        );
        assert_eq!(
            FalVideoModel::Hailuo23Fast.as_str(),
            "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video"
        );
    }

    #[test]
    fn test_seedance_model_as_str() {
        assert_eq!(
            FalVideoModel::SeedancePro.as_str(),
            "fal-ai/bytedance/seedance/v1/pro/text-to-video"
        );
        assert_eq!(
            FalVideoModel::SeedanceLite.as_str(),
            "fal-ai/bytedance/seedance/v1/lite/text-to-video"
        );
        assert_eq!(
            FalVideoModel::Seedance15Pro.as_str(),
            "fal-ai/bytedance/seedance/v1.5/pro/text-to-video"
        );
    }

    #[test]
    fn test_hailuo_i2v_variants() {
        let i2v_std = FalVideoModel::Hailuo23Std.i2v_variant();
        assert!(i2v_std.is_some());
        assert_eq!(
            i2v_std.unwrap().as_str(),
            "fal-ai/minimax/hailuo-2.3/standard/image-to-video"
        );

        let i2v_pro = FalVideoModel::Hailuo23Pro.i2v_variant();
        assert!(i2v_pro.is_some());
        assert_eq!(
            i2v_pro.unwrap().as_str(),
            "fal-ai/minimax/hailuo-2.3/pro/image-to-video"
        );

        // Hailuo23Fast is already i2v, so no variant
        assert!(FalVideoModel::Hailuo23Fast.i2v_variant().is_none());
    }

    #[test]
    fn test_seedance_i2v_variants() {
        let i2v_pro = FalVideoModel::SeedancePro.i2v_variant();
        assert!(i2v_pro.is_some());
        assert_eq!(
            i2v_pro.unwrap().as_str(),
            "fal-ai/bytedance/seedance/v1/pro/image-to-video"
        );

        let i2v_lite = FalVideoModel::SeedanceLite.i2v_variant();
        assert!(i2v_lite.is_some());
        assert_eq!(
            i2v_lite.unwrap().as_str(),
            "fal-ai/bytedance/seedance/v1/lite/image-to-video"
        );

        let i2v_15 = FalVideoModel::Seedance15Pro.i2v_variant();
        assert!(i2v_15.is_some());
        assert_eq!(
            i2v_15.unwrap().as_str(),
            "fal-ai/bytedance/seedance/v1.5/pro/image-to-video"
        );
    }

    #[test]
    fn test_request_serialization_with_new_fields() {
        let req = FalVideoRequest {
            prompt: "Dancing robot".to_string(),
            image_url: None,
            duration: Some(5),
            aspect_ratio: None,
            negative_prompt: None,
            start_image_url: None,
            end_image_url: None,
            resolution: Some("720p".to_string()),
            camera_fixed: Some(true),
            seed: Some(42),
            generate_audio: Some(true),
            prompt_optimizer: Some(false),
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["resolution"], "720p");
        assert_eq!(json["camera_fixed"], true);
        assert_eq!(json["seed"], 42);
        assert_eq!(json["generate_audio"], true);
        assert_eq!(json["prompt_optimizer"], false);
    }

    #[test]
    fn test_auto_model_selection_hailuo_i2v() {
        // Hailuo23Std with source image should switch to i2v via i2v_variant().
        let request = VideoGenerationRequest::new("Animate this")
            .with_source_image("https://example.com/photo.jpg");

        let mut effective = FalVideoModel::Hailuo23Std;
        if request.source_image_url.is_some() {
            if let Some(i2v) = effective.i2v_variant() {
                effective = i2v;
            }
        }
        assert_eq!(
            effective.as_str(),
            "fal-ai/minimax/hailuo-2.3/standard/image-to-video"
        );

        // Hailuo23Fast is already i2v, should stay the same.
        let mut effective = FalVideoModel::Hailuo23Fast;
        if request.source_image_url.is_some() {
            if let Some(i2v) = effective.i2v_variant() {
                effective = i2v;
            }
        }
        assert_eq!(effective, FalVideoModel::Hailuo23Fast);
    }

    #[test]
    fn test_auto_model_selection_seedance_i2v() {
        let request = VideoGenerationRequest::new("Animate this")
            .with_source_image("https://example.com/photo.jpg");

        let mut effective = FalVideoModel::SeedancePro;
        if request.source_image_url.is_some() {
            if let Some(i2v) = effective.i2v_variant() {
                effective = i2v;
            }
        }
        assert_eq!(
            effective.as_str(),
            "fal-ai/bytedance/seedance/v1/pro/image-to-video"
        );

        let mut effective = FalVideoModel::Seedance15Pro;
        if request.source_image_url.is_some() {
            if let Some(i2v) = effective.i2v_variant() {
                effective = i2v;
            }
        }
        assert_eq!(
            effective.as_str(),
            "fal-ai/bytedance/seedance/v1.5/pro/image-to-video"
        );
    }

    // -- health_check tests --

    #[tokio::test]
    async fn test_health_check_empty_key() {
        let provider = FalVideoProvider {
            client: reqwest::Client::new(),
            api_key: String::new(),
            model: FalVideoModel::default(),
            poll_interval: Duration::from_secs(3),
            timeout: Duration::from_secs(600),
        };
        let result = provider.health_check().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_health_check_with_key() {
        let provider = FalVideoProviderBuilder::new()
            .api_key("fal-test-key")
            .build()
            .unwrap();
        let result = provider.health_check().await;
        assert!(result.is_ok());
    }
}
