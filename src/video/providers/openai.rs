//! Sora (OpenAI) video generation provider.

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const BASE_URL: &str = "https://api.openai.com/v1/videos";

/// Sora model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SoraModel {
    /// Sora 2 - OpenAI's video generation model.
    #[default]
    Sora2,
}

impl SoraModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sora2 => "sora-2",
        }
    }
}

/// Builder for SoraProvider.
#[derive(Debug, Clone)]
pub struct SoraProviderBuilder {
    api_key: Option<String>,
    model: SoraModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for SoraProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: SoraModel::default(),
            poll_interval: Duration::from_secs(5),
            timeout: Duration::from_secs(600), // 10 minutes for video
        }
    }
}

impl SoraProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `OPENAI_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the Sora model variant.
    pub fn model(mut self, model: SoraModel) -> Self {
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
    pub fn build(self) -> Result<SoraProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("OPENAI_API_KEY not set and no API key provided".into())
            })?;

        Ok(SoraProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// Sora video generation provider.
pub struct SoraProvider {
    client: reqwest::Client,
    api_key: String,
    model: SoraModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl SoraProvider {
    /// Creates a new `SoraProviderBuilder`.
    pub fn builder() -> SoraProviderBuilder {
        SoraProviderBuilder::new()
    }

    /// Submit a video generation request. Returns the video generation ID.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        let body = SoraRequest::from_request(request, &self.model);

        let response = self
            .client
            .post(BASE_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let submit_response: SoraSubmitResponse = response.json().await?;
        Ok(submit_response.id)
    }

    /// Poll until the video generation is complete. Returns the video ID.
    async fn poll_until_ready(&self, video_id: &str) -> Result<()> {
        let url = format!("{}/{}", BASE_URL, video_id);
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text, &headers));
            }

            let poll_response: SoraPollResponse = response.json().await?;

            match poll_response.status.as_str() {
                "completed" => {
                    return Ok(());
                }
                "failed" => {
                    // Check error object for moderation blocks
                    if let Some(ref err) = poll_response.error {
                        if let Some(ref code) = err.code {
                            if code == "moderation_blocked" || code == "sentinel_block" {
                                let msg = err
                                    .message
                                    .clone()
                                    .unwrap_or_else(|| "Content blocked by moderation".into());
                                return Err(GenVizError::ContentBlocked(msg));
                            }
                        }
                    }

                    let message = poll_response
                        .failure_reason
                        .or_else(|| poll_response.error.and_then(|e| e.message))
                        .unwrap_or_else(|| "Unknown error".into());
                    return Err(GenVizError::VideoGeneration(message));
                }
                "queued" | "in_progress" => {
                    tracing::debug!(
                        video_id = %video_id,
                        status = %poll_response.status,
                        progress = poll_response.progress,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling Sora video generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                }
                other => {
                    return Err(GenVizError::VideoGeneration(format!(
                        "Unexpected status: {}",
                        other
                    )));
                }
            }
        }
    }

    /// Download the video via the content endpoint.
    async fn download(&self, video_id: &str) -> Result<Vec<u8>> {
        let url = format!("{}/{}/content", BASE_URL, video_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(GenVizError::Api {
                status: response.status().as_u16(),
                message: "Failed to download video".into(),
            });
        }

        Ok(response.bytes().await?.to_vec())
    }

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 402 {
            return GenVizError::Billing(sanitize_error_message(&text));
        }
        if status == 429 {
            // Distinguish insufficient_quota (not retryable) from transient rate limit
            if text.contains("insufficient_quota") || text.contains("exceeded your current quota") {
                return GenVizError::Billing(sanitize_error_message(&text));
            }
            let retry_after = parse_retry_after(headers).map(std::time::Duration::from_secs);
            return GenVizError::RateLimited { retry_after };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        let lower = text.to_lowercase();
        if lower.contains("safety") || lower.contains("blocked") || lower.contains("content_policy")
        {
            return GenVizError::ContentBlocked(text);
        }
        GenVizError::Api {
            status,
            message: text,
        }
    }
}

#[async_trait]
impl VideoProvider for SoraProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();

        // Submit the request
        let video_id = self.submit(request).await?;
        tracing::debug!(video_id = %video_id, "submitted Sora video generation request");

        // Poll until ready
        self.poll_until_ready(&video_id).await?;
        tracing::debug!(video_id = %video_id, "Sora video generation complete");

        // Download via /content endpoint
        let data = self.download(&video_id).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::OpenAI,
            VideoMetadata {
                model: Some(self.model.as_str().to_string()),
                duration_ms: Some(duration_ms),
                video_duration_secs: request.duration_secs,
                resolution: None,
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::OpenAI
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.starts_with("sk-") {
            Ok(())
        } else {
            Err(GenVizError::Auth("Invalid API key format".into()))
        }
    }
}

// Request/Response types

#[derive(Debug, Serialize)]
struct SoraRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    /// Video duration: "4", "8", or "12" seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<String>,
}

impl SoraRequest {
    /// Valid Sora duration values in seconds.
    const VALID_DURATIONS: [u32; 3] = [4, 8, 12];

    fn from_request(req: &VideoGenerationRequest, model: &SoraModel) -> Self {
        let size = req.aspect_ratio.as_ref().map(|ar| match ar.as_str() {
            "16:9" => "1920x1080".to_string(),
            "9:16" => "1080x1920".to_string(),
            "1:1" => "1080x1080".to_string(),
            other => other.to_string(),
        });

        // Map duration_secs to nearest valid Sora value ("4", "8", "12")
        let seconds = req.duration_secs.map(|d| {
            let nearest = Self::VALID_DURATIONS
                .iter()
                .min_by_key(|&&v| (v as i64 - d as i64).unsigned_abs())
                .copied()
                .unwrap_or(4);
            nearest.to_string()
        });

        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            size,
            seconds,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SoraSubmitResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct SoraPollResponse {
    status: String,
    #[serde(default)]
    progress: Option<u32>,
    #[serde(default)]
    failure_reason: Option<String>,
    #[serde(default)]
    error: Option<SoraError>,
}

#[derive(Debug, Deserialize)]
struct SoraError {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sora_model_as_str() {
        assert_eq!(SoraModel::Sora2.as_str(), "sora-2");
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = SoraProviderBuilder::new().api_key("sk-test").build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_builder_without_key_fails() {
        std::env::remove_var("OPENAI_API_KEY");
        let provider = SoraProviderBuilder::new().build();
        assert!(provider.is_err());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = SoraProviderBuilder::new()
            .api_key("sk-test")
            .poll_interval(Duration::from_secs(10))
            .timeout(Duration::from_secs(900))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(10));
        assert_eq!(provider.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_request_construction_basic() {
        let req = VideoGenerationRequest::new("A flying bird");
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);

        assert_eq!(sora_req.prompt, "A flying bird");
        assert_eq!(sora_req.model, "sora-2");
        assert!(sora_req.size.is_none());
        assert!(sora_req.seconds.is_none());
    }

    #[test]
    fn test_request_duration_exact_match() {
        let req = VideoGenerationRequest::new("test").with_duration(8);
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        assert_eq!(sora_req.seconds.as_deref(), Some("8"));
    }

    #[test]
    fn test_request_duration_rounds_to_nearest() {
        // 5 is closer to 4 than to 8
        let req = VideoGenerationRequest::new("test").with_duration(5);
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        assert_eq!(sora_req.seconds.as_deref(), Some("4"));

        // 7 is closer to 8 than to 4
        let req = VideoGenerationRequest::new("test").with_duration(7);
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        assert_eq!(sora_req.seconds.as_deref(), Some("8"));

        // 10 is equidistant from 8 and 12; min_by_key picks first (8)
        let req = VideoGenerationRequest::new("test").with_duration(10);
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        assert_eq!(sora_req.seconds.as_deref(), Some("8"));

        // 15 rounds down to 12
        let req = VideoGenerationRequest::new("test").with_duration(15);
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        assert_eq!(sora_req.seconds.as_deref(), Some("12"));
    }

    #[test]
    fn test_request_construction_with_aspect_ratio() {
        let req = VideoGenerationRequest::new("A flying bird").with_aspect_ratio("16:9");
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);

        assert_eq!(sora_req.size.as_deref(), Some("1920x1080"));
    }

    #[test]
    fn test_request_construction_portrait() {
        let req = VideoGenerationRequest::new("test").with_aspect_ratio("9:16");
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);

        assert_eq!(sora_req.size.as_deref(), Some("1080x1920"));
    }

    #[test]
    fn test_request_construction_square() {
        let req = VideoGenerationRequest::new("test").with_aspect_ratio("1:1");
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);

        assert_eq!(sora_req.size.as_deref(), Some("1080x1080"));
    }

    #[test]
    fn test_submit_response_deserialization() {
        let json = r#"{"id": "video_abc123", "status": "queued", "created_at": 1234567890}"#;
        let resp: SoraSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "video_abc123");
    }

    #[test]
    fn test_poll_response_completed() {
        let json = r#"{"status": "completed", "progress": 100}"#;
        let resp: SoraPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "completed");
        assert_eq!(resp.progress, Some(100));
        assert!(resp.failure_reason.is_none());
    }

    #[test]
    fn test_poll_response_in_progress() {
        let json = r#"{"status": "in_progress", "progress": 42}"#;
        let resp: SoraPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "in_progress");
        assert_eq!(resp.progress, Some(42));
    }

    #[test]
    fn test_poll_response_failed() {
        let json = r#"{"status": "failed", "failure_reason": "Content policy violation"}"#;
        let resp: SoraPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "failed");
        assert_eq!(
            resp.failure_reason.as_deref(),
            Some("Content policy violation")
        );
    }

    #[test]
    fn test_request_serialization_skips_none_fields() {
        let req = VideoGenerationRequest::new("test");
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        let json = serde_json::to_value(&sora_req).unwrap();

        assert!(json.get("size").is_none());
        assert!(json.get("seconds").is_none());
        assert!(json.get("prompt").is_some());
        assert!(json.get("model").is_some());
    }

    #[test]
    fn test_request_serialization_with_seconds() {
        let req = VideoGenerationRequest::new("test").with_duration(4);
        let sora_req = SoraRequest::from_request(&req, &SoraModel::Sora2);
        let json = serde_json::to_value(&sora_req).unwrap();

        assert_eq!(json.get("seconds").unwrap().as_str(), Some("4"));
        // Ensure old "duration" field is not present
        assert!(json.get("duration").is_none());
        // Ensure "n" field is not present
        assert!(json.get("n").is_none());
    }
}
