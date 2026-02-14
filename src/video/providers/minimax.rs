//! MiniMax Hailuo video generation provider (direct API).

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const SUBMIT_URL: &str = "https://api.minimax.io/v1/video_generation";
const POLL_URL: &str = "https://api.minimax.io/v1/query/video_generation";
const FILE_URL: &str = "https://api.minimax.io/v1/files/retrieve";

/// MiniMax Hailuo video model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MiniMaxVideoModel {
    /// Hailuo 2.3 — default model.
    #[default]
    Hailuo23,
    /// Hailuo 2.3 Fast — faster generation.
    Hailuo23Fast,
}

impl MiniMaxVideoModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hailuo23 => "hailuo-2.3",
            Self::Hailuo23Fast => "hailuo-2.3-fast",
        }
    }
}

/// Builder for `MiniMaxVideoProvider`.
#[derive(Debug, Clone)]
pub struct MiniMaxVideoProviderBuilder {
    api_key: Option<String>,
    model: MiniMaxVideoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for MiniMaxVideoProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: MiniMaxVideoModel::default(),
            poll_interval: Duration::from_secs(3),
            timeout: Duration::from_secs(600), // 10 minutes for video
        }
    }
}

impl MiniMaxVideoProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `MINIMAX_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the MiniMax video model variant.
    pub fn model(mut self, model: MiniMaxVideoModel) -> Self {
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
    pub fn build(self) -> Result<MiniMaxVideoProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("MINIMAX_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("MINIMAX_API_KEY not set and no API key provided".into())
            })?;

        Ok(MiniMaxVideoProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// MiniMax Hailuo video generation provider.
#[derive(Debug)]
pub struct MiniMaxVideoProvider {
    client: reqwest::Client,
    api_key: String,
    model: MiniMaxVideoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl MiniMaxVideoProvider {
    /// Creates a new `MiniMaxVideoProviderBuilder`.
    pub fn builder() -> MiniMaxVideoProviderBuilder {
        MiniMaxVideoProviderBuilder::new()
    }

    /// Submit a video generation request.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        let body = MiniMaxVideoRequest::from_request(request, &self.model);

        let response = self
            .client
            .post(SUBMIT_URL)
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

        let submit_response: MiniMaxSubmitResponse = response.json().await?;

        // Check base_resp for API-level errors (status_code != 0 means error)
        if let Some(ref base_resp) = submit_response.base_resp {
            if base_resp.status_code != 0 {
                return Err(GenVizError::Api {
                    status: base_resp.status_code as u16,
                    message: sanitize_error_message(&base_resp.status_msg),
                });
            }
        }

        Ok(submit_response.task_id)
    }

    /// Poll until the video is ready, returning the file_id.
    async fn poll_until_ready(&self, task_id: &str) -> Result<String> {
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(POLL_URL)
                .query(&[("task_id", task_id)])
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text, &headers));
            }

            let poll_response: MiniMaxPollResponse = response.json().await?;

            match poll_response.status.as_str() {
                "Processing" => {
                    tracing::debug!(
                        task_id = %task_id,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling MiniMax video generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                    continue;
                }
                "Success" => {
                    let file_id = poll_response
                        .file_id
                        .filter(|id| !id.is_empty())
                        .ok_or_else(|| {
                            GenVizError::UnexpectedResponse(
                                "MiniMax poll returned Success but no file_id".into(),
                            )
                        })?;
                    return Ok(file_id);
                }
                "Fail" => {
                    let msg = poll_response
                        .base_resp
                        .map(|r| r.status_msg)
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| "video generation failed".into());
                    return Err(GenVizError::VideoGeneration(sanitize_error_message(&msg)));
                }
                other => {
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "unknown MiniMax poll status: {other}"
                    )));
                }
            }
        }
    }

    /// Fetch the download URL for a file_id.
    async fn fetch_file_url(&self, file_id: &str) -> Result<String> {
        let response = self
            .client
            .get(FILE_URL)
            .query(&[("file_id", file_id)])
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let file_response: MiniMaxFileResponse = response.json().await?;
        Ok(file_response.file.download_url)
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

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        if status == 422 {
            return GenVizError::InvalidRequest(text);
        }
        if status == 429 {
            let retry_after = parse_retry_after(headers).map(Duration::from_secs);
            return GenVizError::RateLimited { retry_after };
        }
        let lower = text.to_lowercase();
        if lower.contains("safety")
            || lower.contains("blocked")
            || lower.contains("content_policy")
            || lower.contains("moderated")
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
impl VideoProvider for MiniMaxVideoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        // Validate mutually exclusive modes
        let has_image = request.source_image_url.is_some();
        let has_subject_ref = request
            .subject_reference
            .as_ref()
            .is_some_and(|v| !v.is_empty());
        if has_image && has_subject_ref {
            return Err(GenVizError::InvalidRequest(
                "MiniMax: source image and subject_reference are mutually exclusive".into(),
            ));
        }

        let start = Instant::now();

        // Submit the request
        let task_id = self.submit(request).await?;
        tracing::debug!(task_id = %task_id, "submitted MiniMax video generation request");

        // Poll until ready
        let file_id = self.poll_until_ready(&task_id).await?;
        tracing::debug!(file_id = %file_id, "MiniMax video generation complete");

        // Fetch the download URL
        let download_url = self.fetch_file_url(&file_id).await?;
        tracing::debug!(url = %download_url, "fetched MiniMax video download URL");

        // Download the video
        let data = self.download(&download_url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::MiniMax,
            VideoMetadata {
                model: Some(self.model.as_str().to_string()),
                duration_ms: Some(duration_ms),
                video_duration_secs: request.duration_secs,
                resolution: request.resolution.clone(),
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::MiniMax
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.is_empty() {
            Err(GenVizError::Auth("API key is empty".into()))
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MiniMaxVideoRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_frame_image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_frame_image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_optimizer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    subject_reference: Option<Vec<MiniMaxSubjectRef>>,
}

#[derive(Debug, Serialize)]
struct MiniMaxSubjectRef {
    #[serde(rename = "type")]
    ref_type: String,
    image: String,
}

/// Normalize resolution strings to MiniMax format.
/// "720p" → "768P", "1080p" → "1080P", otherwise pass through.
fn normalize_resolution(res: &str) -> String {
    match res {
        "720p" => "768P".to_string(),
        "1080p" => "1080P".to_string(),
        other => other.to_string(),
    }
}

impl MiniMaxVideoRequest {
    fn from_request(req: &VideoGenerationRequest, model: &MiniMaxVideoModel) -> Self {
        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            first_frame_image: req.source_image_url.clone(),
            last_frame_image: req.last_frame_url.clone(),
            resolution: req.resolution.as_deref().map(normalize_resolution),
            prompt_optimizer: req.prompt_optimizer,
            subject_reference: req
                .subject_reference
                .as_ref()
                .filter(|refs| !refs.is_empty())
                .map(|refs| {
                    refs.iter()
                        .map(|r| MiniMaxSubjectRef {
                            ref_type: r.ref_type.clone(),
                            image: r.image.clone(),
                        })
                        .collect()
                }),
        }
    }
}

// Submit response
#[derive(Debug, Deserialize)]
struct MiniMaxSubmitResponse {
    task_id: String,
    #[serde(default)]
    base_resp: Option<MiniMaxBaseResp>,
}

#[derive(Debug, Deserialize)]
struct MiniMaxBaseResp {
    #[serde(default)]
    status_code: i32,
    #[serde(default)]
    status_msg: String,
}

// Poll response
#[derive(Debug, Deserialize)]
struct MiniMaxPollResponse {
    status: String,
    #[serde(default)]
    file_id: Option<String>,
    #[serde(default)]
    base_resp: Option<MiniMaxBaseResp>,
}

// File retrieve response
#[derive(Debug, Deserialize)]
struct MiniMaxFileResponse {
    file: MiniMaxFileInfo,
}

#[derive(Debug, Deserialize)]
struct MiniMaxFileInfo {
    download_url: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_video_model_as_str() {
        assert_eq!(MiniMaxVideoModel::Hailuo23.as_str(), "hailuo-2.3");
        assert_eq!(MiniMaxVideoModel::Hailuo23Fast.as_str(), "hailuo-2.3-fast");
    }

    #[test]
    fn test_default_model() {
        assert_eq!(MiniMaxVideoModel::default(), MiniMaxVideoModel::Hailuo23);
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("mm-test-key")
            .build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_builder_missing_key() {
        // Clear the env var if it exists, then check the builder fails
        let saved = std::env::var("MINIMAX_API_KEY").ok();
        std::env::remove_var("MINIMAX_API_KEY");

        let result = MiniMaxVideoProviderBuilder::new().build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("MINIMAX_API_KEY"),
            "error should mention MINIMAX_API_KEY: {err}"
        );

        // Restore env var if it was set
        if let Some(val) = saved {
            std::env::set_var("MINIMAX_API_KEY", val);
        }
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("mm-test-key")
            .poll_interval(Duration::from_secs(5))
            .timeout(Duration::from_secs(900))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(5));
        assert_eq!(provider.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_builder_default_timeouts() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("mm-test-key")
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(3));
        assert_eq!(provider.timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_builder_with_model() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("mm-test-key")
            .model(MiniMaxVideoModel::Hailuo23Fast)
            .build()
            .unwrap();
        assert_eq!(provider.model, MiniMaxVideoModel::Hailuo23Fast);
    }

    // -- Request construction for all 4 modes --

    #[test]
    fn test_request_t2v_prompt_only() {
        let req = VideoGenerationRequest::new("A sunset over the ocean");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);

        assert_eq!(mm_req.prompt, "A sunset over the ocean");
        assert_eq!(mm_req.model, "hailuo-2.3");
        assert!(mm_req.first_frame_image.is_none());
        assert!(mm_req.last_frame_image.is_none());
        assert!(mm_req.subject_reference.is_none());
        assert!(mm_req.resolution.is_none());
        assert!(mm_req.prompt_optimizer.is_none());
    }

    #[test]
    fn test_request_i2v_with_first_frame() {
        let req = VideoGenerationRequest::new("Animate this scene")
            .with_source_image("https://example.com/photo.jpg");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);

        assert_eq!(mm_req.prompt, "Animate this scene");
        assert_eq!(
            mm_req.first_frame_image.as_deref(),
            Some("https://example.com/photo.jpg")
        );
        assert!(mm_req.last_frame_image.is_none());
        assert!(mm_req.subject_reference.is_none());
    }

    #[test]
    fn test_request_flf_first_and_last_frame() {
        let req = VideoGenerationRequest::new("Transition between frames")
            .with_source_image("https://example.com/start.jpg")
            .with_last_frame_url("https://example.com/end.jpg");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);

        assert_eq!(
            mm_req.first_frame_image.as_deref(),
            Some("https://example.com/start.jpg")
        );
        assert_eq!(
            mm_req.last_frame_image.as_deref(),
            Some("https://example.com/end.jpg")
        );
    }

    #[test]
    fn test_request_subject_reference() {
        let req = VideoGenerationRequest::new("A character walking")
            .with_subject_reference("character", "https://example.com/face.jpg");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23Fast);

        assert_eq!(mm_req.model, "hailuo-2.3-fast");
        let refs = mm_req.subject_reference.unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].ref_type, "character");
        assert_eq!(refs[0].image, "https://example.com/face.jpg");
    }

    // -- Resolution normalization --

    #[test]
    fn test_resolution_normalization_720p() {
        assert_eq!(normalize_resolution("720p"), "768P");
    }

    #[test]
    fn test_resolution_normalization_1080p() {
        assert_eq!(normalize_resolution("1080p"), "1080P");
    }

    #[test]
    fn test_resolution_normalization_passthrough() {
        assert_eq!(normalize_resolution("4K"), "4K");
        assert_eq!(normalize_resolution("768P"), "768P");
    }

    #[test]
    fn test_request_with_resolution() {
        let req = VideoGenerationRequest::new("A scene").with_resolution("720p");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);
        assert_eq!(mm_req.resolution.as_deref(), Some("768P"));
    }

    #[test]
    fn test_request_with_prompt_optimizer() {
        let req = VideoGenerationRequest::new("A scene").with_prompt_optimizer(true);
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);
        assert_eq!(mm_req.prompt_optimizer, Some(true));
    }

    // -- Submit response deserialization --

    #[test]
    fn test_submit_response_success() {
        let json = r#"{"task_id": "task-abc123"}"#;
        let resp: MiniMaxSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.task_id, "task-abc123");
        assert!(resp.base_resp.is_none());
    }

    #[test]
    fn test_submit_response_with_base_resp_ok() {
        let json =
            r#"{"task_id": "task-abc123", "base_resp": {"status_code": 0, "status_msg": ""}}"#;
        let resp: MiniMaxSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.task_id, "task-abc123");
        assert_eq!(resp.base_resp.as_ref().unwrap().status_code, 0);
    }

    #[test]
    fn test_submit_response_with_base_resp_error() {
        let json = r#"{"task_id": "", "base_resp": {"status_code": 1001, "status_msg": "invalid parameter"}}"#;
        let resp: MiniMaxSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.base_resp.as_ref().unwrap().status_code, 1001);
        assert_eq!(
            resp.base_resp.as_ref().unwrap().status_msg,
            "invalid parameter"
        );
    }

    // -- Poll response deserialization --

    #[test]
    fn test_poll_response_processing() {
        let json = r#"{"status": "Processing"}"#;
        let resp: MiniMaxPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Processing");
        assert!(resp.file_id.is_none());
    }

    #[test]
    fn test_poll_response_success() {
        let json = r#"{"status": "Success", "file_id": "file-xyz789"}"#;
        let resp: MiniMaxPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Success");
        assert_eq!(resp.file_id.as_deref(), Some("file-xyz789"));
    }

    #[test]
    fn test_poll_response_fail() {
        let json = r#"{"status": "Fail", "base_resp": {"status_code": 2001, "status_msg": "content moderated"}}"#;
        let resp: MiniMaxPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Fail");
        assert_eq!(
            resp.base_resp.as_ref().unwrap().status_msg,
            "content moderated"
        );
    }

    // -- File response deserialization --

    #[test]
    fn test_file_response() {
        let json = r#"{"file": {"download_url": "https://cdn.minimax.io/video/abc.mp4"}}"#;
        let resp: MiniMaxFileResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.file.download_url,
            "https://cdn.minimax.io/video/abc.mp4"
        );
    }

    // -- Request serialization (skip_serializing_if) --

    #[test]
    fn test_request_serialization_minimal() {
        let req = VideoGenerationRequest::new("A cat");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);
        let json = serde_json::to_value(&mm_req).unwrap();

        // Required fields present
        assert_eq!(json["model"], "hailuo-2.3");
        assert_eq!(json["prompt"], "A cat");

        // Optional fields absent
        assert!(json.get("first_frame_image").is_none());
        assert!(json.get("last_frame_image").is_none());
        assert!(json.get("resolution").is_none());
        assert!(json.get("prompt_optimizer").is_none());
        assert!(json.get("subject_reference").is_none());
    }

    #[test]
    fn test_request_serialization_full() {
        let req = VideoGenerationRequest::new("A scene")
            .with_source_image("https://example.com/img.jpg")
            .with_last_frame_url("https://example.com/end.jpg")
            .with_resolution("1080p")
            .with_prompt_optimizer(true)
            .with_subject_reference("character", "https://example.com/face.jpg");
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23Fast);
        let json = serde_json::to_value(&mm_req).unwrap();

        assert_eq!(json["model"], "hailuo-2.3-fast");
        assert_eq!(json["first_frame_image"], "https://example.com/img.jpg");
        assert_eq!(json["last_frame_image"], "https://example.com/end.jpg");
        assert_eq!(json["resolution"], "1080P");
        assert_eq!(json["prompt_optimizer"], true);

        let refs = json["subject_reference"].as_array().unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0]["type"], "character");
        assert_eq!(refs[0]["image"], "https://example.com/face.jpg");
    }

    // -- parse_error tests --

    #[test]
    fn test_parse_error_auth_401() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(401, "Unauthorized", &headers);
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_parse_error_auth_403() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(403, "Forbidden", &headers);
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_parse_error_invalid_request_422() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(422, "Unprocessable Entity", &headers);
        assert!(matches!(err, GenVizError::InvalidRequest(_)));
    }

    #[test]
    fn test_parse_error_rate_limited_429() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(429, "Too Many Requests", &headers);
        assert!(matches!(err, GenVizError::RateLimited { .. }));
    }

    #[test]
    fn test_parse_error_content_blocked() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(500, "content_blocked by safety filter", &headers);
        assert!(matches!(err, GenVizError::ContentBlocked(_)));
    }

    #[test]
    fn test_parse_error_generic_api_error() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(500, "Internal Server Error", &headers);
        assert!(matches!(err, GenVizError::Api { status: 500, .. }));
    }

    // -- Empty subject_reference vec treated as None --

    #[test]
    fn test_empty_subject_reference_vec_treated_as_none() {
        let mut req = VideoGenerationRequest::new("A scene");
        req.subject_reference = Some(vec![]);
        let mm_req = MiniMaxVideoRequest::from_request(&req, &MiniMaxVideoModel::Hailuo23);
        assert!(
            mm_req.subject_reference.is_none(),
            "empty subject_reference vec should be filtered to None"
        );
    }

    // -- Multiple subject references --

    #[test]
    fn test_multiple_subject_references() {
        let req = VideoGenerationRequest::new("Two characters")
            .with_subject_reference("character", "https://example.com/face1.jpg")
            .with_subject_reference("character", "https://example.com/face2.jpg");
        let refs = req.subject_reference.as_ref().unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].image, "https://example.com/face1.jpg");
        assert_eq!(refs[1].image, "https://example.com/face2.jpg");
    }

    // -- health_check tests --

    #[tokio::test]
    async fn test_health_check_empty_key() {
        let provider = MiniMaxVideoProvider {
            client: reqwest::Client::new(),
            api_key: String::new(),
            model: MiniMaxVideoModel::default(),
            poll_interval: Duration::from_secs(3),
            timeout: Duration::from_secs(600),
        };
        let result = provider.health_check().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_health_check_with_key() {
        let provider = MiniMaxVideoProviderBuilder::new()
            .api_key("mm-test-key")
            .build()
            .unwrap();
        let result = provider.health_check().await;
        assert!(result.is_ok());
    }
}
