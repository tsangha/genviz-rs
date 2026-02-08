//! Veo (Google) video generation provider.

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Veo model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VeoModel {
    /// Veo 3.1 Preview - Google's video generation model.
    #[default]
    Veo31Preview,
}

impl VeoModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Veo31Preview => "veo-3.1-generate-preview",
        }
    }
}

/// Builder for VeoProvider.
#[derive(Debug, Clone)]
pub struct VeoProviderBuilder {
    api_key: Option<String>,
    model: VeoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for VeoProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: VeoModel::default(),
            poll_interval: Duration::from_secs(10),
            timeout: Duration::from_secs(600), // 10 minutes for video
        }
    }
}

impl VeoProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `GOOGLE_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the Veo model variant.
    pub fn model(mut self, model: VeoModel) -> Self {
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
    pub fn build(self) -> Result<VeoProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("GOOGLE_API_KEY not set and no API key provided".into())
            })?;

        Ok(VeoProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// Veo video generation provider.
pub struct VeoProvider {
    client: reqwest::Client,
    api_key: String,
    model: VeoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl VeoProvider {
    /// Creates a new `VeoProviderBuilder`.
    pub fn builder() -> VeoProviderBuilder {
        VeoProviderBuilder::new()
    }

    /// Submit a video generation request.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:predictLongRunning",
            self.model.as_str(),
        );

        let body = VeoRequest::from_request(request);

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
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

        let operation: VeoOperationResponse = response.json().await?;
        Ok(operation.name)
    }

    /// Poll until the video is ready.
    async fn poll_until_ready(&self, operation_name: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}",
            operation_name,
        );
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(&url)
                .header("x-goog-api-key", &self.api_key)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text, &headers));
            }

            let operation: VeoOperationResponse = response.json().await?;

            if operation.done.unwrap_or(false) {
                // Check for error FIRST before checking response
                if let Some(err) = operation.error {
                    return Err(GenVizError::VideoGeneration(
                        err.message.unwrap_or_else(|| "Unknown error".into()),
                    ));
                }

                if let Some(resp) = operation.response {
                    if let Some(gen_resp) = resp.generate_video_response {
                        // Check if content was filtered
                        if gen_resp.rai_media_filtered_count.unwrap_or(0) > 0
                            && gen_resp
                                .generated_samples
                                .as_ref()
                                .is_none_or(|s| s.is_empty())
                        {
                            return Err(GenVizError::ContentBlocked(
                                "Video was filtered by Veo safety filters".into(),
                            ));
                        }

                        if let Some(samples) = gen_resp.generated_samples {
                            if let Some(first) = samples.into_iter().next() {
                                if let Some(uri) = first.video.and_then(|v| v.uri) {
                                    return Ok(uri);
                                }
                            }
                        }
                    }
                }
                return Err(GenVizError::UnexpectedResponse(
                    "Video generation completed but no video URL returned".into(),
                ));
            }

            if let Some(err) = operation.error {
                return Err(GenVizError::VideoGeneration(
                    err.message.unwrap_or_else(|| "Unknown error".into()),
                ));
            }

            tracing::debug!(
                operation = %operation_name,
                elapsed_secs = start.elapsed().as_secs(),
                "polling Veo video generation"
            );
            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Download the video from the given URL.
    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        if url.starts_with("gs://") {
            return Err(GenVizError::VideoGeneration(format!(
                "Veo returned a Google Cloud Storage URI ({}) which cannot be downloaded directly. \
                 Use `gsutil cp` or the Google Cloud Storage API to download the video.",
                url
            )));
        }

        // Append API key as query parameter (known SDK requirement)
        let url = if url.contains('?') {
            format!("{}&key={}", url, self.api_key)
        } else {
            format!("{}?key={}", url, self.api_key)
        };

        let response = self
            .client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
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
        if status == 402 {
            return GenVizError::Billing(
                "Veo billing issue: enable billing at https://aistudio.google.com".into(),
            );
        }
        if status == 404 {
            return GenVizError::InvalidRequest(
                "Veo API not available. Veo requires a paid-tier API key with billing enabled. \
                 Enable it at https://aistudio.google.com by selecting a Google Cloud project with billing."
                    .to_string(),
            );
        }
        let text = sanitize_error_message(text);
        if status == 429 {
            let retry_after = parse_retry_after(headers).map(std::time::Duration::from_secs);
            return GenVizError::RateLimited { retry_after };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        let lower = text.to_lowercase();
        if lower.contains("safety")
            || lower.contains("blocked")
            || lower.contains("content_policy")
            || lower.contains("prohibited")
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
impl VideoProvider for VeoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();

        // Submit the request
        let operation_name = self.submit(request).await?;
        tracing::debug!(operation = %operation_name, "submitted video generation request");

        // Poll until ready
        let video_url = self.poll_until_ready(&operation_name).await?;
        tracing::debug!(url = %video_url, "video generation complete");

        // Download the video
        let data = self.download(&video_url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::Veo,
            VideoMetadata {
                model: Some(self.model.as_str().to_string()),
                duration_ms: Some(duration_ms),
                video_duration_secs: None,
                resolution: request.resolution.clone(),
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::Veo
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}",
            self.model.as_str(),
        );

        let response = self
            .client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;

        match response.status().as_u16() {
            401 | 403 => Err(GenVizError::Auth("Invalid API key".into())),
            _ => Ok(()),
        }
    }
}

// Request/Response types
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoRequest {
    instances: Vec<VeoInstance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<VeoParameters>,
}

#[derive(Debug, Serialize)]
struct VeoInstance {
    prompt: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_seconds: Option<u32>,
}

impl VeoRequest {
    fn from_request(req: &VideoGenerationRequest) -> Self {
        let has_params =
            req.resolution.is_some() || req.aspect_ratio.is_some() || req.duration_secs.is_some();

        let parameters = if has_params {
            Some(VeoParameters {
                aspect_ratio: req.aspect_ratio.clone(),
                resolution: req.resolution.clone(),
                duration_seconds: req.duration_secs,
            })
        } else {
            None
        };

        Self {
            instances: vec![VeoInstance {
                prompt: req.prompt.clone(),
            }],
            parameters,
        }
    }
}

#[derive(Debug, Deserialize)]
struct VeoOperationResponse {
    name: String,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    response: Option<VeoVideoResponse>,
    #[serde(default)]
    error: Option<VeoError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeoVideoResponse {
    /// Response from predictLongRunning endpoint.
    #[serde(default)]
    generate_video_response: Option<VeoGenerateVideoResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeoGenerateVideoResponse {
    #[serde(default)]
    generated_samples: Option<Vec<VeoGeneratedSample>>,
    #[serde(default)]
    rai_media_filtered_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct VeoGeneratedSample {
    #[serde(default)]
    video: Option<VeoVideo>,
}

#[derive(Debug, Deserialize)]
struct VeoVideo {
    #[serde(default)]
    uri: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VeoError {
    #[serde(default)]
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_veo_model_as_str() {
        assert_eq!(VeoModel::Veo31Preview.as_str(), "veo-3.1-generate-preview");
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = VeoProviderBuilder::new().api_key("test-key").build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = VeoProviderBuilder::new()
            .api_key("test-key")
            .poll_interval(Duration::from_secs(30))
            .timeout(Duration::from_secs(900))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(30));
        assert_eq!(provider.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_request_construction_basic() {
        let req = VideoGenerationRequest::new("Ocean waves");
        let veo_req = VeoRequest::from_request(&req);

        assert_eq!(veo_req.instances.len(), 1);
        assert_eq!(veo_req.instances[0].prompt, "Ocean waves");
        assert!(veo_req.parameters.is_none());
    }

    #[test]
    fn test_request_construction_with_resolution() {
        let req = VideoGenerationRequest::new("Ocean waves").with_resolution("720p");
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(params.resolution.as_deref(), Some("720p"));
    }

    #[test]
    fn test_request_construction_with_all_params() {
        let req = VideoGenerationRequest::new("Ocean waves")
            .with_resolution("1080p")
            .with_aspect_ratio("16:9")
            .with_duration(8);
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(params.resolution.as_deref(), Some("1080p"));
        assert_eq!(params.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(params.duration_seconds, Some(8));
    }

    #[test]
    fn test_request_serialization_uses_camel_case() {
        let req = VideoGenerationRequest::new("test")
            .with_resolution("720p")
            .with_duration(6);
        let veo_req = VeoRequest::from_request(&req);
        let json = serde_json::to_value(&veo_req).unwrap();

        // Should use camelCase
        assert!(json.get("instances").is_some());
        let params = json.get("parameters").unwrap();
        assert!(params.get("aspectRatio").is_some() || params.get("resolution").is_some());
        assert!(params.get("durationSeconds").is_some());
    }

    #[test]
    fn test_operation_response_not_done() {
        let json = r#"{"name": "operations/123", "done": false}"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.name, "operations/123");
        assert_eq!(resp.done, Some(false));
        assert!(resp.response.is_none());
    }

    #[test]
    fn test_operation_response_done_with_video() {
        let json = r#"{
            "name": "operations/123",
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [{
                        "video": {"uri": "https://example.com/video.mp4"}
                    }]
                }
            }
        }"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.done, Some(true));

        let samples = resp
            .response
            .unwrap()
            .generate_video_response
            .unwrap()
            .generated_samples
            .unwrap();
        let uri = samples[0].video.as_ref().unwrap().uri.as_deref();
        assert_eq!(uri, Some("https://example.com/video.mp4"));
    }

    #[test]
    fn test_operation_response_with_error() {
        let json = r#"{
            "name": "operations/123",
            "done": false,
            "error": {"message": "Quota exceeded"}
        }"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.error.unwrap().message.as_deref(),
            Some("Quota exceeded")
        );
    }

    #[test]
    fn test_gs_url_returns_error() {
        let provider = VeoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let result = rt.block_on(provider.download("gs://my-bucket/video.mp4"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Google Cloud Storage"),
            "Expected GCS error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_error_404_gives_helpful_message() {
        let provider = VeoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();

        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(404, "Not Found", &headers);
        match err {
            GenVizError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("billing"),
                    "Expected billing hint, got: {}",
                    msg
                );
            }
            _ => panic!("Expected InvalidRequest error, got: {:?}", err),
        }
    }
}
