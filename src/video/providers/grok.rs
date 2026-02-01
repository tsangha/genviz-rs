//! Grok Imagine Video (xAI) video generation provider.

use crate::error::{sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const SUBMIT_URL: &str = "https://api.x.ai/v1/videos/generations";

/// Grok video model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GrokVideoModel {
    #[default]
    GrokImagineVideo,
}

impl GrokVideoModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GrokImagineVideo => "grok-imagine-video",
        }
    }
}

/// Builder for GrokVideoProvider.
#[derive(Debug, Clone)]
pub struct GrokVideoProviderBuilder {
    api_key: Option<String>,
    model: GrokVideoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for GrokVideoProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: GrokVideoModel::default(),
            poll_interval: Duration::from_secs(2),
            timeout: Duration::from_secs(300), // 5 minutes for video
        }
    }
}

impl GrokVideoProviderBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: GrokVideoModel) -> Self {
        self.model = model;
        self
    }

    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn build(self) -> Result<GrokVideoProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("XAI_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("XAI_API_KEY not set and no API key provided".into())
            })?;

        Ok(GrokVideoProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// Grok Imagine Video generation provider.
pub struct GrokVideoProvider {
    client: reqwest::Client,
    api_key: String,
    model: GrokVideoModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl GrokVideoProvider {
    pub fn builder() -> GrokVideoProviderBuilder {
        GrokVideoProviderBuilder::new()
    }

    /// Submit a video generation request.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        let body = GrokVideoRequest::from_request(request, &self.model);

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
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let submit_response: GrokVideoSubmitResponse = response.json().await?;
        Ok(submit_response.request_id)
    }

    /// Poll until the video is ready.
    async fn poll_until_ready(&self, request_id: &str) -> Result<String> {
        let url = format!("https://api.x.ai/v1/videos/{}", request_id);
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

            // 202 means still processing
            if status.as_u16() == 202 {
                tokio::time::sleep(self.poll_interval).await;
                continue;
            }

            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text));
            }

            let result: GrokVideoResultResponse = response.json().await?;
            return Ok(result.video.url);
        }
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

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 429 {
            return GenVizError::RateLimited { retry_after: None };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        if text.contains("safety") || text.contains("blocked") || text.contains("content_policy") {
            return GenVizError::ContentBlocked(text);
        }
        GenVizError::Api {
            status,
            message: text,
        }
    }
}

#[async_trait]
impl VideoProvider for GrokVideoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();

        // Submit the request
        let request_id = self.submit(request).await?;
        tracing::debug!(request_id = %request_id, "submitted video generation request");

        // Poll until ready
        let video_url = self.poll_until_ready(&request_id).await?;
        tracing::debug!(url = %video_url, "video generation complete");

        // Download the video
        let data = self.download(&video_url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::Grok,
            VideoMetadata {
                model: Some(self.model.as_str().to_string()),
                duration_ms: Some(duration_ms),
                video_duration_secs: request.duration_secs,
                resolution: None,
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::Grok
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.starts_with("xai-") {
            Ok(())
        } else {
            Err(GenVizError::Auth("Invalid API key format".into()))
        }
    }
}

// Request/Response types
#[derive(Debug, Serialize)]
struct GrokVideoRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<GrokVideoImage>,
}

#[derive(Debug, Serialize)]
struct GrokVideoImage {
    url: String,
}

impl GrokVideoRequest {
    fn from_request(req: &VideoGenerationRequest, model: &GrokVideoModel) -> Self {
        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            duration: req.duration_secs,
            image: req
                .source_image_url
                .as_ref()
                .map(|url| GrokVideoImage { url: url.clone() }),
        }
    }
}

#[derive(Debug, Deserialize)]
struct GrokVideoSubmitResponse {
    request_id: String,
}

#[derive(Debug, Deserialize)]
struct GrokVideoResultResponse {
    video: GrokVideoResult,
}

#[derive(Debug, Deserialize)]
struct GrokVideoResult {
    url: String,
}
