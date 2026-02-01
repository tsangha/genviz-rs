//! Veo (Google) video generation provider.

use crate::error::{GenVizError, Result};
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
    #[default]
    Veo31Preview,
}

impl VeoModel {
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: VeoModel) -> Self {
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
    pub fn builder() -> VeoProviderBuilder {
        VeoProviderBuilder::new()
    }

    /// Submit a video generation request.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateVideos?key={}",
            self.model.as_str(),
            self.api_key
        );

        let body = VeoRequest::from_request(request);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let operation: VeoOperationResponse = response.json().await?;
        Ok(operation.name)
    }

    /// Poll until the video is ready.
    async fn poll_until_ready(&self, operation_name: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}?key={}",
            operation_name, self.api_key
        );
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self.client.get(&url).send().await?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text));
            }

            let operation: VeoOperationResponse = response.json().await?;

            if operation.done.unwrap_or(false) {
                if let Some(resp) = operation.response {
                    if let Some(videos) = resp.generated_videos {
                        if let Some(first) = videos.into_iter().next() {
                            if let Some(uri) = first.video.and_then(|v| v.uri) {
                                return Ok(uri);
                            }
                        }
                    }
                }
                return Err(GenVizError::VideoGeneration(
                    "Video generation completed but no video URL returned".into(),
                ));
            }

            if let Some(err) = operation.error {
                return Err(GenVizError::VideoGeneration(
                    err.message.unwrap_or_else(|| "Unknown error".into()),
                ));
            }

            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Download the video from the given URL.
    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        // Veo returns gs:// URLs which need to be converted or accessed via the files API
        // For now, we'll try a direct download
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
        if status == 404 {
            return GenVizError::Api {
                status,
                message: "Veo API not available. Veo requires a paid-tier API key with billing enabled. \
                         Enable it at https://aistudio.google.com by selecting a Google Cloud project with billing."
                    .to_string(),
            };
        }
        if status == 429 {
            return GenVizError::RateLimited { retry_after: None };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text.to_string());
        }
        if text.contains("SAFETY") || text.contains("blocked") {
            return GenVizError::ContentBlocked(text.to_string());
        }
        GenVizError::Api {
            status,
            message: text.to_string(),
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
            "https://generativelanguage.googleapis.com/v1beta/models/{}?key={}",
            self.model.as_str(),
            self.api_key
        );

        let response = self.client.get(&url).send().await?;

        match response.status().as_u16() {
            401 | 403 => Err(GenVizError::Auth("Invalid API key".into())),
            _ => Ok(()),
        }
    }
}

// Request/Response types
#[derive(Debug, Serialize)]
struct VeoRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<VeoConfig>,
}

#[derive(Debug, Serialize)]
struct VeoConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    number_of_videos: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
}

impl VeoRequest {
    fn from_request(req: &VideoGenerationRequest) -> Self {
        let config = if req.resolution.is_some() {
            Some(VeoConfig {
                number_of_videos: Some(1),
                resolution: req.resolution.clone(),
            })
        } else {
            None
        };

        Self {
            model: "veo-3.1-generate-preview".to_string(),
            prompt: req.prompt.clone(),
            config,
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
struct VeoVideoResponse {
    #[serde(default)]
    generated_videos: Option<Vec<VeoGeneratedVideo>>,
}

#[derive(Debug, Deserialize)]
struct VeoGeneratedVideo {
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
