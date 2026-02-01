//! Flux (Black Forest Labs) image generation provider.

use crate::error::{GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Flux model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FluxModel {
    #[default]
    FluxPro11,
    FluxPro,
    FluxDev,
}

impl FluxModel {
    fn endpoint(&self) -> &'static str {
        match self {
            Self::FluxPro11 => "flux-pro-1.1",
            Self::FluxPro => "flux-pro",
            Self::FluxDev => "flux-dev",
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.endpoint()
    }
}

/// Builder for FluxProvider.
#[derive(Debug, Clone)]
pub struct FluxProviderBuilder {
    api_key: Option<String>,
    model: FluxModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for FluxProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: FluxModel::default(),
            poll_interval: Duration::from_millis(500),
            timeout: Duration::from_secs(120),
        }
    }
}

impl FluxProviderBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: FluxModel) -> Self {
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

    pub fn build(self) -> Result<FluxProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("BFL_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("BFL_API_KEY not set and no API key provided".into())
            })?;

        Ok(FluxProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// Flux image generation provider.
pub struct FluxProvider {
    client: reqwest::Client,
    api_key: String,
    model: FluxModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl FluxProvider {
    pub fn builder() -> FluxProviderBuilder {
        FluxProviderBuilder::new()
    }

    async fn submit(&self, request: &GenerationRequest) -> Result<String> {
        // Use flux-kontext-pro for image editing, otherwise use the configured model
        let url = if request.input_image.is_some() {
            "https://api.bfl.ml/v1/flux-kontext-pro".to_string()
        } else {
            format!("https://api.bfl.ml/v1/{}", self.model.endpoint())
        };

        let body = FluxRequest::from_generation_request(request);

        let response = self
            .client
            .post(&url)
            .header("x-key", &self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(GenVizError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let submit_response: FluxSubmitResponse = response.json().await?;
        Ok(submit_response.id)
    }

    async fn poll_until_ready(&self, task_id: &str) -> Result<FluxResult> {
        let url = format!("https://api.bfl.ml/v1/get_result?id={}", task_id);
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(&url)
                .header("x-key", &self.api_key)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(GenVizError::Api {
                    status: status.as_u16(),
                    message: text,
                });
            }

            let result: FluxResultResponse = response.json().await?;

            match result.status.as_str() {
                "Ready" => {
                    return result.result.ok_or_else(|| GenVizError::Api {
                        status: 500,
                        message: "Ready status but no result".into(),
                    });
                }
                "Pending" | "Processing" => {
                    tokio::time::sleep(self.poll_interval).await;
                }
                "Error" => {
                    return Err(GenVizError::Api {
                        status: 500,
                        message: result
                            .result
                            .and_then(|r| r.error)
                            .unwrap_or_else(|| "Unknown error".into()),
                    });
                }
                "Content Moderated" => {
                    return Err(GenVizError::ContentBlocked(
                        "Image blocked by content moderation".into(),
                    ));
                }
                other => {
                    return Err(GenVizError::Api {
                        status: 500,
                        message: format!("Unknown status: {}", other),
                    });
                }
            }
        }
    }

    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            if response.status().as_u16() == 403 || response.status().as_u16() == 410 {
                return Err(GenVizError::UrlExpired);
            }
            return Err(GenVizError::Api {
                status: response.status().as_u16(),
                message: "Failed to download image".into(),
            });
        }

        Ok(response.bytes().await?.to_vec())
    }
}

#[async_trait]
impl ImageProvider for FluxProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        let task_id = self.submit(request).await?;
        tracing::debug!(task_id = %task_id, "submitted generation request");

        let result = self.poll_until_ready(&task_id).await?;
        tracing::debug!(url = %result.sample, "generation complete");

        let data = self.download(&result.sample).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let format = result
            .sample
            .split('.')
            .next_back()
            .and_then(ImageFormat::from_extension)
            .unwrap_or(ImageFormat::Jpeg);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Flux,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: result.seed,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Flux
    }

    async fn health_check(&self) -> Result<()> {
        let url = "https://api.bfl.ml/v1/get_result?id=test";
        let response = self
            .client
            .get(url)
            .header("x-key", &self.api_key)
            .send()
            .await?;

        match response.status().as_u16() {
            401 | 403 => Err(GenVizError::Auth("Invalid API key".into())),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Serialize)]
struct FluxRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    /// Input image for editing (base64 encoded).
    #[serde(skip_serializing_if = "Option::is_none")]
    input_image: Option<String>,
}

impl FluxRequest {
    fn from_generation_request(req: &GenerationRequest) -> Self {
        use base64::Engine;

        Self {
            prompt: req.prompt.clone(),
            width: req.width,
            height: req.height,
            seed: req.seed,
            aspect_ratio: req.aspect_ratio.map(|ar| ar.as_str().to_string()),
            input_image: req
                .input_image
                .as_ref()
                .map(|img| base64::engine::general_purpose::STANDARD.encode(img)),
        }
    }
}

#[derive(Debug, Deserialize)]
struct FluxSubmitResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct FluxResultResponse {
    status: String,
    result: Option<FluxResult>,
}

#[derive(Debug, Deserialize)]
struct FluxResult {
    sample: String,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}
