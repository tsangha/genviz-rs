//! Grok Imagine (xAI) image generation provider.

use crate::error::{GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProviderKind,
};
use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::time::Instant;

const GENERATIONS_URL: &str = "https://api.x.ai/v1/images/generations";
const EDITS_URL: &str = "https://api.x.ai/v1/images/edits";

/// Grok Imagine model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GrokModel {
    #[default]
    GrokImagine,
}

impl GrokModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GrokImagine => "grok-imagine-image",
        }
    }
}

/// Builder for GrokProvider.
#[derive(Debug, Clone, Default)]
pub struct GrokProviderBuilder {
    api_key: Option<String>,
    model: GrokModel,
}

impl GrokProviderBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: GrokModel) -> Self {
        self.model = model;
        self
    }

    pub fn build(self) -> Result<GrokProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("XAI_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("XAI_API_KEY not set and no API key provided".into())
            })?;

        Ok(GrokProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
        })
    }
}

/// Grok Imagine image generation provider.
pub struct GrokProvider {
    client: reqwest::Client,
    api_key: String,
    model: GrokModel,
}

impl GrokProvider {
    pub fn builder() -> GrokProviderBuilder {
        GrokProviderBuilder::new()
    }

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
        if status == 429 {
            return GenVizError::RateLimited { retry_after: None };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text.to_string());
        }
        if text.contains("safety") || text.contains("blocked") || text.contains("content_policy") {
            return GenVizError::ContentBlocked(text.to_string());
        }
        GenVizError::Api {
            status,
            message: text.to_string(),
        }
    }
}

#[async_trait]
impl ImageProvider for GrokProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        // Use edits endpoint if input image is provided
        let (url, body) = if request.input_image.is_some() {
            (
                EDITS_URL,
                serde_json::to_value(GrokEditRequest::from_generation_request(request, &self.model))?,
            )
        } else {
            (
                GENERATIONS_URL,
                serde_json::to_value(GrokRequest::from_generation_request(request, &self.model))?,
            )
        };

        let response = self
            .client
            .post(url)
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

        let grok_response: GrokResponse = response.json().await?;

        let image_data = grok_response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| GenVizError::Api {
                status: 500,
                message: "No images in response".into(),
            })?;

        let data = base64::engine::general_purpose::STANDARD
            .decode(&image_data.b64_json)
            .map_err(|e| GenVizError::Decode(e.to_string()))?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Grok,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: None,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Grok
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.starts_with("xai-") {
            Ok(())
        } else {
            Err(GenVizError::Auth("Invalid API key format".into()))
        }
    }
}

#[derive(Debug, Serialize)]
struct GrokRequest {
    model: String,
    prompt: String,
    n: u32,
    response_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
}

impl GrokRequest {
    fn from_generation_request(req: &GenerationRequest, model: &GrokModel) -> Self {
        let aspect_ratio = req.aspect_ratio.map(|ar| ar.as_str().to_string());

        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            n: 1,
            response_format: "b64_json".to_string(),
            aspect_ratio,
        }
    }
}

/// Request for image editing endpoint.
#[derive(Debug, Serialize)]
struct GrokEditRequest {
    model: String,
    prompt: String,
    /// Base64-encoded input image.
    image: String,
    n: u32,
    response_format: String,
}

impl GrokEditRequest {
    fn from_generation_request(req: &GenerationRequest, model: &GrokModel) -> Self {
        let image = req
            .input_image
            .as_ref()
            .map(|img| base64::engine::general_purpose::STANDARD.encode(img))
            .unwrap_or_default();

        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            image,
            n: 1,
            response_format: "b64_json".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct GrokResponse {
    data: Vec<GrokImageData>,
}

#[derive(Debug, Deserialize)]
struct GrokImageData {
    b64_json: String,
}
