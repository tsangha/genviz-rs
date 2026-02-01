//! Gemini (Google) image generation provider.

use crate::error::{GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProviderKind,
};
use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Gemini image model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GeminiModel {
    /// Nano Banana - Gemini 2.5 Flash Image (fast, economical).
    NanoBanana,
    /// Nano Banana Pro - Gemini 3 Pro Image (highest quality).
    #[default]
    NanoBananaPro,
}

impl GeminiModel {
    /// Returns the API model identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NanoBanana => "gemini-2.5-flash-image",
            Self::NanoBananaPro => "nano-banana-pro-preview",
        }
    }
}

/// Builder for GeminiProvider.
#[derive(Debug, Clone, Default)]
pub struct GeminiProviderBuilder {
    api_key: Option<String>,
    model: GeminiModel,
}

impl GeminiProviderBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: GeminiModel) -> Self {
        self.model = model;
        self
    }

    pub fn build(self) -> Result<GeminiProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("GOOGLE_API_KEY not set and no API key provided".into())
            })?;

        Ok(GeminiProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
        })
    }
}

/// Gemini image generation provider.
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: GeminiModel,
}

impl GeminiProvider {
    pub fn builder() -> GeminiProviderBuilder {
        GeminiProviderBuilder::new()
    }

    async fn generate_impl(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model.as_str(),
            self.api_key
        );

        let body = GeminiRequest::from_generation_request(request);

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

        let gemini_response: GeminiResponse = response.json().await?;

        let candidate = gemini_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| GenVizError::Api {
                status: 500,
                message: "No candidates in response".into(),
            })?;

        let inline_data = candidate
            .content
            .parts
            .into_iter()
            .find_map(|p| p.inline_data)
            .ok_or_else(|| GenVizError::Api {
                status: 500,
                message: "No image in response".into(),
            })?;

        let data = base64::engine::general_purpose::STANDARD
            .decode(&inline_data.data)
            .map_err(|e| GenVizError::Decode(e.to_string()))?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let format = match inline_data.mime_type.as_str() {
            "image/png" => ImageFormat::Png,
            "image/jpeg" => ImageFormat::Jpeg,
            "image/webp" => ImageFormat::WebP,
            _ => ImageFormat::Png,
        };

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Gemini,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: None,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
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
impl ImageProvider for GeminiProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        self.generate_impl(request).await
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Gemini
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
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: GeminiConfig,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiConfig {
    response_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

impl GeminiRequest {
    fn from_generation_request(req: &GenerationRequest) -> Self {
        Self {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart {
                    text: req.prompt.clone(),
                }],
            }],
            generation_config: GeminiConfig {
                response_modalities: vec!["IMAGE".to_string()],
                seed: req.seed,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
}

#[derive(Debug, Deserialize)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPartResponse {
    #[serde(default)]
    inline_data: Option<InlineData>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    mime_type: String,
    data: String,
}
