//! ModelsLab image generation provider.
//!
//! Supports Flux, SDXL, Stable Diffusion, and 10,000+ community models
//! via the ModelsLab REST API.
//!
//! # Example
//!
//! ```no_run
//! use genviz::image::{providers::ModelsLabImageProvider, ImageProvider, GenerationRequest};
//!
//! #[tokio::main]
//! async fn main() {
//!     let provider = ModelsLabImageProvider::builder()
//!         .build()
//!         .expect("MODELSLAB_API_KEY must be set");
//!
//!     let request = GenerationRequest::new("A sunset over mountains in watercolor style");
//!     let image = provider.generate(&request).await.unwrap();
//!     std::fs::write("output.png", image.data).unwrap();
//! }
//! ```

use crate::error::{sanitize_error_message, GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    AspectRatio, GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat,
    ImageProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const DEFAULT_BASE_URL: &str = "https://modelslab.com/api/v6";
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(3);
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// ModelsLab image model variants.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ModelsLabImageModel {
    /// Flux — high-quality, photorealistic (default).
    #[default]
    Flux,
    /// Flux Dev variant.
    FluxDev,
    /// Stable Diffusion XL — fast, versatile.
    Sdxl,
    /// Realistic Vision v6 — photorealistic portraits and scenes.
    RealisticVision,
    /// DreamShaper 8 — artistic and creative styles.
    DreamShaper,
    /// Anything v5 — anime and illustration style.
    Anything,
    /// Any community model by its ModelsLab model ID.
    Custom(String),
}

impl ModelsLabImageModel {
    /// Returns the ModelsLab model ID string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Flux => "flux",
            Self::FluxDev => "flux-dev",
            Self::Sdxl => "sdxl",
            Self::RealisticVision => "realistic-vision-v6",
            Self::DreamShaper => "dreamshaper-8",
            Self::Anything => "anything-v5",
            Self::Custom(id) => id,
        }
    }
}

/// Builder for [`ModelsLabImageProvider`].
#[derive(Debug, Clone)]
pub struct ModelsLabImageProviderBuilder {
    api_key: Option<String>,
    model: ModelsLabImageModel,
    base_url: String,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for ModelsLabImageProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: ModelsLabImageModel::default(),
            base_url: DEFAULT_BASE_URL.into(),
            poll_interval: DEFAULT_POLL_INTERVAL,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

impl ModelsLabImageProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `MODELSLAB_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the model variant.
    pub fn model(mut self, model: ModelsLabImageModel) -> Self {
        self.model = model;
        self
    }

    /// Sets the polling interval for async generation (default: 3s).
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the maximum time to wait for generation (default: 5 min).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Overrides the base URL (useful for testing).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Builds the provider, resolving credentials from env if not explicitly set.
    pub fn build(self) -> Result<ModelsLabImageProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("MODELSLAB_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth(
                    "MODELSLAB_API_KEY not set and no API key provided".into(),
                )
            })?;

        Ok(ModelsLabImageProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            base_url: self.base_url,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// ModelsLab image generation provider.
///
/// Supports text-to-image generation using Flux, SDXL, Stable Diffusion,
/// and thousands of community fine-tuned models via the ModelsLab REST API.
pub struct ModelsLabImageProvider {
    client: reqwest::Client,
    api_key: String,
    model: ModelsLabImageModel,
    base_url: String,
    poll_interval: Duration,
    timeout: Duration,
}

impl ModelsLabImageProvider {
    /// Creates a new [`ModelsLabImageProviderBuilder`].
    pub fn builder() -> ModelsLabImageProviderBuilder {
        ModelsLabImageProviderBuilder::new()
    }

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        if status == 429 {
            return GenVizError::RateLimited { retry_after: None };
        }
        GenVizError::Api { status, message: text }
    }

    /// Posts a text-to-image request and returns the API response.
    async fn post_generate(&self, request: &GenerationRequest) -> Result<ModelsLabResponse> {
        let (width, height) = resolve_size(request);

        let body = ModelsLabRequest {
            key: self.api_key.clone(),
            prompt: request.prompt.clone(),
            negative_prompt: None,
            model_id: self.model.as_str().to_string(),
            width: width.to_string(),
            height: height.to_string(),
            samples: "1".into(),
            num_inference_steps: "30".into(),
            guidance_scale: 7.5,
            safety_checker: "no".into(),
            seed: request.seed,
        };

        let url = format!("{}/images/text2img", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();
        if status == 401 || status == 403 || status == 429 {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status, &text));
        }
        if status != 200 {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status, &text));
        }

        let api_resp: ModelsLabResponse = response.json().await?;
        Ok(api_resp)
    }

    /// Polls the fetch endpoint until the image is ready.
    async fn poll_fetch(&self, request_id: u64, start: Instant) -> Result<Vec<String>> {
        let url = format!("{}/images/fetch/{}", self.base_url, request_id);

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            tokio::time::sleep(self.poll_interval).await;

            let response = self
                .client
                .post(&url)
                .json(&serde_json::json!({ "key": self.api_key }))
                .send()
                .await?;

            let status = response.status().as_u16();
            if !response.status().is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status, &text));
            }

            let fetch_resp: ModelsLabResponse = response.json().await?;

            match fetch_resp.status.as_str() {
                "success" => {
                    return Ok(fetch_resp.output.unwrap_or_default());
                }
                "processing" => {
                    tracing::debug!(
                        request_id,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling ModelsLab image generation"
                    );
                }
                other => {
                    let msg = fetch_resp.message.unwrap_or_else(|| other.to_string());
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "ModelsLab unexpected status '{other}': {msg}"
                    )));
                }
            }
        }
    }

    /// Downloads image bytes from a URL.
    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.client.get(url).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(GenVizError::Api {
                status: status.as_u16(),
                message: "Failed to download generated image".into(),
            });
        }
        Ok(response.bytes().await?.to_vec())
    }
}

#[async_trait]
impl ImageProvider for ModelsLabImageProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        let api_resp = self.post_generate(request).await?;

        let image_urls = match api_resp.status.as_str() {
            "success" => api_resp.output.ok_or_else(|| {
                GenVizError::UnexpectedResponse("ModelsLab returned no output URLs".into())
            })?,
            "processing" => {
                let request_id = api_resp.id.ok_or_else(|| {
                    GenVizError::UnexpectedResponse(
                        "ModelsLab returned 'processing' without a request id".into(),
                    )
                })?;
                tracing::debug!(request_id, "ModelsLab image generation is async, polling");
                self.poll_fetch(request_id, start).await?
            }
            "error" => {
                let msg = api_resp
                    .message
                    .unwrap_or_else(|| "unknown error".into());
                return Err(GenVizError::Api {
                    status: 200,
                    message: sanitize_error_message(&msg),
                });
            }
            other => {
                return Err(GenVizError::UnexpectedResponse(format!(
                    "ModelsLab returned unexpected status: {other}"
                )));
            }
        };

        let image_url = image_urls
            .into_iter()
            .next()
            .ok_or_else(|| GenVizError::UnexpectedResponse("ModelsLab returned empty output".into()))?;

        let data = self.download(&image_url).await?;
        let duration_ms = start.elapsed().as_millis() as u64;
        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::ModelsLab,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: request.seed,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::ModelsLab
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(GenVizError::Auth("MODELSLAB_API_KEY is empty".into()));
        }
        Ok(())
    }
}

// ---- Request / Response types ----

#[derive(Debug, Serialize)]
struct ModelsLabRequest {
    key: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    model_id: String,
    width: String,
    height: String,
    samples: String,
    num_inference_steps: String,
    guidance_scale: f32,
    safety_checker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ModelsLabResponse {
    status: String,
    #[serde(default)]
    output: Option<Vec<String>>,
    /// Request id returned when status == "processing".
    #[serde(default)]
    id: Option<u64>,
    /// Error message (status == "error").
    #[serde(default)]
    message: Option<String>,
    /// Typo variant present in some ModelsLab responses.
    #[serde(default, rename = "messege")]
    message_typo: Option<String>,
}

impl ModelsLabResponse {
    #[allow(dead_code)]
    fn error_message(&self) -> Option<&str> {
        self.message
            .as_deref()
            .or(self.message_typo.as_deref())
    }
}

/// Resolves the pixel size from the request's explicit dimensions or aspect ratio.
fn resolve_size(req: &GenerationRequest) -> (u32, u32) {
    if let (Some(w), Some(h)) = (req.width, req.height) {
        return (w, h);
    }
    match req.aspect_ratio.unwrap_or(AspectRatio::Square) {
        AspectRatio::Square => (1024, 1024),
        AspectRatio::Landscape | AspectRatio::Ultrawide => (1216, 832),
        AspectRatio::Portrait => (832, 1216),
        AspectRatio::Standard | AspectRatio::ThreeTwo => (1024, 768),
        AspectRatio::StandardPortrait | AspectRatio::TwoThree => (768, 1024),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_ids() {
        assert_eq!(ModelsLabImageModel::Flux.as_str(), "flux");
        assert_eq!(ModelsLabImageModel::FluxDev.as_str(), "flux-dev");
        assert_eq!(ModelsLabImageModel::Sdxl.as_str(), "sdxl");
        assert_eq!(
            ModelsLabImageModel::RealisticVision.as_str(),
            "realistic-vision-v6"
        );
        assert_eq!(ModelsLabImageModel::DreamShaper.as_str(), "dreamshaper-8");
        assert_eq!(ModelsLabImageModel::Anything.as_str(), "anything-v5");
        assert_eq!(
            ModelsLabImageModel::Custom("my-model".into()).as_str(),
            "my-model"
        );
    }

    #[test]
    fn test_default_model() {
        assert_eq!(ModelsLabImageModel::default(), ModelsLabImageModel::Flux);
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = ModelsLabImageProviderBuilder::new()
            .api_key("test-key")
            .model(ModelsLabImageModel::Sdxl)
            .build();
        assert!(provider.is_ok());
        let p = provider.unwrap();
        assert_eq!(p.model, ModelsLabImageModel::Sdxl);
    }

    #[test]
    fn test_builder_missing_key() {
        std::env::remove_var("MODELSLAB_API_KEY");
        let result = ModelsLabImageProviderBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_size_explicit() {
        let req = GenerationRequest::new("test").with_size(512, 768);
        assert_eq!(resolve_size(&req), (512, 768));
    }

    #[test]
    fn test_resolve_size_square() {
        let req = GenerationRequest::new("test").with_aspect_ratio(AspectRatio::Square);
        assert_eq!(resolve_size(&req), (1024, 1024));
    }

    #[test]
    fn test_resolve_size_landscape() {
        let req = GenerationRequest::new("test").with_aspect_ratio(AspectRatio::Landscape);
        assert_eq!(resolve_size(&req), (1216, 832));
    }

    #[test]
    fn test_response_deserialization_success() {
        let json = r#"{
            "status": "success",
            "output": ["https://cdn.modelslab.com/images/abc123.png"]
        }"#;
        let resp: ModelsLabResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "success");
        assert_eq!(resp.output.unwrap()[0], "https://cdn.modelslab.com/images/abc123.png");
    }

    #[test]
    fn test_response_deserialization_processing() {
        let json = r#"{"status": "processing", "id": 42}"#;
        let resp: ModelsLabResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "processing");
        assert_eq!(resp.id, Some(42));
    }

    #[test]
    fn test_response_deserialization_error_typo() {
        // ModelsLab sometimes returns "messege" (typo)
        let json = r#"{"status": "error", "messege": "invalid key"}"#;
        let resp: ModelsLabResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "error");
        assert_eq!(resp.message_typo.as_deref(), Some("invalid key"));
    }

    #[test]
    fn test_provider_kind() {
        let provider = ModelsLabImageProviderBuilder::new()
            .api_key("test")
            .build()
            .unwrap();
        assert_eq!(provider.kind(), ImageProviderKind::ModelsLab);
        assert_eq!(provider.name(), "ModelsLab");
    }

    #[test]
    fn test_provider_kind_display() {
        assert_eq!(ImageProviderKind::ModelsLab.to_string(), "modelslab");
    }
}
