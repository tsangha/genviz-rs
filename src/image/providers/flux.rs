//! Flux (Black Forest Labs) image generation provider.

use crate::error::{sanitize_error_message, GenVizError, Result};
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
    /// FLUX.1 Pro 1.1 - high quality text-to-image generation.
    #[default]
    FluxPro11,
    /// FLUX.1 Pro 1.1 Ultra - highest quality with 4MP output.
    FluxPro11Ultra,
    /// FLUX.1 Pro - professional quality generation.
    FluxPro,
    /// FLUX.1 Dev - fast development/testing model.
    FluxDev,

    /// FLUX.2 Max - latest generation with editing support.
    Flux2Max,
    /// FLUX.2 Pro - professional quality with editing.
    Flux2Pro,
    /// FLUX.2 Flex - flexible generation and editing.
    Flux2Flex,
    /// FLUX.2 Klein 4B - lightweight 4 billion parameter model.
    Flux2Klein4B,
    /// FLUX.2 Klein 9B - mid-size 9 billion parameter model.
    Flux2Klein9B,

    /// Kontext Pro - context-aware image editing.
    KontextPro,
    /// Kontext Max - highest quality context-aware editing.
    KontextMax,

    /// Fill Pro - inpainting with mask-based editing. Requires input image.
    FillPro,

    /// Expand Pro - outpainting to extend images. Requires input image.
    ExpandPro,
}

impl FluxModel {
    /// Returns the API endpoint path for this model.
    fn endpoint(&self) -> &'static str {
        match self {
            Self::FluxPro11 => "flux-pro-1.1",
            Self::FluxPro11Ultra => "flux-pro-1.1-ultra",
            Self::FluxPro => "flux-pro",
            Self::FluxDev => "flux-dev",
            Self::Flux2Max => "flux-2-max",
            Self::Flux2Pro => "flux-2-pro",
            Self::Flux2Flex => "flux-2-flex",
            Self::Flux2Klein4B => "flux-2-klein-4b",
            Self::Flux2Klein9B => "flux-2-klein-9b",
            Self::KontextPro => "flux-kontext-pro",
            Self::KontextMax => "flux-kontext-max",
            Self::FillPro => "flux-pro-1.0-fill",
            Self::ExpandPro => "flux-pro-1.0-expand",
        }
    }

    /// Returns the API model identifier string.
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
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `BFL_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the Flux model variant.
    pub fn model(mut self, model: FluxModel) -> Self {
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
    /// Creates a new `FluxProviderBuilder`.
    pub fn builder() -> FluxProviderBuilder {
        FluxProviderBuilder::new()
    }

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 429 {
            let retry_after =
                crate::error::parse_retry_after(headers).map(std::time::Duration::from_secs);
            return GenVizError::RateLimited { retry_after };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        if status == 402 {
            return GenVizError::Billing("Insufficient BFL credits. Add credits at bfl.ai".into());
        }
        if status == 422 {
            return GenVizError::InvalidRequest(text);
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

    /// Returns (task_id, polling_url).
    async fn submit(&self, request: &GenerationRequest) -> Result<(String, String)> {
        // Validate Fill/Expand models require input images
        match self.model {
            FluxModel::FillPro => {
                if request.input_image.is_none() {
                    return Err(GenVizError::InvalidRequest(
                        "Fill model requires an input image and mask. Use with_input_image() to provide the source image.".into(),
                    ));
                }
            }
            FluxModel::ExpandPro => {
                if request.input_image.is_none() {
                    return Err(GenVizError::InvalidRequest(
                        "Expand model requires an input image. Use with_input_image() to provide the source image.".into(),
                    ));
                }
            }
            _ => {}
        }

        let url = if request.input_image.is_some() {
            match self.model {
                // FLUX.2 models handle editing natively
                FluxModel::Flux2Max
                | FluxModel::Flux2Pro
                | FluxModel::Flux2Flex
                | FluxModel::Flux2Klein4B
                | FluxModel::Flux2Klein9B => {
                    format!("https://api.bfl.ai/v1/{}", self.model.endpoint())
                }
                // Kontext models are already edit models
                FluxModel::KontextPro | FluxModel::KontextMax => {
                    format!("https://api.bfl.ai/v1/{}", self.model.endpoint())
                }
                // Fill and Expand also take images
                FluxModel::FillPro | FluxModel::ExpandPro => {
                    format!("https://api.bfl.ai/v1/{}", self.model.endpoint())
                }
                // FLUX.1 text-to-image models redirect to Kontext Pro for editing
                _ => "https://api.bfl.ai/v1/flux-kontext-pro".to_string(),
            }
        } else {
            format!("https://api.bfl.ai/v1/{}", self.model.endpoint())
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
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let submit_response: FluxSubmitResponse = response.json().await?;

        // Use the server-provided polling URL (includes correct regional routing),
        // falling back to the global endpoint if not provided.
        let polling_url = submit_response.polling_url.unwrap_or_else(|| {
            format!("https://api.bfl.ai/v1/get_result?id={}", submit_response.id)
        });

        Ok((submit_response.id, polling_url))
    }

    async fn poll_until_ready(&self, polling_url: &str) -> Result<FluxResult> {
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(polling_url)
                .header("x-key", &self.api_key)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text, &headers));
            }

            let result: FluxResultResponse = response.json().await?;

            match result.status.as_str() {
                "Ready" => {
                    return result.result.ok_or_else(|| {
                        GenVizError::UnexpectedResponse(
                            "Flux returned Ready status but no result data".into(),
                        )
                    });
                }
                "Pending" | "Processing" => {
                    tracing::debug!(
                        status = %result.status,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling Flux generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                }
                "Error" => {
                    let message = result
                        .result
                        .and_then(|r| r.error)
                        .unwrap_or_else(|| "Unknown error".into());
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "Flux generation error: {}",
                        message
                    )));
                }
                "Content Moderated" => {
                    return Err(GenVizError::ContentBlocked(
                        "Image blocked by content moderation".into(),
                    ));
                }
                "Request Moderated" => {
                    return Err(GenVizError::ContentBlocked(
                        "Request blocked by content moderation (input flagged). Try modifying your prompt or image.".into(),
                    ));
                }
                "Task not found" => {
                    return Err(GenVizError::UnexpectedResponse(
                        "Task not found or expired. The task may have timed out.".into(),
                    ));
                }
                "Failed" => {
                    let message = result
                        .result
                        .and_then(|r| r.error)
                        .unwrap_or_else(|| "Generation failed".into());
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "Flux generation failed: {}",
                        message
                    )));
                }
                other => {
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "Flux returned unexpected status: {}",
                        other
                    )));
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

        let (task_id, polling_url) = self.submit(request).await?;
        tracing::debug!(task_id = %task_id, polling_url = %polling_url, "submitted generation request");

        let result = self.poll_until_ready(&polling_url).await?;
        tracing::debug!(url = %result.sample, "generation complete");

        let data = self.download(&result.sample).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Jpeg);

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
        let url = "https://api.bfl.ai/v1/get_result?id=test";
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
    /// Server-provided polling URL (includes regional routing).
    #[serde(default)]
    polling_url: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_model_endpoint() {
        assert_eq!(FluxModel::FluxPro11.endpoint(), "flux-pro-1.1");
        assert_eq!(FluxModel::FluxPro11Ultra.endpoint(), "flux-pro-1.1-ultra");
        assert_eq!(FluxModel::FluxPro.endpoint(), "flux-pro");
        assert_eq!(FluxModel::FluxDev.endpoint(), "flux-dev");
        assert_eq!(FluxModel::Flux2Max.endpoint(), "flux-2-max");
        assert_eq!(FluxModel::Flux2Pro.endpoint(), "flux-2-pro");
        assert_eq!(FluxModel::Flux2Flex.endpoint(), "flux-2-flex");
        assert_eq!(FluxModel::Flux2Klein4B.endpoint(), "flux-2-klein-4b");
        assert_eq!(FluxModel::Flux2Klein9B.endpoint(), "flux-2-klein-9b");
        assert_eq!(FluxModel::KontextPro.endpoint(), "flux-kontext-pro");
        assert_eq!(FluxModel::KontextMax.endpoint(), "flux-kontext-max");
        assert_eq!(FluxModel::FillPro.endpoint(), "flux-pro-1.0-fill");
        assert_eq!(FluxModel::ExpandPro.endpoint(), "flux-pro-1.0-expand");
    }

    #[test]
    fn test_flux_model_as_str_matches_endpoint() {
        let models = [
            FluxModel::FluxPro11,
            FluxModel::FluxPro11Ultra,
            FluxModel::FluxPro,
            FluxModel::FluxDev,
            FluxModel::Flux2Max,
            FluxModel::Flux2Pro,
            FluxModel::Flux2Flex,
            FluxModel::Flux2Klein4B,
            FluxModel::Flux2Klein9B,
            FluxModel::KontextPro,
            FluxModel::KontextMax,
            FluxModel::FillPro,
            FluxModel::ExpandPro,
        ];
        for model in models {
            assert_eq!(model.as_str(), model.endpoint());
        }
    }

    #[test]
    fn test_flux_model_default() {
        assert_eq!(FluxModel::default(), FluxModel::FluxPro11);
    }

    #[test]
    fn test_parse_error_rate_limited() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(429, "Too many requests", &headers);
        assert!(matches!(err, GenVizError::RateLimited { .. }));
    }

    #[test]
    fn test_parse_error_auth() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(401, "Unauthorized", &headers);
        assert!(matches!(err, GenVizError::Auth(_)));

        let err = provider.parse_error(403, "Forbidden", &headers);
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_parse_error_billing() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(402, "Payment required", &headers);
        assert!(matches!(err, GenVizError::Billing(_)));
    }

    #[test]
    fn test_parse_error_invalid_request() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(422, "Invalid parameters", &headers);
        assert!(matches!(err, GenVizError::InvalidRequest(_)));
    }

    #[test]
    fn test_parse_error_content_blocked() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(400, "content_policy violation", &headers);
        assert!(matches!(err, GenVizError::ContentBlocked(_)));

        let err = provider.parse_error(400, "Request was blocked by safety filter", &headers);
        assert!(matches!(err, GenVizError::ContentBlocked(_)));
    }

    #[test]
    fn test_parse_error_generic_api() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(500, "Internal server error", &headers);
        assert!(matches!(err, GenVizError::Api { status: 500, .. }));
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = FluxProviderBuilder::new()
            .api_key("test-key")
            .model(FluxModel::FluxDev)
            .build();
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.model, FluxModel::FluxDev);
    }

    #[test]
    fn test_request_construction_basic() {
        let req = GenerationRequest::new("A sunset");
        let flux_req = FluxRequest::from_generation_request(&req);

        assert_eq!(flux_req.prompt, "A sunset");
        assert!(flux_req.width.is_none());
        assert!(flux_req.height.is_none());
        assert!(flux_req.seed.is_none());
        assert!(flux_req.aspect_ratio.is_none());
        assert!(flux_req.input_image.is_none());
    }

    #[test]
    fn test_request_construction_with_options() {
        let req = GenerationRequest::new("A sunset")
            .with_size(1024, 768)
            .with_seed(42)
            .with_aspect_ratio(crate::image::types::AspectRatio::Landscape);

        let flux_req = FluxRequest::from_generation_request(&req);

        assert_eq!(flux_req.width, Some(1024));
        assert_eq!(flux_req.height, Some(768));
        assert_eq!(flux_req.seed, Some(42));
        assert_eq!(flux_req.aspect_ratio.as_deref(), Some("16:9"));
    }

    #[test]
    fn test_request_with_input_image() {
        let req = GenerationRequest::new("Edit this").with_input_image(vec![0x89, 0x50, 0x4E]);
        let flux_req = FluxRequest::from_generation_request(&req);

        assert!(flux_req.input_image.is_some());
    }

    #[test]
    fn test_submit_response_deserialization() {
        let json = r#"{"id": "abc-123"}"#;
        let resp: FluxSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "abc-123");
    }

    #[test]
    fn test_result_response_ready() {
        let json = r#"{
            "status": "Ready",
            "result": {"sample": "https://example.com/image.jpg", "seed": 42}
        }"#;
        let resp: FluxResultResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Ready");
        let result = resp.result.unwrap();
        assert_eq!(result.sample, "https://example.com/image.jpg");
        assert_eq!(result.seed, Some(42));
    }

    #[test]
    fn test_result_response_pending() {
        let json = r#"{"status": "Pending", "result": null}"#;
        let resp: FluxResultResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Pending");
        assert!(resp.result.is_none());
    }

    #[test]
    fn test_result_response_error() {
        let json = r#"{
            "status": "Error",
            "result": {"sample": "", "error": "Something went wrong"}
        }"#;
        let resp: FluxResultResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Error");
        assert_eq!(
            resp.result.unwrap().error.as_deref(),
            Some("Something went wrong")
        );
    }

    #[test]
    fn test_request_serialization_skips_none() {
        let req = GenerationRequest::new("A sunset");
        let flux_req = FluxRequest::from_generation_request(&req);
        let json = serde_json::to_value(&flux_req).unwrap();

        assert!(json.get("prompt").is_some());
        assert!(json.get("width").is_none());
        assert!(json.get("height").is_none());
        assert!(json.get("seed").is_none());
        assert!(json.get("aspect_ratio").is_none());
        assert!(json.get("input_image").is_none());
    }
}
