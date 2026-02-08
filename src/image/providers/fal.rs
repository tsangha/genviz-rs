//! fal.ai image generation provider.

use crate::error::{sanitize_error_message, GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    AspectRatio, GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat,
    ImageProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// fal.ai image model variants.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FalImageModel {
    /// Flux Schnell - fast, efficient image model (default).
    #[default]
    FluxSchnell,
    /// Flux Pro v1.1 - high-quality image model.
    FluxPro,
    /// Flux Pro v1.1 Ultra - highest-quality Flux model.
    FluxProUltra,
    /// Recraft V3 - design-focused image model.
    RecraftV3,
    /// Ideogram V3 - text-in-image capable model.
    Ideogram3,
    /// HiDream I1 Full - high-fidelity dream-like images.
    HiDream,
    /// Custom fal.ai model by ID (e.g., "fal-ai/some-model").
    Custom(String),
}

impl FalImageModel {
    /// Returns the fal.ai model identifier string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::FluxSchnell => "fal-ai/flux/schnell",
            Self::FluxPro => "fal-ai/flux-pro/v1.1",
            Self::FluxProUltra => "fal-ai/flux-pro/v1.1-ultra",
            Self::RecraftV3 => "fal-ai/recraft-v3",
            Self::Ideogram3 => "fal-ai/ideogram/v3",
            Self::HiDream => "fal-ai/hidream-i1-full",
            Self::Custom(id) => id,
        }
    }
}

/// Builder for [`FalImageProvider`].
#[derive(Debug, Clone)]
pub struct FalImageProviderBuilder {
    api_key: Option<String>,
    model: FalImageModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for FalImageProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: FalImageModel::default(),
            poll_interval: Duration::from_secs(1),
            timeout: Duration::from_secs(600), // 10 minutes (HiDream can take >5min)
        }
    }
}

impl FalImageProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `FAL_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the fal.ai model variant.
    pub fn model(mut self, model: FalImageModel) -> Self {
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

    /// Builds the provider, resolving credentials.
    pub fn build(self) -> Result<FalImageProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("FAL_KEY").ok())
            .ok_or_else(|| GenVizError::Auth("FAL_KEY not set and no API key provided".into()))?;

        Ok(FalImageProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// fal.ai image generation provider.
///
/// Uses the fal.ai queue API for asynchronous image generation.
/// Supports Flux, Recraft, Ideogram, HiDream, and custom models.
pub struct FalImageProvider {
    client: reqwest::Client,
    api_key: String,
    model: FalImageModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl FalImageProvider {
    /// Creates a new [`FalImageProviderBuilder`].
    pub fn builder() -> FalImageProviderBuilder {
        FalImageProviderBuilder::new()
    }

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
        let text = sanitize_error_message(text);

        if let Ok(error_resp) = serde_json::from_str::<FalErrorResponse>(&text) {
            let msg = sanitize_error_message(&error_resp.detail);
            if status == 401 || status == 403 {
                return GenVizError::Auth(msg);
            }
            if status == 429 {
                return GenVizError::RateLimited { retry_after: None };
            }
            return GenVizError::Api {
                status,
                message: msg,
            };
        }

        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        if status == 429 {
            return GenVizError::RateLimited { retry_after: None };
        }

        GenVizError::Api {
            status,
            message: text,
        }
    }

    /// Submit an image generation request to the fal.ai queue.
    ///
    /// Returns the submit response with request_id, status_url, and response_url.
    /// fal.ai's queue uses different URL paths than the submit endpoint for models
    /// with nested paths (e.g., `fal-ai/flux/schnell` queues under `fal-ai/flux`),
    /// so we rely on their returned URLs for polling and result fetching.
    async fn submit(&self, request: &GenerationRequest) -> Result<FalSubmitResponse> {
        let model_id = self.model.as_str();
        let url = format!("https://queue.fal.run/{}", model_id);

        let body = FalImageRequest::from_request(request);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Key {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let submit_response: FalSubmitResponse = response.json().await?;

        Ok(submit_response)
    }

    /// Poll the queue until the request is complete.
    async fn poll_until_ready(&self, request_id: &str, status_url: &str) -> Result<()> {
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(status_url)
                .header("Authorization", format!("Key {}", self.api_key))
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text));
            }

            let poll_response: FalStatusResponse = response.json().await?;

            match poll_response.status.as_str() {
                "COMPLETED" => return Ok(()),
                "IN_QUEUE" | "IN_PROGRESS" => {
                    tracing::debug!(
                        request_id = %request_id,
                        status = %poll_response.status,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling fal.ai image generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                }
                "FAILED" => {
                    return Err(GenVizError::UnexpectedResponse(
                        "fal.ai generation failed".into(),
                    ));
                }
                other => {
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "fal.ai returned unexpected status: {}",
                        other
                    )));
                }
            }
        }
    }

    /// Fetch the generation result from the queue.
    ///
    /// Tries the fal.ai-provided `response_url` first. If that returns a 404
    /// (which happens for some models with nested paths), falls back to
    /// constructing the URL from the full model_id.
    async fn fetch_result(
        &self,
        response_url: &str,
        model_id: &str,
        request_id: &str,
    ) -> Result<FalResultResponse> {
        let response = self
            .client
            .get(response_url)
            .header("Authorization", format!("Key {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if status.as_u16() == 404 || status.as_u16() == 405 {
            // Fallback: construct URL from full model_id
            tracing::debug!(
                response_url = %response_url,
                "fal.ai response_url returned {}, falling back to model_id-based URL",
                status.as_u16()
            );
            let fallback_url =
                format!("https://queue.fal.run/{}/requests/{}", model_id, request_id);
            let fallback = self
                .client
                .get(&fallback_url)
                .header("Authorization", format!("Key {}", self.api_key))
                .send()
                .await?;

            let fb_status = fallback.status();
            if !fb_status.is_success() {
                let text = fallback.text().await.unwrap_or_default();
                return Err(self.parse_error(fb_status.as_u16(), &text));
            }

            let result: FalResultResponse = fallback.json().await?;
            return Ok(result);
        }

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let result: FalResultResponse = response.json().await?;
        Ok(result)
    }

    /// Download an image from the given URL.
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
impl ImageProvider for FalImageProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();
        let model_id = self.model.as_str();

        let submit = self.submit(request).await?;
        tracing::debug!(request_id = %submit.request_id, "submitted fal.ai image generation request");

        self.poll_until_ready(&submit.request_id, &submit.status_url)
            .await?;
        tracing::debug!(request_id = %submit.request_id, "fal.ai image generation complete");

        let result = self
            .fetch_result(&submit.response_url, model_id, &submit.request_id)
            .await?;

        let image_info =
            result.images.into_iter().next().ok_or_else(|| {
                GenVizError::UnexpectedResponse("fal.ai returned no images".into())
            })?;

        let data = self.download(&image_info.url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Fal,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: result.seed,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Fal
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(GenVizError::Auth("FAL_KEY is empty".into()));
        }
        Ok(())
    }
}

// -- Request types --

/// Maps an `AspectRatio` to the fal.ai `image_size` string format.
fn aspect_ratio_to_image_size(ratio: AspectRatio) -> String {
    match ratio {
        AspectRatio::Square => "square".into(),
        AspectRatio::Landscape => "landscape_16_9".into(),
        AspectRatio::Portrait => "portrait_9_16".into(),
        AspectRatio::Standard => "landscape_4_3".into(),
        AspectRatio::StandardPortrait => "portrait_4_3".into(),
        AspectRatio::Ultrawide => "landscape_16_9".into(),
        AspectRatio::ThreeTwo => "landscape_4_3".into(),
        AspectRatio::TwoThree => "portrait_4_3".into(),
    }
}

/// Represents the `image_size` field, which can be a named preset or explicit dimensions.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum FalImageSize {
    /// A named preset like "square", "landscape_16_9", etc.
    Named(String),
    /// Explicit pixel dimensions.
    Custom {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
}

#[derive(Debug, Serialize)]
struct FalImageRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_size: Option<FalImageSize>,
    num_images: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
}

impl FalImageRequest {
    fn from_request(req: &GenerationRequest) -> Self {
        use base64::Engine;

        let image_size = if let (Some(w), Some(h)) = (req.width, req.height) {
            Some(FalImageSize::Custom {
                width: w,
                height: h,
            })
        } else {
            req.aspect_ratio
                .map(|ar| FalImageSize::Named(aspect_ratio_to_image_size(ar)))
        };

        let image_url = req.input_image.as_ref().map(|img| {
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(img)
            )
        });

        Self {
            prompt: req.prompt.clone(),
            image_size,
            num_images: 1,
            seed: req.seed,
            image_url,
        }
    }
}

// -- Response types --

#[derive(Debug, Deserialize)]
struct FalSubmitResponse {
    request_id: String,
    /// URL to poll for status (provided by fal.ai).
    status_url: String,
    /// URL to fetch completed result (provided by fal.ai).
    response_url: String,
}

#[derive(Debug, Deserialize)]
struct FalStatusResponse {
    status: String,
}

#[derive(Debug, Deserialize)]
struct FalResultResponse {
    images: Vec<FalImageInfo>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct FalImageInfo {
    url: String,
    #[serde(default)]
    #[allow(dead_code)]
    content_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FalErrorResponse {
    detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fal_image_model_as_str() {
        assert_eq!(FalImageModel::FluxSchnell.as_str(), "fal-ai/flux/schnell");
        assert_eq!(FalImageModel::FluxPro.as_str(), "fal-ai/flux-pro/v1.1");
        assert_eq!(
            FalImageModel::FluxProUltra.as_str(),
            "fal-ai/flux-pro/v1.1-ultra"
        );
        assert_eq!(FalImageModel::RecraftV3.as_str(), "fal-ai/recraft-v3");
        assert_eq!(FalImageModel::Ideogram3.as_str(), "fal-ai/ideogram/v3");
        assert_eq!(FalImageModel::HiDream.as_str(), "fal-ai/hidream-i1-full");
    }

    #[test]
    fn test_fal_image_model_default() {
        assert_eq!(FalImageModel::default(), FalImageModel::FluxSchnell);
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = FalImageProviderBuilder::new()
            .api_key("test-key")
            .model(FalImageModel::FluxPro)
            .build();
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.model, FalImageModel::FluxPro);
    }

    #[test]
    fn test_builder_missing_key() {
        std::env::remove_var("FAL_KEY");

        let result = FalImageProviderBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = FalImageProviderBuilder::new()
            .api_key("test-key")
            .poll_interval(Duration::from_secs(2))
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(2));
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_request_construction_basic() {
        let req = GenerationRequest::new("A sunset");
        let fal_req = FalImageRequest::from_request(&req);

        assert_eq!(fal_req.prompt, "A sunset");
        assert_eq!(fal_req.num_images, 1);
        assert!(fal_req.image_size.is_none());
        assert!(fal_req.seed.is_none());
        assert!(fal_req.image_url.is_none());
    }

    #[test]
    fn test_request_construction_with_size() {
        let req = GenerationRequest::new("A sunset").with_size(1024, 768);
        let fal_req = FalImageRequest::from_request(&req);

        match &fal_req.image_size {
            Some(FalImageSize::Custom { width, height }) => {
                assert_eq!(*width, 1024);
                assert_eq!(*height, 768);
            }
            other => panic!("Expected Custom image_size, got {:?}", other),
        }
    }

    #[test]
    fn test_request_construction_with_aspect_ratio() {
        let req = GenerationRequest::new("A sunset").with_aspect_ratio(AspectRatio::Landscape);
        let fal_req = FalImageRequest::from_request(&req);

        match &fal_req.image_size {
            Some(FalImageSize::Named(name)) => {
                assert_eq!(name, "landscape_16_9");
            }
            other => panic!("Expected Named image_size, got {:?}", other),
        }
    }

    #[test]
    fn test_request_with_input_image() {
        let req = GenerationRequest::new("Edit this").with_input_image(vec![0x89, 0x50, 0x4E]);
        let fal_req = FalImageRequest::from_request(&req);

        assert!(fal_req.image_url.is_some());
        let url = fal_req.image_url.unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_request_serialization_skips_none() {
        let req = GenerationRequest::new("A sunset");
        let fal_req = FalImageRequest::from_request(&req);
        let json = serde_json::to_value(&fal_req).unwrap();

        assert!(json.get("prompt").is_some());
        assert!(json.get("num_images").is_some());
        assert!(json.get("image_size").is_none());
        assert!(json.get("seed").is_none());
        assert!(json.get("image_url").is_none());
    }

    #[test]
    fn test_submit_response_deserialization() {
        let json = r#"{
            "request_id": "abc-123-def",
            "status_url": "https://queue.fal.run/fal-ai/flux/requests/abc-123-def/status",
            "response_url": "https://queue.fal.run/fal-ai/flux/requests/abc-123-def"
        }"#;
        let resp: FalSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.request_id, "abc-123-def");
        assert!(resp.status_url.contains("status"));
        assert!(resp.response_url.contains("abc-123-def"));
    }

    #[test]
    fn test_status_response_deserialization() {
        for status in &["IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED"] {
            let json = format!(r#"{{"status": "{}"}}"#, status);
            let resp: FalStatusResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(resp.status, *status);
        }
    }

    #[test]
    fn test_result_response_deserialization() {
        let json = r#"{
            "images": [{"url": "https://fal.media/files/image.png", "content_type": "image/png"}],
            "seed": 42
        }"#;
        let resp: FalResultResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.images.len(), 1);
        assert_eq!(resp.images[0].url, "https://fal.media/files/image.png");
        assert_eq!(resp.images[0].content_type.as_deref(), Some("image/png"));
        assert_eq!(resp.seed, Some(42));
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"detail": "Invalid API key"}"#;
        let resp: FalErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.detail, "Invalid API key");
    }

    #[test]
    fn test_parse_error_auth() {
        let provider = FalImageProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let err = provider.parse_error(401, r#"{"detail": "Unauthorized"}"#);
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_parse_error_rate_limited() {
        let provider = FalImageProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let err = provider.parse_error(429, r#"{"detail": "Too many requests"}"#);
        assert!(matches!(err, GenVizError::RateLimited { .. }));
    }

    #[test]
    fn test_parse_error_api() {
        let provider = FalImageProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();
        let err = provider.parse_error(500, r#"{"detail": "Internal server error"}"#);
        assert!(matches!(err, GenVizError::Api { .. }));
    }

    #[test]
    fn test_custom_model() {
        let custom = FalImageModel::Custom("fal-ai/my-custom-model".into());
        assert_eq!(custom.as_str(), "fal-ai/my-custom-model");

        let provider = FalImageProviderBuilder::new()
            .api_key("test-key")
            .model(custom.clone())
            .build()
            .unwrap();
        assert_eq!(
            provider.model,
            FalImageModel::Custom("fal-ai/my-custom-model".into())
        );
    }

    #[test]
    fn test_aspect_ratio_mapping() {
        assert_eq!(aspect_ratio_to_image_size(AspectRatio::Square), "square");
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::Landscape),
            "landscape_16_9"
        );
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::Portrait),
            "portrait_9_16"
        );
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::Standard),
            "landscape_4_3"
        );
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::StandardPortrait),
            "portrait_4_3"
        );
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::Ultrawide),
            "landscape_16_9"
        );
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::ThreeTwo),
            "landscape_4_3"
        );
        assert_eq!(
            aspect_ratio_to_image_size(AspectRatio::TwoThree),
            "portrait_4_3"
        );
    }
}
