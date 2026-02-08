//! Grok Imagine (xAI) image generation provider.

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
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
    /// Grok Imagine - xAI's image generation model.
    #[default]
    GrokImagine,
}

impl GrokModel {
    /// Returns the API model identifier string.
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
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `XAI_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the Grok model variant.
    pub fn model(mut self, model: GrokModel) -> Self {
        self.model = model;
        self
    }

    /// Builds the provider, resolving the API key.
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
    /// Creates a new `GrokProviderBuilder`.
    pub fn builder() -> GrokProviderBuilder {
        GrokProviderBuilder::new()
    }

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 402 {
            return GenVizError::Billing(
                "Insufficient credits. Check your xAI billing at console.x.ai".into(),
            );
        }
        if status == 422 {
            return GenVizError::InvalidRequest(text);
        }
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
            || lower.contains("moderated")
            || lower.contains("content moderated")
            || lower.contains("try a different idea")
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
impl ImageProvider for GrokProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        // Use edits endpoint if input image is provided
        let (url, body) = if request.input_image.is_some() {
            (
                EDITS_URL,
                serde_json::to_value(GrokEditRequest::from_generation_request(
                    request,
                    &self.model,
                ))?,
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
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let grok_response: GrokResponse = response.json().await?;

        let image_data = grok_response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| {
                GenVizError::UnexpectedResponse(
                    "No images in Grok response. The model may not have generated output for this prompt.".into(),
                )
            })?;

        // Handle b64_json, image (edit endpoint), or url response formats
        let data = if let Some(b64) = image_data.b64_json {
            base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| GenVizError::Decode(e.to_string()))?
        } else if let Some(b64) = image_data.image {
            // Edit endpoint returns base64 in "image" field
            base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| GenVizError::Decode(e.to_string()))?
        } else if let Some(url) = image_data.url {
            // Download image from URL
            let img_response = self.client.get(&url).send().await?;
            if !img_response.status().is_success() {
                return Err(GenVizError::Api {
                    status: img_response.status().as_u16(),
                    message: "Failed to download image from URL".into(),
                });
            }
            img_response.bytes().await?.to_vec()
        } else {
            return Err(GenVizError::UnexpectedResponse(
                "Grok response contained no image data".into(),
            ));
        };

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
        let response = self
            .client
            .get("https://api.x.ai/v1/models")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        match response.status().as_u16() {
            401 | 403 => Err(GenVizError::Auth("Invalid API key".into())),
            s if !(200..300).contains(&s) => Err(GenVizError::Api {
                status: s,
                message: "Health check failed".into(),
            }),
            _ => Ok(()),
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

/// Image URL structure for edit endpoint.
#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

/// Request for image editing endpoint.
#[derive(Debug, Serialize)]
struct GrokEditRequest {
    model: String,
    prompt: String,
    /// Image as URL object (data URI or public URL).
    image: ImageUrl,
    n: u32,
    response_format: String,
}

impl GrokEditRequest {
    fn from_generation_request(req: &GenerationRequest, model: &GrokModel) -> Self {
        // Format as data URI
        let image_b64 = req
            .input_image
            .as_ref()
            .map(|img| base64::engine::general_purpose::STANDARD.encode(img))
            .unwrap_or_default();

        // Detect mime type from magic bytes
        let mime = req
            .input_image
            .as_ref()
            .and_then(|img| crate::image::types::ImageFormat::from_magic_bytes(img))
            .map(|f| f.mime_type())
            .unwrap_or("image/png");

        let data_uri = format!("data:{};base64,{}", mime, image_b64);

        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            image: ImageUrl { url: data_uri },
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
    /// Base64-encoded image (when response_format is b64_json)
    #[serde(default)]
    b64_json: Option<String>,
    /// Image URL (when response_format is url)
    #[serde(default)]
    url: Option<String>,
    /// Base64-encoded image from edit endpoint
    #[serde(default)]
    image: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grok_model_as_str() {
        assert_eq!(GrokModel::GrokImagine.as_str(), "grok-imagine-image");
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = GrokProviderBuilder::new().api_key("xai-test").build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_request_construction_basic() {
        let req = GenerationRequest::new("A city");
        let grok_req = GrokRequest::from_generation_request(&req, &GrokModel::GrokImagine);

        assert_eq!(grok_req.prompt, "A city");
        assert_eq!(grok_req.model, "grok-imagine-image");
        assert_eq!(grok_req.n, 1);
        assert_eq!(grok_req.response_format, "b64_json");
        assert!(grok_req.aspect_ratio.is_none());
    }

    #[test]
    fn test_request_construction_with_aspect_ratio() {
        let req =
            GenerationRequest::new("A city").with_aspect_ratio(crate::image::AspectRatio::Portrait);
        let grok_req = GrokRequest::from_generation_request(&req, &GrokModel::GrokImagine);

        assert_eq!(grok_req.aspect_ratio.as_deref(), Some("9:16"));
    }

    #[test]
    fn test_edit_request_construction() {
        let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let req = GenerationRequest::new("Edit this").with_input_image(png_data);
        let edit_req = GrokEditRequest::from_generation_request(&req, &GrokModel::GrokImagine);

        assert_eq!(edit_req.prompt, "Edit this");
        assert!(edit_req.image.url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_response_deserialization_b64() {
        let json = r#"{"data": [{"b64_json": "AQID"}]}"#;
        let resp: GrokResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("AQID"));
        assert!(resp.data[0].url.is_none());
    }

    #[test]
    fn test_response_deserialization_url() {
        let json = r#"{"data": [{"url": "https://example.com/img.png"}]}"#;
        let resp: GrokResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://example.com/img.png")
        );
        assert!(resp.data[0].b64_json.is_none());
    }

    #[test]
    fn test_response_deserialization_edit() {
        let json = r#"{"data": [{"image": "AQID"}]}"#;
        let resp: GrokResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data[0].image.as_deref(), Some("AQID"));
    }

    #[test]
    fn test_request_serialization_skips_none_aspect_ratio() {
        let req = GenerationRequest::new("A city");
        let grok_req = GrokRequest::from_generation_request(&req, &GrokModel::GrokImagine);
        let json = serde_json::to_value(&grok_req).unwrap();

        assert!(json.get("aspect_ratio").is_none());
        assert!(json.get("prompt").is_some());
    }
}
