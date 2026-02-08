//! OpenAI image generation provider (gpt-image-1, dall-e-3).

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;

const GENERATIONS_URL: &str = "https://api.openai.com/v1/images/generations";
const EDITS_URL: &str = "https://api.openai.com/v1/images/edits";

/// OpenAI image model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiImageModel {
    /// GPT Image 1 - OpenAI's latest image generation model.
    #[default]
    GptImage1,
    /// DALL-E 3 - high quality image generation.
    DallE3,
}

impl OpenAiImageModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GptImage1 => "gpt-image-1",
            Self::DallE3 => "dall-e-3",
        }
    }
}

/// Builder for OpenAiImageProvider.
#[derive(Debug, Clone, Default)]
pub struct OpenAiImageProviderBuilder {
    api_key: Option<String>,
    model: OpenAiImageModel,
    quality: Option<String>,
}

impl OpenAiImageProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `OPENAI_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the OpenAI image model variant.
    pub fn model(mut self, model: OpenAiImageModel) -> Self {
        self.model = model;
        self
    }

    /// Sets the quality. For gpt-image-1: "low", "medium", "high" (default: "auto").
    /// For dall-e-3: "standard", "hd" (default: "standard").
    pub fn quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    /// Builds the provider, resolving the API key.
    pub fn build(self) -> Result<OpenAiImageProvider> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("OPENAI_API_KEY not set and no API key provided".into())
            })?;

        Ok(OpenAiImageProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            quality: self.quality,
        })
    }
}

/// OpenAI image generation provider.
pub struct OpenAiImageProvider {
    client: reqwest::Client,
    api_key: String,
    model: OpenAiImageModel,
    quality: Option<String>,
}

impl OpenAiImageProvider {
    /// Creates a new `OpenAiImageProviderBuilder`.
    pub fn builder() -> OpenAiImageProviderBuilder {
        OpenAiImageProviderBuilder::new()
    }

    /// Maps width/height or aspect_ratio to an OpenAI size string.
    fn resolve_size(request: &GenerationRequest, model: &OpenAiImageModel) -> Option<String> {
        // Explicit width/height takes priority
        if let (Some(w), Some(h)) = (request.width, request.height) {
            return Some(format!("{}x{}", w, h));
        }

        // Map aspect ratio to closest supported size
        if let Some(ar) = &request.aspect_ratio {
            let size = match model {
                OpenAiImageModel::GptImage1 => match ar {
                    crate::image::AspectRatio::Square => "1024x1024",
                    crate::image::AspectRatio::Landscape => "1536x1024",
                    crate::image::AspectRatio::Portrait => "1024x1536",
                    crate::image::AspectRatio::Standard => "1536x1024",
                    crate::image::AspectRatio::StandardPortrait => "1024x1536",
                    crate::image::AspectRatio::Ultrawide => "1536x1024",
                    crate::image::AspectRatio::ThreeTwo => "1536x1024",
                    crate::image::AspectRatio::TwoThree => "1024x1536",
                },
                OpenAiImageModel::DallE3 => match ar {
                    crate::image::AspectRatio::Square => "1024x1024",
                    crate::image::AspectRatio::Landscape => "1792x1024",
                    crate::image::AspectRatio::Portrait => "1024x1792",
                    crate::image::AspectRatio::Standard => "1792x1024",
                    crate::image::AspectRatio::StandardPortrait => "1024x1792",
                    crate::image::AspectRatio::Ultrawide => "1792x1024",
                    crate::image::AspectRatio::ThreeTwo => "1792x1024",
                    crate::image::AspectRatio::TwoThree => "1024x1792",
                },
            };
            return Some(size.to_string());
        }

        None
    }

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 402 {
            return GenVizError::Billing(sanitize_error_message(&text));
        }
        if status == 413 {
            return GenVizError::InvalidRequest(
                "Image too large. Reduce image size and try again.".into(),
            );
        }
        if status == 429 {
            // Distinguish insufficient_quota (not retryable) from transient rate limit
            if text.contains("insufficient_quota") || text.contains("exceeded your current quota") {
                return GenVizError::Billing(sanitize_error_message(&text));
            }
            let retry_after = parse_retry_after(headers).map(std::time::Duration::from_secs);
            return GenVizError::RateLimited { retry_after };
        }
        if status == 401 || status == 403 {
            return GenVizError::Auth(text);
        }
        let lower = text.to_lowercase();
        if lower.contains("safety") || lower.contains("blocked") || lower.contains("content_policy")
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
impl ImageProvider for OpenAiImageProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        // Use edits endpoint if input image is provided
        if request.is_edit() {
            return self.generate_edit(request, start).await;
        }

        let body = OpenAiImageRequest::from_generation_request(request, &self.model, &self.quality);

        let response = self
            .client
            .post(GENERATIONS_URL)
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

        let openai_response: OpenAiImageResponse = response.json().await?;

        let image_data = openai_response.data.into_iter().next().ok_or_else(|| {
            GenVizError::UnexpectedResponse("No images in OpenAI response".into())
        })?;

        // Handle b64_json or url response formats
        let data = if let Some(b64) = image_data.b64_json {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| GenVizError::Decode(e.to_string()))?
        } else if let Some(url) = image_data.url {
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
                "OpenAI response contained no image data".into(),
            ));
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::OpenAI,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: None,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::OpenAI
    }

    async fn health_check(&self) -> Result<()> {
        if self.api_key.starts_with("sk-") {
            Ok(())
        } else {
            Err(GenVizError::Auth("Invalid API key format".into()))
        }
    }
}

impl OpenAiImageProvider {
    /// Generate an edited image using the edits endpoint (multipart/form-data).
    async fn generate_edit(
        &self,
        request: &GenerationRequest,
        start: Instant,
    ) -> Result<GeneratedImage> {
        let input_image = request
            .input_image
            .as_ref()
            .ok_or_else(|| GenVizError::InvalidRequest("No input image for edit".into()))?;

        // Detect mime type for file extension
        let ext = ImageFormat::from_magic_bytes(input_image)
            .map(|f| f.extension())
            .unwrap_or("png");

        let filename = format!("image.{}", ext);

        let image_part = reqwest::multipart::Part::bytes(input_image.clone())
            .file_name(filename)
            .mime_str(
                ImageFormat::from_magic_bytes(input_image)
                    .map(|f| f.mime_type())
                    .unwrap_or("image/png"),
            )
            .map_err(|e| GenVizError::InvalidRequest(e.to_string()))?;

        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.as_str().to_string())
            .text("prompt", request.prompt.clone())
            .part("image", image_part);

        if let Some(size) = Self::resolve_size(request, &self.model) {
            form = form.text("size", size);
        }

        let response = self
            .client
            .post(EDITS_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let openai_response: OpenAiImageResponse = response.json().await?;

        let image_data = openai_response.data.into_iter().next().ok_or_else(|| {
            GenVizError::UnexpectedResponse("No images in OpenAI edit response".into())
        })?;

        let data = if let Some(b64) = image_data.b64_json {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| GenVizError::Decode(e.to_string()))?
        } else if let Some(url) = image_data.url {
            let img_response = self.client.get(&url).send().await?;
            if !img_response.status().is_success() {
                return Err(GenVizError::Api {
                    status: img_response.status().as_u16(),
                    message: "Failed to download edited image from URL".into(),
                });
            }
            img_response.bytes().await?.to_vec()
        } else {
            return Err(GenVizError::UnexpectedResponse(
                "OpenAI edit response contained no image data".into(),
            ));
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::OpenAI,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: None,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }
}

#[derive(Debug, Serialize)]
struct OpenAiImageRequest {
    model: String,
    prompt: String,
    n: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
}

impl OpenAiImageRequest {
    fn from_generation_request(
        req: &GenerationRequest,
        model: &OpenAiImageModel,
        quality: &Option<String>,
    ) -> Self {
        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            n: 1,
            size: OpenAiImageProvider::resolve_size(req, model),
            quality: quality.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiImageResponse {
    data: Vec<OpenAiImageData>,
}

#[derive(Debug, Deserialize)]
struct OpenAiImageData {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    revised_prompt: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_as_str() {
        assert_eq!(OpenAiImageModel::GptImage1.as_str(), "gpt-image-1");
        assert_eq!(OpenAiImageModel::DallE3.as_str(), "dall-e-3");
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = OpenAiImageProviderBuilder::new().api_key("sk-test").build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_builder_without_key_fails() {
        // Clear env var to ensure it fails
        std::env::remove_var("OPENAI_API_KEY");
        let provider = OpenAiImageProviderBuilder::new().build();
        assert!(provider.is_err());
    }

    #[test]
    fn test_builder_with_model() {
        let provider = OpenAiImageProviderBuilder::new()
            .api_key("sk-test")
            .model(OpenAiImageModel::DallE3)
            .build()
            .unwrap();
        assert_eq!(provider.model, OpenAiImageModel::DallE3);
    }

    #[test]
    fn test_request_construction_basic() {
        let req = GenerationRequest::new("A sunset");
        let openai_req =
            OpenAiImageRequest::from_generation_request(&req, &OpenAiImageModel::GptImage1, &None);

        assert_eq!(openai_req.prompt, "A sunset");
        assert_eq!(openai_req.model, "gpt-image-1");
        assert_eq!(openai_req.n, 1);
        assert!(openai_req.size.is_none());
        assert!(openai_req.quality.is_none());
    }

    #[test]
    fn test_request_construction_with_size() {
        let req = GenerationRequest::new("A sunset").with_size(512, 512);
        let openai_req =
            OpenAiImageRequest::from_generation_request(&req, &OpenAiImageModel::GptImage1, &None);

        assert_eq!(openai_req.size.as_deref(), Some("512x512"));
    }

    #[test]
    fn test_request_construction_with_quality() {
        let req = GenerationRequest::new("A sunset");
        let quality = Some("high".to_string());
        let openai_req = OpenAiImageRequest::from_generation_request(
            &req,
            &OpenAiImageModel::GptImage1,
            &quality,
        );

        assert_eq!(openai_req.quality.as_deref(), Some("high"));
    }

    #[test]
    fn test_size_mapping_gpt_image_aspect_ratio() {
        let req =
            GenerationRequest::new("test").with_aspect_ratio(crate::image::AspectRatio::Landscape);
        let size = OpenAiImageProvider::resolve_size(&req, &OpenAiImageModel::GptImage1);
        assert_eq!(size.as_deref(), Some("1536x1024"));

        let req =
            GenerationRequest::new("test").with_aspect_ratio(crate::image::AspectRatio::Portrait);
        let size = OpenAiImageProvider::resolve_size(&req, &OpenAiImageModel::GptImage1);
        assert_eq!(size.as_deref(), Some("1024x1536"));

        let req =
            GenerationRequest::new("test").with_aspect_ratio(crate::image::AspectRatio::Square);
        let size = OpenAiImageProvider::resolve_size(&req, &OpenAiImageModel::GptImage1);
        assert_eq!(size.as_deref(), Some("1024x1024"));
    }

    #[test]
    fn test_size_mapping_dalle3_aspect_ratio() {
        let req =
            GenerationRequest::new("test").with_aspect_ratio(crate::image::AspectRatio::Landscape);
        let size = OpenAiImageProvider::resolve_size(&req, &OpenAiImageModel::DallE3);
        assert_eq!(size.as_deref(), Some("1792x1024"));

        let req =
            GenerationRequest::new("test").with_aspect_ratio(crate::image::AspectRatio::Portrait);
        let size = OpenAiImageProvider::resolve_size(&req, &OpenAiImageModel::DallE3);
        assert_eq!(size.as_deref(), Some("1024x1792"));
    }

    #[test]
    fn test_response_deserialization_url() {
        let json = r#"{"data": [{"url": "https://example.com/img.png", "revised_prompt": "A beautiful sunset over the ocean"}]}"#;
        let resp: OpenAiImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://example.com/img.png")
        );
        assert!(resp.data[0].revised_prompt.is_some());
    }

    #[test]
    fn test_response_deserialization_b64() {
        let json = r#"{"data": [{"b64_json": "AQID"}]}"#;
        let resp: OpenAiImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("AQID"));
        assert!(resp.data[0].url.is_none());
    }

    #[test]
    fn test_request_serialization_skips_none_fields() {
        let req = GenerationRequest::new("A sunset");
        let openai_req =
            OpenAiImageRequest::from_generation_request(&req, &OpenAiImageModel::GptImage1, &None);
        let json = serde_json::to_value(&openai_req).unwrap();

        assert!(json.get("size").is_none());
        assert!(json.get("quality").is_none());
        assert!(json.get("prompt").is_some());
        assert!(json.get("model").is_some());
    }

    #[test]
    fn test_explicit_size_overrides_aspect_ratio() {
        let req = GenerationRequest::new("test")
            .with_size(256, 256)
            .with_aspect_ratio(crate::image::AspectRatio::Landscape);
        let size = OpenAiImageProvider::resolve_size(&req, &OpenAiImageModel::GptImage1);
        // width/height takes priority
        assert_eq!(size.as_deref(), Some("256x256"));
    }
}
