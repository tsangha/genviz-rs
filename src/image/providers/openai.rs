//! OpenAI image generation provider (gpt-image-2.5-sunburst, gpt-image-2.5-flare, gpt-image-2).

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
    /// GPT Image 2.5 Sunburst — base model, highest quality and editing precision.
    GptImage25Sunburst,
    /// GPT Image 2.5 Flare — fast model, OpenAI's default choice for most apps.
    #[default]
    GptImage25Flare,
    /// GPT Image 2 — previous generation.
    GptImage2,
}

impl OpenAiImageModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GptImage25Sunburst => "gpt-image-2.5-sunburst",
            Self::GptImage25Flare => "gpt-image-2.5-flare",
            Self::GptImage2 => "gpt-image-2",
        }
    }

    /// Parses an API model identifier.
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::GptImage25Sunburst,
            Self::GptImage25Flare,
            Self::GptImage2,
        ]
        .into_iter()
        .find(|m| m.as_str() == name)
    }
}

/// Builder for OpenAiImageProvider.
#[derive(Debug, Clone, Default)]
pub struct OpenAiImageProviderBuilder {
    api_key: Option<String>,
    model: OpenAiImageModel,
    quality: Option<String>,
    background: Option<String>,
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

    /// Sets the quality: "low", "medium", "high", "xhigh", "max" or "auto" (default).
    pub fn quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    /// Sets the background: "transparent", "opaque" or "auto" (default).
    pub fn background(mut self, background: impl Into<String>) -> Self {
        self.background = Some(background.into());
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
            background: self.background,
        })
    }
}

/// OpenAI image generation provider.
pub struct OpenAiImageProvider {
    client: reqwest::Client,
    api_key: String,
    model: OpenAiImageModel,
    quality: Option<String>,
    background: Option<String>,
}

impl OpenAiImageProvider {
    /// Creates a new `OpenAiImageProviderBuilder`.
    pub fn builder() -> OpenAiImageProviderBuilder {
        OpenAiImageProviderBuilder::new()
    }

    /// Maps width/height or aspect_ratio to an OpenAI size string.
    fn resolve_size(request: &GenerationRequest) -> Option<String> {
        if let (Some(w), Some(h)) = (request.width, request.height) {
            return Some(format!("{}x{}", w, h));
        }
        use crate::image::AspectRatio::*;
        request.aspect_ratio.map(|ar| {
            match ar {
                Square => "1024x1024",
                Landscape | Standard | Ultrawide | ThreeTwo => "1536x1024",
                Portrait | StandardPortrait | TwoThree => "1024x1536",
            }
            .to_string()
        })
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

        let body = OpenAiImageRequest::from_generation_request(
            request,
            &self.model,
            &self.quality,
            &self.background,
        );

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
        let field = if request.input_images.len() > 1 {
            "image[]"
        } else {
            "image"
        };
        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.as_str().to_string())
            .text("prompt", request.prompt.clone());
        for (i, input_image) in request.input_images.iter().enumerate() {
            let format = ImageFormat::from_magic_bytes(input_image).unwrap_or(ImageFormat::Png);
            let part = reqwest::multipart::Part::bytes(input_image.clone())
                .file_name(format!("image{}.{}", i, format.extension()))
                .mime_str(format.mime_type())
                .map_err(|e| GenVizError::InvalidRequest(e.to_string()))?;
            form = form.part(field, part);
        }

        for (name, value) in [
            ("size", Self::resolve_size(request)),
            ("quality", self.quality.clone()),
            ("background", self.background.clone()),
        ] {
            if let Some(value) = value {
                form = form.text(name, value);
            }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<String>,
}

impl OpenAiImageRequest {
    fn from_generation_request(
        req: &GenerationRequest,
        model: &OpenAiImageModel,
        quality: &Option<String>,
        background: &Option<String>,
    ) -> Self {
        Self {
            model: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            n: 1,
            size: OpenAiImageProvider::resolve_size(req),
            quality: quality.clone(),
            background: background.clone(),
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
    fn test_model_names_round_trip() {
        for name in [
            "gpt-image-2.5-sunburst",
            "gpt-image-2.5-flare",
            "gpt-image-2",
        ] {
            assert_eq!(OpenAiImageModel::from_name(name).unwrap().as_str(), name);
        }
        assert_eq!(OpenAiImageModel::from_name("dall-e-3"), None);
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = OpenAiImageProviderBuilder::new().api_key("sk-test").build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_builder_without_key_fails() {
        std::env::remove_var("OPENAI_API_KEY");
        let provider = OpenAiImageProviderBuilder::new().build();
        assert!(provider.is_err());
    }

    #[test]
    fn test_request_serialization() {
        let req = GenerationRequest::new("A sunset").with_size(1536, 512);
        let body = OpenAiImageRequest::from_generation_request(
            &req,
            &OpenAiImageModel::GptImage25Sunburst,
            &Some("low".into()),
            &Some("transparent".into()),
        );
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "gpt-image-2.5-sunburst");
        assert_eq!(json["size"], "1536x512");
        assert_eq!(json["quality"], "low");
        assert_eq!(json["background"], "transparent");

        let bare = OpenAiImageRequest::from_generation_request(
            &GenerationRequest::new("A sunset"),
            &OpenAiImageModel::GptImage2,
            &None,
            &None,
        );
        let json = serde_json::to_value(&bare).unwrap();
        assert!(json.get("size").is_none());
        assert!(json.get("background").is_none());
    }

    #[test]
    fn test_size_mapping_aspect_ratio() {
        let req =
            GenerationRequest::new("test").with_aspect_ratio(crate::image::AspectRatio::Portrait);
        assert_eq!(
            OpenAiImageProvider::resolve_size(&req).as_deref(),
            Some("1024x1536")
        );
    }

    #[test]
    fn test_response_deserialization_b64() {
        let json = r#"{"data": [{"b64_json": "AQID"}]}"#;
        let resp: OpenAiImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("AQID"));
    }
}
