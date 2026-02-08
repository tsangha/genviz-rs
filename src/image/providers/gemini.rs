//! Gemini (Google) image generation provider.

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
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
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `GOOGLE_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the Gemini model variant.
    pub fn model(mut self, model: GeminiModel) -> Self {
        self.model = model;
        self
    }

    /// Builds the provider, resolving the API key.
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
    /// Creates a new `GeminiProviderBuilder`.
    pub fn builder() -> GeminiProviderBuilder {
        GeminiProviderBuilder::new()
    }

    async fn generate_impl(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.model.as_str(),
        );

        let body = GeminiRequest::from_generation_request(request);

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
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

        let gemini_response: GeminiResponse = response.json().await?;

        // Check prompt_feedback for blocks (returned as HTTP 200)
        if let Some(ref feedback) = gemini_response.prompt_feedback {
            if let Some(ref reason) = feedback.block_reason {
                let msg = feedback
                    .block_reason_message
                    .clone()
                    .unwrap_or_else(|| format!("Prompt blocked: {}", reason));
                return Err(GenVizError::ContentBlocked(msg));
            }
        }

        // Check finish_reason on the first candidate
        let candidate = gemini_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| {
                GenVizError::UnexpectedResponse("No candidates in Gemini response".into())
            })?;

        if let Some(ref finish_reason) = candidate.finish_reason {
            match finish_reason.as_str() {
                "SAFETY"
                | "IMAGE_SAFETY"
                | "IMAGE_PROHIBITED_CONTENT"
                | "IMAGE_RECITATION"
                | "RECITATION"
                | "PROHIBITED_CONTENT"
                | "BLOCKLIST" => {
                    return Err(GenVizError::ContentBlocked(format!(
                        "Content blocked by Gemini safety filter: {}",
                        finish_reason
                    )));
                }
                "IMAGE_OTHER" | "NO_IMAGE" => {
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "Generation failed: {}. Try a different prompt.",
                        finish_reason
                    )));
                }
                _ => {} // STOP, MAX_TOKENS, etc. are normal
            }
        }

        let content = candidate.content.ok_or_else(|| {
            GenVizError::UnexpectedResponse("No content in Gemini candidate".into())
        })?;

        let inline_data = content
            .parts
            .into_iter()
            .find_map(|p| p.inline_data)
            .ok_or_else(|| {
                GenVizError::UnexpectedResponse("No image data in Gemini response".into())
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

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        let text = sanitize_error_message(text);
        if status == 402 {
            return GenVizError::Billing(
                "Gemini billing issue: enable billing at https://aistudio.google.com".into(),
            );
        }
        if status == 404 {
            return GenVizError::InvalidRequest(
                "Model not found. Verify the model name is correct.".into(),
            );
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
            || lower.contains("prohibited")
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
impl ImageProvider for GeminiProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        self.generate_impl(request).await
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Gemini
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}",
            self.model.as_str(),
        );

        let response = self
            .client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;

        match response.status().as_u16() {
            401 | 403 => Err(GenVizError::Auth("Invalid API key".into())),
            404 => Err(GenVizError::InvalidRequest(
                "Model not found. Verify the model name is correct.".into(),
            )),
            s if !(200..300).contains(&s) => Err(GenVizError::Api {
                status: s,
                message: "Health check failed".into(),
            }),
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
    parts: Vec<GeminiRequestPart>,
}

/// A part in a Gemini request - can be text or inline image data.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GeminiRequestPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
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
        let mut parts = Vec::new();

        // Add input image first if present (for editing)
        if let Some(ref image_data) = req.input_image {
            let mime_type = crate::image::types::ImageFormat::from_magic_bytes(image_data)
                .map(|f| f.mime_type())
                .unwrap_or("image/png")
                .to_string();

            parts.push(GeminiRequestPart::InlineData {
                inline_data: GeminiInlineData {
                    mime_type,
                    data: base64::engine::general_purpose::STANDARD.encode(image_data),
                },
            });
        }

        // Add text prompt
        parts.push(GeminiRequestPart::Text {
            text: req.prompt.clone(),
        });

        Self {
            contents: vec![GeminiContent { parts }],
            generation_config: GeminiConfig {
                response_modalities: vec!["IMAGE".to_string()],
                seed: req.seed,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContentResponse>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    #[serde(default)]
    block_reason: Option<String>,
    #[serde(default)]
    block_reason_message: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_model_as_str() {
        assert_eq!(GeminiModel::NanoBanana.as_str(), "gemini-2.5-flash-image");
        assert_eq!(
            GeminiModel::NanoBananaPro.as_str(),
            "nano-banana-pro-preview"
        );
    }

    #[test]
    fn test_gemini_model_default() {
        assert_eq!(GeminiModel::default(), GeminiModel::NanoBananaPro);
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = GeminiProviderBuilder::new()
            .api_key("test-key")
            .model(GeminiModel::NanoBanana)
            .build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_request_construction_basic() {
        let req = GenerationRequest::new("A puppy");
        let gemini_req = GeminiRequest::from_generation_request(&req);

        assert_eq!(gemini_req.contents.len(), 1);
        assert_eq!(gemini_req.contents[0].parts.len(), 1);
        assert_eq!(
            gemini_req.generation_config.response_modalities,
            vec!["IMAGE"]
        );
        assert!(gemini_req.generation_config.seed.is_none());
    }

    #[test]
    fn test_request_construction_with_seed() {
        let req = GenerationRequest::new("A puppy").with_seed(42);
        let gemini_req = GeminiRequest::from_generation_request(&req);

        assert_eq!(gemini_req.generation_config.seed, Some(42));
    }

    #[test]
    fn test_request_construction_with_input_image() {
        // PNG magic bytes
        let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let req = GenerationRequest::new("Edit this").with_input_image(png_data);
        let gemini_req = GeminiRequest::from_generation_request(&req);

        // Should have 2 parts: inline image + text prompt
        assert_eq!(gemini_req.contents[0].parts.len(), 2);
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgo="
                        }
                    }]
                },
                "finishReason": "STOP"
            }]
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.candidates.len(), 1);
        assert_eq!(resp.candidates[0].finish_reason.as_deref(), Some("STOP"));

        let content = resp.candidates[0].content.as_ref().unwrap();
        let part = &content.parts[0];
        let inline = part.inline_data.as_ref().unwrap();
        assert_eq!(inline.mime_type, "image/png");
    }

    #[test]
    fn test_response_no_image_data() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{}]
                }
            }]
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).unwrap();
        let content = resp.candidates[0].content.as_ref().unwrap();
        let part = &content.parts[0];
        assert!(part.inline_data.is_none());
    }

    #[test]
    fn test_response_with_prompt_feedback_block() {
        let json = r#"{
            "candidates": [],
            "promptFeedback": {
                "blockReason": "SAFETY",
                "blockReasonMessage": "Prompt was blocked due to safety"
            }
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).unwrap();
        assert!(resp.candidates.is_empty());
        let feedback = resp.prompt_feedback.unwrap();
        assert_eq!(feedback.block_reason.as_deref(), Some("SAFETY"));
        assert_eq!(
            feedback.block_reason_message.as_deref(),
            Some("Prompt was blocked due to safety")
        );
    }

    #[test]
    fn test_response_safety_finish_reason() {
        let json = r#"{
            "candidates": [{
                "finishReason": "IMAGE_SAFETY"
            }]
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.candidates[0].finish_reason.as_deref(),
            Some("IMAGE_SAFETY")
        );
        assert!(resp.candidates[0].content.is_none());
    }

    #[test]
    fn test_request_serialization_uses_camel_case() {
        let req = GenerationRequest::new("A puppy").with_seed(1);
        let gemini_req = GeminiRequest::from_generation_request(&req);
        let json = serde_json::to_value(&gemini_req).unwrap();

        // Should use camelCase per serde config
        assert!(json.get("generationConfig").is_some());
        assert!(json.get("generation_config").is_none());
    }
}
