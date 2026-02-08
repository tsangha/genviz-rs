//! Kling AI (Kuaishou) image generation provider.

use crate::error::{sanitize_error_message, GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const BASE_URL: &str = "https://api.klingai.com/v1";

/// Kling image model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KlingImageModel {
    /// Kling V1 - first generation image model.
    KlingV1,
    /// Kling V1.5 - improved image model.
    KlingV1_5,
    /// Kling V2 - latest image model (default).
    #[default]
    KlingV2,
}

impl KlingImageModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::KlingV1 => "kling-v1",
            Self::KlingV1_5 => "kling-v1-5",
            Self::KlingV2 => "kling-v2",
        }
    }
}

/// Builder for KlingImageProvider.
#[derive(Debug, Clone)]
pub struct KlingImageProviderBuilder {
    access_key: Option<String>,
    secret_key: Option<String>,
    model: KlingImageModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for KlingImageProviderBuilder {
    fn default() -> Self {
        Self {
            access_key: None,
            secret_key: None,
            model: KlingImageModel::default(),
            poll_interval: Duration::from_secs(1),
            timeout: Duration::from_secs(120),
        }
    }
}

impl KlingImageProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the access key. Falls back to `KLING_ACCESS_KEY` env var.
    pub fn access_key(mut self, key: impl Into<String>) -> Self {
        self.access_key = Some(key.into());
        self
    }

    /// Sets the secret key. Falls back to `KLING_SECRET_KEY` env var.
    pub fn secret_key(mut self, key: impl Into<String>) -> Self {
        self.secret_key = Some(key.into());
        self
    }

    /// Sets the Kling model variant.
    pub fn model(mut self, model: KlingImageModel) -> Self {
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
    pub fn build(self) -> Result<KlingImageProvider> {
        let access_key = self
            .access_key
            .or_else(|| std::env::var("KLING_ACCESS_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("KLING_ACCESS_KEY not set and no access key provided".into())
            })?;

        let secret_key = self
            .secret_key
            .or_else(|| std::env::var("KLING_SECRET_KEY").ok())
            .ok_or_else(|| {
                GenVizError::Auth("KLING_SECRET_KEY not set and no secret key provided".into())
            })?;

        Ok(KlingImageProvider {
            client: reqwest::Client::new(),
            access_key,
            secret_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// Kling AI image generation provider.
pub struct KlingImageProvider {
    client: reqwest::Client,
    access_key: String,
    secret_key: String,
    model: KlingImageModel,
    poll_interval: Duration,
    timeout: Duration,
}

impl KlingImageProvider {
    /// Creates a new `KlingImageProviderBuilder`.
    pub fn builder() -> KlingImageProviderBuilder {
        KlingImageProviderBuilder::new()
    }

    /// Generates a JWT token for Kling API authentication.
    fn generate_jwt(&self) -> Result<String> {
        use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| GenVizError::Auth(format!("system clock error: {}", e)))?
            .as_secs();

        let claims = KlingJwtClaims {
            iss: self.access_key.clone(),
            exp: now + 1800,
            nbf: now.saturating_sub(5),
        };

        let header = Header::new(Algorithm::HS256);
        let key = EncodingKey::from_secret(self.secret_key.as_bytes());

        encode(&header, &claims, &key)
            .map_err(|e| GenVizError::Auth(format!("JWT generation failed: {}", e)))
    }

    fn parse_error(&self, status: u16, text: &str) -> GenVizError {
        let text = sanitize_error_message(text);

        // Try to extract Kling error code from JSON response
        if let Ok(error_resp) = serde_json::from_str::<KlingErrorResponse>(&text) {
            return self.map_kling_error(error_resp.code, &error_resp.message);
        }

        // Fallback to HTTP status-based error mapping
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

    fn map_kling_error(&self, code: i32, message: &str) -> GenVizError {
        let message = sanitize_error_message(message);
        match code {
            1000..=1004 => GenVizError::Auth(message),
            1100..=1102 => GenVizError::Billing(
                "Insufficient Kling credits. Check billing at klingai.com".into(),
            ),
            1300..=1301 => GenVizError::ContentBlocked(message),
            1302..=1304 => GenVizError::RateLimited { retry_after: None },
            c if c >= 5000 => GenVizError::Api { status: 0, message },
            _ => GenVizError::Api { status: 0, message },
        }
    }

    /// Submit an image generation request.
    async fn submit(&self, request: &GenerationRequest) -> Result<String> {
        let token = self.generate_jwt()?;
        let url = format!("{}/images/generations", BASE_URL);

        let body = KlingImageRequest::from_request(request, &self.model);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text));
        }

        let submit_response: KlingApiResponse<KlingTaskData> = response.json().await?;

        if submit_response.code != 0 {
            return Err(self.map_kling_error(submit_response.code, &submit_response.message));
        }

        submit_response
            .data
            .map(|d| d.task_id)
            .ok_or_else(|| GenVizError::UnexpectedResponse("No task_id in response".into()))
    }

    /// Poll until the image is ready.
    async fn poll_until_ready(&self, task_id: &str) -> Result<KlingImageResult> {
        let url = format!("{}/images/generations/{}", BASE_URL, task_id);
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let token = self.generate_jwt()?;

            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", token))
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text));
            }

            let poll_response: KlingApiResponse<KlingTaskResult> = response.json().await?;

            if poll_response.code != 0 {
                return Err(self.map_kling_error(poll_response.code, &poll_response.message));
            }

            let task = poll_response.data.ok_or_else(|| {
                GenVizError::UnexpectedResponse("No data in poll response".into())
            })?;

            match task.task_status.as_str() {
                "succeed" => {
                    let image = task
                        .task_result
                        .and_then(|r| r.images)
                        .and_then(|imgs| imgs.into_iter().next())
                        .ok_or_else(|| {
                            GenVizError::UnexpectedResponse(
                                "Kling returned succeed but no image data".into(),
                            )
                        })?;
                    return Ok(image);
                }
                "submitted" | "processing" => {
                    tracing::debug!(
                        task_id = %task_id,
                        status = %task.task_status,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling Kling image generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                }
                "failed" => {
                    let message = task
                        .task_status_msg
                        .unwrap_or_else(|| "Generation failed".into());
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "Kling generation failed: {}",
                        message
                    )));
                }
                other => {
                    return Err(GenVizError::UnexpectedResponse(format!(
                        "Kling returned unexpected status: {}",
                        other
                    )));
                }
            }
        }
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
impl ImageProvider for KlingImageProvider {
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();

        let task_id = self.submit(request).await?;
        tracing::debug!(task_id = %task_id, "submitted Kling image generation request");

        let result = self.poll_until_ready(&task_id).await?;
        tracing::debug!(url = %result.url, "Kling image generation complete");

        let data = self.download(&result.url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let format = ImageFormat::from_magic_bytes(&data).unwrap_or(ImageFormat::Png);

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Kling,
            GenerationMetadata {
                model: Some(self.model.as_str().to_string()),
                seed: None,
                duration_ms: Some(duration_ms),
                safety_filtered: false,
            },
        ))
    }

    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::Kling
    }

    async fn health_check(&self) -> Result<()> {
        // Validate JWT generation works (proves credentials are present)
        self.generate_jwt()?;
        Ok(())
    }
}

// JWT claims
#[derive(Debug, Serialize)]
struct KlingJwtClaims {
    iss: String,
    exp: u64,
    nbf: u64,
}

// Request types
#[derive(Debug, Serialize)]
struct KlingImageRequest {
    model_name: String,
    prompt: String,
    n: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
}

impl KlingImageRequest {
    fn from_request(req: &GenerationRequest, model: &KlingImageModel) -> Self {
        use base64::Engine;

        Self {
            model_name: model.as_str().to_string(),
            prompt: req.prompt.clone(),
            n: 1,
            aspect_ratio: req.aspect_ratio.map(|ar| ar.as_str().to_string()),
            // Kling expects raw base64 without data URI prefix
            image: req
                .input_image
                .as_ref()
                .map(|img| base64::engine::general_purpose::STANDARD.encode(img)),
        }
    }
}

// Response types
#[derive(Debug, Deserialize)]
struct KlingApiResponse<T> {
    code: i32,
    message: String,
    data: Option<T>,
}

#[derive(Debug, Deserialize)]
struct KlingTaskData {
    task_id: String,
}

#[derive(Debug, Deserialize)]
struct KlingTaskResult {
    task_status: String,
    #[serde(default)]
    task_status_msg: Option<String>,
    #[serde(default)]
    task_result: Option<KlingImageResultContainer>,
}

#[derive(Debug, Deserialize)]
struct KlingImageResultContainer {
    #[serde(default)]
    images: Option<Vec<KlingImageResult>>,
}

#[derive(Debug, Deserialize)]
struct KlingImageResult {
    url: String,
}

#[derive(Debug, Deserialize)]
struct KlingErrorResponse {
    code: i32,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kling_image_model_as_str() {
        assert_eq!(KlingImageModel::KlingV1.as_str(), "kling-v1");
        assert_eq!(KlingImageModel::KlingV1_5.as_str(), "kling-v1-5");
        assert_eq!(KlingImageModel::KlingV2.as_str(), "kling-v2");
    }

    #[test]
    fn test_kling_image_model_default() {
        assert_eq!(KlingImageModel::default(), KlingImageModel::KlingV2);
    }

    #[test]
    fn test_builder_with_explicit_keys() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .model(KlingImageModel::KlingV1_5)
            .build();
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.model, KlingImageModel::KlingV1_5);
    }

    #[test]
    fn test_builder_missing_access_key() {
        // Ensure env vars are not set for this test
        std::env::remove_var("KLING_ACCESS_KEY");
        std::env::remove_var("KLING_SECRET_KEY");

        let result = KlingImageProviderBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_secret_key() {
        let result = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .build();
        // Will fail because secret_key is also missing from env
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .poll_interval(Duration::from_secs(2))
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(2));
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_jwt_generation() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();

        let token = provider.generate_jwt();
        assert!(token.is_ok());
        let token = token.unwrap();
        // JWT has 3 parts separated by dots
        assert_eq!(token.split('.').count(), 3);
    }

    #[test]
    fn test_request_construction_basic() {
        let req = GenerationRequest::new("A sunset");
        let kling_req = KlingImageRequest::from_request(&req, &KlingImageModel::KlingV2);

        assert_eq!(kling_req.prompt, "A sunset");
        assert_eq!(kling_req.model_name, "kling-v2");
        assert_eq!(kling_req.n, 1);
        assert!(kling_req.aspect_ratio.is_none());
        assert!(kling_req.image.is_none());
    }

    #[test]
    fn test_request_construction_with_aspect_ratio() {
        let req = GenerationRequest::new("A sunset")
            .with_aspect_ratio(crate::image::AspectRatio::ThreeTwo);
        let kling_req = KlingImageRequest::from_request(&req, &KlingImageModel::KlingV2);

        assert_eq!(kling_req.aspect_ratio.as_deref(), Some("3:2"));
    }

    #[test]
    fn test_request_with_input_image() {
        let req = GenerationRequest::new("Edit this").with_input_image(vec![0x89, 0x50, 0x4E]);
        let kling_req = KlingImageRequest::from_request(&req, &KlingImageModel::KlingV2);

        assert!(kling_req.image.is_some());
        // Should be raw base64 (no data URI prefix)
        let b64 = kling_req.image.unwrap();
        assert!(!b64.starts_with("data:"));
    }

    #[test]
    fn test_request_serialization_skips_none() {
        let req = GenerationRequest::new("A sunset");
        let kling_req = KlingImageRequest::from_request(&req, &KlingImageModel::KlingV2);
        let json = serde_json::to_value(&kling_req).unwrap();

        assert!(json.get("prompt").is_some());
        assert!(json.get("model_name").is_some());
        assert!(json.get("n").is_some());
        assert!(json.get("aspect_ratio").is_none());
        assert!(json.get("image").is_none());
    }

    #[test]
    fn test_task_data_deserialization() {
        let json = r#"{"code": 0, "message": "OK", "data": {"task_id": "abc-123"}}"#;
        let resp: KlingApiResponse<KlingTaskData> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.code, 0);
        assert_eq!(resp.data.unwrap().task_id, "abc-123");
    }

    #[test]
    fn test_task_result_succeed() {
        let json = r#"{
            "code": 0,
            "message": "OK",
            "data": {
                "task_status": "succeed",
                "task_result": {
                    "images": [{"url": "https://example.com/image.png"}]
                }
            }
        }"#;
        let resp: KlingApiResponse<KlingTaskResult> = serde_json::from_str(json).unwrap();
        let data = resp.data.unwrap();
        assert_eq!(data.task_status, "succeed");
        let images = data.task_result.unwrap().images.unwrap();
        assert_eq!(images[0].url, "https://example.com/image.png");
    }

    #[test]
    fn test_task_result_processing() {
        let json = r#"{
            "code": 0,
            "message": "OK",
            "data": {
                "task_status": "processing"
            }
        }"#;
        let resp: KlingApiResponse<KlingTaskResult> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.unwrap().task_status, "processing");
    }

    #[test]
    fn test_map_kling_error_auth() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1000, "Invalid API key");
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_map_kling_error_billing() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1100, "Insufficient credits");
        assert!(matches!(err, GenVizError::Billing(_)));
    }

    #[test]
    fn test_map_kling_error_content_blocked() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1300, "Content blocked");
        assert!(matches!(err, GenVizError::ContentBlocked(_)));
    }

    #[test]
    fn test_map_kling_error_rate_limited() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1302, "Rate limited");
        assert!(matches!(err, GenVizError::RateLimited { .. }));
    }

    #[test]
    fn test_map_kling_error_api() {
        let provider = KlingImageProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(5000, "Internal error");
        assert!(matches!(err, GenVizError::Api { .. }));
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"code": 1000, "message": "Invalid API key"}"#;
        let resp: KlingErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.code, 1000);
        assert_eq!(resp.message, "Invalid API key");
    }
}
