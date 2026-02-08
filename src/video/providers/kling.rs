//! Kling AI (Kuaishou) video generation provider (image-to-video).

use crate::error::{sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const BASE_URL: &str = "https://api.klingai.com/v1";

/// Kling video model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KlingVideoModel {
    /// Kling V1 - first generation video model.
    KlingV1,
    /// Kling V1.5 - improved video model.
    KlingV1_5,
    /// Kling V1.6 - latest video model (default).
    #[default]
    KlingV1_6,
}

impl KlingVideoModel {
    /// Returns the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::KlingV1 => "kling-v1",
            Self::KlingV1_5 => "kling-v1-5",
            Self::KlingV1_6 => "kling-v1-6",
        }
    }
}

/// Kling video generation mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KlingVideoMode {
    /// Standard mode - faster, lower cost.
    #[default]
    Std,
    /// Professional mode - higher quality, slower.
    Pro,
}

impl KlingVideoMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Std => "std",
            Self::Pro => "pro",
        }
    }
}

/// Builder for KlingVideoProvider.
#[derive(Debug, Clone)]
pub struct KlingVideoProviderBuilder {
    access_key: Option<String>,
    secret_key: Option<String>,
    model: KlingVideoModel,
    mode: KlingVideoMode,
    poll_interval: Duration,
    timeout: Duration,
}

impl Default for KlingVideoProviderBuilder {
    fn default() -> Self {
        Self {
            access_key: None,
            secret_key: None,
            model: KlingVideoModel::default(),
            mode: KlingVideoMode::default(),
            poll_interval: Duration::from_secs(3),
            timeout: Duration::from_secs(300), // 5 minutes for video
        }
    }
}

impl KlingVideoProviderBuilder {
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

    /// Sets the Kling video model variant.
    pub fn model(mut self, model: KlingVideoModel) -> Self {
        self.model = model;
        self
    }

    /// Sets the generation mode (std or pro).
    pub fn mode(mut self, mode: KlingVideoMode) -> Self {
        self.mode = mode;
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
    pub fn build(self) -> Result<KlingVideoProvider> {
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

        Ok(KlingVideoProvider {
            client: reqwest::Client::new(),
            access_key,
            secret_key,
            model: self.model,
            mode: self.mode,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
        })
    }
}

/// Kling AI video generation provider (image-to-video).
pub struct KlingVideoProvider {
    client: reqwest::Client,
    access_key: String,
    secret_key: String,
    model: KlingVideoModel,
    mode: KlingVideoMode,
    poll_interval: Duration,
    timeout: Duration,
}

impl KlingVideoProvider {
    /// Creates a new `KlingVideoProviderBuilder`.
    pub fn builder() -> KlingVideoProviderBuilder {
        KlingVideoProviderBuilder::new()
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

        if let Ok(error_resp) = serde_json::from_str::<KlingErrorResponse>(&text) {
            return self.map_kling_error(error_resp.code, &error_resp.message);
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

    /// Submit a video generation request.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        // Kling video is image-to-video only
        let image_url = request.source_image_url.as_ref().ok_or_else(|| {
            GenVizError::InvalidRequest(
                "Kling video requires a source image (image-to-video only). Provide source_image_url."
                    .into(),
            )
        })?;

        let token = self.generate_jwt()?;
        let url = format!("{}/videos/image2video", BASE_URL);

        let body = KlingVideoRequest {
            model_name: self.model.as_str().to_string(),
            image: image_url.clone(),
            prompt: request.prompt.clone(),
            mode: self.mode.as_str().to_string(),
            duration: request.duration_secs.map(|d| d.to_string()),
        };

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

    /// Poll until the video is ready.
    async fn poll_until_ready(&self, task_id: &str) -> Result<String> {
        let url = format!("{}/videos/image2video/{}", BASE_URL, task_id);
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

            let poll_response: KlingApiResponse<KlingVideoTaskResult> = response.json().await?;

            if poll_response.code != 0 {
                return Err(self.map_kling_error(poll_response.code, &poll_response.message));
            }

            let task = poll_response.data.ok_or_else(|| {
                GenVizError::UnexpectedResponse("No data in poll response".into())
            })?;

            match task.task_status.as_str() {
                "succeed" => {
                    let video_url = task
                        .task_result
                        .and_then(|r| r.videos)
                        .and_then(|vids| vids.into_iter().next())
                        .map(|v| v.url)
                        .ok_or_else(|| {
                            GenVizError::UnexpectedResponse(
                                "Kling returned succeed but no video data".into(),
                            )
                        })?;
                    return Ok(video_url);
                }
                "submitted" | "processing" => {
                    tracing::debug!(
                        task_id = %task_id,
                        status = %task.task_status,
                        elapsed_secs = start.elapsed().as_secs(),
                        "polling Kling video generation"
                    );
                    tokio::time::sleep(self.poll_interval).await;
                }
                "failed" => {
                    let message = task
                        .task_status_msg
                        .unwrap_or_else(|| "Video generation failed".into());
                    return Err(GenVizError::VideoGeneration(format!(
                        "Kling video failed: {}",
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

    /// Download the video from the given URL.
    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(GenVizError::Api {
                status: response.status().as_u16(),
                message: "Failed to download video".into(),
            });
        }

        Ok(response.bytes().await?.to_vec())
    }
}

#[async_trait]
impl VideoProvider for KlingVideoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();

        let task_id = self.submit(request).await?;
        tracing::debug!(task_id = %task_id, "submitted Kling video generation request");

        let video_url = self.poll_until_ready(&task_id).await?;
        tracing::debug!(url = %video_url, "Kling video generation complete");

        let data = self.download(&video_url).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::Kling,
            VideoMetadata {
                model: Some(self.model.as_str().to_string()),
                duration_ms: Some(duration_ms),
                video_duration_secs: request.duration_secs,
                resolution: None,
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::Kling
    }

    async fn health_check(&self) -> Result<()> {
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
struct KlingVideoRequest {
    model_name: String,
    image: String,
    prompt: String,
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<String>,
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
struct KlingVideoTaskResult {
    task_status: String,
    #[serde(default)]
    task_status_msg: Option<String>,
    #[serde(default)]
    task_result: Option<KlingVideoResultContainer>,
}

#[derive(Debug, Deserialize)]
struct KlingVideoResultContainer {
    #[serde(default)]
    videos: Option<Vec<KlingVideoResult>>,
}

#[derive(Debug, Deserialize)]
struct KlingVideoResult {
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
    fn test_kling_video_model_as_str() {
        assert_eq!(KlingVideoModel::KlingV1.as_str(), "kling-v1");
        assert_eq!(KlingVideoModel::KlingV1_5.as_str(), "kling-v1-5");
        assert_eq!(KlingVideoModel::KlingV1_6.as_str(), "kling-v1-6");
    }

    #[test]
    fn test_kling_video_model_default() {
        assert_eq!(KlingVideoModel::default(), KlingVideoModel::KlingV1_6);
    }

    #[test]
    fn test_kling_video_mode_as_str() {
        assert_eq!(KlingVideoMode::Std.as_str(), "std");
        assert_eq!(KlingVideoMode::Pro.as_str(), "pro");
    }

    #[test]
    fn test_builder_with_explicit_keys() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .model(KlingVideoModel::KlingV1_5)
            .mode(KlingVideoMode::Pro)
            .build();
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.model, KlingVideoModel::KlingV1_5);
        assert_eq!(provider.mode, KlingVideoMode::Pro);
    }

    #[test]
    fn test_builder_missing_keys() {
        std::env::remove_var("KLING_ACCESS_KEY");
        std::env::remove_var("KLING_SECRET_KEY");

        let result = KlingVideoProviderBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .poll_interval(Duration::from_secs(5))
            .timeout(Duration::from_secs(600))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(5));
        assert_eq!(provider.timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_jwt_generation() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();

        let token = provider.generate_jwt();
        assert!(token.is_ok());
        let token = token.unwrap();
        assert_eq!(token.split('.').count(), 3);
    }

    #[test]
    fn test_request_serialization() {
        let req = KlingVideoRequest {
            model_name: "kling-v1-6".to_string(),
            image: "https://example.com/photo.jpg".to_string(),
            prompt: "Animate this".to_string(),
            mode: "std".to_string(),
            duration: Some("5".to_string()),
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["model_name"], "kling-v1-6");
        assert_eq!(json["image"], "https://example.com/photo.jpg");
        assert_eq!(json["prompt"], "Animate this");
        assert_eq!(json["mode"], "std");
        assert_eq!(json["duration"], "5");
    }

    #[test]
    fn test_request_serialization_no_duration() {
        let req = KlingVideoRequest {
            model_name: "kling-v1-6".to_string(),
            image: "https://example.com/photo.jpg".to_string(),
            prompt: "Animate this".to_string(),
            mode: "std".to_string(),
            duration: None,
        };
        let json = serde_json::to_value(&req).unwrap();

        assert!(json.get("duration").is_none());
    }

    #[test]
    fn test_task_data_deserialization() {
        let json = r#"{"code": 0, "message": "OK", "data": {"task_id": "vid-123"}}"#;
        let resp: KlingApiResponse<KlingTaskData> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.code, 0);
        assert_eq!(resp.data.unwrap().task_id, "vid-123");
    }

    #[test]
    fn test_video_task_result_succeed() {
        let json = r#"{
            "code": 0,
            "message": "OK",
            "data": {
                "task_status": "succeed",
                "task_result": {
                    "videos": [{"url": "https://example.com/video.mp4"}]
                }
            }
        }"#;
        let resp: KlingApiResponse<KlingVideoTaskResult> = serde_json::from_str(json).unwrap();
        let data = resp.data.unwrap();
        assert_eq!(data.task_status, "succeed");
        let videos = data.task_result.unwrap().videos.unwrap();
        assert_eq!(videos[0].url, "https://example.com/video.mp4");
    }

    #[test]
    fn test_video_task_result_processing() {
        let json = r#"{
            "code": 0,
            "message": "OK",
            "data": {
                "task_status": "processing"
            }
        }"#;
        let resp: KlingApiResponse<KlingVideoTaskResult> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.unwrap().task_status, "processing");
    }

    #[test]
    fn test_map_kling_error_auth() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1000, "Invalid API key");
        assert!(matches!(err, GenVizError::Auth(_)));
    }

    #[test]
    fn test_map_kling_error_billing() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1100, "Insufficient credits");
        assert!(matches!(err, GenVizError::Billing(_)));
    }

    #[test]
    fn test_map_kling_error_content_blocked() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1300, "Content blocked");
        assert!(matches!(err, GenVizError::ContentBlocked(_)));
    }

    #[test]
    fn test_map_kling_error_rate_limited() {
        let provider = KlingVideoProviderBuilder::new()
            .access_key("test-ak")
            .secret_key("test-sk")
            .build()
            .unwrap();
        let err = provider.map_kling_error(1302, "Rate limited");
        assert!(matches!(err, GenVizError::RateLimited { .. }));
    }
}
