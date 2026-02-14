//! Veo (Google) video generation provider.

use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
use crate::video::provider::VideoProvider;
use crate::video::types::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Which Google API backend to use for Veo video generation.
#[derive(Debug, Clone, Default)]
pub enum VeoBackend {
    /// Gemini Developer API (generativelanguage.googleapis.com).
    /// Uses API key auth. Has daily rate limits (10 RPD).
    #[default]
    Gemini,
    /// Vertex AI (aiplatform.googleapis.com).
    /// Uses gcloud CLI for auth. No daily rate limits, pay-per-second.
    Vertex {
        /// GCP project ID.
        project: String,
        /// GCP location (e.g. "us-central1").
        location: String,
    },
}

/// Veo model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VeoModel {
    /// Veo 3.1 Preview - Google's video generation model.
    #[default]
    Veo31Preview,
}

impl VeoModel {
    /// Returns the Gemini Developer API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Veo31Preview => "veo-3.1-generate-preview",
        }
    }

    /// Returns the Vertex AI model identifier string.
    pub fn vertex_id(&self) -> &'static str {
        match self {
            Self::Veo31Preview => "veo-3.1-generate-001",
        }
    }
}

/// Builder for VeoProvider.
#[derive(Debug, Clone)]
pub struct VeoProviderBuilder {
    api_key: Option<String>,
    model: VeoModel,
    poll_interval: Duration,
    timeout: Duration,
    backend: Option<VeoBackend>,
    project: Option<String>,
    location: Option<String>,
}

impl Default for VeoProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            model: VeoModel::default(),
            poll_interval: Duration::from_secs(10),
            timeout: Duration::from_secs(600), // 10 minutes for video
            backend: None,
            project: None,
            location: None,
        }
    }
}

impl VeoProviderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key. Falls back to `GOOGLE_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the Veo model variant.
    pub fn model(mut self, model: VeoModel) -> Self {
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

    /// Explicitly sets the backend (Gemini or Vertex AI).
    pub fn backend(mut self, backend: VeoBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Sets the GCP project ID (implies Vertex AI backend).
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Sets the GCP location (implies Vertex AI backend, defaults to "us-central1").
    pub fn location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Builds the provider, resolving auth and backend.
    ///
    /// Backend auto-detection: if `VERTEX_AI_PROJECT` env var is set (and no
    /// explicit backend), defaults to Vertex AI. Otherwise uses Gemini Developer API.
    pub fn build(self) -> Result<VeoProvider> {
        // Resolve backend
        let backend = if let Some(b) = self.backend {
            b
        } else if let Some(project) = self.project.clone() {
            let location = self
                .location
                .clone()
                .unwrap_or_else(|| "us-central1".to_string());
            VeoBackend::Vertex { project, location }
        } else if let Ok(project) = std::env::var("VERTEX_AI_PROJECT") {
            let location =
                std::env::var("VERTEX_AI_LOCATION").unwrap_or_else(|_| "us-central1".to_string());
            VeoBackend::Vertex { project, location }
        } else {
            VeoBackend::Gemini
        };

        // Resolve API key — required for Gemini, optional for Vertex
        let api_key = match &backend {
            VeoBackend::Gemini => {
                let key = self
                    .api_key
                    .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
                    .ok_or_else(|| {
                        GenVizError::Auth(
                            "GOOGLE_API_KEY not set and no API key provided. \
                             Set GOOGLE_API_KEY for Gemini API, or VERTEX_AI_PROJECT for Vertex AI."
                                .into(),
                        )
                    })?;
                Some(key)
            }
            VeoBackend::Vertex { .. } => {
                // API key is optional for Vertex — auth uses gcloud CLI
                self.api_key
                    .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            }
        };

        Ok(VeoProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            poll_interval: self.poll_interval,
            timeout: self.timeout,
            backend,
        })
    }
}

/// Veo video generation provider.
pub struct VeoProvider {
    client: reqwest::Client,
    /// API key — Some for Gemini (required), optional for Vertex.
    api_key: Option<String>,
    model: VeoModel,
    poll_interval: Duration,
    timeout: Duration,
    backend: VeoBackend,
}

/// Get a bearer token by running `gcloud auth print-access-token`.
fn gcloud_access_token() -> Result<String> {
    let output = std::process::Command::new("gcloud")
        .args(["auth", "print-access-token"])
        .output()
        .map_err(|e| {
            GenVizError::Auth(format!(
                "Failed to run gcloud CLI: {}. Install it from https://cloud.google.com/sdk/docs/install",
                e
            ))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(GenVizError::Auth(format!("gcloud auth failed: {}", stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

impl VeoProvider {
    /// Creates a new `VeoProviderBuilder`.
    pub fn builder() -> VeoProviderBuilder {
        VeoProviderBuilder::new()
    }

    /// Returns a reference to the active backend.
    pub fn backend(&self) -> &VeoBackend {
        &self.backend
    }

    /// Submit a video generation request.
    async fn submit(&self, request: &VideoGenerationRequest) -> Result<String> {
        let data = VeoRequestData::from_request(request);

        match &self.backend {
            VeoBackend::Gemini => self.submit_gemini(&data).await,
            VeoBackend::Vertex { project, location } => {
                self.submit_vertex(&data, project, location).await
            }
        }
    }

    async fn submit_gemini(&self, data: &VeoRequestData) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:predictLongRunning",
            self.model.as_str(),
        );

        let body = VeoRequest::from_data(data);
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            GenVizError::Auth("GOOGLE_API_KEY required for Gemini backend".into())
        })?;

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", api_key)
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

        let operation: VeoOperationResponse = response.json().await?;
        Ok(operation.name)
    }

    async fn submit_vertex(
        &self,
        data: &VeoRequestData,
        project: &str,
        location: &str,
    ) -> Result<String> {
        let model_id = self.model.vertex_id();
        let url = format!(
            "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_id}:predictLongRunning",
        );

        let body = VertexRequest::from_data(data);
        let token = gcloud_access_token()?;

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
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let operation: VeoOperationResponse = response.json().await?;
        Ok(operation.name)
    }

    async fn poll_gemini(&self, operation_name: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}",
            operation_name,
        );
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            GenVizError::Auth("GOOGLE_API_KEY required for Gemini backend".into())
        })?;
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let response = self
                .client
                .get(&url)
                .header("x-goog-api-key", api_key)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text, &headers));
            }

            let operation: VeoOperationResponse = response.json().await?;

            if operation.done.unwrap_or(false) {
                return self.extract_gemini_video_url(operation);
            }

            if let Some(err) = operation.error {
                return Err(GenVizError::VideoGeneration(
                    err.message.unwrap_or_else(|| "Unknown error".into()),
                ));
            }

            tracing::debug!(
                operation = %operation_name,
                elapsed_secs = start.elapsed().as_secs(),
                "polling Veo video generation (Gemini)"
            );
            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Poll Vertex AI and return the video bytes directly.
    ///
    /// Vertex AI can return video data in three ways:
    /// 1. Inline base64 (`bytesBase64Encoded`) — the default
    /// 2. GCS URI (`gcsUri`) — when `storage_uri` is set
    /// 3. HTTPS URL (`uri`) — alternative download URL
    async fn poll_and_download_vertex(
        &self,
        operation_name: &str,
        project: &str,
        location: &str,
    ) -> Result<Vec<u8>> {
        let model_id = self.model.vertex_id();
        let url = format!(
            "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_id}:fetchPredictOperation",
        );
        let start = Instant::now();

        loop {
            if start.elapsed() > self.timeout {
                return Err(GenVizError::Timeout(self.timeout));
            }

            let token = gcloud_access_token()?;
            let body = VertexFetchOperationRequest {
                operation_name: operation_name.to_string(),
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
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(self.parse_error(status.as_u16(), &text, &headers));
            }

            let operation: VeoOperationResponse = response.json().await?;

            if operation.done.unwrap_or(false) {
                // Check for safety filtering
                if let Some(ref resp) = operation.response {
                    if let Some(ref gen_resp) = resp.generate_video_response {
                        if gen_resp.rai_media_filtered_count.unwrap_or(0) > 0 {
                            let has_results = resp.videos.as_ref().is_some_and(|v| !v.is_empty())
                                || gen_resp
                                    .generated_samples
                                    .as_ref()
                                    .is_some_and(|s| !s.is_empty());
                            if !has_results {
                                return Err(GenVizError::ContentBlocked(
                                    "Video was filtered by Veo safety filters".into(),
                                ));
                            }
                        }
                    }
                }

                // Extract video data from the response
                if let Some(resp) = operation.response {
                    if let Some(videos) = resp.videos {
                        if let Some(first) = videos.into_iter().next() {
                            // Inline base64 data — decode directly (most common)
                            if let Some(b64) = first.bytes_base64_encoded {
                                use base64::Engine;
                                return base64::engine::general_purpose::STANDARD
                                    .decode(&b64)
                                    .map_err(|e| {
                                        GenVizError::UnexpectedResponse(format!(
                                            "Failed to decode inline video data: {}",
                                            e
                                        ))
                                    });
                            }
                            // GCS URI or HTTPS URL — download with bearer token
                            if let Some(download_url) =
                                first.gcs_uri.as_ref().or(first.uri.as_ref())
                            {
                                return self.download_vertex(download_url).await;
                            }
                        }
                    }
                }
                return Err(GenVizError::UnexpectedResponse(
                    "Video generation completed but no video data returned".into(),
                ));
            }

            if let Some(err) = operation.error {
                return Err(GenVizError::VideoGeneration(
                    err.message.unwrap_or_else(|| "Unknown error".into()),
                ));
            }

            tracing::debug!(
                operation = %operation_name,
                elapsed_secs = start.elapsed().as_secs(),
                "polling Veo video generation (Vertex AI)"
            );
            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Extract video URL from a completed Gemini-format operation response.
    fn extract_gemini_video_url(&self, operation: VeoOperationResponse) -> Result<String> {
        // Check for error FIRST before checking response
        if let Some(err) = operation.error {
            return Err(GenVizError::VideoGeneration(
                err.message.unwrap_or_else(|| "Unknown error".into()),
            ));
        }

        if let Some(resp) = operation.response {
            if let Some(gen_resp) = resp.generate_video_response {
                // Check if content was filtered
                if gen_resp.rai_media_filtered_count.unwrap_or(0) > 0
                    && gen_resp
                        .generated_samples
                        .as_ref()
                        .is_none_or(|s| s.is_empty())
                {
                    return Err(GenVizError::ContentBlocked(
                        "Video was filtered by Veo safety filters".into(),
                    ));
                }

                if let Some(samples) = gen_resp.generated_samples {
                    if let Some(first) = samples.into_iter().next() {
                        if let Some(uri) = first.video.and_then(|v| v.uri) {
                            return Ok(uri);
                        }
                    }
                }
            }
        }
        Err(GenVizError::UnexpectedResponse(
            "Video generation completed but no video URL returned".into(),
        ))
    }

    async fn download_gemini(&self, url: &str) -> Result<Vec<u8>> {
        if url.starts_with("gs://") {
            return Err(GenVizError::VideoGeneration(format!(
                "Veo returned a Google Cloud Storage URI ({}) which cannot be downloaded directly. \
                 Use `gsutil cp` or the Google Cloud Storage API to download the video.",
                url
            )));
        }

        let api_key = self.api_key.as_ref().ok_or_else(|| {
            GenVizError::Auth("GOOGLE_API_KEY required for Gemini backend".into())
        })?;

        // Append API key as query parameter (known SDK requirement)
        let url = if url.contains('?') {
            format!("{}&key={}", url, api_key)
        } else {
            format!("{}?key={}", url, api_key)
        };

        let response = self
            .client
            .get(&url)
            .header("x-goog-api-key", api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(GenVizError::Api {
                status: response.status().as_u16(),
                message: "Failed to download video".into(),
            });
        }

        Ok(response.bytes().await?.to_vec())
    }

    async fn download_vertex(&self, url: &str) -> Result<Vec<u8>> {
        let token = gcloud_access_token()?;

        // Convert gs:// URI to HTTPS URL
        let https_url = if let Some(path) = url.strip_prefix("gs://") {
            format!("https://storage.googleapis.com/{}", path)
        } else {
            url.to_string()
        };

        let response = self
            .client
            .get(&https_url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(GenVizError::Api {
                status: response.status().as_u16(),
                message: format!("Failed to download video from {}", https_url),
            });
        }

        Ok(response.bytes().await?.to_vec())
    }

    fn parse_error(
        &self,
        status: u16,
        text: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> GenVizError {
        if status == 402 {
            return GenVizError::Billing(
                "Veo billing issue: enable billing at https://aistudio.google.com".into(),
            );
        }
        if status == 404 {
            return GenVizError::InvalidRequest(
                "Veo API not available. Veo requires a paid-tier API key with billing enabled. \
                 Enable it at https://aistudio.google.com by selecting a Google Cloud project with billing."
                    .to_string(),
            );
        }
        let text = sanitize_error_message(text);
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
impl VideoProvider for VeoProvider {
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo> {
        let start = Instant::now();

        // Submit the request
        let operation_name = self.submit(request).await?;
        tracing::debug!(operation = %operation_name, "submitted video generation request");

        // Poll and retrieve video data — Vertex may return inline base64
        let data = match &self.backend {
            VeoBackend::Gemini => {
                let video_url = self.poll_gemini(&operation_name).await?;
                tracing::debug!(url = %video_url, "video generation complete");
                self.download_gemini(&video_url).await?
            }
            VeoBackend::Vertex { project, location } => {
                let project = project.clone();
                let location = location.clone();
                self.poll_and_download_vertex(&operation_name, &project, &location)
                    .await?
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        let model_name = match &self.backend {
            VeoBackend::Gemini => self.model.as_str(),
            VeoBackend::Vertex { .. } => self.model.vertex_id(),
        };

        Ok(GeneratedVideo::new(
            data,
            "video/mp4",
            VideoProviderKind::Veo,
            VideoMetadata {
                model: Some(model_name.to_string()),
                duration_ms: Some(duration_ms),
                video_duration_secs: None,
                resolution: request.resolution.clone(),
            },
        ))
    }

    fn kind(&self) -> VideoProviderKind {
        VideoProviderKind::Veo
    }

    async fn health_check(&self) -> Result<()> {
        match &self.backend {
            VeoBackend::Gemini => {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{}",
                    self.model.as_str(),
                );
                let api_key = self.api_key.as_ref().ok_or_else(|| {
                    GenVizError::Auth("GOOGLE_API_KEY required for Gemini backend".into())
                })?;

                let response = self
                    .client
                    .get(&url)
                    .header("x-goog-api-key", api_key)
                    .send()
                    .await?;

                match response.status().as_u16() {
                    401 | 403 => Err(GenVizError::Auth("Invalid API key".into())),
                    _ => Ok(()),
                }
            }
            VeoBackend::Vertex { project, location } => {
                // Verify gcloud auth works and project is accessible
                let token = gcloud_access_token()?;
                let model_id = self.model.vertex_id();
                let url = format!(
                    "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_id}",
                );

                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .send()
                    .await?;

                match response.status().as_u16() {
                    401 | 403 => Err(GenVizError::Auth(
                        "Vertex AI auth failed. Run `gcloud auth login` and verify project access."
                            .into(),
                    )),
                    _ => Ok(()),
                }
            }
        }
    }
}

// ── Shared intermediate representation ──────────────────────────────────────

/// Backend-agnostic intermediate representation of a Veo request.
/// Contains the shared business logic (duration forcing, extension constraints, etc.)
struct VeoRequestData {
    prompt: String,
    first_frame: Option<(String, String)>, // (base64, mime)
    last_frame: Option<(String, String)>,
    video: Option<(String, String)>,
    reference_images: Vec<(String, String)>,
    aspect_ratio: Option<String>,
    resolution: Option<String>,
    duration_seconds: Option<u32>,
    negative_prompt: Option<String>,
    person_generation: Option<String>,
    num_videos: Option<u32>,
    storage_uri: Option<String>,
    enhance_prompt: Option<bool>,
    generate_audio: Option<bool>,
}

impl VeoRequestData {
    /// Extract shared business logic from a VideoGenerationRequest.
    fn from_request(req: &VideoGenerationRequest) -> Self {
        let detect = |b64: &str, fallback: &str| -> (String, String) {
            let mime = detect_mime_from_base64(b64, fallback);
            (b64.to_string(), mime)
        };

        let first_frame = req.image.as_ref().map(|b64| detect(b64, "image/png"));
        let last_frame = req.last_frame.as_ref().map(|b64| detect(b64, "image/png"));
        let video = req.video.as_ref().map(|b64| detect(b64, "video/mp4"));

        let reference_images: Vec<(String, String)> = req
            .reference_images
            .as_ref()
            .map(|imgs| imgs.iter().map(|b64| detect(b64, "image/png")).collect())
            .unwrap_or_default();

        let has_frames = first_frame.is_some() || last_frame.is_some();
        let is_extension = video.is_some();
        let has_advanced = is_extension || !reference_images.is_empty();

        // Force duration to 8s when using frames, extensions, or reference images
        let duration_seconds = if has_frames || has_advanced {
            Some(8)
        } else {
            req.duration_secs
        };

        // Video extension requires 720p resolution (API constraint)
        let resolution = if is_extension {
            Some("720p".to_string())
        } else {
            req.resolution.clone()
        };

        // Video extension requires numberOfVideos/sampleCount: 1
        let num_videos = if is_extension {
            Some(1)
        } else {
            req.number_of_videos
        };

        Self {
            prompt: req.prompt.clone(),
            first_frame,
            last_frame,
            video,
            reference_images,
            aspect_ratio: req.aspect_ratio.clone(),
            resolution,
            duration_seconds,
            negative_prompt: req.negative_prompt.clone(),
            person_generation: req.person_generation.clone(),
            num_videos,
            storage_uri: req.storage_uri.clone(),
            enhance_prompt: req.enhance_prompt,
            generate_audio: req.generate_audio,
        }
    }
}

// ── Gemini Developer API wire format ────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoRequest {
    instances: Vec<VeoInstance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<VeoParameters>,
}

/// Inline data wrapper for the Gemini API (`{"inlineData": {"mimeType": "...", "data": "..."}}`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoInlineData {
    mime_type: String,
    data: String,
}

/// Media payload wrapping `inlineData` — used for images and videos in Veo requests.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoMediaData {
    inline_data: VeoInlineData,
}

impl VeoMediaData {
    /// Build from base64 data and a pre-resolved MIME type.
    fn new(b64: &str, mime: &str) -> Self {
        Self {
            inline_data: VeoInlineData {
                mime_type: mime.to_string(),
                data: b64.to_string(),
            },
        }
    }
}

/// A reference image entry for `parameters.referenceImages[]`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoReferenceImage {
    image: VeoMediaData,
    reference_type: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoInstance {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<VeoMediaData>,
    /// Video input for video extension (continue an existing video).
    #[serde(skip_serializing_if = "Option::is_none")]
    video: Option<VeoMediaData>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_seconds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_frame: Option<VeoMediaData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_images: Option<Vec<VeoReferenceImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    person_generation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    number_of_videos: Option<u32>,
}

/// Detect MIME type from base64-encoded data using magic byte detection.
/// Falls back to the provided default if detection fails.
fn detect_mime_from_base64(base64_data: &str, fallback: &str) -> String {
    use crate::image::ImageFormat;
    use base64::Engine;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(base64_data))
        .unwrap_or_default();

    ImageFormat::from_magic_bytes(&bytes)
        .map(|f| f.mime_type().to_string())
        .unwrap_or_else(|| fallback.to_string())
}

impl VeoRequest {
    /// Build from the shared intermediate representation (Gemini wire format).
    fn from_data(data: &VeoRequestData) -> Self {
        let first_frame = data
            .first_frame
            .as_ref()
            .map(|(b64, mime)| VeoMediaData::new(b64, mime));

        let last_frame = data
            .last_frame
            .as_ref()
            .map(|(b64, mime)| VeoMediaData::new(b64, mime));

        let video_ext = data
            .video
            .as_ref()
            .map(|(b64, mime)| VeoMediaData::new(b64, mime));

        let reference_images: Option<Vec<VeoReferenceImage>> = if data.reference_images.is_empty() {
            None
        } else {
            Some(
                data.reference_images
                    .iter()
                    .map(|(b64, mime)| VeoReferenceImage {
                        image: VeoMediaData::new(b64, mime),
                        reference_type: "asset".to_string(),
                    })
                    .collect(),
            )
        };

        let has_params = data.resolution.is_some()
            || data.aspect_ratio.is_some()
            || data.duration_seconds.is_some()
            || last_frame.is_some()
            || reference_images.is_some()
            || data.negative_prompt.is_some()
            || data.person_generation.is_some()
            || data.num_videos.is_some();

        let parameters = if has_params {
            Some(VeoParameters {
                aspect_ratio: data.aspect_ratio.clone(),
                resolution: data.resolution.clone(),
                duration_seconds: data.duration_seconds,
                last_frame,
                reference_images,
                negative_prompt: data.negative_prompt.clone(),
                person_generation: data.person_generation.clone(),
                number_of_videos: data.num_videos,
            })
        } else {
            None
        };

        Self {
            instances: vec![VeoInstance {
                prompt: data.prompt.clone(),
                image: first_frame,
                video: video_ext,
            }],
            parameters,
        }
    }

    /// Build directly from a VideoGenerationRequest (convenience for tests).
    #[cfg(test)]
    fn from_request(req: &VideoGenerationRequest) -> Self {
        Self::from_data(&VeoRequestData::from_request(req))
    }
}

// ── Vertex AI wire format ───────────────────────────────────────────────────

/// Vertex AI media format — flat `bytesBase64Encoded` + `mimeType`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexMediaData {
    bytes_base64_encoded: String,
    mime_type: String,
}

impl VertexMediaData {
    fn new(b64: &str, mime: &str) -> Self {
        Self {
            bytes_base64_encoded: b64.to_string(),
            mime_type: mime.to_string(),
        }
    }
}

/// Vertex AI reference image format.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexReferenceImage {
    image: VertexMediaData,
    reference_type: String,
}

/// Vertex AI instance (uses `bytesBase64Encoded` format).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexInstance {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<VertexMediaData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    video: Option<VertexMediaData>,
}

/// Vertex AI parameters (uses `sampleCount` instead of `numberOfVideos`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_seconds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_frame: Option<VertexMediaData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_images: Option<Vec<VertexReferenceImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    person_generation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enhance_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate_audio: Option<bool>,
}

/// Vertex AI request body.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexRequest {
    instances: Vec<VertexInstance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<VertexParameters>,
}

impl VertexRequest {
    /// Build from the shared intermediate representation (Vertex AI wire format).
    fn from_data(data: &VeoRequestData) -> Self {
        let first_frame = data
            .first_frame
            .as_ref()
            .map(|(b64, mime)| VertexMediaData::new(b64, mime));

        let last_frame = data
            .last_frame
            .as_ref()
            .map(|(b64, mime)| VertexMediaData::new(b64, mime));

        let video_ext = data
            .video
            .as_ref()
            .map(|(b64, mime)| VertexMediaData::new(b64, mime));

        let reference_images: Option<Vec<VertexReferenceImage>> =
            if data.reference_images.is_empty() {
                None
            } else {
                Some(
                    data.reference_images
                        .iter()
                        .map(|(b64, mime)| VertexReferenceImage {
                            image: VertexMediaData::new(b64, mime),
                            reference_type: "asset".to_string(),
                        })
                        .collect(),
                )
            };

        let has_params = data.resolution.is_some()
            || data.aspect_ratio.is_some()
            || data.duration_seconds.is_some()
            || last_frame.is_some()
            || reference_images.is_some()
            || data.negative_prompt.is_some()
            || data.person_generation.is_some()
            || data.num_videos.is_some()
            || data.storage_uri.is_some()
            || data.enhance_prompt.is_some()
            || data.generate_audio.is_some();

        let parameters = if has_params {
            Some(VertexParameters {
                aspect_ratio: data.aspect_ratio.clone(),
                resolution: data.resolution.clone(),
                duration_seconds: data.duration_seconds,
                last_frame,
                reference_images,
                negative_prompt: data.negative_prompt.clone(),
                person_generation: data.person_generation.clone(),
                sample_count: data.num_videos,
                storage_uri: data.storage_uri.clone(),
                enhance_prompt: data.enhance_prompt,
                generate_audio: data.generate_audio,
            })
        } else {
            None
        };

        Self {
            instances: vec![VertexInstance {
                prompt: data.prompt.clone(),
                image: first_frame,
                video: video_ext,
            }],
            parameters,
        }
    }
}

/// Vertex AI poll request body.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexFetchOperationRequest {
    operation_name: String,
}

// ── Response types (shared + Vertex-specific fields) ────────────────────────

#[derive(Debug, Deserialize)]
struct VeoOperationResponse {
    name: String,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    response: Option<VeoVideoResponse>,
    #[serde(default)]
    error: Option<VeoError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeoVideoResponse {
    /// Gemini: response from predictLongRunning endpoint.
    #[serde(default)]
    generate_video_response: Option<VeoGenerateVideoResponse>,
    /// Vertex AI: videos array in response.
    #[serde(default)]
    videos: Option<Vec<VertexVideo>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeoGenerateVideoResponse {
    #[serde(default)]
    generated_samples: Option<Vec<VeoGeneratedSample>>,
    #[serde(default)]
    rai_media_filtered_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct VeoGeneratedSample {
    #[serde(default)]
    video: Option<VeoVideo>,
}

#[derive(Debug, Deserialize)]
struct VeoVideo {
    #[serde(default)]
    uri: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VeoError {
    #[serde(default)]
    message: Option<String>,
}

/// Vertex AI video entry in response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexVideo {
    /// GCS URI (gs://...) — returned when `storage_uri` is set.
    #[serde(default)]
    gcs_uri: Option<String>,
    /// HTTPS download URL — sometimes returned instead of GCS URI.
    #[serde(default)]
    uri: Option<String>,
    /// Inline base64-encoded video data — the default Vertex AI response format.
    #[serde(default)]
    bytes_base64_encoded: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_veo_model_as_str() {
        assert_eq!(VeoModel::Veo31Preview.as_str(), "veo-3.1-generate-preview");
    }

    #[test]
    fn test_veo_model_vertex_id() {
        assert_eq!(VeoModel::Veo31Preview.vertex_id(), "veo-3.1-generate-001");
    }

    #[test]
    fn test_builder_with_explicit_key() {
        let provider = VeoProviderBuilder::new().api_key("test-key").build();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_builder_custom_timeouts() {
        let provider = VeoProviderBuilder::new()
            .api_key("test-key")
            .poll_interval(Duration::from_secs(30))
            .timeout(Duration::from_secs(900))
            .build()
            .unwrap();
        assert_eq!(provider.poll_interval, Duration::from_secs(30));
        assert_eq!(provider.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_builder_vertex_backend_with_project() {
        let provider = VeoProviderBuilder::new()
            .project("my-project")
            .location("us-east1")
            .build()
            .unwrap();
        match &provider.backend {
            VeoBackend::Vertex { project, location } => {
                assert_eq!(project, "my-project");
                assert_eq!(location, "us-east1");
            }
            _ => panic!("Expected Vertex backend"),
        }
        // API key should be None (no env var set in test)
        // This is fine for Vertex — auth uses gcloud
    }

    #[test]
    fn test_builder_vertex_backend_default_location() {
        let provider = VeoProviderBuilder::new()
            .project("my-project")
            .build()
            .unwrap();
        match &provider.backend {
            VeoBackend::Vertex { location, .. } => {
                assert_eq!(location, "us-central1");
            }
            _ => panic!("Expected Vertex backend"),
        }
    }

    #[test]
    fn test_builder_explicit_backend() {
        let provider = VeoProviderBuilder::new()
            .backend(VeoBackend::Vertex {
                project: "explicit-project".to_string(),
                location: "europe-west4".to_string(),
            })
            .build()
            .unwrap();
        match &provider.backend {
            VeoBackend::Vertex { project, location } => {
                assert_eq!(project, "explicit-project");
                assert_eq!(location, "europe-west4");
            }
            _ => panic!("Expected Vertex backend"),
        }
    }

    #[test]
    fn test_builder_gemini_requires_api_key() {
        // Without GOOGLE_API_KEY env var and no explicit key, Gemini should fail
        // (Only if GOOGLE_API_KEY and VERTEX_AI_PROJECT are not set in the env)
        if std::env::var("GOOGLE_API_KEY").is_err() && std::env::var("VERTEX_AI_PROJECT").is_err() {
            let result = VeoProviderBuilder::new().build();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_request_construction_basic() {
        let req = VideoGenerationRequest::new("Ocean waves");
        let veo_req = VeoRequest::from_request(&req);

        assert_eq!(veo_req.instances.len(), 1);
        assert_eq!(veo_req.instances[0].prompt, "Ocean waves");
        assert!(veo_req.parameters.is_none());
    }

    #[test]
    fn test_request_construction_with_resolution() {
        let req = VideoGenerationRequest::new("Ocean waves").with_resolution("720p");
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(params.resolution.as_deref(), Some("720p"));
    }

    #[test]
    fn test_request_construction_with_all_params() {
        let req = VideoGenerationRequest::new("Ocean waves")
            .with_resolution("1080p")
            .with_aspect_ratio("16:9")
            .with_duration(8);
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(params.resolution.as_deref(), Some("1080p"));
        assert_eq!(params.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(params.duration_seconds, Some(8));
    }

    #[test]
    fn test_request_serialization_uses_camel_case() {
        let req = VideoGenerationRequest::new("test")
            .with_resolution("720p")
            .with_duration(6);
        let veo_req = VeoRequest::from_request(&req);
        let json = serde_json::to_value(&veo_req).unwrap();

        assert!(json.get("instances").is_some());
        let params = json.get("parameters").unwrap();
        assert!(params.get("aspectRatio").is_some() || params.get("resolution").is_some());
        assert!(params.get("durationSeconds").is_some());
    }

    #[test]
    fn test_operation_response_not_done() {
        let json = r#"{"name": "operations/123", "done": false}"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.name, "operations/123");
        assert_eq!(resp.done, Some(false));
        assert!(resp.response.is_none());
    }

    #[test]
    fn test_operation_response_done_with_video() {
        let json = r#"{
            "name": "operations/123",
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [{
                        "video": {"uri": "https://example.com/video.mp4"}
                    }]
                }
            }
        }"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.done, Some(true));

        let samples = resp
            .response
            .unwrap()
            .generate_video_response
            .unwrap()
            .generated_samples
            .unwrap();
        let uri = samples[0].video.as_ref().unwrap().uri.as_deref();
        assert_eq!(uri, Some("https://example.com/video.mp4"));
    }

    #[test]
    fn test_operation_response_with_error() {
        let json = r#"{
            "name": "operations/123",
            "done": false,
            "error": {"message": "Quota exceeded"}
        }"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.error.unwrap().message.as_deref(),
            Some("Quota exceeded")
        );
    }

    #[test]
    fn test_gs_url_returns_error_gemini() {
        let provider = VeoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let result = rt.block_on(provider.download_gemini("gs://my-bucket/video.mp4"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Google Cloud Storage"),
            "Expected GCS error, got: {}",
            err
        );
    }

    #[test]
    fn test_request_construction_with_first_frame() {
        // Use a minimal PNG header as base64 for MIME detection
        let png_b64 = base64_encode_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

        let req = VideoGenerationRequest::new("Animate this frame").with_image(png_b64.clone());
        let veo_req = VeoRequest::from_request(&req);

        // First frame should be in instances
        let instance = &veo_req.instances[0];
        let image = instance.image.as_ref().unwrap();
        assert_eq!(image.inline_data.mime_type, "image/png");
        assert_eq!(image.inline_data.data, png_b64);

        // Duration forced to 8
        let params = veo_req.parameters.unwrap();
        assert_eq!(params.duration_seconds, Some(8));
        assert!(params.last_frame.is_none());
    }

    #[test]
    fn test_request_construction_with_last_frame() {
        // JPEG magic bytes (>= 12 bytes for from_magic_bytes)
        let jpeg_b64 = base64_encode_bytes(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let req =
            VideoGenerationRequest::new("End at this frame").with_last_frame(jpeg_b64.clone());
        let veo_req = VeoRequest::from_request(&req);

        // No first frame
        assert!(veo_req.instances[0].image.is_none());

        // Last frame in parameters
        let params = veo_req.parameters.unwrap();
        let last_frame = params.last_frame.as_ref().unwrap();
        assert_eq!(last_frame.inline_data.mime_type, "image/jpeg");
        assert_eq!(last_frame.inline_data.data, jpeg_b64);

        // Duration forced to 8
        assert_eq!(params.duration_seconds, Some(8));
    }

    #[test]
    fn test_request_construction_with_both_frames() {
        let b64 = base64_encode_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

        let req = VideoGenerationRequest::new("Interpolate between frames")
            .with_image(b64.clone())
            .with_last_frame(b64);
        let veo_req = VeoRequest::from_request(&req);

        assert!(veo_req.instances[0].image.is_some());
        let params = veo_req.parameters.unwrap();
        assert!(params.last_frame.is_some());
        assert_eq!(params.duration_seconds, Some(8));
    }

    #[test]
    fn test_request_serialization_with_frames_uses_inline_data() {
        let png_b64 = base64_encode_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        let jpeg_b64 = base64_encode_bytes(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let req = VideoGenerationRequest::new("test")
            .with_aspect_ratio("16:9")
            .with_image(png_b64.clone())
            .with_last_frame(jpeg_b64.clone());
        let veo_req = VeoRequest::from_request(&req);
        let json = serde_json::to_value(&veo_req).unwrap();

        // Verify inlineData format (Gemini API format)
        let instance = &json["instances"][0];
        assert_eq!(instance["image"]["inlineData"]["mimeType"], "image/png");
        assert_eq!(instance["image"]["inlineData"]["data"], png_b64);
        // Must NOT have bytesBase64Encoded (Vertex AI format)
        assert!(instance["image"]["bytesBase64Encoded"].is_null());

        let params = &json["parameters"];
        assert_eq!(params["lastFrame"]["inlineData"]["mimeType"], "image/jpeg");
        assert_eq!(params["lastFrame"]["inlineData"]["data"], jpeg_b64);
        assert_eq!(params["durationSeconds"], 8);
        assert_eq!(params["aspectRatio"], "16:9");
    }

    #[test]
    fn test_duration_not_forced_without_frames() {
        let req = VideoGenerationRequest::new("No frames").with_duration(5);
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(params.duration_seconds, Some(5));
    }

    #[test]
    fn test_request_with_video_extension() {
        let video_b64 = base64_encode_bytes(&[0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70]);

        let req = VideoGenerationRequest::new("Continue this video").with_video(video_b64.clone());
        let veo_req = VeoRequest::from_request(&req);

        let instance = &veo_req.instances[0];
        let video = instance.video.as_ref().unwrap();
        assert_eq!(video.inline_data.data, video_b64);

        // Duration forced to 8, resolution forced to 720p, numberOfVideos forced to 1
        let params = veo_req.parameters.unwrap();
        assert_eq!(params.duration_seconds, Some(8));
        assert_eq!(params.resolution.as_deref(), Some("720p"));
        assert_eq!(params.number_of_videos, Some(1));
    }

    #[test]
    fn test_request_with_reference_images() {
        let ref_b64 = base64_encode_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

        let req = VideoGenerationRequest::new("Style like these references")
            .with_reference_image(ref_b64.clone())
            .with_reference_image(ref_b64.clone());
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        let refs = params.reference_images.unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].reference_type, "asset");
        assert_eq!(refs[0].image.inline_data.mime_type, "image/png");
        // Duration forced to 8 for reference images
        assert_eq!(params.duration_seconds, Some(8));
    }

    #[test]
    fn test_request_with_negative_prompt() {
        let req =
            VideoGenerationRequest::new("A sunset").with_negative_prompt("blurry, low quality");
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(
            params.negative_prompt.as_deref(),
            Some("blurry, low quality")
        );
    }

    #[test]
    fn test_request_with_person_generation() {
        let req =
            VideoGenerationRequest::new("People walking").with_person_generation("allow_adult");
        let veo_req = VeoRequest::from_request(&req);

        let params = veo_req.parameters.unwrap();
        assert_eq!(params.person_generation.as_deref(), Some("allow_adult"));
    }

    #[test]
    fn test_video_extension_json_wire_format() {
        let video_b64 = base64_encode_bytes(&[0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70]);

        let req = VideoGenerationRequest::new("Continue walking through park")
            .with_video(video_b64.clone())
            .with_aspect_ratio("16:9");
        let veo_req = VeoRequest::from_request(&req);

        let json = serde_json::to_value(&veo_req).unwrap();

        // Verify instance structure: video is in instances[0].video.inlineData
        let instance = &json["instances"][0];
        assert!(instance["video"]["inlineData"]["data"].is_string());
        assert_eq!(
            instance["video"]["inlineData"]["mimeType"],
            "video/mp4" // mp4 magic bytes detected
        );

        // Verify parameters: resolution forced to 720p, numberOfVideos = 1, durationSeconds = 8
        let params = &json["parameters"];
        assert_eq!(params["resolution"], "720p");
        assert_eq!(params["numberOfVideos"], 1);
        assert_eq!(params["durationSeconds"], 8);
        assert_eq!(params["aspectRatio"], "16:9");

        // Verify no unexpected image field
        assert!(instance["image"].is_null());
    }

    #[test]
    fn test_parse_error_404_gives_helpful_message() {
        let provider = VeoProviderBuilder::new()
            .api_key("test-key")
            .build()
            .unwrap();

        let headers = reqwest::header::HeaderMap::new();
        let err = provider.parse_error(404, "Not Found", &headers);
        match err {
            GenVizError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("billing"),
                    "Expected billing hint, got: {}",
                    msg
                );
            }
            _ => panic!("Expected InvalidRequest error, got: {:?}", err),
        }
    }

    #[test]
    fn test_detect_mime_from_base64_png() {
        let png_b64 = base64_encode_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(detect_mime_from_base64(&png_b64, "image/png"), "image/png");
    }

    #[test]
    fn test_detect_mime_from_base64_jpeg() {
        let jpeg_b64 = base64_encode_bytes(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            detect_mime_from_base64(&jpeg_b64, "image/png"),
            "image/jpeg"
        );
    }

    #[test]
    fn test_detect_mime_from_base64_unknown_uses_fallback() {
        let unknown_b64 = base64_encode_bytes(&[0x00, 0x01, 0x02, 0x03]);
        assert_eq!(
            detect_mime_from_base64(&unknown_b64, "video/mp4"),
            "video/mp4"
        );
    }

    // ── Vertex AI wire format tests ─────────────────────────────────────────

    #[test]
    fn test_vertex_request_basic() {
        let req = VideoGenerationRequest::new("Ocean waves");
        let data = VeoRequestData::from_request(&req);
        let vertex_req = VertexRequest::from_data(&data);

        assert_eq!(vertex_req.instances.len(), 1);
        assert_eq!(vertex_req.instances[0].prompt, "Ocean waves");
        assert!(vertex_req.parameters.is_none());
    }

    #[test]
    fn test_vertex_request_uses_sample_count() {
        let req = VideoGenerationRequest::new("test")
            .with_resolution("720p")
            .with_duration(8);
        let data = VeoRequestData::from_request(&req);
        let vertex_req = VertexRequest::from_data(&data);

        let json = serde_json::to_value(&vertex_req).unwrap();
        let params = &json["parameters"];
        assert_eq!(params["durationSeconds"], 8);
        assert_eq!(params["resolution"], "720p");
        // Must use sampleCount, NOT numberOfVideos
        assert!(params.get("numberOfVideos").is_none() || params["numberOfVideos"].is_null());
    }

    #[test]
    fn test_vertex_request_uses_bytes_base64_encoded() {
        let png_b64 = base64_encode_bytes(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        let req = VideoGenerationRequest::new("Animate this").with_image(png_b64.clone());
        let data = VeoRequestData::from_request(&req);
        let vertex_req = VertexRequest::from_data(&data);

        let json = serde_json::to_value(&vertex_req).unwrap();
        let instance = &json["instances"][0];

        // Must use bytesBase64Encoded format (Vertex AI), NOT inlineData (Gemini)
        assert_eq!(instance["image"]["bytesBase64Encoded"], png_b64);
        assert_eq!(instance["image"]["mimeType"], "image/png");
        assert!(instance["image"]["inlineData"].is_null());
    }

    #[test]
    fn test_vertex_request_with_extra_params() {
        let req = VideoGenerationRequest::new("test")
            .with_duration(8)
            .with_storage_uri("gs://my-bucket/output/")
            .with_enhance_prompt(true)
            .with_generate_audio(true);
        let data = VeoRequestData::from_request(&req);
        let vertex_req = VertexRequest::from_data(&data);

        let json = serde_json::to_value(&vertex_req).unwrap();
        let params = &json["parameters"];
        assert_eq!(params["storageUri"], "gs://my-bucket/output/");
        assert_eq!(params["enhancePrompt"], true);
        assert_eq!(params["generateAudio"], true);
    }

    #[test]
    fn test_vertex_video_response_deserialization() {
        let json = r#"{
            "name": "projects/123/locations/us-central1/operations/456",
            "done": true,
            "response": {
                "videos": [
                    {"gcsUri": "gs://bucket/video.mp4"},
                    {"uri": "https://example.com/video.mp4"}
                ]
            }
        }"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert!(resp.done.unwrap());
        let videos = resp.response.unwrap().videos.unwrap();
        assert_eq!(videos.len(), 2);
        assert_eq!(videos[0].gcs_uri.as_deref(), Some("gs://bucket/video.mp4"));
        assert_eq!(
            videos[1].uri.as_deref(),
            Some("https://example.com/video.mp4")
        );
    }

    #[test]
    fn test_vertex_video_response_inline_base64() {
        let json = r#"{
            "name": "projects/123/locations/us-central1/operations/456",
            "done": true,
            "response": {
                "raiMediaFilteredCount": 0,
                "videos": [
                    {"bytesBase64Encoded": "AQID"}
                ]
            }
        }"#;
        let resp: VeoOperationResponse = serde_json::from_str(json).unwrap();
        assert!(resp.done.unwrap());
        let videos = resp.response.unwrap().videos.unwrap();
        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].bytes_base64_encoded.as_deref(), Some("AQID"));
        assert!(videos[0].gcs_uri.is_none());
        assert!(videos[0].uri.is_none());

        // Verify decoding works
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(videos[0].bytes_base64_encoded.as_ref().unwrap())
            .unwrap();
        assert_eq!(decoded, vec![1, 2, 3]);
    }

    #[test]
    fn test_vertex_fetch_operation_request_serialization() {
        let req = VertexFetchOperationRequest {
            operation_name: "projects/123/operations/456".to_string(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["operationName"], "projects/123/operations/456");
    }

    #[test]
    fn test_vertex_video_extension_forces_constraints() {
        let video_b64 = base64_encode_bytes(&[0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70]);
        let req = VideoGenerationRequest::new("Continue this").with_video(video_b64.clone());
        let data = VeoRequestData::from_request(&req);
        let vertex_req = VertexRequest::from_data(&data);

        let json = serde_json::to_value(&vertex_req).unwrap();
        let params = &json["parameters"];
        assert_eq!(params["durationSeconds"], 8);
        assert_eq!(params["resolution"], "720p");
        assert_eq!(params["sampleCount"], 1);
    }

    /// Helper: base64-encode raw bytes for test assertions.
    fn base64_encode_bytes(bytes: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }
}
