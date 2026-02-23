//! Gemini (Google) image generation provider.
//!
//! Supports two backends:
//! - **Gemini Developer API** (`generativelanguage.googleapis.com`) — API key auth
//! - **Vertex AI** (`aiplatform.googleapis.com`) — gcloud CLI auth, pay-per-use
//!
//! Auto-detects Vertex AI when `VERTEX_AI_PROJECT` env var is set.

use crate::auth::gcloud_access_token;
use crate::error::{parse_retry_after, sanitize_error_message, GenVizError, Result};
use crate::image::provider::ImageProvider;
use crate::image::types::{
    GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProviderKind,
};
use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Which Google API backend to use for Gemini image generation.
#[derive(Debug, Clone, Default)]
pub enum GeminiBackend {
    /// Gemini Developer API (generativelanguage.googleapis.com).
    /// Uses API key auth.
    #[default]
    Gemini,
    /// Vertex AI (aiplatform.googleapis.com).
    /// Uses gcloud CLI for auth. No daily rate limits, pay-per-use.
    Vertex {
        /// GCP project ID.
        project: String,
        /// GCP location (e.g. "us-central1").
        location: String,
    },
}

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
    /// Returns the Gemini Developer API model identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NanoBanana => "gemini-2.5-flash-image",
            Self::NanoBananaPro => "nano-banana-pro-preview",
        }
    }

    /// Returns the Vertex AI model identifier.
    pub fn vertex_id(&self) -> &'static str {
        match self {
            Self::NanoBanana => "gemini-2.5-flash-image",
            Self::NanoBananaPro => "gemini-3-pro-image-preview",
        }
    }
}

/// Builder for GeminiProvider.
#[derive(Debug, Clone, Default)]
pub struct GeminiProviderBuilder {
    api_key: Option<String>,
    model: GeminiModel,
    backend: Option<GeminiBackend>,
    project: Option<String>,
    location: Option<String>,
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

    /// Explicitly sets the backend (Gemini or Vertex AI).
    pub fn backend(mut self, backend: GeminiBackend) -> Self {
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
    pub fn build(self) -> Result<GeminiProvider> {
        // Resolve backend
        let backend = if let Some(b) = self.backend {
            b
        } else if let Some(project) = self.project.clone() {
            let location = self
                .location
                .clone()
                .unwrap_or_else(|| "us-central1".to_string());
            GeminiBackend::Vertex { project, location }
        } else if let Ok(project) = std::env::var("VERTEX_AI_PROJECT") {
            let location =
                std::env::var("VERTEX_AI_LOCATION").unwrap_or_else(|_| "us-central1".to_string());
            GeminiBackend::Vertex { project, location }
        } else {
            GeminiBackend::Gemini
        };

        // Resolve API key — required for Gemini, optional for Vertex
        let api_key = match &backend {
            GeminiBackend::Gemini => {
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
            GeminiBackend::Vertex { .. } => {
                // API key is optional for Vertex — auth uses gcloud CLI
                self.api_key
                    .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            }
        };

        Ok(GeminiProvider {
            client: reqwest::Client::new(),
            api_key,
            model: self.model,
            backend,
        })
    }
}

/// Gemini image generation provider.
pub struct GeminiProvider {
    client: reqwest::Client,
    /// API key — Some for Gemini (required), optional for Vertex.
    api_key: Option<String>,
    model: GeminiModel,
    backend: GeminiBackend,
}

impl GeminiProvider {
    /// Creates a new `GeminiProviderBuilder`.
    pub fn builder() -> GeminiProviderBuilder {
        GeminiProviderBuilder::new()
    }

    /// Returns a reference to the active backend.
    pub fn backend(&self) -> &GeminiBackend {
        &self.backend
    }

    async fn generate_impl(&self, request: &GenerationRequest) -> Result<GeneratedImage> {
        let start = Instant::now();
        let body = GeminiRequest::from_generation_request(request);

        let response = match &self.backend {
            GeminiBackend::Gemini => {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
                    self.model.as_str(),
                );
                let api_key = self.api_key.as_ref().ok_or_else(|| {
                    GenVizError::Auth("GOOGLE_API_KEY required for Gemini backend".into())
                })?;

                self.client
                    .post(&url)
                    .header("x-goog-api-key", api_key)
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await?
            }
            GeminiBackend::Vertex { project, location } => {
                let model_id = self.model.vertex_id();
                // Use global endpoint — Gemini 3 models are only available there,
                // and Gemini 2.x also works on global.
                let url = format!(
                    "https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model_id}:generateContent",
                );
                let _ = location; // location kept for batch API which requires regional endpoints
                let token = gcloud_access_token()?;

                self.client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await?
            }
        };

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let text = response.text().await.unwrap_or_default();
            return Err(self.parse_error(status.as_u16(), &text, &headers));
        }

        let gemini_response: GeminiResponse = response.json().await?;
        self.extract_image(gemini_response, start)
    }

    /// Extract image data from a Gemini response, handling safety filters.
    fn extract_image(
        &self,
        gemini_response: GeminiResponse,
        start: Instant,
    ) -> Result<GeneratedImage> {
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

        let model_name = match &self.backend {
            GeminiBackend::Gemini => self.model.as_str(),
            GeminiBackend::Vertex { .. } => self.model.vertex_id(),
        };

        Ok(GeneratedImage::new(
            data,
            format,
            ImageProviderKind::Gemini,
            GenerationMetadata {
                model: Some(model_name.to_string()),
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
        match &self.backend {
            GeminiBackend::Gemini => {
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
            GeminiBackend::Vertex { project, location } => {
                let token = gcloud_access_token()?;
                let model_id = self.model.vertex_id();
                let _ = location;
                let url = format!(
                    "https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model_id}",
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

// ── Batch Prediction (Vertex AI only, CLI feature) ──────────────────────────

/// Submit a batch image generation job to Vertex AI.
///
/// Creates JSONL input from prompts, uploads to GCS, and submits
/// a batch prediction job. Returns the job name for status polling.
#[cfg(feature = "cli")]
pub async fn submit_batch(
    project: &str,
    location: &str,
    prompts: &[String],
    gcs_bucket: &str,
    model: GeminiModel,
) -> Result<BatchSubmitResult> {
    let client = reqwest::Client::new();
    let token = gcloud_access_token()?;
    let batch_id = format!(
        "genviz-batch-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );

    // Build JSONL input
    let mut jsonl = String::new();
    for prompt in prompts {
        let line = serde_json::json!({
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]}
            }
        });
        jsonl.push_str(&serde_json::to_string(&line).map_err(|e| {
            GenVizError::InvalidRequest(format!("Failed to serialize batch input: {}", e))
        })?);
        jsonl.push('\n');
    }

    // Upload JSONL to GCS
    let (bucket_name, _) = parse_gcs_bucket(gcs_bucket)?;
    let input_object = format!("genviz-batch/{}/input.jsonl", batch_id);
    let upload_url = format!(
        "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}",
        bucket_name,
        urlencoded(&input_object),
    );

    let upload_resp = client
        .post(&upload_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/x-ndjson")
        .body(jsonl)
        .send()
        .await?;

    if !upload_resp.status().is_success() {
        let text = upload_resp.text().await.unwrap_or_default();
        return Err(GenVizError::Api {
            status: 0,
            message: format!(
                "Failed to upload batch input to GCS: {}",
                sanitize_error_message(&text)
            ),
        });
    }

    let input_uri = format!("gs://{}/{}", bucket_name, input_object);
    let output_uri_prefix = format!("gs://{}/genviz-batch/{}/output/", bucket_name, batch_id);

    // Submit batch prediction job
    let model_id = model.vertex_id();
    let batch_url = format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/batchPredictionJobs",
    );

    let batch_body = serde_json::json!({
        "displayName": batch_id,
        "model": format!("publishers/google/models/{}", model_id),
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": { "uris": [input_uri] }
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": { "outputUriPrefix": output_uri_prefix }
        }
    });

    // Re-fetch token in case the upload took a while
    let token = gcloud_access_token()?;
    let batch_resp = client
        .post(&batch_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(&batch_body)
        .send()
        .await?;

    let batch_status = batch_resp.status();
    if !batch_status.is_success() {
        let text = batch_resp.text().await.unwrap_or_default();
        return Err(GenVizError::Api {
            status: batch_status.as_u16(),
            message: format!(
                "Failed to submit batch job: {}",
                sanitize_error_message(&text)
            ),
        });
    }

    let job: BatchPredictionJobResponse = batch_resp.json().await?;

    Ok(BatchSubmitResult {
        job_name: job.name,
        display_name: job.display_name,
        state: job.state,
        num_prompts: prompts.len(),
        input_uri,
    })
}

/// Check the status of a batch prediction job.
#[cfg(feature = "cli")]
pub async fn get_batch_status(job_name: &str) -> Result<BatchJobStatus> {
    let client = reqwest::Client::new();
    let token = gcloud_access_token()?;

    // The job_name is a full resource path like:
    // projects/{project}/locations/{location}/batchPredictionJobs/{id}
    let location = extract_location_from_job_name(job_name)?;
    let url = format!("https://{location}-aiplatform.googleapis.com/v1/{job_name}",);

    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err(GenVizError::Api {
            status,
            message: format!(
                "Failed to get batch job status: {}",
                sanitize_error_message(&text)
            ),
        });
    }

    let job: BatchPredictionJobResponse = resp.json().await?;

    let output_uri_prefix = job.output_info.and_then(|o| o.gcs_output_directory);

    Ok(BatchJobStatus {
        job_name: job.name,
        display_name: job.display_name,
        state: job.state,
        create_time: job.create_time,
        update_time: job.update_time,
        output_uri_prefix,
    })
}

/// Download and save batch prediction results from GCS.
///
/// Reads the output JSONL, extracts images, and saves them to `output_dir`.
/// Returns metadata about each saved image.
#[cfg(feature = "cli")]
pub async fn download_batch_results(
    output_uri_prefix: &str,
    output_dir: &str,
) -> Result<Vec<BatchImageResult>> {
    let client = reqwest::Client::new();
    let token = gcloud_access_token()?;

    // List objects in the output prefix
    let (bucket, prefix) = crate::auth::parse_gcs_uri(output_uri_prefix)?;
    let list_url = format!(
        "https://storage.googleapis.com/storage/v1/b/{}/o?prefix={}",
        bucket,
        urlencoded(prefix),
    );

    let list_resp = client
        .get(&list_url)
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await?;

    if !list_resp.status().is_success() {
        let text = list_resp.text().await.unwrap_or_default();
        return Err(GenVizError::Api {
            status: 0,
            message: format!(
                "Failed to list batch output: {}",
                sanitize_error_message(&text)
            ),
        });
    }

    let list_body: GcsListResponse = list_resp.json().await?;
    let items = list_body.items.unwrap_or_default();

    // Find prediction JSONL files
    let jsonl_objects: Vec<&str> = items
        .iter()
        .filter(|item| item.name.ends_with(".jsonl"))
        .map(|item| item.name.as_str())
        .collect();

    if jsonl_objects.is_empty() {
        return Err(GenVizError::UnexpectedResponse(
            "No output JSONL files found in batch results".into(),
        ));
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    let mut results = Vec::new();
    let mut image_index = 0;

    for object_name in jsonl_objects {
        // Download the JSONL file
        let download_url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
            bucket,
            urlencoded(object_name),
        );

        let token = gcloud_access_token()?;
        let download_resp = client
            .get(&download_url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await?;

        if !download_resp.status().is_success() {
            continue;
        }

        let body = download_resp.text().await.unwrap_or_default();

        // Parse each line as a batch output entry
        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let entry: BatchOutputLine = match serde_json::from_str(line) {
                Ok(e) => e,
                Err(_) => continue,
            };

            if let Some(resp) = entry.response {
                // Extract image from the response (same structure as online response)
                if let Some(candidate) = resp.candidates.into_iter().next() {
                    if let Some(content) = candidate.content {
                        if let Some(inline_data) =
                            content.parts.into_iter().find_map(|p| p.inline_data)
                        {
                            let data = base64::engine::general_purpose::STANDARD
                                .decode(&inline_data.data)
                                .map_err(|e| GenVizError::Decode(e.to_string()))?;

                            let ext = match inline_data.mime_type.as_str() {
                                "image/jpeg" => "jpg",
                                "image/webp" => "webp",
                                _ => "png",
                            };

                            let filename = format!("image_{:04}.{}", image_index, ext);
                            let filepath = format!("{}/{}", output_dir, filename);
                            std::fs::write(&filepath, &data)?;

                            results.push(BatchImageResult {
                                index: image_index,
                                path: filepath,
                                size_bytes: data.len(),
                                format: ext.to_string(),
                            });

                            image_index += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Extract the location from a full batch job resource name.
#[cfg(feature = "cli")]
fn extract_location_from_job_name(job_name: &str) -> Result<String> {
    // Format: projects/{project}/locations/{location}/batchPredictionJobs/{id}
    let parts: Vec<&str> = job_name.split('/').collect();
    if parts.len() >= 4 && parts[2] == "locations" {
        Ok(parts[3].to_string())
    } else {
        Err(GenVizError::InvalidRequest(format!(
            "Cannot extract location from job name: {}",
            job_name
        )))
    }
}

/// Parse a GCS bucket reference (with or without gs:// prefix) into a bucket name.
#[cfg(feature = "cli")]
fn parse_gcs_bucket(bucket: &str) -> Result<(String, String)> {
    let stripped = bucket.strip_prefix("gs://").unwrap_or(bucket);
    let stripped = stripped.trim_end_matches('/');
    if stripped.contains('/') {
        // bucket/prefix format
        let (name, prefix) = stripped.split_once('/').unwrap();
        Ok((name.to_string(), prefix.to_string()))
    } else {
        Ok((stripped.to_string(), String::new()))
    }
}

/// URL-encode a string for use in GCS API paths.
#[cfg(feature = "cli")]
fn urlencoded(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('/', "%2F")
        .replace('?', "%3F")
        .replace('#', "%23")
        .replace('&', "%26")
        .replace('=', "%3D")
}

// ── Batch types ─────────────────────────────────────────────────────────────

/// Result of submitting a batch prediction job.
#[cfg(feature = "cli")]
#[derive(Debug, Serialize)]
pub struct BatchSubmitResult {
    /// Full resource name of the batch job.
    pub job_name: String,
    /// Display name of the batch job.
    pub display_name: String,
    /// Current state (e.g., "JOB_STATE_PENDING").
    pub state: String,
    /// Number of prompts submitted.
    pub num_prompts: usize,
    /// GCS URI of the input JSONL.
    pub input_uri: String,
}

/// Status of a batch prediction job.
#[cfg(feature = "cli")]
#[derive(Debug, Serialize)]
pub struct BatchJobStatus {
    /// Full resource name of the batch job.
    pub job_name: String,
    /// Display name of the batch job.
    pub display_name: String,
    /// Current state (e.g., "JOB_STATE_SUCCEEDED", "JOB_STATE_RUNNING").
    pub state: String,
    /// When the job was created.
    pub create_time: Option<String>,
    /// When the job was last updated.
    pub update_time: Option<String>,
    /// GCS prefix where output files are written (available when complete).
    pub output_uri_prefix: Option<String>,
}

/// Result of downloading a single image from batch output.
#[cfg(feature = "cli")]
#[derive(Debug, Serialize)]
pub struct BatchImageResult {
    /// Index of the image in the batch.
    pub index: usize,
    /// Path where the image was saved.
    pub path: String,
    /// Size of the image in bytes.
    pub size_bytes: usize,
    /// Image format extension.
    pub format: String,
}

/// Vertex AI batch prediction job response.
#[cfg(feature = "cli")]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchPredictionJobResponse {
    name: String,
    display_name: String,
    state: String,
    #[serde(default)]
    create_time: Option<String>,
    #[serde(default)]
    update_time: Option<String>,
    #[serde(default)]
    output_info: Option<BatchOutputInfo>,
}

#[cfg(feature = "cli")]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchOutputInfo {
    gcs_output_directory: Option<String>,
}

/// A single line from the batch output JSONL.
#[cfg(feature = "cli")]
#[derive(Debug, Deserialize)]
struct BatchOutputLine {
    #[serde(default)]
    response: Option<GeminiResponse>,
}

/// GCS list objects response.
#[cfg(feature = "cli")]
#[derive(Debug, Deserialize)]
struct GcsListResponse {
    #[serde(default)]
    items: Option<Vec<GcsObject>>,
}

#[cfg(feature = "cli")]
#[derive(Debug, Deserialize)]
struct GcsObject {
    name: String,
}

// ── Request/Response types ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: GeminiConfig,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: &'static str,
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
            contents: vec![GeminiContent {
                role: "user",
                parts,
            }],
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
    fn test_gemini_model_vertex_id() {
        assert_eq!(
            GeminiModel::NanoBanana.vertex_id(),
            "gemini-2.5-flash-image"
        );
        assert_eq!(
            GeminiModel::NanoBananaPro.vertex_id(),
            "gemini-3-pro-image-preview"
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
    fn test_builder_vertex_backend_with_project() {
        let provider = GeminiProviderBuilder::new()
            .project("my-project")
            .location("us-east1")
            .build()
            .unwrap();
        match &provider.backend {
            GeminiBackend::Vertex { project, location } => {
                assert_eq!(project, "my-project");
                assert_eq!(location, "us-east1");
            }
            _ => panic!("Expected Vertex backend"),
        }
    }

    #[test]
    fn test_builder_vertex_backend_default_location() {
        let provider = GeminiProviderBuilder::new()
            .project("my-project")
            .build()
            .unwrap();
        match &provider.backend {
            GeminiBackend::Vertex { location, .. } => {
                assert_eq!(location, "us-central1");
            }
            _ => panic!("Expected Vertex backend"),
        }
    }

    #[test]
    fn test_builder_explicit_backend() {
        let provider = GeminiProviderBuilder::new()
            .backend(GeminiBackend::Vertex {
                project: "explicit-project".to_string(),
                location: "europe-west4".to_string(),
            })
            .build()
            .unwrap();
        match &provider.backend {
            GeminiBackend::Vertex { project, location } => {
                assert_eq!(project, "explicit-project");
                assert_eq!(location, "europe-west4");
            }
            _ => panic!("Expected Vertex backend"),
        }
    }

    #[test]
    fn test_builder_gemini_requires_api_key() {
        if std::env::var("GOOGLE_API_KEY").is_err() && std::env::var("VERTEX_AI_PROJECT").is_err() {
            let result = GeminiProviderBuilder::new().build();
            assert!(result.is_err());
        }
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

    #[cfg(feature = "cli")]
    #[test]
    fn test_extract_location_from_job_name() {
        let job_name = "projects/my-project/locations/us-central1/batchPredictionJobs/12345";
        let location = extract_location_from_job_name(job_name).unwrap();
        assert_eq!(location, "us-central1");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_extract_location_from_job_name_invalid() {
        assert!(extract_location_from_job_name("invalid/path").is_err());
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_parse_gcs_bucket_simple() {
        let (bucket, prefix) = parse_gcs_bucket("my-bucket").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_parse_gcs_bucket_with_prefix() {
        let (bucket, prefix) = parse_gcs_bucket("gs://my-bucket").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_parse_gcs_bucket_with_gs_prefix_and_path() {
        let (bucket, prefix) = parse_gcs_bucket("gs://my-bucket/some/path/").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "some/path");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_batch_output_line_deserialization() {
        let json = r#"{
            "status": "",
            "response": {
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
            }
        }"#;
        let line: BatchOutputLine = serde_json::from_str(json).unwrap();
        assert!(line.response.is_some());
        let resp = line.response.unwrap();
        assert_eq!(resp.candidates.len(), 1);
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_batch_prediction_job_response_deserialization() {
        let json = r#"{
            "name": "projects/my-proj/locations/us-central1/batchPredictionJobs/123",
            "displayName": "genviz-batch-1234567890",
            "state": "JOB_STATE_RUNNING",
            "createTime": "2026-01-01T00:00:00Z",
            "updateTime": "2026-01-01T00:01:00Z"
        }"#;
        let resp: BatchPredictionJobResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.state, "JOB_STATE_RUNNING");
        assert!(resp.output_info.is_none());
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_batch_prediction_job_response_with_output() {
        let json = r#"{
            "name": "projects/my-proj/locations/us-central1/batchPredictionJobs/123",
            "displayName": "genviz-batch-1234567890",
            "state": "JOB_STATE_SUCCEEDED",
            "outputInfo": {
                "gcsOutputDirectory": "gs://my-bucket/genviz-batch/123/output/"
            }
        }"#;
        let resp: BatchPredictionJobResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.state, "JOB_STATE_SUCCEEDED");
        let output = resp.output_info.unwrap();
        assert_eq!(
            output.gcs_output_directory.as_deref(),
            Some("gs://my-bucket/genviz-batch/123/output/")
        );
    }
}
