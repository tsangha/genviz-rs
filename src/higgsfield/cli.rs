//! Thin async wrapper around the `higgsfield` CLI binary.
//!
//! The CLI accepts `--json` and prints structured output, so we shell out and
//! parse stdout rather than re-implementing Higgsfield's HTTP/auth layer.

use crate::error::{GenVizError, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

/// Higgsfield top-level surface to invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiggsfieldSurface {
    /// Generic `higgsfield generate create <model>` — accepts every model.
    Generate,
    /// `higgsfield product-photoshoot create --mode <mode>` — brand/product imagery.
    ProductPhotoshoot,
    /// `higgsfield marketplace-cards create` — marketplace listing cards.
    MarketplaceCards,
}

/// Mode for the `product-photoshoot` surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiggsfieldMode {
    /// Studio product shot.
    ProductShot,
    /// Lifestyle scene featuring the product.
    LifestyleScene,
    /// Close-up of product with a person's hands/face.
    CloseupProductWithPerson,
    /// Pinterest-style moodboard pin.
    MoodboardPin,
    /// Hero banner image.
    HeroBanner,
    /// Social carousel image.
    SocialCarousel,
    /// Paid social ad creative pack.
    AdCreativePack,
    /// Virtual try-on / model wearing product.
    VirtualModelTryout,
    /// Surreal/CGI conceptual product shot.
    ConceptualProduct,
    /// Restyle an existing scene.
    Restyle,
}

impl HiggsfieldMode {
    /// Returns the CLI mode string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProductShot => "product_shot",
            Self::LifestyleScene => "lifestyle_scene",
            Self::CloseupProductWithPerson => "closeup_product_with_person",
            Self::MoodboardPin => "moodboard_pin",
            Self::HeroBanner => "hero_banner",
            Self::SocialCarousel => "social_carousel",
            Self::AdCreativePack => "ad_creative_pack",
            Self::VirtualModelTryout => "virtual_model_tryout",
            Self::ConceptualProduct => "conceptual_product",
            Self::Restyle => "restyle",
        }
    }
}

/// A completed Higgsfield job — what we pull out of `generate ... --wait --json`.
#[derive(Debug, Clone, Deserialize)]
pub struct HiggsfieldJob {
    /// Job UUID.
    pub id: String,
    /// Job status, e.g. "completed".
    pub status: String,
    /// Direct CDN URL to the produced asset (may be missing while running).
    #[serde(default)]
    pub result_url: Option<String>,
    /// `job_set_type` — what model produced this job.
    #[serde(default)]
    pub job_set_type: Option<String>,
}

/// CLI wrapper.
///
/// `binary` defaults to `"higgsfield"`. Override for alternative install paths.
#[derive(Debug, Clone)]
pub struct HiggsfieldCli {
    binary: PathBuf,
    /// Soft cap on wait time per generation. The CLI's own `--wait-timeout`
    /// flag enforces this server-side.
    pub wait_timeout: Duration,
}

impl Default for HiggsfieldCli {
    fn default() -> Self {
        Self {
            binary: PathBuf::from("higgsfield"),
            wait_timeout: Duration::from_secs(900),
        }
    }
}

impl HiggsfieldCli {
    /// Construct with a custom binary path.
    pub fn with_binary(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            wait_timeout: Duration::from_secs(900),
        }
    }

    /// Override the wait timeout.
    pub fn with_wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    /// Returns a fresh `Command` for the CLI with `--json` already appended.
    fn cmd(&self) -> Command {
        let mut c = Command::new(&self.binary);
        c.arg("--json");
        c.arg("--no-color");
        c
    }

    /// Probe `higgsfield version`. Returns the version string on success.
    /// Surfaces a clear error if the binary is missing.
    pub async fn version(&self) -> Result<String> {
        let out = self
            .cmd()
            .arg("version")
            .output()
            .await
            .map_err(|e| missing_binary_err(&self.binary, e))?;
        if !out.status.success() {
            return Err(GenVizError::ProviderNotAvailable(format!(
                "higgsfield version failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// Run `generate create <model> [...] --wait --json`. Returns the final job.
    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        extra_args: &[String],
    ) -> Result<HiggsfieldJob> {
        let mut cmd = self.cmd();
        cmd.args(["generate", "create", model, "--prompt", prompt]);
        for a in extra_args {
            cmd.arg(a);
        }
        cmd.arg("--wait");
        cmd.args([
            "--wait-timeout",
            &format_duration_compact(self.wait_timeout),
        ]);

        self.run_and_parse_job(cmd, "higgsfield generate create")
            .await
    }

    /// Run `product-photoshoot create --mode <mode> ...`.
    pub async fn product_photoshoot(
        &self,
        mode: HiggsfieldMode,
        prompt: &str,
        extra_args: &[String],
    ) -> Result<HiggsfieldJob> {
        let mut cmd = self.cmd();
        cmd.args([
            "product-photoshoot",
            "create",
            "--mode",
            mode.as_str(),
            "--prompt",
            prompt,
        ]);
        for a in extra_args {
            cmd.arg(a);
        }
        cmd.arg("--wait");
        cmd.args([
            "--wait-timeout",
            &format_duration_compact(self.wait_timeout),
        ]);

        self.run_and_parse_job(cmd, "higgsfield product-photoshoot create")
            .await
    }

    /// Run `marketplace-cards create ...`.
    pub async fn marketplace_cards(
        &self,
        prompt: &str,
        extra_args: &[String],
    ) -> Result<HiggsfieldJob> {
        let mut cmd = self.cmd();
        cmd.args(["marketplace-cards", "create", "--prompt", prompt]);
        for a in extra_args {
            cmd.arg(a);
        }
        cmd.arg("--wait");
        cmd.args([
            "--wait-timeout",
            &format_duration_compact(self.wait_timeout),
        ]);

        self.run_and_parse_job(cmd, "higgsfield marketplace-cards create")
            .await
    }

    /// Upload a local file. Returns the upload UUID.
    pub async fn upload(&self, path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();
        let mut cmd = self.cmd();
        cmd.args(["upload", "create"]).arg(path);

        let out = cmd
            .output()
            .await
            .map_err(|e| missing_binary_err(&self.binary, e))?;
        if !out.status.success() {
            return Err(GenVizError::ProviderNotAvailable(format!(
                "higgsfield upload create {}: {}",
                path.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }

        #[derive(Deserialize)]
        struct UploadResponse {
            id: String,
        }
        let parsed: UploadResponse = serde_json::from_slice(&out.stdout)
            .map_err(|e| GenVizError::Json(format!("upload response: {e}")))?;
        Ok(parsed.id)
    }

    /// Train a Soul Character from a set of local face images.
    /// Returns the soul reference id when the training job completes.
    pub async fn soul_id_train(
        &self,
        name: &str,
        images: &[PathBuf],
        extra_args: &[String],
    ) -> Result<String> {
        let mut cmd = self.cmd();
        cmd.args(["soul-id", "train", "--name", name]);
        for img in images {
            cmd.arg("--image").arg(img);
        }
        for a in extra_args {
            cmd.arg(a);
        }
        cmd.arg("--wait");
        cmd.args([
            "--wait-timeout",
            &format_duration_compact(self.wait_timeout),
        ]);

        let out = cmd
            .output()
            .await
            .map_err(|e| missing_binary_err(&self.binary, e))?;
        if !out.status.success() {
            return Err(GenVizError::ProviderNotAvailable(format!(
                "higgsfield soul-id train: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        #[derive(Deserialize)]
        struct SoulResponse {
            #[serde(alias = "reference_id", alias = "id")]
            id: String,
        }
        let parsed: SoulResponse = serde_json::from_slice(&out.stdout)
            .map_err(|e| GenVizError::Json(format!("soul-id train response: {e}")))?;
        Ok(parsed.id)
    }

    /// List existing Soul refs. Returns `(id, name)` tuples.
    pub async fn soul_id_list(&self) -> Result<Vec<(String, String)>> {
        let out = self
            .cmd()
            .args(["soul-id", "list"])
            .output()
            .await
            .map_err(|e| missing_binary_err(&self.binary, e))?;
        if !out.status.success() {
            return Err(GenVizError::ProviderNotAvailable(format!(
                "higgsfield soul-id list: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        #[derive(Deserialize)]
        struct Soul {
            id: String,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            display_name: Option<String>,
        }
        let parsed: Vec<Soul> = serde_json::from_slice(&out.stdout)
            .map_err(|e| GenVizError::Json(format!("soul-id list response: {e}")))?;
        Ok(parsed
            .into_iter()
            .map(|s| {
                let name = s.name.or(s.display_name).unwrap_or_default();
                (s.id, name)
            })
            .collect())
    }

    /// Internal: run the command, parse stdout as a job (or array of jobs, taking the first).
    async fn run_and_parse_job(&self, mut cmd: Command, context: &str) -> Result<HiggsfieldJob> {
        let out = cmd
            .output()
            .await
            .map_err(|e| missing_binary_err(&self.binary, e))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            return Err(map_cli_failure(context, &stderr));
        }

        parse_job_stdout(&out.stdout, context)
    }
}

/// Parse `generate create --wait --json` stdout. The CLI returns either a
/// single job object or an array (when `--count > 1`).
fn parse_job_stdout(stdout: &[u8], context: &str) -> Result<HiggsfieldJob> {
    // Try single-job shape first; fall back to array.
    if let Ok(job) = serde_json::from_slice::<HiggsfieldJob>(stdout) {
        return ensure_completed(job, context);
    }
    if let Ok(jobs) = serde_json::from_slice::<Vec<HiggsfieldJob>>(stdout) {
        if let Some(first) = jobs.into_iter().next() {
            return ensure_completed(first, context);
        }
        return Err(GenVizError::UnexpectedResponse(format!(
            "{context}: empty job array"
        )));
    }
    Err(GenVizError::UnexpectedResponse(format!(
        "{context}: unparseable JSON ({})",
        String::from_utf8_lossy(stdout)
            .chars()
            .take(200)
            .collect::<String>()
    )))
}

fn ensure_completed(job: HiggsfieldJob, context: &str) -> Result<HiggsfieldJob> {
    match job.status.as_str() {
        "completed" | "succeeded" | "done" => {
            if job.result_url.is_some() {
                Ok(job)
            } else {
                Err(GenVizError::UnexpectedResponse(format!(
                    "{context}: completed job {} has no result_url",
                    job.id
                )))
            }
        }
        "failed" | "error" => Err(GenVizError::Api {
            status: 0,
            message: format!("{context}: job {} failed", job.id),
        }),
        other => Err(GenVizError::Timeout(Duration::from_secs(0))).map_err(|_| GenVizError::Api {
            status: 0,
            message: format!(
                "{context}: job {} did not complete (status={})",
                job.id, other
            ),
        }),
    }
}

fn missing_binary_err(binary: &Path, err: std::io::Error) -> GenVizError {
    if err.kind() == std::io::ErrorKind::NotFound {
        GenVizError::ProviderNotAvailable(format!(
            "`{}` not found on PATH. Install with `npm i -g @higgsfield/cli` and run `higgsfield auth login`.",
            binary.display()
        ))
    } else {
        GenVizError::Io(err)
    }
}

fn map_cli_failure(context: &str, stderr: &str) -> GenVizError {
    let lower = stderr.to_lowercase();
    if lower.contains("unauthorized") || lower.contains("not logged in") || lower.contains("auth") {
        return GenVizError::Auth(format!("{context}: {stderr}"));
    }
    if lower.contains("insufficient credits") || lower.contains("not enough credits") {
        return GenVizError::Billing(format!("{context}: {stderr}"));
    }
    if lower.contains("rate limit") {
        return GenVizError::RateLimited { retry_after: None };
    }
    GenVizError::Api {
        status: 0,
        message: format!("{context}: {stderr}"),
    }
}

/// Format a duration as a compact string the higgsfield CLI accepts ("20m", "30s").
fn format_duration_compact(d: Duration) -> String {
    let secs = d.as_secs();
    if secs % 60 == 0 && secs >= 60 {
        format!("{}m", secs / 60)
    } else {
        format!("{}s", secs)
    }
}

/// Download `result_url` into bytes.
pub(crate) async fn download(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| GenVizError::Network(e.to_string()))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(GenVizError::Api {
            status: status.as_u16(),
            message: format!("Failed to download {url}"),
        });
    }
    Ok(resp
        .bytes()
        .await
        .map_err(|e| GenVizError::Network(e.to_string()))?
        .to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_strings() {
        assert_eq!(HiggsfieldMode::ProductShot.as_str(), "product_shot");
        assert_eq!(HiggsfieldMode::Restyle.as_str(), "restyle");
        assert_eq!(
            HiggsfieldMode::VirtualModelTryout.as_str(),
            "virtual_model_tryout"
        );
    }

    #[test]
    fn test_parse_single_job() {
        let json = br#"{"id":"abc","status":"completed","result_url":"https://x/y.png","job_set_type":"gpt_image_2"}"#;
        let j = parse_job_stdout(json, "ctx").unwrap();
        assert_eq!(j.id, "abc");
        assert_eq!(j.result_url.as_deref(), Some("https://x/y.png"));
    }

    #[test]
    fn test_parse_array_job() {
        let json = br#"[{"id":"abc","status":"completed","result_url":"https://x/y.mp4","job_set_type":"seedance_2_0"}]"#;
        let j = parse_job_stdout(json, "ctx").unwrap();
        assert_eq!(j.id, "abc");
        assert_eq!(j.result_url.as_deref(), Some("https://x/y.mp4"));
    }

    #[test]
    fn test_parse_failed_job_errors() {
        let json = br#"{"id":"abc","status":"failed"}"#;
        let res = parse_job_stdout(json, "ctx");
        assert!(matches!(res, Err(GenVizError::Api { .. })));
    }

    #[test]
    fn test_parse_completed_no_url_errors() {
        let json = br#"{"id":"abc","status":"completed"}"#;
        let res = parse_job_stdout(json, "ctx");
        assert!(matches!(res, Err(GenVizError::UnexpectedResponse(_))));
    }

    #[test]
    fn test_parse_unparseable_errors() {
        let res = parse_job_stdout(b"not json", "ctx");
        assert!(matches!(res, Err(GenVizError::UnexpectedResponse(_))));
    }

    #[test]
    fn test_map_cli_failure_auth() {
        let e = map_cli_failure("ctx", "Error: unauthorized (401)");
        assert!(matches!(e, GenVizError::Auth(_)));
    }

    #[test]
    fn test_map_cli_failure_billing() {
        let e = map_cli_failure("ctx", "insufficient credits to run job");
        assert!(matches!(e, GenVizError::Billing(_)));
    }

    #[test]
    fn test_map_cli_failure_rate_limit() {
        let e = map_cli_failure("ctx", "rate limit exceeded");
        assert!(matches!(e, GenVizError::RateLimited { .. }));
    }

    #[test]
    fn test_format_duration_compact() {
        assert_eq!(format_duration_compact(Duration::from_secs(60)), "1m");
        assert_eq!(format_duration_compact(Duration::from_secs(900)), "15m");
        assert_eq!(format_duration_compact(Duration::from_secs(45)), "45s");
    }

    #[tokio::test]
    async fn test_missing_binary_returns_provider_not_available() {
        let cli = HiggsfieldCli::with_binary("/definitely/not/a/real/binary/higgsfield");
        let err = cli.version().await.unwrap_err();
        assert!(matches!(err, GenVizError::ProviderNotAvailable(_)));
    }
}
