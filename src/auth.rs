//! Shared authentication utilities for Google Cloud backends.
//!
//! Used by providers that support Vertex AI (e.g., Gemini images, Veo video).

use crate::error::{GenVizError, Result};

/// Get a bearer token by running `gcloud auth print-access-token`.
///
/// Requires the `gcloud` CLI to be installed and authenticated.
/// Install from <https://cloud.google.com/sdk/docs/install>.
pub fn gcloud_access_token() -> Result<String> {
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

/// Parse a `gs://bucket/path/to/object` URI into (bucket, object_path).
#[cfg(feature = "cli")]
pub fn parse_gcs_uri(uri: &str) -> Result<(&str, &str)> {
    let path = uri
        .strip_prefix("gs://")
        .ok_or_else(|| GenVizError::InvalidRequest(format!("Invalid GCS URI: {}", uri)))?;
    let (bucket, object) = path
        .split_once('/')
        .ok_or_else(|| GenVizError::InvalidRequest(format!("GCS URI missing path: {}", uri)))?;
    Ok((bucket, object))
}
