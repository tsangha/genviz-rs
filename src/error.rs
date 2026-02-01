//! Error types for media generation.

use std::time::Duration;

/// Errors that can occur during media generation.
#[derive(Debug, thiserror::Error)]
pub enum GenVizError {
    /// API key missing or invalid.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// API returned an error response.
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },

    /// Rate limit exceeded.
    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Option<Duration> },

    /// Operation timed out (e.g., Flux/video polling).
    #[error("operation timed out after {0:?}")]
    Timeout(Duration),

    /// Download URL expired before download.
    #[error("download URL expired")]
    UrlExpired,

    /// Content was blocked by safety filters.
    #[error("content blocked: {0}")]
    ContentBlocked(String),

    /// Invalid request parameters.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Network or HTTP error.
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Failed to decode base64 data.
    #[error("failed to decode: {0}")]
    Decode(String),

    /// I/O error (e.g., saving file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Provider not available (feature not enabled).
    #[error("provider not available: {0}")]
    ProviderNotAvailable(String),

    /// Video generation specific error.
    #[error("video generation failed: {0}")]
    VideoGeneration(String),
}

impl GenVizError {
    /// Returns true if this error is likely transient and worth retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. } | Self::Timeout(_) | Self::Network(_)
        )
    }

    /// Returns the suggested retry delay, if available.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after } => *retry_after,
            Self::Timeout(_) => Some(Duration::from_secs(1)),
            Self::Network(_) => Some(Duration::from_secs(2)),
            _ => None,
        }
    }
}

/// Result type alias for media generation operations.
pub type Result<T> = std::result::Result<T, GenVizError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_retryable() {
        assert!(GenVizError::RateLimited { retry_after: None }.is_retryable());
        assert!(GenVizError::Timeout(Duration::from_secs(30)).is_retryable());

        assert!(!GenVizError::Auth("bad key".into()).is_retryable());
        assert!(!GenVizError::ContentBlocked("nsfw".into()).is_retryable());
        assert!(!GenVizError::UrlExpired.is_retryable());
        assert!(!GenVizError::Decode("bad base64".into()).is_retryable());
    }

    #[test]
    fn test_retry_after() {
        let rate_limited = GenVizError::RateLimited {
            retry_after: Some(Duration::from_secs(60)),
        };
        assert_eq!(rate_limited.retry_after(), Some(Duration::from_secs(60)));

        let rate_limited_no_hint = GenVizError::RateLimited { retry_after: None };
        assert_eq!(rate_limited_no_hint.retry_after(), None);

        let timeout = GenVizError::Timeout(Duration::from_secs(30));
        assert_eq!(timeout.retry_after(), Some(Duration::from_secs(1)));

        let auth = GenVizError::Auth("bad".into());
        assert_eq!(auth.retry_after(), None);
    }

    #[test]
    fn test_error_display() {
        let err = GenVizError::Api {
            status: 404,
            message: "Not found".into(),
        };
        assert_eq!(err.to_string(), "API error: 404 - Not found");

        let err = GenVizError::ContentBlocked("Safety filter triggered".into());
        assert_eq!(err.to_string(), "content blocked: Safety filter triggered");
    }
}
