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
    Api {
        /// HTTP status code from the API response.
        status: u16,
        /// Error message from the API response.
        message: String,
    },

    /// Billing or quota error (HTTP 402, insufficient credits/quota).
    #[error("billing error: {0}")]
    Billing(String),

    /// Rate limit exceeded.
    #[error("rate limited{}", retry_after.map(|d| format!(", retry after {}s", d.as_secs())).unwrap_or_default())]
    RateLimited {
        /// Suggested delay before retrying, parsed from Retry-After header.
        retry_after: Option<Duration>,
    },

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
    Network(String),

    /// Failed to decode base64 data.
    #[error("failed to decode: {0}")]
    Decode(String),

    /// I/O error (e.g., saving file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(String),

    /// Provider not available (feature not enabled).
    #[error("provider not available: {0}")]
    ProviderNotAvailable(String),

    /// Unexpected response from the API (valid HTTP status but unexpected body shape).
    #[error("unexpected API response: {0}")]
    UnexpectedResponse(String),

    /// Video generation specific error.
    #[error("video generation failed: {0}")]
    VideoGeneration(String),
}

/// Maximum length for error messages before truncation.
const MAX_ERROR_LENGTH: usize = 500;

/// Truncates error messages that may contain base64 or other large payloads.
///
/// API errors sometimes include the full request body (e.g., base64 images),
/// which can bloat logs and session transcripts.
pub fn sanitize_error_message(text: &str) -> String {
    if text.len() > MAX_ERROR_LENGTH {
        format!(
            "{}... [truncated {} bytes]",
            &text[..MAX_ERROR_LENGTH],
            text.len() - MAX_ERROR_LENGTH
        )
    } else {
        text.to_string()
    }
}

impl From<reqwest::Error> for GenVizError {
    fn from(err: reqwest::Error) -> Self {
        // Sanitize the error message to strip any URL query parameters
        // that may contain API keys (e.g., ?key=... from Gemini/Veo).
        let msg = err.to_string();
        GenVizError::Network(sanitize_url_params(&msg))
    }
}

/// Strips query parameters from URLs embedded in error messages.
///
/// reqwest's Display includes the full URL. If the URL has `?key=...`
/// (e.g., Google APIs), this would leak the API key into logs.
fn sanitize_url_params(msg: &str) -> String {
    // Replace URL query strings: anything after '?' up to whitespace or end
    let mut result = String::with_capacity(msg.len());
    let mut chars = msg.chars().peekable();
    let mut in_url = false;

    while let Some(ch) = chars.next() {
        if ch == '?' && in_url {
            result.push_str("?[REDACTED]");
            // Skip until whitespace, quote, or end
            for next in chars.by_ref() {
                if next.is_whitespace() || next == '\'' || next == '"' || next == ')' {
                    result.push(next);
                    break;
                }
            }
            in_url = false;
        } else {
            // Detect URL start (http:// or https://)
            if ch == 'h' {
                let rest: String = std::iter::once(ch).chain(chars.clone().take(7)).collect();
                if rest.starts_with("https://") || rest.starts_with("http://") {
                    in_url = true;
                }
            }
            if ch.is_whitespace() {
                in_url = false;
            }
            result.push(ch);
        }
    }
    result
}

impl From<serde_json::Error> for GenVizError {
    fn from(err: serde_json::Error) -> Self {
        // Sanitize serde errors which may contain large payloads (e.g., base64 images)
        GenVizError::Json(sanitize_error_message(&err.to_string()))
    }
}

impl GenVizError {
    /// Returns true if this error is likely transient and worth retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. } | Self::Timeout(_) | Self::Network(_)
        ) || matches!(self, Self::Api { status, .. } if (500..=599).contains(status))
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

/// Extract Retry-After header value in seconds from an HTTP response.
pub fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
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
        assert!(GenVizError::Api {
            status: 500,
            message: "Internal".into()
        }
        .is_retryable());
        assert!(GenVizError::Api {
            status: 503,
            message: "Unavailable".into()
        }
        .is_retryable());

        assert!(!GenVizError::Auth("bad key".into()).is_retryable());
        assert!(!GenVizError::ContentBlocked("nsfw".into()).is_retryable());
        assert!(!GenVizError::UrlExpired.is_retryable());
        assert!(!GenVizError::Decode("bad base64".into()).is_retryable());
        assert!(!GenVizError::Api {
            status: 400,
            message: "Bad request".into()
        }
        .is_retryable());
        assert!(!GenVizError::Billing("quota".into()).is_retryable());
        assert!(!GenVizError::UnexpectedResponse("weird".into()).is_retryable());
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

        let err = GenVizError::RateLimited { retry_after: None };
        assert_eq!(err.to_string(), "rate limited");

        let err = GenVizError::RateLimited {
            retry_after: Some(Duration::from_secs(30)),
        };
        assert_eq!(err.to_string(), "rate limited, retry after 30s");

        let err = GenVizError::Billing("quota exceeded".into());
        assert_eq!(err.to_string(), "billing error: quota exceeded");

        let err = GenVizError::UnexpectedResponse("no image".into());
        assert_eq!(err.to_string(), "unexpected API response: no image");
    }

    #[test]
    fn test_sanitize_error_message() {
        // Short messages pass through unchanged
        let short = "Not found";
        assert_eq!(sanitize_error_message(short), "Not found");

        // Long messages (e.g., containing base64) get truncated
        let long = "x".repeat(1000);
        let result = sanitize_error_message(&long);
        assert!(result.len() < 600);
        assert!(result.contains("[truncated"));
        assert!(result.contains("500 bytes]"));
    }

    #[test]
    fn test_sanitize_url_params() {
        // URL with API key gets redacted
        let msg = "error sending request for url (https://api.example.com/v1/model?key=SECRET123)";
        let result = sanitize_url_params(msg);
        assert!(!result.contains("SECRET123"));
        assert!(result.contains("?[REDACTED]"));

        // URL without query params passes through
        let msg = "error sending request for url (https://api.example.com/v1/model)";
        let result = sanitize_url_params(msg);
        assert_eq!(result, msg);

        // Non-URL question marks are preserved
        let msg = "what happened? something broke";
        let result = sanitize_url_params(msg);
        assert_eq!(result, msg);
    }

    #[test]
    fn test_network_error_is_retryable() {
        let err = GenVizError::Network("connection refused".into());
        assert!(err.is_retryable());
        assert_eq!(err.retry_after(), Some(Duration::from_secs(2)));
    }
}
