//! Video provider trait and utilities.

use crate::error::Result;
use crate::video::types::{GeneratedVideo, VideoGenerationRequest, VideoProviderKind};
use async_trait::async_trait;

/// Trait for video generation providers.
#[async_trait]
pub trait VideoProvider: Send + Sync {
    /// Generates a video from the given request.
    async fn generate(&self, request: &VideoGenerationRequest) -> Result<GeneratedVideo>;

    /// Returns the kind of this provider.
    fn kind(&self) -> VideoProviderKind;

    /// Returns the name of this provider for display.
    fn name(&self) -> &str {
        match self.kind() {
            VideoProviderKind::Grok => "Grok Imagine Video (xAI)",
            VideoProviderKind::Veo => "Veo (Google)",
        }
    }

    /// Checks if the provider is reachable and authenticated.
    async fn health_check(&self) -> Result<()>;
}

/// Extension trait for providers with retry logic.
#[async_trait]
pub trait VideoProviderExt: VideoProvider {
    /// Generates with automatic retries on transient failures.
    async fn generate_with_retries(
        &self,
        request: &VideoGenerationRequest,
        max_retries: u32,
    ) -> Result<GeneratedVideo> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match self.generate(request).await {
                Ok(video) => return Ok(video),
                Err(e) if e.is_retryable() && attempt < max_retries => {
                    let delay = e.retry_after().unwrap_or(std::time::Duration::from_secs(1));
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries,
                        delay_ms = delay.as_millis(),
                        "retrying after transient error: {e}"
                    );
                    tokio::time::sleep(delay).await;
                    last_error = Some(e);
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_error.expect("should have error after retries"))
    }
}

impl<T: VideoProvider> VideoProviderExt for T {}
