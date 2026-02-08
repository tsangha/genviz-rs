//! Image provider trait and utilities.

use crate::error::Result;
use crate::image::types::{GeneratedImage, GenerationRequest, ImageProviderKind};
use async_trait::async_trait;

/// Trait for image generation providers.
#[async_trait]
pub trait ImageProvider: Send + Sync {
    /// Generates an image from the given request.
    async fn generate(&self, request: &GenerationRequest) -> Result<GeneratedImage>;

    /// Returns the kind of this provider.
    fn kind(&self) -> ImageProviderKind;

    /// Returns the name of this provider for display.
    fn name(&self) -> &str {
        match self.kind() {
            ImageProviderKind::Flux => "Flux (Black Forest Labs)",
            ImageProviderKind::Gemini => "Gemini (Google)",
            ImageProviderKind::Grok => "Grok Imagine (xAI)",
            ImageProviderKind::OpenAI => "OpenAI (gpt-image)",
            ImageProviderKind::Kling => "Kling AI (Kuaishou)",
            ImageProviderKind::Fal => "fal.ai",
        }
    }

    /// Checks if the provider is reachable and authenticated.
    async fn health_check(&self) -> Result<()>;
}

/// Extension trait for providers with retry logic.
#[async_trait]
pub trait ImageProviderExt: ImageProvider {
    /// Generates with automatic retries on transient failures.
    async fn generate_with_retries(
        &self,
        request: &GenerationRequest,
        max_retries: u32,
    ) -> Result<GeneratedImage> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match self.generate(request).await {
                Ok(image) => return Ok(image),
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

impl<T: ImageProvider> ImageProviderExt for T {}
