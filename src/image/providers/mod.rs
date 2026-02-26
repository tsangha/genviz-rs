//! Image generation providers.

#[cfg(feature = "flux-image")]
mod flux;
#[cfg(feature = "gemini-image")]
pub(crate) mod gemini;
#[cfg(feature = "grok-image")]
mod grok;
#[cfg(feature = "openai-image")]
mod openai;

#[cfg(feature = "flux-image")]
pub use flux::{FluxModel, FluxProvider, FluxProviderBuilder};

#[cfg(feature = "gemini-image")]
pub use gemini::{GeminiBackend, GeminiModel, GeminiProvider, GeminiProviderBuilder};

#[cfg(all(feature = "gemini-image", feature = "cli"))]
pub use gemini::{
    download_batch_results, get_batch_status, submit_batch, BatchImageResult, BatchJobStatus,
    BatchSubmitResult,
};

#[cfg(feature = "grok-image")]
pub use grok::{GrokModel, GrokProvider, GrokProviderBuilder};

#[cfg(feature = "openai-image")]
pub use openai::{OpenAiImageModel, OpenAiImageProvider, OpenAiImageProviderBuilder};

#[cfg(feature = "kling-image")]
mod kling;
#[cfg(feature = "kling-image")]
pub use kling::{KlingImageModel, KlingImageProvider, KlingImageProviderBuilder};

#[cfg(feature = "fal-image")]
mod fal;
#[cfg(feature = "fal-image")]
pub use fal::{FalImageModel, FalImageProvider, FalImageProviderBuilder};

#[cfg(feature = "modelslab-image")]
mod modelslab;
#[cfg(feature = "modelslab-image")]
pub use modelslab::{ModelsLabImageModel, ModelsLabImageProvider, ModelsLabImageProviderBuilder};
