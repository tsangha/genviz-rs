//! Image generation providers.

#[cfg(feature = "flux-image")]
mod flux;
#[cfg(feature = "gemini-image")]
mod gemini;
#[cfg(feature = "grok-image")]
mod grok;
#[cfg(feature = "openai-image")]
mod openai;

#[cfg(feature = "flux-image")]
pub use flux::{FluxModel, FluxProvider, FluxProviderBuilder};

#[cfg(feature = "gemini-image")]
pub use gemini::{GeminiModel, GeminiProvider, GeminiProviderBuilder};

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
