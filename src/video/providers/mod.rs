//! Video generation providers.

#[cfg(feature = "grok-video")]
mod grok;
#[cfg(feature = "openai-video")]
mod openai;
#[cfg(feature = "veo")]
mod veo;

#[cfg(feature = "grok-video")]
pub use grok::{GrokVideoModel, GrokVideoProvider, GrokVideoProviderBuilder};

#[cfg(feature = "openai-video")]
pub use openai::{SoraModel, SoraProvider, SoraProviderBuilder};

#[cfg(feature = "veo")]
pub use veo::{VeoModel, VeoProvider, VeoProviderBuilder};

#[cfg(feature = "kling-video")]
mod kling;
#[cfg(feature = "kling-video")]
pub use kling::{KlingVideoModel, KlingVideoProvider, KlingVideoProviderBuilder};

#[cfg(feature = "fal-video")]
mod fal;
#[cfg(feature = "fal-video")]
pub use fal::{FalVideoModel, FalVideoProvider, FalVideoProviderBuilder};
