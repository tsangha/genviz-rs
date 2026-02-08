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
