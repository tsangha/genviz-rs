//! Video generation providers.

#[cfg(feature = "grok-video")]
mod grok;
#[cfg(feature = "veo")]
mod veo;

#[cfg(feature = "grok-video")]
pub use grok::{GrokVideoModel, GrokVideoProvider, GrokVideoProviderBuilder};

#[cfg(feature = "veo")]
pub use veo::{VeoModel, VeoProvider, VeoProviderBuilder};
