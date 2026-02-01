//! Video generation module.

mod provider;
pub mod providers;
mod types;

pub use provider::{VideoProvider, VideoProviderExt};
pub use types::{GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProviderKind};
