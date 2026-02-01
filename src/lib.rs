//! GenViz - Unified AI media generation (image + video).
//!
//! This crate provides a unified interface for generating images and videos
//! using different AI providers.
//!
//! # Quick Start - Images
//!
//! ```no_run
//! use genviz::{GeminiProvider, GenerationRequest, ImageProvider};
//!
//! #[tokio::main]
//! async fn main() -> genviz::Result<()> {
//!     let provider = GeminiProvider::builder().build()?;
//!     let request = GenerationRequest::new("A golden retriever puppy");
//!     let image = provider.generate(&request).await?;
//!     image.save("puppy.jpg")?;
//!     Ok(())
//! }
//! ```
//!
//! # Quick Start - Videos
//!
//! ```no_run
//! use genviz::{GrokVideoProvider, VideoGenerationRequest, VideoProvider};
//!
//! #[tokio::main]
//! async fn main() -> genviz::Result<()> {
//!     let provider = GrokVideoProvider::builder().build()?;
//!     let request = VideoGenerationRequest::new("A cat playing with a ball")
//!         .with_duration(6);
//!     let video = provider.generate(&request).await?;
//!     video.save("cat.mp4")?;
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! ## Image Generation
//! - `flux-image`: Flux (Black Forest Labs)
//! - `gemini-image`: Gemini (Google)
//! - `grok-image`: Grok Imagine (xAI)
//!
//! ## Video Generation
//! - `grok-video`: Grok Imagine Video (xAI)
//! - `veo`: Veo (Google)
//!
//! ## Meta Features
//! - `image`: All image providers
//! - `video`: All video providers
//! - `cli`: Command-line interface

mod error;

#[cfg(feature = "image")]
pub mod image;

#[cfg(feature = "video")]
pub mod video;

#[cfg(feature = "cli")]
pub mod mcp;

// Re-export error types at crate root
pub use error::{GenVizError, Result};

// Re-export commonly used image types
#[cfg(feature = "image")]
pub use image::{
    AspectRatio, GeneratedImage, GenerationMetadata, GenerationRequest, ImageFormat, ImageProvider,
    ImageProviderExt, ImageProviderKind,
};

#[cfg(feature = "flux-image")]
pub use image::providers::{FluxModel, FluxProvider, FluxProviderBuilder};

#[cfg(feature = "gemini-image")]
pub use image::providers::{GeminiModel, GeminiProvider, GeminiProviderBuilder};

#[cfg(feature = "grok-image")]
pub use image::providers::{GrokModel, GrokProvider, GrokProviderBuilder};

// Re-export commonly used video types
#[cfg(feature = "video")]
pub use video::{
    GeneratedVideo, VideoGenerationRequest, VideoMetadata, VideoProvider, VideoProviderExt,
    VideoProviderKind,
};

#[cfg(feature = "grok-video")]
pub use video::providers::{GrokVideoModel, GrokVideoProvider, GrokVideoProviderBuilder};

#[cfg(feature = "veo")]
pub use video::providers::{VeoModel, VeoProvider, VeoProviderBuilder};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::error::{GenVizError, Result};

    #[cfg(feature = "image")]
    pub use crate::image::{GeneratedImage, GenerationRequest, ImageProvider, ImageProviderExt};

    #[cfg(feature = "video")]
    pub use crate::video::{
        GeneratedVideo, VideoGenerationRequest, VideoProvider, VideoProviderExt,
    };

    #[cfg(feature = "flux-image")]
    pub use crate::image::providers::FluxProvider;

    #[cfg(feature = "gemini-image")]
    pub use crate::image::providers::GeminiProvider;

    #[cfg(feature = "grok-image")]
    pub use crate::image::providers::GrokProvider;

    #[cfg(feature = "grok-video")]
    pub use crate::video::providers::GrokVideoProvider;

    #[cfg(feature = "veo")]
    pub use crate::video::providers::VeoProvider;
}
