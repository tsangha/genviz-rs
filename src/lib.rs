#![warn(missing_docs)]
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
//! - `openai-image`: OpenAI (gpt-image-1, dall-e-3)
//! - `fal-image`: fal.ai (Flux, Recraft, Ideogram, HiDream)
//!
//! ## Video Generation
//! - `grok-video`: Grok Imagine Video (xAI)
//! - `openai-video`: Sora (OpenAI)
//! - `veo`: Veo (Google)
//! - `kling-video`: Kling AI (Kuaishou)
//! - `fal-video`: fal.ai (Wan, Hailuo 2.3, Seedance, LTX Video, Kling)
//! - `minimax-video`: MiniMax Hailuo (direct API, subject reference)
//!
//! ## Meta Features
//! - `image`: All image providers
//! - `video`: All video providers
//! - `cli`: Command-line interface

mod error;

#[cfg(any(
    feature = "image",
    feature = "flux-image",
    feature = "gemini-image",
    feature = "grok-image",
    feature = "openai-image",
    feature = "kling-image",
    feature = "fal-image"
))]
pub mod image;

#[cfg(any(
    feature = "video",
    feature = "grok-video",
    feature = "veo",
    feature = "openai-video",
    feature = "kling-video",
    feature = "fal-video",
    feature = "minimax-video"
))]
pub mod video;

#[cfg(feature = "cli")]
#[doc(hidden)]
pub mod mcp;

// Re-export error types at crate root
pub use error::{GenVizError, Result};

// Re-export commonly used image types
#[cfg(any(
    feature = "image",
    feature = "flux-image",
    feature = "gemini-image",
    feature = "grok-image",
    feature = "openai-image",
    feature = "kling-image",
    feature = "fal-image"
))]
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

#[cfg(feature = "openai-image")]
pub use image::providers::{OpenAiImageModel, OpenAiImageProvider, OpenAiImageProviderBuilder};

#[cfg(feature = "kling-image")]
pub use image::providers::{KlingImageModel, KlingImageProvider, KlingImageProviderBuilder};

#[cfg(feature = "fal-image")]
pub use image::providers::{FalImageModel, FalImageProvider, FalImageProviderBuilder};

// Re-export commonly used video types
#[cfg(any(
    feature = "video",
    feature = "grok-video",
    feature = "veo",
    feature = "openai-video",
    feature = "kling-video",
    feature = "fal-video",
    feature = "minimax-video"
))]
pub use video::{
    GeneratedVideo, SubjectReference, VideoGenerationRequest, VideoMetadata, VideoProvider,
    VideoProviderExt, VideoProviderKind,
};

#[cfg(feature = "grok-video")]
pub use video::providers::{GrokVideoModel, GrokVideoProvider, GrokVideoProviderBuilder};

#[cfg(feature = "openai-video")]
pub use video::providers::{SoraModel, SoraProvider, SoraProviderBuilder};

#[cfg(feature = "veo")]
pub use video::providers::{VeoBackend, VeoModel, VeoProvider, VeoProviderBuilder};

#[cfg(feature = "kling-video")]
pub use video::providers::{KlingVideoModel, KlingVideoProvider, KlingVideoProviderBuilder};

#[cfg(feature = "fal-video")]
pub use video::providers::{FalVideoModel, FalVideoProvider, FalVideoProviderBuilder};

#[cfg(feature = "minimax-video")]
pub use video::providers::{MiniMaxVideoModel, MiniMaxVideoProvider, MiniMaxVideoProviderBuilder};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::error::{GenVizError, Result};

    #[cfg(any(
        feature = "image",
        feature = "flux-image",
        feature = "gemini-image",
        feature = "grok-image",
        feature = "openai-image",
        feature = "kling-image",
        feature = "fal-image"
    ))]
    pub use crate::image::{GeneratedImage, GenerationRequest, ImageProvider, ImageProviderExt};

    #[cfg(any(
        feature = "video",
        feature = "grok-video",
        feature = "veo",
        feature = "openai-video",
        feature = "kling-video",
        feature = "fal-video",
        feature = "minimax-video"
    ))]
    pub use crate::video::{
        GeneratedVideo, VideoGenerationRequest, VideoProvider, VideoProviderExt,
    };

    #[cfg(feature = "flux-image")]
    pub use crate::image::providers::FluxProvider;

    #[cfg(feature = "gemini-image")]
    pub use crate::image::providers::GeminiProvider;

    #[cfg(feature = "grok-image")]
    pub use crate::image::providers::GrokProvider;

    #[cfg(feature = "openai-image")]
    pub use crate::image::providers::OpenAiImageProvider;

    #[cfg(feature = "kling-image")]
    pub use crate::image::providers::KlingImageProvider;

    #[cfg(feature = "fal-image")]
    pub use crate::image::providers::FalImageProvider;

    #[cfg(feature = "grok-video")]
    pub use crate::video::providers::GrokVideoProvider;

    #[cfg(feature = "openai-video")]
    pub use crate::video::providers::SoraProvider;

    #[cfg(feature = "veo")]
    pub use crate::video::providers::VeoProvider;

    #[cfg(feature = "kling-video")]
    pub use crate::video::providers::KlingVideoProvider;

    #[cfg(feature = "fal-video")]
    pub use crate::video::providers::FalVideoProvider;

    #[cfg(feature = "minimax-video")]
    pub use crate::video::providers::MiniMaxVideoProvider;
}
