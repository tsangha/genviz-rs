//! Higgsfield provider — wraps the local `higgsfield` CLI.
//!
//! Higgsfield is a creative-AI workflow service that bundles many third-party
//! models (GPT Image 2, Nano Banana Pro, Seedance 2.0, Veo 3.1, Kling 3.0,
//! Soul V2, Wan 2.7, Happy-Horse-equivalent, etc.) behind a single API with
//! brand-aware prompt enhancement, marketing-studio templates, marketplace
//! card workflows, and trainable Soul Character identities.
//!
//! Instead of reimplementing Higgsfield's HTTP layer, this module shells out
//! to the official `higgsfield` CLI (must be on `$PATH`). Auth is whatever the
//! CLI is logged into (`higgsfield auth login`).

pub mod cli;
pub mod image;
pub mod video;

pub use cli::{HiggsfieldCli, HiggsfieldJob, HiggsfieldMode, HiggsfieldSurface};
pub use image::{HiggsfieldImageModel, HiggsfieldImageProvider, HiggsfieldImageProviderBuilder};
pub use video::{HiggsfieldVideoModel, HiggsfieldVideoProvider, HiggsfieldVideoProviderBuilder};
