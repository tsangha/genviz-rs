//! Basic video generation example.
//!
//! Run with: `cargo run --example generate_video`
//!
//! Requires `XAI_API_KEY` environment variable.

use genviz::{GrokVideoProvider, VideoGenerationRequest, VideoProvider};

#[tokio::main]
async fn main() -> genviz::Result<()> {
    let provider = GrokVideoProvider::builder().build()?;

    let request = VideoGenerationRequest::new("Ocean waves crashing on a rocky shore at sunset")
        .with_duration(5);

    println!("Generating video (this may take a few minutes)...");
    let video = provider.generate(&request).await?;

    video.save("output.mp4")?;
    println!(
        "Generated video: {} bytes, duration: {:?}",
        video.size(),
        video.metadata.video_duration_secs
    );

    Ok(())
}
