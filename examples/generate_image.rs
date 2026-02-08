//! Basic image generation example.
//!
//! Run with: `cargo run --example generate_image`
//!
//! Requires `GOOGLE_API_KEY` environment variable.

use genviz::{GeminiProvider, GenerationRequest, ImageProvider};

#[tokio::main]
async fn main() -> genviz::Result<()> {
    let provider = GeminiProvider::builder().build()?;

    let request = GenerationRequest::new("A golden retriever puppy playing in snow");
    let image = provider.generate(&request).await?;

    image.save("output.png")?;
    println!(
        "Generated image: {} bytes, format: {:?}",
        image.size(),
        image.format
    );

    Ok(())
}
