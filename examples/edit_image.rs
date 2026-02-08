//! Image editing example - modifies an existing image with a text prompt.
//!
//! Run with: `cargo run --example edit_image -- <input_image.png>`
//!
//! Requires `GOOGLE_API_KEY` environment variable.

use genviz::{GeminiProvider, GenerationRequest, ImageProvider};

#[tokio::main]
async fn main() -> genviz::Result<()> {
    let input_path = std::env::args()
        .nth(1)
        .expect("Usage: edit_image <input_image.png>");

    let input_bytes = std::fs::read(&input_path)?;

    let provider = GeminiProvider::builder().build()?;

    let request = GenerationRequest::new("Make the colors more vibrant and add a warm sunset glow")
        .with_input_image(input_bytes);

    let image = provider.generate(&request).await?;
    image.save("edited.png")?;
    println!("Edited image saved to edited.png ({} bytes)", image.size());

    Ok(())
}
