//! Demonstrates using the trait interface to swap providers.
//!
//! Run with: `cargo run --example multi_provider`
//!
//! Requires at least one of: `GOOGLE_API_KEY`, `BFL_API_KEY`,
//! `XAI_API_KEY`, or `OPENAI_API_KEY`.

use genviz::{GenerationRequest, ImageProvider};

#[tokio::main]
async fn main() -> genviz::Result<()> {
    // Try to build whichever provider has a key available
    let provider: Box<dyn ImageProvider> = if std::env::var("GOOGLE_API_KEY").is_ok() {
        println!("Using Gemini provider");
        Box::new(genviz::GeminiProvider::builder().build()?)
    } else if std::env::var("BFL_API_KEY").is_ok() {
        println!("Using Flux provider");
        Box::new(genviz::FluxProvider::builder().build()?)
    } else if std::env::var("XAI_API_KEY").is_ok() {
        println!("Using Grok provider");
        Box::new(genviz::GrokProvider::builder().build()?)
    } else if std::env::var("OPENAI_API_KEY").is_ok() {
        println!("Using OpenAI provider");
        Box::new(genviz::OpenAiImageProvider::builder().build()?)
    } else {
        eprintln!("Set at least one API key environment variable.");
        std::process::exit(1);
    };

    println!("Provider: {} ({:?})", provider.name(), provider.kind());

    let request = GenerationRequest::new("A serene mountain lake at dawn");
    let image = provider.generate(&request).await?;

    let filename = format!("{:?}_output.png", provider.kind());
    image.save(&filename)?;
    println!("Saved to {} ({} bytes)", filename, image.size());

    Ok(())
}
