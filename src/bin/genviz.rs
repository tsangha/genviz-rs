//! CLI for GenViz - AI media generation.

use clap::{Args, Parser, Subcommand, ValueEnum};
use genviz::image::{AspectRatio, GenerationRequest, ImageFormat, ImageProvider};
use genviz::video::{VideoGenerationRequest, VideoProvider};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "genviz")]
#[command(about = "Generate images and videos via AI APIs (Flux, Gemini, Grok, Veo)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate an image from a text prompt
    Image(ImageArgs),

    /// Generate a video from a text prompt
    Video(VideoArgs),

    /// List available providers
    Providers,

    /// Run as MCP server (for AI agent integration)
    Mcp,
}

#[derive(Args)]
struct ImageArgs {
    /// The text prompt describing the image
    prompt: String,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,

    /// Provider to use
    #[arg(short, long, value_enum, default_value = "gemini")]
    provider: ImageProviderArg,

    /// Image width in pixels
    #[arg(long)]
    width: Option<u32>,

    /// Image height in pixels
    #[arg(long)]
    height: Option<u32>,

    /// Seed for deterministic generation
    #[arg(long)]
    seed: Option<u64>,

    /// Aspect ratio (alternative to width/height)
    #[arg(long, value_enum)]
    aspect_ratio: Option<AspectRatioArg>,

    /// Input image for editing (path to image file)
    #[arg(short, long)]
    input: Option<PathBuf>,
}

#[derive(Args)]
struct VideoArgs {
    /// The text prompt describing the video
    prompt: String,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,

    /// Provider to use
    #[arg(short, long, value_enum, default_value = "grok")]
    provider: VideoProviderArg,

    /// Video duration in seconds (1-15 for Grok)
    #[arg(short, long)]
    duration: Option<u32>,

    /// Aspect ratio (e.g., 16:9)
    #[arg(long)]
    aspect_ratio: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageProviderArg {
    Flux,
    Gemini,
    Grok,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VideoProviderArg {
    Grok,
    Veo,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AspectRatioArg {
    #[value(name = "1:1")]
    Square,
    #[value(name = "16:9")]
    Landscape,
    #[value(name = "9:16")]
    Portrait,
    #[value(name = "4:3")]
    Standard,
    #[value(name = "3:4")]
    StandardPortrait,
    #[value(name = "21:9")]
    Ultrawide,
}

impl From<AspectRatioArg> for AspectRatio {
    fn from(arg: AspectRatioArg) -> Self {
        match arg {
            AspectRatioArg::Square => AspectRatio::Square,
            AspectRatioArg::Landscape => AspectRatio::Landscape,
            AspectRatioArg::Portrait => AspectRatio::Portrait,
            AspectRatioArg::Standard => AspectRatio::Standard,
            AspectRatioArg::StandardPortrait => AspectRatio::StandardPortrait,
            AspectRatioArg::Ultrawide => AspectRatio::Ultrawide,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Image(args) => {
            generate_image(args, cli.json).await?;
        }
        Commands::Video(args) => {
            generate_video(args, cli.json).await?;
        }
        Commands::Providers => {
            list_providers(cli.json)?;
        }
        Commands::Mcp => {
            run_mcp_server().await?;
        }
    }

    Ok(())
}

fn validate_image_args(args: &ImageArgs) -> anyhow::Result<()> {
    match args.provider {
        ImageProviderArg::Gemini => {
            if args.aspect_ratio.is_some() {
                anyhow::bail!("Gemini does not support --aspect-ratio");
            }
            if args.width.is_some() || args.height.is_some() {
                anyhow::bail!("Gemini does not support --width/--height");
            }
        }
        ImageProviderArg::Grok => {
            if args.width.is_some() || args.height.is_some() {
                anyhow::bail!(
                    "Grok does not support --width/--height (use --aspect-ratio instead)"
                );
            }
            if args.seed.is_some() {
                anyhow::bail!("Grok does not support --seed");
            }
            // Grok edit endpoint doesn't support aspect_ratio
            if args.input.is_some() && args.aspect_ratio.is_some() {
                anyhow::bail!("Grok does not support --aspect-ratio when editing images");
            }
        }
        ImageProviderArg::Flux => {
            // Flux supports all options
        }
    }
    Ok(())
}

async fn generate_image(args: ImageArgs, json_output: bool) -> anyhow::Result<()> {
    // Validate provider/flag compatibility before execution
    validate_image_args(&args)?;

    let mut request = GenerationRequest::new(&args.prompt);

    if let (Some(w), Some(h)) = (args.width, args.height) {
        request = request.with_size(w, h);
    }
    if let Some(s) = args.seed {
        request = request.with_seed(s);
    }
    if let Some(ar) = args.aspect_ratio {
        request = request.with_aspect_ratio(ar.into());
    }

    // Read input image for editing
    if let Some(ref input_path) = args.input {
        let input_data = std::fs::read(input_path)?;
        request = request.with_input_image(input_data);
    }

    if let Some(ext) = args.output.extension().and_then(|e| e.to_str()) {
        if let Some(format) = ImageFormat::from_extension(ext) {
            request = request.with_format(format);
        }
    }

    let image = match args.provider {
        ImageProviderArg::Flux => {
            #[cfg(feature = "flux-image")]
            {
                let provider = genviz::FluxProvider::builder().build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "flux-image"))]
            {
                anyhow::bail!("Flux provider not enabled");
            }
        }
        ImageProviderArg::Gemini => {
            #[cfg(feature = "gemini-image")]
            {
                let provider = genviz::GeminiProvider::builder().build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "gemini-image"))]
            {
                anyhow::bail!("Gemini provider not enabled");
            }
        }
        ImageProviderArg::Grok => {
            #[cfg(feature = "grok-image")]
            {
                let provider = genviz::GrokProvider::builder().build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "grok-image"))]
            {
                anyhow::bail!("Grok provider not enabled");
            }
        }
    };

    image.save(&args.output)?;

    if json_output {
        let result = serde_json::json!({
            "type": "image",
            "success": true,
            "output": args.output.display().to_string(),
            "size_bytes": image.size(),
            "format": image.format.extension(),
            "provider": image.provider.to_string(),
            "model": image.metadata.model,
            "duration_ms": image.metadata.duration_ms,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Generated image: {} ({} bytes) via {}",
            args.output.display(),
            image.size(),
            image.provider
        );
        if let Some(duration) = image.metadata.duration_ms {
            println!("Duration: {}ms", duration);
        }
    }

    Ok(())
}

async fn generate_video(args: VideoArgs, json_output: bool) -> anyhow::Result<()> {
    let mut request = VideoGenerationRequest::new(&args.prompt);

    if let Some(d) = args.duration {
        request = request.with_duration(d);
    }
    if let Some(ar) = args.aspect_ratio {
        request = request.with_aspect_ratio(ar);
    }

    let video = match args.provider {
        VideoProviderArg::Grok => {
            #[cfg(feature = "grok-video")]
            {
                let provider = genviz::GrokVideoProvider::builder().build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "grok-video"))]
            {
                anyhow::bail!("Grok video provider not enabled");
            }
        }
        VideoProviderArg::Veo => {
            #[cfg(feature = "veo")]
            {
                let provider = genviz::VeoProvider::builder().build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "veo"))]
            {
                anyhow::bail!("Veo provider not enabled");
            }
        }
    };

    video.save(&args.output)?;

    if json_output {
        let result = serde_json::json!({
            "type": "video",
            "success": true,
            "output": args.output.display().to_string(),
            "size_bytes": video.size(),
            "provider": video.provider.to_string(),
            "model": video.metadata.model,
            "duration_ms": video.metadata.duration_ms,
            "video_duration_secs": video.metadata.video_duration_secs,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "Generated video: {} ({} bytes) via {}",
            args.output.display(),
            video.size(),
            video.provider
        );
        if let Some(duration) = video.metadata.duration_ms {
            println!("Generation time: {}ms", duration);
        }
    }

    Ok(())
}

async fn run_mcp_server() -> anyhow::Result<()> {
    eprintln!("[genviz-mcp] Starting MCP server...");
    let mut server = genviz::mcp::McpServer::new();
    server.run().await?;
    Ok(())
}

fn list_providers(json_output: bool) -> anyhow::Result<()> {
    #[derive(serde::Serialize)]
    struct ProviderInfo {
        name: &'static str,
        kind: &'static str,
        media_type: &'static str,
        env_var: &'static str,
        enabled: bool,
    }

    let providers = vec![
        // Image providers
        ProviderInfo {
            name: "Flux (Black Forest Labs)",
            kind: "flux",
            media_type: "image",
            env_var: "BFL_API_KEY",
            enabled: cfg!(feature = "flux-image"),
        },
        ProviderInfo {
            name: "Gemini (Google)",
            kind: "gemini",
            media_type: "image",
            env_var: "GOOGLE_API_KEY",
            enabled: cfg!(feature = "gemini-image"),
        },
        ProviderInfo {
            name: "Grok Imagine (xAI)",
            kind: "grok",
            media_type: "image",
            env_var: "XAI_API_KEY",
            enabled: cfg!(feature = "grok-image"),
        },
        // Video providers
        ProviderInfo {
            name: "Grok Imagine Video (xAI)",
            kind: "grok",
            media_type: "video",
            env_var: "XAI_API_KEY",
            enabled: cfg!(feature = "grok-video"),
        },
        ProviderInfo {
            name: "Veo (Google)",
            kind: "veo",
            media_type: "video",
            env_var: "GOOGLE_API_KEY",
            enabled: cfg!(feature = "veo"),
        },
    ];

    if json_output {
        println!("{}", serde_json::to_string_pretty(&providers)?);
    } else {
        println!("Available providers:\n");
        println!("IMAGE:");
        for p in providers.iter().filter(|p| p.media_type == "image") {
            let status = if p.enabled { "✓" } else { "✗" };
            println!("  {} {} ({})", status, p.name, p.kind);
            println!("    API key: {}", p.env_var);
        }
        println!("\nVIDEO:");
        for p in providers.iter().filter(|p| p.media_type == "video") {
            let status = if p.enabled { "✓" } else { "✗" };
            println!("  {} {} ({})", status, p.name, p.kind);
            println!("    API key: {}", p.env_var);
        }
    }

    Ok(())
}
