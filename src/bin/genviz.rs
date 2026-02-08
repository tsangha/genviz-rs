//! CLI for GenViz - AI media generation.

use clap::{Args, Parser, Subcommand, ValueEnum};
use genviz::image::{AspectRatio, GenerationRequest, ImageFormat, ImageProvider};
use genviz::video::{VideoGenerationRequest, VideoProvider};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "genviz")]
#[command(
    about = "Generate images and videos via AI APIs (Flux, Gemini, Grok, OpenAI, Kling, fal.ai, Veo, Sora)"
)]
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

    /// Model variant (flux: flux-pro-1.1, flux-pro-1.1-ultra, flux-pro, flux-dev, flux-2-max, flux-2-pro, flux-2-flex, flux-2-klein-4b, flux-2-klein-9b, flux-kontext-pro, flux-kontext-max, flux-fill-pro, flux-expand-pro; gemini: nano-banana, nano-banana-pro; grok: grok-imagine; openai: gpt-image-1, dall-e-3; kling: kling-v1, kling-v1.5, kling-v2; fal: flux-schnell, flux-pro, flux-pro-ultra, recraft-v3, ideogram-v3, hidream)
    #[arg(long)]
    model: Option<String>,
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

    /// Source image URL for image-to-video generation
    #[arg(long)]
    source_image_url: Option<String>,

    /// Model variant (grok: grok-imagine-video; openai: sora-2; veo: veo-3.1-generate-preview; fal: wan-2.1, wan-2.1-i2v, minimax, ltx-video)
    #[arg(long)]
    model: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageProviderArg {
    Flux,
    Gemini,
    Grok,
    Openai,
    Kling,
    Fal,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VideoProviderArg {
    Grok,
    Openai,
    Veo,
    Kling,
    Fal,
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
    // Initialize tracing subscriber (respects RUST_LOG, defaults to warn)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

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
        ImageProviderArg::Openai => {
            if args.seed.is_some() {
                anyhow::bail!("OpenAI does not support --seed");
            }
        }
        ImageProviderArg::Flux => {
            // Flux supports all options
        }
        ImageProviderArg::Kling => {
            if args.width.is_some() || args.height.is_some() {
                anyhow::bail!(
                    "Kling does not support --width/--height (use --aspect-ratio instead)"
                );
            }
            if args.seed.is_some() {
                anyhow::bail!("Kling does not support --seed");
            }
        }
        ImageProviderArg::Fal => {
            // fal.ai supports all options
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

    if !json_output {
        let action = if args.input.is_some() {
            "Editing"
        } else {
            "Generating"
        };
        eprint!("{} image via {:?}... ", action, args.provider);
    }

    let image = match args.provider {
        ImageProviderArg::Flux => {
            #[cfg(feature = "flux-image")]
            {
                let mut builder = genviz::FluxProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "flux-pro-1.1" => genviz::FluxModel::FluxPro11,
                        "flux-pro-1.1-ultra" => genviz::FluxModel::FluxPro11Ultra,
                        "flux-pro" => genviz::FluxModel::FluxPro,
                        "flux-dev" => genviz::FluxModel::FluxDev,
                        "flux-2-max" | "flux2-max" => genviz::FluxModel::Flux2Max,
                        "flux-2-pro" | "flux2-pro" => genviz::FluxModel::Flux2Pro,
                        "flux-2-flex" | "flux2-flex" => genviz::FluxModel::Flux2Flex,
                        "flux-2-klein-4b" | "flux2-klein-4b" => genviz::FluxModel::Flux2Klein4B,
                        "flux-2-klein-9b" | "flux2-klein-9b" => genviz::FluxModel::Flux2Klein9B,
                        "flux-kontext-pro" | "kontext-pro" => genviz::FluxModel::KontextPro,
                        "flux-kontext-max" | "kontext-max" => genviz::FluxModel::KontextMax,
                        "flux-fill-pro" | "fill-pro" => genviz::FluxModel::FillPro,
                        "flux-expand-pro" | "expand-pro" => genviz::FluxModel::ExpandPro,
                        _ => anyhow::bail!(
                            "Unknown Flux model: {}. Options: flux-pro-1.1, flux-pro-1.1-ultra, flux-pro, flux-dev, flux-2-max, flux-2-pro, flux-2-flex, flux-2-klein-4b, flux-2-klein-9b, flux-kontext-pro, flux-kontext-max, flux-fill-pro, flux-expand-pro",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
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
                let mut builder = genviz::GeminiProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "nano-banana" => genviz::GeminiModel::NanoBanana,
                        "nano-banana-pro" => genviz::GeminiModel::NanoBananaPro,
                        _ => anyhow::bail!(
                            "Unknown Gemini model: {}. Options: nano-banana, nano-banana-pro",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
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
                let mut builder = genviz::GrokProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "grok-imagine" => genviz::GrokModel::GrokImagine,
                        _ => anyhow::bail!("Unknown Grok model: {}. Options: grok-imagine", m),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "grok-image"))]
            {
                anyhow::bail!("Grok provider not enabled");
            }
        }
        ImageProviderArg::Openai => {
            #[cfg(feature = "openai-image")]
            {
                let mut builder = genviz::OpenAiImageProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "gpt-image-1" => genviz::OpenAiImageModel::GptImage1,
                        "dall-e-3" => genviz::OpenAiImageModel::DallE3,
                        _ => anyhow::bail!(
                            "Unknown OpenAI model: {}. Options: gpt-image-1, dall-e-3",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "openai-image"))]
            {
                anyhow::bail!("OpenAI image provider not enabled");
            }
        }
        ImageProviderArg::Kling => {
            #[cfg(feature = "kling-image")]
            {
                let mut builder = genviz::KlingImageProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "kling-v1" => genviz::KlingImageModel::KlingV1,
                        "kling-v1.5" | "kling-v1-5" => genviz::KlingImageModel::KlingV1_5,
                        "kling-v2" => genviz::KlingImageModel::KlingV2,
                        _ => anyhow::bail!(
                            "Unknown Kling model: {}. Options: kling-v1, kling-v1.5, kling-v2",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "kling-image"))]
            {
                anyhow::bail!("Kling image provider not enabled");
            }
        }
        ImageProviderArg::Fal => {
            #[cfg(feature = "fal-image")]
            {
                let mut builder = genviz::FalImageProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "flux-schnell" => genviz::FalImageModel::FluxSchnell,
                        "flux-pro" => genviz::FalImageModel::FluxPro,
                        "flux-pro-ultra" => genviz::FalImageModel::FluxProUltra,
                        "recraft-v3" => genviz::FalImageModel::RecraftV3,
                        "ideogram-v3" => genviz::FalImageModel::Ideogram3,
                        "hidream" => genviz::FalImageModel::HiDream,
                        s if s.starts_with("fal-ai/") => {
                            genviz::FalImageModel::Custom(s.to_string())
                        }
                        _ => anyhow::bail!(
                            "Unknown fal.ai model: {}. Options: flux-schnell, flux-pro, flux-pro-ultra, recraft-v3, ideogram-v3, hidream, or fal-ai/...",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "fal-image"))]
            {
                anyhow::bail!("fal.ai image provider not enabled");
            }
        }
    };

    if !json_output {
        eprintln!("done.");
    }

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
    if let Some(url) = args.source_image_url {
        request = request.with_source_image(url);
    }

    if !json_output {
        eprint!(
            "Generating video via {:?} (this may take a few minutes)... ",
            args.provider
        );
    }

    let video = match args.provider {
        VideoProviderArg::Grok => {
            #[cfg(feature = "grok-video")]
            {
                let mut builder = genviz::GrokVideoProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "grok-imagine-video" => genviz::GrokVideoModel::GrokImagineVideo,
                        _ => anyhow::bail!(
                            "Unknown Grok video model: {}. Options: grok-imagine-video",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "grok-video"))]
            {
                anyhow::bail!("Grok video provider not enabled");
            }
        }
        VideoProviderArg::Openai => {
            #[cfg(feature = "openai-video")]
            {
                let mut builder = genviz::SoraProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "sora-2" => genviz::SoraModel::Sora2,
                        _ => anyhow::bail!("Unknown Sora model: {}. Options: sora-2", m),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "openai-video"))]
            {
                anyhow::bail!("OpenAI video (Sora) provider not enabled");
            }
        }
        VideoProviderArg::Veo => {
            #[cfg(feature = "veo")]
            {
                let mut builder = genviz::VeoProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "veo-3.1-generate-preview" => genviz::VeoModel::Veo31Preview,
                        _ => anyhow::bail!(
                            "Unknown Veo model: {}. Options: veo-3.1-generate-preview",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "veo"))]
            {
                anyhow::bail!("Veo provider not enabled");
            }
        }
        VideoProviderArg::Kling => {
            #[cfg(feature = "kling-video")]
            {
                let builder = genviz::KlingVideoProvider::builder();
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "kling-video"))]
            {
                anyhow::bail!("Kling video provider not enabled");
            }
        }
        VideoProviderArg::Fal => {
            #[cfg(feature = "fal-video")]
            {
                let mut builder = genviz::FalVideoProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "wan-2.1" | "wan" => genviz::FalVideoModel::Wan21,
                        "wan-2.1-i2v" | "wan-i2v" => genviz::FalVideoModel::Wan21I2V,
                        "minimax" => genviz::FalVideoModel::MiniMax,
                        "ltx-video" | "ltx" => genviz::FalVideoModel::LtxVideo,
                        s if s.starts_with("fal-ai/") => {
                            genviz::FalVideoModel::Custom(s.to_string())
                        }
                        _ => anyhow::bail!(
                            "Unknown fal.ai video model: {}. Options: wan-2.1, wan-2.1-i2v, minimax, ltx-video, or fal-ai/...",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "fal-video"))]
            {
                anyhow::bail!("fal.ai video provider not enabled");
            }
        }
    };

    if !json_output {
        eprintln!("done.");
    }

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
            name: "Flux (Black Forest Labs) - models: flux-pro-1.1, flux-pro-1.1-ultra, flux-pro, flux-dev, flux-2-max, flux-2-pro, flux-2-flex, flux-2-klein-4b, flux-2-klein-9b, flux-kontext-pro, flux-kontext-max, flux-fill-pro, flux-expand-pro",
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
        ProviderInfo {
            name: "OpenAI (gpt-image-1, dall-e-3)",
            kind: "openai",
            media_type: "image",
            env_var: "OPENAI_API_KEY",
            enabled: cfg!(feature = "openai-image"),
        },
        ProviderInfo {
            name: "Kling (kling-v1, kling-v1.5, kling-v2)",
            kind: "kling",
            media_type: "image",
            env_var: "KLING_ACCESS_KEY",
            enabled: cfg!(feature = "kling-image"),
        },
        ProviderInfo {
            name: "fal.ai (flux-schnell, flux-pro, flux-pro-ultra, recraft-v3, ideogram-v3, hidream)",
            kind: "fal",
            media_type: "image",
            env_var: "FAL_KEY",
            enabled: cfg!(feature = "fal-image"),
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
            name: "Sora (OpenAI)",
            kind: "openai",
            media_type: "video",
            env_var: "OPENAI_API_KEY",
            enabled: cfg!(feature = "openai-video"),
        },
        ProviderInfo {
            name: "Veo (Google)",
            kind: "veo",
            media_type: "video",
            env_var: "GOOGLE_API_KEY",
            enabled: cfg!(feature = "veo"),
        },
        ProviderInfo {
            name: "Kling Video",
            kind: "kling",
            media_type: "video",
            env_var: "KLING_ACCESS_KEY",
            enabled: cfg!(feature = "kling-video"),
        },
        ProviderInfo {
            name: "fal.ai (wan-2.1, minimax, ltx-video)",
            kind: "fal",
            media_type: "video",
            env_var: "FAL_KEY",
            enabled: cfg!(feature = "fal-video"),
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
