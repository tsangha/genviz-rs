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
    Video(Box<VideoArgs>),

    /// Batch image generation via Vertex AI
    Batch {
        #[command(subcommand)]
        action: BatchCommands,
    },

    /// List available providers
    Providers,

    /// Run as MCP server (for AI agent integration)
    Mcp,
}

#[derive(Subcommand)]
enum BatchCommands {
    /// Submit a batch of prompts for image generation
    Submit(BatchSubmitArgs),

    /// Check the status of a batch job
    Status(BatchStatusArgs),

    /// Download results from a completed batch job
    Download(BatchDownloadArgs),
}

#[derive(Args)]
struct BatchSubmitArgs {
    /// Prompts to generate (provide multiple, or use --prompts-file)
    #[arg(required_unless_present = "prompts_file")]
    prompts: Vec<String>,

    /// File with one prompt per line
    #[arg(long)]
    prompts_file: Option<PathBuf>,

    /// GCS bucket for input/output (e.g., "my-bucket" or "gs://my-bucket")
    #[arg(long)]
    gcs_bucket: String,

    /// Model variant (nano-banana, nano-banana-pro)
    #[arg(long, default_value = "nano-banana-pro")]
    model: String,

    /// GCP project ID (defaults to VERTEX_AI_PROJECT env var)
    #[arg(long, env = "VERTEX_AI_PROJECT")]
    project: Option<String>,

    /// GCP location (defaults to VERTEX_AI_LOCATION or us-central1)
    #[arg(long, env = "VERTEX_AI_LOCATION")]
    location: Option<String>,
}

#[derive(Args)]
struct BatchStatusArgs {
    /// Full batch job resource name (from submit output)
    job_name: String,
}

#[derive(Args)]
struct BatchDownloadArgs {
    /// Full batch job resource name (from submit output)
    job_name: String,

    /// Directory to save downloaded images
    #[arg(short, long, default_value = "./batch_output")]
    output_dir: String,
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

    /// Model variant (flux: flux-pro-1.1, flux-pro-1.1-ultra, flux-pro, flux-dev, flux-2-max, flux-2-pro, flux-2-flex, flux-2-klein-4b, flux-2-klein-9b, flux-kontext-pro, flux-kontext-max, flux-fill-pro, flux-expand-pro; gemini: nano-banana, nano-banana-pro; grok: grok-imagine; openai: gpt-image-2, gpt-image-1, dall-e-3; kling: kling-v1, kling-v1.5, kling-v2; fal: flux-schnell, flux-pro, flux-pro-ultra, recraft-v3, recraft-v4.1, recraft-v4.1-pro, ideogram-v3, hidream)
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

    /// Source image URL for image-to-video (Grok, Kling, fal.ai, Sora)
    #[arg(long)]
    source_image_url: Option<String>,

    /// URL of last frame image for interpolation (fal.ai Wan FLF2V)
    #[arg(long)]
    last_frame_url: Option<String>,

    /// First frame image file path (Veo only, duration forced to 8s)
    #[arg(long)]
    image: Option<PathBuf>,

    /// Last frame image file path (Veo only, duration forced to 8s)
    #[arg(long)]
    last_frame: Option<PathBuf>,

    /// Video file path for extension/continuation (Veo only, extends by up to 8s)
    #[arg(long)]
    video: Option<PathBuf>,

    /// Reference image 1 for style/asset guidance (Veo only, up to 3 total)
    #[arg(long)]
    reference_image_1: Option<PathBuf>,

    /// Reference image 2 (Veo only)
    #[arg(long)]
    reference_image_2: Option<PathBuf>,

    /// Reference image 3 (Veo only)
    #[arg(long)]
    reference_image_3: Option<PathBuf>,

    /// Negative prompt — what to avoid (Veo, fal.ai)
    #[arg(long)]
    negative_prompt: Option<String>,

    /// Person generation policy: allow_all or allow_adult (Veo only)
    #[arg(long)]
    person_generation: Option<String>,

    /// Google Cloud project ID for Vertex AI backend
    #[arg(long, env = "VERTEX_AI_PROJECT")]
    vertex_project: Option<String>,

    /// Google Cloud location for Vertex AI backend (default: us-central1)
    #[arg(long, env = "VERTEX_AI_LOCATION")]
    vertex_location: Option<String>,

    /// GCS bucket URI for video output (Vertex AI only)
    #[arg(long)]
    storage_uri: Option<String>,

    /// Enable prompt enhancement (Vertex AI only)
    #[arg(long)]
    enhance_prompt: Option<bool>,

    /// Enable audio generation (Vertex AI only)
    #[arg(long)]
    generate_audio: Option<bool>,

    /// Seed for deterministic generation (Seedance via fal.ai)
    #[arg(long)]
    seed: Option<i64>,

    /// Lock camera position (Seedance via fal.ai)
    #[arg(long)]
    camera_fixed: Option<bool>,

    /// Enable prompt enhancement (MiniMax Hailuo)
    #[arg(long)]
    prompt_optimizer: Option<bool>,

    /// URL of subject reference image for character consistency (MiniMax direct API)
    #[arg(long)]
    subject_reference_url: Option<String>,

    /// Video resolution (e.g., "720p", "1080p")
    #[arg(long)]
    resolution: Option<String>,

    /// Model variant (grok: grok-imagine-video; openai: sora-2; veo: veo-3.1-generate-preview, veo-3.1-lite-generate-preview; fal: wan-2.1, wan-2.1-i2v, wan-2.7, happy-horse, hailuo-std, hailuo-pro, hailuo-fast, seedance-pro, seedance-lite, seedance-1.5, seedance-2.0, seedance-2.0-fast, ltx-video, kling-std, kling-pro; minimax: hailuo-2.3, hailuo-2.3-fast)
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
    Higgsfield,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VideoProviderArg {
    Grok,
    Openai,
    Veo,
    Kling,
    Fal,
    Minimax,
    Higgsfield,
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
            generate_video(*args, cli.json).await?;
        }
        Commands::Batch { action } => {
            run_batch(action, cli.json).await?;
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
        ImageProviderArg::Higgsfield => {
            if args.width.is_some() || args.height.is_some() {
                anyhow::bail!(
                    "Higgsfield does not support --width/--height (use --aspect-ratio instead)"
                );
            }
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
                        "gpt-image-2" => genviz::OpenAiImageModel::GptImage2,
                        "gpt-image-1" => genviz::OpenAiImageModel::GptImage1,
                        "dall-e-3" => genviz::OpenAiImageModel::DallE3,
                        _ => anyhow::bail!(
                            "Unknown OpenAI model: {}. Options: gpt-image-2, gpt-image-1, dall-e-3",
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
                        "recraft-v4.1" | "recraft-v41" => genviz::FalImageModel::RecraftV41,
                        "recraft-v4.1-pro" | "recraft-v41-pro" => {
                            genviz::FalImageModel::RecraftV41Pro
                        }
                        "ideogram-v3" => genviz::FalImageModel::Ideogram3,
                        "hidream" => genviz::FalImageModel::HiDream,
                        s if s.starts_with("fal-ai/") => {
                            genviz::FalImageModel::Custom(s.to_string())
                        }
                        _ => anyhow::bail!(
                            "Unknown fal.ai model: {}. Options: flux-schnell, flux-pro, flux-pro-ultra, recraft-v3, recraft-v4.1, recraft-v4.1-pro, ideogram-v3, hidream, or fal-ai/...",
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
        ImageProviderArg::Higgsfield => {
            #[cfg(feature = "higgsfield-image")]
            {
                let mut builder = genviz::HiggsfieldImageProvider::builder();
                if let Some(ref m) = args.model {
                    let model = parse_higgsfield_image_model(m)?;
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "higgsfield-image"))]
            {
                anyhow::bail!("Higgsfield image provider not enabled");
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
    use base64::Engine;

    // Validate provider-specific flags
    if args.subject_reference_url.is_some() && !matches!(args.provider, VideoProviderArg::Minimax) {
        anyhow::bail!("--subject-reference-url is only supported by the minimax provider");
    }

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
    if let Some(url) = args.last_frame_url {
        request = request.with_last_frame_url(url);
    }
    if let Some(seed) = args.seed {
        request = request.with_seed(seed);
    }
    if let Some(fixed) = args.camera_fixed {
        request = request.with_camera_fixed(fixed);
    }
    if let Some(optimize) = args.prompt_optimizer {
        request = request.with_prompt_optimizer(optimize);
    }
    if let Some(url) = args.subject_reference_url {
        request = request.with_subject_reference("character", url);
    }
    if let Some(res) = args.resolution {
        request = request.with_resolution(res);
    }

    // Read and base64-encode first frame image
    if let Some(ref path) = args.image {
        let data = std::fs::read(path)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        request = request.with_image(b64);
    }

    // Read and base64-encode last frame image
    if let Some(ref path) = args.last_frame {
        let data = std::fs::read(path)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        request = request.with_last_frame(b64);
    }

    // Read and base64-encode video for extension
    if let Some(ref path) = args.video {
        let data = std::fs::read(path)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        request = request.with_video(b64);
    }

    // Read and base64-encode reference images
    for path in [
        &args.reference_image_1,
        &args.reference_image_2,
        &args.reference_image_3,
    ]
    .into_iter()
    .flatten()
    {
        let data = std::fs::read(path)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        request = request.with_reference_image(b64);
    }

    // Pass through Veo-specific text params
    if let Some(ref neg) = args.negative_prompt {
        request = request.with_negative_prompt(neg.clone());
    }
    if let Some(ref pg) = args.person_generation {
        request = request.with_person_generation(pg.clone());
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
                        "veo-3.1-generate-preview" | "veo-3.1" => genviz::VeoModel::Veo31Preview,
                        "veo-3.1-lite-generate-preview" | "veo-3.1-lite" => {
                            genviz::VeoModel::Veo31LitePreview
                        }
                        _ => anyhow::bail!(
                            "Unknown Veo model: {}. Options: veo-3.1-generate-preview, veo-3.1-lite-generate-preview",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                // Configure Vertex AI backend if project specified
                if let Some(ref project) = args.vertex_project {
                    builder = builder.project(project.clone());
                    if let Some(ref location) = args.vertex_location {
                        builder = builder.location(location.clone());
                    }
                }
                let provider = builder.build()?;
                // Pass through Vertex AI-specific params
                if let Some(ref uri) = args.storage_uri {
                    request = request.with_storage_uri(uri.clone());
                }
                if let Some(enhance) = args.enhance_prompt {
                    request = request.with_enhance_prompt(enhance);
                }
                if let Some(audio) = args.generate_audio {
                    request = request.with_generate_audio(audio);
                }
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
                        "wan-2.7" => genviz::FalVideoModel::Wan27,
                        "happy-horse" | "happyhorse" | "happyhorse-1.0" => {
                            genviz::FalVideoModel::HappyHorse
                        }
                        "hailuo-std" | "hailuo" => genviz::FalVideoModel::Hailuo23Std,
                        "hailuo-pro" => genviz::FalVideoModel::Hailuo23Pro,
                        "hailuo-fast" => genviz::FalVideoModel::Hailuo23Fast,
                        "seedance-pro" | "seedance" => genviz::FalVideoModel::SeedancePro,
                        "seedance-lite" => genviz::FalVideoModel::SeedanceLite,
                        "seedance-1.5" | "seedance-1.5-pro" => genviz::FalVideoModel::Seedance15Pro,
                        "seedance-2.0" | "seedance-2" => genviz::FalVideoModel::Seedance20,
                        "seedance-2.0-fast" | "seedance-2-fast" => {
                            genviz::FalVideoModel::Seedance20Fast
                        }
                        "minimax" => genviz::FalVideoModel::Hailuo23Std, // backward compat alias
                        "ltx-video" | "ltx" => genviz::FalVideoModel::LtxVideo,
                        "kling-std" | "kling" => genviz::FalVideoModel::KlingStd,
                        "kling-pro" => genviz::FalVideoModel::KlingPro,
                        s if s.starts_with("fal-ai/") => {
                            genviz::FalVideoModel::Custom(s.to_string())
                        }
                        _ => anyhow::bail!(
                            "Unknown fal.ai video model: {}. Options: wan-2.1, wan-2.1-i2v, wan-2.7, happy-horse, hailuo-std, hailuo-pro, hailuo-fast, seedance-pro, seedance-lite, seedance-1.5, seedance-2.0, seedance-2.0-fast, ltx-video, kling-std, kling-pro, or fal-ai/...",
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
        VideoProviderArg::Minimax => {
            #[cfg(feature = "minimax-video")]
            {
                let mut builder = genviz::MiniMaxVideoProvider::builder();
                if let Some(ref m) = args.model {
                    let model = match m.as_str() {
                        "hailuo-2.3" | "hailuo" => genviz::MiniMaxVideoModel::Hailuo23,
                        "hailuo-2.3-fast" | "hailuo-fast" => {
                            genviz::MiniMaxVideoModel::Hailuo23Fast
                        }
                        _ => anyhow::bail!(
                            "Unknown MiniMax video model: {}. Options: hailuo-2.3, hailuo-2.3-fast",
                            m
                        ),
                    };
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "minimax-video"))]
            {
                anyhow::bail!("MiniMax video provider not enabled");
            }
        }
        VideoProviderArg::Higgsfield => {
            #[cfg(feature = "higgsfield-video")]
            {
                let mut builder = genviz::HiggsfieldVideoProvider::builder();
                if let Some(ref m) = args.model {
                    let model = parse_higgsfield_video_model(m)?;
                    builder = builder.model(model);
                }
                let provider = builder.build()?;
                provider.generate(&request).await?
            }
            #[cfg(not(feature = "higgsfield-video"))]
            {
                anyhow::bail!("Higgsfield video provider not enabled");
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

async fn run_batch(action: BatchCommands, json_output: bool) -> anyhow::Result<()> {
    use genviz::{download_batch_results, get_batch_status, submit_batch};

    match action {
        BatchCommands::Submit(args) => {
            let project = args.project.ok_or_else(|| {
                anyhow::anyhow!(
                    "Batch requires Vertex AI. Set --project or VERTEX_AI_PROJECT env var."
                )
            })?;
            let location = args.location.unwrap_or_else(|| "us-central1".to_string());

            // Collect prompts from args or file
            let mut prompts = args.prompts;
            if let Some(ref path) = args.prompts_file {
                let content = std::fs::read_to_string(path)?;
                prompts.extend(
                    content
                        .lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty())
                        .map(String::from),
                );
            }

            if prompts.is_empty() {
                anyhow::bail!("No prompts provided. Pass them as arguments or use --prompts-file.");
            }

            let model = match args.model.as_str() {
                "nano-banana" => genviz::GeminiModel::NanoBanana,
                "nano-banana-pro" => genviz::GeminiModel::NanoBananaPro,
                _ => anyhow::bail!(
                    "Unknown model: {}. Options: nano-banana, nano-banana-pro",
                    args.model
                ),
            };

            if !json_output {
                eprint!(
                    "Submitting batch of {} prompts to Vertex AI ({}/{})... ",
                    prompts.len(),
                    project,
                    location
                );
            }

            let result =
                submit_batch(&project, &location, &prompts, &args.gcs_bucket, model).await?;

            if !json_output {
                eprintln!("done.");
            }

            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("Batch job submitted:");
                println!("  Job name:   {}", result.job_name);
                println!("  Display:    {}", result.display_name);
                println!("  State:      {}", result.state);
                println!("  Prompts:    {}", result.num_prompts);
                println!("  Input URI:  {}", result.input_uri);
                println!();
                println!("Check status with:");
                println!("  genviz batch status \"{}\"", result.job_name);
            }
        }
        BatchCommands::Status(args) => {
            if !json_output {
                eprint!("Checking batch job status... ");
            }

            let status = get_batch_status(&args.job_name).await?;

            if !json_output {
                eprintln!("done.");
            }

            if json_output {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("Batch job status:");
                println!("  Job name:   {}", status.job_name);
                println!("  Display:    {}", status.display_name);
                println!("  State:      {}", status.state);
                if let Some(ref ct) = status.create_time {
                    println!("  Created:    {}", ct);
                }
                if let Some(ref ut) = status.update_time {
                    println!("  Updated:    {}", ut);
                }

                if status.state == "JOB_STATE_SUCCEEDED" {
                    if let Some(ref prefix) = status.output_uri_prefix {
                        println!("  Output:     {}", prefix);
                        println!();
                        println!("Download results with:");
                        println!("  genviz batch download \"{}\"", status.job_name);
                    }
                }
            }
        }
        BatchCommands::Download(args) => {
            // First get the job to find the output URI
            if !json_output {
                eprint!("Fetching job info... ");
            }

            let status = get_batch_status(&args.job_name).await?;

            if status.state != "JOB_STATE_SUCCEEDED" {
                anyhow::bail!(
                    "Job is not complete (state: {}). Wait for JOB_STATE_SUCCEEDED.",
                    status.state
                );
            }

            let output_prefix = status
                .output_uri_prefix
                .ok_or_else(|| anyhow::anyhow!("Job succeeded but no output URI found"))?;

            if !json_output {
                eprint!("downloading results to {}... ", args.output_dir);
            }

            let results: Vec<genviz::BatchImageResult> =
                download_batch_results(&output_prefix, &args.output_dir).await?;

            if !json_output {
                eprintln!("done.");
            }

            if json_output {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                println!(
                    "Downloaded {} images to {}/",
                    results.len(),
                    args.output_dir
                );
                for r in &results {
                    println!("  {} ({} bytes, {})", r.path, r.size_bytes, r.format);
                }
            }
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
            name: "fal.ai (flux-schnell, flux-pro, flux-pro-ultra, recraft-v3, recraft-v4.1, ideogram-v3, hidream)",
            kind: "fal",
            media_type: "image",
            env_var: "FAL_KEY",
            enabled: cfg!(feature = "fal-image"),
        },
        ProviderInfo {
            name: "Higgsfield (gpt-image-2, nano-banana-pro, soul-v2, flux-2, seedream-4.5, …) — via `higgsfield` CLI",
            kind: "higgsfield",
            media_type: "image",
            env_var: "(uses `higgsfield auth login`)",
            enabled: cfg!(feature = "higgsfield-image"),
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
            name: "Veo (Google) — Gemini Dev API or Vertex AI",
            kind: "veo",
            media_type: "video",
            env_var: "GOOGLE_API_KEY or VERTEX_AI_PROJECT",
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
            name: "fal.ai (wan-2.1, hailuo-std/pro/fast, seedance-pro/lite/1.5, ltx-video, kling-std/pro)",
            kind: "fal",
            media_type: "video",
            env_var: "FAL_KEY",
            enabled: cfg!(feature = "fal-video"),
        },
        ProviderInfo {
            name: "MiniMax Hailuo (hailuo-2.3, hailuo-2.3-fast) — subject reference support",
            kind: "minimax",
            media_type: "video",
            env_var: "MINIMAX_API_KEY",
            enabled: cfg!(feature = "minimax-video"),
        },
        ProviderInfo {
            name: "Higgsfield video (seedance-2.0, veo3-1, veo3-1-lite, kling3-0, wan2-7, soul-cast, …) — via `higgsfield` CLI",
            kind: "higgsfield",
            media_type: "video",
            env_var: "(uses `higgsfield auth login`)",
            enabled: cfg!(feature = "higgsfield-video"),
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

#[cfg(feature = "higgsfield-image")]
fn parse_higgsfield_image_model(s: &str) -> anyhow::Result<genviz::HiggsfieldImageModel> {
    use genviz::HiggsfieldImageModel as M;
    Ok(match s {
        "gpt-image-2" | "gpt_image_2" => M::GptImage2,
        "nano-banana-pro" | "nano_banana_2" => M::NanoBananaPro,
        "nano-banana-2" | "nano-banana-flash" | "nano_banana_flash" => M::NanoBanana2,
        "flux-2" | "flux_2" => M::Flux2,
        "flux-kontext" | "flux_kontext" => M::FluxKontext,
        "soul-v2" | "soul_v2" | "text2image_soul_v2" => M::SoulV2,
        "soul-cinematic" | "soul_cinematic" => M::SoulCinematic,
        "soul-location" | "soul_location" => M::SoulLocation,
        "seedream-4.5" | "seedream_v4_5" => M::Seedream45,
        "seedream-5-lite" | "seedream_v5_lite" => M::Seedream5Lite,
        "kling-omni" | "kling_omni_image" => M::KlingOmni,
        "cinematic-studio-2.5" | "cinematic_studio_2_5" => M::CinematicStudio25,
        "grok-image" | "grok_image" => M::GrokImage,
        "z-image" | "z_image" => M::ZImage,
        other => M::Custom(other.to_string()),
    })
}

#[cfg(feature = "higgsfield-video")]
fn parse_higgsfield_video_model(s: &str) -> anyhow::Result<genviz::HiggsfieldVideoModel> {
    use genviz::HiggsfieldVideoModel as M;
    Ok(match s {
        "seedance-2.0" | "seedance-2" | "seedance_2_0" => M::Seedance20,
        "seedance-1.5" | "seedance1_5" => M::Seedance15Pro,
        "veo3-1" | "veo-3.1" | "veo3_1" => M::Veo31,
        "veo3-1-lite" | "veo-3.1-lite" | "veo3_1_lite" => M::Veo31Lite,
        "veo3" | "veo-3" => M::Veo3,
        "kling3-0" | "kling-3.0" | "kling3_0" => M::Kling30,
        "kling2-6" | "kling-2.6" | "kling2_6" => M::Kling26,
        "wan2-7" | "wan-2.7" | "wan2_7" => M::Wan27,
        "wan2-6" | "wan-2.6" | "wan2_6" => M::Wan26,
        "hailuo" | "minimax-hailuo" | "minimax_hailuo" => M::Hailuo,
        "grok-video" | "grok_video" => M::GrokVideo,
        "soul-cast" | "soul_cast" => M::SoulCast,
        "cinematic-studio-3.0" | "cinematic_studio_3_0" => M::CinematicStudio3,
        "cinematic-studio-video-v2" | "cinematic_studio_video_v2" => M::CinematicStudioVideoV2,
        "marketing-studio-video" | "marketing_studio_video" => M::MarketingStudioVideo,
        other => M::Custom(other.to_string()),
    })
}
