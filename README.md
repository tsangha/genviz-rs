# genviz-rs

[![CI](https://github.com/tsangha/genviz-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tsangha/genviz-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/genviz.svg)](https://crates.io/crates/genviz)
[![docs.rs](https://docs.rs/genviz/badge.svg)](https://docs.rs/genviz)
[![License: MIT](https://img.shields.io/crates/l/genviz.svg)](LICENSE)

Unified Rust library for AI media generation (images and video) via multiple providers.

## Features

- **6 providers**: Flux, Gemini, Grok, OpenAI (gpt-image-1/dall-e-3), Sora, and Veo
- **13 Flux models**: FLUX.1, FLUX.2, Kontext, Fill, Expand, Ultra variants
- **Unified trait interface**: Same API regardless of provider
- **Image editing**: Edit existing images with text prompts (all providers)
- **Format validation**: Verify image magic bytes match claimed format
- **Robust error handling**: Typed errors with retryability, billing detection, rate limit headers
- **CLI included**: Generate images and videos from the command line
- **MCP server**: Integrate with Claude Code and other AI agents

## Installation

```toml
[dependencies]
genviz = "0.2.0"
```

Feature flags:
- `flux-image` (default) - Enable Flux image provider (13 models)
- `gemini-image` (default) - Enable Gemini image provider
- `grok-image` (default) - Enable Grok Imagine provider
- `openai-image` (default) - Enable OpenAI image provider (gpt-image-1, dall-e-3)
- `grok-video` (default) - Enable Grok video provider
- `openai-video` (default) - Enable Sora video provider
- `veo` (default) - Enable Veo video provider
- `cli` - Build CLI binary

## Quick Start

```rust
use genviz::{GeminiProvider, GenerationRequest, ImageProvider};

#[tokio::main]
async fn main() -> genviz::Result<()> {
    // Uses GOOGLE_API_KEY env var
    let provider = GeminiProvider::builder().build()?;

    let request = GenerationRequest::new("A golden retriever puppy in snow");
    let image = provider.generate(&request).await?;

    image.save("puppy.png")?;
    println!("Generated {} bytes", image.size());
    Ok(())
}
```

## Image Providers

### Gemini (Google)

```rust
use genviz::{GeminiProvider, GeminiModel};

// Nano Banana Pro (default, highest quality)
let provider = GeminiProvider::builder().build()?;

// Nano Banana (faster, cheaper)
let provider = GeminiProvider::builder()
    .model(GeminiModel::NanoBanana)
    .build()?;
```

Requires `GOOGLE_API_KEY` env var. Get one at [Google AI Studio](https://aistudio.google.com/apikey).

### Flux (Black Forest Labs)

```rust
use genviz::{FluxProvider, FluxModel};

// Flux Pro 1.1 (default)
let provider = FluxProvider::builder().build()?;

// FLUX.2 Max (latest, editing + generation)
let provider = FluxProvider::builder()
    .model(FluxModel::Flux2Max)
    .build()?;

// Kontext Pro (context-aware editing)
let provider = FluxProvider::builder()
    .model(FluxModel::KontextPro)
    .build()?;
```

Available models: `FluxPro11`, `FluxPro11Ultra`, `FluxPro`, `FluxDev`, `Flux2Max`, `Flux2Pro`, `Flux2Flex`, `Flux2Klein4B`, `Flux2Klein9B`, `KontextPro`, `KontextMax`, `FillPro`, `ExpandPro`

Requires `BFL_API_KEY` env var. Get one at [api.bfl.ai](https://api.bfl.ai).

### Grok Imagine (xAI)

```rust
use genviz::GrokProvider;

let provider = GrokProvider::builder().build()?;
```

Requires `XAI_API_KEY` env var. Get one at [x.ai](https://x.ai).

### OpenAI (gpt-image-1, dall-e-3)

```rust
use genviz::{OpenAiImageProvider, OpenAiImageModel};

// gpt-image-1 (default)
let provider = OpenAiImageProvider::builder().build()?;

// DALL-E 3
let provider = OpenAiImageProvider::builder()
    .model(OpenAiImageModel::DallE3)
    .build()?;
```

Requires `OPENAI_API_KEY` env var. Get one at [platform.openai.com](https://platform.openai.com).

## Video Providers

### Grok Imagine Video (xAI)

```rust
use genviz::{GrokVideoProvider, VideoGenerationRequest, VideoProvider};

let provider = GrokVideoProvider::builder().build()?;
let request = VideoGenerationRequest::new("Ocean waves").with_duration(5);
let video = provider.generate(&request).await?;
video.save("waves.mp4")?;
```

### Sora (OpenAI)

```rust
use genviz::{SoraProvider, VideoGenerationRequest, VideoProvider};

let provider = SoraProvider::builder().build()?;
let request = VideoGenerationRequest::new("A cat jumping")
    .with_duration(8)
    .with_aspect_ratio("16:9");
let video = provider.generate(&request).await?;
video.save("cat.mp4")?;
```

Requires `OPENAI_API_KEY` env var.

### Veo (Google)

```rust
use genviz::{VeoProvider, VideoGenerationRequest, VideoProvider};

let provider = VeoProvider::builder().build()?;
let request = VideoGenerationRequest::new("A timelapse of clouds")
    .with_resolution("720p");
let video = provider.generate(&request).await?;
video.save("clouds.mp4")?;
```

Requires `GOOGLE_API_KEY` env var with billing enabled.

## CLI Usage

```bash
# Install
cargo install genviz --features cli

# Generate images
genviz image "A cat wearing sunglasses" -o cat.png
genviz image "A sunset" -o sunset.png -p flux --model flux-2-max
genviz image "A portrait" -o portrait.png -p openai --model dall-e-3
genviz image "A city" -o city.png -p grok --aspect-ratio 16:9

# Image editing (all providers support --input)
genviz image "Change the sofa to leather" -o edited.png -p gemini --input living_room.png
genviz image "Add a hat" -o with_hat.png -p flux --model flux-kontext-pro --input portrait.jpg

# Generate videos
genviz video "Ocean waves" -o waves.mp4 -p grok
genviz video "A cat jumping" -o cat.mp4 -p openai --duration 8
genviz video "A timelapse of clouds" -o clouds.mp4 -p veo

# List providers
genviz providers

# JSON output
genviz image "A dog" -o dog.png --json
```

**Provider capabilities:**

| Provider | `--aspect-ratio` | `--width`/`--height` | `--seed` | `--input` (editing) |
|----------|------------------|----------------------|----------|---------------------|
| Gemini   | -                | -                    | yes      | yes                 |
| Flux     | yes              | yes                  | yes      | yes                 |
| Grok     | yes              | -                    | -        | yes                 |
| OpenAI   | yes              | yes                  | -        | yes                 |

## MCP Server (AI Agent Integration)

Run as an MCP server for integration with Claude Code and other AI agents:

```bash
genviz mcp
```

### Claude Code Configuration

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "genviz": {
      "command": "genviz",
      "args": ["mcp"],
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key",
        "BFL_API_KEY": "your-bfl-api-key",
        "XAI_API_KEY": "your-xai-api-key",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

### Available Tools

#### `generate_image`

```json
{
  "name": "generate_image",
  "parameters": {
    "prompt": "A sunset over mountains",
    "provider": "flux",
    "output_path": "/tmp/sunset.png",
    "aspect_ratio": "16:9"
  }
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Text description of the image |
| `provider` | No | `gemini` (default), `flux`, `grok`, or `openai` |
| `output_path` | No | Save path (returns base64 if omitted). Use `{n}` placeholder for batch. |
| `model` | No | Model variant |
| `width`/`height` | No | Dimensions in pixels (Flux/OpenAI) |
| `aspect_ratio` | No | `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `21:9` |
| `seed` | No | For deterministic generation (Gemini/Flux) |
| `count` | No | Number of images (1-10, default 1) |
| `concurrency` | No | Max parallel requests (1-5, default 3) |
| `input_image` | No | Base64-encoded image for editing (all providers) |

#### `generate_video`

```json
{
  "name": "generate_video",
  "parameters": {
    "prompt": "Ocean waves at sunset",
    "provider": "grok",
    "output_path": "/tmp/waves.mp4",
    "duration": 5
  }
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Text description of the video |
| `provider` | No | `grok` (default), `openai`, or `veo` |
| `output_path` | No | Save path |
| `duration` | No | Video duration in seconds |
| `aspect_ratio` | No | e.g., `16:9`, `9:16` |

## Request Options

```rust
let request = GenerationRequest::new("A serene lake")
    .with_aspect_ratio(AspectRatio::Landscape)  // 16:9
    .with_seed(42)                               // Deterministic
    .with_format(ImageFormat::Png);              // Output format
```

## Error Handling

```rust
use genviz::GenVizError;

match provider.generate(&request).await {
    Ok(image) => image.save("out.png")?,
    Err(GenVizError::RateLimited { retry_after }) => {
        println!("Rate limited, retry after {:?}", retry_after);
    }
    Err(GenVizError::ContentBlocked(reason)) => {
        println!("Content blocked: {}", reason);
    }
    Err(GenVizError::Billing(msg)) => {
        println!("Billing issue: {}", msg);
    }
    Err(e) if e.is_retryable() => {
        println!("Transient error, retrying: {}", e);
    }
    Err(e) => return Err(e.into()),
}
```

## License

MIT
