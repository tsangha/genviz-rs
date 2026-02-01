# genviz-rs

Unified Rust library for AI media generation (images and video) via multiple providers.

## Features

- **Multiple providers**: Gemini, Flux (Black Forest Labs), Grok, and Veo
- **Unified trait interface**: Same API regardless of provider
- **Automatic fallback**: Try multiple providers in sequence
- **Format validation**: Verify image magic bytes match claimed format
- **CLI included**: Generate images from the command line

## Installation

```toml
[dependencies]
genviz = "0.1.1"
```

Feature flags:
- `flux-image` (default) - Enable Flux image provider
- `gemini-image` (default) - Enable Gemini image provider
- `grok-image` (default) - Enable Grok Imagine provider
- `grok-video` (default) - Enable Grok video provider
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

## Providers

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

// Flux Dev (faster)
let provider = FluxProvider::builder()
    .model(FluxModel::FluxDev)
    .build()?;
```

Requires `BFL_API_KEY` env var. Get one at [api.bfl.ml](https://api.bfl.ml).

### Grok Imagine (xAI)

```rust
use genviz::GrokProvider;

let provider = GrokProvider::builder().build()?;
```

Requires `XAI_API_KEY` env var. Get one at [x.ai](https://x.ai).

## Fallback Support

```rust
use genviz::{ImageClient, FluxProvider, GeminiProvider, GenerationRequest};

let client = ImageClient::new(FluxProvider::builder().build()?)
    .with_fallback(GeminiProvider::builder().build()?);

// Tries Flux first, falls back to Gemini on transient errors
let image = client.generate(&GenerationRequest::new("A sunset")).await?;
```

## CLI Usage

```bash
# Install
cargo install genviz --features cli

# Generate image with Gemini (default)
genviz image "A cat wearing sunglasses" -o cat.png

# Gemini supports seed for deterministic output
genviz image "A dog" -o dog.png -p gemini --seed 42

# Flux supports aspect ratio and explicit dimensions
genviz image "A sunset" -o sunset.png -p flux --aspect-ratio 16:9
genviz image "A portrait" -o portrait.png -p flux --width 768 --height 1024

# Grok supports aspect ratio
genviz image "A futuristic city" -o city.png -p grok --aspect-ratio 9:16

# Image editing (all providers support --input)
genviz image "Change the sofa to leather" -o edited.png -p gemini --input living_room.png
genviz image "Add a hat to the person" -o with_hat.png -p flux --input portrait.jpg
genviz image "Make the background sunset" -o sunset_bg.png -p grok --input photo.png

# Generate video with Veo (requires gs:// output path)
genviz video "A timelapse of clouds" -o gs://my-bucket/output/clouds.mp4 -p veo

# Generate video with Grok
genviz video "Ocean waves" -o waves.mp4 -p grok

# List providers
genviz providers

# JSON output
genviz image "A dog" -o dog.png --json
```

**Provider capabilities:**

| Provider | `--aspect-ratio` | `--width`/`--height` | `--seed` | `--input` (editing) |
|----------|------------------|----------------------|----------|---------------------|
| Gemini   | ❌               | ❌                   | ✅       | ✅                  |
| Flux     | ✅               | ✅                   | ✅       | ✅                  |
| Grok     | ✅               | ❌                   | ❌       | ✅                  |

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
        "XAI_API_KEY": "your-xai-api-key"
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
| `provider` | No | `gemini` (default), `flux`, or `grok` |
| `output_path` | No | Save path (returns base64 if omitted). Use `{n}` placeholder for batch. |
| `model` | No | Model variant |
| `width`/`height` | No | Dimensions in pixels (Flux only) |
| `aspect_ratio` | No | `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `21:9` (Flux/Grok only) |
| `seed` | No | For deterministic generation (Gemini/Flux only) |
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
| `provider` | No | `grok` (default) or `veo` |
| `output_path` | No | Save path (Veo requires `gs://` path) |
| `duration` | No | Video duration in seconds (Grok: 1-15) |
| `aspect_ratio` | No | e.g., `16:9`, `9:16` |

### Parallel Generation

Generate multiple variations with a single call:

```json
{
  "name": "generate_image",
  "parameters": {
    "prompt": "A colorful parrot",
    "count": 5,
    "concurrency": 3,
    "output_path": "/tmp/parrot_{n}.png"
  }
}
```

This generates 5 images, 3 at a time, saving to `parrot_0.png` through `parrot_4.png`.

**Guardrails:**
- Max 10 images per call (prevents runaway costs)
- Max 5 concurrent requests (respects API rate limits)
- `{n}` placeholder required in `output_path` when `count > 1`

## Request Options

```rust
let request = GenerationRequest::new("A serene lake")
    .with_aspect_ratio(AspectRatio::Landscape)  // 16:9
    .with_seed(42)                               // Deterministic
    .with_format(ImageFormat::Png);              // Output format
```

## Format Validation

Verify the API returned the correct image format:

```rust
let image = provider.generate(&request).await?;

// Check magic bytes match claimed format
if !image.validate_format() {
    eprintln!("Warning: format mismatch, detected {:?}", image.detected_format());
}

// Or auto-detect format from bytes
let image = GeneratedImage::from_bytes(data, provider, metadata)?;
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
    Err(e) if e.is_retryable() => {
        println!("Transient error, retrying: {}", e);
    }
    Err(e) => return Err(e.into()),
}
```

## License

MIT
