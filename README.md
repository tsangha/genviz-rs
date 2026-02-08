# genviz

[![CI](https://github.com/tsangha/genviz-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tsangha/genviz-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/genviz.svg)](https://crates.io/crates/genviz)
[![docs.rs](https://docs.rs/genviz/badge.svg)](https://docs.rs/genviz)
[![License: MIT](https://img.shields.io/crates/l/genviz.svg)](LICENSE)

One interface for AI image and video generation. Write once, swap providers freely.

```rust
use genviz::{GeminiProvider, GenerationRequest, ImageProvider};

let provider = GeminiProvider::builder().build()?;
let image = provider.generate(&GenerationRequest::new("A golden retriever in snow")).await?;
image.save("puppy.png")?;
```

Switch to Flux, Grok, or OpenAI by changing one line — your request code stays the same.

## Why genviz

- **Unified trait API** across 6 providers — no vendor lock-in
- **Image generation + editing** with text prompts (all providers)
- **Video generation** via Grok, Sora, and Veo
- **Typed error handling** with retryability detection, rate limit parsing, and billing errors
- **Feature-flag granularity** — compile only the providers you use
- **CLI and MCP server** included for scripting and AI agent integration

## Providers

### Images

| Provider | Models | Env Var | Editing | Aspect Ratio | Seed |
|----------|--------|---------|---------|--------------|------|
| [Gemini](https://aistudio.google.com/apikey) | `NanoBanana`, `NanoBananaPro` | `GOOGLE_API_KEY` | yes | - | yes |
| [Flux](https://api.bfl.ai) | 13 models (Pro, Ultra, Kontext, Fill, Expand) | `BFL_API_KEY` | yes | yes | yes |
| [Grok](https://x.ai) | `GrokImagine` | `XAI_API_KEY` | yes | yes | - |
| [OpenAI](https://platform.openai.com) | `GptImage1`, `DallE3` | `OPENAI_API_KEY` | yes | yes | - |

### Videos

| Provider | Env Var | Duration | Aspect Ratio | Image-to-Video |
|----------|---------|----------|--------------|----------------|
| Grok | `XAI_API_KEY` | 1–15s | yes | yes |
| [Sora](https://platform.openai.com) | `OPENAI_API_KEY` | yes | yes | - |
| [Veo](https://aistudio.google.com/apikey) | `GOOGLE_API_KEY` | - | - | resolution control |

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
genviz = "0.2"
```

All image and video providers are enabled by default. To include only what you need:

```toml
[dependencies]
genviz = { version = "0.2", default-features = false, features = ["gemini-image", "grok-video"] }
```

<details>
<summary>All feature flags</summary>

| Flag | Default | What it enables |
|------|---------|-----------------|
| `flux-image` | yes | Flux (13 models) |
| `gemini-image` | yes | Gemini |
| `grok-image` | yes | Grok Imagine |
| `openai-image` | yes | OpenAI (gpt-image-1, DALL-E 3) |
| `grok-video` | yes | Grok video |
| `openai-video` | yes | Sora |
| `veo` | yes | Veo |
| `cli` | no | CLI binary |

</details>

Set the API key for your chosen provider as an environment variable, then:

```rust
use genviz::prelude::*;

// Images — any provider, same interface
let gemini = GeminiProvider::builder().build()?;
let flux = FluxProvider::builder().model(FluxModel::Flux2Max).build()?;

let request = GenerationRequest::new("A serene mountain lake at dawn")
    .with_aspect_ratio(AspectRatio::Landscape)
    .with_seed(42);

let image = gemini.generate(&request).await?;
image.save("lake.png")?;

// Edit an existing image
let input = std::fs::read("photo.png")?;
let edit_request = GenerationRequest::new("Make it sunset")
    .with_input_image(input);
let edited = gemini.generate(&edit_request).await?;

// Videos
let video_provider = GrokVideoProvider::builder().build()?;
let video = video_provider
    .generate(&VideoGenerationRequest::new("Ocean waves crashing").with_duration(5))
    .await?;
video.save("waves.mp4")?;
```

## Error Handling

Errors are typed and actionable — you can match on specific failure modes and decide whether to retry:

```rust
use genviz::GenVizError;

match provider.generate(&request).await {
    Ok(image) => image.save("out.png")?,
    Err(GenVizError::RateLimited { retry_after }) => { /* back off */ }
    Err(GenVizError::ContentBlocked(reason)) => { /* prompt was rejected */ }
    Err(GenVizError::Billing(msg)) => { /* account issue */ }
    Err(e) if e.is_retryable() => { /* transient failure, safe to retry */ }
    Err(e) => return Err(e.into()),
}
```

## CLI

```bash
cargo install genviz --features cli
```

```bash
# Generate
genviz image "A cat wearing sunglasses" -o cat.png
genviz image "A sunset" -o sunset.png -p flux --model flux-2-max
genviz video "Ocean waves" -o waves.mp4 -p grok --duration 5

# Edit
genviz image "Change the sofa to leather" -o edited.png --input living_room.png

# List available providers
genviz providers
```

## MCP Server

Expose image and video generation as tools for Claude Code and other AI agents:

```bash
genviz mcp
```

Add to your Claude Code config (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "genviz": {
      "command": "genviz",
      "args": ["mcp"]
    }
  }
}
```

This gives AI agents access to `generate_image` and `generate_video` tools. See [docs.rs](https://docs.rs/genviz) for full tool schemas and parameters.

## Documentation

- **API Reference**: [docs.rs/genviz](https://docs.rs/genviz)
- **Examples**: [`examples/`](examples/) directory
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT
