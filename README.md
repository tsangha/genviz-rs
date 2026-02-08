# genviz

[![CI](https://github.com/tsangha/genviz-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tsangha/genviz-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/genviz.svg)](https://crates.io/crates/genviz)
[![docs.rs](https://docs.rs/genviz/badge.svg)](https://docs.rs/genviz)
[![License: MIT](https://img.shields.io/crates/l/genviz.svg)](LICENSE)

Generate images and videos with AI. Works as a **CLI**, a **Rust library**, or an **MCP server** for AI agents like Claude Code. Six providers, one interface, no lock-in.

### Install

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/tsangha/genviz-rs/releases/latest/download/genviz-installer.sh | sh
```

<details>
<summary>Windows / cargo / manual options</summary>

```powershell
# Windows
powershell -ExecutionPolicy Bypass -Command "irm https://github.com/tsangha/genviz-rs/releases/latest/download/genviz-installer.ps1 | iex"
```

```bash
# Or build from source
cargo install genviz --features cli
```

</details>

Set an API key for the provider you want to use:

```bash
export GOOGLE_API_KEY="..."  # or BFL_API_KEY, XAI_API_KEY, OPENAI_API_KEY
```

### Command line

```bash
$ genviz image "A cat wearing sunglasses" -o cat.png
Generated cat.png (1024x1024, 1.2MB) via gemini in 3.4s

$ genviz image "Make it a painting" -o painting.png --input cat.png
Generated painting.png (1024x1024, 980KB) via gemini in 2.8s

$ genviz video "Ocean waves crashing on rocks at sunset" -o waves.mp4 -p grok
Generated waves.mp4 (5s, 2.1MB) via grok in 12.3s
```

### Claude Code / MCP

```bash
claude mcp add genviz -- genviz mcp
```

Then just ask Claude: *"Generate an image of a sunset over mountains"* — it discovers available providers, models, and capabilities automatically via the `list_providers` tool.

### Rust library

```rust
use genviz::prelude::*;

let provider = GeminiProvider::builder().build()?;
let image = provider.generate(&GenerationRequest::new("A mountain lake at dawn")).await?;
image.save("lake.png")?;
```

Swap `GeminiProvider` for `FluxProvider`, `GrokProvider`, or `OpenAiImageProvider` — request code stays the same.

---

Works with [Gemini](https://aistudio.google.com/apikey), [Flux](https://api.bfl.ai) (13 models), [Grok](https://x.ai), [OpenAI](https://platform.openai.com) (gpt-image-1, DALL-E 3), [Sora](https://platform.openai.com), and [Veo](https://aistudio.google.com/apikey). You only need an API key for the provider you want to use.

## CLI

```bash
# Generate images
genviz image "A sunset over mountains" -o sunset.png
genviz image "A portrait" -o portrait.png -p openai --model dall-e-3
genviz image "A city skyline" -o city.png -p flux --model flux-2-max --aspect-ratio 16:9

# Edit existing images
genviz image "Change the sky to sunset" -o edited.png --input photo.png
genviz image "Add a hat" -o hat.png -p flux --model flux-kontext-pro --input portrait.jpg

# Generate videos
genviz video "Ocean waves" -o waves.mp4 -p grok --duration 5
genviz video "A cat jumping" -o cat.mp4 -p openai
genviz video "Timelapse of clouds" -o clouds.mp4 -p veo

# See what's available
genviz providers
```

## Rust Library

For use in your own Rust projects:

```toml
[dependencies]
genviz = "0.2"
```

```rust
use genviz::prelude::*;

// Generate an image — swap GeminiProvider for FluxProvider, GrokProvider, etc.
let provider = GeminiProvider::builder().build()?;
let image = provider.generate(&GenerationRequest::new("A mountain lake at dawn")).await?;
image.save("lake.png")?;

// Edit an existing image
let input = std::fs::read("photo.png")?;
let edited = provider
    .generate(&GenerationRequest::new("Make it sunset").with_input_image(input))
    .await?;

// Generate a video
let video_provider = GrokVideoProvider::builder().build()?;
let video = video_provider
    .generate(&VideoGenerationRequest::new("Ocean waves").with_duration(5))
    .await?;
video.save("waves.mp4")?;
```

All providers implement the same `ImageProvider` / `VideoProvider` traits, so switching is a one-line change.

<details>
<summary>Feature flags</summary>

All providers are enabled by default. To slim down compile times:

```toml
[dependencies]
genviz = { version = "0.2", default-features = false, features = ["gemini-image", "grok-video"] }
```

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

<details>
<summary>Error handling</summary>

Errors are typed and actionable:

```rust
use genviz::GenVizError;

match provider.generate(&request).await {
    Ok(image) => image.save("out.png")?,
    Err(GenVizError::RateLimited { retry_after }) => { /* back off */ }
    Err(GenVizError::ContentBlocked(reason)) => { /* prompt was rejected */ }
    Err(GenVizError::Billing(msg)) => { /* account issue */ }
    Err(e) if e.is_retryable() => { /* transient, safe to retry */ }
    Err(e) => return Err(e.into()),
}
```

</details>

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
| Grok | `XAI_API_KEY` | 1-15s | yes | yes |
| [Sora](https://platform.openai.com) | `OPENAI_API_KEY` | yes | yes | - |
| [Veo](https://aistudio.google.com/apikey) | `GOOGLE_API_KEY` | - | - | resolution control |

## Documentation

- **API Reference**: [docs.rs/genviz](https://docs.rs/genviz)
- **Examples**: [`examples/`](examples/) directory
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT
