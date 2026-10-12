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

Works with [Gemini](https://aistudio.google.com/apikey), [Flux](https://api.bfl.ai) (FLUX.2 pro/klein previews + 12 more), [Grok](https://x.ai) (Imagine Image 2.0), [OpenAI](https://platform.openai.com) (gpt-image-2.5 Sunburst/Flare, gpt-image-2), [Sora](https://platform.openai.com), [Veo](https://aistudio.google.com/apikey) (3.1 + 3.1 Lite), [Kling](https://app.klingai.com), [fal.ai](https://fal.ai) (Flux, Recraft V4.1, Wan 2.7, Seedance 2.0, Happy Horse, Hailuo 2.3, Kling, LTX, …), [MiniMax](https://platform.minimax.io) (Hailuo 2.3), and [Higgsfield](https://higgsfield.ai) (GPT Image 2, Nano Banana Pro, Soul V2, Seedance 2.0, Veo 3.1, Kling 3.0 + product-photoshoot / marketplace-card / Soul ID workflows). You only need an API key for the provider you want to use (Higgsfield uses the local `higgsfield` CLI session).

## CLI

```bash
# Generate images
genviz image "A sunset over mountains" -o sunset.png
genviz image "A portrait" -o portrait.png -p openai --model gpt-image-2.5-sunburst --quality high
genviz image "A city skyline" -o city.png -p flux --model flux-2-pro-preview --aspect-ratio 16:9

# Edit existing images (repeat --input for multi-reference: OpenAI, Gemini, Grok, FLUX.2)
genviz image "Change the sky to sunset" -o edited.png --input photo.png
genviz image "Put the hat on the dog" -o dog.png -p openai --input dog.png --input hat.png

# Transparent background, one JSON line on stdout:
# {"path","width","height","provider","model","durationMs"}
genviz image "Robot mascot, 4 poses side by side" -o strip.png -p openai \
  --model gpt-image-2.5-sunburst --background transparent --width 1536 --height 512 --json

# Slice a strip into a uniform WebP sprite strip (N*W x H). Figures are split at transparent
# gaps (rows first with --rows), keep their height against a shared baseline so jumps survive,
# and share one scale. --key auto (default) removes a flat background from the corners when the
# strip has no transparency. Prints {"path","frames","cell","loopDelta","heightSpread"} with --json;
# exits non-zero unless exactly --frames figures are found.
genviz sprite strip.png --frames 4 --cell 256x256 -o sprite.webp --json

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
genviz = "0.4"
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
genviz = { version = "0.4", default-features = false, features = ["gemini-image", "grok-video"] }
```

| Flag | Default | What it enables |
|------|---------|-----------------|
| `flux-image` | yes | Flux (14 models) |
| `gemini-image` | yes | Gemini |
| `grok-image` | yes | Grok Imagine |
| `openai-image` | yes | OpenAI (gpt-image-2.5-sunburst, gpt-image-2.5-flare, gpt-image-2) |
| `higgsfield-image` | yes | Higgsfield image (CLI-backed; gpt-image-2, nano-banana-pro, soul-v2, …) |
| `higgsfield-video` | yes | Higgsfield video (CLI-backed; seedance-2.0, veo3-1-lite, kling3-0, soul-cast, …) |
| `grok-video` | yes | Grok video |
| `openai-video` | yes | Sora |
| `veo` | yes | Veo |
| `sprite` | no | Sprite strip packing (`genviz::sprite`, pure-Rust `image` crate) |
| `cli` | no | CLI binary (includes `sprite`) |

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
| [Gemini](https://aistudio.google.com/apikey) | `NanoBananaPro` (`gemini-3-pro-image`), `NanoBanana21` (`gemini-nano-banana-2.1`), `NanoBanana2` (`gemini-3.1-flash-image`), `NanoBanana` | `GOOGLE_API_KEY` | yes, multi-reference | - | yes |
| [Flux](https://api.bfl.ai) | 14 models (`flux-2-pro-preview` default, `flux-2-klein-9b-preview`, FLUX.2, FLUX1.1, Kontext, Fill, Expand) | `BFL_API_KEY` | yes, FLUX.2 multi-reference (klein 4, others 8) | yes | yes |
| [Grok](https://x.ai) | `GrokImagineImage2` (`grok-imagine-image-2.0`, default), `GrokImagine` | `XAI_API_KEY` | yes, up to 5 references | yes | - |
| [OpenAI](https://platform.openai.com) | `GptImage25Sunburst`, `GptImage25Flare` (default), `GptImage2` | `OPENAI_API_KEY` | yes, multi-reference; `--background`, `--quality` | yes | - |

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
