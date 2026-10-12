# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-12

### Added
- OpenAI `gpt-image-2.5-sunburst` and `gpt-image-2.5-flare` (released 2026-09-08); Flare is the new OpenAI default
- Gemini `nano-banana-2.1` (`gemini-nano-banana-2.1`) and `nano-banana-2` (`gemini-3.1-flash-image`)
- Grok `grok-imagine-image-2.0` (new Grok default)
- Flux `flux-2-pro-preview` (new Flux default) and `flux-2-klein-9b-preview`
- Multi-reference input: `GenerationRequest::with_input_image` now appends; CLI `--input` is repeatable. OpenAI, Gemini, Grok (≤5) and FLUX.2 (klein ≤4, others ≤8) receive every image; single-image providers fail with a clear error
- CLI `--background transparent|opaque|auto` and `--quality low|medium|high|xhigh|max|auto` (OpenAI only)
- `genviz sprite <strip> --frames N [--rows R] --cell WxH -o out.webp [--key auto|none]` and the `sprite` feature / `genviz::sprite` module: slices figures at transparent gaps, keeps a shared baseline, one shared scale, packs a lossless WebP strip, reports `loopDelta` and `heightSpread`

### Changed
- **BREAKING** — `genviz image --json` prints one line `{"path","width","height","provider","model","durationMs"}`
- **BREAKING** — `GenerationRequest::input_image: Option<Vec<u8>>` is now `input_images: Vec<Vec<u8>>`
- Nano Banana Pro uses the GA id `gemini-3-pro-image` on both the Gemini API and Vertex AI; `GeminiModel::vertex_id` removed
- Model names are parsed by `from_name` on each model enum (shared by CLI and MCP)
- MCP `generate_image` validates parameters before checking for API keys

### Removed
- OpenAI `dall-e-3` (shut down 2026-05-12) and `gpt-image-1` (shuts down 2026-10-23)
- Flux `flux-pro` (FLUX.1 [pro] 1.0, deprecated by BFL)
- Gemini `nano-banana-pro-preview` / `gemini-3-pro-image-preview` ids (shut down 2026-06-25)

## [0.3.0] - 2026-05-16

### Added
- **Higgsfield provider** (CLI-backed) — wraps the local `higgsfield` binary; no extra HTTP client
  - `HiggsfieldImageProvider` with 14 named image models (gpt-image-2, nano-banana-pro, soul-v2, flux-2, seedream-4.5, …) plus `Custom(String)` escape hatch
  - `HiggsfieldVideoProvider` with 15 named video models (seedance-2.0, veo3-1, veo3-1-lite, kling3-0, wan2-7, soul-cast, …) plus `Custom(String)`
  - Three surfaces: free-form `generate`, `product-photoshoot` (10 modes: product_shot, lifestyle_scene, hero_banner, ad_creative_pack, virtual_model_tryout, …), and `marketplace-cards`
  - Soul Character support: `HiggsfieldImageProviderBuilder::soul_id(id)` and `HiggsfieldVideoProviderBuilder::soul_id(id)` for identity-faithful generation
  - New feature flags `higgsfield-image` and `higgsfield-video` (added to the umbrella `image` / `video` sets)
  - New variants `ImageProviderKind::Higgsfield` and `VideoProviderKind::Higgsfield`
  - CLI: `genviz image|video --provider higgsfield --model <…>`
  - MCP: `provider: "higgsfield"` accepted on `generate_image` and `generate_video`; surfaces/models reported in `list_providers`
  - Auth is delegated to the `higgsfield` CLI session (`higgsfield auth login`); no new env vars
- `.claude/skills/genviz/references/higgsfield.md` — lazy-loaded reference (only read when the task involves Higgsfield)

### Changed
- **BREAKING** — `ImageProviderKind` and `VideoProviderKind` gain a `Higgsfield` variant. Downstream `match` arms without a `_` wildcard need updating.
- `tokio` now requires the `process` feature (transparent for users via genviz; called out for re-exporters).

## [0.2.2] - 2026-05-16

### Added
- OpenAI `gpt-image-2` image model (released 2026-04-21); promoted to default for the OpenAI provider
- Veo `veo-3.1-lite-generate-preview` video model (released 2026-03-31) — cost-efficient sibling of Veo 3.1
- fal.ai image: `recraft-v4.1`, `recraft-v4.1-pro` (released 2026-05-14)
- fal.ai video: `wan-2.7` (released 2026-04), `seedance-2.0`, `seedance-2.0-fast` (released 2026-04), `happy-horse` (HappyHorse-1.0 by Alibaba, released 2026-04-26)
- Image-to-video auto-routing for the new fal.ai video models (passing a source image transparently switches to the i2v endpoint)
- MCP `generate_video` now honors the `model` parameter for the Veo provider

### Changed
- MCP `list_providers` now reports `default_model` and `models` for the Veo video provider
- CLI/MCP model enums and help text updated to expose the new models

## [0.2.1] - 2026-02-07

### Added
- Pre-built binaries for macOS (ARM + Intel), Linux (x86_64), and Windows (x86_64)
- One-line install scripts for shell (macOS/Linux) and PowerShell (Windows)
- GitHub Actions release workflow via cargo-dist

## [0.2.0] - 2026-02-07

### Added
- OpenAI image provider (gpt-image-1, dall-e-3) with size mapping and quality options
- Sora video provider (OpenAI) with polling-based generation
- Image editing support for all providers via `with_input_image()`
- `AspectRatio` enum for type-safe aspect ratio selection
- `ImageProviderExt` and `VideoProviderExt` traits with automatic retry logic
- MCP server for Claude Code and AI agent integration
- CLI `providers` subcommand to list available providers
- Batch image generation with configurable concurrency in MCP server

### Changed
- Upgraded to `thiserror` 2.x
- Improved error sanitization to prevent base64 and API key leakage
- Grok provider now handles both URL and b64_json response formats

### Fixed
- Grok edit endpoint now uses ImageUrl struct with data URI
- Flux provider correctly uses server-provided polling URLs

## [0.1.1] - 2026-02-06

### Fixed
- Sanitize error messages to prevent base64 data leakage in logs and AI agent transcripts
- URL parameter sanitization to strip API keys from error messages

## [0.1.0] - 2026-02-06

### Added
- Initial release
- Image generation via Flux (13 models), Gemini, and Grok providers
- Video generation via Grok and Veo providers
- Unified `ImageProvider` and `VideoProvider` traits
- Builder pattern for all providers
- Typed error handling with `GenVizError`
- Feature flags for individual provider selection
- CLI binary with `image`, `video`, and `mcp` subcommands
- Format detection via magic bytes
