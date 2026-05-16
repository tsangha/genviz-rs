# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
