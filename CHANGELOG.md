# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
