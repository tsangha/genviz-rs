//! MCP (Model Context Protocol) server implementation.
//!
//! Exposes image and video generation as tools that AI agents can call.

use crate::image::{GeneratedImage, GenerationRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Maximum images per request (hard limit).
const MAX_COUNT: u32 = 10;
/// Maximum concurrent generations (hard limit).
const MAX_CONCURRENCY: u32 = 5;
/// Default concurrent generations.
const DEFAULT_CONCURRENCY: u32 = 3;

/// JSON-RPC 2.0 request.
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error.
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// MCP tool definition.
#[derive(Debug, Serialize)]
struct Tool {
    name: &'static str,
    description: &'static str,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

/// Generate image tool parameters.
#[derive(Debug, Clone, Deserialize)]
struct GenerateImageParams {
    prompt: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    output_path: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    aspect_ratio: Option<String>,
    #[serde(default)]
    model: Option<String>,
    /// Number of images to generate (1-10, default 1).
    #[serde(default)]
    count: Option<u32>,
    /// Max concurrent generations (1-5, default 3).
    #[serde(default)]
    concurrency: Option<u32>,
}

/// Generate video tool parameters.
#[derive(Debug, Clone, Deserialize)]
struct GenerateVideoParams {
    prompt: String,
    #[serde(default)]
    provider: Option<String>,
    output_path: String,
    #[serde(default)]
    duration: Option<u32>,
    #[serde(default)]
    aspect_ratio: Option<String>,
    #[serde(default)]
    resolution: Option<String>,
    #[serde(default)]
    source_image_url: Option<String>,
}

/// MCP server for media generation.
pub struct McpServer {
    initialized: bool,
}

impl McpServer {
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Run the MCP server, reading from stdin and writing to stdout.
    pub async fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let response = self.handle_message(&line).await;
            if let Some(resp) = response {
                let json = serde_json::to_string(&resp).unwrap_or_else(|e| {
                    json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32603, "message": e.to_string()}}).to_string()
                });
                writeln!(stdout, "{}", json)?;
                stdout.flush()?;
            }
        }

        Ok(())
    }

    async fn handle_message(&mut self, message: &str) -> Option<JsonRpcResponse> {
        let request: JsonRpcRequest = match serde_json::from_str(message) {
            Ok(r) => r,
            Err(e) => {
                return Some(JsonRpcResponse::error(
                    Value::Null,
                    -32700,
                    format!("Parse error: {}", e),
                ));
            }
        };

        if request.jsonrpc != "2.0" {
            return Some(JsonRpcResponse::error(
                request.id.unwrap_or(Value::Null),
                -32600,
                "Invalid JSON-RPC version",
            ));
        }

        let id = request.id.clone().unwrap_or(Value::Null);

        match request.method.as_str() {
            "initialize" => Some(self.handle_initialize(id, &request.params)),
            "initialized" => {
                // Notification, no response
                None
            }
            "tools/list" => Some(self.handle_tools_list(id)),
            "tools/call" => Some(self.handle_tools_call(id, &request.params).await),
            "ping" => Some(JsonRpcResponse::success(id, json!({}))),
            _ => Some(JsonRpcResponse::error(
                id,
                -32601,
                format!("Method not found: {}", request.method),
            )),
        }
    }

    fn handle_initialize(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        self.initialized = true;

        // Log client info if provided
        if let Some(client_info) = params.get("clientInfo") {
            eprintln!(
                "[genviz-mcp] Client: {} v{}",
                client_info
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown"),
                client_info
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
        }

        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "genviz",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
    }

    fn handle_tools_list(&self, id: Value) -> JsonRpcResponse {
        let tools = vec![
            Tool {
                name: "generate_image",
                description:
                    "Generate an image from a text prompt using AI (Flux, Gemini, or Grok)",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the image to generate"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["flux", "gemini", "grok"],
                            "description": "AI provider to use (default: gemini)"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the generated image (optional, returns base64 if not provided)"
                        },
                        "width": {
                            "type": "integer",
                            "description": "Image width in pixels"
                        },
                        "height": {
                            "type": "integer",
                            "description": "Image height in pixels"
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Seed for deterministic generation"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
                            "description": "Aspect ratio (alternative to width/height)"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model variant (gemini: nano-banana, nano-banana-pro; flux: flux-pro-1.1, flux-pro, flux-dev; grok: grok-imagine)"
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Number of images to generate (1-10, default 1). Use {n} in output_path as placeholder."
                        },
                        "concurrency": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "Max concurrent generations (1-5, default 3)"
                        }
                    },
                    "required": ["prompt"]
                }),
            },
            Tool {
                name: "generate_video",
                description: "Generate a video from a text prompt using AI (Grok or Veo)",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the video to generate"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["grok", "veo"],
                            "description": "AI provider to use (default: grok)"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the generated video (required for video)"
                        },
                        "duration": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 15,
                            "description": "Video duration in seconds (1-15 for Grok)"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio (e.g., 16:9)"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Video resolution (for Veo)"
                        },
                        "source_image_url": {
                            "type": "string",
                            "description": "URL of source image for image-to-video (Grok only)"
                        }
                    },
                    "required": ["prompt", "output_path"]
                }),
            },
        ];

        JsonRpcResponse::success(id, json!({ "tools": tools }))
    }

    async fn handle_tools_call(&self, id: Value, params: &Value) -> JsonRpcResponse {
        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        match tool_name {
            "generate_image" => self.generate_image(id, arguments).await,
            "generate_video" => self.generate_video(id, arguments).await,
            _ => JsonRpcResponse::error(id, -32602, format!("Unknown tool: {}", tool_name)),
        }
    }

    async fn generate_image(&self, id: Value, arguments: Value) -> JsonRpcResponse {
        let params: GenerateImageParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(id, -32602, format!("Invalid parameters: {}", e));
            }
        };

        // Validate provider/param compatibility
        let provider_name = params.provider.as_deref().unwrap_or("gemini");
        if let Err(e) = validate_image_params(provider_name, &params) {
            return JsonRpcResponse::error(id, -32602, e);
        }

        // Validate and clamp count/concurrency
        let count = params.count.unwrap_or(1).clamp(1, MAX_COUNT);
        let concurrency = params
            .concurrency
            .unwrap_or(DEFAULT_CONCURRENCY)
            .clamp(1, MAX_CONCURRENCY);

        // Validate output_path template if count > 1
        if count > 1 {
            if let Some(path) = &params.output_path {
                if !path.contains("{n}") {
                    return JsonRpcResponse::error(
                        id,
                        -32602,
                        "output_path must contain {n} placeholder when count > 1",
                    );
                }
            }
        }

        // Build base request
        let mut request = GenerationRequest::new(&params.prompt);

        if let (Some(w), Some(h)) = (params.width, params.height) {
            request = request.with_size(w, h);
        }

        if let Some(ar) = &params.aspect_ratio {
            if let Some(ratio) = parse_aspect_ratio(ar) {
                request = request.with_aspect_ratio(ratio);
            }
        }

        let model = params.model.clone();

        // Create semaphore for concurrency limiting
        let semaphore = Arc::new(Semaphore::new(concurrency as usize));

        // Spawn generation tasks
        let mut handles = Vec::with_capacity(count as usize);

        for i in 0..count {
            let sem = Arc::clone(&semaphore);
            let req = if let Some(seed) = params.seed {
                // Vary seed for each image
                request.clone().with_seed(seed.wrapping_add(i as u64))
            } else {
                request.clone()
            };
            let provider = provider_name.to_string();
            let model = model.clone();
            let output_path = params.output_path.as_ref().map(|p| {
                if count > 1 {
                    p.replace("{n}", &i.to_string())
                } else {
                    p.clone()
                }
            });

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                generate_single(&req, &provider, model.as_deref(), output_path, i).await
            });

            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::with_capacity(count as usize);
        let mut errors = Vec::new();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => errors.push(format!("Image {}: {}", i, e)),
                Err(e) => errors.push(format!("Image {}: task failed: {}", i, e)),
            }
        }

        // Build response
        if results.is_empty() {
            let content = json!([{
                "type": "text",
                "text": format!("All {} generations failed:\n{}", count, errors.join("\n"))
            }]);
            return JsonRpcResponse::success(id, json!({ "content": content, "isError": true }));
        }

        let response = json!({
            "success": true,
            "requested": count,
            "succeeded": results.len(),
            "failed": errors.len(),
            "images": results,
            "errors": if errors.is_empty() { None } else { Some(errors) }
        });

        let content = json!([{
            "type": "text",
            "text": serde_json::to_string_pretty(&response).unwrap_or_default()
        }]);

        JsonRpcResponse::success(id, json!({ "content": content }))
    }

    async fn generate_video(&self, id: Value, arguments: Value) -> JsonRpcResponse {
        let params: GenerateVideoParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(id, -32602, format!("Invalid parameters: {}", e));
            }
        };

        let provider_name = params.provider.as_deref().unwrap_or("grok");

        let result = match provider_name {
            "grok" => generate_video_with_grok(&params).await,
            "veo" => generate_video_with_veo(&params).await,
            _ => Err(format!("Unknown video provider: {}", provider_name)),
        };

        match result {
            Ok(video_result) => {
                let content = json!([{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&video_result).unwrap_or_default()
                }]);
                JsonRpcResponse::success(id, json!({ "content": content }))
            }
            Err(e) => {
                let content = json!([{
                    "type": "text",
                    "text": format!("Video generation failed: {}", e)
                }]);
                JsonRpcResponse::success(id, json!({ "content": content, "isError": true }))
            }
        }
    }
}

/// Result of a single image generation.
#[derive(Debug, Serialize)]
struct GenerationResult {
    index: u32,
    provider: String,
    format: String,
    model: Option<String>,
    seed: Option<u64>,
    duration_ms: Option<u64>,
    output: Value,
}

/// Generate a single image (called from spawned task).
async fn generate_single(
    request: &GenerationRequest,
    provider: &str,
    model: Option<&str>,
    output_path: Option<String>,
    index: u32,
) -> Result<GenerationResult, String> {
    let image = match provider {
        "flux" => generate_with_flux(request, model).await?,
        "gemini" => generate_with_gemini(request, model).await?,
        "grok" => generate_with_grok(request, model).await?,
        _ => return Err(format!("Unknown provider: {}", provider)),
    };

    let output = if let Some(path) = &output_path {
        image
            .save(path)
            .map_err(|e| format!("Failed to save: {}", e))?;
        json!({
            "saved_to": path,
            "size_bytes": image.size()
        })
    } else {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&image.data);
        json!({
            "base64": b64,
            "mime_type": image.format.mime_type(),
            "size_bytes": image.size()
        })
    };

    Ok(GenerationResult {
        index,
        provider: image.provider.to_string(),
        format: image.format.extension().to_string(),
        model: image.metadata.model,
        seed: image.metadata.seed,
        duration_ms: image.metadata.duration_ms,
        output,
    })
}

#[cfg(feature = "flux-image")]
async fn generate_with_flux(
    request: &GenerationRequest,
    model: Option<&str>,
) -> Result<GeneratedImage, String> {
    use crate::image::ImageProvider;
    use crate::{FluxModel, FluxProvider};

    let mut builder = FluxProvider::builder();

    if let Some(m) = model {
        let flux_model = match m {
            "flux-pro-1.1" => FluxModel::FluxPro11,
            "flux-pro" => FluxModel::FluxPro,
            "flux-dev" => FluxModel::FluxDev,
            _ => return Err(format!("Unknown Flux model: {}", m)),
        };
        builder = builder.model(flux_model);
    }

    let provider = builder.build().map_err(|e| e.to_string())?;
    provider.generate(request).await.map_err(|e| e.to_string())
}

#[cfg(not(feature = "flux-image"))]
async fn generate_with_flux(
    _request: &GenerationRequest,
    _model: Option<&str>,
) -> Result<GeneratedImage, String> {
    Err("Flux provider not enabled".to_string())
}

#[cfg(feature = "gemini-image")]
async fn generate_with_gemini(
    request: &GenerationRequest,
    model: Option<&str>,
) -> Result<GeneratedImage, String> {
    use crate::image::ImageProvider;
    use crate::{GeminiModel, GeminiProvider};

    let mut builder = GeminiProvider::builder();

    if let Some(m) = model {
        let gemini_model = match m {
            "nano-banana" => GeminiModel::NanoBanana,
            "nano-banana-pro" => GeminiModel::NanoBananaPro,
            _ => return Err(format!("Unknown Gemini model: {}", m)),
        };
        builder = builder.model(gemini_model);
    }

    let provider = builder.build().map_err(|e| e.to_string())?;
    provider.generate(request).await.map_err(|e| e.to_string())
}

#[cfg(not(feature = "gemini-image"))]
async fn generate_with_gemini(
    _request: &GenerationRequest,
    _model: Option<&str>,
) -> Result<GeneratedImage, String> {
    Err("Gemini provider not enabled".to_string())
}

#[cfg(feature = "grok-image")]
async fn generate_with_grok(
    request: &GenerationRequest,
    model: Option<&str>,
) -> Result<GeneratedImage, String> {
    use crate::image::ImageProvider;
    use crate::{GrokModel, GrokProvider};

    let mut builder = GrokProvider::builder();

    if let Some(m) = model {
        let grok_model = match m {
            "grok-imagine" => GrokModel::GrokImagine,
            _ => return Err(format!("Unknown Grok model: {}", m)),
        };
        builder = builder.model(grok_model);
    }

    let provider = builder.build().map_err(|e| e.to_string())?;
    provider.generate(request).await.map_err(|e| e.to_string())
}

#[cfg(not(feature = "grok-image"))]
async fn generate_with_grok(
    _request: &GenerationRequest,
    _model: Option<&str>,
) -> Result<GeneratedImage, String> {
    Err("Grok provider not enabled".to_string())
}

fn parse_aspect_ratio(s: &str) -> Option<crate::image::AspectRatio> {
    use crate::image::AspectRatio;
    match s {
        "1:1" => Some(AspectRatio::Square),
        "16:9" => Some(AspectRatio::Landscape),
        "9:16" => Some(AspectRatio::Portrait),
        "4:3" => Some(AspectRatio::Standard),
        "3:4" => Some(AspectRatio::StandardPortrait),
        "21:9" => Some(AspectRatio::Ultrawide),
        _ => None,
    }
}

/// Result of video generation.
#[derive(Debug, Serialize)]
struct VideoGenerationResult {
    success: bool,
    provider: String,
    output_path: String,
    size_bytes: usize,
    model: Option<String>,
    duration_ms: Option<u64>,
    video_duration_secs: Option<u32>,
}

#[cfg(feature = "grok-video")]
async fn generate_video_with_grok(
    params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    use crate::video::{VideoGenerationRequest, VideoProvider};
    use crate::GrokVideoProvider;

    let mut request = VideoGenerationRequest::new(&params.prompt);

    if let Some(d) = params.duration {
        request = request.with_duration(d);
    }
    if let Some(ar) = &params.aspect_ratio {
        request = request.with_aspect_ratio(ar.clone());
    }
    if let Some(url) = &params.source_image_url {
        request = request.with_source_image(url.clone());
    }

    let provider = GrokVideoProvider::builder()
        .build()
        .map_err(|e| e.to_string())?;

    let video = provider
        .generate(&request)
        .await
        .map_err(|e| e.to_string())?;

    video.save(&params.output_path).map_err(|e| e.to_string())?;

    Ok(VideoGenerationResult {
        success: true,
        provider: "grok".to_string(),
        output_path: params.output_path.clone(),
        size_bytes: video.size(),
        model: video.metadata.model,
        duration_ms: video.metadata.duration_ms,
        video_duration_secs: video.metadata.video_duration_secs,
    })
}

#[cfg(not(feature = "grok-video"))]
async fn generate_video_with_grok(
    _params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    Err("Grok video provider not enabled".to_string())
}

#[cfg(feature = "veo")]
async fn generate_video_with_veo(
    params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    use crate::video::{VideoGenerationRequest, VideoProvider};
    use crate::VeoProvider;

    let mut request = VideoGenerationRequest::new(&params.prompt);

    if let Some(res) = &params.resolution {
        request = request.with_resolution(res.clone());
    }

    let provider = VeoProvider::builder().build().map_err(|e| e.to_string())?;

    let video = provider
        .generate(&request)
        .await
        .map_err(|e| e.to_string())?;

    video.save(&params.output_path).map_err(|e| e.to_string())?;

    Ok(VideoGenerationResult {
        success: true,
        provider: "veo".to_string(),
        output_path: params.output_path.clone(),
        size_bytes: video.size(),
        model: video.metadata.model,
        duration_ms: video.metadata.duration_ms,
        video_duration_secs: video.metadata.video_duration_secs,
    })
}

#[cfg(not(feature = "veo"))]
async fn generate_video_with_veo(
    _params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    Err("Veo video provider not enabled".to_string())
}

/// Validate image generation params against provider capabilities.
fn validate_image_params(provider: &str, params: &GenerateImageParams) -> Result<(), String> {
    match provider {
        "gemini" => {
            if params.aspect_ratio.is_some() {
                return Err("Gemini does not support aspect_ratio".to_string());
            }
            if params.width.is_some() || params.height.is_some() {
                return Err("Gemini does not support width/height".to_string());
            }
        }
        "grok" => {
            if params.width.is_some() || params.height.is_some() {
                return Err("Grok does not support width/height (use aspect_ratio instead)".to_string());
            }
            if params.seed.is_some() {
                return Err("Grok does not support seed".to_string());
            }
        }
        "flux" => {
            // Flux supports all options
        }
        _ => {
            return Err(format!("Unknown provider: {}", provider));
        }
    }
    Ok(())
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}
