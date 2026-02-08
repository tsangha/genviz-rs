//! MCP (Model Context Protocol) server implementation.
//!
//! Exposes image and video generation as tools that AI agents can call.

use crate::image::{GeneratedImage, GenerationRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Maximum images per request (hard limit).
const MAX_COUNT: u32 = 10;

/// Timeout for a single image generation (including provider polling).
const IMAGE_GENERATION_TIMEOUT: Duration = Duration::from_secs(600);

/// Timeout for video generation (including provider polling).
const VIDEO_GENERATION_TIMEOUT: Duration = Duration::from_secs(600);

/// Decodes a base64 string that may be imperfectly formatted.
///
/// LLMs frequently send base64 with issues that strict decoders reject:
/// - Data URI prefix (`data:image/png;base64,...`)
/// - Missing padding (`=` characters)
/// - Embedded whitespace or newlines
///
/// This function normalizes all of these before decoding.
fn decode_base64_lenient(input: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::Engine;

    // Strip data URI prefix if present (e.g. "data:image/png;base64,")
    let b64 = match input.find(";base64,") {
        Some(pos) => &input[pos + 8..],
        None => input,
    };

    // Strip whitespace (newlines, spaces, tabs)
    let cleaned: String = b64.chars().filter(|c| !c.is_ascii_whitespace()).collect();

    // Try standard decoding first (fast path)
    if let Ok(data) = base64::engine::general_purpose::STANDARD.decode(&cleaned) {
        return Ok(data);
    }

    // Fall back to no-pad decoding (handles missing `=`)
    base64::engine::general_purpose::STANDARD_NO_PAD.decode(&cleaned)
}
/// Returns the environment variable name for a provider's API key.
///
/// Used for pre-flight checks before spawning generation tasks,
/// so missing keys fail fast with a clear error.
fn api_key_env_var(provider: &str) -> Option<&'static str> {
    match provider {
        "gemini" | "veo" => Some("GOOGLE_API_KEY"),
        "flux" => Some("BFL_API_KEY"),
        "grok" => Some("XAI_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        "kling" => Some("KLING_ACCESS_KEY"),
        "fal" => Some("FAL_KEY"),
        _ => None,
    }
}

/// Maximum concurrent generations (hard limit).
const MAX_CONCURRENCY: u32 = 5;
/// Default concurrent generations.
const DEFAULT_CONCURRENCY: u32 = 3;

/// Validates that an output path is safe to write to.
///
/// Rejects paths containing directory traversal (`..`) components
/// to prevent writing outside the intended directory.
fn validate_output_path(path: &str) -> std::result::Result<(), String> {
    let path = std::path::Path::new(path);
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err("Path must not contain '..' components".into());
        }
    }
    Ok(())
}

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
    /// Input image for editing (base64 encoded).
    #[serde(default)]
    input_image: Option<String>,
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
                name: "list_providers",
                description: "List available image and video providers with their models, capabilities, and API key status",
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
            Tool {
                name: "generate_image",
                description:
                    "Generate an image from a text prompt using AI (Flux, Gemini, Grok, Kling, fal.ai, or OpenAI). Call list_providers first to see available models and which API keys are configured.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the image to generate"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["flux", "gemini", "grok", "openai", "kling", "fal"],
                            "description": "AI provider to use (default: gemini)"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the generated image (optional, returns base64 if not provided)"
                        },
                        "width": {
                            "type": "integer",
                            "description": "Image width in pixels (Flux and OpenAI only)"
                        },
                        "height": {
                            "type": "integer",
                            "description": "Image height in pixels (Flux and OpenAI only)"
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Seed for deterministic generation (Gemini and Flux only)"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "3:2", "2:3"],
                            "description": "Aspect ratio (Flux, Grok, and OpenAI only)"
                        },
                        "model": {
                            "type": "string",
                            "enum": [
                                "nano-banana", "nano-banana-pro",
                                "flux-pro-1.1", "flux-pro-1.1-ultra", "flux-pro", "flux-dev",
                                "flux-2-max", "flux-2-pro", "flux-2-flex",
                                "flux-2-klein-4b", "flux-2-klein-9b",
                                "flux-kontext-pro", "flux-kontext-max",
                                "flux-fill-pro", "flux-expand-pro",
                                "grok-imagine",
                                "kling-v1", "kling-v1.5", "kling-v2",
                                "gpt-image-1", "dall-e-3",
                                "flux-schnell", "flux-pro-ultra", "recraft-v3", "ideogram-v3", "hidream"
                            ],
                            "description": "Model variant. Must match the selected provider. Call list_providers to see which models belong to which provider."
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
                        },
                        "input_image": {
                            "type": "string",
                            "description": "Base64-encoded input image for editing. Accepts raw base64 or data URIs (e.g. data:image/png;base64,...). All providers support this."
                        }
                    },
                    "required": ["prompt"]
                }),
            },
            Tool {
                name: "generate_video",
                description:
                    "Generate a video from a text prompt using AI (Grok, Kling, fal.ai, OpenAI/Sora, or Veo). Call list_providers first to see available providers and which API keys are configured.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the video to generate"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["grok", "openai", "veo", "kling", "fal"],
                            "description": "AI provider to use (default: grok)"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the generated video"
                        },
                        "duration": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 15,
                            "description": "Video duration in seconds (Grok: 1-15, Sora: varies)"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio (e.g., 16:9). Supported by Grok and Sora."
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Video resolution (Veo only, e.g., 720p)"
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
            "list_providers" => self.list_providers(id),
            "generate_image" => self.generate_image(id, arguments).await,
            "generate_video" => self.generate_video(id, arguments).await,
            _ => JsonRpcResponse::error(id, -32602, format!("Unknown tool: {}", tool_name)),
        }
    }

    fn list_providers(&self, id: Value) -> JsonRpcResponse {
        let check_key = |var: &str| -> bool { std::env::var(var).is_ok() };

        let providers = json!({
            "image_providers": [
                {
                    "name": "gemini",
                    "api_key_env": "GOOGLE_API_KEY",
                    "api_key_set": check_key("GOOGLE_API_KEY"),
                    "default_model": "nano-banana-pro",
                    "models": ["nano-banana", "nano-banana-pro"],
                    "capabilities": {
                        "editing": true,
                        "aspect_ratio": false,
                        "width_height": false,
                        "seed": true
                    }
                },
                {
                    "name": "flux",
                    "api_key_env": "BFL_API_KEY",
                    "api_key_set": check_key("BFL_API_KEY"),
                    "default_model": "flux-pro-1.1",
                    "models": [
                        "flux-pro-1.1", "flux-pro-1.1-ultra", "flux-pro", "flux-dev",
                        "flux-2-max", "flux-2-pro", "flux-2-flex",
                        "flux-2-klein-4b", "flux-2-klein-9b",
                        "flux-kontext-pro", "flux-kontext-max",
                        "flux-fill-pro", "flux-expand-pro"
                    ],
                    "capabilities": {
                        "editing": true,
                        "aspect_ratio": true,
                        "width_height": true,
                        "seed": true
                    }
                },
                {
                    "name": "grok",
                    "api_key_env": "XAI_API_KEY",
                    "api_key_set": check_key("XAI_API_KEY"),
                    "default_model": "grok-imagine",
                    "models": ["grok-imagine"],
                    "capabilities": {
                        "editing": true,
                        "aspect_ratio": true,
                        "width_height": false,
                        "seed": false
                    }
                },
                {
                    "name": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "api_key_set": check_key("OPENAI_API_KEY"),
                    "default_model": "gpt-image-1",
                    "models": ["gpt-image-1", "dall-e-3"],
                    "capabilities": {
                        "editing": true,
                        "aspect_ratio": true,
                        "width_height": true,
                        "seed": false
                    }
                },
                {
                    "name": "kling",
                    "api_key_env": "KLING_ACCESS_KEY",
                    "api_key_set": check_key("KLING_ACCESS_KEY"),
                    "default_model": "kling-v2",
                    "models": ["kling-v1", "kling-v1.5", "kling-v2"],
                    "capabilities": {
                        "editing": true,
                        "aspect_ratio": true,
                        "width_height": false,
                        "seed": false
                    }
                },
                {
                    "name": "fal",
                    "api_key_env": "FAL_KEY",
                    "api_key_set": check_key("FAL_KEY"),
                    "default_model": "flux-schnell",
                    "models": ["flux-schnell", "flux-pro", "flux-pro-ultra", "recraft-v3", "ideogram-v3", "hidream"],
                    "capabilities": {
                        "editing": true,
                        "aspect_ratio": true,
                        "width_height": true,
                        "seed": true
                    }
                }
            ],
            "video_providers": [
                {
                    "name": "grok",
                    "api_key_env": "XAI_API_KEY",
                    "api_key_set": check_key("XAI_API_KEY"),
                    "capabilities": {
                        "duration": "1-15s",
                        "aspect_ratio": true,
                        "image_to_video": true
                    }
                },
                {
                    "name": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "api_key_set": check_key("OPENAI_API_KEY"),
                    "capabilities": {
                        "duration": true,
                        "aspect_ratio": true,
                        "image_to_video": false
                    }
                },
                {
                    "name": "veo",
                    "api_key_env": "GOOGLE_API_KEY",
                    "api_key_set": check_key("GOOGLE_API_KEY"),
                    "capabilities": {
                        "duration": false,
                        "aspect_ratio": false,
                        "resolution": true,
                        "image_to_video": false
                    }
                },
                {
                    "name": "kling",
                    "api_key_env": "KLING_ACCESS_KEY",
                    "api_key_set": check_key("KLING_ACCESS_KEY"),
                    "capabilities": {
                        "duration": true,
                        "aspect_ratio": false,
                        "image_to_video": true
                    }
                },
                {
                    "name": "fal",
                    "api_key_env": "FAL_KEY",
                    "api_key_set": check_key("FAL_KEY"),
                    "capabilities": {
                        "duration": true,
                        "aspect_ratio": true,
                        "image_to_video": true
                    }
                }
            ]
        });

        let content = json!([{
            "type": "text",
            "text": serde_json::to_string_pretty(&providers).unwrap_or_default()
        }]);

        JsonRpcResponse::success(id, json!({ "content": content }))
    }

    async fn generate_image(&self, id: Value, arguments: Value) -> JsonRpcResponse {
        let params: GenerateImageParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(id, -32602, format!("Invalid parameters: {}", e));
            }
        };

        // Validate output path safety
        if let Some(path) = &params.output_path {
            if let Err(msg) = validate_output_path(path) {
                return JsonRpcResponse::error(id, -32602, &msg);
            }
        }

        // Validate provider/param compatibility
        let provider_name = params.provider.as_deref().unwrap_or("gemini");
        if let Err(e) = validate_image_params(provider_name, &params) {
            return JsonRpcResponse::error(id, -32602, e);
        }

        // Pre-flight API key check (fail fast before spawning tasks)
        if let Some(env_var) = api_key_env_var(provider_name) {
            if std::env::var(env_var).is_err() {
                return JsonRpcResponse::error(
                    id,
                    -32602,
                    format!(
                        "{} requires {} environment variable to be set",
                        provider_name, env_var
                    ),
                );
            }
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
            match parse_aspect_ratio(ar) {
                Some(ratio) => request = request.with_aspect_ratio(ratio),
                None => {
                    return JsonRpcResponse::error(
                        id,
                        -32602,
                        format!(
                            "Invalid aspect_ratio '{}'. Valid values: 1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 3:2, 2:3",
                            ar
                        ),
                    );
                }
            }
        }

        // Decode and add input image for editing
        if let Some(ref b64_image) = params.input_image {
            match decode_base64_lenient(b64_image) {
                Ok(data) => request = request.with_input_image(data),
                Err(e) => {
                    return JsonRpcResponse::error(
                        id,
                        -32602,
                        format!("Invalid base64 in input_image: {}", e),
                    );
                }
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
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|_| "semaphore closed".to_string())?;
                match tokio::time::timeout(
                    IMAGE_GENERATION_TIMEOUT,
                    generate_single(&req, &provider, model.as_deref(), output_path, i),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => Err(format!(
                        "timed out after {}s",
                        IMAGE_GENERATION_TIMEOUT.as_secs()
                    )),
                }
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
            return JsonRpcResponse::error(
                id,
                -32603,
                format!("All {} generations failed:\n{}", count, errors.join("\n")),
            );
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

        // Validate output path safety
        if let Err(msg) = validate_output_path(&params.output_path) {
            return JsonRpcResponse::error(id, -32602, &msg);
        }

        let provider_name = params.provider.as_deref().unwrap_or("grok");

        // Pre-flight API key check
        if let Some(env_var) = api_key_env_var(provider_name) {
            if std::env::var(env_var).is_err() {
                return JsonRpcResponse::error(
                    id,
                    -32602,
                    format!(
                        "{} requires {} environment variable to be set",
                        provider_name, env_var
                    ),
                );
            }
        }

        let result = match tokio::time::timeout(VIDEO_GENERATION_TIMEOUT, async {
            match provider_name {
                "grok" => generate_video_with_grok(&params).await,
                "openai" => generate_video_with_openai(&params).await,
                "veo" => generate_video_with_veo(&params).await,
                "kling" => generate_video_with_kling(&params).await,
                "fal" => generate_video_with_fal(&params).await,
                _ => Err(format!("Unknown video provider: {}", provider_name)),
            }
        })
        .await
        {
            Ok(result) => result,
            Err(_) => Err(format!(
                "timed out after {}s",
                VIDEO_GENERATION_TIMEOUT.as_secs()
            )),
        };

        match result {
            Ok(video_result) => {
                let content = json!([{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&video_result).unwrap_or_default()
                }]);
                JsonRpcResponse::success(id, json!({ "content": content }))
            }
            Err(e) => JsonRpcResponse::error(id, -32603, format!("Video generation failed: {}", e)),
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
        "openai" => generate_with_openai(request, model).await?,
        "kling" => generate_with_kling(request, model).await?,
        "fal" => generate_with_fal(request, model).await?,
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
            "flux-pro-1.1-ultra" => FluxModel::FluxPro11Ultra,
            "flux-pro" => FluxModel::FluxPro,
            "flux-dev" => FluxModel::FluxDev,
            "flux-2-max" | "flux2-max" => FluxModel::Flux2Max,
            "flux-2-pro" | "flux2-pro" => FluxModel::Flux2Pro,
            "flux-2-flex" | "flux2-flex" => FluxModel::Flux2Flex,
            "flux-2-klein-4b" | "flux2-klein-4b" => FluxModel::Flux2Klein4B,
            "flux-2-klein-9b" | "flux2-klein-9b" => FluxModel::Flux2Klein9B,
            "flux-kontext-pro" | "kontext-pro" => FluxModel::KontextPro,
            "flux-kontext-max" | "kontext-max" => FluxModel::KontextMax,
            "flux-fill-pro" | "fill-pro" => FluxModel::FillPro,
            "flux-expand-pro" | "expand-pro" => FluxModel::ExpandPro,
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

#[cfg(feature = "openai-image")]
async fn generate_with_openai(
    request: &GenerationRequest,
    model: Option<&str>,
) -> Result<GeneratedImage, String> {
    use crate::image::ImageProvider;
    use crate::{OpenAiImageModel, OpenAiImageProvider};

    let mut builder = OpenAiImageProvider::builder();

    if let Some(m) = model {
        let openai_model = match m {
            "gpt-image-1" => OpenAiImageModel::GptImage1,
            "dall-e-3" => OpenAiImageModel::DallE3,
            _ => return Err(format!("Unknown OpenAI model: {}", m)),
        };
        builder = builder.model(openai_model);
    }

    let provider = builder.build().map_err(|e| e.to_string())?;
    provider.generate(request).await.map_err(|e| e.to_string())
}

#[cfg(not(feature = "openai-image"))]
async fn generate_with_openai(
    _request: &GenerationRequest,
    _model: Option<&str>,
) -> Result<GeneratedImage, String> {
    Err("OpenAI image provider not enabled".to_string())
}

#[cfg(feature = "kling-image")]
async fn generate_with_kling(
    request: &GenerationRequest,
    model: Option<&str>,
) -> Result<GeneratedImage, String> {
    use crate::image::ImageProvider;
    use crate::{KlingImageModel, KlingImageProvider};

    let mut builder = KlingImageProvider::builder();

    if let Some(m) = model {
        let kling_model = match m {
            "kling-v1" => KlingImageModel::KlingV1,
            "kling-v1.5" | "kling-v1-5" => KlingImageModel::KlingV1_5,
            "kling-v2" => KlingImageModel::KlingV2,
            _ => return Err(format!("Unknown Kling model: {}", m)),
        };
        builder = builder.model(kling_model);
    }

    let provider = builder.build().map_err(|e| e.to_string())?;
    provider.generate(request).await.map_err(|e| e.to_string())
}

#[cfg(not(feature = "kling-image"))]
async fn generate_with_kling(
    _request: &GenerationRequest,
    _model: Option<&str>,
) -> Result<GeneratedImage, String> {
    Err("Kling image provider not enabled".to_string())
}

#[cfg(feature = "fal-image")]
async fn generate_with_fal(
    request: &GenerationRequest,
    model: Option<&str>,
) -> Result<GeneratedImage, String> {
    use crate::image::ImageProvider;
    use crate::{FalImageModel, FalImageProvider};

    let mut builder = FalImageProvider::builder();

    if let Some(m) = model {
        let fal_model = match m {
            "flux-schnell" => FalImageModel::FluxSchnell,
            "flux-pro" => FalImageModel::FluxPro,
            "flux-pro-ultra" => FalImageModel::FluxProUltra,
            "recraft-v3" => FalImageModel::RecraftV3,
            "ideogram-v3" => FalImageModel::Ideogram3,
            "hidream" => FalImageModel::HiDream,
            s if s.starts_with("fal-ai/") => FalImageModel::Custom(s.to_string()),
            _ => return Err(format!("Unknown fal.ai model: {}", m)),
        };
        builder = builder.model(fal_model);
    }

    let provider = builder.build().map_err(|e| e.to_string())?;
    provider.generate(request).await.map_err(|e| e.to_string())
}

#[cfg(not(feature = "fal-image"))]
async fn generate_with_fal(
    _request: &GenerationRequest,
    _model: Option<&str>,
) -> Result<GeneratedImage, String> {
    Err("fal.ai image provider not enabled".to_string())
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
        "3:2" => Some(AspectRatio::ThreeTwo),
        "2:3" => Some(AspectRatio::TwoThree),
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

#[cfg(feature = "openai-video")]
async fn generate_video_with_openai(
    params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    use crate::video::{VideoGenerationRequest, VideoProvider};
    use crate::SoraProvider;

    let mut request = VideoGenerationRequest::new(&params.prompt);

    if let Some(d) = params.duration {
        request = request.with_duration(d);
    }
    if let Some(ar) = &params.aspect_ratio {
        request = request.with_aspect_ratio(ar.clone());
    }

    let provider = SoraProvider::builder().build().map_err(|e| e.to_string())?;

    let video = provider
        .generate(&request)
        .await
        .map_err(|e| e.to_string())?;

    video.save(&params.output_path).map_err(|e| e.to_string())?;

    Ok(VideoGenerationResult {
        success: true,
        provider: "openai".to_string(),
        output_path: params.output_path.clone(),
        size_bytes: video.size(),
        model: video.metadata.model,
        duration_ms: video.metadata.duration_ms,
        video_duration_secs: video.metadata.video_duration_secs,
    })
}

#[cfg(not(feature = "openai-video"))]
async fn generate_video_with_openai(
    _params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    Err("OpenAI video (Sora) provider not enabled".to_string())
}

#[cfg(feature = "kling-video")]
async fn generate_video_with_kling(
    params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    use crate::video::{VideoGenerationRequest, VideoProvider};
    use crate::KlingVideoProvider;

    let mut request = VideoGenerationRequest::new(&params.prompt);

    if let Some(d) = params.duration {
        request = request.with_duration(d);
    }
    if let Some(url) = &params.source_image_url {
        request = request.with_source_image(url.clone());
    }

    let provider = KlingVideoProvider::builder()
        .build()
        .map_err(|e| e.to_string())?;

    let video = provider
        .generate(&request)
        .await
        .map_err(|e| e.to_string())?;

    video.save(&params.output_path).map_err(|e| e.to_string())?;

    Ok(VideoGenerationResult {
        success: true,
        provider: "kling".to_string(),
        output_path: params.output_path.clone(),
        size_bytes: video.size(),
        model: video.metadata.model,
        duration_ms: video.metadata.duration_ms,
        video_duration_secs: video.metadata.video_duration_secs,
    })
}

#[cfg(not(feature = "kling-video"))]
async fn generate_video_with_kling(
    _params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    Err("Kling video provider not enabled".to_string())
}

#[cfg(feature = "fal-video")]
async fn generate_video_with_fal(
    params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    use crate::video::{VideoGenerationRequest, VideoProvider};
    use crate::FalVideoProvider;

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

    let provider = FalVideoProvider::builder()
        .build()
        .map_err(|e| e.to_string())?;

    let video = provider
        .generate(&request)
        .await
        .map_err(|e| e.to_string())?;

    video.save(&params.output_path).map_err(|e| e.to_string())?;

    Ok(VideoGenerationResult {
        success: true,
        provider: "fal".to_string(),
        output_path: params.output_path.clone(),
        size_bytes: video.size(),
        model: video.metadata.model,
        duration_ms: video.metadata.duration_ms,
        video_duration_secs: video.metadata.video_duration_secs,
    })
}

#[cfg(not(feature = "fal-video"))]
async fn generate_video_with_fal(
    _params: &GenerateVideoParams,
) -> Result<VideoGenerationResult, String> {
    Err("fal.ai video provider not enabled".to_string())
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
                return Err(
                    "Grok does not support width/height (use aspect_ratio instead)".to_string(),
                );
            }
            if params.seed.is_some() {
                return Err("Grok does not support seed".to_string());
            }
        }
        "openai" => {
            if params.seed.is_some() {
                return Err("OpenAI does not support seed".to_string());
            }
        }
        "flux" => {
            // Flux supports all options
        }
        "fal" => {
            // fal.ai supports all options
        }
        "kling" => {
            if params.width.is_some() || params.height.is_some() {
                return Err(
                    "Kling does not support width/height (use aspect_ratio instead)".to_string(),
                );
            }
            if params.seed.is_some() {
                return Err("Kling does not support seed".to_string());
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> McpServer {
        McpServer::new()
    }

    #[tokio::test]
    async fn test_initialize() {
        let mut server = make_server();
        let resp = server
            .handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#)
            .await
            .unwrap();

        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "genviz");
        assert!(server.initialized);
    }

    #[tokio::test]
    async fn test_ping() {
        let mut server = make_server();
        let resp = server
            .handle_message(r#"{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}"#)
            .await
            .unwrap();

        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn test_tools_list() {
        let server = make_server();
        let resp = server.handle_tools_list(json!(1));

        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);

        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"list_providers"));
        assert!(names.contains(&"generate_image"));
        assert!(names.contains(&"generate_video"));
    }

    #[tokio::test]
    async fn test_list_providers() {
        let server = make_server();
        let resp = server.list_providers(json!(1));

        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        let content = result["content"].as_array().unwrap();
        let text = content[0]["text"].as_str().unwrap();
        let providers: Value = serde_json::from_str(text).unwrap();

        // Verify structure
        let image_providers = providers["image_providers"].as_array().unwrap();
        let video_providers = providers["video_providers"].as_array().unwrap();
        assert_eq!(image_providers.len(), 6);
        assert_eq!(video_providers.len(), 5);

        // Verify each image provider has required fields
        for p in image_providers {
            assert!(p["name"].is_string());
            assert!(p["api_key_env"].is_string());
            assert!(p["api_key_set"].is_boolean());
            assert!(p["models"].is_array());
            assert!(p["capabilities"].is_object());
        }

        // Verify provider names
        let names: Vec<&str> = image_providers
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec!["gemini", "flux", "grok", "openai", "kling", "fal"]
        );
    }

    #[tokio::test]
    async fn test_generate_image_model_enum() {
        // Verify model field has enum constraint in schema
        let server = make_server();
        let resp = server.handle_tools_list(json!(1));
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        let image_tool = tools
            .iter()
            .find(|t| t["name"] == "generate_image")
            .unwrap();
        let model_schema = &image_tool["inputSchema"]["properties"]["model"];
        let model_enum = model_schema["enum"].as_array().unwrap();

        // Should contain models from all providers
        let models: Vec<&str> = model_enum.iter().map(|m| m.as_str().unwrap()).collect();
        assert!(models.contains(&"nano-banana-pro")); // gemini
        assert!(models.contains(&"flux-2-max")); // flux
        assert!(models.contains(&"grok-imagine")); // grok
        assert!(models.contains(&"gpt-image-1")); // openai
        assert!(models.contains(&"dall-e-3")); // openai
        assert!(models.contains(&"flux-schnell")); // fal
    }

    #[tokio::test]
    async fn test_invalid_jsonrpc_version() {
        let mut server = make_server();
        let resp = server
            .handle_message(r#"{"jsonrpc":"1.0","id":1,"method":"ping","params":{}}"#)
            .await
            .unwrap();

        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32600);
    }

    #[tokio::test]
    async fn test_parse_error() {
        let mut server = make_server();
        let resp = server.handle_message("not json").await.unwrap();

        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32700);
    }

    #[tokio::test]
    async fn test_method_not_found() {
        let mut server = make_server();
        let resp = server
            .handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"nonexistent","params":{}}"#)
            .await
            .unwrap();

        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_initialized_notification_returns_none() {
        let mut server = make_server();
        let resp = server
            .handle_message(r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#)
            .await;

        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let server = make_server();
        let resp = server
            .handle_tools_call(
                json!(1),
                &json!({"name": "nonexistent_tool", "arguments": {}}),
            )
            .await;

        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_generate_image_invalid_params() {
        let server = make_server();
        let resp = server
            .generate_image(json!(1), json!({"not_prompt": "missing required field"}))
            .await;

        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_generate_image_unknown_provider() {
        let server = make_server();
        let resp = server
            .generate_image(json!(1), json!({"prompt": "test", "provider": "unknown"}))
            .await;

        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_generate_image_count_requires_placeholder() {
        let server = make_server();
        let resp = server
            .generate_image(
                json!(1),
                json!({"prompt": "test", "count": 3, "output_path": "/tmp/out.png"}),
            )
            .await;

        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("{n}"));
    }

    #[test]
    fn test_validate_image_params_gemini_rejects_aspect_ratio() {
        let params = GenerateImageParams {
            prompt: "test".into(),
            provider: None,
            output_path: None,
            width: None,
            height: None,
            seed: None,
            aspect_ratio: Some("16:9".into()),
            model: None,
            count: None,
            concurrency: None,
            input_image: None,
        };
        assert!(validate_image_params("gemini", &params).is_err());
    }

    #[test]
    fn test_validate_image_params_grok_rejects_seed() {
        let params = GenerateImageParams {
            prompt: "test".into(),
            provider: None,
            output_path: None,
            width: None,
            height: None,
            seed: Some(42),
            aspect_ratio: None,
            model: None,
            count: None,
            concurrency: None,
            input_image: None,
        };
        assert!(validate_image_params("grok", &params).is_err());
    }

    #[test]
    fn test_validate_image_params_flux_accepts_all() {
        let params = GenerateImageParams {
            prompt: "test".into(),
            provider: None,
            output_path: None,
            width: Some(1024),
            height: Some(768),
            seed: Some(42),
            aspect_ratio: Some("16:9".into()),
            model: None,
            count: None,
            concurrency: None,
            input_image: None,
        };
        assert!(validate_image_params("flux", &params).is_ok());
    }

    #[test]
    fn test_validate_image_params_fal_accepts_all() {
        let params = GenerateImageParams {
            prompt: "test".into(),
            provider: None,
            output_path: None,
            width: Some(1024),
            height: Some(768),
            seed: Some(42),
            aspect_ratio: Some("16:9".into()),
            model: None,
            count: None,
            concurrency: None,
            input_image: None,
        };
        assert!(validate_image_params("fal", &params).is_ok());
    }

    #[test]
    fn test_parse_aspect_ratio() {
        use crate::image::AspectRatio;
        assert_eq!(parse_aspect_ratio("1:1"), Some(AspectRatio::Square));
        assert_eq!(parse_aspect_ratio("16:9"), Some(AspectRatio::Landscape));
        assert_eq!(parse_aspect_ratio("9:16"), Some(AspectRatio::Portrait));
        assert_eq!(parse_aspect_ratio("4:3"), Some(AspectRatio::Standard));
        assert_eq!(
            parse_aspect_ratio("3:4"),
            Some(AspectRatio::StandardPortrait)
        );
        assert_eq!(parse_aspect_ratio("21:9"), Some(AspectRatio::Ultrawide));
        assert_eq!(parse_aspect_ratio("3:2"), Some(AspectRatio::ThreeTwo));
        assert_eq!(parse_aspect_ratio("2:3"), Some(AspectRatio::TwoThree));
        assert_eq!(parse_aspect_ratio("invalid"), None);
    }

    #[test]
    fn test_validate_output_path_rejects_traversal() {
        assert!(validate_output_path("../etc/passwd").is_err());
        assert!(validate_output_path("/tmp/../etc/passwd").is_err());
        assert!(validate_output_path("foo/../../bar").is_err());
    }

    #[test]
    fn test_validate_output_path_accepts_safe_paths() {
        assert!(validate_output_path("/tmp/output.png").is_ok());
        assert!(validate_output_path("output.png").is_ok());
        assert!(validate_output_path("./images/output.png").is_ok());
        assert!(validate_output_path("images/sub/output.png").is_ok());
    }

    #[tokio::test]
    async fn test_generate_image_rejects_path_traversal() {
        let server = make_server();
        let resp = server
            .generate_image(
                json!(1),
                json!({"prompt": "test", "output_path": "../etc/evil.png"}),
            )
            .await;

        assert!(resp.error.is_some());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32602);
        assert!(err.message.contains(".."));
    }

    #[tokio::test]
    async fn test_generate_video_rejects_path_traversal() {
        let server = make_server();
        let resp = server
            .generate_video(
                json!(1),
                json!({"prompt": "test", "output_path": "../etc/evil.mp4"}),
            )
            .await;

        assert!(resp.error.is_some());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32602);
        assert!(err.message.contains(".."));
    }

    #[test]
    fn test_decode_base64_lenient_standard() {
        // Standard padded base64
        let data = decode_base64_lenient("aGVsbG8=").unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_decode_base64_lenient_no_padding() {
        // Missing padding (the exact error Claude hits)
        let data = decode_base64_lenient("aGVsbG8").unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_decode_base64_lenient_data_uri() {
        // Data URI prefix (common from browser/screenshot tools)
        let data = decode_base64_lenient("data:image/png;base64,aGVsbG8=").unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_decode_base64_lenient_data_uri_no_padding() {
        // Data URI + missing padding (worst case combo)
        let data = decode_base64_lenient("data:image/jpeg;base64,aGVsbG8").unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_decode_base64_lenient_whitespace() {
        // Embedded newlines/spaces
        let data = decode_base64_lenient("aGVs\nbG8=").unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_decode_base64_lenient_invalid() {
        // Truly invalid base64 should still error
        assert!(decode_base64_lenient("!!!not-base64!!!").is_err());
    }

    #[test]
    fn test_api_key_env_var() {
        assert_eq!(api_key_env_var("gemini"), Some("GOOGLE_API_KEY"));
        assert_eq!(api_key_env_var("flux"), Some("BFL_API_KEY"));
        assert_eq!(api_key_env_var("grok"), Some("XAI_API_KEY"));
        assert_eq!(api_key_env_var("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(api_key_env_var("veo"), Some("GOOGLE_API_KEY"));
        assert_eq!(api_key_env_var("kling"), Some("KLING_ACCESS_KEY"));
        assert_eq!(api_key_env_var("fal"), Some("FAL_KEY"));
        assert_eq!(api_key_env_var("unknown"), None);
    }

    #[tokio::test]
    async fn test_generate_image_invalid_aspect_ratio() {
        let server = make_server();
        let resp = server
            .generate_image(
                json!(1),
                json!({"prompt": "test", "provider": "flux", "aspect_ratio": "5:7"}),
            )
            .await;

        assert!(resp.error.is_some());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32602);
        assert!(err.message.contains("Invalid aspect_ratio"));
        assert!(err.message.contains("5:7"));
    }
}
