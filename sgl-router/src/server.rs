use crate::router::PolicyConfig;
use crate::router::Router;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use bytes::Bytes;
use env_logger::Builder;
use log::{info, LevelFilter};
use std::collections::HashMap;
use std::io::Write;

use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: Option<f32>,       // Optional, default is 1.0
    top_p: Option<f32>,            // Optional, default is 1.0
    n: Option<u32>,               // Optional, default is 1
    stream: Option<bool>,         // Optional, default is false
    stop: Option<StopSequence>,   // Optional, can be a string or array of strings
    max_tokens: Option<u32>,      // Optional, default is infinity
    presence_penalty: Option<f32>, // Optional, default is 0
    frequency_penalty: Option<f32>, // Optional, default is 0
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug)]
pub struct AppState {
    router: Router,
    client: reqwest::Client,
}

impl AppState {
    pub fn new(
        worker_urls: Vec<String>,
        client: reqwest::Client,
        policy_config: PolicyConfig,
    ) -> Result<Self, String> {
        // Create router based on policy
        let router = Router::new(worker_urls, policy_config)?;
        Ok(Self { router, client })
    }
}

#[get("/health")]
async fn health(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/health", &req)
        .await
}

#[get("/health_generate")]
async fn health_generate(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/health_generate", &req)
        .await
}

#[get("/get_server_info")]
async fn get_server_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/get_server_info", &req)
        .await
}

#[get("/v1/models")]
async fn v1_models(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/v1/models", &req)
        .await
}

#[get("/get_model_info")]
async fn get_model_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/get_model_info", &req)
        .await
}

#[post("/generate")]
async fn generate(req: HttpRequest, body: Bytes, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/generate")
        .await
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {

    // Parse the request body into a ChatCompletionRequest struct
    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Invalid request body: {}", e),
                    "type": "invalid_request_error",
                    "param": null,
                    "code": null
                }
            }));
        }
    };

    // Validate the request
    if request.model.is_empty() {
        return HttpResponse::BadRequest().json(json!({
            "error": {
                "message": "Invalid model specified.",
                "type": "invalid_request_error",
                "param": "model",
                "code": null
            }
        }));
    }

    if request.messages.is_empty() {
        return HttpResponse::BadRequest().json(json!({
            "error": {
                "message": "Messages field cannot be empty.",
                "type": "invalid_request_error",
                "param": "messages",
                "code": null
            }
        }));
    }

    for (i, message) in request.messages.iter().enumerate() {
        if !["system", "user", "assistant"].contains(&message.role.as_str()) {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Invalid role in message {}: {}", i, message.role),
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": null
                }
            }));
        }

        if message.content.is_empty() {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Message {} has an empty content.", i),
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": null
                }
            }));
        }
    }

    // Validate optional fields
    if let Some(temperature) = request.temperature {
        if !(0.0..=2.0).contains(&temperature) {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Temperature must be between 0 and 2, got {}", temperature),
                    "type": "invalid_request_error",
                    "param": "temperature",
                    "code": null
                }
            }));
        }
    }

    if let Some(top_p) = request.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Top-p must be between 0 and 1, got {}", top_p),
                    "type": "invalid_request_error",
                    "param": "top_p",
                    "code": null
                }
            }));
        }
    }

    if let Some(n) = request.n {
        if n == 0 {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": "N must be greater than 0.",
                    "type": "invalid_request_error",
                    "param": "n",
                    "code": null
                }
            }));
        }
    }

    if let Some(stop) = &request.stop {
        match stop {
            StopSequence::Single(s) => {
                if s.is_empty() {
                    return HttpResponse::BadRequest().json(json!({
                        "error": {
                            "message": "Stop sequence cannot be an empty string.",
                            "type": "invalid_request_error",
                            "param": "stop",
                            "code": null
                        }
                    }));
                }
            }
            StopSequence::Multiple(seqs) => {
                if seqs.iter().any(|s| s.is_empty()) {
                    return HttpResponse::BadRequest().json(json!({
                        "error": {
                            "message": "Stop sequences cannot contain empty strings.",
                            "type": "invalid_request_error",
                            "param": "stop",
                            "code": null
                        }
                    }));
                }
            }
        }
    }

    if let Some(max_tokens) = request.max_tokens {
        if max_tokens == 0 {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": "Max tokens must be greater than 0.",
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": null
                }
            }));
        }
    }

    if let Some(presence_penalty) = request.presence_penalty {
        if !(f32::NEG_INFINITY..=2.0).contains(&presence_penalty) || presence_penalty < -2.0 {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Presence penalty must be between -2 and 2, got {}", presence_penalty),
                    "type": "invalid_request_error",
                    "param": "presence_penalty",
                    "code": null
                }
            }));
        }
    }

    if let Some(frequency_penalty) = request.frequency_penalty {
        if !(f32::NEG_INFINITY..=2.0).contains(&frequency_penalty) || frequency_penalty < -2.0 {
            return HttpResponse::BadRequest().json(json!({
                "error": {
                    "message": format!("Frequency penalty must be between -2 and 2, got {}", frequency_penalty),
                    "type": "invalid_request_error",
                    "param": "frequency_penalty",
                    "code": null
                }
            }));
        }
    }

    data.router
        .route_generate_request(&data.client, &req, &body, "/v1/chat/completions")
        .await
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/v1/completions")
        .await
}

#[post("/add_worker")]
async fn add_worker(
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url = match query.get("url") {
        Some(url) => url.to_string(),
        None => {
            return HttpResponse::BadRequest()
                .body("Worker URL required. Provide 'url' query parameter")
        }
    };

    match data.router.add_worker(&worker_url).await {
        Ok(message) => HttpResponse::Ok().body(message),
        Err(error) => HttpResponse::BadRequest().body(error),
    }
}

#[post("/remove_worker")]
async fn remove_worker(
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url = match query.get("url") {
        Some(url) => url.to_string(),
        None => return HttpResponse::BadRequest().finish(),
    };
    data.router.remove_worker(&worker_url);
    HttpResponse::Ok().body(format!("Successfully removed worker: {}", worker_url))
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub worker_urls: Vec<String>,
    pub policy_config: PolicyConfig,
    pub verbose: bool,
    pub max_payload_size: usize,
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
    // Initialize logger
    Builder::new()
        .format(|buf, record| {
            use chrono::Local;
            writeln!(
                buf,
                "[Router (Rust)] {} - {} - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(
            None,
            if config.verbose {
                LevelFilter::Debug
            } else {
                LevelFilter::Info
            },
        )
        .init();

    info!("🚧 Initializing router on {}:{}", config.host, config.port);
    info!("🚧 Initializing workers on {:?}", config.worker_urls);
    info!("🚧 Policy Config: {:?}", config.policy_config);
    info!(
        "🚧 Max payload size: {} MB",
        config.max_payload_size / (1024 * 1024)
    );

    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    let app_state = web::Data::new(
        AppState::new(
            config.worker_urls.clone(),
            client,
            config.policy_config.clone(),
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
    );

    info!("✅ Serving router on {}:{}", config.host, config.port);
    info!("✅ Serving workers on {:?}", config.worker_urls);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(web::JsonConfig::default().limit(config.max_payload_size))
            .app_data(web::PayloadConfig::default().limit(config.max_payload_size))
            .service(generate)
            .service(v1_chat_completions)
            .service(v1_completions)
            .service(v1_models)
            .service(get_model_info)
            .service(health)
            .service(health_generate)
            .service(get_server_info)
            .service(add_worker)
            .service(remove_worker)
    })
    .bind((config.host, config.port))?
    .run()
    .await
}
