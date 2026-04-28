use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};
use http::Request;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::{MakeSpan, TraceLayer},
};
use tracing::Span;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod handlers;
mod store;
mod stream;
mod translate;
mod util;
mod web_fetch;

use config::Config;
use handlers::{
    AppState, chat_completions_handler, cop_get_handler, cop_post_handler, health_handler,
    models_handler, responses_handler,
};
use store::ResponseStore;
use util::uid;

#[derive(Clone)]
struct SpanMaker;

impl<B> MakeSpan<B> for SpanMaker {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        let request_id = format!("req_{}", uid());
        tracing::info_span!(
            "HTTP",
            request_id = %request_id,
            method = %request.method(),
            path = %request.uri().path(),
            query = %request.uri().query().unwrap_or("-"),
        )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "minimax_proxy=info,tower_http=warn".into());

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().compact())
        .init();

    let config = Arc::new(Config::from_env());

    tracing::info!(
        "[proxy] starting on port {} | providers: {:?}",
        config.port,
        config.enabled_providers
    );

    let state = AppState {
        config: config.clone(),
        store: Arc::new(ResponseStore::new()),
        client: reqwest::Client::builder().timeout(std::time::Duration::from_mins(2)).build()?,
    };

    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/cop", get(cop_get_handler))
        .route("/cop", post(cop_post_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http().make_span_with(SpanMaker))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("[proxy] listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
