use std::sync::Arc;

use axum::{
    Json,
    body::Body,
    extract::{Query, State},
    http::{Response, StatusCode, header},
    response::IntoResponse,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    config::Config,
    store::{MAX_CONSECUTIVE_TOOL_CALLS, ResponseStore},
    stream::{
        RequestContext, handle_streaming_response, pipe_responses_stream_and_capture,
        send_response_as_stream_bytes,
    },
    translate::{
        chat_completion_to_response, extract_web_fetch_calls, get_fetch_url,
        normalize_input_to_array, normalize_minimax_chat_messages,
        responses_request_to_chat_completions, strip_web_fetch_calls, web_fetch_urls_key,
    },
    util::uid,
    web_fetch::{
        MAX_FETCH_LOOPS, conversation_has_urls, ensure_web_fetch_hint, ensure_web_fetch_tool,
        execute_web_fetch,
    },
};

// ─── Shared application state ───────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub store: Arc<ResponseStore>,
    pub client: reqwest::Client,
}

// ─── Helper: JSON response ───────────────────────────────────────────────────

#[allow(clippy::needless_pass_by_value)]
fn json_response(status: u16, body: Value) -> axum::response::Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

/// Build an SSE streaming response from a channel of bytes
fn sse_stream_response(rx: mpsc::UnboundedReceiver<Bytes>) -> axum::response::Response {
    let stream = UnboundedReceiverStream::new(rx);
    #[allow(clippy::unwrap_used)]
    Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(tokio_stream::StreamExt::map(stream, |b| {
            Ok::<_, std::io::Error>(b)
        })))
        .unwrap()
}

/// Forward an upstream error to the client
async fn forward_upstream_error(upstream_res: reqwest::Response) -> axum::response::Response {
    let status = upstream_res.status().as_u16();
    let ct = upstream_res
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let body = upstream_res.text().await.unwrap_or_default();
    tracing::error!("[proxy] upstream error: {} {}", status, body);
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        [(header::CONTENT_TYPE, ct)],
        body,
    )
        .into_response()
}

// ─── GET /health ─────────────────────────────────────────────────────────────

pub async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let providers: Vec<&str> =
        state.config.enabled_providers.iter().map(std::string::String::as_str).collect();
    let default_provider =
        state.config.get_fallback_provider().unwrap_or_else(|_| "none".to_string());
    Json(json!({
        "status": "ok",
        "proxy": "codex-minimax-proxy",
        "providers": providers,
        "default_provider": default_provider
    }))
}

// ─── GET /v1/models ──────────────────────────────────────────────────────────

pub async fn models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let default_provider =
        state.config.get_fallback_provider().unwrap_or_else(|_| "none".to_string());
    Json(json!({
        "object": "list",
        "data": state.config.model_catalog,
        "default_provider": default_provider
    }))
}

// ─── GET+POST /cop ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CopQuery {
    url: Option<String>,
}

pub async fn cop_get_handler(
    State(state): State<AppState>,
    Query(q): Query<CopQuery>,
) -> axum::response::Response {
    let url = match q.url {
        Some(u) if !u.is_empty() => u,
        _ => {
            return json_response(400, json!({ "error": "url parameter required" }));
        },
    };
    tracing::info!("[proxy] /cop GET {}", url);
    let args = json!({ "url": url, "method": "GET" });
    let content = execute_web_fetch(&state.client, &state.config.github_token, &args).await;
    (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], content).into_response()
}

pub async fn cop_post_handler(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let url = match body.get("url").and_then(|u| u.as_str()) {
        Some(u) => u.to_string(),
        None => {
            return json_response(400, json!({ "error": "url parameter required" }));
        },
    };
    let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("GET").to_string();
    tracing::info!("[proxy] /cop {} {}", method, url);
    let args = json!({
        "url": url,
        "method": method,
        "headers": body.get("headers").cloned().unwrap_or_else(|| json!({})),
        "body": body.get("body").cloned()
    });
    let content = execute_web_fetch(&state.client, &state.config.github_token, &args).await;
    (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], content).into_response()
}

// ─── POST /v1/responses ───────────────────────────────────────────────────────

pub async fn responses_handler(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> axum::response::Response {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
    let provider = state.config.resolve_provider_for_model(&model);
    let original_input = normalize_input_to_array(body.get("input").unwrap_or(&Value::Null));

    match provider.as_str() {
        "openai" => {
            if state.config.openai_key.is_empty() {
                return json_response(
                    400,
                    json!({ "error": { "message": "OPENAI_API_KEY is not configured" } }),
                );
            }
            let original_prev_id = body
                .get("previous_response_id")
                .and_then(|p| p.as_str())
                .map(std::string::ToString::to_string);
            maybe_resolve_previous_response_chain(&mut body, "openai", &state.store);
            tracing::info!(
                "[proxy] responses openai({}) | stream={}",
                model,
                body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false)
            );
            forward_openai_responses(body, state, original_input, original_prev_id).await
        },
        _ => handle_minimax_responses(body, state, original_input).await,
    }
}

// ─── POST /v1/chat/completions ────────────────────────────────────────────────

pub async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
    let provider = state.config.resolve_provider_for_model(&model);

    match provider.as_str() {
        "openai" => {
            if state.config.openai_key.is_empty() {
                return json_response(
                    400,
                    json!({ "error": { "message": "OPENAI_API_KEY is not configured" } }),
                );
            }
            tracing::info!(
                "[proxy] chat/completions openai({}) | stream={}",
                model,
                body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false)
            );
            forward_openai_chat_completions(body, state).await
        },
        _ => handle_minimax_chat_completions(body, state).await,
    }
}

// ─── OpenAI forwarding ────────────────────────────────────────────────────────

async fn forward_openai_responses(
    body: Value,
    state: AppState,
    original_input: Vec<Value>,
    original_prev_id: Option<String>,
) -> axum::response::Response {
    let is_stream = body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let url = format!("{}/responses", state.config.openai_base);

    let result = state
        .client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", state.config.openai_key))
        .json(&body)
        .send()
        .await;

    match result {
        Err(e) => json_response(502, json!({ "error": { "message": e.to_string() } })),
        Ok(upstream_res) => {
            if !upstream_res.status().is_success() {
                return forward_upstream_error(upstream_res).await;
            }

            if is_stream {
                let request_id = format!("req_{}", uid());
                let ctx = RequestContext {
                    request_id: request_id.clone(),
                    model: body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string(),
                    provider: "openai".to_string(),
                    stream: true,
                    message_count: original_input.len(),
                    upstream_url: url.clone(),
                };
                tracing::info!("[proxy] pipe stream started | {}", ctx);
                let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
                let store = state.store.clone();
                let orig_input = original_input.clone();
                let orig_prev = original_prev_id.clone();
                let ctx2 = ctx;

                tokio::spawn(async move {
                    pipe_responses_stream_and_capture(upstream_res, &tx, |completed_response| {
                        if let (Some(id), Some(output)) = (
                            completed_response.get("id").and_then(|i| i.as_str()),
                            completed_response.get("output").and_then(|o| o.as_array()),
                        ) {
                            tracing::info!(
                                "[proxy] response stored | req_id={} | provider=openai | response_id={} | output_count={}",
                                request_id,
                                id,
                                output.len()
                            );
                            store.store(
                                id,
                                "openai",
                                orig_input.clone(),
                                output.clone(),
                                orig_prev.clone(),
                            );
                        }
                    }, ctx2)
                    .await;
                });

                return sse_stream_response(rx);
            }

            // Non-streaming
            match upstream_res.json::<Value>().await {
                Err(e) => json_response(502, json!({ "error": { "message": e.to_string() } })),
                Ok(response) => {
                    if let (Some(id), Some(output)) = (
                        response.get("id").and_then(|i| i.as_str()),
                        response.get("output").and_then(|o| o.as_array()),
                    ) {
                        state.store.store(
                            id,
                            "openai",
                            original_input,
                            output.clone(),
                            original_prev_id,
                        );
                    }
                    json_response(200, response)
                },
            }
        },
    }
}

async fn forward_openai_chat_completions(body: Value, state: AppState) -> axum::response::Response {
    let is_stream = body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let url = format!("{}/chat/completions", state.config.openai_base);

    let result = state
        .client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", state.config.openai_key))
        .json(&body)
        .send()
        .await;

    match result {
        Err(e) => json_response(502, json!({ "error": { "message": e.to_string() } })),
        Ok(upstream_res) => {
            if !upstream_res.status().is_success() {
                return forward_upstream_error(upstream_res).await;
            }

            if is_stream {
                let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
                tokio::spawn(async move {
                    let mut stream = upstream_res.bytes_stream();
                    while let Some(chunk) = stream.next().await {
                        if let Ok(bytes) = chunk {
                            let _ = tx.send(bytes);
                        }
                    }
                });
                return sse_stream_response(rx);
            }

            match upstream_res.json::<Value>().await {
                Err(e) => json_response(502, json!({ "error": { "message": e.to_string() } })),
                Ok(response) => json_response(200, response),
            }
        },
    }
}

// ─── MiniMax Responses handler ────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
async fn handle_minimax_responses(
    mut body: Value,
    state: AppState,
    original_input: Vec<Value>,
) -> axum::response::Response {
    if state.config.minimax_key.is_empty() {
        return json_response(
            400,
            json!({ "error": { "message": "MINIMAX_API_KEY is not configured" } }),
        );
    }

    let original_prev_id = body
        .get("previous_response_id")
        .and_then(|p| p.as_str())
        .map(std::string::ToString::to_string);

    maybe_resolve_previous_response_chain(&mut body, "minimax", &state.store);

    // Circuit breaker: check consecutive tool calls
    if let Some(prev_id) = &original_prev_id
        && let Some(prev) = state.store.get(prev_id)
    {
        let consecutive_tc = prev.consecutive_tool_calls;
        if consecutive_tc >= MAX_CONSECUTIVE_TOOL_CALLS {
            tracing::warn!(
                "[proxy] CIRCUIT BREAKER: {} consecutive tool-call-only responses — injecting stop-loop nudge",
                consecutive_tc
            );
            let nudge = json!({
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": format!(
                        "[SYSTEM: You have made {} consecutive tool calls without responding to the user. You MUST now stop making tool calls and provide a text response summarizing your progress, findings, and any remaining work. Do NOT make any more tool calls in this response.]",
                        consecutive_tc
                    )
                }]
            });
            let current = normalize_input_to_array(body.get("input").unwrap_or(&Value::Null));
            let mut new_input = current;
            new_input.push(nudge);
            body["input"] = Value::Array(new_input);
        } else if consecutive_tc >= MAX_CONSECUTIVE_TOOL_CALLS * 3 / 4 {
            tracing::warn!(
                "[proxy] tool-call loop warning: {}/{} consecutive tool-call responses",
                consecutive_tc,
                MAX_CONSECUTIVE_TOOL_CALLS
            );
        }
    }

    let mut chat_req = responses_request_to_chat_completions(&body);
    chat_req["model"] = json!(
        state.config.minimax_models.first().map_or("MiniMax-M2.7", std::string::String::as_str)
    );
    let is_stream = chat_req.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);

    chat_req["reasoning_split"] = json!(true);
    let upstream_url = format!("{}/chat/completions", state.config.minimax_base);
    let upstream_key = state.config.minimax_key.clone();
    let route_label =
        format!("minimax({})", chat_req.get("model").and_then(|m| m.as_str()).unwrap_or(""));

    // Hard circuit breaker: strip all tools after many extra consecutive TCs
    if let Some(prev_id) = &original_prev_id
        && let Some(prev) = state.store.get(prev_id)
        && prev.consecutive_tool_calls >= MAX_CONSECUTIVE_TOOL_CALLS + 3
    {
        tracing::warn!("[proxy] HARD CIRCUIT BREAKER: stripping all tools to force text response");
        if let Some(obj) = chat_req.as_object_mut() {
            obj.remove("tools");
            obj.remove("tool_choice");
        }
    }

    // web_fetch injection for conversations with URLs
    let messages = chat_req.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    let has_conversation_urls = conversation_has_urls(&messages);

    if has_conversation_urls {
        let tools = chat_req.get("tools").cloned();
        chat_req["tools"] = ensure_web_fetch_tool(tools.as_ref());
        let msgs = chat_req.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
        chat_req["messages"] = Value::Array(ensure_web_fetch_hint(msgs));
    }

    let messages = chat_req.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();

    tracing::info!(
        "[proxy] {} | stream={} | messages={}{}",
        route_label,
        is_stream,
        messages.len(),
        if has_conversation_urls { " | web_fetch_injected" } else { "" }
    );

    let model_name = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
    let metadata = body.get("metadata").cloned().unwrap_or_else(|| json!({}));

    // web_fetch loop mode (non-streaming upstream loop, then convert)
    if has_conversation_urls {
        let result = run_web_fetch_loop(
            &state.client,
            &state.config.github_token,
            &upstream_url,
            &upstream_key,
            &chat_req,
            messages,
            "[proxy]",
        )
        .await;

        let final_cc = match result {
            Ok(cc) => cc,
            Err(resp) => return resp,
        };

        let responses_response = chat_completion_to_response(
            &final_cc,
            &model_name,
            original_prev_id.as_deref(),
            Some(&metadata),
        );
        state.store.store(
            responses_response.get("id").and_then(|i| i.as_str()).unwrap_or(""),
            "minimax",
            original_input,
            responses_response
                .get("output")
                .and_then(|o| o.as_array())
                .cloned()
                .unwrap_or_default(),
            original_prev_id,
        );

        if is_stream {
            let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
            for bytes in send_response_as_stream_bytes(&responses_response) {
                let _ = tx.send(bytes);
            }
            return sse_stream_response(rx);
        }
        return json_response(200, responses_response);
    }

    // Standard path (streaming or non-streaming)
    let upstream_res = match state
        .client
        .post(&upstream_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {upstream_key}"))
        .json(&chat_req)
        .send()
        .await
    {
        Err(e) => return json_response(502, json!({ "error": { "message": e.to_string() } })),
        Ok(r) => r,
    };

    if !upstream_res.status().is_success() {
        return forward_upstream_error(upstream_res).await;
    }

    if is_stream {
        let request_id = format!("req_{}", uid());
        let ctx = RequestContext {
            request_id: request_id.clone(),
            model: model_name.clone(),
            provider: "minimax".to_string(),
            stream: true,
            message_count: messages.len(),
            upstream_url: upstream_url.clone(),
        };
        tracing::info!("[proxy] stream started | {} | response_id=pending", ctx);
        let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
        let store = state.store.clone();
        let model_name2 = model_name.clone();
        let orig_input2 = original_input.clone();
        let orig_prev2 = original_prev_id.clone();
        let ctx2 = ctx;

        tokio::spawn(async move {
            let result = handle_streaming_response(
                upstream_res,
                model_name2,
                orig_prev2.clone(),
                metadata,
                tx.clone(),
                ctx2,
            )
            .await;

            tracing::info!(
                "[proxy] stream completed | req_id={} | response_id={} | output_items={}",
                request_id,
                result.response_id,
                result.output.len()
            );
            store.store(&result.response_id, "minimax", orig_input2, result.output, orig_prev2);
        });

        return sse_stream_response(rx);
    }

    // Non-streaming
    match upstream_res.json::<Value>().await {
        Err(e) => json_response(502, json!({ "error": { "message": e.to_string() } })),
        Ok(cc) => {
            let responses_response = chat_completion_to_response(
                &cc,
                &model_name,
                original_prev_id.as_deref(),
                Some(&metadata),
            );
            state.store.store(
                responses_response.get("id").and_then(|i| i.as_str()).unwrap_or(""),
                "minimax",
                original_input,
                responses_response
                    .get("output")
                    .and_then(|o| o.as_array())
                    .cloned()
                    .unwrap_or_default(),
                original_prev_id,
            );
            json_response(200, responses_response)
        },
    }
}

// ─── MiniMax Chat Completions handler ─────────────────────────────────────────

#[allow(clippy::too_many_lines)]
async fn handle_minimax_chat_completions(
    mut body: Value,
    state: AppState,
) -> axum::response::Response {
    if state.config.minimax_key.is_empty() {
        return json_response(
            400,
            json!({ "error": { "message": "MINIMAX_API_KEY is not configured" } }),
        );
    }

    let model = body.get("model").and_then(|m| m.as_str()).map_or_else(
        || {
            state
                .config
                .minimax_models
                .first()
                .cloned()
                .unwrap_or_else(|| "MiniMax-M2.7".to_string())
        },
        std::string::ToString::to_string,
    );
    body["model"] = json!(model);

    let is_stream = body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);

    let messages = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    let normalized = normalize_minimax_chat_messages(messages);
    body["messages"] = Value::Array(normalized.clone());
    body["reasoning_split"] = json!(true);
    if body.get("max_tokens").is_none() {
        body["max_tokens"] = json!(16384);
    }

    let has_urls = conversation_has_urls(&normalized);

    if has_urls {
        let tools = body.get("tools").cloned();
        body["tools"] = ensure_web_fetch_tool(tools.as_ref());
        let msgs = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
        body["messages"] = Value::Array(ensure_web_fetch_hint(msgs));
    }

    let messages_final =
        body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();

    tracing::info!(
        "[proxy] chat/completions minimax({}) | stream={} | messages={}{}",
        model,
        is_stream,
        messages_final.len(),
        if has_urls { " | web_fetch_injected" } else { "" }
    );

    let upstream_url = format!("{}/chat/completions", state.config.minimax_base);

    if has_urls {
        let result = run_web_fetch_loop(
            &state.client,
            &state.config.github_token,
            &upstream_url,
            &state.config.minimax_key,
            &body,
            messages_final,
            "[proxy] cc:",
        )
        .await;

        let final_cc = match result {
            Ok(cc) => cc,
            Err(resp) => return resp,
        };

        if is_stream {
            // Fake streaming from the completed response
            let msg = final_cc
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("message"))
                .cloned()
                .unwrap_or_else(|| json!({}));

            let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                for (i, tc) in tool_calls.iter().enumerate() {
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    let arguments = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .unwrap_or("");
                    let id = tc.get("id").and_then(|id| id.as_str()).unwrap_or("");
                    let _ = tx.send(Bytes::from(format!(
                        "data: {}\n\n",
                        json!({ "choices": [{ "index": 0, "delta": { "tool_calls": [{ "index": i, "id": id, "type": "function", "function": { "name": name, "arguments": "" } }] } }] })
                    )));
                    let _ = tx.send(Bytes::from(format!(
                        "data: {}\n\n",
                        json!({ "choices": [{ "index": 0, "delta": { "tool_calls": [{ "index": i, "function": { "arguments": arguments } }] } }] })
                    )));
                }
            }
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                let _ = tx.send(Bytes::from(format!(
                    "data: {}\n\n",
                    json!({ "choices": [{ "index": 0, "delta": { "content": content } }] })
                )));
            }
            let finish_reason = final_cc
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("finish_reason"))
                .cloned()
                .unwrap_or_else(|| json!("stop"));
            let usage = final_cc.get("usage").cloned().unwrap_or(json!(null));
            let _ = tx.send(Bytes::from(format!(
                "data: {}\n\n",
                json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }], "usage": usage })
            )));
            let _ = tx.send(Bytes::from("data: [DONE]\n\n"));

            return sse_stream_response(rx);
        }

        return json_response(200, final_cc);
    }

    // Standard (no URL fetch loop)
    let upstream_res = match state
        .client
        .post(&upstream_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", state.config.minimax_key))
        .json(&body)
        .send()
        .await
    {
        Err(e) => return json_response(502, json!({ "error": { "message": e.to_string() } })),
        Ok(r) => r,
    };

    if !upstream_res.status().is_success() {
        return forward_upstream_error(upstream_res).await;
    }

    if is_stream {
        let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = upstream_res.bytes_stream();
            while let Some(chunk) = stream.next().await {
                if let Ok(bytes) = chunk {
                    let _ = tx.send(bytes);
                }
            }
        });
        return sse_stream_response(rx);
    }

    match upstream_res.json::<Value>().await {
        Err(e) => json_response(502, json!({ "error": { "message": e.to_string() } })),
        Ok(data) => json_response(200, data),
    }
}

// ─── Web fetch loop ────────────────────────────────────────────────────────────

/// Run the iterative `web_fetch` loop: call upstream non-streaming, execute any
/// `web_fetch` tool calls, repeat up to `MAX_FETCH_LOOPS` times.
/// Returns the final `ChatCompletion` response, or an error response.
#[allow(clippy::too_many_lines, clippy::unwrap_used)]
async fn run_web_fetch_loop(
    client: &reqwest::Client,
    github_token: &str,
    upstream_url: &str,
    upstream_key: &str,
    base_req: &Value,
    initial_messages: Vec<Value>,
    log_prefix: &str,
) -> Result<Value, axum::response::Response> {
    let mut loop_messages = initial_messages;
    let mut final_cc: Option<Value> = None;
    let mut fetch_loop_count = 0;
    let mut fetch_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut prev_fetch_urls = String::new();

    for loop_idx in 0..=MAX_FETCH_LOOPS {
        let mut loop_req = base_req.clone();
        loop_req["messages"] = Value::Array(loop_messages.clone());
        loop_req["stream"] = json!(false);

        let upstream_res = client
            .post(upstream_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {upstream_key}"))
            .json(&loop_req)
            .send()
            .await
            .map_err(|e| json_response(502, json!({ "error": { "message": e.to_string() } })))?;

        if !upstream_res.status().is_success() {
            return Err(forward_upstream_error(upstream_res).await);
        }

        let mut cc = upstream_res
            .json::<Value>()
            .await
            .map_err(|e| json_response(502, json!({ "error": { "message": e.to_string() } })))?;

        let web_fetch_calls = extract_web_fetch_calls(&cc);
        let current_fetch_urls = web_fetch_urls_key(&web_fetch_calls);
        let is_stuck = !web_fetch_calls.is_empty() && current_fetch_urls == prev_fetch_urls;

        if web_fetch_calls.is_empty() || loop_idx == MAX_FETCH_LOOPS || is_stuck {
            if is_stuck {
                tracing::warn!(
                    "{} web_fetch loop stuck — model re-requested same URL(s), breaking early at loop {}",
                    log_prefix,
                    loop_idx + 1
                );
            }
            if loop_idx == MAX_FETCH_LOOPS && !web_fetch_calls.is_empty() {
                tracing::warn!(
                    "{} web_fetch MAX_FETCH_LOOPS ({}) exhausted — stripping remaining fetches",
                    log_prefix,
                    MAX_FETCH_LOOPS
                );
            }

            if !web_fetch_calls.is_empty() {
                strip_web_fetch_calls(&mut cc);
            }

            final_cc = Some(cc);
            fetch_loop_count = loop_idx;
            break;
        }

        prev_fetch_urls = current_fetch_urls;
        tracing::info!(
            "{} executing {} web_fetch call(s) (loop {}/{})",
            log_prefix,
            web_fetch_calls.len(),
            loop_idx + 1,
            MAX_FETCH_LOOPS
        );

        let mut results: Vec<Value> = Vec::new();
        for tc in &web_fetch_calls {
            let args_str = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let fetch_url = get_fetch_url(args_str);
            let tc_id = tc.get("id").and_then(|id| id.as_str()).unwrap_or("");

            let content = if let Some(cached) = fetch_cache.get(&fetch_url) {
                tracing::info!(
                    "{} web_fetch {} -> {} chars (cached)",
                    log_prefix,
                    fetch_url,
                    cached.len()
                );
                cached.clone()
            } else {
                let args: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
                let fetched = execute_web_fetch(client, github_token, &args).await;
                tracing::info!("{} web_fetch {} -> {} chars", log_prefix, fetch_url, fetched.len());
                fetch_cache.insert(fetch_url.clone(), fetched.clone());
                fetched
            };

            results.push(json!({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": content
            }));
        }

        loop_messages.push(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": web_fetch_calls
        }));
        loop_messages.extend(results);
    }

    if fetch_loop_count > 0 {
        tracing::info!("{} web_fetch resolved after {} loop(s)", log_prefix, fetch_loop_count);
    }

    Ok(final_cc.unwrap())
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn maybe_resolve_previous_response_chain(
    body: &mut Value,
    target_provider: &str,
    store: &ResponseStore,
) {
    let prev_id = match body.get("previous_response_id").and_then(|p| p.as_str()) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => return,
    };

    let Some(previous) = store.get(&prev_id) else {
        if target_provider == "minimax" {
            tracing::warn!(
                "[proxy] previous_response_id {} missing; MiniMax request will continue without restored history",
                prev_id
            );
        }
        return;
    };

    let needs_local_resolution =
        target_provider == "minimax" || previous.provider != target_provider;
    if !needs_local_resolution {
        return;
    }

    let chain_items = store.resolve_chain(&prev_id);
    if chain_items.is_empty() {
        return;
    }

    let current = normalize_input_to_array(body.get("input").unwrap_or(&Value::Null));
    let mut new_input = chain_items.clone();
    new_input.extend(current);
    body["input"] = Value::Array(new_input);
    if let Some(obj) = body.as_object_mut() {
        obj.remove("previous_response_id");
    }

    tracing::info!(
        "[proxy] locally resolved previous_response_id across provider boundary -> {} ({} items prepended)",
        target_provider,
        chain_items.len()
    );
}
