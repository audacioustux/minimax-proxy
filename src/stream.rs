use std::collections::HashMap;

use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    translate::{finish_reason_to_status, translate_usage},
    util::{now_unix, uid},
};

// Request context for structured logging

#[derive(Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub model: String,
    pub provider: String,
    pub stream: bool,
    pub message_count: usize,
    pub upstream_url: String,
}

impl std::fmt::Display for RequestContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "req_id={} provider={} model={} stream={} msgs={} url={}",
            self.request_id,
            self.provider,
            self.model,
            self.stream,
            self.message_count,
            self.upstream_url
        )
    }
}

// ─── SSE event formatters ───────────────────────────────────────────────────

#[allow(clippy::needless_pass_by_value)]
fn sse(event_type: &str, data: Value) -> Bytes {
    Bytes::from(format!("event: {event_type}\ndata: {data}\n\n"))
}

pub fn event_created(response_id: &str, model: &str, prev_id: Option<&str>, meta: &Value) -> Bytes {
    let base = base_response(response_id, model, prev_id, meta, "in_progress");
    sse("response.created", json!({ "type": "response.created", "response": base }))
}

pub fn event_in_progress(
    response_id: &str,
    model: &str,
    prev_id: Option<&str>,
    meta: &Value,
) -> Bytes {
    let base = base_response(response_id, model, prev_id, meta, "in_progress");
    sse("response.in_progress", json!({ "type": "response.in_progress", "response": base }))
}

fn base_response(
    id: &str,
    model: &str,
    prev_id: Option<&str>,
    meta: &Value,
    status: &str,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now_unix(),
        "status": status,
        "model": model,
        "output": [],
        "previous_response_id": prev_id,
        "metadata": meta,
        "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
    })
}

#[allow(clippy::needless_pass_by_value)]
pub fn event_output_item_added(out_idx: usize, item: Value) -> Bytes {
    sse(
        "response.output_item.added",
        json!({
            "type": "response.output_item.added",
            "output_index": out_idx,
            "item": item
        }),
    )
}

#[allow(clippy::needless_pass_by_value)]
pub fn event_content_part_added(out_idx: usize, content_idx: usize, part: Value) -> Bytes {
    sse(
        "response.content_part.added",
        json!({
            "type": "response.content_part.added",
            "output_index": out_idx,
            "content_index": content_idx,
            "part": part
        }),
    )
}

pub fn event_text_delta(out_idx: usize, content_idx: usize, delta: &str) -> Bytes {
    sse(
        "response.output_text.delta",
        json!({
            "type": "response.output_text.delta",
            "output_index": out_idx,
            "content_index": content_idx,
            "delta": delta
        }),
    )
}

pub fn event_text_done(out_idx: usize, content_idx: usize, text: &str) -> Bytes {
    sse(
        "response.output_text.done",
        json!({
            "type": "response.output_text.done",
            "output_index": out_idx,
            "content_index": content_idx,
            "text": text
        }),
    )
}

#[allow(clippy::needless_pass_by_value)]
pub fn event_content_part_done(out_idx: usize, content_idx: usize, part: Value) -> Bytes {
    sse(
        "response.content_part.done",
        json!({
            "type": "response.content_part.done",
            "output_index": out_idx,
            "content_index": content_idx,
            "part": part
        }),
    )
}

#[allow(clippy::needless_pass_by_value)]
pub fn event_output_item_done(out_idx: usize, item: Value) -> Bytes {
    sse(
        "response.output_item.done",
        json!({
            "type": "response.output_item.done",
            "output_index": out_idx,
            "item": item
        }),
    )
}

pub fn event_fn_args_delta(out_idx: usize, call_id: &str, delta: &str) -> Bytes {
    sse(
        "response.function_call_arguments.delta",
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": out_idx,
            "call_id": call_id,
            "delta": delta
        }),
    )
}

pub fn event_fn_args_done(out_idx: usize, call_id: &str, arguments: &str) -> Bytes {
    sse(
        "response.function_call_arguments.done",
        json!({
            "type": "response.function_call_arguments.done",
            "output_index": out_idx,
            "call_id": call_id,
            "arguments": arguments
        }),
    )
}

#[allow(clippy::needless_pass_by_value)]
pub fn event_completed(response: Value) -> Bytes {
    sse("response.completed", json!({ "type": "response.completed", "response": response }))
}

// ─── Tool call tracking during streaming ───────────────────────────────────

#[derive(Debug, Clone)]
pub struct StreamToolCall {
    pub id: String,      // fc_... ID
    pub call_id: String, // original tc.id
    pub name: String,
    pub arguments: String,
    pub output_idx: usize,
}

// ─── Handle streaming response (chat completions SSE → Responses API SSE) ──

pub struct StreamResult {
    pub response_id: String,
    pub output: Vec<Value>,
}

/// Spawns a task that reads the upstream streaming chat-completions response,
/// translates events to Responses API SSE, sends them on `tx`, and returns
/// the completed output items via a oneshot channel.
#[allow(clippy::too_many_lines)]
pub async fn handle_streaming_response(
    upstream_response: reqwest::Response,
    model: String,
    previous_response_id: Option<String>,
    metadata: Value,
    tx: mpsc::UnboundedSender<Bytes>,
    ctx: RequestContext,
) -> StreamResult {
    let response_id = format!("resp_{}", uid());
    let prev_id_str = previous_response_id.as_deref();

    tracing::info!("[proxy] stream started | {} | response_id={}", ctx, response_id);

    let _ = tx.send(event_created(&response_id, &model, prev_id_str, &metadata));
    let _ = tx.send(event_in_progress(&response_id, &model, prev_id_str, &metadata));

    let mut full_text = String::new();
    let mut in_think = false;
    let mut message_started = false;
    let mut completion_sent = false;
    let mut tool_calls: HashMap<usize, StreamToolCall> = HashMap::new();
    let output_index: usize = 0;
    let mut text_output_idx: i64 = -1;
    let mut buffer = String::new();
    let mut final_output: Vec<Value> = Vec::new();

    let mut chunk_count: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut stream = upstream_response.bytes_stream();

    'outer: while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    "[proxy] stream read error | {} | response_id={} | bytes_received={} | error={}",
                    ctx,
                    response_id,
                    total_bytes,
                    e
                );
                break;
            },
        };

        chunk_count += 1;
        total_bytes += chunk.len() as u64;

        if chunk_count.is_multiple_of(50) {
            tracing::debug!(
                "[proxy] stream progress | {} | response_id={} | chunks={} bytes={}",
                ctx,
                response_id,
                chunk_count,
                total_bytes
            );
        }

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        loop {
            match buffer.find('\n') {
                None => break,
                Some(idx) => {
                    let line = buffer[..idx].trim_end_matches('\r').to_string();
                    buffer = buffer[idx + 1..].to_string();

                    if !line.starts_with("data: ") {
                        continue;
                    }

                    let data = line[6..].trim();

                    if data == "[DONE]" {
                        if !completion_sent {
                            completion_sent = true;
                            final_output = send_completion_events(
                                &tx,
                                &response_id,
                                &model,
                                &full_text,
                                &tool_calls,
                                output_index,
                                text_output_idx,
                                None,
                                None,
                                previous_response_id.as_deref(),
                                &metadata,
                            );
                        }
                        break 'outer;
                    }

                    let parsed: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let delta = parsed
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                        .and_then(|c| c.get("delta"));
                    let finish_reason = parsed
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                        .and_then(|c| c.get("finish_reason"))
                        .and_then(|r| r.as_str());

                    if delta.is_none() && finish_reason.is_none() {
                        continue;
                    }

                    // Tool calls delta
                    if let Some(tc_deltas) =
                        delta.and_then(|d| d.get("tool_calls")).and_then(|tc| tc.as_array())
                    {
                        for tc in tc_deltas {
                            let idx = tc
                                .get("index")
                                .and_then(serde_json::Value::as_u64)
                                .and_then(|v| usize::try_from(v).ok())
                                .unwrap_or(0);
                            let tc_out_idx = if message_started && text_output_idx == 0 {
                                output_index + idx + 1
                            } else {
                                output_index + idx
                            };

                            tool_calls.entry(idx).or_insert_with(|| {
                                let call_id = tc.get("id").and_then(|id| id.as_str()).map_or_else(
                                    || format!("call_{}", uid()),
                                    std::string::ToString::to_string,
                                );
                                let fc_id = format!("fc_{}", uid());
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                let _ = tx.send(event_output_item_added(
                                    tc_out_idx,
                                    json!({
                                        "type": "function_call",
                                        "id": fc_id,
                                        "call_id": call_id,
                                        "name": name,
                                        "arguments": "",
                                        "status": "in_progress"
                                    }),
                                ));

                                StreamToolCall {
                                    id: fc_id,
                                    call_id,
                                    name,
                                    arguments: String::new(),
                                    output_idx: tc_out_idx,
                                }
                            });

                            if let Some(args_delta) = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                && let Some(tc_data) = tool_calls.get_mut(&idx)
                            {
                                tc_data.arguments.push_str(args_delta);
                                let _ = tx.send(event_fn_args_delta(
                                    tc_data.output_idx,
                                    &tc_data.call_id.clone(),
                                    args_delta,
                                ));
                            }
                        }

                        if let Some(reason) = finish_reason
                            && !completion_sent
                        {
                            completion_sent = true;
                            let usage = parsed.get("usage").cloned();
                            final_output = send_completion_events(
                                &tx,
                                &response_id,
                                &model,
                                &full_text,
                                &tool_calls,
                                output_index,
                                text_output_idx,
                                Some(reason),
                                usage.as_ref(),
                                previous_response_id.as_deref(),
                                &metadata,
                            );
                        }
                        continue;
                    }

                    // Skip reasoning content
                    if delta.and_then(|d| d.get("reasoning_content")).is_some() {
                        continue;
                    }

                    // Text content delta
                    if let Some(content) =
                        delta.and_then(|d| d.get("content")).and_then(|c| c.as_str())
                    {
                        let mut text = content.to_string();

                        if text.contains("<think>") {
                            in_think = true;
                            text = text.replace("<think>", "");
                        }
                        if text.contains("</think>") {
                            in_think = false;
                            text = text.replace("</think>", "");
                        }

                        if in_think || text.is_empty() {
                            // still check finish_reason below
                        } else {
                            if !message_started {
                                message_started = true;
                                text_output_idx =
                                    i64::try_from(output_index + tool_calls.len()).unwrap_or(0);
                                let tidx = usize::try_from(text_output_idx).unwrap_or(0);
                                let _ = tx.send(event_output_item_added(
                                    tidx,
                                    json!({
                                        "type": "message",
                                        "id": format!("msg_{}", uid()),
                                        "status": "in_progress",
                                        "role": "assistant",
                                        "content": []
                                    }),
                                ));
                                let _ = tx.send(event_content_part_added(
                                    tidx,
                                    0,
                                    json!({ "type": "output_text", "text": "", "annotations": [] }),
                                ));
                            }
                            full_text.push_str(&text);
                            let _ = tx.send(event_text_delta(
                                usize::try_from(text_output_idx).unwrap_or(0),
                                0,
                                &text,
                            ));
                        }
                    }

                    if let Some(reason) = finish_reason
                        && !completion_sent
                    {
                        completion_sent = true;
                        let usage = parsed.get("usage").cloned();
                        final_output = send_completion_events(
                            &tx,
                            &response_id,
                            &model,
                            &full_text,
                            &tool_calls,
                            output_index,
                            text_output_idx,
                            Some(reason),
                            usage.as_ref(),
                            previous_response_id.as_deref(),
                            &metadata,
                        );
                    }
                },
            }
        }
    }

    // Stream ended without explicit DONE/finish_reason
    if !completion_sent {
        let was_generating = !full_text.is_empty() || !tool_calls.is_empty();
        let fallback = if was_generating { "length" } else { "stop" };
        tracing::warn!(
            "[proxy] stream ended without finish_reason (wasGenerating={}, reason={})",
            was_generating,
            fallback
        );
        final_output = send_completion_events(
            &tx,
            &response_id,
            &model,
            &full_text,
            &tool_calls,
            output_index,
            text_output_idx,
            Some(fallback),
            None,
            previous_response_id.as_deref(),
            &metadata,
        );
    }

    StreamResult { response_id, output: final_output }
}

#[allow(clippy::too_many_arguments)]
fn send_completion_events(
    tx: &mpsc::UnboundedSender<Bytes>,
    response_id: &str,
    model: &str,
    full_text: &str,
    tool_calls: &HashMap<usize, StreamToolCall>,
    output_index: usize,
    text_output_idx: i64,
    finish_reason: Option<&str>,
    usage: Option<&Value>,
    previous_response_id: Option<&str>,
    metadata: &Value,
) -> Vec<Value> {
    // Finalize tool calls
    let mut sorted_tcs: Vec<(&usize, &StreamToolCall)> = tool_calls.iter().collect();
    sorted_tcs.sort_by_key(|(idx, _)| *idx);

    for (idx, tc) in &sorted_tcs {
        let tc_idx = tc.output_idx;
        let _ = tx.send(event_fn_args_done(tc_idx, &tc.call_id, &tc.arguments));
        let _ = tx.send(event_output_item_done(
            tc_idx,
            json!({
                "type": "function_call",
                "id": tc.id,
                "call_id": tc.call_id,
                "name": tc.name,
                "arguments": tc.arguments,
                "status": "completed"
            }),
        ));
        let _ = idx; // suppress warning
    }

    let msg_out_idx = if text_output_idx >= 0 {
        usize::try_from(text_output_idx).unwrap_or(output_index + tool_calls.len())
    } else {
        output_index + tool_calls.len()
    };

    let trimmed = full_text.trim();
    if !trimmed.is_empty() {
        let done_part = json!({ "type": "output_text", "text": trimmed, "annotations": [] });
        let _ = tx.send(event_text_done(msg_out_idx, 0, trimmed));
        let _ = tx.send(event_content_part_done(msg_out_idx, 0, done_part.clone()));
        let _ = tx.send(event_output_item_done(
            msg_out_idx,
            json!({
                "type": "message",
                "id": format!("msg_{}", uid()),
                "status": "completed",
                "role": "assistant",
                "content": [done_part]
            }),
        ));
    }

    // Build sorted output items
    let mut output_items: Vec<(usize, Value)> = Vec::new();

    for (idx, tc) in &sorted_tcs {
        output_items.push((
            tc.output_idx,
            json!({
                "type": "function_call",
                "id": tc.id,
                "call_id": tc.call_id,
                "name": tc.name,
                "arguments": tc.arguments,
                "status": "completed"
            }),
        ));
        let _ = idx;
    }

    if !trimmed.is_empty() {
        output_items.push((
            msg_out_idx,
            json!({
                "type": "message",
                "id": format!("msg_{}", uid()),
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": trimmed, "annotations": [] }]
            }),
        ));
    }

    output_items.sort_by_key(|(idx, _)| *idx);
    let final_output: Vec<Value> = output_items.into_iter().map(|(_, v)| v).collect();

    let fr = finish_reason.unwrap_or("stop");
    let (status, incomplete_details) = finish_reason_to_status(fr);

    let final_response = json!({
        "id": response_id,
        "object": "response",
        "created_at": now_unix(),
        "status": status,
        "model": model,
        "output": final_output,
        "previous_response_id": previous_response_id,
        "metadata": metadata,
        "usage": translate_usage(usage),
        "incomplete_details": incomplete_details
    });

    let _ = tx.send(event_completed(final_response));

    final_output
}

// ─── Convert a completed Response to SSE stream ────────────────────────────

/// Stream a pre-built Responses API response object as SSE
pub fn send_response_as_stream_bytes(response: &Value) -> Vec<Bytes> {
    let mut events: Vec<Bytes> = Vec::new();

    let response_id = response.get("id").and_then(|id| id.as_str()).unwrap_or("");
    let model = response.get("model").and_then(|m| m.as_str()).unwrap_or("");
    let prev_id = response.get("previous_response_id").and_then(|p| p.as_str());
    let meta = response.get("metadata").cloned().unwrap_or_else(|| json!({}));

    events.push(event_created(response_id, model, prev_id, &meta));
    events.push(event_in_progress(response_id, model, prev_id, &meta));

    let output = response.get("output").and_then(|o| o.as_array()).cloned().unwrap_or_default();

    for (i, item) in output.iter().enumerate() {
        let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match item_type {
            "function_call" => {
                let call_id = item.get("call_id").and_then(|c| c.as_str()).unwrap_or("");
                let arguments = item.get("arguments").and_then(|a| a.as_str()).unwrap_or("");

                let mut in_progress = item.clone();
                if let Some(obj) = in_progress.as_object_mut() {
                    obj.insert("status".to_string(), json!("in_progress"));
                    obj.insert("arguments".to_string(), json!(""));
                }
                events.push(event_output_item_added(i, in_progress.clone()));
                events.push(event_fn_args_delta(i, call_id, arguments));
                events.push(event_fn_args_done(i, call_id, arguments));
                events.push(event_output_item_done(i, item.clone()));
            },
            "message" => {
                let mut in_progress = item.clone();
                if let Some(obj) = in_progress.as_object_mut() {
                    obj.insert("status".to_string(), json!("in_progress"));
                    obj.insert("content".to_string(), json!([]));
                }
                events.push(event_output_item_added(i, in_progress.clone()));

                let content =
                    item.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default();
                for (ci, part) in content.iter().enumerate() {
                    if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        events.push(event_content_part_added(
                            i,
                            ci,
                            json!({ "type": "output_text", "text": "", "annotations": [] }),
                        ));
                        // Send in 80-char chunks
                        let chars: Vec<char> = text.chars().collect();
                        for chunk in chars.chunks(80) {
                            let s: String = chunk.iter().collect();
                            events.push(event_text_delta(i, ci, &s));
                        }
                        events.push(event_text_done(i, ci, text));
                        events.push(event_content_part_done(i, ci, part.clone()));
                    }
                }
                events.push(event_output_item_done(i, item.clone()));
            },
            _ => {},
        }
    }

    events.push(event_completed(response.clone()));
    events
}

// ─── Pipe OpenAI Responses SSE stream through, capturing completed response ─

pub async fn pipe_responses_stream_and_capture<F>(
    upstream_response: reqwest::Response,
    tx: &mpsc::UnboundedSender<Bytes>,
    mut on_completed: F,
    ctx: RequestContext,
) where
    F: FnMut(Value),
{
    let mut buffer = String::new();
    let mut stream = upstream_response.bytes_stream();
    let mut chunk_count: u64 = 0;
    let mut total_bytes: u64 = 0;

    tracing::info!("[proxy] pipe stream started | {}", ctx);

    while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    "[proxy] pipe stream error | {} | chunks={} bytes={} | error={}",
                    ctx,
                    chunk_count,
                    total_bytes,
                    e
                );
                break;
            },
        };

        chunk_count += 1;
        total_bytes += chunk.len() as u64;

        if chunk_count.is_multiple_of(50) {
            tracing::debug!(
                "[proxy] pipe progress | {} | chunks={} bytes={}",
                ctx,
                chunk_count,
                total_bytes
            );
        }

        let _ = tx.send(Bytes::copy_from_slice(&chunk));
        buffer.push_str(&String::from_utf8_lossy(&chunk).replace("\r\n", "\n"));

        loop {
            match buffer.find("\n\n") {
                None => break,
                Some(idx) => {
                    let block = buffer[..idx].to_string();
                    buffer = buffer[idx + 2..].to_string();
                    handle_sse_block(&block, &mut on_completed);
                },
            }
        }
    }

    if !buffer.trim().is_empty() {
        handle_sse_block(&buffer, &mut on_completed);
    }

    tracing::info!(
        "[proxy] pipe stream ended | {} | chunks={} bytes={}",
        ctx,
        chunk_count,
        total_bytes
    );
}

fn handle_sse_block<F: FnMut(Value)>(block: &str, on_completed: &mut F) {
    let mut event_type = String::new();
    let mut data_lines: Vec<&str> = Vec::new();

    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_type = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }

    let data = data_lines.join("\n");
    if data.is_empty() || data == "[DONE]" {
        return;
    }

    if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
        let is_completed = event_type == "response.completed"
            || parsed.get("type").and_then(|t| t.as_str()) == Some("response.completed");
        if is_completed {
            let response = parsed.get("response").cloned().unwrap_or(parsed);
            on_completed(response);
        }
    }
}
