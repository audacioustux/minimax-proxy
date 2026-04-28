use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::util::{now_unix, uid};

#[allow(clippy::expect_used)]
static THINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>.*?\s*").expect("valid regex"));

const TOOL_OUTPUT_MAX: usize = 2_000;
const KEEP_RECENT_FULL: usize = 10;
const MAX_MESSAGES: usize = 55;

// ─── Input normalization ────────────────────────────────────────────────────

/// Convert a Responses API `input` field into a Vec<Value> of items
pub fn normalize_input_to_array(input: &Value) -> Vec<Value> {
    match input {
        Value::Array(arr) => arr.clone(),
        Value::String(s) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": s }]
        })],
        _ => vec![],
    }
}

// ─── Responses API → Chat Completions ──────────────────────────────────────

/// Full translation of a Responses API request body to a Chat Completions request body
#[allow(clippy::too_many_lines)]
pub fn responses_request_to_chat_completions(body: &Value) -> Value {
    let mut messages: Vec<Option<Value>> = Vec::new();

    // instructions → first user message
    if let Some(instructions) = body.get("instructions").and_then(|i| i.as_str()) {
        messages.push(Some(json!({
            "role": "user",
            "content": format!("[System Instructions] {}\n\nNote: Be efficient with tool calls. Avoid repeating the same tool call unnecessarily.", instructions)
        })));
    }

    let input = body.get("input").cloned().unwrap_or(Value::Null);

    match &input {
        Value::String(s) => {
            messages.push(Some(json!({ "role": "user", "content": s })));
        },
        Value::Array(items) => {
            let mut pending_tool_calls: Vec<Value> = Vec::new();

            for item in items {
                let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

                match item_type {
                    "message" => {
                        let raw_role = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                        let role = if raw_role == "developer" || raw_role == "system" {
                            "user"
                        } else {
                            raw_role
                        };

                        let content = normalize_content_blocks(item.get("content"));

                        if !pending_tool_calls.is_empty() {
                            messages.push(Some(json!({
                                "role": "assistant",
                                "content": null,
                                "tool_calls": pending_tool_calls
                            })));
                            pending_tool_calls.clear();
                        }

                        messages.push(Some(json!({ "role": role, "content": content })));
                    },
                    "function_call" => {
                        let call_id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name =
                            item.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let arguments = item
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}")
                            .to_string();

                        pending_tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments }
                        }));
                    },
                    "function_call_output" => {
                        if !pending_tool_calls.is_empty() {
                            messages.push(Some(json!({
                                "role": "assistant",
                                "content": null,
                                "tool_calls": pending_tool_calls
                            })));
                            pending_tool_calls.clear();
                        }
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        let output = item.get("output").and_then(|o| o.as_str()).unwrap_or("");
                        messages.push(Some(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output
                        })));
                    },
                    _ => {},
                }
            }

            if !pending_tool_calls.is_empty() {
                messages.push(Some(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": pending_tool_calls
                })));
            }
        },
        _ => {},
    }

    // Repair tool call ordering (pull orphan tool results after their assistant)
    let fixed = repair_tool_order(messages);

    // Merge consecutive same-role messages
    let merged = merge_consecutive(fixed);

    // Truncate old tool outputs
    let truncated = truncate_old_tool_outputs(merged);

    // Trim to MAX_MESSAGES
    let trimmed = trim_messages(truncated);

    // Validate (remove orphan tool messages)
    let validated = validate_tool_sequence(trimmed);

    // Build the chat completions request
    let mut req = json!({
        "model": body.get("model").cloned().unwrap_or(Value::Null),
        "messages": validated,
        "stream": body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false),
        "max_tokens": body.get("max_output_tokens").and_then(serde_json::Value::as_u64).unwrap_or(16384)
    });

    if let Some(temp) = body.get("temperature")
        && !temp.is_null()
    {
        req["temperature"] = temp.clone();
    }
    if let Some(top_p) = body.get("top_p")
        && !top_p.is_null()
    {
        req["top_p"] = top_p.clone();
    }

    // Tools - filter to function type
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let supported: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"))
            .map(normalize_tool)
            .collect();
        if !supported.is_empty() {
            req["tools"] = Value::Array(supported);
        }
    }

    if let Some(tc) = body.get("tool_choice")
        && !tc.is_null()
    {
        if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
            req["tool_choice"] = json!({ "type": "function", "function": { "name": name } });
        } else {
            req["tool_choice"] = tc.clone();
        }
    }

    if let Some(effort) =
        body.get("reasoning").and_then(|r| r.get("effort")).and_then(|e| e.as_str())
    {
        req["reasoning_effort"] = json!(effort);
    }

    if let Some(ptc) = body.get("parallel_tool_calls")
        && !ptc.is_null()
    {
        req["parallel_tool_calls"] = ptc.clone();
    }

    req
}

fn normalize_content_blocks(content: Option<&Value>) -> Value {
    match content {
        None => Value::Null,
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(arr)) => {
            let mapped: Vec<Value> = arr
                .iter()
                .map(|block| {
                    let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match btype {
                        "input_text" | "output_text" => json!({
                            "type": "text",
                            "text": block.get("text").cloned().unwrap_or(Value::Null)
                        }),
                        "input_image" => {
                            let url = block
                                .get("image_url")
                                .or_else(|| block.get("url"))
                                .cloned()
                                .unwrap_or(Value::Null);
                            json!({ "type": "image_url", "image_url": { "url": url } })
                        },
                        _ => block.clone(),
                    }
                })
                .collect();

            // If single text element, unwrap to string
            if mapped.len() == 1
                && let Some(text) = mapped[0].get("text").and_then(|t| t.as_str())
                && mapped[0].get("type").and_then(|t| t.as_str()) == Some("text")
            {
                return Value::String(text.to_string());
            }

            Value::Array(mapped)
        },
        Some(other) => other.clone(),
    }
}

fn normalize_tool(t: &Value) -> Value {
    if t.get("function").is_some() {
        t.clone()
    } else {
        json!({
            "type": "function",
            "function": {
                "name": t.get("name").cloned().unwrap_or(Value::Null),
                "description": t.get("description").cloned().unwrap_or(Value::Null),
                "parameters": t.get("parameters").cloned().unwrap_or(Value::Null)
            }
        })
    }
}

/// Repair tool call ordering: ensure tool results immediately follow their assistant
fn repair_tool_order(messages: Vec<Option<Value>>) -> Vec<Value> {
    let mut fixed: Vec<Value> = Vec::new();
    let mut messages = messages;

    let len = messages.len();
    for i in 0..len {
        let Some(msg) = messages[i].take() else { continue };

        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if role == "assistant" && msg.get("tool_calls").is_some() {
            // Collect call IDs for this assistant message
            let call_ids: std::collections::HashSet<String> = msg
                .get("tool_calls")
                .and_then(|tc| tc.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()))
                        .map(std::string::ToString::to_string)
                        .collect()
                })
                .unwrap_or_default();

            fixed.push(msg);

            // Pull forward any matching tool results
            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..messages.len() {
                if let Some(m) = messages[j].take() {
                    let m_role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    let m_tc_id = m.get("tool_call_id").and_then(|id| id.as_str()).unwrap_or("");
                    if m_role == "tool" && call_ids.contains(m_tc_id) {
                        fixed.push(m);
                    } else {
                        messages[j] = Some(m);
                    }
                }
            }
        } else if role == "tool" {
            // Find the last assistant-with-tool-calls in fixed and insert after its tool results
            if let Some(last_tc_idx) = fixed.iter().rposition(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("assistant")
                    && m.get("tool_calls").is_some()
            }) {
                let mut insert_idx = last_tc_idx + 1;
                while insert_idx < fixed.len()
                    && fixed[insert_idx].get("role").and_then(|r| r.as_str()) == Some("tool")
                {
                    insert_idx += 1;
                }
                fixed.insert(insert_idx, msg);
            }
            // else orphan tool message – drop it
        } else {
            fixed.push(msg);
        }
    }

    fixed
}

/// Merge consecutive messages of the same role
fn merge_consecutive(messages: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if let Some(prev) = merged.last_mut() {
            let prev_role = prev.get("role").and_then(|r| r.as_str()).unwrap_or("");

            // Merge consecutive user messages
            if prev_role == "user"
                && role == "user"
                && prev.get("content").and_then(|c| c.as_str()).is_some()
                && msg.get("content").and_then(|c| c.as_str()).is_some()
            {
                #[allow(clippy::unwrap_used)]
                let combined = format!(
                    "{}\n\n{}",
                    prev["content"].as_str().unwrap(),
                    msg["content"].as_str().unwrap()
                );
                prev["content"] = Value::String(combined);
                continue;
            }

            // Merge consecutive assistant text messages
            if prev_role == "assistant"
                && role == "assistant"
                && prev.get("tool_calls").is_none()
                && msg.get("tool_calls").is_none()
                && prev.get("content").and_then(|c| c.as_str()).is_some()
                && msg.get("content").and_then(|c| c.as_str()).is_some()
            {
                #[allow(clippy::unwrap_used)]
                let combined = format!(
                    "{}\n\n{}",
                    prev["content"].as_str().unwrap(),
                    msg["content"].as_str().unwrap()
                );
                prev["content"] = Value::String(combined);
                continue;
            }

            // Assistant text → assistant with tool_calls: replace
            if prev_role == "assistant"
                && role == "assistant"
                && prev.get("tool_calls").is_none()
                && msg.get("tool_calls").is_some()
            {
                *prev = msg;
                continue;
            }

            // Drop text-only assistant that follows tool calls
            if prev_role == "assistant"
                && role == "assistant"
                && prev.get("tool_calls").is_some()
                && msg.get("tool_calls").is_none()
            {
                continue;
            }
        }

        merged.push(msg);
    }

    merged
}

/// Truncate old tool outputs (beyond `KEEP_RECENT_FULL` from end)
fn truncate_old_tool_outputs(mut messages: Vec<Value>) -> Vec<Value> {
    let len = messages.len();
    let cutoff = len.saturating_sub(KEEP_RECENT_FULL);

    for msg in &mut messages[..cutoff] {
        if msg.get("role").and_then(|r| r.as_str()) == Some("tool")
            && let Some(content) = msg.get("content").and_then(|c| c.as_str())
            && content.len() > TOOL_OUTPUT_MAX
        {
            let truncated = content.chars().take(TOOL_OUTPUT_MAX).collect::<String>();
            let removed = content.len() - truncated.len();
            let trimmed = format!("{truncated}\n...[output truncated, {removed} chars removed]");
            msg["content"] = Value::String(trimmed);
        }
    }

    messages
}

/// Trim conversation to `MAX_MESSAGES` keeping head + tail
fn trim_messages(messages: Vec<Value>) -> Vec<Value> {
    if messages.len() <= MAX_MESSAGES {
        return messages;
    }

    let head: Vec<Value> = messages[..2].to_vec();
    let mut tail: Vec<Value> = messages[messages.len().saturating_sub(MAX_MESSAGES - 3)..].to_vec();

    // Don't start tail with orphan tool results
    while !tail.is_empty() && tail[0].get("role").and_then(|r| r.as_str()) == Some("tool") {
        tail.remove(0);
    }

    let orig_len = messages.len();
    let mut result = head;
    result.push(json!({
        "role": "user",
        "content": "[Earlier conversation trimmed. Do not repeat previous statements or tool calls you already made. Continue with the current task. If you have enough information, respond to the user instead of making more tool calls.]"
    }));
    result.extend(tail);

    tracing::info!("[proxy] trimmed {} -> {} messages", orig_len, result.len());

    result
}

/// Remove orphan tool messages (not preceded by assistant or another tool)
fn validate_tool_sequence(messages: Vec<Value>) -> Vec<Value> {
    let mut validated: Vec<Value> = Vec::new();
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) == Some("tool") {
            if let Some(prev) = validated.last() {
                let prev_role = prev.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if prev_role == "tool"
                    || (prev_role == "assistant" && prev.get("tool_calls").is_some())
                {
                    validated.push(msg);
                }
                // else orphan – drop
            }
            // else very first message is a tool – drop
        } else {
            validated.push(msg);
        }
    }
    validated
}

// ─── Chat Completions → Responses API ──────────────────────────────────────

/// Translate a Chat Completions response to Responses API format
pub fn chat_completion_to_response(
    cc: &Value,
    model: &str,
    previous_response_id: Option<&str>,
    metadata: Option<&Value>,
) -> Value {
    let response_id = format!("resp_{}", uid());
    let created_at = cc.get("created").and_then(serde_json::Value::as_i64).unwrap_or_else(now_unix);

    let Some(choice) = cc.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()) else {
        return json!({
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": model,
            "output": [],
            "usage": translate_usage(cc.get("usage"))
        });
    };

    let Some(msg) = choice.get("message") else {
        return json!({
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": model,
            "output": [],
            "usage": translate_usage(cc.get("usage"))
        });
    };

    let mut output: Vec<Value> = Vec::new();

    // Tool calls first
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
        for tc in tool_calls {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let call_id = tc.get("id").and_then(|id| id.as_str()).unwrap_or("");

            output.push(json!({
                "type": "function_call",
                "id": format!("fc_{}", uid()),
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed"
            }));
        }
    }

    // Text content
    let raw_text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    let text = THINK_RE.replace_all(raw_text, "").trim().to_string();
    if !text.is_empty() {
        output.push(json!({
            "type": "message",
            "id": format!("msg_{}", uid()),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }]
        }));
    }

    // Refusal
    if let Some(refusal) = msg.get("refusal").and_then(|r| r.as_str())
        && !refusal.is_empty()
    {
        if let Some(msg_item) =
            output.iter_mut().find(|o| o.get("type").and_then(|t| t.as_str()) == Some("message"))
        {
            if let Some(content) = msg_item.get_mut("content").and_then(|c| c.as_array_mut()) {
                content.push(json!({ "type": "refusal", "refusal": refusal }));
            }
        } else {
            output.push(json!({
                "type": "message",
                "id": format!("msg_{}", uid()),
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "refusal", "refusal": refusal }]
            }));
        }
    }

    let finish_reason = choice.get("finish_reason").and_then(|r| r.as_str()).unwrap_or("stop");
    let (status, incomplete_details) = finish_reason_to_status(finish_reason);

    json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "previous_response_id": previous_response_id,
        "metadata": metadata.cloned().unwrap_or_else(|| json!({})),
        "usage": translate_usage(cc.get("usage")),
        "incomplete_details": incomplete_details
    })
}

pub fn finish_reason_to_status(finish_reason: &str) -> (&'static str, Value) {
    match finish_reason {
        "length" => ("incomplete", json!({ "reason": "max_output_tokens" })),
        "content_filter" => ("incomplete", json!({ "reason": "content_filter" })),
        _ => ("completed", Value::Null),
    }
}

/// Translate Chat Completions usage to Responses API usage format
pub fn translate_usage(u: Option<&Value>) -> Value {
    u.map_or_else(
        || json!({ "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }),
        |u| {
            let input = u.get("prompt_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0);
            let output =
                u.get("completion_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0);
            let total = u.get("total_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0);
            let cached = u
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let reasoning = u
                .get("completion_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            json!({
                "input_tokens": input,
                "output_tokens": output,
                "total_tokens": total,
                "input_tokens_details": { "cached_tokens": cached },
                "output_tokens_details": { "reasoning_tokens": reasoning }
            })
        },
    )
}

/// Normalize and repair messages for `MiniMax` chat completions path
/// (same tool-order repair + argument normalization)
pub fn normalize_minimax_chat_messages(messages: Vec<Value>) -> Vec<Value> {
    // Repair tool order
    let with_options: Vec<Option<Value>> = messages.into_iter().map(Some).collect();
    let fixed = repair_tool_order(with_options);
    let merged = merge_consecutive(fixed);

    // Validate
    let validated = validate_tool_sequence(merged);

    // Normalize tool call arguments
    let mut result: Vec<Value> = Vec::new();
    for mut msg in validated {
        if msg.get("role").and_then(|r| r.as_str()) == Some("assistant")
            && let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut())
        {
            for tc in tool_calls.iter_mut() {
                if let Some(func) = tc.get_mut("function") {
                    let args = func.get("arguments").cloned().unwrap_or(Value::Null);
                    let normalized = match &args {
                        Value::Null => "{}".to_string(),
                        Value::String(s) if s.is_empty() => "{}".to_string(),
                        Value::String(s) => {
                            // Validate JSON
                            if serde_json::from_str::<Value>(s).is_ok() {
                                s.clone()
                            } else {
                                let name =
                                    func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                                tracing::warn!(
                                    "[proxy] invalid tool_call arguments for {}, wrapping as JSON",
                                    name
                                );
                                json!({ "input": s }).to_string()
                            }
                        },
                        other => other.to_string(),
                    };
                    func["arguments"] = Value::String(normalized);
                }
            }
        }

        // Ensure tool content is a string
        if msg.get("role").and_then(|r| r.as_str()) == Some("tool")
            && let Some(content) = msg.get("content")
            && !content.is_string()
        {
            msg["content"] = Value::String(content.to_string());
        }

        result.push(msg);
    }

    result
}

/// Extract `web_fetch` tool calls from a chat completion response message
pub fn extract_web_fetch_calls(cc: &Value) -> Vec<Value> {
    cc.get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
        .map(|calls| {
            calls
                .iter()
                .filter(|tc| {
                    tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str())
                        == Some("web_fetch")
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Strip `web_fetch` calls from a completion and update `finish_reason` if needed
pub fn strip_web_fetch_calls(cc: &mut Value) {
    if let Some(choice) =
        cc.get_mut("choices").and_then(|c| c.as_array_mut()).and_then(|a| a.first_mut())
        && let Some(msg) = choice.get_mut("message")
        && let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut())
    {
        tool_calls.retain(|tc| {
            tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str())
                != Some("web_fetch")
        });
        if tool_calls.is_empty() {
            #[allow(clippy::unwrap_used)]
            msg.as_object_mut().unwrap().remove("tool_calls");
            if choice.get("finish_reason").and_then(|r| r.as_str()) == Some("tool_calls") {
                choice["finish_reason"] = Value::String("stop".to_string());
            }
        }
    }
}

/// Get the URLs from a list of `web_fetch` tool calls (sorted, joined with |)
pub fn web_fetch_urls_key(calls: &[Value]) -> String {
    let mut urls: Vec<String> = calls
        .iter()
        .filter_map(|tc| {
            let args =
                tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str())?;
            let parsed: Value = serde_json::from_str(args).ok()?;
            parsed.get("url")?.as_str().map(std::string::ToString::to_string)
        })
        .collect();
    urls.sort();
    urls.join("|")
}

/// Get the URL from a single `web_fetch` tool call arguments string
pub fn get_fetch_url(args_str: &str) -> String {
    serde_json::from_str::<Value>(args_str)
        .ok()
        .and_then(|v| v.get("url").and_then(|u| u.as_str()).map(std::string::ToString::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}
