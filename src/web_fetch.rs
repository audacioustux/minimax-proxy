use std::{collections::HashMap, sync::LazyLock, time::Duration};

use regex::Regex;
use serde_json::Value;

#[allow(clippy::expect_used)]
static URL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://").expect("valid regex"));

const FETCH_TIMEOUT_SECS: u64 = 15;
const FETCH_MAX_BODY: usize = 50_000;
pub const MAX_FETCH_LOOPS: usize = 5;

/// Fetch a URL and return content as plain text/markdown
/// Tries Accept: text/markdown first, falls back to text/plain
pub async fn fetch_url(client: &reqwest::Client, url: &str) -> String {
    let result = tokio::time::timeout(
        Duration::from_secs(FETCH_TIMEOUT_SECS),
        client
            .get(url)
            .header("Accept", "text/markdown")
            .header("User-Agent", "Mozilla/5.0 (compatible; CodexProxy/1.0)")
            .send(),
    )
    .await;

    match result {
        Err(_) => "Fetch error: request timed out".to_string(),
        Ok(Err(e)) => format!("Fetch error: {e}"),
        Ok(Ok(resp)) => {
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return format!(
                    "HTTP {} {}\n{}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or(""),
                    body
                )[..FETCH_MAX_BODY.min(
                    format!(
                        "HTTP {} {}\n{}",
                        status.as_u16(),
                        status.canonical_reason().unwrap_or(""),
                        body
                    )
                    .len(),
                )]
                    .to_string();
            }
            match resp.text().await {
                Err(e) => format!("Read error: {e}"),
                Ok(text) => {
                    if text.len() > FETCH_MAX_BODY {
                        format!(
                            "{}\n...[content truncated, {} chars omitted]",
                            &text[..FETCH_MAX_BODY],
                            text.len() - FETCH_MAX_BODY
                        )
                    } else {
                        text
                    }
                },
            }
        },
    }
}

/// Raw HTTP fetch (for non-GET requests)
pub async fn raw_fetch(
    client: &reqwest::Client,
    github_token: &str,
    url: &str,
    method: &str,
    headers: &HashMap<String, String>,
    body: Option<&str>,
) -> String {
    let method_upper = method.to_uppercase();
    let Ok(req_method) = reqwest::Method::from_bytes(method_upper.as_bytes()) else {
        return format!("Fetch error: invalid method '{method}'");
    };

    let mut req = client
        .request(req_method.clone(), url)
        .header(
            "User-Agent",
            headers
                .get("User-Agent")
                .or_else(|| headers.get("user-agent"))
                .map_or("Mozilla/5.0 (compatible; CodexProxy/1.0)", std::string::String::as_str),
        )
        .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS));

    // GitHub API auth injection
    if !github_token.is_empty()
        && url.contains("api.github.com")
        && !headers.contains_key("Authorization")
        && !headers.contains_key("authorization")
    {
        req = req.header("Authorization", format!("Bearer {github_token}"));
    }

    for (k, v) in headers {
        if k.to_lowercase() != "user-agent" {
            req = req.header(k.as_str(), v.as_str());
        }
    }

    if let Some(body_str) = body
        && matches!(method_upper.as_str(), "POST" | "PUT" | "PATCH")
    {
        req = req.body(body_str.to_string());
    }

    let result = req.send().await;
    match result {
        Err(e) => {
            if e.is_timeout() {
                "Fetch error: request timed out".to_string()
            } else {
                format!("Fetch error: {e}")
            }
        },
        Ok(resp) => {
            let status_code = resp.status().as_u16();
            let status_text = resp.status().canonical_reason().unwrap_or("").to_string();
            let ct = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            let status_line = format!("HTTP {status_code} {status_text}");

            if matches!(method_upper.as_str(), "HEAD" | "OPTIONS") {
                let hdrs: Vec<String> = resp
                    .headers()
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_str().unwrap_or("")))
                    .collect();
                return format!("{}\n{}", status_line, hdrs.join("\n"));
            }

            if ct.contains("image")
                || ct.contains("audio")
                || ct.contains("video")
                || ct.contains("octet-stream")
            {
                return format!("{status_line}\nContent-Type: {ct}\n(binary content, not shown)");
            }

            match resp.text().await {
                Err(e) => format!("{status_line}\n\nRead error: {e}"),
                Ok(text) => {
                    if text.len() > FETCH_MAX_BODY {
                        format!(
                            "{}\n\n{}\n...[truncated, {} chars omitted]",
                            status_line,
                            &text[..FETCH_MAX_BODY],
                            text.len() - FETCH_MAX_BODY
                        )
                    } else {
                        format!("{status_line}\n\n{text}")
                    }
                },
            }
        },
    }
}

/// Execute a `web_fetch` tool call (routes to jina or raw based on method)
pub async fn execute_web_fetch(
    client: &reqwest::Client,
    github_token: &str,
    args: &Value,
) -> String {
    let url = match args.get("url").and_then(|u| u.as_str()) {
        Some(u) => u.to_string(),
        None => return "Error: no URL provided".to_string(),
    };

    let method = args.get("method").and_then(|m| m.as_str()).unwrap_or("GET");

    let headers: HashMap<String, String> = args
        .get("headers")
        .and_then(|h| h.as_object())
        .map(|obj| {
            obj.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect()
        })
        .unwrap_or_default();

    let body = args.get("body").and_then(|b| b.as_str()).map(std::string::ToString::to_string);

    if method.to_uppercase() == "GET" {
        fetch_url(client, &url).await
    } else {
        raw_fetch(client, github_token, &url, method, &headers, body.as_deref()).await
    }
}

/// Check if a string contains a URL
fn str_has_url(s: &str) -> bool {
    URL_RE.is_match(s)
}

/// Check if a `serde_json` content value contains any URLs
pub fn content_has_url(content: &Value) -> bool {
    match content {
        Value::String(s) => str_has_url(s),
        Value::Array(arr) => arr.iter().any(|part| {
            if let Some(text) = part.get("text").and_then(|t| t.as_str())
                && str_has_url(text)
            {
                return true;
            }
            if let Some(url) = part.get("url").and_then(|u| u.as_str())
                && str_has_url(url)
            {
                return true;
            }
            if let Some(url) = part.get("image_url").and_then(|u| u.as_str())
                && str_has_url(url)
            {
                return true;
            }
            if let Some(url) =
                part.get("image_url").and_then(|u| u.get("url")).and_then(|u| u.as_str())
                && str_has_url(url)
            {
                return true;
            }
            false
        }),
        _ => false,
    }
}

/// Check if any message in a conversation contains URLs
pub fn conversation_has_urls(messages: &[Value]) -> bool {
    messages.iter().any(|msg| msg.get("content").is_some_and(content_has_url))
}

/// The `web_fetch` tool definition to inject
pub fn web_fetch_tool_def() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL over HTTP/HTTPS. Use this when you need to retrieve content from a web URL. Returns HTTP status and response body as clean markdown. Supports all HTTP methods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch (http:// or https://)" },
                    "method": { "type": "string", "enum": ["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"], "description": "HTTP method (default: GET)" },
                    "headers": { "type": "object", "description": "Optional HTTP headers as key-value pairs" },
                    "body": { "type": "string", "description": "Request body for POST/PUT/PATCH requests" }
                },
                "required": ["url"]
            }
        }
    })
}

/// Add `web_fetch` tool if not already present
pub fn ensure_web_fetch_tool(tools: Option<&Value>) -> Value {
    let mut list: Vec<Value> = tools.and_then(|t| t.as_array()).cloned().unwrap_or_default();

    let already_present = list.iter().any(|tool| {
        tool.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str())
            == Some("web_fetch")
            || tool.get("name").and_then(|n| n.as_str()) == Some("web_fetch")
    });

    if !already_present {
        list.push(web_fetch_tool_def());
    }
    Value::Array(list)
}

/// Add `web_fetch` hint message if not already present
pub fn ensure_web_fetch_hint(messages: Vec<Value>) -> Vec<Value> {
    let hint = "[System: You have a `web_fetch` tool available for making HTTP requests. Use it instead of curl, wget, or other shell-based HTTP tools. Call web_fetch with {\"url\": \"...\"} to fetch any URL. It supports GET, HEAD, POST, PUT, DELETE, PATCH, and OPTIONS methods.]";
    let already = messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content").and_then(|c| c.as_str()) == Some(hint)
    });
    if already {
        return messages;
    }
    let mut msgs = messages;
    msgs.push(serde_json::json!({ "role": "user", "content": hint }));
    msgs
}
