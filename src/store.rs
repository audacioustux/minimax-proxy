use std::sync::Mutex;

use indexmap::IndexMap;
use serde_json::Value;

const STORE_TTL_SECS: u64 = 60 * 60; // 1 hour
const STORE_MAX: usize = 500;
pub const MAX_CONSECUTIVE_TOOL_CALLS: u32 = 20;

#[derive(Debug, Clone)]
pub struct ResponseEntry {
    pub provider: String,
    pub input: Vec<Value>,
    pub output: Vec<Value>,
    pub previous_response_id: Option<String>,
    pub stored_at: std::time::Instant,
    pub consecutive_tool_calls: u32,
}

pub struct ResponseStore {
    data: Mutex<IndexMap<String, ResponseEntry>>,
}

#[allow(clippy::unwrap_used)]
impl ResponseStore {
    pub fn new() -> Self {
        Self { data: Mutex::new(IndexMap::new()) }
    }

    pub fn get(&self, id: &str) -> Option<ResponseEntry> {
        let data = self.data.lock().unwrap();
        data.get(id).cloned()
    }

    pub fn store(
        &self,
        id: &str,
        provider: &str,
        input: Vec<Value>,
        output: Vec<Value>,
        previous_response_id: Option<String>,
    ) {
        if id.is_empty() {
            return;
        }

        let mut data = self.data.lock().unwrap();

        // Evict expired entries if at capacity
        if data.len() >= STORE_MAX {
            let now = std::time::Instant::now();
            let ttl = std::time::Duration::from_secs(STORE_TTL_SECS);
            data.retain(|_, v| now.duration_since(v.stored_at) <= ttl);

            // If still at capacity, remove oldest
            if data.len() >= STORE_MAX
                && let Some(oldest_key) = data.keys().next().cloned()
            {
                data.shift_remove(&oldest_key);
            }
        }

        // Calculate consecutive tool calls for circuit breaker
        let is_tool_call_only = !output.is_empty()
            && output
                .iter()
                .all(|o| o.get("type").and_then(|t| t.as_str()) == Some("function_call"));

        let consecutive_tool_calls = if is_tool_call_only {
            previous_response_id
                .as_ref()
                .and_then(|prev_id| data.get(prev_id.as_str()))
                .map_or(1, |p| p.consecutive_tool_calls + 1)
        } else {
            0
        };

        let entry = ResponseEntry {
            provider: provider.to_string(),
            input,
            output,
            previous_response_id,
            stored_at: std::time::Instant::now(),
            consecutive_tool_calls,
        };

        tracing::info!(
            "[proxy] stored response {} (provider={}, store size: {}{})",
            id,
            provider,
            data.len() + 1,
            if consecutive_tool_calls > 0 {
                format!(", consecutive_tc: {consecutive_tool_calls}")
            } else {
                String::new()
            }
        );

        data.insert(id.to_string(), entry);
    }

    /// Walk the `previous_response_id` chain and collect all input+output items in order
    pub fn resolve_chain(&self, previous_response_id: &str) -> Vec<Value> {
        let data = self.data.lock().unwrap();

        let mut chain: Vec<ResponseEntry> = Vec::new();
        let mut current_id = previous_response_id.to_string();
        let mut visited = std::collections::HashSet::new();

        loop {
            if visited.contains(&current_id) {
                break;
            }
            visited.insert(current_id.clone());

            if let Some(entry) = data.get(&current_id) {
                let next = entry.previous_response_id.clone();
                chain.push(entry.clone());
                match next {
                    Some(next_id) => current_id = next_id,
                    None => break,
                }
            } else {
                tracing::warn!("[proxy] previous_response_id {} not found in store", current_id);
                break;
            }
        }

        // chain is newest-first; reverse to get oldest-first
        chain.reverse();

        let mut items = Vec::new();
        for entry in chain {
            items.extend(entry.input);
            items.extend(entry.output);
        }
        items
    }
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::new()
    }
}
