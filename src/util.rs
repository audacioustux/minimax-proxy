use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;

/// Generate a 12-byte base64url-encoded random ID (same as Node crypto.randomBytes(12).toString("base64url"))
pub fn uid() -> String {
    let mut bytes = [0u8; 12];
    rand::rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Current Unix timestamp in seconds
pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .cast_signed()
}

/// Parse a comma-separated string into a deduplicated Vec<String>
pub fn parse_csv(value: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && seen.insert(s.clone()))
        .collect()
}

/// Lowercase + trim a model ID for comparison
pub fn normalize_model_id(model: &str) -> String {
    model.trim().to_lowercase()
}
