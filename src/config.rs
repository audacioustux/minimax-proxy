use std::collections::{HashMap, HashSet};

use crate::util::{normalize_model_id, parse_csv};

pub struct Config {
    pub port: u16,

    pub minimax_base: String,
    pub minimax_key: String,
    pub minimax_models: Vec<String>,

    pub openai_base: String,
    pub openai_key: String,
    pub openai_model_prefixes: Vec<String>,

    pub default_provider: String,
    pub github_token: String,

    // Derived
    pub enabled_providers: HashSet<String>,
    pub explicit_model_provider: HashMap<String, String>,
    pub model_catalog: Vec<serde_json::Value>,
}

impl Config {
    pub fn from_env() -> Self {
        let port: u16 =
            std::env::var("PROXY_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);

        let minimax_base = std::env::var("MINIMAX_BASE_URL")
            .unwrap_or_else(|_| "https://api.minimax.io/v1".to_string());
        let minimax_key = std::env::var("MINIMAX_API_KEY").unwrap_or_default();
        let minimax_models = parse_csv(
            &std::env::var("MINIMAX_MODELS").unwrap_or_else(|_| "MiniMax-M2.7".to_string()),
        );

        let openai_base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let openai_model_prefixes = parse_csv(
            &std::env::var("OPENAI_MODEL_PREFIXES")
                .unwrap_or_else(|_| "gpt-,o1,o3,o4,codex-,chatgpt-".to_string()),
        );

        let default_provider =
            std::env::var("DEFAULT_PROVIDER").unwrap_or_default().trim().to_lowercase();

        let github_token = std::env::var("GITHUB_TOKEN").unwrap_or_else(|_| {
            std::process::Command::new("gh")
                .args(["auth", "token"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        });

        if minimax_key.is_empty() && openai_key.is_empty() {
            eprintln!(
                "At least one upstream provider key is required: set MINIMAX_API_KEY and/or OPENAI_API_KEY"
            );
            std::process::exit(1);
        }

        let mut enabled_providers = HashSet::new();
        if !minimax_key.is_empty() {
            enabled_providers.insert("minimax".to_string());
        }
        if !openai_key.is_empty() {
            enabled_providers.insert("openai".to_string());
        }

        let mut explicit_model_provider = HashMap::new();
        for m in &minimax_models {
            explicit_model_provider.insert(normalize_model_id(m), "minimax".to_string());
        }

        let mut model_catalog = vec![];
        for m in &minimax_models {
            model_catalog
                .push(serde_json::json!({ "id": m, "object": "model", "owned_by": "minimax" }));
        }

        Self {
            port,
            minimax_base,
            minimax_key,
            minimax_models,
            openai_base,
            openai_key,
            openai_model_prefixes,
            default_provider,
            github_token,
            enabled_providers,
            explicit_model_provider,
            model_catalog,
        }
    }

    pub fn get_fallback_provider(&self) -> anyhow::Result<String> {
        if !self.default_provider.is_empty()
            && self.enabled_providers.contains(&self.default_provider)
        {
            return Ok(self.default_provider.clone());
        }
        if self.enabled_providers.contains("openai") {
            return Ok("openai".to_string());
        }
        if self.enabled_providers.contains("minimax") {
            return Ok("minimax".to_string());
        }
        anyhow::bail!("No providers are enabled")
    }

    pub fn resolve_provider_for_model(&self, model: &str) -> String {
        let normalized = normalize_model_id(model);
        if !normalized.is_empty() {
            if let Some(provider) = self.explicit_model_provider.get(&normalized)
                && self.enabled_providers.contains(provider)
            {
                return provider.clone();
            }
            if self.enabled_providers.contains("minimax") && normalized.contains("minimax") {
                return "minimax".to_string();
            }
            if self.enabled_providers.contains("openai") {
                let looks_openai = self
                    .openai_model_prefixes
                    .iter()
                    .any(|p| normalized.starts_with(&p.to_lowercase()));
                if looks_openai {
                    return "openai".to_string();
                }
            }
        }
        self.get_fallback_provider().unwrap_or_else(|_| "openai".to_string())
    }
}
