//! Agent configuration.
//!
//! Values are loaded from `config.toml` (next to the binary / in the working
//! dir) and can be overridden by environment variables. Environment variables
//! take precedence so deployments can be configured without touching files.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level agent configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Bridge WebSocket connection settings.
    pub bridge: BridgeConfig,
    /// OpenAI-compatible chat completion backend.
    pub llm: LlmConfig,
    /// Conversation history retention.
    pub history: HistoryConfig,
    /// Observability / tracing settings.
    pub observability: ObservabilityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BridgeConfig {
    /// WebSocket URL of the OpenClaw bridge, e.g. `ws://127.0.0.1:9300/bridge`.
    pub url: String,
    /// Channel id to subscribe to (e.g. `liangzimixin`, `openclaw-weixin`).
    pub channel: String,
    /// Account id within the channel (defaults to `default`).
    pub account_id: String,
    /// Reconnect backoff base, in milliseconds.
    pub reconnect_ms: u64,
    /// Maximum reconnect backoff, in milliseconds.
    pub reconnect_max_ms: u64,
    /// Heartbeat ping interval, in milliseconds. 0 disables.
    pub heartbeat_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// OpenAI-compatible base URL, e.g. `http://172.17.0.1:8317/v1`.
    pub base_url: String,
    /// Model id to use for completions.
    pub model: String,
    /// Optional API key. When empty, no `Authorization` header is sent — the
    /// local gateway at 172.17.0.1:8317 does not require one and in fact
    /// rejects unknown bearer tokens.
    pub api_key: Option<String>,
    /// System prompt prepended to every conversation.
    pub system_prompt: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Per-request timeout, in seconds.
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HistoryConfig {
    /// Maximum messages kept per sender (excluding the system prompt).
    pub max_messages: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservabilityConfig {
    /// `text`, `json`, or `pretty`.
    pub log_format: String,
    /// tracing filter directive, e.g. `info,sloth_agent=debug`.
    pub log_filter: String,
    /// Service name emitted on structured log lines.
    pub service_name: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            url: "ws://127.0.0.1:9300/bridge".to_string(),
            channel: "liangzimixin".to_string(),
            account_id: "default".to_string(),
            reconnect_ms: 1_000,
            reconnect_max_ms: 30_000,
            heartbeat_ms: 25_000,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://172.17.0.1:8317/v1".to_string(),
            model: "glm-5.2".to_string(),
            api_key: None,
            system_prompt: "You are Sloth, a friendly, concise AI assistant replying over a \
                chat bridge. Keep replies short and natural."
                .to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1024),
            timeout_secs: 60,
        }
    }
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self { max_messages: 20 }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_format: "text".to_string(),
            log_filter: "info,sloth_agent=debug".to_string(),
            service_name: "sloth-agent".to_string(),
        }
    }
}

impl Config {
    /// Load config from `config.toml` (if present) with environment overrides.
    #[allow(dead_code)]
    pub fn load() -> Result<Self> {
        let mut cfg = Self::load_from_file("config.toml");
        cfg.apply_env();
        Ok(cfg)
    }

    /// Load from a specific TOML file, falling back to defaults if missing.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Self {
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<Self>(&text) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!(
                        "warning: failed to parse {} ({e}); using defaults",
                        path.as_ref().display()
                    );
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Apply `SLOTH_*` environment overrides on top of file config.
    pub fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("SLOTH_BRIDGE_URL") {
            self.bridge.url = v;
        }
        if let Ok(v) = std::env::var("SLOTH_CHANNEL") {
            self.bridge.channel = v;
        }
        if let Ok(v) = std::env::var("SLOTH_ACCOUNT_ID") {
            self.bridge.account_id = v;
        }
        if let Ok(v) = std::env::var("SLOTH_LLM_BASE_URL") {
            self.llm.base_url = v;
        }
        if let Ok(v) = std::env::var("SLOTH_LLM_MODEL") {
            self.llm.model = v;
        }
        if let Ok(v) = std::env::var("SLOTH_LLM_API_KEY") {
            // Empty string means "no key".
            self.llm.api_key = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = std::env::var("SLOTH_LLM_SYSTEM_PROMPT") {
            self.llm.system_prompt = v;
        }
        if let Ok(v) = std::env::var("SLOTH_LOG_FORMAT") {
            self.observability.log_format = v;
        }
        if let Ok(v) = std::env::var("SLOTH_LOG_FILTER") {
            self.observability.log_filter = v;
        }
    }

    /// Validate the configuration, returning a helpful error if invalid.
    pub fn validate(&self) -> Result<()> {
        if url::Url::parse(&self.bridge.url).is_err() {
            anyhow::bail!("bridge.url is not a valid URL: {}", self.bridge.url);
        }
        if self.bridge.channel.trim().is_empty() {
            anyhow::bail!("bridge.channel must not be empty");
        }
        // base_url need not be a full URL with host, but should be non-empty.
        if self.llm.base_url.trim().is_empty() {
            anyhow::bail!("llm.base_url must not be empty");
        }
        if self.llm.model.trim().is_empty() {
            anyhow::bail!("llm.model must not be empty");
        }
        Ok(())
    }
}

/// Load config, surfacing file-read errors only when the path was explicitly
/// provided (e.g. via `--config`). The implicit `config.toml` is optional.
pub fn load_optional_explicit(explicit: Option<&str>) -> Result<Config> {
    let mut cfg = match explicit {
        Some(p) => {
            let text =
                std::fs::read_to_string(p).with_context(|| format!("reading config file {p}"))?;
            toml::from_str(&text).with_context(|| format!("parsing config file {p}"))?
        }
        None => Config::load_from_file("config.toml"),
    };
    cfg.apply_env();
    Ok(cfg)
}
