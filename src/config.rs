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
    /// Remote MCP server registry (hot-reloaded from this list).
    pub mcp: McpConfig,
    /// Time-based scheduler (cron) settings.
    pub scheduler: SchedulerConfig,
    /// Session management settings.
    pub sessions: SessionConfig,
    /// Human-in-the-loop confirmation settings.
    pub hitl: HitlConfig,
    /// Skill system (markdown skill files, hot-reloaded).
    pub skills: SkillsConfig,
    /// A2A (Agent2Agent) remote agent registry.
    pub a2a: A2aConfig,
    /// Model catalog (auto model selection by cost/capacity/benchmarks).
    pub models: ModelCatalogConfig,
    /// Auto-compaction of conversation history.
    pub compact: CompactConfig,
    /// Persistent memory.
    pub memory: MemoryConfig,
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

/// A single remote MCP server entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServerConfig {
    /// Human-readable name; also used as the tool-name prefix.
    pub name: String,
    /// Streamable-HTTP endpoint URL, e.g. `http://127.0.0.1:8080/mcp`.
    pub url: String,
    /// Optional bearer token sent as `Authorization: Bearer {token}`.
    pub token: Option<String>,
    /// Connect timeout, seconds.
    pub timeout_secs: u64,
}

/// Remote MCP registry config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpConfig {
    /// Registered remote MCP servers.
    pub servers: Vec<McpServerConfig>,
    /// Reconnect poll interval, seconds (0 = reconnect only on change).
    pub poll_secs: u64,
    /// When true, the agent exposes remote MCP tools to the LLM as callable
    /// functions (prefixed `mcp_<name>__`).
    pub expose_tools: bool,
}

/// Scheduler (cron) config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerConfig {
    /// When true, the time-based scheduler is enabled and runs cron jobs.
    pub enabled: bool,
    /// Tick resolution, seconds. The scheduler wakes this often to check for
    /// due jobs. Lower = more responsive, more CPU.
    pub tick_secs: u64,
    /// Default session id a scheduled job's prompt is dispatched into when the
    /// job itself specifies none.
    pub default_session: String,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tick_secs: 5,
            default_session: "default".to_string(),
        }
    }
}

/// Session management config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    /// Directory sessions persist state under (workspace metadata, etc.).
    pub store_dir: String,
    /// Default session id created on startup.
    pub default_session: String,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            store_dir: "sessions".to_string(),
            default_session: "default".to_string(),
        }
    }
}

/// Human-in-the-loop config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HitlConfig {
    /// When true, gated tools (scheduler add/remove, mcp call) require human
    /// approval before executing.
    pub enabled: bool,
    /// Seconds to wait for a human decision before auto-rejecting.
    pub timeout_secs: u64,
    /// Glob patterns of tool names that require confirmation (others run
    /// automatically). Empty list = confirm all gated tools.
    pub confirm_tools: Vec<String>,
}

impl Default for HitlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_secs: 120,
            confirm_tools: Vec::new(),
        }
    }
}

/// Skill system config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SkillsConfig {
    /// Directory containing skill files (markdown with frontmatter). Skills
    /// are hot-reloaded when files in this directory change.
    pub dir: Option<String>,
    /// Re-scan the skills directory this often (seconds). 0 = scan on demand
    /// only (the runtime still triggers a rescan when files change).
    pub poll_secs: u64,
    /// When true, expose loaded skills to the LLM as invocable tools
    /// (`skill_<name>`).
    pub expose_tools: bool,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            dir: None,
            poll_secs: 5,
            expose_tools: true,
        }
    }
}

/// A single remote A2A agent entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct A2aAgentConfig {
    /// Human-readable name; used as the tool-name prefix.
    pub name: String,
    /// Base URL of the remote A2A agent (Agent Card is fetched from
    /// `{url}/.well-known/agent-card.json`).
    pub url: String,
    /// Optional bearer token.
    pub token: Option<String>,
    /// Connect timeout, seconds.
    pub timeout_secs: u64,
}

/// A2A (Agent2Agent) registry config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct A2aConfig {
    /// Registered remote A2A agents.
    pub agents: Vec<A2aAgentConfig>,
    /// When true, expose each remote agent to the LLM as a callable function
    /// (`a2a_<name>`).
    pub expose_tools: bool,
}

/// Model-catalog config (auto model selection).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelCatalogConfig {
    /// Directory of YAML catalog files. When set, the agent picks a model
    /// automatically instead of using the fixed `llm.model`.
    pub dir: Option<String>,
    /// Re-scan the catalog this often (seconds). 0 = scan on demand only.
    pub poll_secs: u64,
    /// Selection strategy: best_score | best_score_under_budget |
    /// cheapest_above_floor | best_value.
    pub strategy: String,
    /// Minimum acceptable benchmark score. 0 = no floor.
    pub min_score: f64,
    /// Maximum blended cost-per-token (USD). None/0 = no cap.
    pub max_cost_per_token: Option<f64>,
    /// Minimum required context window (tokens). None/0 = no requirement.
    pub min_context_window: Option<u32>,
    /// When true, expose a `model_list` / `model_pick` tool to the LLM.
    pub expose_tools: bool,
}

impl Default for ModelCatalogConfig {
    fn default() -> Self {
        Self {
            dir: None,
            poll_secs: 60,
            strategy: "best_score".to_string(),
            min_score: 0.0,
            max_cost_per_token: None,
            min_context_window: None,
            expose_tools: true,
        }
    }
}

/// Auto-compaction config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactConfig {
    /// When true, conversation history is summarized once it exceeds the
    /// threshold, replacing old turns with a compact summary.
    pub enabled: bool,
    /// Number of stored messages (turns × 2) that triggers compaction.
    pub threshold_messages: usize,
    /// Messages to retain verbatim after compaction (most recent).
    pub keep_recent: usize,
    /// System instruction appended to the compaction prompt.
    pub prompt: String,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_messages: 20,
            keep_recent: 6,
            prompt: "Summarize the preceding conversation in a concise paragraph, \
                preserving facts, decisions, names, and any open tasks."
                .to_string(),
        }
    }
}

/// Persistent memory config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Directory under which memory files are persisted (one per sender).
    pub dir: Option<String>,
    /// When true, inject recalled memory into the system prompt.
    pub inject_into_prompt: bool,
    /// When true, expose `memory_set` / `memory_recall` tools to the LLM.
    pub expose_tools: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            dir: None,
            inject_into_prompt: true,
            expose_tools: true,
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
        if let Ok(v) = std::env::var("SLOTH_SCHEDULER_ENABLED") {
            self.scheduler.enabled = matches!(v.as_str(), "1" | "true" | "yes");
        }
        if let Ok(v) = std::env::var("SLOTH_SCHEDULER_TICK_SECS")
            && let Ok(n) = v.parse()
        {
            self.scheduler.tick_secs = n;
        }
        if let Ok(v) = std::env::var("SLOTH_SESSION_DEFAULT") {
            self.sessions.default_session = v;
        }
        if let Ok(v) = std::env::var("SLOTH_HITL_ENABLED") {
            self.hitl.enabled = matches!(v.as_str(), "1" | "true" | "yes");
        }
        if let Ok(v) = std::env::var("SLOTH_HITL_TIMEOUT_SECS")
            && let Ok(n) = v.parse()
        {
            self.hitl.timeout_secs = n;
        }
        if let Ok(v) = std::env::var("SLOTH_MCP_EXPOSE_TOOLS") {
            self.mcp.expose_tools = matches!(v.as_str(), "1" | "true" | "yes");
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
