//! Function-calling tool registry.
//!
//! The agent exposes a fixed set of built-in tools to the LLM and, when
//! configured, the remote MCP tools. Each built-in tool has a name, a JSON
//! Schema for its arguments, and an async executor. The [`ToolRouter`]
//! resolves a tool call into an execution, applying HITL confirmation
//! gating where appropriate.
//!
//! Tools are grouped:
//! - **scheduler** — `scheduler_add_job`, `scheduler_remove_job`,
//!   `scheduler_list_jobs`: set up and use the time-based scheduler.
//! - **mcp** — `mcp_<server>__<tool>`: call remote MCP tools.
//! - **session** — `session_switch`, `session_set_workspace`, `session_list`:
//!   manage sessions.
//! - **mcp_admin** — `mcp_list_servers`, `mcp_reload`: list/reload remote MCP
//!   servers (the hot-reload entry point the LLM can drive).

use std::sync::Arc;

use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::info;

use crate::a2a::A2aRegistry;
use crate::config::{A2aAgentConfig, McpServerConfig};
use crate::hitl::{HitlBroker, Outcome};
use crate::mcp::McpRegistry;
use crate::scheduler::Scheduler;
use crate::session::SessionManager;
use crate::skill::SkillRegistry;

/// Outcome of executing a tool call.
#[derive(Debug, Clone, Serialize)]
pub struct ToolOutcome {
    /// Text returned to the model as the tool result.
    pub content: String,
    /// True if execution was blocked/refused (HITL denial, error, etc.).
    pub is_error: bool,
}

impl ToolOutcome {
    fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }
    fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Built-in tool name constants.
pub mod names {
    pub const SCHED_ADD: &str = "scheduler_add_job";
    pub const SCHED_REMOVE: &str = "scheduler_remove_job";
    pub const SCHED_LIST: &str = "scheduler_list_jobs";
    pub const SESS_SWITCH: &str = "session_switch";
    pub const SESS_SET_WS: &str = "session_set_workspace";
    pub const SESS_LIST: &str = "session_list";
    pub const MCP_LIST_SERVERS: &str = "mcp_list_servers";
    pub const MCP_RELOAD: &str = "mcp_reload";
    pub const SKILL_LIST: &str = "skill_list";
    pub const SKILL_RELOAD: &str = "skill_reload";
    pub const A2A_LIST: &str = "a2a_list_agents";
    pub const A2A_RELOAD: &str = "a2a_reload";
    pub const MODEL_LIST: &str = "model_list";
    pub const MODEL_PICK: &str = "model_pick";
    pub const MEM_SET: &str = "memory_set";
    pub const MEM_RECALL: &str = "memory_recall";
    /// Prefix for invocable skill tools.
    pub const SKILL_PREFIX: &str = "skill_";
    /// Prefix for invocable A2A agent tools.
    pub const A2A_PREFIX: &str = "a2a_";
}

/// The tool router. Cloneable; all collaborators shared behind `Arc`s.
#[derive(Clone)]
pub struct ToolRouter {
    pub scheduler: Arc<Scheduler>,
    pub sessions: Arc<SessionManager>,
    pub mcp: Arc<McpRegistry>,
    pub skills: Arc<SkillRegistry>,
    pub a2a: Arc<A2aRegistry>,
    pub catalog: Arc<crate::model_catalog::Catalog>,
    pub memory: Arc<crate::memory::MemoryStore>,
    pub hitl: Arc<HitlBroker>,
    /// The desired MCP server list, used by the `mcp_reload` tool.
    mcp_desired: Arc<tokio::sync::Mutex<Vec<McpServerConfig>>>,
    /// The desired A2A agent list, used by the `a2a_reload` tool.
    a2a_desired: Arc<tokio::sync::Mutex<Vec<A2aAgentConfig>>>,
    expose_mcp: bool,
    expose_skills: bool,
    expose_a2a: bool,
    expose_models: bool,
    expose_memory: bool,
    /// Model-selection options mirrored from config (used by `model_pick`).
    model_opts: Arc<crate::model_catalog::PickOptions>,
}

impl ToolRouter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scheduler: Arc<Scheduler>,
        sessions: Arc<SessionManager>,
        mcp: Arc<McpRegistry>,
        skills: Arc<SkillRegistry>,
        a2a: Arc<A2aRegistry>,
        catalog: Arc<crate::model_catalog::Catalog>,
        memory: Arc<crate::memory::MemoryStore>,
        hitl: Arc<HitlBroker>,
        mcp_desired: Vec<McpServerConfig>,
        a2a_desired: Vec<A2aAgentConfig>,
        expose_mcp: bool,
        expose_skills: bool,
        expose_a2a: bool,
        expose_models: bool,
        expose_memory: bool,
    ) -> Self {
        Self::with_model_opts(
            scheduler,
            sessions,
            mcp,
            skills,
            a2a,
            catalog,
            memory,
            hitl,
            mcp_desired,
            a2a_desired,
            expose_mcp,
            expose_skills,
            expose_a2a,
            expose_models,
            expose_memory,
            crate::model_catalog::PickOptions::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_model_opts(
        scheduler: Arc<Scheduler>,
        sessions: Arc<SessionManager>,
        mcp: Arc<McpRegistry>,
        skills: Arc<SkillRegistry>,
        a2a: Arc<A2aRegistry>,
        catalog: Arc<crate::model_catalog::Catalog>,
        memory: Arc<crate::memory::MemoryStore>,
        hitl: Arc<HitlBroker>,
        mcp_desired: Vec<McpServerConfig>,
        a2a_desired: Vec<A2aAgentConfig>,
        expose_mcp: bool,
        expose_skills: bool,
        expose_a2a: bool,
        expose_models: bool,
        expose_memory: bool,
        model_opts: crate::model_catalog::PickOptions,
    ) -> Self {
        Self {
            scheduler,
            sessions,
            mcp,
            skills,
            a2a,
            catalog,
            memory,
            hitl,
            mcp_desired: Arc::new(tokio::sync::Mutex::new(mcp_desired)),
            a2a_desired: Arc::new(tokio::sync::Mutex::new(a2a_desired)),
            expose_mcp,
            expose_skills,
            expose_a2a,
            expose_models,
            expose_memory,
            model_opts: Arc::new(model_opts),
        }
    }

    /// The OpenAI tool definitions to send to the model.
    pub async fn tool_definitions(&self) -> Vec<ChatCompletionTools> {
        let mut tools = Vec::new();

        // Scheduler tools.
        tools.push(tool_def(
            names::SCHED_ADD,
            "Schedule a time-based job that runs a prompt on a cron schedule (UTC). Use this to set up recurring or delayed tasks.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Human-readable job name." },
                    "cron": { "type": "string", "description": "Cron expression in UTC. 5-field minute-granular (minute hour dom month dow, e.g. '0 9 * * 1-5') OR 6-field second-precision (second minute hour dom month dow, e.g. '*/10 * * * * *' fires every 10 seconds)." },
                    "prompt": { "type": "string", "description": "Prompt to run each time the job fires." },
                    "session_id": { "type": "string", "description": "Session id the prompt runs under. Optional; defaults to 'default'." }
                },
                "required": ["name", "cron", "prompt"]
            }),
        ));
        tools.push(tool_def(
            names::SCHED_REMOVE,
            "Remove a scheduled job by id.",
            json!({
                "type": "object",
                "properties": { "id": { "type": "string" } },
                "required": ["id"]
            }),
        ));
        tools.push(tool_def(
            names::SCHED_LIST,
            "List all currently scheduled jobs.",
            json!({ "type": "object", "properties": {} }),
        ));

        // Session tools.
        tools.push(tool_def(
            names::SESS_SWITCH,
            "Switch the active session for the current user to an existing session.",
            json!({
                "type": "object",
                "properties": {
                    "sender_id": { "type": "string" },
                    "session_id": { "type": "string" }
                },
                "required": ["sender_id", "session_id"]
            }),
        ));
        tools.push(tool_def(
            names::SESS_SET_WS,
            "Set the working directory (workspace) of a session. Creates the directory if missing.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "workspace": { "type": "string", "description": "Absolute directory path." }
                },
                "required": ["session_id", "workspace"]
            }),
        ));
        tools.push(tool_def(
            names::SESS_LIST,
            "List all sessions.",
            json!({ "type": "object", "properties": {} }),
        ));

        // MCP admin tools.
        tools.push(tool_def(
            names::MCP_LIST_SERVERS,
            "List connected remote MCP servers and their tool counts.",
            json!({ "type": "object", "properties": {} }),
        ));
        tools.push(tool_def(
            names::MCP_RELOAD,
            "Hot-reload the remote MCP server registry: connect newly-added servers and drop removed ones, without restarting.",
            json!({
                "type": "object",
                "properties": {
                    "servers": {
                        "type": "array",
                        "description": "The desired full server list. Omit to reload from current configuration.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "url": { "type": "string" },
                                "token": { "type": "string" },
                                "timeout_secs": { "type": "integer" }
                            },
                            "required": ["name", "url"]
                        }
                    }
                }
            }),
        ));

        // Remote MCP tools (one per server tool), when exposed.
        if self.expose_mcp {
            for rt in self.mcp.routed_tools().await {
                let schema = if rt.input_schema.is_object() {
                    rt.input_schema.clone()
                } else {
                    json!({ "type": "object", "properties": {} })
                };
                tools.push(tool_def(
                    &rt.qualified_name,
                    rt.description
                        .as_deref()
                        .unwrap_or("Call a remote MCP tool."),
                    schema,
                ));
            }
        }

        // Skill admin tools.
        tools.push(tool_def(
            names::SKILL_LIST,
            "List all loaded skills (markdown skill files, hot-reloaded).",
            json!({ "type": "object", "properties": {} }),
        ));
        tools.push(tool_def(
            names::SKILL_RELOAD,
            "Hot-reload skills from the skills directory: pick up added/edited/removed skill files without restarting.",
            json!({
                "type": "object",
                "properties": {
                    "dir": { "type": "string", "description": "Optional: override the skills directory to scan." }
                }
            }),
        ));
        // Invocable skill tools (one per loaded skill), when exposed.
        if self.expose_skills {
            for s in self.skills.list().await {
                let name = format!("{}{}", names::SKILL_PREFIX, s.name);
                let desc = s
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Invoke the '{}' skill.", s.name));
                tools.push(tool_def(&name, &desc, s.input_schema()));
            }
        }

        // A2A admin tools.
        tools.push(tool_def(
            names::A2A_LIST,
            "List connected remote A2A agents.",
            json!({ "type": "object", "properties": {} }),
        ));
        tools.push(tool_def(
            names::A2A_RELOAD,
            "Hot-reload the remote A2A agent registry: connect newly-added agents and drop removed ones.",
            json!({
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "description": "Desired full agent list. Omit to reload from current configuration.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "url": { "type": "string" },
                                "token": { "type": "string" },
                                "timeout_secs": { "type": "integer" }
                            },
                            "required": ["name", "url"]
                        }
                    }
                }
            }),
        ));
        // Invocable A2A agent tools (one per connected agent), when exposed.
        if self.expose_a2a {
            for name in self.a2a.agent_names().await {
                let tool = format!("{}{}", names::A2A_PREFIX, name);
                tools.push(
                    tool_def(
                        &tool,
                        &format!("Send a prompt to the remote A2A agent '{name}' and return its reply."),
                        json!({
                            "type": "object",
                            "properties": {
                                "prompt": { "type": "string", "description": "Prompt text to send to the agent." }
                            },
                            "required": ["prompt"]
                        }),
                    ),
                );
            }
        }

        // Model catalog tools.
        if self.expose_models {
            tools.push(tool_def(
                names::MODEL_LIST,
                "List all models in the catalog with their cost, context window, and benchmark scores.",
                json!({ "type": "object", "properties": {} }),
            ));
            tools.push(tool_def(
                names::MODEL_PICK,
                "Pick the best model from the catalog given optional constraints. Returns the model id + rationale.",
                json!({
                    "type": "object",
                    "properties": {
                        "strategy": { "type": "string", "description": "best_score | best_score_under_budget | cheapest_above_floor | best_value" },
                        "min_score": { "type": "number" },
                        "max_cost_per_token": { "type": "number" },
                        "min_context_window": { "type": "integer" }
                    }
                }),
            ));
        }

        // Memory tools.
        if self.expose_memory {
            tools.push(tool_def(
                names::MEM_SET,
                "Persist a fact (key/value) about the current user so the agent remembers it across conversations.",
                json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string", "description": "Fact key, e.g. 'timezone' or 'preferred_language'." },
                        "value": { "type": "string", "description": "Fact value." }
                    },
                    "required": ["key", "value"]
                }),
            ));
            tools.push(tool_def(
                names::MEM_RECALL,
                "Recall a stored fact by key for the current user, or list all facts when no key is given.",
                json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string", "description": "Optional fact key; omit to list all." }
                    }
                }),
            ));
        }

        tools
    }

    /// Execute a tool call by name with the given arguments object.
    /// Applies HITL gating where configured.
    pub async fn execute(&self, tool: &str, args: &Value, sender_id: &str) -> ToolOutcome {
        // HITL gate.
        if self.hitl.requires_confirmation(tool) {
            let summary = summarize_call(tool, args);
            let pending = self.hitl.new_pending(tool, &summary, "default", sender_id);
            info!(%tool, hitl_id = %pending.id, "HITL confirmation requested");
            // Register + surface to the runtime; it asks the human and calls
            // `hitl.resolve(...)` with the decision.
            let rx = self.hitl.register(pending.clone()).await;
            self.hitl.publish(pending).await;
            let outcome = self.hitl.await_decision(rx).await;
            match outcome {
                Outcome::Approved => info!(%tool, "HITL approved"),
                Outcome::Denied => {
                    return ToolOutcome::err(format!("Tool call {tool} was denied by the human."));
                }
                Outcome::TimedOut => {
                    return ToolOutcome::err(format!(
                        "Tool call {tool} timed out waiting for human approval."
                    ));
                }
            }
        }

        self.dispatch(tool, args, sender_id).await
    }

    async fn dispatch(&self, tool: &str, args: &Value, sender_id: &str) -> ToolOutcome {
        match tool {
            names::SCHED_ADD => {
                let Some(name) = args.get("name").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'name'");
                };
                let Some(cron) = args.get("cron").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'cron'");
                };
                let Some(prompt) = args.get("prompt").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'prompt'");
                };
                let session_id = args
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                match self.scheduler.add(crate::scheduler::ScheduledJob {
                    id: String::new(),
                    name: name.to_string(),
                    cron: cron.to_string(),
                    prompt: prompt.to_string(),
                    session_id,
                    // Reply to the user who scheduled the job, so a fired job's
                    // output reaches them rather than broadcasting to the channel.
                    reply_to: Some(sender_id.to_string()),
                }) {
                    Ok(id) => ToolOutcome::ok(json!({ "id": id, "scheduled": true }).to_string()),
                    Err(e) => ToolOutcome::err(format!("failed to schedule: {e:#}")),
                }
            }
            names::SCHED_REMOVE => {
                let Some(id) = args.get("id").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'id'");
                };
                let removed = self.scheduler.remove(id);
                ToolOutcome::ok(json!({ "removed": removed }).to_string())
            }
            names::SCHED_LIST => {
                let jobs = self.scheduler.list();
                ToolOutcome::ok(serde_json::to_string(&json!({ "jobs": jobs })).unwrap_or_default())
            }
            names::SESS_SWITCH => {
                let Some(session_id) = args.get("session_id").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'session_id'");
                };
                match self.sessions.switch(sender_id, session_id).await {
                    Ok(s) => ToolOutcome::ok(serde_json::to_string(&s).unwrap_or_default()),
                    Err(e) => ToolOutcome::err(format!("{e:#}")),
                }
            }
            names::SESS_SET_WS => {
                let Some(session_id) = args.get("session_id").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'session_id'");
                };
                let Some(workspace) = args.get("workspace").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'workspace'");
                };
                match self
                    .sessions
                    .set_workspace(session_id, workspace.into())
                    .await
                {
                    Ok(s) => ToolOutcome::ok(serde_json::to_string(&s).unwrap_or_default()),
                    Err(e) => ToolOutcome::err(format!("{e:#}")),
                }
            }
            names::SESS_LIST => {
                let list = self.sessions.list().await;
                ToolOutcome::ok(serde_json::to_string(&list).unwrap_or_default())
            }
            names::MCP_LIST_SERVERS => {
                let tools = self.mcp.routed_tools().await;
                let servers: Vec<Value> = tools
                    .iter()
                    .fold(
                        std::collections::HashMap::<String, u32>::new(),
                        |mut acc, t| {
                            *acc.entry(t.server.clone()).or_default() += 1;
                            acc
                        },
                    )
                    .into_iter()
                    .map(|(s, n)| json!({ "server": s, "tools": n }))
                    .collect();
                ToolOutcome::ok(json!({ "servers": servers }).to_string())
            }
            names::MCP_RELOAD => {
                let desired = if let Some(servers) = args.get("servers") {
                    parse_servers(servers)
                } else {
                    self.mcp_desired.lock().await.clone()
                };
                match self.mcp.reload(&desired).await {
                    Ok(r) => {
                        // Update the cached desired list.
                        *self.mcp_desired.lock().await = desired;
                        ToolOutcome::ok(serde_json::to_string(&r).unwrap_or_default())
                    }
                    Err(e) => ToolOutcome::err(format!("reload failed: {e:#}")),
                }
            }
            names::SKILL_LIST => {
                let skills = self.skills.list().await;
                let summary: Vec<Value> = skills
                    .iter()
                    .map(|s| {
                        json!({
                            "name": s.name,
                            "description": s.description,
                            "args": s.arguments.iter().map(|a| a.name.clone()).collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                ToolOutcome::ok(json!({ "skills": summary }).to_string())
            }
            names::SKILL_RELOAD => {
                if let Some(dir) = args.get("dir").and_then(|v| v.as_str()) {
                    self.skills.set_dir(dir).await;
                }
                match self.skills.reload().await {
                    Ok(n) => ToolOutcome::ok(json!({ "loaded": n }).to_string()),
                    Err(e) => ToolOutcome::err(format!("skill reload failed: {e:#}")),
                }
            }
            names::A2A_LIST => {
                let names_list = self.a2a.agent_names().await;
                ToolOutcome::ok(json!({ "agents": names_list }).to_string())
            }
            names::A2A_RELOAD => {
                let desired = if let Some(agents) = args.get("agents") {
                    parse_agents(agents)
                } else {
                    self.a2a_desired.lock().await.clone()
                };
                match self.a2a.reload(&desired).await {
                    Ok(r) => {
                        *self.a2a_desired.lock().await = desired;
                        ToolOutcome::ok(serde_json::to_string(&r).unwrap_or_default())
                    }
                    Err(e) => ToolOutcome::err(format!("a2a reload failed: {e:#}")),
                }
            }
            other if other.starts_with("mcp_") && other.contains("__") => {
                let args = args.clone();
                match self.mcp.call_qualified(other, args).await {
                    Ok(r) => ToolOutcome::ok(r.text),
                    Err(e) => ToolOutcome::err(format!("MCP tool call failed: {e:#}")),
                }
            }
            other
                if other.starts_with(names::A2A_PREFIX)
                    && other != names::A2A_LIST
                    && other != names::A2A_RELOAD =>
            {
                let agent = other
                    .strip_prefix(names::A2A_PREFIX)
                    .unwrap_or("")
                    .to_string();
                let Some(prompt) = args.get("prompt").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'prompt'");
                };
                match self.a2a.send(&agent, prompt).await {
                    Ok(r) => {
                        let mut msg = r.text;
                        if let Some(state) = r.state {
                            msg = format!("{msg}\n[task state: {state}]");
                        }
                        ToolOutcome::ok(msg)
                    }
                    Err(e) => ToolOutcome::err(format!("A2A agent call failed: {e:#}")),
                }
            }
            other
                if other.starts_with(names::SKILL_PREFIX)
                    && other != names::SKILL_LIST
                    && other != names::SKILL_RELOAD =>
            {
                let skill = other
                    .strip_prefix(names::SKILL_PREFIX)
                    .unwrap_or("")
                    .to_string();
                match self.skills.invoke(&skill, args).await {
                    Ok(body) => ToolOutcome::ok(body),
                    Err(e) => ToolOutcome::err(format!("skill invoke failed: {e:#}")),
                }
            }
            names::MODEL_LIST => {
                let models = self.catalog.list().await;
                ToolOutcome::ok(crate::model_catalog::catalog_json(&models).to_string())
            }
            names::MODEL_PICK => {
                let opts = crate::model_catalog::PickOptions {
                    strategy: args
                        .get("strategy")
                        .and_then(|v| v.as_str())
                        .map(parse_strategy_arg)
                        .unwrap_or(self.model_opts.strategy),
                    min_score: args
                        .get("min_score")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(self.model_opts.min_score),
                    max_cost_per_token: args
                        .get("max_cost_per_token")
                        .and_then(|v| v.as_f64())
                        .or(self.model_opts.max_cost_per_token),
                    min_context_window: args
                        .get("min_context_window")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as u32)
                        .or(self.model_opts.min_context_window),
                };
                let all = self.catalog.list().await;
                let picked = self.catalog.pick(&opts).await;
                let explanation = crate::model_catalog::explain_pick(picked.as_ref(), &all, &opts);
                let id = picked.as_ref().map(|m| m.id.clone()).unwrap_or_default();
                ToolOutcome::ok(json!({ "model": id, "explanation": explanation }).to_string())
            }
            names::MEM_SET => {
                let Some(key) = args.get("key").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'key'");
                };
                let Some(value) = args.get("value").and_then(|v| v.as_str()) else {
                    return ToolOutcome::err("missing 'value'");
                };
                match self.memory.set(sender_id, key, value).await {
                    Ok(()) => ToolOutcome::ok(json!({ "stored": true, "key": key }).to_string()),
                    Err(e) => ToolOutcome::err(format!("memory set failed: {e:#}")),
                }
            }
            names::MEM_RECALL => match self.memory.recall(sender_id).await {
                Ok(mem) => {
                    if let Some(key) = args.get("key").and_then(|v| v.as_str()) {
                        let value = mem.facts.get(key).cloned();
                        ToolOutcome::ok(json!({ "key": key, "value": value }).to_string())
                    } else {
                        ToolOutcome::ok(serde_json::to_string(&mem).unwrap_or_default())
                    }
                }
                Err(e) => ToolOutcome::err(format!("memory recall failed: {e:#}")),
            },
            other => ToolOutcome::err(format!("unknown tool: {other}")),
        }
    }
}

/// Wrap a definition into an OpenAI Tool.
fn tool_def(name: &str, desc: &str, schema: Value) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: name.to_string(),
            description: Some(desc.to_string()),
            parameters: Some(schema),
            strict: None,
        },
    })
}

fn summarize_call(tool: &str, args: &Value) -> String {
    match tool {
        names::SCHED_ADD => format!(
            "schedule job '{}' (cron {})",
            args.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
            args.get("cron").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        names::SCHED_REMOVE => format!(
            "remove job {}",
            args.get("id").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        names::MCP_RELOAD => "reload remote MCP servers".to_string(),
        _ => format!("call {tool}"),
    }
}

fn parse_servers(servers: &Value) -> Vec<McpServerConfig> {
    let Some(arr) = servers.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| {
            let name = s.get("name")?.as_str()?.to_string();
            let url = s.get("url")?.as_str()?.to_string();
            let token = s.get("token").and_then(|v| v.as_str()).map(String::from);
            let timeout_secs = s.get("timeout_secs").and_then(|v| v.as_u64()).unwrap_or(10);
            Some(McpServerConfig {
                name,
                url,
                token,
                timeout_secs,
            })
        })
        .collect()
}

fn parse_strategy_arg(s: &str) -> crate::model_catalog::Strategy {
    use crate::model_catalog::Strategy;
    match s.trim().to_ascii_lowercase().as_str() {
        "best_score_under_budget" => Strategy::BestScoreUnderBudget,
        "cheapest_above_floor" => Strategy::CheapestAboveFloor,
        "best_value" => Strategy::BestValue,
        _ => Strategy::BestScore,
    }
}

fn parse_agents(agents: &Value) -> Vec<A2aAgentConfig> {
    let Some(arr) = agents.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| {
            let name = s.get("name")?.as_str()?.to_string();
            let url = s.get("url")?.as_str()?.to_string();
            let token = s.get("token").and_then(|v| v.as_str()).map(String::from);
            let timeout_secs = s.get("timeout_secs").and_then(|v| v.as_u64()).unwrap_or(30);
            Some(A2aAgentConfig {
                name,
                url,
                token,
                timeout_secs,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_servers_ok() {
        let v = json!([
            { "name": "weather", "url": "http://x/mcp", "timeout_secs": 5 },
            { "name": "empty", "url": "http://y/mcp" },
            { "url": "http://no-name/mcp" },
        ]);
        let s = parse_servers(&v);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].name, "weather");
        assert_eq!(s[1].timeout_secs, 10);
    }
}
