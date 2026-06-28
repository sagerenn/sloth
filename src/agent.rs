//! Chat-completion agent backed by an OpenAI-compatible endpoint.
//!
//! Maintains a short per-sender conversation history (capped to the most
//! recent `history.max_messages` turns) and produces a reply for an inbound
//! user message. Uses the `async-openai` SDK configured with a custom base URL
//! pointing at the local gateway at `http://172.17.0.1:8317/v1`.
//!
//! Note on auth: the local gateway does not require an API key and in fact
//! rejects unknown bearer tokens. `async-openai` always sends an
//! `Authorization: Bearer {key}` header; with an empty key it sends
//! `Bearer ` (empty), which the gateway accepts. So we configure an empty key
//! when none is provided rather than trying to suppress the header.

use std::sync::Arc;

use anyhow::Result;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, CompletionUsage,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};
use dashmap::DashMap;
use tokio::sync::Mutex;

use crate::compact::Compactor;
use crate::config::LlmConfig;
use crate::memory::MemoryStore;
use crate::tools::ToolRouter;

/// A single stored message in a conversation.
#[derive(Debug, Clone)]
pub enum Stored {
    User(String),
    Assistant(String),
}

/// Per-sender conversation state.
#[derive(Debug, Default)]
struct Conversation {
    messages: Vec<Stored>,
}

impl Conversation {
    fn push(&mut self, m: Stored, cap: usize) {
        self.messages.push(m);
        // Keep only the most recent `cap` messages (each entry is one role).
        if self.messages.len() > cap {
            let drop = self.messages.len() - cap;
            self.messages.drain(..drop);
        }
    }
}

/// The agent. Cheaply cloneable (history map is shared).
#[derive(Clone)]
pub struct ChatAgent {
    inner: Arc<Inner>,
}

struct Inner {
    client: Client<OpenAIConfig>,
    model: String,
    system_prompt: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    history_cap: usize,
    conversations: DashMap<String, Mutex<Conversation>>,
    /// Optional auto-compactor; when present, history is summarized past a
    /// threshold so context stays bounded.
    compactor: Option<Compactor>,
    /// Optional persistent memory; when present, per-sender recalled facts are
    /// injected into the system prompt.
    memory: Option<MemoryStore>,
    /// Whether to inject recalled memory into the prompt.
    inject_memory: bool,
}

/// Token usage reported by the backend (independent of the SDK's own types so
/// the rest of the crate doesn't depend on `async-openai` internals).
#[derive(Debug, Default, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<&CompletionUsage> for TokenUsage {
    fn from(u: &CompletionUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

/// Result of generating a reply.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Reply {
    /// Generated assistant text. Empty if nothing was produced.
    pub text: String,
    /// Token usage, if reported by the backend.
    pub usage: Option<TokenUsage>,
    /// The model that produced the response.
    pub model: String,
}

impl ChatAgent {
    /// Build the agent from config.
    pub fn new(cfg: &LlmConfig, history_cap: usize) -> Result<Self> {
        Self::with_compactor(cfg, history_cap, None)
    }

    /// Build the agent with an optional auto-compactor.
    pub fn with_compactor(
        cfg: &LlmConfig,
        history_cap: usize,
        compactor: Option<Compactor>,
    ) -> Result<Self> {
        Self::with_compactor_and_memory(cfg, history_cap, compactor, None, false)
    }

    /// Build the agent with an optional compactor and persistent memory store.
    pub fn with_compactor_and_memory(
        cfg: &LlmConfig,
        history_cap: usize,
        compactor: Option<Compactor>,
        memory: Option<MemoryStore>,
        inject_memory: bool,
    ) -> Result<Self> {
        // Empty key (when none provided) → `Bearer ` header, which the gateway
        // accepts. A real key is forwarded as-is.
        let api_key = cfg.api_key.clone().unwrap_or_default();
        let oc = OpenAIConfig::new()
            .with_api_base(cfg.base_url.clone())
            .with_api_key(api_key);
        let client = Client::with_config(oc);

        Ok(Self {
            inner: Arc::new(Inner {
                client,
                model: cfg.model.clone(),
                system_prompt: cfg.system_prompt.clone(),
                temperature: cfg.temperature,
                max_tokens: cfg.max_tokens,
                history_cap,
                conversations: DashMap::new(),
                compactor,
                memory,
                inject_memory,
            }),
        })
    }

    /// Generate a reply for `user_text` from `sender_id`.
    ///
    /// Runs inside a span `chat.complete` carrying sender + message length +
    /// token usage so it surfaces in tracing output.
    pub async fn reply(&self, sender_id: &str, user_text: &str) -> Result<Reply> {
        let span = tracing::info_span!(
            "chat.complete",
            sender = %sender_id,
            inbound_chars = user_text.len(),
            model = %self.inner.model,
        );
        let _enter = span.enter();

        let request = self.build_request(sender_id, user_text).await?;
        let response = self
            .inner
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| anyhow::anyhow!("chat completion request failed: {e}"))?;

        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default()
            .trim()
            .to_string();

        let usage = response.usage.as_ref().map(TokenUsage::from);

        tracing::info!(
            reply_chars = text.len(),
            prompt_tokens = usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
            completion_tokens = usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            total_tokens = usage.as_ref().map(|u| u.total_tokens).unwrap_or(0),
            finish_reason = ?response.choices.first().map(|c| c.finish_reason),
            "chat completion done"
        );

        // Persist the exchange into history.
        self.record_turn(sender_id, user_text, &text).await;

        Ok(Reply {
            text,
            usage,
            model: response.model.clone(),
        })
    }

    /// Generate a reply with function-calling support.
    ///
    /// Runs an agentic loop: send the conversation + tool definitions → if the
    /// model returns `tool_calls`, execute each via the [`ToolRouter`] (which
    /// applies HITL gating), append the tool results, and re-query — up to
    /// `max_steps` rounds. The final assistant text message is the reply.
    ///
    /// This is the "structured output" path: the model emits structured
    /// `tool_call` objects (name + JSON arguments) that we dispatch
    /// deterministically, rather than free-text.
    pub async fn reply_with_tools(
        &self,
        sender_id: &str,
        user_text: &str,
        router: &ToolRouter,
        max_steps: usize,
    ) -> Result<Reply> {
        let span = tracing::info_span!(
            "chat.complete.tools",
            sender = %sender_id,
            inbound_chars = user_text.len(),
            model = %self.inner.model,
        );
        let _enter = span.enter();

        let tools = router.tool_definitions().await;
        let mut messages = self.history_messages(sender_id).await?;
        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_text)
                .build()?
                .into(),
        );

        let mut total_usage: Option<TokenUsage> = None;
        let mut steps = 0usize;
        let final_text: String;

        loop {
            steps += 1;
            if steps > max_steps {
                tracing::warn!(steps, "tool loop hit max_steps; stopping");
                final_text = "I reached the maximum number of tool steps; stopping here."
                    .to_string();
                break;
            }

            let mut args = CreateChatCompletionRequestArgs::default();
            args.model(self.inner.model.clone()).messages(messages.clone());
            if !tools.is_empty() {
                args.tools(tools.clone());
            }
            if let Some(t) = self.inner.temperature {
                args.temperature(t);
            }
            if let Some(m) = self.inner.max_tokens {
                args.max_tokens(m);
            }
            let request = args.build()?;

            let response = self
                .inner
                .client
                .chat()
                .create(request)
                .await
                .map_err(|e| anyhow::anyhow!("chat completion request failed: {e}"))?;

            if let Some(u) = response.usage.as_ref() {
                let new = TokenUsage::from(u);
                total_usage = Some(match total_usage {
                    Some(prev) => TokenUsage {
                        prompt_tokens: prev.prompt_tokens + new.prompt_tokens,
                        completion_tokens: prev.completion_tokens + new.completion_tokens,
                        total_tokens: prev.total_tokens + new.total_tokens,
                    },
                    None => new,
                });
            }

            let Some(choice) = response.choices.first() else {
                anyhow::bail!("no choices in completion response");
            };
            let msg = &choice.message;
            let tool_calls = msg.tool_calls.clone().unwrap_or_default();

            // If there are no tool calls, the model is done — take its text.
            if tool_calls.is_empty() {
                final_text = msg.content.clone().unwrap_or_default().trim().to_string();
                break;
            }

            // Append the assistant message carrying the tool calls (including
            // any text content the model emitted alongside).
            let mut assistant_args = ChatCompletionRequestAssistantMessageArgs::default();
            if let Some(c) = &msg.content {
                assistant_args.content(c.clone());
            } else {
                assistant_args.content("");
            }
            assistant_args.tool_calls(tool_calls.clone());
            messages.push(assistant_args.build()?.into());

            // Execute each tool call and append its result. Tool calls are an
            // enum; we only handle the `Function` variant (ignore custom).
            for tc in &tool_calls {
                let ChatCompletionMessageToolCalls::Function(call) = tc else {
                    continue;
                };
                let outcome = self.run_tool_call(call, router, sender_id).await;
                let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                    .content(outcome.content)
                    .tool_call_id(call.id.clone())
                    .build()?
                    .into();
                messages.push(tool_msg);
            }
        }

        tracing::info!(
            reply_chars = final_text.len(),
            steps,
            prompt_tokens = total_usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
            completion_tokens = total_usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            total_tokens = total_usage.as_ref().map(|u| u.total_tokens).unwrap_or(0),
            "tool-augmented chat completion done"
        );

        // Persist the user turn + final reply into history (intermediate tool
        // rounds are not stored — they're transient scaffolding).
        self.record_turn(sender_id, user_text, &final_text).await;

        Ok(Reply {
            text: final_text,
            usage: total_usage,
            model: self.inner.model.clone(),
        })
    }

    /// Execute a single tool call, parsing arguments defensively.
    async fn run_tool_call(
        &self,
        tc: &ChatCompletionMessageToolCall,
        router: &ToolRouter,
        sender_id: &str,
    ) -> crate::tools::ToolOutcome {
        let name = &tc.function.name;
        let args: serde_json::Value = if tc.function.arguments.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null)
        };
        tracing::info!(%name, args = ?args, "executing tool call");
        router.execute(name, &args, sender_id).await
    }

    /// Build the system prompt + prior-history messages (no current turn).
    async fn history_messages(
        &self,
        sender_id: &str,
    ) -> Result<Vec<async_openai::types::chat::ChatCompletionRequestMessage>> {
        let mut messages = Vec::new();
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt_for(sender_id).await?)
                .build()?
                .into(),
        );
        let history = if let Some(entry) = self.inner.conversations.get(sender_id) {
            let conv = entry.value();
            let guard = conv.lock().await;
            guard.messages.clone()
        } else {
            Vec::new()
        };
        for m in history {
            match m {
                Stored::User(t) => messages.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(t)
                        .build()?
                        .into(),
                ),
                Stored::Assistant(t) => messages.push(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(t)
                        .build()?
                        .into(),
                ),
            }
        }
        Ok(messages)
    }

    async fn build_request(
        &self,
        sender_id: &str,
        user_text: &str,
    ) -> Result<CreateChatCompletionRequest> {
        let mut messages = Vec::new();
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt_for(sender_id).await?)
                .build()?
                .into(),
        );

        // Snapshot prior history for this sender under its lock, then drop the
        // DashMap shard guard before building the rest (builders are sync).
        let history = if let Some(entry) = self.inner.conversations.get(sender_id) {
            let conv = entry.value();
            let guard = conv.lock().await;
            guard.messages.clone()
        } else {
            Vec::new()
        };
        for m in history {
            match m {
                Stored::User(t) => messages.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(t)
                        .build()?
                        .into(),
                ),
                Stored::Assistant(t) => messages.push(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(t)
                        .build()?
                        .into(),
                ),
            }
        }

        // Current user turn.
        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_text)
                .build()?
                .into(),
        );

        let mut args = CreateChatCompletionRequestArgs::default();
        args.model(self.inner.model.clone()).messages(messages);
        if let Some(t) = self.inner.temperature {
            args.temperature(t);
        }
        if let Some(m) = self.inner.max_tokens {
            args.max_tokens(m);
        }
        Ok(args.build()?)
    }

    /// The effective system prompt for `sender`: the base prompt +, when
    /// memory injection is on, the recalled facts for that sender.
    async fn system_prompt_for(&self, sender_id: &str) -> Result<String> {
        let base = self.inner.system_prompt.clone();
        if let Some(mem) = &self.inner.memory
            && self.inner.inject_memory
            && let Ok(m) = mem.recall(sender_id).await
            && let Some(snippet) = m.to_prompt_snippet()
        {
            return Ok(format!("{base}\n\n{snippet}"));
        }
        Ok(base)
    }

    /// Public accessor to the effective system prompt (for tests/inspection).
    pub async fn system_prompt_for_pub(&self, sender_id: &str) -> Result<String> {
        self.system_prompt_for(sender_id).await
    }

    async fn record_turn(&self, sender_id: &str, user_text: &str, reply: &str) {
        // `entry().or_default()` inserts a `Mutex<Conversation>`; lock and push.
        let entry = self
            .inner
            .conversations
            .entry(sender_id.to_string())
            .or_default();
        let mut guard = entry.value().lock().await;
        let cap = self.inner.history_cap;
        guard.push(Stored::User(user_text.to_string()), cap);
        guard.push(Stored::Assistant(reply.to_string()), cap);

        // Auto-compact when the history crosses the threshold. Compaction
        // itself calls the LLM, so we run it after the turn is recorded and
        // hold the per-sender lock to avoid racing other turns for this sender.
        if let Some(compactor) = &self.inner.compactor {
            let pre = guard.messages.len();
            if compactor.should_compact(pre) {
                match compactor.compact(&mut guard.messages).await {
                    Ok(true) => {
                        tracing::info!(
                            sender = %sender_id,
                            before = pre,
                            after = guard.messages.len(),
                            "history auto-compacted"
                        );
                    }
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(error = %e, sender = %sender_id, "auto-compact failed; keeping full history");
                    }
                }
            }
        }
    }
}
