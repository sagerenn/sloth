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
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CompletionUsage, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs,
};
use dashmap::DashMap;
use tokio::sync::Mutex;

use crate::config::LlmConfig;

/// A single stored message in a conversation.
#[derive(Debug, Clone)]
enum Stored {
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

    async fn build_request(
        &self,
        sender_id: &str,
        user_text: &str,
    ) -> Result<CreateChatCompletionRequest> {
        let mut messages = Vec::new();
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.inner.system_prompt.clone())
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
    }
}
