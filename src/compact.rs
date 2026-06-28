//! Auto-compaction of conversation history.
//!
//! Conversation histories grow unboundedly as a sender talks to the agent;
//! every prior turn is replayed into the model's context on each request.
//! Beyond a point this wastes tokens (cost + latency) and eventually overruns
//! the context window. [`Compactor`] summarizes the older portion of a
//! conversation into a single compact summary, keeping only the most recent
//! `keep_recent` turns verbatim — so the model retains the gist of the past
//! without paying for every word.
//!
//! The compactor uses the same OpenAI-compatible client the chat agent uses.
//! It is invoked automatically by the agent when the per-sender history
//! crosses [`CompactConfig::threshold_messages`].

use anyhow::Result;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use tracing::info;

use crate::agent::{Stored, TokenUsage};
use crate::config::{CompactConfig, LlmConfig};

/// Compactor: summarizes older conversation turns.
#[derive(Clone)]
pub struct Compactor {
    client: Client<OpenAIConfig>,
    model: String,
    cfg: CompactConfig,
}

impl Compactor {
    pub fn new(llm: &LlmConfig, cfg: CompactConfig) -> Self {
        let api_key = llm.api_key.clone().unwrap_or_default();
        let oc = OpenAIConfig::new()
            .with_api_base(llm.base_url.clone())
            .with_api_key(api_key);
        Self {
            client: Client::with_config(oc),
            model: llm.model.clone(),
            cfg,
        }
    }

    pub fn config(&self) -> &CompactConfig {
        &self.cfg
    }

    /// Whether a conversation with `len` stored messages should be compacted.
    pub fn should_compact(&self, len: usize) -> bool {
        self.cfg.enabled
            && self.cfg.threshold_messages > 0
            && len >= self.cfg.threshold_messages
            && self.cfg.keep_recent < len
    }

    /// Summarize the older portion of `messages` (everything except the last
    /// `keep_recent` entries) into a single paragraph. Returns the summary.
    pub async fn summarize(&self, messages: &[Stored]) -> Result<String> {
        let drop = messages.len().saturating_sub(self.cfg.keep_recent);
        let old = &messages[..drop];
        if old.is_empty() {
            return Ok(String::new());
        }

        // Render the old turns as a transcript the summarizer can read.
        let mut transcript = String::new();
        for m in old {
            match m {
                Stored::User(t) => {
                    transcript.push_str("User: ");
                    transcript.push_str(t);
                }
                Stored::Assistant(t) => {
                    transcript.push_str("Assistant: ");
                    transcript.push_str(t);
                }
            }
            transcript.push('\n');
        }

        let req = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(format!(
                        "You are a conversation summarizer. {} \
                         Reply with only the summary — no preamble.",
                        self.cfg.prompt
                    ))
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(transcript)
                    .build()?
                    .into(),
            ])
            .max_tokens(512u32)
            .temperature(0.0)
            .build()?;

        let resp = self
            .client
            .chat()
            .create(req)
            .await
            .map_err(|e| anyhow::anyhow!("compaction request failed: {e}"))?;
        let summary = resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default()
            .trim()
            .to_string();
        let _ = TokenUsage::default(); // (usage not surfaced here)
        info!(chars = summary.len(), "conversation compacted");
        Ok(summary)
    }

    /// Compact `messages` in place: replace the older portion with a single
    /// summary entry (encoded as an assistant message prefixed with a marker),
    /// keeping the most recent `keep_recent` entries verbatim. Returns whether
    /// a compaction actually happened.
    pub async fn compact(&self, messages: &mut Vec<Stored>) -> Result<bool> {
        if !self.should_compact(messages.len()) {
            return Ok(false);
        }
        let keep = self.cfg.keep_recent;
        let summary = self.summarize(messages).await?;
        if summary.is_empty() {
            return Ok(false);
        }
        let summary_entry = Stored::Assistant(format!("[summary of earlier conversation]\n{summary}"));
        let tail: Vec<Stored> = messages
            .iter()
            .skip(messages.len() - keep)
            .cloned()
            .collect();
        messages.clear();
        messages.push(summary_entry);
        messages.extend(tail);
        Ok(true)
    }
}
