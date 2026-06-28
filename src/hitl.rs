//! Human-in-the-Loop (HITL) confirmation gating.
//!
//! When a tool call is flagged for confirmation, the agent asks the user (over
//! the bridge) for approval before executing. This module provides the
//! in-process plumbing: a request channel + a decision channel keyed by a
//! pending id, a timeout fallback, and a [`HitlGate`] that decides whether a
//! given tool needs confirmation.
//!
//! The runtime wires the user-facing side: it receives a pending confirmation,
//! sends a question to the bridge, parses the reply, and calls back into
//! [`HitlBroker::resolve`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;

use crate::config::HitlConfig;

/// A pending tool call awaiting human approval.
#[derive(Debug, Clone, Serialize)]
pub struct PendingConfirmation {
    /// Unique id for this confirmation.
    pub id: String,
    /// Fully-qualified tool name (e.g. `scheduler_add_job`, `mcp_weather__forecast`).
    pub tool: String,
    /// Human-readable summary of what the tool will do.
    pub summary: String,
    /// The session (and sender) the request originated from.
    pub session_id: String,
    pub sender_id: String,
}

/// A human decision on a pending confirmation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Proceed with the tool call.
    Approve,
    /// Abort the tool call.
    Deny,
}

/// Decision outcome for a tool call, including the timeout case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Approved,
    Denied,
    /// No human response within the configured timeout; treated as a denial.
    TimedOut,
}

/// Broker for pending confirmations. Cloneable; state shared behind an `Arc`.
#[derive(Clone)]
pub struct HitlBroker {
    inner: Arc<Mutex<Inner>>,
    pending_tx: Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<PendingConfirmation>>>>,
    cfg: HitlConfig,
}

#[derive(Default)]
struct Inner {
    pending: HashMap<String, oneshot::Sender<Outcome>>,
}

impl HitlBroker {
    pub fn new(cfg: HitlConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::default())),
            pending_tx: Arc::new(tokio::sync::Mutex::new(None)),
            cfg,
        }
    }

    /// Subscribe to pending-confirmations. The runtime drains this to surface
    /// requests to the human over the bridge and call [`Self::resolve`].
    pub fn pending_channel(&self) -> tokio::sync::mpsc::UnboundedReceiver<PendingConfirmation> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        // Replace any prior sender.
        // We can't await here easily from a non-async caller, so use try_lock.
        if let Ok(mut g) = self.pending_tx.try_lock() {
            *g = Some(tx);
        }
        rx
    }

    /// Async variant of `pending_channel` (avoids try_lock edge cases).
    pub async fn pending_channel_async(&self) -> tokio::sync::mpsc::UnboundedReceiver<PendingConfirmation> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        *self.pending_tx.lock().await = Some(tx);
        rx
    }

    /// Publish a pending confirmation to the runtime channel (if subscribed).
    pub async fn publish(&self, p: PendingConfirmation) {
        let g = self.pending_tx.lock().await;
        if let Some(tx) = g.as_ref()
            && tx.send(p).is_err() {
                // Receiver dropped — the runtime will not see this; the
                // await_decision timeout will eventually deny it.
            }
    }

    /// Whether HITL is enabled at all.
    pub fn enabled(&self) -> bool {
        self.cfg.enabled
    }

    /// Decides whether a given tool name requires confirmation.
    ///
    /// When `confirm_tools` is empty, every tool is gated (when enabled). When
    /// it lists patterns, only matching tools are gated. Patterns use simple
    /// glob: `*` matches any run of chars, `?` matches one.
    pub fn requires_confirmation(&self, tool: &str) -> bool {
        if !self.cfg.enabled {
            return false;
        }
        if self.cfg.confirm_tools.is_empty() {
            return true;
        }
        self.cfg
            .confirm_tools
            .iter()
            .any(|pat| glob_match(pat, tool))
    }

    /// Register a pending confirmation and return a receiver for its outcome.
    ///
    /// The caller is expected to surface the [`PendingConfirmation`] to the
    /// human (via the bridge) and call [`Self::resolve`] when they answer.
    pub async fn register(&self, p: PendingConfirmation) -> oneshot::Receiver<Outcome> {
        let (tx, rx) = oneshot::channel();
        let mut g = self.inner.lock().await;
        g.pending.insert(p.id.clone(), tx);
        rx
    }

    /// Await a decision, applying the configured timeout as an auto-deny.
    pub async fn await_decision(
        &self,
        rx: oneshot::Receiver<Outcome>,
    ) -> Outcome {
        let to = Duration::from_secs(self.cfg.timeout_secs.max(1));
        match tokio::time::timeout(to, rx).await {
            Ok(Ok(o)) => o,
            Ok(Err(_)) => Outcome::Denied, // sender dropped
            Err(_) => Outcome::TimedOut,
        }
    }

    /// Resolve a pending confirmation with the human's decision (or timeout).
    /// Returns false if no pending confirmation with that id exists.
    pub async fn resolve(&self, id: &str, decision: Outcome) -> bool {
        let mut g = self.inner.lock().await;
        if let Some(tx) = g.pending.remove(id) {
            let _ = tx.send(decision);
            true
        } else {
            false
        }
    }

    /// Build a new pending confirmation with a fresh id.
    pub fn new_pending(
        &self,
        tool: &str,
        summary: &str,
        session_id: &str,
        sender_id: &str,
    ) -> PendingConfirmation {
        PendingConfirmation {
            id: format!("hitl-{}", Uuid::new_v4().simple()),
            tool: tool.to_string(),
            summary: summary.to_string(),
            session_id: session_id.to_string(),
            sender_id: sender_id.to_string(),
        }
    }
}

/// Minimal glob matcher: `*` = any run, `?` = one char. Case-sensitive.
fn glob_match(pat: &str, s: &str) -> bool {
    glob_inner(pat.as_bytes(), s.as_bytes())
}

fn glob_inner(pat: &[u8], s: &[u8]) -> bool {
    let (mut pi, mut si) = (0, 0);
    let (mut star_pi, mut star_si): (Option<usize>, usize) = (None, 0);
    while si < s.len() {
        if pi < pat.len() && (pat[pi] == b'?' || pat[pi] == s[si]) {
            pi += 1;
            si += 1;
        } else if pi < pat.len() && pat[pi] == b'*' {
            star_pi = Some(pi);
            star_si = si;
            pi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_si += 1;
            si = star_si;
        } else {
            return false;
        }
    }
    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }
    pi == pat.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_basics() {
        assert!(glob_match("scheduler_*", "scheduler_add_job"));
        assert!(glob_match("mcp_*__*", "mcp_weather__forecast"));
        assert!(!glob_match("scheduler_*", "mcp_weather__forecast"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
    }

    #[tokio::test]
    async fn approval_flow() {
        let cfg = HitlConfig {
            enabled: true,
            timeout_secs: 5,
            confirm_tools: vec![],
        };
        let b = HitlBroker::new(cfg);
        assert!(b.requires_confirmation("scheduler_add_job"));

        let p = b.new_pending("scheduler_add_job", "add daily 9am job", "default", "u1");
        let rx = b.register(p.clone()).await;
        // Simulate the human approving.
        assert!(b.resolve(&p.id, Outcome::Approved).await);
        let outcome = b.await_decision(rx).await;
        assert_eq!(outcome, Outcome::Approved);
    }

    #[tokio::test]
    async fn timeout_denies() {
        let cfg = HitlConfig {
            enabled: true,
            timeout_secs: 1,
            confirm_tools: vec![],
        };
        let b = HitlBroker::new(cfg);
        let p = b.new_pending("x", "y", "default", "u1");
        let rx = b.register(p).await;
        // Never resolve → should time out.
        let outcome = b.await_decision(rx).await;
        assert_eq!(outcome, Outcome::TimedOut);
    }

    #[test]
    fn pattern_gates_only_matching() {
        let cfg = HitlConfig {
            enabled: true,
            timeout_secs: 5,
            confirm_tools: vec!["scheduler_*".to_string(), "mcp_*".to_string()],
        };
        let b = HitlBroker::new(cfg);
        assert!(b.requires_confirmation("scheduler_add_job"));
        assert!(b.requires_confirmation("mcp_weather__forecast"));
        assert!(!b.requires_confirmation("session_list"));
    }
}
