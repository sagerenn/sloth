//! A2A (Agent2Agent) protocol support — backed by the **official `a2a-rs`
//! SDK** (a2aproject/a2a-rs), via the vendored `a2a-client-lf` crate.
//!
//! We use the SDK's `A2AClientFactory` to negotiate a transport from each
//! agent's published Agent Card (`{base_url}/.well-known/agent-card.json`),
//! then `A2AClient::send_text` to dispatch prompts. [`A2aRegistry`] holds
//! multiple named remote agents and supports **hot reload**: changing the
//! configured agent list reconnects/drops entries without restarting.
//!
//! Each remote agent is exposed to the LLM as an invocable tool
//! `a2a_<name>` whose single argument is the prompt text to send.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use a2a_client::A2AClient;
use a2a_client::A2AClientFactory;
use a2a_client::auth::AuthInterceptor;
use a2a_client::client::SendMessageExt;
use a2a_client::middleware::CallInterceptor;
use a2a_client::transport::Transport;
use a2a::{AgentCard, SendMessageResponse, Task};
#[cfg(test)]
use a2a::TaskState;
use serde::Serialize;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::A2aAgentConfig;

/// A handle to one connected remote A2A agent.
struct AgentEntry {
    client: Arc<A2AClient<Box<dyn Transport>>>,
}

impl Drop for AgentEntry {
    fn drop(&mut self) {
        // Best-effort transport cleanup; can't await in Drop, so spawn.
        let client = self.client.clone();
        tokio::spawn(async move {
            let _ = client.destroy().await;
        });
    }
}

/// Extract text from a `Task`'s status message or artifacts.
fn task_text(task: &Task) -> String {
    if let Some(msg) = &task.status.message
        && let Some(t) = msg.text()
    {
        return t.to_string();
    }
    if let Some(arts) = &task.artifacts {
        let mut parts = Vec::new();
        for a in arts {
            for p in &a.parts {
                if let Some(t) = p.as_text() {
                    parts.push(t.to_string());
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }
    format!("[task {} — {:?}, no text]", task.id, task.status.state)
}

/// Result of an A2A send.
#[derive(Debug, Clone, Serialize)]
pub struct A2aResult {
    pub text: String,
    pub state: Option<String>,
}

/// Registry of connected remote A2A agents. Cloneable (state shared).
#[derive(Clone, Default)]
pub struct A2aRegistry {
    inner: Arc<Mutex<RegistryState>>,
}

#[derive(Default)]
struct RegistryState {
    agents: HashMap<String, AgentEntry>,
}

impl A2aRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Connect to one remote A2A agent (fetch its Agent Card, negotiate
    /// transport) and add it to the registry.
    pub async fn add_agent(&self, cfg: &A2aAgentConfig) -> Result<()> {
        let client = connect_a2a(cfg).await?;
        info!(agent = %cfg.name, "A2A agent connected (a2a-rs)");
        let mut g = self.inner.lock().await;
        g.agents.insert(cfg.name.clone(), AgentEntry { client: Arc::new(client) });
        Ok(())
    }

    /// Hot reload: reconcile the live set against `desired`.
    pub async fn reload(&self, desired: &[A2aAgentConfig]) -> Result<ReloadReport> {
        let mut report = ReloadReport::default();

        // Removals.
        let to_remove: Vec<String> = {
            let g = self.inner.lock().await;
            let desired_names: std::collections::HashSet<&str> =
                desired.iter().map(|d| d.name.as_str()).collect();
            g.agents
                .keys()
                .filter(|n| !desired_names.contains(n.as_str()))
                .cloned()
                .collect()
        };
        for name in &to_remove {
            self.inner.lock().await.agents.remove(name);
            info!(agent = %name, "A2A agent removed (hot reload)");
            report.removed.push(name.clone());
        }

        // Additions.
        for d in desired {
            let exists = self.inner.lock().await.agents.contains_key(&d.name);
            if !exists {
                if let Err(e) = self.add_agent(d).await {
                    warn!(agent = %d.name, error = %e, "A2A agent connect failed on reload");
                    report.failed.push((d.name.clone(), format!("{e:#}")));
                } else {
                    report.added.push(d.name.clone());
                }
            }
        }
        Ok(report)
    }

    /// Number of connected agents.
    pub async fn agent_count(&self) -> usize {
        self.inner.lock().await.agents.len()
    }

    /// Names of connected agents.
    pub async fn agent_names(&self) -> Vec<String> {
        self.inner.lock().await.agents.keys().cloned().collect()
    }

    /// Send a prompt to a named agent, returning its textual reply.
    pub async fn send(&self, name: &str, prompt: &str) -> Result<A2aResult> {
        let client = {
            let g = self.inner.lock().await;
            g.agents
                .get(name)
                .map(|e| e.client.clone())
                .ok_or_else(|| anyhow!("unknown A2A agent: {name}"))?
        };
        let resp = client
            .send_text(prompt)
            .await
            .map_err(|e| anyhow!("a2a send_message failed: {e}"))?;
        match resp {
            SendMessageResponse::Task(t) => {
                let state = format!("{:?}", t.status.state);
                Ok(A2aResult { text: task_text(&t), state: Some(state) })
            }
            SendMessageResponse::Message(m) => Ok(A2aResult {
                text: m.text().unwrap_or("").to_string(),
                state: None,
            }),
        }
    }
}

/// Summary of a hot-reload operation.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ReloadReport {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub failed: Vec<(String, String)>,
}

/// Fetch the Agent Card and negotiate a transport using the official SDK.
async fn connect_a2a(cfg: &A2aAgentConfig) -> Result<A2AClient<Box<dyn Transport>>> {
    let factory = build_factory(cfg);
    let card = fetch_card(&cfg.url, cfg.token.as_deref()).await?;
    let client = factory
        .create_from_card(&card)
        .await
        .map_err(|e| anyhow!("a2a transport negotiation failed: {e}"))?;
    Ok(client)
}

/// Build the factory; injects a Bearer auth interceptor when a token is set.
fn build_factory(cfg: &A2aAgentConfig) -> A2AClientFactory {
    let mut builder = A2AClientFactory::builder();
    if let Some(tok) = &cfg.token
        && !tok.is_empty()
    {
        let interceptor: Arc<dyn CallInterceptor> = Arc::new(AuthInterceptor::bearer(tok));
        builder = builder.with_interceptor(interceptor);
    }
    let _ = cfg.timeout_secs;
    builder.build()
}

/// Fetch and parse the Agent Card from `{base_url}/.well-known/agent-card.json`.
async fn fetch_card(base_url: &str, token: Option<&str>) -> Result<AgentCard> {
    let url = format!(
        "{}/.well-known/agent-card.json",
        base_url.trim_end_matches('/')
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("building reqwest client for agent card")?;
    let mut req = client.get(&url);
    if let Some(tok) = token
        && !tok.is_empty()
    {
        req = req.bearer_auth(tok);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("fetching agent card {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("agent card fetch returned HTTP {}", resp.status());
    }
    let card: AgentCard = resp
        .json()
        .await
        .context("parsing agent card JSON")?;
    Ok(card)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_text_extracts_status_message() {
        let task = Task {
            id: "t1".into(),
            context_id: "c1".into(),
            status: a2a::TaskStatus {
                state: TaskState::Completed,
                message: Some(a2a::Message::new(
                    a2a::Role::Agent,
                    vec![a2a::Part::text("hello there")],
                )),
                timestamp: None,
            },
            artifacts: None,
            history: None,
            metadata: None,
        };
        assert_eq!(task_text(&task), "hello there");
    }

    #[test]
    fn task_text_falls_back_to_state() {
        let task = Task {
            id: "t9".into(),
            context_id: "c1".into(),
            status: a2a::TaskStatus {
                state: TaskState::Failed,
                message: None,
                timestamp: None,
            },
            artifacts: None,
            history: None,
            metadata: None,
        };
        let t = task_text(&task);
        assert!(t.contains("t9"));
    }
}
