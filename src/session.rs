//! Session management.
//!
//! A session is a named conversation context with its own history and an
//! optional **workspace** — a working directory the agent operates in for
//! that session (e.g. for file-relative tool calls). Sessions can be created,
//! listed, switched, and deleted; the active session per sender is tracked so
//! a user can resume a context or jump between them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Per-session metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique id (also the history key).
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Working directory for the session (workspace). None = inherit cwd.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// Epoch-second creation time.
    pub created_at: i64,
}

/// Session manager. Cloneable; state shared behind an `Arc`.
#[derive(Clone)]
pub struct SessionManager {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    sessions: HashMap<String, Session>,
    /// Active session id per sender.
    active: HashMap<String, String>,
    /// Default store dir for persistence (reserved for future on-disk state).
    #[allow(dead_code)]
    store_dir: PathBuf,
}

impl SessionManager {
    /// Create a manager seeded with a default session.
    pub fn new(default_session: &str, store_dir: impl Into<PathBuf>) -> Self {
        let now = crate::scheduler::wall_secs();
        let mut sessions = HashMap::new();
        sessions.insert(
            default_session.to_string(),
            Session {
                id: default_session.to_string(),
                label: "Default".to_string(),
                workspace: None,
                created_at: now,
            },
        );
        Self {
            inner: Arc::new(Mutex::new(Inner {
                sessions,
                active: HashMap::new(),
                store_dir: store_dir.into(),
            })),
        }
    }

    /// Create a new session. Returns the session id. If `id` is empty, one is
    /// generated.
    pub async fn create(&self, id: Option<String>, label: String) -> Result<Session> {
        let mut g = self.inner.lock().await;
        let id = match id {
            Some(s) if !s.is_empty() => s,
            _ => format!("sess-{}", Uuid::new_v4().simple()),
        };
        if g.sessions.contains_key(&id) {
            anyhow::bail!("session {id} already exists");
        }
        let session = Session {
            id: id.clone(),
            label,
            workspace: None,
            created_at: crate::scheduler::wall_secs(),
        };
        g.sessions.insert(id, session.clone());
        Ok(session)
    }

    /// Set a session's workspace directory. Creates the directory if missing.
    pub async fn set_workspace(&self, id: &str, workspace: PathBuf) -> Result<Session> {
        let mut g = self.inner.lock().await;
        let session = g
            .sessions
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("session {id} not found"))?;
        if !workspace.exists() {
            std::fs::create_dir_all(&workspace)
                .with_context(|| format!("creating workspace {}", workspace.display()))?;
        }
        session.workspace = Some(workspace.clone());
        Ok(session.clone())
    }

    /// Switch the sender's active session. Returns the activated session.
    pub async fn switch(&self, sender: &str, id: &str) -> Result<Session> {
        let mut g = self.inner.lock().await;
        let session = g
            .sessions
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("session {id} not found"))?;
        g.active.insert(sender.to_string(), id.to_string());
        Ok(session)
    }

    /// Active session id for a sender, defaulting to the first/default session.
    pub async fn active_id(&self, sender: &str) -> String {
        let g = self.inner.lock().await;
        g.active
            .get(sender)
            .cloned()
            .unwrap_or_else(|| g.sessions.keys().next().cloned().unwrap_or_default())
    }

    /// Resolve a sender's active session.
    pub async fn active(&self, sender: &str) -> Option<Session> {
        let g = self.inner.lock().await;
        let id = g
            .active
            .get(sender)
            .cloned()
            .or_else(|| g.sessions.keys().next().cloned())?;
        g.sessions.get(&id).cloned()
    }

    /// List all sessions.
    pub async fn list(&self) -> Vec<Session> {
        let g = self.inner.lock().await;
        let mut v: Vec<_> = g.sessions.values().cloned().collect();
        v.sort_by_key(|s| s.created_at);
        v
    }

    /// Delete a session (the default session is protected).
    pub async fn delete(&self, id: &str, default_session: &str) -> Result<()> {
        let mut g = self.inner.lock().await;
        if id == default_session {
            anyhow::bail!("cannot delete the default session");
        }
        if g.sessions.remove(id).is_none() {
            anyhow::bail!("session {id} not found");
        }
        // Re-point any senders that were active in the deleted session.
        for v in g.active.values_mut() {
            if v == id {
                *v = default_session.to_string();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_switch_workspace_delete() {
        let dir = std::env::temp_dir().join(format!("sloth-sess-{}", uuid::Uuid::new_v4()));
        let sm = SessionManager::new("default", dir);
        let s = sm.create(Some("work".into()), "Work".into()).await.unwrap();
        assert_eq!(s.id, "work");
        let active = sm.switch("alice", "work").await.unwrap();
        assert_eq!(active.id, "work");
        assert_eq!(sm.active_id("alice").await, "work");

        let ws = std::env::temp_dir().join(format!("sloth-ws-{}", uuid::Uuid::new_v4()));
        let s = sm.set_workspace("work", ws.clone()).await.unwrap();
        assert_eq!(s.workspace.as_ref(), Some(&ws));
        assert!(ws.exists());

        sm.delete("default", "default").await.unwrap_err();
        sm.delete("work", "default").await.unwrap();
        assert!(sm.list().await.iter().all(|s| s.id != "work"));
    }
}
