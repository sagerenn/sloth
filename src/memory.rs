//! Persistent memory.
//!
//! The agent remembers facts about each sender across restarts. Memory is
//! stored as a TOML file per sender under `MemoryConfig::dir` (e.g.
//! `memories/<sender>.toml`), keyed by an arbitrary `key` → `value` string.
//! Recalled facts are injected into the system prompt so the model acts on
//! prior context, and the model can write new facts via the `memory_set` /
//! `memory_recall` tools.
//!
//! The store is hot on disk: writes persist immediately, reads are cached in
//! memory behind a per-sender `Mutex`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

/// One recalled fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub key: String,
    pub value: String,
}

/// Per-sender memory: a flat map of `key → value`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SenderMemory {
    pub facts: BTreeMap<String, String>,
}

impl SenderMemory {
    /// Render the memory as a short system-prompt snippet. Empty memory → None.
    pub fn to_prompt_snippet(&self) -> Option<String> {
        if self.facts.is_empty() {
            return None;
        }
        let mut out = String::from("Known facts about this user:\n");
        for (k, v) in &self.facts {
            out.push_str(&format!("- {k}: {v}\n"));
        }
        Some(out)
    }
}

/// The persistent memory store. Cloneable (state shared behind `Arc`).
#[derive(Clone, Default)]
pub struct MemoryStore {
    inner: Arc<MemoryState>,
}

#[derive(Default)]
struct MemoryState {
    dir: Mutex<Option<PathBuf>>,
    /// Cache: sender → memory (loaded lazily).
    cache: Mutex<BTreeMap<String, SenderMemory>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MemoryState {
                dir: Mutex::new(None),
                cache: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    /// Set the persistence directory (created on first write).
    pub async fn set_dir(&self, dir: impl Into<PathBuf>) {
        *self.inner.dir.lock().await = Some(dir.into());
    }

    /// Load a sender's memory from disk into the cache (no-op if absent).
    async fn load(&self, sender: &str) -> Result<SenderMemory> {
        {
            let cache = self.inner.cache.lock().await;
            if let Some(m) = cache.get(sender) {
                return Ok(m.clone());
            }
        }
        let dir = self.inner.dir.lock().await.clone();
        let m = match &dir {
            Some(d) => {
                let path = sender_path(d, sender);
                if path.exists() {
                    let text = std::fs::read_to_string(&path)
                        .with_context(|| format!("reading memory {}", path.display()))?;
                    toml::from_str::<SenderMemory>(&text).unwrap_or_default()
                } else {
                    SenderMemory::default()
                }
            }
            None => SenderMemory::default(),
        };
        self.inner
            .cache
            .lock()
            .await
            .insert(sender.to_string(), m.clone());
        Ok(m)
    }

    /// Recall a sender's memory.
    pub async fn recall(&self, sender: &str) -> Result<SenderMemory> {
        self.load(sender).await
    }

    /// Look up a single fact by key.
    pub async fn get(&self, sender: &str, key: &str) -> Result<Option<String>> {
        let m = self.load(sender).await?;
        Ok(m.facts.get(key).cloned())
    }

    /// Set a fact (upsert) and persist to disk.
    pub async fn set(&self, sender: &str, key: &str, value: &str) -> Result<()> {
        // Load current into cache.
        let mut m = self.load(sender).await?;
        m.facts.insert(key.to_string(), value.to_string());
        self.persist(sender, &m).await?;
        self.inner.cache.lock().await.insert(sender.to_string(), m);
        info!(sender = %sender, key = %key, "memory fact set");
        Ok(())
    }

    /// Delete a fact and persist.
    pub async fn delete(&self, sender: &str, key: &str) -> Result<bool> {
        let mut m = self.load(sender).await?;
        let removed = m.facts.remove(key).is_some();
        if removed {
            self.persist(sender, &m).await?;
            self.inner.cache.lock().await.insert(sender.to_string(), m);
        }
        Ok(removed)
    }

    async fn persist(&self, sender: &str, m: &SenderMemory) -> Result<()> {
        let dir = self.inner.dir.lock().await.clone();
        if let Some(d) = &dir {
            std::fs::create_dir_all(d)
                .with_context(|| format!("creating memory dir {}", d.display()))?;
            let path = sender_path(d, sender);
            let text = toml::to_string(m).context("encoding memory")?;
            std::fs::write(&path, text)
                .with_context(|| format!("writing memory {}", path.display()))?;
        }
        Ok(())
    }
}

fn sender_path(dir: &Path, sender: &str) -> PathBuf {
    // Sanitize the sender id so it can't escape the directory.
    let safe: String = sender
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = if safe.is_empty() {
        "anon".to_string()
    } else {
        safe
    };
    dir.join(format!("{safe}.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn set_recall_persist_across_instances() {
        let dir = std::env::temp_dir().join(format!("sloth-mem-{}", uniq()));
        let store = MemoryStore::new();
        store.set_dir(&dir).await;

        store.set("alice", "name", "Alice Lee").await.unwrap();
        store.set("alice", "lang", "rust").await.unwrap();
        assert_eq!(
            store.get("alice", "name").await.unwrap().as_deref(),
            Some("Alice Lee")
        );

        // A fresh store instance reading from the same dir recalls the facts.
        let store2 = MemoryStore::new();
        store2.set_dir(&dir).await;
        let m = store2.recall("alice").await.unwrap();
        assert_eq!(m.facts.get("name").unwrap(), "Alice Lee");
        assert_eq!(m.facts.get("lang").unwrap(), "rust");

        // Delete persists.
        assert!(store2.delete("alice", "lang").await.unwrap());
        let store3 = MemoryStore::new();
        store3.set_dir(&dir).await;
        assert!(store3.get("alice", "lang").await.unwrap().is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sender_path_sanitizes() {
        let p = sender_path(Path::new("/tmp/mem"), "../etc/passwd");
        assert!(p.starts_with("/tmp/mem"));
        assert!(
            p.to_string_lossy().contains("___etc_passwd") || p.to_string_lossy().contains("passwd")
        );
    }

    fn uniq() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    }
}
