//! Model catalog with automatic model selection.
//!
//! The catalog is a **collection of YAML files** (one per file, each may list
//! one or more models) describing every model the agent *could* use, with its
//! cost, context capacity, and benchmark scores. The agent picks which model to
//! send a request to automatically — driven by a [`Strategy`] and optional
//! constraints (score floor, max cost, required context window) — rather than
//! hard-coding a single model id.
//!
//! ## Catalog file format
//!
//! Models live under `dir/*.yaml`; each file is either a single model map or a
//! top-level `models:` list. Example:
//!
//! ```yaml
//! models:
//!   - id: glm-4.6
//!     provider: zhipu
//!     context_window: 128000
//!     max_output: 8192
//!     pricing:
//!       prompt_per_1m: 0.5
//!       completion_per_1m: 1.5
//!     scores:
//!       mmlu: 86.5
//!       human_eval: 82.0
//!       average: 84.2
//!   - id: glm-5.2
//!     provider: zhipu
//!     context_window: 128000
//!     max_output: 8192
//!     pricing:
//!       prompt_per_1m: 2.0
//!       completion_per_1m: 8.0
//!     scores:
//!       mmlu: 90.1
//!       human_eval: 88.5
//!       average: 89.3
//! ```
//!
//! `pricing` is per 1M tokens (USD). `scores` is a free-form map; the special
//! `average` key (if present) is the default score used for ranking.
//!
//! The catalog parses with `serde_yaml` and is hot-reloadable via [`Catalog::reload`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Pricing for a model (per 1M tokens, USD).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Pricing {
    pub prompt_per_1m: f64,
    pub completion_per_1m: f64,
}

/// One model entry in the catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelInfo {
    /// Model id used in chat-completion requests.
    pub id: String,
    /// Provider / backend name (informational).
    pub provider: Option<String>,
    /// Maximum input context window, in tokens.
    pub context_window: u32,
    /// Maximum output tokens per request.
    pub max_output: Option<u32>,
    /// Per-1M-token pricing.
    pub pricing: Pricing,
    /// Benchmark scores. Free-form; `average` (if present) is the default
    /// ranking score. Other keys (e.g. `mmlu`, `human_eval`) are available for
    /// targeted selection.
    pub scores: std::collections::BTreeMap<String, f64>,
}

impl ModelInfo {
    /// The blended average score, falling back to the `average` key, then to
    /// the mean of all listed scores, then 0.
    pub fn score(&self) -> f64 {
        if let Some(v) = self.scores.get("average") {
            return *v;
        }
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.values().sum::<f64>() / self.scores.len() as f64
    }

    /// Effective cost per token (prompt + blended completion). Used only for
    /// ranking when a real per-request cost isn't yet known; we weight
    /// completion at a typical 1:3 completion:prompt ratio.
    pub fn blended_cost_per_token(&self) -> f64 {
        let p = self.pricing.prompt_per_1m / 1_000_000.0;
        let c = self.pricing.completion_per_1m / 1_000_000.0;
        // 1 prompt token + ~0.33 completion tokens of expected weight.
        p * 0.75 + c * 0.25
    }
}

/// Top-level shape of a catalog YAML file: either a bare model or a `models:` list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CatalogFile {
    List { models: Vec<ModelInfo> },
    Single(ModelInfo),
}

/// Selection strategy for [`Catalog::pick`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Pick the model with the highest benchmark score (cost-agnostic).
    #[default]
    BestScore,
    /// Pick the highest-quality model whose blended cost stays under `max_cost`.
    BestScoreUnderBudget,
    /// Pick the cheapest model whose score is at least `min_score`.
    CheapestAboveFloor,
    /// Best "score per dollar" — maximize `score / blended_cost_per_token`.
    BestValue,
}

/// Constraints/options passed to [`Catalog::pick`].
#[derive(Debug, Clone, Default)]
pub struct PickOptions {
    pub strategy: Strategy,
    /// Reject models scoring below this. 0 = no floor.
    pub min_score: f64,
    /// Reject models whose blended cost-per-token exceeds this. None = no cap.
    pub max_cost_per_token: Option<f64>,
    /// Reject models whose context window is smaller than this. None = no req.
    pub min_context_window: Option<u32>,
}

/// The model catalog. Cloneable (state shared behind an `Arc`+`Mutex`).
#[derive(Clone)]
pub struct Catalog {
    inner: std::sync::Arc<Mutex<CatalogState>>,
}

#[derive(Default)]
struct CatalogState {
    models: Vec<ModelInfo>,
    dir: Option<PathBuf>,
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(CatalogState::default())),
        }
    }

    /// Set the catalog directory (does not load — call [`Self::reload`]).
    pub async fn set_dir(&self, dir: impl Into<PathBuf>) {
        self.inner.lock().await.dir = Some(dir.into());
    }

    /// Re-scan the catalog directory. Returns the count loaded.
    pub async fn reload(&self) -> Result<usize> {
        let dir = self.inner.lock().await.dir.clone();
        let Some(dir) = dir else {
            self.inner.lock().await.models.clear();
            return Ok(0);
        };
        let models = load_dir(&dir)?;
        let count = models.len();
        let mut g = self.inner.lock().await;
        g.models = models;
        info!(dir = %dir.display(), count, "model catalog reloaded");
        Ok(count)
    }

    /// All loaded models.
    pub async fn list(&self) -> Vec<ModelInfo> {
        self.inner.lock().await.models.clone()
    }

    /// Look up a model by id.
    pub async fn get(&self, id: &str) -> Option<ModelInfo> {
        self.inner
            .lock()
            .await
            .models
            .iter()
            .find(|m| m.id == id)
            .cloned()
    }

    /// Pick the best model subject to `opts`. Returns `None` if the catalog is
    /// empty or no model satisfies the constraints.
    pub async fn pick(&self, opts: &PickOptions) -> Option<ModelInfo> {
        let g = self.inner.lock().await;
        let candidates: Vec<&ModelInfo> = g
            .models
            .iter()
            .filter(|m| {
                if opts.min_score > 0.0 && m.score() < opts.min_score {
                    return false;
                }
                if let Some(max) = opts.max_cost_per_token
                    && m.blended_cost_per_token() > max
                {
                    return false;
                }
                if let Some(ctx) = opts.min_context_window
                    && m.context_window < ctx
                {
                    return false;
                }
                true
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let pick = match opts.strategy {
            Strategy::BestScore => candidates
                .iter()
                .copied()
                .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap(),
            Strategy::BestScoreUnderBudget => candidates
                .iter()
                .copied()
                .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap(),
            Strategy::CheapestAboveFloor => candidates
                .iter()
                .copied()
                .min_by(|a, b| {
                    a.blended_cost_per_token()
                        .partial_cmp(&b.blended_cost_per_token())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap(),
            Strategy::BestValue => candidates
                .iter()
                .copied()
                .max_by(|a, b| {
                    let va = a.score() / a.blended_cost_per_token().max(1e-12);
                    let vb = b.score() / b.blended_cost_per_token().max(1e-12);
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap(),
        };
        Some(pick.clone())
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

/// A short, human-readable explanation of why a model was (or wasn't) picked.
pub fn explain_pick(picked: Option<&ModelInfo>, all: &[ModelInfo], opts: &PickOptions) -> String {
    match picked {
        Some(m) => format!(
            "picked '{}' (score {:.1}, ~${:.4}/tok, ctx {}) via {:?} from {} candidates",
            m.id,
            m.score(),
            m.blended_cost_per_token(),
            m.context_window,
            opts.strategy,
            all.len(),
        ),
        None => format!("no model satisfied {:?} among {} candidates", opts.strategy, all.len()),
    }
}

/// Render the catalog as a JSON value (for tools / inspection).
pub fn catalog_json(models: &[ModelInfo]) -> Value {
    json!(models)
}

fn load_dir(dir: &Path) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    if !dir.exists() {
        return Ok(models);
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading catalog dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("yaml") | Some("yml")
            )
        })
        .collect();
    entries.sort();
    for path in entries {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "failed to read catalog file");
                continue;
            }
        };
        let parsed: CatalogFile = match serde_yaml::from_str(&text) {
            Ok(c) => c,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "failed to parse catalog file");
                continue;
            }
        };
        match parsed {
            CatalogFile::List { models: ms } => models.extend(ms),
            CatalogFile::Single(m) => models.push(m),
        }
    }
    // Drop models without an id (invalid entries).
    models.retain(|m| !m.id.trim().is_empty());
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(unique: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sloth-catalog-{unique}"))
    }

    #[tokio::test]
    async fn reload_picks_best_score() {
        let dir = tmp("best");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.yaml"),
            "models:\n  - id: cheap\n    context_window: 8000\n    pricing: {prompt_per_1m: 0.1, completion_per_1m: 0.2}\n    scores: {average: 70.0}\n  - id: smart\n    context_window: 128000\n    pricing: {prompt_per_1m: 2.0, completion_per_1m: 8.0}\n    scores: {average: 90.0}\n",
        ).unwrap();
        let cat = Catalog::new();
        cat.set_dir(&dir).await;
        assert_eq!(cat.reload().await.unwrap(), 2);
        let pick = cat.pick(&PickOptions::default()).await.unwrap();
        assert_eq!(pick.id, "smart");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn pick_cheapest_above_floor() {
        let dir = tmp("cheap");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("m.yaml"),
            "models:\n  - id: cheap\n    context_window: 8000\n    pricing: {prompt_per_1m: 0.1, completion_per_1m: 0.2}\n    scores: {average: 75.0}\n  - id: smart\n    context_window: 128000\n    pricing: {prompt_per_1m: 2.0, completion_per_1m: 8.0}\n    scores: {average: 90.0}\n",
        ).unwrap();
        let cat = Catalog::new();
        cat.set_dir(&dir).await;
        cat.reload().await.unwrap();
        let opts = PickOptions {
            strategy: Strategy::CheapestAboveFloor,
            min_score: 70.0,
            ..Default::default()
        };
        assert_eq!(cat.pick(&opts).await.unwrap().id, "cheap");
        // Floor excludes the cheap one → falls back to smart.
        let opts = PickOptions {
            strategy: Strategy::CheapestAboveFloor,
            min_score: 80.0,
            ..Default::default()
        };
        assert_eq!(cat.pick(&opts).await.unwrap().id, "smart");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn pick_respects_context_requirement() {
        let dir = tmp("ctx");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("m.yaml"),
            "models:\n  - id: small\n    context_window: 8000\n    scores: {average: 90.0}\n  - id: big\n    context_window: 128000\n    scores: {average: 85.0}\n",
        ).unwrap();
        let cat = Catalog::new();
        cat.set_dir(&dir).await;
        cat.reload().await.unwrap();
        let opts = PickOptions {
            min_context_window: Some(50_000),
            ..Default::default()
        };
        assert_eq!(cat.pick(&opts).await.unwrap().id, "big");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn score_fallbacks() {
        let mut m = ModelInfo::default();
        assert_eq!(m.score(), 0.0);
        m.scores.insert("mmlu".into(), 80.0);
        m.scores.insert("humaneval".into(), 60.0);
        assert!((m.score() - 70.0).abs() < 1e-6);
        m.scores.insert("average".into(), 91.0);
        assert!((m.score() - 91.0).abs() < 1e-6);
    }
}
