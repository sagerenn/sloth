//! Time-based scheduler: a small in-process cron engine.
//!
//! Each [`ScheduledJob`] carries a name, a cron expression (5-field
//! minute-granular or 6-field second-precision), a prompt, and the session the
//! prompt should be dispatched into when the job fires.
//! The scheduler tracks each job's next fire time and emits [`FiredJob`]
//! events as they come due. It is driven by a background task in [`Scheduler::start`]
//! but its due-job computation is exposed synchronously via [`Scheduler::evaluate`]
//! for deterministic testing without a wall clock.

use std::sync::Arc;

use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::cron::Cron;

/// A scheduled job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduledJob {
    /// Stable id (assigned on insertion).
    #[serde(default)]
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Cron expression (UTC): 5-field (minute-granular) or 6-field
    /// (second-precision, leading seconds field).
    pub cron: String,
    /// Prompt dispatched to the agent session when the job fires.
    pub prompt: String,
    /// Session id the firing prompt runs under.
    pub session_id: String,
    /// Tenant id the job belongs to. Used to scope `list`/`remove` so a
    /// sender only sees/operates on their own tenant's jobs. `None` is
    /// treated as the empty tenant (legacy / unscoped jobs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// Outbound `to` target the fired job's reply is sent to. When `None`,
    /// the runtime falls back to the channel (broadcast). Set from the
    /// originating sender so a fired job replies to the user who scheduled it
    /// rather than the whole channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
}

/// Internal entry: job + parsed cron + next fire time (epoch secs).
#[derive(Debug, Clone)]
struct Entry {
    job: ScheduledJob,
    cron: Cron,
    next_fire: i64,
}

/// An event emitted when a job comes due.
#[derive(Debug, Clone)]
pub struct FiredJob {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub session_id: String,
    /// Tenant id the job belongs to (copied from the job). The runtime uses
    /// it to dispatch the fired prompt under the correct tenant's principal.
    pub tenant_id: Option<String>,
    /// Outbound `to` target for the fired job's reply (copied from the job).
    pub reply_to: Option<String>,
    /// The instant (epoch secs) the job fired at.
    pub fired_at: i64,
}

/// The scheduler. Cheaply cloneable (state is shared behind an `Arc`).
#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    jobs: DashMap<String, Entry>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner::default()),
        }
    }

    /// Add a job. Parses + validates the cron expression and computes the
    /// next fire time after `now`. Returns the assigned job id.
    pub fn add_at(&self, mut job: ScheduledJob, now: i64) -> Result<String> {
        if job.id.is_empty() {
            job.id = format!("job-{}", Uuid::new_v4().simple());
        }
        let cron = Cron::parse(&job.cron).with_context(|| format!("invalid cron: {}", job.cron))?;
        let next_fire = cron.next_after(now);
        let id = job.id.clone();
        self.inner.jobs.insert(
            id.clone(),
            Entry {
                job,
                cron,
                next_fire,
            },
        );
        Ok(id)
    }

    /// Add a job using the wall clock for `now`.
    pub fn add(self: &Scheduler, job: ScheduledJob) -> Result<String> {
        self.add_at(job, wall_secs())
    }

    /// Remove a job by id. Returns true if it existed.
    pub fn remove(&self, id: &str) -> bool {
        self.inner.jobs.remove(id).is_some()
    }

    /// Remove a job by id only if it belongs to `tenant`. Returns true if
    /// removed. `tenant=None` matches only unscoped (legacy) jobs.
    pub fn remove_for(&self, id: &str, tenant: Option<&str>) -> bool {
        if let Some(entry) = self.inner.jobs.get(id) {
            if entry.job.tenant_id.as_deref() == tenant {
                drop(entry);
                self.inner.jobs.remove(id).is_some()
            } else {
                false
            }
        } else {
            false
        }
    }

    /// List all scheduled jobs (parsed cron not exposed).
    pub fn list(&self) -> Vec<ScheduledJob> {
        self.inner.jobs.iter().map(|e| e.job.clone()).collect()
    }

    /// List only jobs belonging to `tenant`. `tenant=None` lists unscoped jobs.
    pub fn list_for(&self, tenant: Option<&str>) -> Vec<ScheduledJob> {
        self.inner
            .jobs
            .iter()
            .filter(|e| e.job.tenant_id.as_deref() == tenant)
            .map(|e| e.job.clone())
            .collect()
    }

    /// Get a job by id.
    pub fn get(&self, id: &str) -> Option<ScheduledJob> {
        self.inner.jobs.get(id).map(|e| e.job.clone())
    }

    /// Synchronously evaluate which jobs are due at `now`, advance their next
    /// fire times, and return the fired events. Idempotent: a job whose
    /// `next_fire` is in the future produces nothing; one whose `next_fire`
    /// has passed fires once and reschedules to its next future match.
    pub fn evaluate(&self, now: i64) -> Vec<FiredJob> {
        let mut fired = Vec::new();
        for mut entry in self.inner.jobs.iter_mut() {
            let e = entry.value_mut();
            // Fire while due (catches up missed ticks, capped to avoid loops).
            let mut guard = 0;
            while e.next_fire <= now && guard < 60 {
                fired.push(FiredJob {
                    id: e.job.id.clone(),
                    name: e.job.name.clone(),
                    prompt: e.job.prompt.clone(),
                    session_id: e.job.session_id.clone(),
                    tenant_id: e.job.tenant_id.clone(),
                    reply_to: e.job.reply_to.clone(),
                    fired_at: e.next_fire,
                });
                e.next_fire = e.cron.next_after(e.next_fire);
                guard += 1;
                // Stop after the first future reschedule if we've caught up —
                // we don't want to emit dozens of events for one tick.
                if e.next_fire > now {
                    break;
                }
            }
        }
        fired
    }

    /// Spawn the background ticker. Emits [`FiredJob`]s on the returned
    /// channel; stops when `shutdown` completes or the receiver is dropped.
    pub fn start(
        &self,
        tick_secs: u64,
        shutdown: tokio::sync::oneshot::Receiver<()>,
    ) -> mpsc::UnboundedReceiver<FiredJob> {
        let (tx, rx) = mpsc::unbounded_channel();
        let me = self.clone();
        let tick = tick_secs.max(1);
        tokio::spawn(async move {
            let mut shutdown = shutdown;
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(tick));
            interval.tick().await; // immediate first tick
            loop {
                tokio::select! {
                    _ = &mut shutdown => break,
                    _ = interval.tick() => {
                        let now = wall_secs();
                        for f in me.evaluate(now) {
                            tracing::info!(job = %f.id, name = %f.name, "scheduled job fired");
                            if tx.send(f).is_err() {
                                return; // receiver gone
                            }
                        }
                    }
                }
            }
        });
        rx
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Wall clock as epoch seconds (UTC).
pub fn wall_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn start_ticker_emits_due_second_precision_job() {
        // Regression guard: the background ticker spawned by `start` must keep
        // running (its shutdown sender must not be dropped prematurely) and
        // actually emit a FiredJob for a due job. A 6-field `*/1 * * * * *`
        // job (every second) should fire within a few seconds.
        let s = Scheduler::new();
        let id = s
            .add(ScheduledJob {
                id: String::new(),
                name: "every-second".into(),
                cron: "*/1 * * * * *".into(),
                prompt: "tick".into(),
                session_id: "default".into(),
                ..Default::default()
            })
            .unwrap();
        let (_stx, srx) = tokio::sync::oneshot::channel::<()>();
        let mut rx = s.start(1, srx);
        // The next fire is at most ~1s out (every-second cron); allow slack.
        let fired = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await;
        let fired = fired
            .expect("ticker never emitted a FiredJob within 5s")
            .expect("channel closed");
        assert_eq!(fired.id, id);
        assert_eq!(fired.prompt, "tick");
    }

    #[test]
    fn add_and_evaluate_fires_due_job_once() {
        let s = Scheduler::new();
        let now = 1_000_000_i64;
        s.add_at(
            ScheduledJob {
                id: String::new(),
                name: "tick".into(),
                cron: "* * * * *".into(),
                prompt: "hello".into(),
                session_id: "default".into(),
                ..Default::default()
            },
            now,
        )
        .unwrap();

        // next_fire after now(1_000_000) for every-minute = 1_000_060.
        let fires = s.evaluate(now + 60);
        assert_eq!(fires.len(), 1);
        assert_eq!(fires[0].prompt, "hello");
        // Re-evaluating at the same now must not refire.
        let again = s.evaluate(now + 60);
        assert!(again.is_empty());
        // Next fire should now be +120.
        let next = s.evaluate(now + 120);
        assert_eq!(next.len(), 1);
    }

    #[test]
    fn remove_and_list() {
        let s = Scheduler::new();
        let id = s
            .add_at(
                ScheduledJob {
                    id: String::new(),
                    name: "j".into(),
                    cron: "*/5 * * * *".into(),
                    prompt: "p".into(),
                    session_id: "default".into(),
                    ..Default::default()
                },
                0,
            )
            .unwrap();
        assert_eq!(s.list().len(), 1);
        assert!(s.remove(&id));
        assert!(s.list().is_empty());
        assert!(!s.remove(&id));
    }

    #[test]
    fn invalid_cron_rejected() {
        let s = Scheduler::new();
        assert!(
            s.add_at(
                ScheduledJob {
                    id: String::new(),
                    name: "x".into(),
                    cron: "not a cron".into(),
                    prompt: "p".into(),
                    session_id: "default".into(),
                    ..Default::default()
                },
                0,
            )
            .is_err()
        );
    }
}
