//! Time-based scheduler: a small in-process cron engine.
//!
//! Each [`ScheduledJob`] carries a name, a 5-field cron expression, a prompt,
//! and the session the prompt should be dispatched into when the job fires.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    /// Stable id (assigned on insertion).
    #[serde(default)]
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// 5-field cron expression (UTC).
    pub cron: String,
    /// Prompt dispatched to the agent session when the job fires.
    pub prompt: String,
    /// Session id the firing prompt runs under.
    pub session_id: String,
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

    /// List all scheduled jobs (parsed cron not exposed).
    pub fn list(&self) -> Vec<ScheduledJob> {
        self.inner.jobs.iter().map(|e| e.job.clone()).collect()
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
    use super::*;

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
                },
                0,
            )
            .is_err()
        );
    }
}
