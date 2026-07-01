//! Runtime wiring: connect to the bridge, subscribe, handle inbound messages
//! with the chat agent, and send replies back. Includes a reconnect loop with
//! exponential backoff and a heartbeat.

use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};
use tokio_tungstenite::tungstenite::Message;

use std::sync::Arc;

use crate::a2a::A2aRegistry;
use crate::agent::ChatAgent;
use crate::bridge::{Envelope, Inbound, SendTextPayload, SubscribePayload};
use crate::compact::Compactor;
use crate::config::Config;
use crate::hitl::{HitlBroker, Outcome};
use crate::mcp::McpRegistry;
use crate::memory::MemoryStore;
use crate::model_catalog::Catalog;
use crate::scheduler::{FiredJob, Scheduler};
use crate::session::SessionManager;
use crate::skill::SkillRegistry;
use crate::tools::ToolRouter;

/// Shared agent services: chat agent + the tool-calling subsystem (scheduler,
/// sessions, remote MCP registry, skill registry, A2A registry, HITL gate).
/// Built once and passed through every bridge session so tool/scheduler state
/// survives reconnects.
#[derive(Clone)]
pub struct AgentContext {
    pub agent: ChatAgent,
    pub router: ToolRouter,
    pub scheduler: Arc<Scheduler>,
    pub sessions: Arc<SessionManager>,
    pub mcp: Arc<McpRegistry>,
    pub skills: Arc<SkillRegistry>,
    pub a2a: Arc<A2aRegistry>,
    pub catalog: Arc<Catalog>,
    pub memory: Arc<MemoryStore>,
    pub hitl: Arc<HitlBroker>,
    /// Multi-tenancy & RBAC registry.
    pub tenants: Arc<crate::tenant::Tenants>,
    /// Whether RBAC enforcement is on (mirrors `cfg.tenancy.enabled`).
    pub tenancy_enabled: bool,
    /// Maximum tool-call rounds per reply.
    pub max_tool_steps: usize,
    /// Effective model id (either the fixed `llm.model` or catalog-picked).
    pub model: String,
}

impl AgentContext {
    /// Build the context from config. Connects configured remote MCP / A2A
    /// servers (best-effort: failures are logged, not fatal), loads skills,
    /// the model catalog, and memory, and starts no background tasks — callers
    /// drive the scheduler/HITL via the channels in [`start`].
    pub async fn build(cfg: &Config) -> Result<Self> {
        // Model catalog: if configured, pick a model automatically; otherwise
        // fall back to the fixed `llm.model`.
        let catalog = Arc::new(Catalog::new());
        let mut model = cfg.llm.model.clone();
        if let Some(dir) = &cfg.models.dir {
            catalog.set_dir(dir).await;
            if let Err(e) = catalog.reload().await {
                tracing::warn!(error = %e, "initial model catalog reload reported errors");
            }
            let opts = crate::model_catalog::PickOptions {
                strategy: parse_strategy(&cfg.models.strategy),
                min_score: cfg.models.min_score,
                max_cost_per_token: cfg.models.max_cost_per_token.filter(|v| *v > 0.0),
                min_context_window: cfg.models.min_context_window.filter(|v| *v > 0),
            };
            if let Some(picked) = catalog.pick(&opts).await {
                tracing::info!(
                    picked = %picked.id,
                    score = picked.score(),
                    "{}",
                    crate::model_catalog::explain_pick(Some(&picked), &catalog.list().await, &opts)
                );
                model = picked.id;
            } else {
                tracing::warn!(
                    "model catalog produced no pick; falling back to llm.model = {model}"
                );
            }
        }

        // Build the chat agent with an optional compactor.
        let compactor = if cfg.compact.enabled {
            Some(Compactor::new(&cfg.llm, cfg.compact.clone()))
        } else {
            None
        };

        let scheduler = Arc::new(Scheduler::new());
        let sessions = Arc::new(SessionManager::new(
            &cfg.sessions.default_session,
            &cfg.sessions.store_dir,
        ));
        let mcp = Arc::new(McpRegistry::new());
        let skills = Arc::new(SkillRegistry::new());
        let a2a = Arc::new(A2aRegistry::new());
        let memory = Arc::new(MemoryStore::new());
        let hitl = Arc::new(HitlBroker::new(cfg.hitl.clone()));
        // The agent needs a handle to memory for system-prompt injection.
        let agent = ChatAgent::with_compactor_and_memory(
            &cfg.llm,
            cfg.history.max_messages,
            compactor,
            Some((*memory).clone()),
            cfg.memory.inject_into_prompt,
        )
        .context("failed to build chat agent")?;

        // Eagerly connect any preconfigured MCP servers (best-effort).
        if !cfg.mcp.servers.is_empty()
            && let Err(e) = mcp.reload(&cfg.mcp.servers).await
        {
            tracing::warn!(error = %e, "initial MCP reload reported errors");
        }

        // Load skills (best-effort).
        if let Some(dir) = &cfg.skills.dir {
            skills.set_dir(dir).await;
            if let Err(e) = skills.reload().await {
                tracing::warn!(error = %e, "initial skills reload reported errors");
            }
        }

        // Eagerly connect any preconfigured A2A agents (best-effort).
        if !cfg.a2a.agents.is_empty()
            && let Err(e) = a2a.reload(&cfg.a2a.agents).await
        {
            tracing::warn!(error = %e, "initial A2A reload reported errors");
        }

        // Memory store directory.
        if let Some(dir) = &cfg.memory.dir {
            memory.set_dir(dir).await;
        }

        // Multi-tenancy & RBAC registry. Built from config; enforcement is on
        // only when `[tenancy].enabled = true`. When off, the router treats
        // every principal as authorized (single-tenant behavior).
        let tenants = Arc::new(crate::tenant::Tenants::from_config(&cfg.tenancy));
        let tenancy_enabled = cfg.tenancy.enabled;
        if tenancy_enabled {
            tracing::info!(
                default_role = %cfg.tenancy.default_role,
                members = cfg.tenancy.members.len(),
                "multi-tenancy & RBAC enabled"
            );
        } else {
            tracing::info!("multi-tenancy & RBAC disabled (single-tenant mode)");
        }

        let router = ToolRouter::new(
            scheduler.clone(),
            sessions.clone(),
            mcp.clone(),
            skills.clone(),
            a2a.clone(),
            catalog.clone(),
            memory.clone(),
            hitl.clone(),
            cfg.mcp.servers.clone(),
            cfg.a2a.agents.clone(),
            cfg.mcp.expose_tools,
            cfg.skills.expose_tools,
            cfg.a2a.expose_tools,
            cfg.models.expose_tools,
            cfg.memory.expose_tools,
        )
        .with_tenants(tenants.clone());
        // When tenancy is disabled, turn enforcement back off (with_tenants
        // flips it on; we re-establish the configured state).
        let router = if tenancy_enabled {
            router
        } else {
            // Reconstruct with enforcement off but keep the registry for
            // `tenant_whoami` diagnostics (reports rbac_enabled=false).
            let mut r = router;
            r.set_tenancy_enabled(false);
            r
        };

        Ok(Self {
            agent,
            router,
            scheduler,
            sessions,
            mcp,
            skills,
            a2a,
            catalog,
            memory,
            hitl,
            tenants,
            tenancy_enabled,
            max_tool_steps: 8,
            model,
        })
    }
}

fn parse_strategy(s: &str) -> crate::model_catalog::Strategy {
    use crate::model_catalog::Strategy;
    match s.trim().to_ascii_lowercase().as_str() {
        "best_score_under_budget" => Strategy::BestScoreUnderBudget,
        "cheapest_above_floor" => Strategy::CheapestAboveFloor,
        "best_value" => Strategy::BestValue,
        _ => Strategy::BestScore,
    }
}

/// Run the agent until shutdown.
pub async fn run(cfg: Config) -> Result<()> {
    run_with_shutdown(cfg, std::future::pending::<()>()).await
}

/// Run the agent until the `shutdown` future completes (or the runtime errors
/// unrecoverably, which it currently does not — it reconnects).
///
/// Exposed so callers (and tests) can stop the reconnect loop gracefully.
pub async fn run_with_shutdown<F>(cfg: Config, shutdown: F) -> Result<()>
where
    F: std::future::Future<Output = ()>,
{
    let ctx = AgentContext::build(&cfg).await?;
    run_ctx_with_shutdown(cfg, ctx, shutdown).await
}

/// Like [`run_with_shutdown`] but with a prebuilt [`AgentContext`].
pub async fn run_ctx_with_shutdown<F>(cfg: Config, ctx: AgentContext, shutdown: F) -> Result<()>
where
    F: std::future::Future<Output = ()>,
{
    let channel = cfg.bridge.channel.clone();
    let account_id = cfg.bridge.account_id.clone();

    tokio::pin!(shutdown);

    // Start the scheduler if enabled: it emits fired jobs we feed back into
    // the agent as synthetic inbound prompts. `_sched_handle` must outlive the
    // whole run: it's the oneshot sender whose drop stops the ticker, so we
    // bind it in the outer scope (not inside the `if` block, where it would be
    // dropped immediately and kill the ticker the instant it starts).
    let mut _sched_handle: Option<tokio::sync::oneshot::Sender<()>> = None;
    let mut sched_rx = if cfg.scheduler.enabled {
        let (stx, srx) = tokio::sync::oneshot::channel::<()>();
        let rx = ctx.scheduler.start(cfg.scheduler.tick_secs, srx);
        _sched_handle = Some(stx);
        Some(rx)
    } else {
        None
    };

    // Skills hot-reload ticker: periodically rescan the skills directory so
    // added/edited/removed skill files take effect without a restart. Only
    // started when a directory is configured and a non-zero poll interval set.
    let _skills_ticker = {
        let skills = ctx.skills.clone();
        let poll = cfg.skills.poll_secs;
        let has_dir = cfg.skills.dir.is_some();
        tokio::spawn(async move {
            if !has_dir || poll == 0 {
                return;
            }
            let mut tick = interval(Duration::from_secs(poll.max(1)));
            tick.tick().await; // skip immediate first tick
            loop {
                tick.tick().await;
                if let Err(e) = skills.reload().await {
                    tracing::warn!(error = %e, "skills hot-reload sweep failed");
                }
            }
        })
    };

    // Model-catalog hot-reload ticker: periodically rescan the catalog so new
    // model YAML files / edited pricing/scores are picked up live.
    let _catalog_ticker = {
        let catalog = ctx.catalog.clone();
        let poll = cfg.models.poll_secs;
        let has_dir = cfg.models.dir.is_some();
        tokio::spawn(async move {
            if !has_dir || poll == 0 {
                return;
            }
            let mut tick = interval(Duration::from_secs(poll.max(1)));
            tick.tick().await;
            loop {
                tick.tick().await;
                if let Err(e) = catalog.reload().await {
                    tracing::warn!(error = %e, "model catalog hot-reload sweep failed");
                }
            }
        })
    };

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received; stopping agent");
                return Ok(());
            }
            outcome = run_session(&cfg, &ctx, channel.clone(), account_id.clone(), sched_rx.as_mut()) => {
                match &outcome {
                    Ok(()) => tracing::info!("bridge session ended cleanly; reconnecting"),
                    Err(e) => tracing::warn!(error = %e, "bridge session ended with error"),
                }
            }
        }

        // Reconnect backoff (exponential, capped).
        let mut backoff = cfg.bridge.reconnect_ms.max(1_000);
        backoff = backoff.min(cfg.bridge.reconnect_max_ms);
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received during backoff; stopping agent");
                return Ok(());
            }
            _ = tokio::time::sleep(Duration::from_millis(backoff)) => {}
        }
    }
}

/// Parse a user reply to a HITL confirmation into a [`Outcome`].
///
/// Accepts yes/no, y/n, approve/deny, confirm/cancel (case-insensitive).
/// Anything else maps to `Denied` (fail-closed).
pub fn parse_hitl_reply(text: &str) -> Outcome {
    let t = text.trim().to_lowercase();
    let head = t.split_whitespace().next().unwrap_or("");
    match head {
        "yes" | "y" | "approve" | "approved" | "ok" | "confirm" | "confirmed" | "allow" | "1" => {
            Outcome::Approved
        }
        "no" | "n" | "deny" | "denied" | "cancel" | "cancelled" | "canceled" | "reject"
        | "rejected" | "block" | "0" => Outcome::Denied,
        _ => Outcome::Denied,
    }
}

/// One connection lifecycle: connect → subscribe → pump messages until the
/// socket closes or errors.
async fn run_session(
    cfg: &Config,
    ctx: &AgentContext,
    channel: String,
    account_id: String,
    mut sched_rx: Option<&mut mpsc::UnboundedReceiver<FiredJob>>,
) -> Result<()> {
    let span = tracing::info_span!("connect", url = %cfg.bridge.url);
    let _enter = span.enter();

    tracing::info!("connecting to bridge");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&cfg.bridge.url)
        .await
        .map_err(|e| anyhow::anyhow!("failed to connect to bridge ws: {e}"))?;
    tracing::info!("bridge websocket connected");

    let (mut sink, mut stream) = ws_stream.split();

    // Send subscribe.
    let sub_payload = serde_json::to_value(SubscribePayload {
        channel: &channel,
        account_id: &account_id,
    })?;
    let sub_env = Envelope::outgoing("subscribe", &channel, sub_payload);
    send_text(&mut sink, &sub_env).await?;
    tracing::info!(%channel, %account_id, "subscribed to channel account");

    // Channel for outbound replies produced by message handlers.
    let (tx, mut rx) = mpsc::channel::<Envelope>(64);

    // Spawn a writer task that drains the reply channel.
    let writer = tokio::spawn(async move {
        while let Some(env) = rx.recv().await {
            if let Err(e) = send_text(&mut sink, &env).await {
                tracing::error!(error = %e, "failed to send outbound envelope");
                return;
            }
        }
        // Close the sink gracefully when the channel closes.
        let _ = sink.close().await;
    });

    // Heartbeat task.
    let heartbeat_ms = cfg.bridge.heartbeat_ms;
    let hb_tx = tx.clone();
    let hb_channel = channel.clone();
    let heartbeat = tokio::spawn(async move {
        if heartbeat_ms == 0 {
            return;
        }
        let mut tick = interval(Duration::from_millis(heartbeat_ms));
        tick.tick().await; // first immediate tick
        loop {
            tick.tick().await;
            let env = Envelope::outgoing("ping", &hb_channel, serde_json::Value::Null);
            if hb_tx.send(env).await.is_err() {
                break;
            }
        }
    });

    // HITL pending-confirmation channel: surface requests to the user. The
    // broker publishes a pending confirmation; the reply loop resolves it
    // when the human answers. There is exactly one consumer (this session).
    let mut hitl_rx = ctx.hitl.pending_channel_async().await;

    // Main receive loop — also drains fired-job prompts (from the scheduler)
    // and HITL confirmations.
    loop {
        let msg = tokio::select! {
            // Fired scheduled job → inject as a synthetic agent prompt.
            job = async { match sched_rx.as_mut() { Some(r) => r.recv().await, None => None } } => {
                if let Some(job) = job {
                    handle_fired_job(ctx, &tx, &channel, &account_id, job).await;
                }
                continue;
            }
            // Pending HITL confirmation → ask the human over the bridge.
            pending = hitl_rx.recv() => {
                if let Some(p) = pending {
                    ask_hitl(&tx, &channel, &account_id, &p, cfg.hitl.timeout_secs).await;
                }
                continue;
            }
            // Bridge frame.
            msg = timeout(
                Duration::from_secs((cfg.bridge.heartbeat_ms.max(5_000) * 3 / 1000).max(60)),
                stream.next(),
            ) => match msg {
                Ok(Some(m)) => m,
                Ok(None) => {
                    tracing::info!("bridge websocket stream closed");
                    break;
                }
                Err(_) => {
                    tracing::warn!("bridge read timed out; closing session");
                    break;
                }
            },
        };

        match msg {
            Ok(Message::Text(text)) => {
                handle_text(
                    text.to_string(),
                    ctx,
                    &mut hitl_rx,
                    &tx,
                    &channel,
                    &account_id,
                )
                .await;
            }
            Ok(Message::Binary(b)) => {
                tracing::debug!(len = b.len(), "ignoring binary frame");
            }
            Ok(Message::Ping(p)) => {
                tracing::debug!(?p, "ws ping (auto-ponged by tungstenite)");
            }
            Ok(Message::Pong(p)) => {
                tracing::debug!(?p, "ws pong");
            }
            Ok(Message::Close(reason)) => {
                tracing::info!(?reason, "bridge sent close frame");
                break;
            }
            Ok(Message::Frame(f)) => {
                tracing::debug!(?f, "raw frame");
            }
            Err(e) => {
                tracing::warn!(error = %e, "websocket error reading frame");
                break;
            }
        }
    }

    // Shutdown helpers.
    heartbeat.abort();
    drop(tx);
    let _ = writer.await;
    Ok(())
}

/// A scheduled job fired: run its prompt through the agent and reply to the
/// channel (so the scheduler's output is visible to the user).
async fn handle_fired_job(
    ctx: &AgentContext,
    tx: &mpsc::Sender<Envelope>,
    channel: &str,
    account_id: &str,
    job: FiredJob,
) {
    tracing::info!(job = %job.id, name = %job.name, session = %job.session_id, "dispatching fired job");
    let prompt = format!("[scheduled task: {}]\n{}", job.name, job.prompt);
    // Reconstruct the principal that owns the job: its tenant + the sender who
    // scheduled it (captured as `reply_to`). History for the fired prompt is
    // keyed by this principal's scope, matching the key used when the job was
    // created (see `tools.rs::SCHED_ADD`).
    let tenant_id = job.tenant_id.clone().unwrap_or_default();
    let sender_id = job
        .reply_to
        .clone()
        .unwrap_or_else(|| job.session_id.clone());
    let principal = crate::tenant::Principal::new(tenant_id, sender_id);
    match ctx
        .agent
        .reply_with_tools(&principal, &prompt, &ctx.router, ctx.max_tool_steps)
        .await
    {
        Ok(reply) => {
            // Route the scheduled reply to the user who scheduled the job when
            // we captured a target; otherwise fall back to the channel.
            let to = job
                .reply_to
                .as_deref()
                .map(|r| format_reply_target(channel, r))
                .unwrap_or_else(|| channel.to_string());
            let env = build_send_text(channel, account_id, &to, &reply.text, None, None);
            if tx.send(env).await.is_err() {
                tracing::warn!("outbound channel closed; dropping scheduled reply");
            }
        }
        Err(e) => {
            tracing::error!(error = %e, job = %job.id, "scheduled job reply failed");
        }
    }
}

/// Send a HITL confirmation question to the channel. The human's next text
/// message is matched against any pending confirmation in [`handle_text`].
async fn ask_hitl(
    tx: &mpsc::Sender<Envelope>,
    channel: &str,
    account_id: &str,
    p: &crate::hitl::PendingConfirmation,
    timeout_secs: u64,
) {
    let text = format!(
        "🔑 Approval needed for `{}` (id `{}`):\n{}\nReply `yes` to approve or `no` to deny (auto-denies in {timeout_secs}s).",
        p.tool, p.id, p.summary
    );
    let env = build_send_text(channel, account_id, channel, &text, None, None);
    let _ = tx.send(env).await;
}

/// Parse a text frame as a bridge envelope and dispatch.
async fn handle_text(
    text: String,
    ctx: &AgentContext,
    hitl_rx: &mut mpsc::UnboundedReceiver<crate::hitl::PendingConfirmation>,
    tx: &mpsc::Sender<Envelope>,
    channel: &str,
    account_id: &str,
) {
    let env: Envelope = match serde_json::from_str(&text) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, frame = %text, "failed to parse bridge envelope");
            return;
        }
    };

    let event = env.into_event();
    match event {
        Inbound::Welcome(w) => {
            tracing::info!(
                version = ?w.version,
                "received welcome from bridge"
            );
        }
        Inbound::Status(s) => {
            tracing::info!(status = %s.status, detail = ?s.detail, error = ?s.error, "channel status");
        }
        Inbound::SendAck(ack) => {
            tracing::debug!(request_id = %ack.request_id, message_id = ?ack.message_id, "send ack");
        }
        Inbound::SendError(err) => {
            tracing::warn!(
                request_id = %err.request_id,
                code = %err.code,
                message = %err.message,
                "send error from bridge"
            );
        }
        Inbound::Pong => {
            tracing::trace!("pong");
        }
        Inbound::Other { message_type, .. } => {
            tracing::debug!(%message_type, "unhandled message type");
        }
        Inbound::Message(msg) => {
            // Only handle text-like messages; ignore media/voice/etc.
            if msg.msg_type != "text" && msg.msg_type != "markdown" {
                tracing::info!(
                    msg_type = %msg.msg_type,
                    sender = %msg.sender_id,
                    "ignoring non-text inbound message"
                );
                return;
            }
            let user_text = msg.text.trim();
            if user_text.is_empty() {
                tracing::debug!(sender = %msg.sender_id, "ignoring empty inbound message");
                return;
            }

            tracing::info!(
                sender = %msg.sender_id,
                chat = %msg.chat_id,
                message_id = %msg.message_id,
                msg_type = %msg.msg_type,
                chars = user_text.len(),
                "inbound user message"
            );

            // First check if this is a reply to a pending HITL confirmation.
            // A user's plain text message is interpreted as the human's
            // decision to the most recent pending request. We only peek
            // non-blockingly: if no confirmation is waiting, this is a normal
            // prompt; if one is waiting, consume it as the human's decision.
            let reply_target = resolve_reply_target(channel, &msg);
            if let Ok(pending) = hitl_rx.try_recv() {
                let decision = if user_text.eq_ignore_ascii_case("yes")
                    || user_text.eq_ignore_ascii_case("y")
                    || user_text.eq_ignore_ascii_case("approve")
                {
                    Outcome::Approved
                } else {
                    parse_hitl_reply(user_text)
                };
                let acknowledged = ctx.hitl.resolve(&pending.id, decision).await;
                if acknowledged {
                    let note = match decision {
                        Outcome::Approved => "✅ Approved. Proceeding.",
                        Outcome::Denied => "🚫 Denied.",
                        Outcome::TimedOut => "⏱️ Timed out.",
                    };
                    let env = build_send_text(
                        channel,
                        account_id,
                        &reply_target,
                        note,
                        msg.reply_to_message_id.as_deref(),
                        msg.context_token.as_deref(),
                    );
                    let _ = tx.send(env).await;
                    return;
                }
                // Not the relevant pending request: fall through and treat the
                // message as a normal prompt.
            }
            // Generate a reply (tool-augmented). Derive the principal from the
            // subscription (tenant) + inbound sender id; RBAC + state
            // namespacing flow from it.
            let sender_id = msg.sender_id.clone();
            let tenant_id = crate::tenant::tenant_id_from_subscription(channel, account_id);
            let principal = crate::tenant::Principal::new(tenant_id, sender_id.clone());
            let reply = match ctx
                .agent
                .reply_with_tools(&principal, user_text, &ctx.router, ctx.max_tool_steps)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(error = %e, sender = %sender_id, "chat completion failed");
                    // Still acknowledge the user with an error notice.
                    let err_text =
                        "⚠️ Sorry, I couldn't generate a reply just now. Please try again.";
                    let env = build_send_text(
                        channel,
                        account_id,
                        &reply_target,
                        err_text,
                        msg.reply_to_message_id.as_deref(),
                        msg.context_token.as_deref(),
                    );
                    let _ = tx.send(env).await;
                    return;
                }
            };

            tracing::info!(
                sender = %sender_id,
                reply_chars = reply.text.len(),
                model = %reply.model,
                "outbound reply generated"
            );

            let env = build_send_text(
                channel,
                account_id,
                &reply_target,
                &reply.text,
                msg.reply_to_message_id.as_deref(),
                msg.context_token.as_deref(),
            );
            if tx.send(env).await.is_err() {
                tracing::warn!("outbound channel closed; dropping reply");
            }
        }
    }
}

/// Resolve the outbound `to` target for replying to `msg`.
///
/// Prefers the bridge-provided `replyTo` field — a ready-to-echo target the
/// channel adapter computes from the inbound context (e.g. mattermost sets
/// `user:<id>` for DMs, `channel:<id>` for groups). Using it keeps the agent
/// free of per-channel target-format knowledge: the bridge is the single
/// source of truth. Falls back to [`format_reply_target`] for channels or
/// bridge versions that don't populate `replyTo`.
fn resolve_reply_target(channel: &str, msg: &crate::bridge::InboundMessage) -> String {
    if let Some(rt) = msg.reply_to.as_deref()
        && !rt.is_empty()
    {
        return rt.to_string();
    }
    format_reply_target(channel, &msg.sender_id)
}

/// Fallback outbound `to` target when the bridge supplies no `replyTo`.
///
/// The bridge protocol's `send_text` `to` field is a channel-specific target
/// string. For DM channels like mattermost, a bare user id is rejected with
/// `403 Forbidden` — the plugin's outbound resolver only recognizes the
/// `user:<id>` form for direct messages (and `channel:<id>` for group
/// channels). Other channels (e.g. liangzimixin) take the sender id verbatim,
/// so we only rewrite when the channel expects the `user:` prefix. We also
/// avoid double-prefixing if the inbound sender id already carries a scheme.
fn format_reply_target(channel: &str, sender_id: &str) -> String {
    let bare = !sender_id.contains(':');
    if bare && channel == "mattermost" {
        format!("user:{sender_id}")
    } else {
        sender_id.to_string()
    }
}

/// Build a `send_text` envelope for a reply.
fn build_send_text(
    channel: &str,
    account_id: &str,
    to: &str,
    text: &str,
    reply_to_message_id: Option<&str>,
    context_token: Option<&str>,
) -> Envelope {
    let payload = serde_json::to_value(SendTextPayload {
        to,
        text,
        reply_to_message_id,
        context_token,
    })
    .expect("send_text payload is always serializable");
    Envelope::outgoing("send_text", channel, payload).with_account(account_id)
}

/// Serialize and send an envelope as a text frame.
async fn send_text<S>(sink: &mut S, env: &Envelope) -> Result<()>
where
    S: SinkExt<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    let text = env.to_text().context("serializing envelope")?;
    sink.send(Message::Text(text.into()))
        .await
        .map_err(|e| anyhow::anyhow!("ws send failed: {e}"))?;
    Ok(())
}
