//! Runtime wiring: connect to the bridge, subscribe, handle inbound messages
//! with the chat agent, and send replies back. Includes a reconnect loop with
//! exponential backoff and a heartbeat.

use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};
use tokio_tungstenite::tungstenite::Message;

use crate::agent::ChatAgent;
use crate::bridge::{Envelope, Inbound, SendTextPayload, SubscribePayload};
use crate::config::Config;

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
    let agent =
        ChatAgent::new(&cfg.llm, cfg.history.max_messages).context("failed to build chat agent")?;

    let channel = cfg.bridge.channel.clone();
    let account_id = cfg.bridge.account_id.clone();

    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received; stopping agent");
                return Ok(());
            }
            outcome = run_session(&cfg, &agent, channel.clone(), account_id.clone()) => {
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

/// One connection lifecycle: connect → subscribe → pump messages until the
/// socket closes or errors.
async fn run_session(
    cfg: &Config,
    agent: &ChatAgent,
    channel: String,
    account_id: String,
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

    // Main receive loop.
    let mut backoff = cfg.bridge.reconnect_ms;
    loop {
        let msg = match timeout(
            // Use a generous read timeout; the heartbeat keeps the socket alive.
            Duration::from_secs((cfg.bridge.heartbeat_ms.max(5_000) * 3 / 1000).max(60)),
            stream.next(),
        )
        .await
        {
            Ok(Some(m)) => m,
            Ok(None) => {
                tracing::info!("bridge websocket stream closed");
                break;
            }
            Err(_) => {
                tracing::warn!("bridge read timed out; closing session");
                break;
            }
        };

        match msg {
            Ok(Message::Text(text)) => {
                handle_text(text.to_string(), agent, &tx, &channel, &account_id).await;
                backoff = cfg.bridge.reconnect_ms;
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
        let _ = backoff; // backoff currently unused for in-session errors
    }

    // Shutdown helpers.
    heartbeat.abort();
    drop(tx);
    let _ = writer.await;
    Ok(())
}

/// Parse a text frame as a bridge envelope and dispatch.
async fn handle_text(
    text: String,
    agent: &ChatAgent,
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

            // Generate a reply.
            let sender_id = msg.sender_id.clone();
            let reply_target = resolve_reply_target(channel, &msg);
            let reply = match agent.reply(&sender_id, user_text).await {
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
