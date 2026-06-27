//! OpenClaw bridge WebSocket protocol.
//!
//! Typed wrappers around the JSON envelope described in the bridge's
//! `src/protocol/messages.ts`. Only the fields the agent actually uses are
//! modeled; unknown fields deserialize through `#[serde(flatten)]` capture /
//! `serde_json::Value` and are ignored.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Protocol version. The bridge is currently at v1.
pub const PROTOCOL_VERSION: u32 = 1;

/// Bridge envelope.
///
/// The wire protocol uses camelCase (`accountId`). `type` and `v` are
/// lower-case keywords; `message_type` is renamed to `type` explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Envelope {
    pub v: u32,
    /// Correlation id — echoed by the server in responses.
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub channel: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ts: Option<u64>,
}

// ─── Client → Server payloads ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscribePayload<'a> {
    pub channel: &'a str,
    pub account_id: &'a str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SendTextPayload<'a> {
    pub to: &'a str,
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to_message_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_token: Option<&'a str>,
}

// ─── Server → Client payloads (the pieces we read) ───────────────────────────

/// `inbound_message` payload.
///
/// Fields mirror the bridge protocol (camelCase on the wire) — not all are
/// read by the agent but are kept for completeness and future use.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct InboundMessage {
    pub message_id: String,
    #[serde(default)]
    pub chat_id: String,
    pub sender_id: String,
    #[serde(default)]
    pub sender_name: Option<String>,
    /// e.g. "text" | "markdown" | "image" | ...
    pub msg_type: String,
    #[serde(default)]
    pub text: String,
    pub timestamp: u64,
    #[serde(default)]
    pub context_token: Option<String>,
    #[serde(default)]
    pub media_url: Option<String>,
    #[serde(default)]
    pub reply_to_message_id: Option<String>,
}

/// `channel_status` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ChannelStatus {
    pub status: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// `welcome` payload (subset).
#[derive(Debug, Clone, Deserialize)]
pub struct Welcome {
    #[serde(default)]
    pub version: Option<String>,
}

/// `send_ack` payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendAck {
    pub request_id: String,
    #[serde(default)]
    pub message_id: Option<String>,
}

/// `send_error` payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendError {
    pub request_id: String,
    pub code: String,
    pub message: String,
}

// ─── Builders ────────────────────────────────────────────────────────────────

impl Envelope {
    /// Build an outbound envelope with a fresh correlation id.
    pub fn outgoing(message_type: &str, channel: &str, payload: Value) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: format!("sloth-{}", uuid::Uuid::new_v4().simple()),
            message_type: message_type.to_string(),
            channel: channel.to_string(),
            account_id: None,
            payload,
            ts: None,
        }
    }

    /// Serialize to a WebSocket text frame.
    pub fn to_text(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Set the account id on this envelope (builder-style).
    pub fn with_account(mut self, account_id: &str) -> Self {
        self.account_id = Some(account_id.to_string());
        self
    }
}

/// A typed view over an inbound envelope's payload, for dispatch.
#[derive(Debug)]
#[allow(dead_code)]
pub enum Inbound {
    /// A user message from a backend channel.
    Message(InboundMessage),
    /// Channel connection status change.
    Status(ChannelStatus),
    /// Welcome envelope on connect.
    Welcome(Welcome),
    /// Acknowledgement that an outbound send succeeded.
    SendAck(SendAck),
    /// An outbound send failed.
    SendError(SendError),
    /// Pong reply to our ping.
    Pong,
    /// Any other message type we don't specifically handle.
    Other {
        message_type: String,
        payload: Value,
    },
}

impl Envelope {
    /// Classify this envelope into a typed inbound event.
    pub fn into_event(self) -> Inbound {
        let Envelope {
            message_type,
            payload,
            ..
        } = self;
        match message_type.as_str() {
            "inbound_message" => match serde_json::from_value::<InboundMessage>(payload.clone()) {
                Ok(m) => Inbound::Message(m),
                Err(e) => {
                    tracing::warn!(error = %e, "malformed inbound_message payload");
                    Inbound::Other {
                        message_type,
                        payload,
                    }
                }
            },
            "channel_status" => match serde_json::from_value::<ChannelStatus>(payload.clone()) {
                Ok(s) => Inbound::Status(s),
                Err(e) => {
                    tracing::warn!(error = %e, "malformed channel_status payload");
                    Inbound::Other {
                        message_type,
                        payload,
                    }
                }
            },
            "welcome" => {
                let w =
                    serde_json::from_value::<Welcome>(payload).unwrap_or(Welcome { version: None });
                Inbound::Welcome(w)
            }
            "send_ack" => {
                let a = serde_json::from_value::<SendAck>(payload).unwrap_or(SendAck {
                    request_id: String::new(),
                    message_id: None,
                });
                Inbound::SendAck(a)
            }
            "send_error" => {
                let e = serde_json::from_value::<SendError>(payload).unwrap_or(SendError {
                    request_id: String::new(),
                    code: "unknown".to_string(),
                    message: "malformed send_error".to_string(),
                });
                Inbound::SendError(e)
            }
            "pong" => Inbound::Pong,
            other => Inbound::Other {
                message_type: other.to_string(),
                payload,
            },
        }
    }
}
