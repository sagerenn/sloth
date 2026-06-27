//! End-to-end test: a mock OpenClaw bridge WebSocket server + the agent,
//! wired through the real LLM gateway.
//!
//! Flow:
//!   1. Start a mock WS server speaking the bridge envelope protocol.
//!   2. Server receives a `subscribe` from the agent and replies with a
//!      `welcome` + `channel_status(connected)`.
//!   3. Server sends an `inbound_message` from a fake sender.
//!   4. The agent calls the LLM (live `glm-5.2`) and sends back a `send_text`.
//!   5. Server asserts the `send_text` payload `to` matches the sender and
//!      `text` is non-empty, then signals shutdown.
//!
//! Requires the live LLM gateway; skips otherwise.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use sloth_agent::config::{BridgeConfig, Config, HistoryConfig, LlmConfig, ObservabilityConfig};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message;

async fn endpoint_reachable() -> bool {
    let base = std::env::var("SLOTH_LLM_BASE_URL")
        .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".to_string());
    let url = format!("{}/models", base.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(&url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn test_config(ws_url: String) -> Config {
    Config {
        bridge: BridgeConfig {
            url: ws_url,
            channel: "test-channel".to_string(),
            account_id: "default".to_string(),
            reconnect_ms: 1_000,
            reconnect_max_ms: 1_000,
            heartbeat_ms: 0, // disable heartbeat for deterministic test
        },
        llm: LlmConfig {
            base_url: std::env::var("SLOTH_LLM_BASE_URL")
                .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".to_string()),
            model: std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".to_string()),
            api_key: std::env::var("SLOTH_LLM_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            system_prompt: "You are a concise test assistant. Reply with one short sentence."
                .to_string(),
            temperature: Some(0.0),
            max_tokens: Some(512),
            timeout_secs: 60,
        },
        history: HistoryConfig { max_messages: 10 },
        observability: ObservabilityConfig {
            log_format: "text".to_string(),
            log_filter: "info,sloth_agent=debug".to_string(),
            service_name: "sloth-agent-test".to_string(),
        },
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway"]
async fn bridge_round_trip_replies_to_inbound_message() {
    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }

    // Bind the mock server on an ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let ws_url = format!("ws://127.0.0.1:{port}/bridge");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let sender_id = "user-123@im.test".to_string();
    let sender_id_for_server = sender_id.clone();

    // Mock bridge server.
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();

        // Send welcome.
        let welcome = json!({
            "v": 1,
            "id": "srv-1",
            "type": "welcome",
            "channel": "test-channel",
            "payload": { "version": "test-1.0", "channels": {} },
            "ts": 1,
        });
        ws.send(Message::Text(welcome.to_string().into()))
            .await
            .unwrap();

        // Wait for subscribe.
        let mut got_reply: Option<Value> = None;
        for _ in 0..200 {
            tokio::select! {
                biased;
                msg = ws.next() => {
                    let Some(Ok(Message::Text(txt))) = msg else { continue };
                    let env: Value = serde_json::from_str(&txt).unwrap();
                    match env["type"].as_str() {
                        Some("subscribe") => {
                            // Acknowledge with a connected channel_status.
                            let status = json!({
                                "v": 1,
                                "id": env["id"].clone(),
                                "type": "channel_status",
                                "channel": "test-channel",
                                "payload": { "status": "connected" },
                                "ts": 2,
                            });
                            ws.send(Message::Text(status.to_string().into())).await.unwrap();

                            // Send an inbound user message.
                            let inbound = json!({
                                "v": 1,
                                "id": "srv-msg-1",
                                "type": "inbound_message",
                                "channel": "test-channel",
                                "accountId": "default",
                                "payload": {
                                    "messageId": "m-1",
                                    "chatId": "c-1",
                                    "senderId": sender_id_for_server,
                                    "senderName": "Tester",
                                    "msgType": "text",
                                    "text": "Hello! In one short sentence, who are you?",
                                    "timestamp": 1719400000000u64,
                                },
                                "ts": 3,
                            });
                            ws.send(Message::Text(inbound.to_string().into())).await.unwrap();
                        }
                        Some("send_text") => {
                            got_reply = Some(env.clone());
                            // Acknowledge the send.
                            let ack = json!({
                                "v": 1,
                                "id": "srv-ack",
                                "type": "send_ack",
                                "channel": "test-channel",
                                "payload": { "requestId": env["id"], "messageId": "out-1" },
                                "ts": 4,
                            });
                            ws.send(Message::Text(ack.to_string().into())).await.unwrap();
                            break;
                        }
                        _ => {}
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(30)) => break,
            }
        }

        let _ = ws.close(None).await;
        got_reply
    });

    // Initialise tracing for visibility.
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,sloth_agent=debug,warn")
        .try_init();

    let cfg = test_config(ws_url);
    let agent_task = tokio::spawn(async move {
        sloth_agent::runtime::run_with_shutdown(cfg, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let reply = server.await.expect("server task panicked");
    let _ = shutdown_tx.send(());

    // Give the agent a moment to observe shutdown.
    tokio::time::timeout(Duration::from_secs(5), agent_task)
        .await
        .expect("agent did not shut down in time")
        .expect("agent task panicked")
        .expect("agent returned error");

    let reply = reply.expect("server never received a send_text from the agent");

    assert_eq!(reply["type"], "send_text");
    assert_eq!(reply["channel"], "test-channel");
    assert_eq!(reply["accountId"], "default");
    assert_eq!(reply["payload"]["to"], sender_id);

    let text = reply["payload"]["text"]
        .as_str()
        .expect("send_text payload.text is a string");
    assert!(!text.is_empty(), "reply text must not be empty");
    println!("agent -> bridge reply: {text:?}");
}
