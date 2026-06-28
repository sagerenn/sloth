//! Live end-to-end test: the sloth agent answering a **real Mattermost user**
//! through the OpenClaw bridge running as the published Docker image.
//!
//! This is the sloth-side mirror of the openclaw-bridge Mattermost E2E. Where
//! the bridge's own suite drives a stub "agent" over the WebSocket, this one
//! wires the *real* sloth runtime + live LLM into the pipeline:
//!
//! ```text
//!   Mattermost user --[REST post]--> Mattermost --[posted WS]--> bridge
//!        ^                                ^                          | inbound_message
//!        |                                |                          v
//!        |                          bot posts reply            sloth-agent --> LLM (glm-5.2)
//!        |                                ^                          | send_text
//!        +--[posted WS]-- bot's post -----+                          |
//! ```
//!
//! The bridge (`ghcr.io/sagerenn/openclaw-bridge`) and a Mattermost server
//! (`mattermost/mattermost-preview`) run as Docker containers on a shared
//! network so the bridge can reach Mattermost by container hostname. The sloth
//! agent runs in-process (its real `runtime::run_with_shutdown`) and connects
//! to the bridge's published WebSocket port from the host. The "real IM user"
//! is a tiny in-process Mattermost REST + WebSocket client driven by a Personal
//! Access Token: it receives the bot's outbound DMs via the `posted` event and
//! posts replies over REST to generate inbound traffic.
//!
//! Three full round-trips are exchanged: for each, the user DMs the bot, the
//! agent (via the LLM) replies through the bridge, and the user receives the
//! reply. This validates the complete inbound -> sloth -> LLM -> outbound path
//! over the mattermost channel plugin.
//!
//! ## Prerequisites
//!
//! - The LLM gateway reachable at `SLOTH_LLM_BASE_URL` (default
//!   `http://172.17.0.1:8317/v1`). The test skips when it is not.
//! - Docker available on the host (the `docker` binary). The `docker run`-
//!   driven containers are cleaned up on exit.
//! - `docker pull` access to `ghcr.io/sagerenn/openclaw-bridge` and
//!   `mattermost/mattermost-preview`.
//!
//! Overrides (env vars):
//!   SLOTH_LLM_BASE_URL   LLM gateway base URL (default http://172.17.0.1:8317/v1)
//!   SLOTH_LLM_MODEL      model id (default glm-5.2)
//!   SLOTH_LLM_API_KEY    optional API key
//!   E2E_BRIDGE_IMAGE     bridge image (default ghcr.io/sagerenn/openclaw-bridge:latest)
//!   E2E_MM_IMAGE         mattermost image (default mattermost/mattermost-preview:latest)
//!   E2E_BRIDGE_PORT      host port published for the bridge WS (default 19499)
//!   E2E_BRIDGE_PORT_MM   host port published for Mattermost (default 18065)
//!   E2E_MM_SETTLE_MS     settle delay before replying (default 600)
//!
//! Run with: `cargo test --test mattermost_e2e -- --nocapture --ignored`

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use sloth_agent::config::{
    A2aConfig, BridgeConfig, CompactConfig, Config, HistoryConfig, HitlConfig, LlmConfig,
    McpConfig, MemoryConfig, ModelCatalogConfig, ObservabilityConfig, SchedulerConfig,
    SessionConfig, SkillsConfig,
};
use tokio::sync::{Mutex, oneshot};
use tokio_tungstenite::tungstenite::Message;

// Unique suffix per process so repeated runs don't collide on container names.
fn run_token() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

// ─── Docker orchestration helpers ────────────────────────────────────────────

/// Run `docker` with the given args, returning combined stdout (trimmed).
fn docker(args: &[&str]) -> Result<String> {
    let out = Command::new("docker")
        .args(args)
        .output()
        .context("failed to spawn `docker`")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("docker {} failed: {stderr}", args.join(" "));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn docker_ok(args: &[&str]) -> Result<()> {
    docker(args)?;
    Ok(())
}

/// Whether the LLM gateway answers /models. Used to skip-not-fail in CI.
async fn llm_reachable() -> bool {
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

/// Best-effort cleanup of all containers + network for a given run token.
/// Never fails the test.
fn teardown(net: &str, mm: &str, bridge: &str) {
    for c in [mm, bridge] {
        let _ = Command::new("docker").args(["rm", "-f", c]).output();
    }
    let _ = Command::new("docker").args(["network", "rm", net]).output();
}

/// Bring up a Mattermost preview server + the OpenClaw bridge container on a
/// shared network. Returns the host-reachable URLs and resolved credential
/// environment once the bridge reports healthy.
struct Stack {
    net: String,
    mm_container: String,
    bridge_container: String,
    mm_host_url: String, // Mattermost as seen from the host (provisioning + sender)
    mm_internal_url: String, // Mattermost as seen from the bridge container (hostname)
    bridge_ws_url: String, // bridge WS as seen from the host (sloth connects here)
}

fn bring_up_stack() -> Result<Stack> {
    let token = run_token();
    let net = format!("sloth-e2e-mm-{token}");
    let mm_container = format!("sloth-e2e-mm-{token}");
    let bridge_container = format!("sloth-e2e-bridge-{token}");

    let mm_image = std::env::var("E2E_MM_IMAGE")
        .unwrap_or_else(|_| "mattermost/mattermost-preview:latest".to_string());

    let mm_port: u16 = std::env::var("E2E_BRIDGE_PORT_MM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18065);
    let bridge_port: u16 = std::env::var("E2E_BRIDGE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(19499);

    // Fresh network. (Ignore failure then re-create to be robust.)
    let _ = Command::new("docker")
        .args(["network", "rm", &net])
        .output();
    docker_ok(&["network", "create", &net]).context("create docker network")?;

    // Mattermost preview server with E2E-friendly settings.
    docker_ok(&[
        "run",
        "-d",
        "--name",
        &mm_container,
        "--network",
        &net,
        "-p",
        &format!("{mm_port}:8065"),
        "-e",
        "MM_SERVICESETTINGS_SITEURL=http://127.0.0.1:8065",
        "-e",
        "MM_SERVICESETTINGS_ENABLEUSERACCESSTOKENS=true",
        "-e",
        "MM_SERVICESETTINGS_ENABLEBOTACCOUNTCREATION=true",
        "-e",
        "MM_SERVICESETTINGS_ALLOWUNTRUSTEDINTERNALCONNECTIONS=true",
        "-e",
        "MM_SERVICESETTINGS_ENABLEOPENSERVER=true",
        "-e",
        "MM_SERVICESETTINGS_ENABLEUSERCREATION=true",
        "-e",
        "MM_EMAILSETTINGS_SMTPSERVER=",
        "-e",
        "MM_PLUGINSETTINGS_ENABLEUPLOADS=false",
        &mm_image,
    ])
    .context("start mattermost container")?;

    let mm_host_url = format!("http://127.0.0.1:{mm_port}");
    // The bridge reaches Mattermost over the shared network by container name.
    let mm_internal_url = format!("http://{mm_container}:8065");

    Ok(Stack {
        net,
        mm_container,
        bridge_container,
        mm_host_url,
        mm_internal_url,
        bridge_ws_url: format!("ws://127.0.0.1:{bridge_port}/bridge"),
    })
}

/// Wait for a URL to answer `GET /api/v4/system/ping` with 200.
async fn wait_for_mattermost(url: &str) -> Result<()> {
    let url = format!("{}/api/v4/system/ping", url.trim_end_matches('/'));
    let deadline = std::time::Instant::now() + Duration::from_secs(240);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    loop {
        if let Ok(r) = client.get(&url).send().await
            && r.status().is_success()
        {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            bail!("mattermost never became reachable at {url}");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// Wait for the bridge to answer its health/spec endpoint.
async fn wait_for_bridge(host_port: u16) -> Result<()> {
    let url = format!("http://127.0.0.1:{host_port}/spec/openapi.json");
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()?;
    loop {
        if let Ok(r) = client.get(&url).send().await
            && r.status().is_success()
        {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            bail!("bridge never became reachable at {url}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Start the bridge container with the mattermost account config baked in.
///
/// The config is passed **inline** (base64-decoded inside the container by a
/// `sh -c` startup command) rather than via a bind mount. This keeps the test
/// portable across Docker installs — notably the snap-packaged Docker, which
/// silently turns host bind-mounts of disallowed paths (e.g. `/tmp`) into empty
/// directories and would crash the bridge with `EISDIR`.
fn start_bridge_container(
    stack: &Stack,
    bot_token: &str,
    mm_url: &str,
    host_port: u16,
) -> Result<()> {
    let cfg = json!({
        "server": {
            "host": "0.0.0.0",
            "port": 9300,
            "path": "/bridge",
        },
        "channels": {
            "mattermost": {
                "enabled": true,
                "accounts": {
                    "default": {
                        "botToken": bot_token,
                        "baseUrl": mm_url,
                        "dmPolicy": "open",
                        "groupPolicy": "open",
                        "allowFrom": ["*"],
                        // Mattermost lives on a private/docker network; the
                        // openclaw runtime's SSRF guard would otherwise block it.
                        "network": { "dangerouslyAllowPrivateNetwork": true }
                    }
                }
            }
        },
        "logging": { "level": "info" }
    });
    let cfg_bytes = serde_json::to_vec(&cfg).context("serialize bridge config")?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&cfg_bytes);

    // Decode the config to /home/openclaw/config.json (the image runs as a
    // non-root `openclaw` user, so /app is read-only — the home dir is
    // writable) and launch the bridge with `--config` pointing at it. The
    // config is base64 (shell-word-safe) so it needs no quoting; passing
    // `sh`, `-c`, <script> as separate argv elements avoids tini treating the
    // whole command as a single filename. CWD stays /app so plugin discovery
    // (which scans node_modules relative to cwd) still finds the mattermost
    // plugin; ContactStore writes contacts.json next to the config (home dir).
    let startup = format!(
        "echo {b64} | base64 -d > /home/openclaw/config.json && exec node /app/dist/server.js --config /home/openclaw/config.json"
    );

    let port = format!("{host_port}:9300");
    let image = stack_image();

    docker_ok(&[
        "run",
        "-d",
        "--name",
        &stack.bridge_container,
        "--network",
        &stack.net,
        "-p",
        &port,
        &image,
        "sh",
        "-c",
        &startup,
    ])
    .context("start bridge container")?;
    Ok(())
}

/// The published openclaw-bridge image, overridable for local testing.
fn stack_image() -> String {
    std::env::var("E2E_BRIDGE_IMAGE")
        .unwrap_or_else(|_| "ghcr.io/sagerenn/openclaw-bridge:latest".to_string())
}

// ─── Provisioning (REST + psql role promotion) ───────────────────────────────

const ADMIN_USER: &str = "e2e-admin";
const ADMIN_EMAIL: &str = "e2e-admin@e2e.local";
const ADMIN_PASS: &str = "E2e-Admin!pass1";
const BOT_USER: &str = "e2e-bot";
const BOT_DISPLAY: &str = "OpenClaw E2E Bot";
const SENDER_USER: &str = "e2e-sender";
const SENDER_EMAIL: &str = "e2e-sender@e2e.local";
const SENDER_PASS: &str = "E2e-Sender!pass1";

#[derive(Clone)]
struct MmUser {
    token: String,
    user_id: String,
}

#[derive(Clone)]
struct Provisions {
    bot: MmUser,
    sender: MmUser,
}

/// Provision a System Admin + bot + sender on a fresh mattermost-preview
/// server, returning their tokens + ids. Mirrors scripts/provision-mattermost.sh.
async fn provision(stack: &Stack) -> Result<Provisions> {
    let url = stack.mm_host_url.clone();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    // 1. Register admin via REST (open-server signup).
    let reg = client
        .post(format!("{}/api/v4/users", url))
        .json(&json!({
            "email": ADMIN_EMAIL, "username": ADMIN_USER, "password": ADMIN_PASS
        }))
        .send()
        .await?;
    let sc = reg.status().as_u16();
    if sc != 201 && sc != 400 {
        bail!("admin signup returned {sc} (open-server signup must be enabled)");
    }

    // 2. Promote the admin role to system_admin via psql inside the container.
    docker_ok(&[
        "exec",
        "-e",
        "PGPASSWORD=mostest",
        &stack.mm_container,
        "psql",
        "-h",
        "localhost",
        "-U",
        "mmuser",
        "-d",
        "mattermost_test",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        &format!(
            "UPDATE users SET roles = 'system_user system_admin' WHERE username = '{ADMIN_USER}';"
        ),
    ])
    .context("promote admin role via psql")?;

    // 3. Restart so the in-memory user/role cache reloads, then re-wait.
    docker_ok(&["restart", &stack.mm_container]).context("restart mattermost")?;
    wait_for_mattermost(&url).await?;

    // 4. Admin login -> capture the Token response header.
    let admin_token = mm_login(&client, &url, ADMIN_USER, ADMIN_PASS).await?;
    let auth = format!("Bearer {admin_token}");

    // Sender (human) account + PAT.
    ensure_user(&client, &url, &auth, SENDER_EMAIL, SENDER_USER, SENDER_PASS).await?;
    let sender_id = mm_user_id_by_username(&client, &url, &auth, SENDER_USER).await?;
    let sender_token = mm_mint_pat(&client, &url, &auth, &sender_id, "e2e sender PAT").await?;

    // Bot account + bot token.
    let bot_id = mm_create_bot(&client, &url, &auth, BOT_USER, BOT_DISPLAY).await?;
    let bot_token = mm_mint_bot_token(&client, &url, &auth, &bot_id, "e2e bot token").await?;

    Ok(Provisions {
        bot: MmUser {
            token: bot_token,
            user_id: bot_id,
        },
        sender: MmUser {
            token: sender_token,
            user_id: sender_id,
        },
    })
}

async fn mm_login(
    client: &reqwest::Client,
    url: &str,
    login_id: &str,
    pass: &str,
) -> Result<String> {
    let resp = client
        .post(format!("{}/api/v4/users/login", url))
        .json(&json!({ "login_id": login_id, "password": pass }))
        .send()
        .await?;
    if let Some(t) = resp.headers().get("token").and_then(|v| v.to_str().ok())
        && !t.is_empty()
    {
        return Ok(t.to_string());
    }
    bail!("admin login returned no token header");
}

async fn ensure_user(
    client: &reqwest::Client,
    url: &str,
    auth: &str,
    email: &str,
    user: &str,
    pass: &str,
) -> Result<()> {
    let r = client
        .post(format!("{}/api/v4/users", url))
        .header("Authorization", auth)
        .json(&json!({ "email": email, "username": user, "password": pass }))
        .send()
        .await?;
    let sc = r.status().as_u16();
    // 201 created, or 400 if the user already exists.
    if sc != 201 && sc != 400 {
        bail!("ensure_user({user}) returned {sc}");
    }
    Ok(())
}

async fn mm_user_id_by_username(
    client: &reqwest::Client,
    url: &str,
    auth: &str,
    username: &str,
) -> Result<String> {
    let v: Value = client
        .get(format!("{}/api/v4/users/username/{username}", url))
        .header("Authorization", auth)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    v["id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("no user id for {username}"))
}

async fn mm_mint_pat(
    client: &reqwest::Client,
    url: &str,
    auth: &str,
    user_id: &str,
    desc: &str,
) -> Result<String> {
    let v: Value = client
        .post(format!("{}/api/v4/users/{user_id}/tokens", url))
        .header("Authorization", auth)
        .json(&json!({ "description": desc }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    v["token"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("no PAT minted for {user_id}"))
}

async fn mm_create_bot(
    client: &reqwest::Client,
    url: &str,
    auth: &str,
    username: &str,
    display: &str,
) -> Result<String> {
    let r = client
        .post(format!("{}/api/v4/bots", url))
        .header("Authorization", auth)
        .json(&json!({ "username": username, "display_name": display, "description": "e2e bot" }))
        .send()
        .await?;
    let sc = r.status().as_u16();
    if sc == 201 {
        let v: Value = r.json().await?;
        return v["user_id"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("bot create: no user_id"));
    }
    // Already exists -> look up by username.
    let v: Value = client
        .get(format!("{}/api/v4/bots/username/{username}", url))
        .header("Authorization", auth)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    v["user_id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("bot lookup: no user_id"))
}

async fn mm_mint_bot_token(
    client: &reqwest::Client,
    url: &str,
    auth: &str,
    bot_id: &str,
    desc: &str,
) -> Result<String> {
    let r = client
        .post(format!("{}/api/v4/bots/{bot_id}/tokens", url))
        .header("Authorization", auth)
        .json(&json!({ "description": desc }))
        .send()
        .await?;
    let v: Value = r.json().await?;
    if let Some(t) = v["token"].as_str()
        && !t.is_empty()
    {
        return Ok(t.to_string());
    }
    // Fallback: PAT for the bot's underlying user.
    let v2: Value = client
        .post(format!("{}/api/v4/users/{bot_id}/tokens", url))
        .header("Authorization", auth)
        .json(&json!({ "description": desc }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    v2["token"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("no bot token minted"))
}

// ─── Minimal Mattermost client (the "real IM user") ──────────────────────────

/// A tiny Mattermost REST + WebSocket-events client used as the far-end human
/// user. Authenticated with a PAT; it opens the event stream to receive the
/// bot's outbound DMs (`posted` events) and posts replies over REST.
struct MattermostUser {
    base_url: String,
    token: String,
    my_user_id: String,
    bot_user_id: String,
    client: reqwest::Client,
    /// Posts authored by the bot, received on the sender's event stream.
    received: Arc<Mutex<Vec<(String, String)>>>, // (post_id, text)
    dm_channel: Arc<Mutex<Option<String>>>,
    ws_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MattermostUser {
    fn new(base_url: String, token: String, my_user_id: String, bot_user_id: String) -> Self {
        Self {
            base_url,
            token,
            my_user_id,
            bot_user_id,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(20))
                .build()
                .unwrap(),
            received: Arc::new(Mutex::new(Vec::new())),
            dm_channel: Arc::new(Mutex::new(None)),
            ws_handle: None,
        }
    }

    /// Open the WS event stream, authenticate the PAT, and resolve once the
    /// server responds with `hello`/OK. Continues draining events in a task.
    async fn connect(&mut self) -> Result<()> {
        let ws_url = self
            .base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/api/v4/websocket";
        let (ws, _) = tokio_tungstenite::connect_async(&ws_url)
            .await
            .map_err(|e| anyhow!("mattermost ws connect failed: {e}"))?;
        let (mut sink, mut stream) = ws.split();

        // Authenticate the connection with the PAT.
        let auth = json!({
            "seq": 1, "action": "authentication_challenge",
            "data": { "token": self.token }
        });
        sink.send(Message::Text(auth.to_string().into())).await?;

        let bot_user_id = self.bot_user_id.clone();
        let received = self.received.clone();
        let (hello_tx, hello_rx) = oneshot::channel::<()>();

        let handle = tokio::spawn(async move {
            let mut hello_tx = Some(hello_tx);
            while let Some(Ok(msg)) = stream.next().await {
                let Message::Text(txt) = msg else { continue };
                let v: Value = match serde_json::from_str(&txt) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Resolve the connect promise once the server acknowledges auth.
                if let Some(tx) = hello_tx.take() {
                    let is_hello = v["event"].as_str() == Some("hello") || v["status"] == "OK";
                    if is_hello {
                        let _ = tx.send(());
                    }
                }
                if v["event"].as_str() == Some("posted") {
                    let raw = &v["data"]["post"];
                    if raw.is_null() {
                        continue;
                    }
                    let post: Value = match raw {
                        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
                        other => other.clone(),
                    };
                    if post["user_id"].as_str() == Some(&bot_user_id) {
                        let id = post["id"].as_str().unwrap_or("").to_string();
                        let text = post["message"].as_str().unwrap_or("").to_string();
                        if !id.is_empty() {
                            received.lock().await.push((id, text));
                        }
                    }
                }
            }
        });
        self.ws_handle = Some(handle);

        // Wait (bounded) for the hello.
        match tokio::time::timeout(Duration::from_secs(15), hello_rx).await {
            Ok(Ok(())) => Ok(()),
            _ => bail!("mattermost WS never sent hello / auth OK"),
        }
    }

    /// Create (or look up) the DM channel between this user and the bot.
    async fn ensure_dm(&self) -> Result<String> {
        if let Some(id) = self.dm_channel.lock().await.clone() {
            return Ok(id);
        }
        let v: Value = self
            .client
            .post(format!("{}/api/v4/channels/direct", self.base_url))
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&json!([self.my_user_id, self.bot_user_id]))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let id = v["id"]
            .as_str()
            .ok_or_else(|| anyhow!("no dm channel id"))?
            .to_string();
        *self.dm_channel.lock().await = Some(id.clone());
        Ok(id)
    }

    /// Post a message to the bot's DM channel.
    async fn post_to_bot(&self, text: &str) -> Result<()> {
        let channel_id = self.ensure_dm().await?;
        let r = self
            .client
            .post(format!("{}/api/v4/posts", self.base_url))
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&json!({ "channel_id": channel_id, "message": text }))
            .send()
            .await?;
        if !r.status().is_success() {
            bail!(
                "post to bot failed: {} {}",
                r.status(),
                r.text().await.unwrap_or_default()
            );
        }
        Ok(())
    }
}

impl Drop for MattermostUser {
    fn drop(&mut self) {
        if let Some(h) = self.ws_handle.take() {
            h.abort();
        }
    }
}

// ─── sloth config + test runner ──────────────────────────────────────────────

fn sloth_config(bridge_ws_url: String) -> Config {
    Config {
        bridge: BridgeConfig {
            url: bridge_ws_url,
            channel: "mattermost".to_string(),
            account_id: "default".to_string(),
            reconnect_ms: 1_000,
            reconnect_max_ms: 3_000,
            // Heartbeat keeps the WS healthy; the bridge enforces one too.
            heartbeat_ms: 25_000,
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
            service_name: "sloth-e2e-mattermost".to_string(),
        },
        mcp: McpConfig::default(),
        scheduler: SchedulerConfig::default(),
        sessions: SessionConfig::default(),
        hitl: HitlConfig::default(),
        skills: SkillsConfig::default(),
        a2a: A2aConfig::default(),
        models: ModelCatalogConfig::default(),
        compact: CompactConfig::default(),
        memory: MemoryConfig::default(),
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway + docker"]
async fn mattermost_round_trip_through_bridge() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,sloth_agent=debug,warn")
        .try_init();

    if !llm_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }
    if docker(&["version"]).is_err() {
        eprintln!("skipping: docker not available");
        return;
    }

    // Bring up the stack up-front so teardown always has names to clean.
    let stack = bring_up_stack().expect("bring up docker stack");
    let bridge_port: u16 = std::env::var("E2E_BRIDGE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(19499);

    // Ensure cleanup runs no matter how we exit.
    let result = run_against_stack(&stack, bridge_port).await;

    teardown(&stack.net, &stack.mm_container, &stack.bridge_container);
    match result {
        Ok(()) => println!("--- Mattermost E2E: 3/3 round-trips succeeded ---"),
        Err(e) => {
            // Dump container logs to aid debugging before cleanup.
            let _ = Command::new("docker")
                .args(["logs", "--tail", "200", &stack.bridge_container])
                .status();
            let _ = Command::new("docker")
                .args(["logs", "--tail", "120", &stack.mm_container])
                .status();
            panic!("Mattermost E2E failed: {e:#}");
        }
    }
}

async fn run_against_stack(stack: &Stack, bridge_port: u16) -> Result<()> {
    eprintln!("[e2e] waiting for mattermost...");
    wait_for_mattermost(&stack.mm_host_url).await?;

    eprintln!("[e2e] provisioning bot + sender...");
    let p = provision(stack).await?;
    eprintln!("[e2e] bot={} sender={}", &p.bot.user_id, &p.sender.user_id);

    eprintln!("[e2e] starting bridge container...");
    start_bridge_container(stack, &p.bot.token, &stack.mm_internal_url, bridge_port)?;
    wait_for_bridge(bridge_port).await?;
    eprintln!("[e2e] bridge healthy on port {bridge_port}");

    // Start the sloth agent runtime in-process, pointed at the bridge.
    let cfg = sloth_config(stack.bridge_ws_url.clone());
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let agent_task = tokio::spawn(async move {
        sloth_agent::runtime::run_with_shutdown(cfg, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    // Give the agent a moment to connect + subscribe.
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Connect the far-end Mattermost user (the human sender).
    let mut sender = MattermostUser::new(
        stack.mm_host_url.clone(),
        p.sender.token.clone(),
        p.sender.user_id.clone(),
        p.bot.user_id.clone(),
    );
    sender.connect().await?;
    sender.ensure_dm().await?;
    eprintln!("[e2e] sender connected (WS event stream + DM channel)");

    // Exchange 3 full round-trips. Each: user DMs the bot -> sloth+LLM replies
    // through the bridge -> the bot posts the reply -> sender receives it.
    let prompts = [
        "Hello! In one short sentence, who are you?",
        "What is 2 plus 3? Just the number.",
        "Say hi back in three words.",
    ];

    let mut seen_post_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (i, prompt) in prompts.iter().enumerate() {
        eprintln!("[e2e] --- round {} ---", i + 1);
        sender.post_to_bot(prompt).await?;
        eprintln!("[e2e] sent prompt #{}: {:?}", i + 1, prompt);

        // Wait for a *fresh* bot reply (a post id not seen in a prior round) to
        // arrive on the sender's event stream. Requiring a new post id — not
        // just any reply — ensures each round genuinely produced its own reply
        // rather than re-observing a stale post from an earlier round.
        let waited = tokio::time::timeout(
            Duration::from_secs(90),
            wait_for_fresh_reply(&sender, prompt, &mut seen_post_ids),
        )
        .await;
        match waited {
            Ok(Ok(text)) => eprintln!("[e2e] received bot reply #{}: {:?}", i + 1, text),
            Ok(Err(e)) => bail!("round {} did not produce a reply: {e}", i + 1),
            Err(_) => bail!("round {} timed out waiting for the bot's reply", i + 1),
        }
    }

    // Shutdown the agent cleanly.
    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(10), agent_task).await;
    Ok(())
}

/// Poll until the sender has received a *fresh* bot-authored post — one whose
/// id is not already in `seen`, whose text is non-empty and not the echoed
/// prompt — recording its id in `seen` and returning the text. Requiring a new
/// post id guarantees each round observed its own reply, not a stale one.
async fn wait_for_fresh_reply(
    sender: &MattermostUser,
    prompt: &str,
    seen: &mut std::collections::HashSet<String>,
) -> Result<String> {
    loop {
        let received = sender.received.lock().await.clone();
        if let Some((id, text)) = received
            .iter()
            .rev()
            .find(|(id, t)| !t.is_empty() && t != prompt && !seen.contains(id))
        {
            seen.insert(id.clone());
            return Ok(text.clone());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}
