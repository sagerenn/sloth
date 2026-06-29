//! Shared helpers for the live Mattermost E2E tests.
//!
//! Brings up a Mattermost preview server + the published OpenClaw bridge image
//! on a shared Docker network, provisions a bot + human sender, and provides a
//! minimal Mattermost REST + WebSocket client acting as the far-end user. The
//! sloth agent itself runs in-process (its real `runtime::run_with_shutdown`)
//! and connects to the bridge's published WebSocket port from the host.

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::{Mutex, oneshot};
use tokio_tungstenite::tungstenite::Message;

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
pub async fn llm_reachable() -> bool {
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

/// Whether `docker` is usable on the host.
pub fn docker_available() -> bool {
    docker(&["version"]).is_ok()
}

/// Unique suffix per process so repeated runs don't collide on container names.
fn run_token() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

/// Best-effort cleanup of all containers + network for a given run token.
/// Never fails the test.
pub fn teardown(net: &str, mm: &str, bridge: &str) {
    for c in [mm, bridge] {
        let _ = Command::new("docker").args(["rm", "-f", c]).output();
    }
    let _ = Command::new("docker").args(["network", "rm", net]).output();
}

/// The published openclaw-bridge image, overridable for local testing.
fn stack_image() -> String {
    std::env::var("E2E_BRIDGE_IMAGE")
        .unwrap_or_else(|_| "ghcr.io/sagerenn/openclaw-bridge:latest".to_string())
}

/// Bring up a Mattermost preview server + the OpenClaw bridge container on a
/// shared network. Returns the host-reachable URLs etc. once provisioned.
pub struct Stack {
    pub net: String,
    pub mm_container: String,
    pub bridge_container: String,
    pub mm_host_url: String,
    pub mm_internal_url: String,
    pub bridge_ws_url: String,
}

/// Bring up the docker network + mattermost container. Bridge is started later
/// (it needs the bot token from provisioning).
pub fn bring_up_stack() -> Result<Stack> {
    let token = run_token();
    let net = format!("sloth-e2e-mcp-{token}");
    let mm_container = format!("sloth-e2e-mm-mcp-{token}");
    let bridge_container = format!("sloth-e2e-bridge-mcp-{token}");

    let mm_image = std::env::var("E2E_MM_IMAGE")
        .unwrap_or_else(|_| "mattermost/mattermost-preview:latest".to_string());
    let mm_port: u16 = std::env::var("E2E_BRIDGE_PORT_MM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18067);
    let bridge_port: u16 = std::env::var("E2E_BRIDGE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(19501);

    let _ = Command::new("docker")
        .args(["network", "rm", &net])
        .output();
    docker_ok(&["network", "create", &net]).context("create docker network")?;

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

/// The host port the bridge publishes its WS on (default 19501).
pub fn bridge_port() -> u16 {
    std::env::var("E2E_BRIDGE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(19501)
}

/// Wait for a URL to answer `GET /api/v4/system/ping` with 200.
pub async fn wait_for_mattermost(url: &str) -> Result<()> {
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

/// Wait for the bridge to answer its spec endpoint.
pub async fn wait_for_bridge(host_port: u16) -> Result<()> {
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
pub fn start_bridge_container(
    stack: &Stack,
    bot_token: &str,
    mm_url: &str,
    host_port: u16,
) -> Result<()> {
    let cfg = json!({
        "server": { "host": "0.0.0.0", "port": 9300, "path": "/bridge" },
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
                        "network": { "dangerouslyAllowPrivateNetwork": true }
                    }
                }
            }
        },
        "logging": { "level": "info" }
    });
    let cfg_bytes = serde_json::to_vec(&cfg).context("serialize bridge config")?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&cfg_bytes);
    let startup = format!(
        "echo {b64} | base64 -d > /home/openclaw/config.json && exec node /app/dist/server.js --config /home/openclaw/config.json"
    );
    let port = format!("{host_port}:9300");
    docker_ok(&[
        "run",
        "-d",
        "--name",
        &stack.bridge_container,
        "--network",
        &stack.net,
        "-p",
        &port,
        &stack_image(),
        "sh",
        "-c",
        &startup,
    ])
    .context("start bridge container")?;
    Ok(())
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
pub struct MmUser {
    pub token: String,
    pub user_id: String,
}

#[derive(Clone)]
pub struct Provisions {
    pub bot: MmUser,
    pub sender: MmUser,
}

/// Provision a System Admin + bot + sender on a fresh mattermost-preview.
pub async fn provision(stack: &Stack) -> Result<Provisions> {
    let url = stack.mm_host_url.clone();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let reg = client
        .post(format!("{}/api/v4/users", url))
        .json(&json!({ "email": ADMIN_EMAIL, "username": ADMIN_USER, "password": ADMIN_PASS }))
        .send()
        .await?;
    let sc = reg.status().as_u16();
    if sc != 201 && sc != 400 {
        bail!("admin signup returned {sc}");
    }

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
    ])?;

    docker_ok(&["restart", &stack.mm_container])?;
    wait_for_mattermost(&url).await?;

    let admin_token = mm_login(&client, &url, ADMIN_USER, ADMIN_PASS).await?;
    let auth = format!("Bearer {admin_token}");

    ensure_user(&client, &url, &auth, SENDER_EMAIL, SENDER_USER, SENDER_PASS).await?;
    let sender_id = mm_user_id_by_username(&client, &url, &auth, SENDER_USER).await?;
    let sender_token = mm_mint_pat(&client, &url, &auth, &sender_id, "e2e sender PAT").await?;
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
pub struct MattermostUser {
    base_url: String,
    token: String,
    pub my_user_id: String,
    bot_user_id: String,
    client: reqwest::Client,
    /// Posts authored by the bot, received on the sender's event stream.
    pub received: Arc<Mutex<Vec<(String, String)>>>, // (post_id, text)
    dm_channel: Arc<Mutex<Option<String>>>,
    ws_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MattermostUser {
    pub fn new(base_url: String, token: String, my_user_id: String, bot_user_id: String) -> Self {
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

    /// Open the WS event stream, authenticate the PAT, resolve once the server
    /// responds with `hello`/OK. Continues draining events in a task.
    pub async fn connect(&mut self) -> Result<()> {
        let ws_url = self
            .base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/api/v4/websocket";
        let (ws, _) = tokio_tungstenite::connect_async(&ws_url)
            .await
            .map_err(|e| anyhow!("mattermost ws connect failed: {e}"))?;
        let (mut sink, mut stream) = ws.split();
        let auth = json!({ "seq": 1, "action": "authentication_challenge", "data": { "token": self.token } });
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
        match tokio::time::timeout(Duration::from_secs(15), hello_rx).await {
            Ok(Ok(())) => Ok(()),
            _ => bail!("mattermost WS never sent hello / auth OK"),
        }
    }

    /// Create (or look up) the DM channel between this user and the bot.
    pub async fn ensure_dm(&self) -> Result<String> {
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
    pub async fn post_to_bot(&self, text: &str) -> Result<()> {
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

/// Poll until the sender has received a *fresh* bot-authored post — one whose
/// id is not already in `seen`, whose text is non-empty (and optionally not
/// matching an excluded string) — recording its id in `seen` and returning the
/// text. Requiring a new post id guarantees each observation is its own reply.
pub async fn wait_for_fresh_reply(
    sender: &MattermostUser,
    exclude: &str,
    seen: &mut std::collections::HashSet<String>,
) -> Result<String> {
    loop {
        let received = sender.received.lock().await.clone();
        if let Some((id, text)) = received
            .iter()
            .rev()
            .find(|(id, t)| !t.is_empty() && t != exclude && !seen.contains(id))
        {
            seen.insert(id.clone());
            return Ok(text.clone());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

// ─── Mock MCP server (Streamable HTTP, recording calls) ──────────────────────
//
// A mock MCP server speaking the Streamable HTTP transport over a hand-rolled
// HTTP/1.1 parser. It exposes one tool, `echo`, and records every call so a
// test can assert the tool was actually invoked (e.g. from a scheduled job).

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Option<(String, Vec<u8>)> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).to_string();
            let body_start = end + 4;
            let clen = head
                .to_lowercase()
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let mut body = buf[body_start..].to_vec();
            while body.len() < clen {
                let n = stream.read(&mut tmp).await.ok()?;
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            body.truncate(clen);
            return Some((head, body));
        }
        if buf.len() > 1024 * 1024 {
            return None;
        }
    }
}

async fn write_response(stream: &mut tokio::net::TcpStream, ct: &str, body: &[u8]) {
    use tokio::io::AsyncWriteExt;
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.flush().await;
}

/// A handle to a running mock MCP server: its base URL plus access to the list
/// of `message` arguments passed to its `echo` tool.
#[derive(Clone)]
pub struct MockMcp {
    pub url: String,
    pub calls: Arc<Mutex<Vec<String>>>,
}

/// Start a mock MCP server whose `echo` tool records each `message` argument.
/// Returns its base URL and the shared call log.
pub async fn start_mock_mcp_recording() -> MockMcp {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let calls_task = calls.clone();
    tokio::spawn(async move {
        loop {
            let (mut stream, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let calls = calls_task.clone();
            tokio::spawn(async move {
                while let Some((head, body)) = read_http_request(&mut stream).await {
                    let req_line = head.lines().next().unwrap_or("");
                    if !req_line.starts_with("POST") {
                        write_response(&mut stream, "application/json", b"{}").await;
                        return;
                    }
                    let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
                    match method {
                        "initialize" => {
                            let resp = json!({
                                "jsonrpc": "2.0", "id": v["id"], "result": {
                                    "protocolVersion": "2024-11-05", "capabilities": {},
                                    "serverInfo": { "name": "mock-mcp-a", "version": "0.0.1" },
                                }
                            });
                            write_response(
                                &mut stream,
                                "application/json",
                                serde_json::to_vec(&resp).unwrap().as_slice(),
                            )
                            .await;
                        }
                        "notifications/initialized" => {
                            write_response(&mut stream, "application/json", b"{}").await;
                            return;
                        }
                        "tools/list" => {
                            let resp = json!({
                                "jsonrpc": "2.0", "id": v["id"], "result": { "tools": [{
                                    "name": "echo",
                                    "description": "Echo back the given message verbatim.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": { "message": { "type": "string" } },
                                        "required": ["message"]
                                    }
                                }] }
                            });
                            write_response(
                                &mut stream,
                                "application/json",
                                serde_json::to_vec(&resp).unwrap().as_slice(),
                            )
                            .await;
                        }
                        "tools/call" => {
                            let name = v["params"]["name"].as_str().unwrap_or("");
                            let msg = v["params"]["arguments"]["message"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            if name == "echo" {
                                calls.lock().await.push(msg.clone());
                                let result = json!({
                                    "content": [{ "type": "text", "text": format!("echo:{msg}") }],
                                    "isError": false,
                                });
                                let resp =
                                    json!({ "jsonrpc": "2.0", "id": v["id"], "result": result });
                                let data =
                                    format!("data: {}\n\n", serde_json::to_string(&resp).unwrap());
                                write_response(&mut stream, "text/event-stream", data.as_bytes())
                                    .await;
                            } else {
                                let resp = json!({ "jsonrpc": "2.0", "id": v["id"], "result": { "content": [{ "type": "text", "text": "unknown tool" }], "isError": true } });
                                write_response(
                                    &mut stream,
                                    "application/json",
                                    serde_json::to_vec(&resp).unwrap().as_slice(),
                                )
                                .await;
                            }
                        }
                        _ => {
                            let resp = json!({ "jsonrpc": "2.0", "id": v["id"], "error": { "code": -32601, "message": "method not found" } });
                            write_response(
                                &mut stream,
                                "application/json",
                                serde_json::to_vec(&resp).unwrap().as_slice(),
                            )
                            .await;
                        }
                    }
                }
            });
        }
    });
    MockMcp {
        url: format!("http://127.0.0.1:{port}/mcp"),
        calls,
    }
}
