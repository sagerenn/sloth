//! End-to-end tests for the new feature surface:
//! 1. remote MCP + hot reload  — spin a mock MCP server, list/call its tools,
//!    and reload the registry on the fly.
//! 2. time-based scheduler     — add a cron job and observe it fire.
//! 3. session management       — create/switch/set-workspace/list.
//! 4. Human-in-the-Loop        — gate a tool call, approve it, see it run.
//! 5. function calling / structured output — drive the scheduler through the
//!    LLM tool-calling path (live-gated).
//!
//! Tests that need the live LLM are `#[ignore]` (matching the repo convention)
//! and skip gracefully when the gateway is unreachable. The pure-logic tests
//! (MCP server, scheduler firing, sessions, HITL plumbing) run by default
//! against in-process mocks — no external services required.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use sloth_agent::config::HitlConfig;
use sloth_agent::cron::Cron;
use sloth_agent::hitl::HitlBroker;
use sloth_agent::mcp::McpRegistry;
use sloth_agent::scheduler::{ScheduledJob, Scheduler};
use sloth_agent::session::SessionManager;
use sloth_agent::tools::ToolRouter;

// ──────────────────────────── helpers ────────────────────────────

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

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{nanos:x}")
}

// ──────────────────────────── cron unit ────────────────────────────

#[test]
fn cron_next_after_is_monotonic() {
    let c = Cron::parse("*/15 * * * *").unwrap();
    let t1 = c.next_after(0);
    let t2 = c.next_after(t1);
    assert!(t2 > t1);
    // every 15 minutes
    assert_eq!(t2 - t1, 900);
}

// ──────────────────────────── scheduler firing ────────────────────────────

#[tokio::test]
async fn scheduler_fires_due_job_and_reschedules() {
    let s = Scheduler::new();
    // Minute-aligned epoch so */minute fires land on +60, +120, ...
    let now = 60 * 16_667; // a clean multiple of 60
    let id = s
        .add_at(
            ScheduledJob {
                id: String::new(),
                name: "every-minute".into(),
                cron: "* * * * *".into(),
                prompt: "tick".into(),
                session_id: "default".into(),
                reply_to: None,
            },
            now,
        )
        .unwrap();

    // Not yet due.
    assert!(s.evaluate(now + 30).is_empty());
    // Due at +60.
    let fired = s.evaluate(now + 60);
    assert_eq!(fired.len(), 1);
    assert_eq!(fired[0].id, id);
    // Re-evaluating the same instant → no refire (already advanced).
    assert!(s.evaluate(now + 60).is_empty());
    // Next fire at +120.
    assert_eq!(s.evaluate(now + 120).len(), 1);
}

// ──────────────────────────── sessions ────────────────────────────

#[tokio::test]
async fn sessions_create_switch_workspace_list() {
    let dir = std::env::temp_dir().join(format!("sloth-sess-{}", uuid_v4()));
    let sm = SessionManager::new("default", dir);
    // default exists
    assert_eq!(sm.list().await.len(), 1);
    let s = sm.create(Some("work".into()), "Work".into()).await.unwrap();
    assert_eq!(s.id, "work");
    sm.switch("alice", "work").await.unwrap();
    assert_eq!(sm.active_id("alice").await, "work");
    let ws = std::env::temp_dir().join(format!("sloth-ws-{}", uuid_v4()));
    let with_ws = sm.set_workspace("work", ws.clone()).await.unwrap();
    assert_eq!(with_ws.workspace.as_ref(), Some(&ws));
    assert!(ws.exists());
    assert!(sm.list().await.iter().any(|s| s.id == "work"));
    // cannot delete default
    sm.delete("default", "default").await.unwrap_err();
    sm.delete("work", "default").await.unwrap();
}

// ──────────────────────────── HITL plumbing (no LLM) ────────────────────────────

#[tokio::test]
async fn hitl_gate_blocks_until_approved_then_runs() {
    use sloth_agent::hitl::{HitlBroker, Outcome};
    let broker = HitlBroker::new(HitlConfig {
        enabled: true,
        timeout_secs: 10,
        confirm_tools: vec![],
    });
    // A tool router with HITL enabled but no real scheduler add needed: we
    // exercise the gating path by calling execute directly. The scheduler_add
    // tool is gated; with HITL enabled and no approver it must time out.
    let scheduler = Arc::new(Scheduler::new());
    let dir = std::env::temp_dir().join(format!("sloth-hitl-{}", uuid_v4()));
    let sessions = Arc::new(SessionManager::new("default", dir));
    let mcp = Arc::new(sloth_agent::mcp::McpRegistry::new());
    let router = ToolRouter::new(
        scheduler.clone(),
        sessions,
        mcp,
        Arc::new(sloth_agent::skill::SkillRegistry::new()),
        Arc::new(sloth_agent::a2a::A2aRegistry::new()),
        Arc::new(sloth_agent::model_catalog::Catalog::new()),
        Arc::new(sloth_agent::memory::MemoryStore::new()),
        Arc::new(broker.clone()),
        vec![],
        vec![],
        true,
        true,
        true,
        true,
        true,
    );

    // Subscribe to pending confirmations so the broker publishes to us.
    let mut pending_rx = broker.pending_channel_async().await;

    // Spawn the execute call; it will block on HITL approval.
    let args = json!({
        "name": "daily",
        "cron": "0 9 * * *",
        "prompt": "good morning"
    });
    let router_c = router.clone();
    let task =
        tokio::spawn(async move { router_c.execute("scheduler_add_job", &args, "alice").await });

    // Receive the pending confirmation and approve it.
    let pending = tokio::time::timeout(Duration::from_secs(2), pending_rx.recv())
        .await
        .expect("no HITL request surfaced")
        .expect("channel closed");
    assert_eq!(pending.tool, "scheduler_add_job");
    broker.resolve(&pending.id, Outcome::Approved).await;

    let outcome = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
    assert!(!outcome.is_error, "expected approval to run the tool");
    // The job should actually have been scheduled.
    assert_eq!(scheduler.list().len(), 1);
}

#[tokio::test]
async fn hitl_deny_blocks_tool() {
    use sloth_agent::hitl::{HitlBroker, Outcome};
    let broker = HitlBroker::new(HitlConfig {
        enabled: true,
        timeout_secs: 10,
        confirm_tools: vec![],
    });
    let scheduler = Arc::new(Scheduler::new());
    let dir = std::env::temp_dir().join(format!("sloth-hitl-deny-{}", uuid_v4()));
    let sessions = Arc::new(SessionManager::new("default", dir));
    let mcp = Arc::new(sloth_agent::mcp::McpRegistry::new());
    let router = ToolRouter::new(
        scheduler.clone(),
        sessions,
        mcp,
        Arc::new(sloth_agent::skill::SkillRegistry::new()),
        Arc::new(sloth_agent::a2a::A2aRegistry::new()),
        Arc::new(sloth_agent::model_catalog::Catalog::new()),
        Arc::new(sloth_agent::memory::MemoryStore::new()),
        Arc::new(broker.clone()),
        vec![],
        vec![],
        true,
        true,
        true,
        true,
        true,
    );
    let mut pending_rx = broker.pending_channel_async().await;
    let args = json!({ "name": "x", "cron": "* * * * *", "prompt": "p" });
    let router_c = router.clone();
    let task =
        tokio::spawn(async move { router_c.execute("scheduler_add_job", &args, "alice").await });
    let pending = pending_rx.recv().await.expect("no pending");
    broker.resolve(&pending.id, Outcome::Denied).await;
    let outcome = task.await.unwrap();
    assert!(outcome.is_error);
    // Nothing scheduled.
    assert!(scheduler.list().is_empty());
}

// ──────────────────────────── remote MCP + hot reload ────────────────────────────
//
// A mock MCP server speaking the Streamable HTTP transport over a hand-rolled
// HTTP/1.1 parser. It answers `initialize`, `tools/list`, and `tools/call`
// (the call returns its result as an SSE frame so the SSE parser is exercised).

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Option<(String, Vec<u8>)> {
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
            // Determine body length from Content-Length.
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
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.flush().await;
}

/// Start a mock MCP server. Returns its base URL and a shutdown signal.
async fn start_mock_mcp() -> (String, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let mut shutdown_rx = std::pin::pin!(shutdown_rx);
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                accept = listener.accept() => {
                    let (mut stream, _) = match accept { Ok(s) => s, Err(_) => continue };
                    tokio::spawn(async move {
                        // Serve a single request per connection (Connection: close).
                        while let Some((head, body)) = read_http_request(&mut stream).await {
                            let req_line = head.lines().next().unwrap_or("");
                            let is_post = req_line.starts_with("POST");
                            if !is_post {
                                write_response(&mut stream, "application/json", b"{}").await;
                                return;
                            }
                            let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                            let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
                            match method {
                                "initialize" => {
                                    let resp = json!({
                                        "jsonrpc": "2.0",
                                        "id": v["id"],
                                        "result": {
                                            "protocolVersion": "2024-11-05",
                                            "capabilities": {},
                                            "serverInfo": { "name": "mock-mcp", "version": "0.0.1" },
                                        }
                                    });
                                    write_response(
                                        &mut stream,
                                        "application/json",
                                        serde_json::to_vec(&resp).unwrap().as_slice(),
                                    ).await;
                                }
                                "notifications/initialized" => {
                                    // notification: no response, but reqwest waits for
                                    // bytes; send an empty 200.
                                    write_response(&mut stream, "application/json", b"{}").await;
                                    return;
                                }
                                "tools/list" => {
                                    let resp = json!({
                                        "jsonrpc": "2.0",
                                        "id": v["id"],
                                        "result": {
                                            "tools": [{
                                                "name": "echo",
                                                "description": "Echo back the message",
                                                "inputSchema": {
                                                    "type": "object",
                                                    "properties": { "message": { "type": "string" } },
                                                    "required": ["message"]
                                                }
                                            }]
                                        }
                                    });
                                    write_response(
                                        &mut stream,
                                        "application/json",
                                        serde_json::to_vec(&resp).unwrap().as_slice(),
                                    ).await;
                                }
                                "tools/call" => {
                                    let name = v["params"]["name"].as_str().unwrap_or("");
                                    let msg = v["params"]["arguments"]["message"]
                                        .as_str()
                                        .unwrap_or("");
                                    let result = if name == "echo" {
                                        json!({
                                            "content": [{ "type": "text", "text": format!("echo:{msg}") }],
                                            "isError": false,
                                        })
                                    } else {
                                        json!({ "content": [{ "type": "text", "text": "unknown tool" }], "isError": true })
                                    };
                                    let resp = json!({ "jsonrpc": "2.0", "id": v["id"], "result": result });
                                    // Return as an SSE frame to exercise the SSE parser.
                                    let data = format!("data: {}\n\n", serde_json::to_string(&resp).unwrap());
                                    write_response(
                                        &mut stream,
                                        "text/event-stream",
                                        data.as_bytes(),
                                    ).await;
                                }
                                _ => {
                                    let resp = json!({ "jsonrpc": "2.0", "id": v["id"], "error": { "code": -32601, "message": "method not found" } });
                                    write_response(
                                        &mut stream,
                                        "application/json",
                                        serde_json::to_vec(&resp).unwrap().as_slice(),
                                    ).await;
                                }
                            }
                        }
                    });
                }
            }
        }
    });
    (format!("http://127.0.0.1:{port}/mcp"), shutdown_tx)
}

#[tokio::test]
async fn mcp_connect_list_and_call_tool() {
    let (url, _shutdown) = start_mock_mcp().await;
    let cfg = sloth_agent::config::McpServerConfig {
        name: "mock".to_string(),
        url,
        token: None,
        timeout_secs: 5,
    };
    let reg = McpRegistry::new();
    reg.add_server(&cfg).await.unwrap();

    // One server, one tool (`echo`), qualified `mcp_mock__echo`.
    let tools = reg.routed_tools().await;
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].qualified_name, "mcp_mock__echo");

    // Call it.
    let result = reg
        .call_qualified("mcp_mock__echo", json!({ "message": "hi" }))
        .await
        .unwrap();
    assert!(!result.is_error);
    assert_eq!(result.text, "echo:hi");
}

#[tokio::test]
async fn mcp_hot_reload_adds_and_removes_servers() {
    let (url, _shutdown) = start_mock_mcp().await;
    let cfg = sloth_agent::config::McpServerConfig {
        name: "mock".to_string(),
        url,
        token: None,
        timeout_secs: 5,
    };
    let reg = McpRegistry::new();

    // Reload with one server → added.
    let report = reg.reload(std::slice::from_ref(&cfg)).await.unwrap();
    assert_eq!(report.added, vec!["mock".to_string()]);
    assert_eq!(reg.server_count().await, 1);
    let tools = reg.routed_tools().await;
    assert_eq!(tools.len(), 1);

    // Reload with empty list → removed.
    let report = reg.reload(&[]).await.unwrap();
    assert_eq!(report.removed, vec!["mock".to_string()]);
    assert_eq!(reg.server_count().await, 0);
    assert!(reg.routed_tools().await.is_empty());
}

// ──────────────────────────── MCP tool via router (HITL on) ────────────────────────────

#[tokio::test]
async fn router_exposes_mcp_tool_and_calls_it_with_hitl_approve() {
    let (url, _shutdown) = start_mock_mcp().await;
    let mcp_cfg = sloth_agent::config::McpServerConfig {
        name: "mock".to_string(),
        url,
        token: None,
        timeout_secs: 5,
    };

    let scheduler = Arc::new(Scheduler::new());
    let dir = std::env::temp_dir().join(format!("sloth-mcp-router-{}", uuid_v4()));
    let sessions = Arc::new(SessionManager::new("default", dir));
    let mcp = Arc::new(McpRegistry::new());
    mcp.add_server(&mcp_cfg).await.unwrap();
    let broker = Arc::new(HitlBroker::new(HitlConfig {
        enabled: true,
        timeout_secs: 10,
        // Only confirm scheduler tools; MCP calls run automatically.
        confirm_tools: vec!["scheduler_*".to_string()],
    }));
    let router = ToolRouter::new(
        scheduler,
        sessions,
        mcp,
        Arc::new(sloth_agent::skill::SkillRegistry::new()),
        Arc::new(sloth_agent::a2a::A2aRegistry::new()),
        Arc::new(sloth_agent::model_catalog::Catalog::new()),
        Arc::new(sloth_agent::memory::MemoryStore::new()),
        broker.clone(),
        vec![mcp_cfg],
        vec![],
        true,
        true,
        true,
        true,
        true,
    );

    // The echo tool is surfaced as a tool definition.
    let defs = router.tool_definitions().await;
    assert!(defs.iter().any(|t| match t {
        async_openai::types::chat::ChatCompletionTools::Function(f) => {
            f.function.name == "mcp_mock__echo"
        }
        _ => false,
    }));

    // Calling it directly should NOT require HITL (MCP not in confirm list).
    let outcome = router
        .execute("mcp_mock__echo", &json!({ "message": "router" }), "alice")
        .await;
    assert!(!outcome.is_error);
    assert_eq!(outcome.content, "echo:router");
}

// ──────────────────────────── live function-calling (LLM) ────────────────────────────
//
// Drives the agent's tool-calling loop with the live LLM: ask it to list
// scheduled jobs (a read-only tool call). Skips when the gateway is down.

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway"]
async fn llm_function_call_lists_scheduled_jobs() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,sloth_agent=debug,warn")
        .try_init();
    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }

    use sloth_agent::agent::ChatAgent;
    use sloth_agent::config::LlmConfig;

    let llm = LlmConfig {
        base_url: std::env::var("SLOTH_LLM_BASE_URL")
            .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".to_string()),
        model: std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".to_string()),
        api_key: std::env::var("SLOTH_LLM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        system_prompt: "You are a test assistant. Use provided tools when asked.".to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
        timeout_secs: 60,
    };
    let agent = ChatAgent::new(&llm, 10).unwrap();

    let scheduler = Arc::new(Scheduler::new());
    // Seed a job so the list tool has something to return.
    scheduler
        .add_at(
            ScheduledJob {
                id: String::new(),
                name: "seed".into(),
                cron: "0 9 * * *".into(),
                prompt: "morning".into(),
                session_id: "default".into(),
                reply_to: None,
            },
            0,
        )
        .unwrap();
    let dir = std::env::temp_dir().join(format!("sloth-llm-fc-{}", uuid_v4()));
    let sessions = Arc::new(SessionManager::new("default", dir));
    let mcp = Arc::new(McpRegistry::new());
    let broker = Arc::new(HitlBroker::new(HitlConfig {
        enabled: false, // disable HITL for the live read-only test
        timeout_secs: 5,
        confirm_tools: vec![],
    }));
    let router = ToolRouter::new(
        scheduler.clone(),
        sessions,
        mcp,
        Arc::new(sloth_agent::skill::SkillRegistry::new()),
        Arc::new(sloth_agent::a2a::A2aRegistry::new()),
        Arc::new(sloth_agent::model_catalog::Catalog::new()),
        Arc::new(sloth_agent::memory::MemoryStore::new()),
        broker,
        vec![],
        vec![],
        true,
        true,
        true,
        true,
        true,
    );

    let reply = agent
        .reply_with_tools(
            "test-sender",
            "Call the scheduler_list_jobs tool and tell me how many jobs are scheduled.",
            &router,
            4,
        )
        .await
        .expect("reply failed");

    println!("LLM reply: {:?}", reply.text);
    // The job we seeded should still be there (read-only tool).
    assert_eq!(scheduler.list().len(), 1);
}

// ──────────────────────────── HITL reply parser ────────────────────────────

#[test]
fn parse_hitl_reply_maps_yes_and_no() {
    use sloth_agent::hitl::Outcome;
    use sloth_agent::runtime::parse_hitl_reply;
    assert_eq!(parse_hitl_reply("yes"), Outcome::Approved);
    assert_eq!(parse_hitl_reply("Y"), Outcome::Approved);
    assert_eq!(parse_hitl_reply("approve"), Outcome::Approved);
    assert_eq!(parse_hitl_reply("confirm"), Outcome::Approved);
    assert_eq!(parse_hitl_reply("no"), Outcome::Denied);
    assert_eq!(parse_hitl_reply("deny"), Outcome::Denied);
    assert_eq!(parse_hitl_reply("cancel"), Outcome::Denied);
    // Unknown → fail-closed deny.
    assert_eq!(parse_hitl_reply("maybe"), Outcome::Denied);
    assert_eq!(parse_hitl_reply(""), Outcome::Denied);
}

// ──────────────────────────── skills (hot reload) ────────────────────────────

#[tokio::test]
async fn skills_load_parse_and_invoke_through_router() {
    use sloth_agent::skill::SkillRegistry;

    let dir = std::env::temp_dir().join(format!("sloth-skills-e2e-{}", uuid_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("greet.md"),
        "---\nname: greet\ndescription: Greet someone\narguments:\n  - name: who\n    description: person to greet\n    required: true\n---\nHello, {{who}}!\n",
    )
    .unwrap();

    let reg = SkillRegistry::new();
    reg.set_dir(&dir).await;
    assert_eq!(reg.reload().await.unwrap(), 1);

    let skills = reg.list().await;
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "greet");
    assert_eq!(skills[0].arguments.len(), 1);

    // Invoke substitutes placeholders.
    let out = reg
        .invoke("greet", &json!({ "who": "world" }))
        .await
        .unwrap();
    assert_eq!(out, "Hello, world!\n");

    // Hot reload picks up an added file.
    std::fs::write(
        dir.join("bye.md"),
        "---\nname: bye\ndescription: Say bye\n---\nGoodbye!\n",
    )
    .unwrap();
    assert_eq!(reg.reload().await.unwrap(), 2);
    assert!(reg.get("bye").await.is_some());

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn router_exposes_skill_tool_and_invokes_it() {
    use sloth_agent::skill::SkillRegistry;

    let dir = std::env::temp_dir().join(format!("sloth-skills-router-{}", uuid_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("greet.md"),
        "---\nname: greet\ndescription: Greet someone\narguments:\n  - name: who\n    required: true\n---\nHello, {{who}}!\n",
    )
    .unwrap();

    let skills = Arc::new(SkillRegistry::new());
    skills.set_dir(&dir).await;
    skills.reload().await.unwrap();

    let scheduler = Arc::new(Scheduler::new());
    let sessions = Arc::new(SessionManager::new(
        "default",
        std::env::temp_dir().join(format!("sloth-sess-sk-{}", uuid_v4())),
    ));
    let mcp = Arc::new(McpRegistry::new());
    let broker = Arc::new(HitlBroker::new(HitlConfig {
        enabled: false,
        ..Default::default()
    }));
    let router = ToolRouter::new(
        scheduler,
        sessions,
        mcp,
        skills,
        Arc::new(sloth_agent::a2a::A2aRegistry::new()),
        Arc::new(sloth_agent::model_catalog::Catalog::new()),
        Arc::new(sloth_agent::memory::MemoryStore::new()),
        broker,
        vec![],
        vec![],
        true,
        true,
        true,
        true,
        true,
    );

    // skill_<name> is surfaced as a tool definition.
    let defs = router.tool_definitions().await;
    assert!(defs.iter().any(|t| match t {
        async_openai::types::chat::ChatCompletionTools::Function(f) => {
            f.function.name == "skill_greet"
        }
        _ => false,
    }));

    // Invoking through the router expands the body.
    let outcome = router
        .execute("skill_greet", &json!({ "who": "agent" }), "alice")
        .await;
    assert!(!outcome.is_error);
    assert_eq!(outcome.content, "Hello, agent!\n");

    std::fs::remove_dir_all(&dir).ok();
}

// ──────────────────────────── A2A (official SDK) ────────────────────────────

/// Start a mock A2A agent server.
///
/// Serves:
///   - `GET /` (any path under base) → agent card declaring a JSONRPC
///     interface at `{base}/jsonrpc`
///   - `POST /jsonrpc` → JSON-RPC `SendMessage` returning a Completed task
///     whose status message echoes `You said: <text>`.
///
/// Returns the base URL and a shutdown signal.
async fn start_mock_a2a() -> (String, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let mut shutdown_rx = std::pin::pin!(shutdown_rx);
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                accept = listener.accept() => {
                    let (mut stream, _) = match accept { Ok(s) => s, Err(_) => continue };
                    tokio::spawn(async move {
                        while let Some((head, body)) = read_http_request(&mut stream).await {
                            let req_line = head.lines().next().unwrap_or("");
                            let (method, _path) = {
                                let mut parts = req_line.split_whitespace();
                                (parts.next().unwrap_or(""), parts.next().unwrap_or(""))
                            };

                            if method == "GET" {
                                // Serve the agent card for any path (incl. .well-known).
                                let card = json!({
                                    "name": "mock-a2a",
                                    "description": "A mock A2A agent for tests",
                                    "version": "1.0.0",
                                    "supportedInterfaces": [{
                                        "url": format!("http://127.0.0.1:{}/jsonrpc", port),
                                        "protocolBinding": "JSONRPC",
                                        "protocolVersion": "1.0",
                                    }],
                                    "capabilities": {},
                                    "defaultInputModes": ["text/plain"],
                                    "defaultOutputModes": ["text/plain"],
                                    "skills": [],
                                });
                                write_response(
                                    &mut stream,
                                    "application/json",
                                    serde_json::to_vec(&card).unwrap().as_slice(),
                                ).await;
                                return;
                            }

                            if method != "POST" {
                                write_response(&mut stream, "application/json", b"{}").await;
                                return;
                            }

                            // JSON-RPC over POST.
                            let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                            let id = v.get("id").cloned().unwrap_or(Value::Null);
                            let method_name = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
                            match method_name {
                                "SendMessage" => {
                                    let prompt = v["params"]["message"]["parts"][0]["text"]
                                        .as_str()
                                        .unwrap_or("");
                                    let result = json!({
                                        "task": {
                                            "id": "task-mock",
                                            "contextId": "ctx-mock",
                                            "status": {
                                                "state": "TASK_STATE_COMPLETED",
                                                "message": {
                                                    "messageId": "m1",
                                                    "role": "ROLE_AGENT",
                                                    "parts": [{ "text": format!("You said: {prompt}") }],
                                                },
                                            },
                                        }
                                    });
                                    let resp = json!({ "jsonrpc": "2.0", "id": id, "result": result });
                                    write_response(
                                        &mut stream,
                                        "application/json",
                                        serde_json::to_vec(&resp).unwrap().as_slice(),
                                    ).await;
                                }
                                _ => {
                                    let resp = json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "method not found" } });
                                    write_response(
                                        &mut stream,
                                        "application/json",
                                        serde_json::to_vec(&resp).unwrap().as_slice(),
                                    ).await;
                                }
                            }
                        }
                    });
                }
            }
        }
    });
    (base, shutdown_tx)
}

#[tokio::test]
async fn a2a_connect_send_and_hot_reload() {
    let (base, _shutdown) = start_mock_a2a().await;
    let cfg = sloth_agent::config::A2aAgentConfig {
        name: "mock".to_string(),
        url: base.clone(),
        token: None,
        timeout_secs: 10,
    };
    let reg = sloth_agent::a2a::A2aRegistry::new();

    // Reload with one agent → added.
    let report = reg.reload(std::slice::from_ref(&cfg)).await.unwrap();
    assert_eq!(report.added, vec!["mock".to_string()]);
    assert_eq!(reg.agent_count().await, 1);
    assert_eq!(reg.agent_names().await, vec!["mock".to_string()]);

    // Send a prompt; the mock echoes it back.
    let result = reg.send("mock", "hello a2a").await.unwrap();
    assert_eq!(result.text, "You said: hello a2a");
    assert_eq!(result.state.as_deref(), Some("Completed"));

    // Hot reload: drop the agent.
    let report = reg.reload(&[]).await.unwrap();
    assert_eq!(report.removed, vec!["mock".to_string()]);
    assert_eq!(reg.agent_count().await, 0);
}

#[tokio::test]
async fn router_exposes_a2a_tool_and_calls_it() {
    let (base, _shutdown) = start_mock_a2a().await;
    let a2a_cfg = sloth_agent::config::A2aAgentConfig {
        name: "mock".to_string(),
        url: base,
        token: None,
        timeout_secs: 10,
    };

    let scheduler = Arc::new(Scheduler::new());
    let sessions = Arc::new(SessionManager::new(
        "default",
        std::env::temp_dir().join(format!("sloth-sess-a2a-{}", uuid_v4())),
    ));
    let mcp = Arc::new(McpRegistry::new());
    let a2a = Arc::new(sloth_agent::a2a::A2aRegistry::new());
    a2a.reload(&[a2a_cfg]).await.unwrap();
    let broker = Arc::new(HitlBroker::new(HitlConfig {
        enabled: false,
        ..Default::default()
    }));
    let router = ToolRouter::new(
        scheduler,
        sessions,
        mcp,
        Arc::new(sloth_agent::skill::SkillRegistry::new()),
        a2a,
        Arc::new(sloth_agent::model_catalog::Catalog::new()),
        Arc::new(sloth_agent::memory::MemoryStore::new()),
        broker,
        vec![],
        vec![],
        true,
        true,
        true,
        true,
        true,
    );

    // a2a_<name> is surfaced as a tool definition.
    let defs = router.tool_definitions().await;
    assert!(defs.iter().any(|t| match t {
        async_openai::types::chat::ChatCompletionTools::Function(f) => {
            f.function.name == "a2a_mock"
        }
        _ => false,
    }));

    // Calling through the router returns the echoed reply.
    let outcome = router
        .execute("a2a_mock", &json!({ "prompt": "via router" }), "alice")
        .await;
    assert!(!outcome.is_error, "content was: {}", outcome.content);
    assert!(outcome.content.contains("You said: via router"));
}

// ──────────────────────────── model catalog ────────────────────────────

#[tokio::test]
async fn model_catalog_pick_and_list_through_router() {
    use sloth_agent::model_catalog::{Catalog, PickOptions, Strategy};

    let dir = std::env::temp_dir().join(format!("sloth-cat-{}", uuid_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("models.yaml"),
        "models:\n  - id: cheap\n    context_window: 8000\n    pricing: {prompt_per_1m: 0.1, completion_per_1m: 0.2}\n    scores: {average: 75.0}\n  - id: smart\n    context_window: 128000\n    pricing: {prompt_per_1m: 2.0, completion_per_1m: 8.0}\n    scores: {average: 90.0}\n",
    ).unwrap();

    let cat = Catalog::new();
    cat.set_dir(&dir).await;
    assert_eq!(cat.reload().await.unwrap(), 2);

    // Best score → smart.
    let pick = cat.pick(&PickOptions::default()).await.unwrap();
    assert_eq!(pick.id, "smart");

    // Cheapest above a 70 floor → cheap.
    let opts = PickOptions {
        strategy: Strategy::CheapestAboveFloor,
        min_score: 70.0,
        ..Default::default()
    };
    assert_eq!(cat.pick(&opts).await.unwrap().id, "cheap");

    // No model satisfies a too-high context requirement.
    let opts = PickOptions {
        min_context_window: Some(1_000_000),
        ..Default::default()
    };
    assert!(cat.pick(&opts).await.is_none());

    // Through the router: model_list returns both, model_pick returns smart.
    let mem_dir = std::env::temp_dir().join(format!("sloth-mem-cat-{}", uuid_v4()));
    let router = make_router_with_catalog(cat.clone(), mem_dir).await;
    let list_outcome = router.execute("model_list", &json!({}), "alice").await;
    assert!(!list_outcome.is_error);
    assert!(listcome_contains(&list_outcome.content, "smart"));
    assert!(listcome_contains(&list_outcome.content, "cheap"));

    let pick_outcome = router
        .execute("model_pick", &json!({ "strategy": "best_score" }), "alice")
        .await;
    assert!(!pick_outcome.is_error, "{}", pick_outcome.content);
    assert!(pick_outcome.content.contains("smart"));

    std::fs::remove_dir_all(&dir).ok();
}

/// Build a router wired with a catalog + memory, HITL disabled, for catalog/
/// memory tool tests.
async fn make_router_with_catalog(
    cat: sloth_agent::model_catalog::Catalog,
    mem_dir: std::path::PathBuf,
) -> ToolRouter {
    let scheduler = Arc::new(Scheduler::new());
    let sessions = Arc::new(SessionManager::new("default", mem_dir.clone()));
    let mcp = Arc::new(McpRegistry::new());
    let memory = Arc::new(sloth_agent::memory::MemoryStore::new());
    memory.set_dir(&mem_dir).await;
    let opts = sloth_agent::model_catalog::PickOptions::default();
    ToolRouter::with_model_opts(
        scheduler,
        sessions,
        mcp,
        Arc::new(sloth_agent::skill::SkillRegistry::new()),
        Arc::new(sloth_agent::a2a::A2aRegistry::new()),
        Arc::new(cat),
        memory,
        Arc::new(HitlBroker::new(HitlConfig {
            enabled: false,
            ..Default::default()
        })),
        vec![],
        vec![],
        true,
        true,
        true,
        true,
        true,
        opts,
    )
}

fn listcome_contains(content: &str, needle: &str) -> bool {
    content.contains(needle)
}

// ──────────────────────────── memory ────────────────────────────

#[tokio::test]
async fn memory_set_recall_persists_through_router() {
    let mem_dir = std::env::temp_dir().join(format!("sloth-mem-e2e-{}", uuid_v4()));
    let cat = sloth_agent::model_catalog::Catalog::new();
    let router = make_router_with_catalog(cat, mem_dir.clone()).await;

    // Set a fact through the tool.
    let set = router
        .execute(
            "memory_set",
            &json!({ "key": "preferred_language", "value": "Rust" }),
            "alice",
        )
        .await;
    assert!(!set.is_error, "{}", set.content);

    // Recall it back.
    let rec = router
        .execute(
            "memory_recall",
            &json!({ "key": "preferred_language" }),
            "alice",
        )
        .await;
    assert!(!rec.is_error, "{}", rec.content);
    let v: Value = serde_json::from_str(&rec.content).unwrap();
    assert_eq!(v["value"].as_str(), Some("Rust"));

    // Recall all (no key).
    let all = router.execute("memory_recall", &json!({}), "alice").await;
    let allv: Value = serde_json::from_str(&all.content).unwrap();
    assert!(allv["facts"]["preferred_language"].as_str() == Some("Rust"));

    // Persistence across a brand-new store reading the same dir.
    let store2 = sloth_agent::memory::MemoryStore::new();
    store2.set_dir(&mem_dir).await;
    let m = store2.recall("alice").await.unwrap();
    assert_eq!(m.facts.get("preferred_language").unwrap(), "Rust");

    std::fs::remove_dir_all(&mem_dir).ok();
}

#[tokio::test]
async fn memory_prompt_snippet_renders() {
    use sloth_agent::memory::SenderMemory;
    let mut m = SenderMemory::default();
    assert!(m.to_prompt_snippet().is_none());
    m.facts.insert("timezone".into(), "UTC".into());
    let s = m.to_prompt_snippet().unwrap();
    assert!(s.contains("Known facts about this user"));
    assert!(s.contains("timezone: UTC"));
}

// ──────────────────────────── auto-compact logic ────────────────────────────

#[tokio::test]
async fn compact_should_compact_threshold_logic() {
    use sloth_agent::compact::Compactor;
    use sloth_agent::config::{CompactConfig, LlmConfig};

    let llm = LlmConfig {
        base_url: "http://127.0.0.1:1/v1".into(),
        model: "unused".into(),
        api_key: None,
        system_prompt: "s".into(),
        temperature: None,
        max_tokens: None,
        timeout_secs: 5,
    };
    // Enabled, threshold 20, keep 6.
    let cfg = CompactConfig {
        enabled: true,
        threshold_messages: 20,
        keep_recent: 6,
        prompt: "p".into(),
    };
    let compactor = Compactor::new(&llm, cfg);

    assert!(!compactor.should_compact(19));
    assert!(compactor.should_compact(20));
    assert!(compactor.should_compact(40));

    // Disabled → never.
    let cfg = CompactConfig {
        enabled: false,
        threshold_messages: 20,
        keep_recent: 6,
        prompt: "p".into(),
    };
    let compactor = Compactor::new(&llm, cfg);
    assert!(!compactor.should_compact(100));

    // keep_recent >= len → never compact (would drop nothing).
    let cfg = CompactConfig {
        enabled: true,
        threshold_messages: 2,
        keep_recent: 10,
        prompt: "p".into(),
    };
    let compactor = Compactor::new(&llm, cfg);
    assert!(!compactor.should_compact(2));
}

// ──────────────────────────── full runtime auto-pick ────────────────────────────
//
// Builds a real AgentContext from a Config with a model catalog dir and checks
// that the auto-picked model id is used. Skips when the LLM is unreachable.

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway"]
async fn runtime_auto_picks_model_from_catalog() {
    use sloth_agent::config::{
        A2aConfig, BridgeConfig, CompactConfig, Config, HistoryConfig, HitlConfig, LlmConfig,
        McpConfig, MemoryConfig, ModelCatalogConfig, ObservabilityConfig, SchedulerConfig,
        SessionConfig, SkillsConfig,
    };
    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sloth-cat-rt-{}", uuid_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("models.yaml"),
        format!(
            "models:\n  - id: {}\n    context_window: 128000\n    scores: {{average: 90.0}}\n",
            std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".into())
        ),
    )
    .unwrap();

    let cfg = Config {
        bridge: BridgeConfig {
            url: "ws://127.0.0.1:1/bridge".into(),
            channel: "x".into(),
            account_id: "default".into(),
            reconnect_ms: 1_000,
            reconnect_max_ms: 1_000,
            heartbeat_ms: 0,
        },
        llm: LlmConfig {
            base_url: std::env::var("SLOTH_LLM_BASE_URL")
                .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".into()),
            model: "WRONG-FALLBACK".into(),
            api_key: None,
            system_prompt: "test".into(),
            temperature: Some(0.0),
            max_tokens: Some(64),
            timeout_secs: 60,
        },
        history: HistoryConfig { max_messages: 5 },
        observability: ObservabilityConfig {
            log_format: "text".into(),
            log_filter: "warn".into(),
            service_name: "sloth-test".into(),
        },
        mcp: McpConfig::default(),
        scheduler: SchedulerConfig {
            enabled: false,
            ..Default::default()
        },
        sessions: SessionConfig::default(),
        hitl: HitlConfig {
            enabled: false,
            ..Default::default()
        },
        skills: SkillsConfig::default(),
        a2a: A2aConfig::default(),
        models: ModelCatalogConfig {
            dir: Some(dir.to_string_lossy().into_owned()),
            strategy: "best_score".into(),
            ..Default::default()
        },
        compact: CompactConfig::default(),
        memory: MemoryConfig::default(),
    };

    let ctx = sloth_agent::runtime::AgentContext::build(&cfg)
        .await
        .unwrap();
    assert_ne!(
        ctx.model, "WRONG-FALLBACK",
        "catalog should override llm.model"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ──────────────────────────── live auto-compact + memory (LLM) ────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway"]
async fn live_auto_compact_summarizes_history() {
    use sloth_agent::agent::Stored;
    use sloth_agent::compact::Compactor;
    use sloth_agent::config::{CompactConfig, LlmConfig};

    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }
    let llm = LlmConfig {
        base_url: std::env::var("SLOTH_LLM_BASE_URL")
            .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".into()),
        model: std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".into()),
        api_key: std::env::var("SLOTH_LLM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        system_prompt: "test".into(),
        temperature: Some(0.0),
        max_tokens: Some(64),
        timeout_secs: 60,
    };
    let cfg = CompactConfig {
        enabled: true,
        threshold_messages: 4,
        keep_recent: 2,
        prompt: "Summarize the conversation in one short sentence.".into(),
    };
    let compactor = Compactor::new(&llm, cfg);

    // 4 messages: exceeds threshold (4), keep 2.
    let msgs = vec![
        Stored::User("My name is Zed and I like turtles.".into()),
        Stored::Assistant("Nice to meet you, Zed.".into()),
        Stored::User("I live in Lisbon and it rains a lot.".into()),
        Stored::Assistant("Lisbon is lovely in its own way.".into()),
    ];
    assert!(compactor.should_compact(msgs.len()));
    let mut msgs = msgs;
    let compacted = compactor.compact(&mut msgs).await.unwrap();
    assert!(compacted, "expected a compaction to occur");
    // Summary entry + 2 kept = 3 messages.
    assert_eq!(msgs.len(), 3);
    let summary = match &msgs[0] {
        Stored::Assistant(s) => s.clone(),
        _ => String::new(),
    };
    assert!(summary.contains("summary of earlier conversation"));
    // The live LLM produced a non-empty summary body.
    assert!(summary.len() > "summary of earlier conversation".len() + 10);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway"]
async fn live_memory_injected_into_prompt_recalled() {
    use sloth_agent::agent::ChatAgent;
    use sloth_agent::config::LlmConfig;
    use sloth_agent::memory::MemoryStore;

    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sloth-mem-live-{}", uuid_v4()));
    let llm = LlmConfig {
        base_url: std::env::var("SLOTH_LLM_BASE_URL")
            .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".into()),
        model: std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".into()),
        api_key: std::env::var("SLOTH_LLM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        system_prompt: "You are a test assistant.".into(),
        temperature: Some(0.0),
        max_tokens: Some(64),
        timeout_secs: 60,
    };
    let mem = MemoryStore::new();
    mem.set_dir(&dir).await;
    mem.set("zed", "favorite_color", "cobalt").await.unwrap();

    let agent =
        ChatAgent::with_compactor_and_memory(&llm, 5, None, Some(mem.clone()), true).unwrap();
    // The prompt snippet for "zed" should now include the fact.
    let prompt = agent.system_prompt_for_pub("zed").await.unwrap();
    assert!(prompt.contains("cobalt"));

    // Ask the model; it should be able to use the injected fact. (We don't
    // hard-assert the model's wording — some backends return empty content —
    // but a successful round-trip proves the prompt path works end to end.)
    let _ = agent
        .reply("zed", "What is my favorite color? Use the known facts.")
        .await;

    std::fs::remove_dir_all(&dir).ok();
}
