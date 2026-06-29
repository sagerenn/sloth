//! Live E2E: a Mattermost user asks the agent to, after a short delay, use a
//! remote MCP server — exercising cron scheduling, the MCP tool-call path, and
//! the complete inbound -> LLM -> (schedule) -> fire -> MCP -> reply flow.
//!
//! Covered in one user-driven round trip:
//! 1. remote MCP (mock server "A" registered into sloth, surfaced as a tool),
//! 2. the time-based scheduler with **second-level** cron (the LLM schedules a
//!    job firing ~10s out via `scheduler_add_job`),
//! 3. the complete message flow: Mattermost user -> bridge -> sloth -> LLM ->
//!    schedule; then scheduler fires -> injected prompt -> LLM calls the MCP
//!    tool -> reply posted back to the user over the DM.
//!
//! See module docs in `mattermost_e2e.rs` for the stack topology. This test
//! reuses the same Docker orchestration (mattermost-preview + the published
//! openclaw-bridge image) but inlines only what it needs.
//!
//! Run with: `cargo test --test mattermost_mcp_cron_e2e -- --nocapture --ignored`

#[path = "common/mod.rs"]
mod common;

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::sync::oneshot;

use common::{
    MattermostUser, MockMcp, bridge_port, bring_up_stack, docker_available, llm_reachable,
    provision, start_bridge_container, wait_for_bridge, wait_for_fresh_reply, wait_for_mattermost,
};
use sloth_agent::config::{
    A2aConfig, BridgeConfig, CompactConfig, Config, HistoryConfig, HitlConfig, LlmConfig,
    McpConfig, McpServerConfig, MemoryConfig, ModelCatalogConfig, ObservabilityConfig,
    SchedulerConfig, SessionConfig, SkillsConfig,
};

/// Sloth config pointed at the live bridge, with MCP server A registered and
/// the second-precise scheduler enabled (1s tick so a ~10s job fires promptly).
fn sloth_config(bridge_ws_url: String, mcp: &MockMcp) -> Config {
    Config {
        bridge: BridgeConfig {
            url: bridge_ws_url,
            channel: "mattermost".to_string(),
            account_id: "default".to_string(),
            reconnect_ms: 1_000,
            reconnect_max_ms: 3_000,
            heartbeat_ms: 25_000,
        },
        llm: LlmConfig {
            base_url: std::env::var("SLOTH_LLM_BASE_URL")
                .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".to_string()),
            model: std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".to_string()),
            api_key: std::env::var("SLOTH_LLM_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            system_prompt: SYSTEM_PROMPT.to_string(),
            temperature: Some(0.0),
            max_tokens: Some(2048),
            timeout_secs: 60,
        },
        history: HistoryConfig { max_messages: 10 },
        observability: ObservabilityConfig {
            log_format: "text".to_string(),
            log_filter: "info,sloth_agent=debug".to_string(),
            service_name: "sloth-e2e-mcp-cron".to_string(),
        },
        mcp: McpConfig {
            servers: vec![McpServerConfig {
                name: "A".to_string(),
                url: mcp.url.clone(),
                token: None,
                timeout_secs: 10,
            }],
            expose_tools: true,
            poll_secs: 0,
        },
        scheduler: SchedulerConfig {
            enabled: true,
            // 1s tick so a second-precision cron fires without lag.
            tick_secs: 1,
            default_session: "default".to_string(),
        },
        sessions: SessionConfig::default(),
        // Disable HITL so the LLM can schedule + call MCP autonomously.
        hitl: HitlConfig {
            enabled: false,
            ..Default::default()
        },
        skills: SkillsConfig::default(),
        a2a: A2aConfig::default(),
        models: ModelCatalogConfig::default(),
        compact: CompactConfig::default(),
        memory: MemoryConfig::default(),
    }
}

/// Prompt instructing the model to (a) schedule a near-term second-precision
/// job using the MCP tool, and (b) when that job fires, actually invoke the
/// MCP server A `echo` tool. This keeps the live LLM on a deterministic path.
const SYSTEM_PROMPT: &str = "\
You are a test assistant driving a scheduler and a remote MCP tool.\n\
The scheduler_add_job tool accepts a 6-field cron expression:\n\
second minute hour day-of-month month day-of-week, UTC. Second-level precision\n\
is supported. For example \"*/10 * * * * *\" fires every 10 seconds.\n\
There is a remote MCP server named A exposing the tool mcp_A__echo (argument\n\
\"message\", a string). When asked to do something after a delay, schedule a\n\
near-term job whose prompt instructs you to call mcp_A__echo, then confirm to\n\
the user that the task is scheduled. When a scheduled task fires, actually call\n\
mcp_A__echo with a short message and tell the user you did it. Keep replies very\n\
short.";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires live LLM gateway + docker"]
async fn mattermost_user_schedules_mcp_task_via_cron() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,sloth_agent=debug,warn")
        .try_init();

    if !llm_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }
    if !docker_available() {
        eprintln!("skipping: docker not available");
        return;
    }

    // Start the mock MCP server "A" on the host (sloth reaches it directly).
    let mcp = common::start_mock_mcp_recording().await;
    eprintln!("[e2e] mock MCP server A at {}", mcp.url);

    let stack = bring_up_stack().expect("bring up docker stack");
    let bp = bridge_port();
    let result = run_mcp_cron_scenario(&stack, bp, &mcp).await;
    common::teardown(&stack.net, &stack.mm_container, &stack.bridge_container);
    match result {
        Ok(()) => {
            println!("--- Mattermost MCP+cron E2E: schedule -> fire -> MCP call -> reply OK ---")
        }
        Err(e) => {
            let _ = std::process::Command::new("docker")
                .args(["logs", "--tail", "200", &stack.bridge_container])
                .status();
            let _ = std::process::Command::new("docker")
                .args(["logs", "--tail", "120", &stack.mm_container])
                .status();
            panic!("Mattermost MCP+cron E2E failed: {e:#}");
        }
    }
}

async fn run_mcp_cron_scenario(stack: &common::Stack, bp: u16, mcp: &MockMcp) -> Result<()> {
    eprintln!("[e2e] waiting for mattermost...");
    wait_for_mattermost(&stack.mm_host_url).await?;

    eprintln!("[e2e] provisioning bot + sender...");
    let p = provision(stack).await?;

    eprintln!("[e2e] starting bridge container...");
    start_bridge_container(stack, &p.bot.token, &stack.mm_internal_url, bp)?;
    wait_for_bridge(bp).await?;

    // Start the real sloth runtime in-process, MCP server A registered.
    let cfg = sloth_config(stack.bridge_ws_url.clone(), mcp);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let agent_task = tokio::spawn(async move {
        sloth_agent::runtime::run_with_shutdown(cfg, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });
    tokio::time::sleep(Duration::from_secs(1)).await;

    let mut sender = MattermostUser::new(
        stack.mm_host_url.clone(),
        p.sender.token.clone(),
        p.sender.user_id.clone(),
        p.bot.user_id.clone(),
    );
    sender.connect().await?;
    sender.ensure_dm().await?;
    eprintln!("[e2e] sender connected");

    let prompt = "after 10s, use mcp A to do something";
    sender.post_to_bot(prompt).await?;
    eprintln!("[e2e] sent: {prompt:?}");

    // 1. The user must receive an initial reply (inbound -> LLM -> outbound).
    let mut seen: HashSet<String> = HashSet::new();
    let first = tokio::time::timeout(
        Duration::from_secs(90),
        wait_for_fresh_reply(&sender, prompt, &mut seen),
    )
    .await;
    match first {
        Ok(Ok(text)) => eprintln!("[e2e] initial reply: {text:?}"),
        Ok(Err(e)) => bail!("no initial reply: {e}"),
        Err(_) => bail!("timed out waiting for the initial reply"),
    }

    // 2. The scheduled job should fire and the LLM must call mcp_A__echo.
    //    Allow generous headroom: schedule + LLM tool loop + MCP round trip.
    let deadline = std::time::Instant::now() + Duration::from_secs(90);
    loop {
        if std::time::Instant::now() > deadline {
            // Diagnostic: did the scheduler ever add a job? did MCP see a call?
            bail!(
                "MCP echo was never called within 90s (calls so far: {:?})",
                mcp.calls.lock().await.clone()
            );
        }
        if !mcp.calls.lock().await.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let calls = mcp.calls.lock().await.clone();
    eprintln!("[e2e] MCP echo called: {calls:?}");
    assert!(
        calls
            .iter()
            .any(|c| c.starts_with("echo:") || !c.is_empty())
    );

    // 3. The fired job's reply must reach the user over the DM.
    let second = tokio::time::timeout(
        Duration::from_secs(90),
        wait_for_fresh_reply(&sender, prompt, &mut seen),
    )
    .await;
    match second {
        Ok(Ok(text)) => eprintln!("[e2e] scheduled reply: {text:?}"),
        Ok(Err(e)) => bail!("no scheduled reply reached the user: {e}"),
        Err(_) => bail!("timed out waiting for the scheduled reply"),
    }

    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(10), agent_task).await;
    Ok(())
}
