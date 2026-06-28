# Sloth Agent

A Rust AI agent that bridges the OpenClaw bridge to an OpenAI-compatible chat completion backend.

```
IM user -> [OpenClaw bridge] --[WebSocket]--> sloth-agent --[chat completions]--> glm-5.2
                                   <--send_text--          <--completion--
```

The agent:

1. Exchanges messages with the OpenClaw bridge over WebSocket — connects,
   subscribes to a `channel:accountId`, handles the `welcome` / `channel_status`
   / `inbound_message` / `send_ack` / `send_error` envelope protocol, and sends
   `send_text` replies back. Includes a heartbeat (`ping`/`pong`) and an
   exponential-backoff reconnect loop.
2. Uses OpenAI SDK chat completions (via the `async-openai` crate) with a
   custom base URL (`http://172.17.0.1:8317/v1`) and model `glm-5.2` to answer
   each inbound user message. Per-sender conversation history is retained
   (capped to the most recent N turns).
3. Is observable — structured `tracing` logs (text, pretty, or JSON) with spans
   (`connect`, `chat.complete`) carrying sender, message lengths, token usage,
   finish reason, and model. Filter via `RUST_LOG` / `SLOTH_LOG_FILTER`.

## Build

Rust stable toolchain required.

```bash
cargo build --release
```

## Configure

Copy the example and edit:

```bash
cp config.example.toml config.toml
```

Any value can be overridden by a `SLOTH_*` environment variable
(`SLOTH_BRIDGE_URL`, `SLOTH_CHANNEL`, `SLOTH_ACCOUNT_ID`, `SLOTH_LLM_BASE_URL`,
`SLOTH_LLM_MODEL`, `SLOTH_LLM_API_KEY`, `SLOTH_LLM_SYSTEM_PROMPT`,
`SLOTH_LOG_FORMAT`, `SLOTH_LOG_FILTER`).

Auth note: the local gateway at `172.17.0.1:8317` does not require an API key
and rejects unknown bearer tokens. `async-openai` always sends an
`Authorization: Bearer {key}` header; with no key configured the agent sends an
empty bearer (`Bearer `), which the gateway accepts. Leave `api_key` unset
unless your gateway actually validates one.

## Tools & agent features

Beyond the base inbound→LLM→reply loop, the agent exposes five capability
areas to the model via OpenAI-compatible function calling (structured
`tool_call` objects dispatched deterministically, not free text):

1. **Remote MCP with hot reload** — `[[mcp.servers]]` entries are connected
   over the Streamable HTTP transport; each server's tools are surfaced to the
   LLM prefixed `mcp_<server>__<tool>`. The registry hot-reloads: add or
   remove a server and the agent reconnects without restarting. The LLM can
   also drive a reload at runtime via `mcp_reload` and inspect servers via
   `mcp_list_servers`.
2. **Time-based scheduler (cron)** — `scheduler_add_job` / `scheduler_remove_job` /
   `scheduler_list_jobs` let the LLM set up 5-field UNIX-cron jobs (UTC). A
   background ticker fires due jobs and injects their prompts into the named
   session, producing a reply on the channel.
3. **Session management** — `session_switch` / `session_set_workspace` /
   `session_list` manage named conversation contexts with per-session working
   directories. Each sender has an active session; history is keyed by session.
4. **Human-in-the-Loop** — gated tools (configurable via `[hitl].confirm_tools`
   glob patterns) require human approval before executing. The runtime asks
   the user over the bridge (`yes`/`no`) and auto-denies on timeout
   (fail-closed).
5. **Function calling / structured output** — the agent runs an agentic
   tool-call loop (configurable step cap): it sends tool definitions, executes
   any returned `tool_calls` through the router (with HITL gating), feeds
   results back, and re-queries until the model produces a final text reply.

See `config.example.toml` for the `[mcp]`, `[scheduler]`, `[sessions]`, and
`[hitl]` sections, and the `SLOTH_SCHEDULER_*`, `SLOTH_HITL_*`,
`SLOTH_MCP_EXPOSE_TOOLS`, and `SLOTH_SESSION_DEFAULT` env overrides.

## Run

```bash
cargo run --release            # reads ./config.toml
cargo run --release -- --config /path/to/config.toml
```

By default it connects to `ws://127.0.0.1:9300/bridge`, subscribes to the
`liangzimixin:default` account, and replies to inbound text/markdown messages.

## Tests

```bash
cargo test                    # non-live tests (default; live tests are #[ignore]d)
cargo test -- --ignored       # live tests against the real LLM gateway
```

The live end-to-end test (`tests/bridge_e2e.rs`) spins up a mock OpenClaw bridge
WS server and verifies the full inbound -> LLM -> reply round trip.

### Mattermost E2E (real server + published bridge image)

`tests/mattermost_e2e.rs` exercises the full pipeline against real services: it
launches a `mattermost/mattermost-preview` server and the published
`ghcr.io/sagerenn/openclaw-bridge` Docker image on a shared network, provisions
a bot + human sender, runs the real sloth runtime + live LLM in-process, and
exchanges 3 messages round-trip:

```
Mattermost user -> bridge -> sloth -> LLM -> reply -> user
```

It skips (rather than fails) when the LLM gateway or Docker is unavailable.
Docker is required on the host.

```bash
cargo test --test mattermost_e2e -- --nocapture --ignored
```

Overrides (env vars): `SLOTH_LLM_BASE_URL` / `SLOTH_LLM_MODEL` / `SLOTH_LLM_API_KEY`
(the gateway), `E2E_BRIDGE_IMAGE` (default `ghcr.io/sagerenn/openclaw-bridge:latest`),
`E2E_MM_IMAGE` (default `mattermost/mattermost-preview:latest`), `E2E_BRIDGE_PORT`
(default 19499), `E2E_BRIDGE_PORT_MM` (default 18065). The job runs in CI; set the
`SLOTH_LLM_*` secrets to actually exercise it (otherwise it skips).

## Layout

```
src/
  config.rs      config.toml + SLOTH_* env overrides
  bridge.rs      typed OpenClaw WS envelope protocol (camelCase on the wire)
  cron.rs        5-field UNIX-cron parser (UTC, self-contained)
  scheduler.rs   in-process cron engine: add/remove/list + fire events
  session.rs     session manager: create/switch/workspace/list/delete
  mcp.rs         remote MCP client (Streamable HTTP) + hot-reload registry
  hitl.rs        Human-in-the-Loop confirmation broker + timeout
  tools.rs       function-call tool router (scheduler + mcp + session tools)
  agent.rs       async-openai chat completions + tool-calling loop + history
  runtime.rs     connect / subscribe / heartbeat / reconnect + tool wiring
  main.rs        thin binary: init tracing, drive runtime, signal handling
tests/
  agent_live.rs       live ChatAgent tests against the gateway
  bridge_e2e.rs       live end-to-end round trip through a mock bridge
  features_e2e.rs     unit + e2e for cron, scheduler, sessions, HITL, remote MCP
                       (mock MCP server, hot reload, router) + a live LLM
                       function-calling test (#[ignore]d)
  mattermost_e2e.rs   live round trip through a real Mattermost server + bridge image
```

## License

MIT
