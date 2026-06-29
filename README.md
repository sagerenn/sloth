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

Beyond the base inbound→LLM→reply loop, the agent exposes ten capability
areas to the model via OpenAI-compatible function calling (structured
`tool_call` objects dispatched deterministically, not free text):

1. **Remote MCP with hot reload** — `[[mcp.servers]]` entries are connected
   over the Streamable HTTP transport; each server's tools are surfaced to the
   LLM prefixed `mcp_<server>__<tool>`. The registry hot-reloads: add or
   remove a server and the agent reconnects without restarting. The LLM can
   also drive a reload at runtime via `mcp_reload` and inspect servers via
   `mcp_list_servers`.
2. **Time-based scheduler (cron)** — `scheduler_add_job` /
   `scheduler_remove_job` / `scheduler_list_jobs` let the LLM set up 5-field
   UNIX-cron jobs (UTC). A background ticker fires due jobs and injects their
   prompts into the named session, producing a reply on the channel.
3. **Session management** — `session_switch` / `session_set_workspace` /
   `session_list` manage named conversation contexts with per-session working
   directories. Each sender has an active session; history is keyed by session.
4. **Human-in-the-Loop** — gated tools (configurable via `[hitl].confirm_tools`
   glob patterns) require human approval before executing. The runtime asks
   the user over the bridge (`yes`/`no`) and auto-denies on timeout
   (fail-closed).
5. **Skills with hot reload** — `[skills]` points at a directory of markdown
   skill files (YAML-like frontmatter `name`/`description`/`arguments` + a body
   with `{{arg}}` placeholders). Each loaded skill is surfaced as an invocable
   `skill_<name>` tool; `skill_list` / `skill_reload` let the LLM inspect and
   refresh the registry. The directory is hot-reloaded on change.
6. **A2A (Agent2Agent) remote agents** — `[[a2a.agents]]` entries are reached
   via the official `a2a-rs` SDK: the Agent Card is fetched from
   `{url}/.well-known/agent-card.json` and a transport negotiated
   automatically. Each agent is surfaced as an `a2a_<name>` tool that sends a
   prompt and returns the reply (plus task state). `a2a_list_agents` /
   `a2a_reload` drive the registry at runtime.
7. **Model catalog (auto model selection)** — `[models]` points at a directory
   of YAML catalog files (see `models.example.yaml`); each lists a model's
   pricing, context window, max output, and benchmark `scores`. The agent picks
   a model automatically at startup instead of using the fixed `llm.model`.
   Selection strategy: `best_score` (default) | `best_score_under_budget` |
   `cheapest_above_floor` | `best_value`, with optional `min_score`,
   `max_cost_per_token`, and `min_context_window` constraints. The LLM can
   browse and re-pick via `model_list` / `model_pick`.
8. **Auto-compaction of history** — when `[compact]` is on and a sender's
   stored history crosses `threshold_messages`, the older turns are summarized
   into a single compact summary, retaining the `keep_recent` most recent turns
   verbatim. Keeps long-running sessions inside the model's context window.
9. **Persistent memory** — `[memory]` stores per-sender facts (one file per
   sender). Recalled facts are injected into the system prompt when
   `inject_into_prompt` is on, and the LLM can read/write them via
   `memory_set` / `memory_recall`.
10. **Function calling / structured output** — the agent runs an agentic
    tool-call loop (configurable step cap): it sends tool definitions, executes
    any returned `tool_calls` through the router (with HITL gating), feeds
    results back, and re-queries until the model produces a final text reply.

See `config.example.toml` for the `[mcp]`, `[scheduler]`, `[sessions]`,
`[hitl]`, `[skills]`, `[a2a]`, `[models]`, `[compact]`, and `[memory]` sections,
and the `SLOTH_SCHEDULER_*`, `SLOTH_HITL_*`, `SLOTH_MCP_EXPOSE_TOOLS`, and
`SLOTH_SESSION_DEFAULT` env overrides. `models.example.yaml` shows the catalog
file format.

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
  config.rs        config.toml + SLOTH_* env overrides
  bridge.rs        typed OpenClaw WS envelope protocol (camelCase on the wire)
  cron.rs          5-field UNIX-cron parser (UTC, self-contained)
  scheduler.rs     in-process cron engine: add/remove/list + fire events
  session.rs       session manager: create/switch/workspace/list/delete
  mcp.rs           remote MCP client (Streamable HTTP) + hot-reload registry
  hitl.rs          Human-in-the-Loop confirmation broker + timeout
  skill.rs         markdown skill registry (frontmatter + body templates, hot-reloaded)
  a2a.rs           Agent2Agent remote-agent registry (a2a-rs SDK, Agent Card discovery)
  model_catalog.rs YAML model catalog + auto-pick strategies (cost/capacity/benchmarks)
  compact.rs       conversation-history auto-compactor (summarize old turns)
  memory.rs        per-sender persistent memory + system-prompt injection
  tools.rs         function-call tool router (all built-in + dynamic tools)
  agent.rs         async-openai chat completions + tool-calling loop + history,
                   compaction, and memory injection
  runtime.rs       connect / subscribe / heartbeat / reconnect + tool wiring
  main.rs          thin binary: init tracing, drive runtime, signal handling
tests/
  agent_live.rs       live ChatAgent tests against the gateway
  bridge_e2e.rs       live end-to-end round trip through a mock bridge
  features_e2e.rs     unit + e2e for cron, scheduler, sessions, HITL, remote MCP,
                       skills, A2A, model catalog, compaction, and memory
                       (mock servers, hot reload, router) + a live LLM
                       function-calling test (#[ignore]d)
  mattermost_e2e.rs   live round trip through a real Mattermost server + bridge image
```

## License

MIT
