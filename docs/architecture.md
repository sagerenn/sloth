# Architecture

Sloth is a single Rust process (`src/main.rs` → `sloth_agent::runtime::run`)
that bridges an **OpenClaw WebSocket bridge** to an **OpenAI-compatible chat
completion backend**. The model is a stateful chat agent that can also call
tools: scheduling, sessions, memory, remote MCP servers, remote A2A agents,
skills, and model selection. Everything runs on one Tokio runtime.

```
 IM user ──► chat platform ──► [OpenClaw bridge] ──WS──► sloth ──chat completions──► glm-5.2
                                  ▲                       │
                                  │                       │ tool_calls (OpenAI function calling)
                                  └── send_text ──────────┘
```

## Component map

```
                       main.rs            tracing init + tokio runtime + signals
                         │
                         ▼
 runtime.rs ────────────────────────────────────────────────────────────────
   run / run_ctx_with_shutdown
   ├── AgentContext::build()        build all subsystems once, wire the router
   ├── scheduler.start()            background cron ticker → FiredJob channel
   ├── skills / catalog hot-reload tickers
   └── loop { run_session }         one WS connection lifecycle
        ├── connect → subscribe
        ├── writer task             drains outbound Envelopes → WS frames
        ├── heartbeat task          periodic `ping`
        └── receive select!         bridge frames │ fired jobs │ HITL requests

 agent.rs ──── ChatAgent           async-openai completions + tool loop + history
 compact.rs ── Compactor           summarize old history turns
 tools.rs ──── ToolRouter          resolve tool_call → execute, apply HITL gate
 hitl.rs ───── HitlBroker          pending-confirmation broker + timeout/auto-deny

 bridge.rs ─── Envelope / Inbound  typed WS protocol (camelCase on the wire)
 config.rs ─── Config              config.toml + SLOTH_* env overrides

 cron.rs ───── Cron                UNIX cron parser (5-field minute, or 6-field second-precision)
 scheduler.rs  Scheduler           in-process cron engine: add/remove/list + fire
 session.rs ── SessionManager      named conversation contexts + workspaces
 memory.rs ─── MemoryStore         per-sender persistent facts (TOML on disk)
 model_catalog.rs Catalog          YAML model catalog + auto-pick strategies
 mcp.rs ────── McpRegistry          remote MCP (rmcp Streamable HTTP) + hot reload
 a2a.rs ────── A2aRegistry          remote A2A agents (a2a-rs SDK) + hot reload
 skill.rs ──── SkillRegistry        markdown skills (frontmatter+body) + hot reload
```

## The shared core: `AgentContext`

`AgentContext` (`runtime.rs:34`) is built **once** in `AgentContext::build`
(`runtime.rs:56`) and held for the life of the process. It is a `Clone`able
bundle of `Arc`-shared subsystems:

| field | type | role |
|-------|------|------|
| `agent` | `ChatAgent` | completions + tool loop + history |
| `router` | `ToolRouter` | resolves/executes tool calls, applies HITL |
| `scheduler` | `Arc<Scheduler>` | cron engine |
| `sessions` | `Arc<SessionManager>` | named contexts + workspaces |
| `mcp` | `Arc<McpRegistry>` | remote MCP tools |
| `skills` | `Arc<SkillRegistry>` | invocable skills |
| `a2a` | `Arc<A2aRegistry>` | remote agents |
| `catalog` | `Arc<Catalog>` | model catalog + auto-pick |
| `memory` | `Arc<MemoryStore>` | per-sender persistent facts |
| `hitl` | `Arc<HitlBroker>` | human-confirmation gating |
| `max_tool_steps` | `usize` | cap on the tool loop (default 8) |
| `model` | `String` | effective model id (catalog-picked or `llm.model`) |

Building it does **eager, best-effort** setup: it connects configured MCP/A2A
servers, loads skills and the model catalog, and configures memory — but a
failure in any of these is logged, not fatal (`runtime.rs:114-139`). The
router is then constructed with handles to every subsystem plus the desired
MCP/A2A server lists (so the LLM can drive a hot reload at runtime via the
`mcp_reload` / `a2a_reload` tools).

### Model selection at build time

If `[models].dir` is set, the catalog is loaded and a model is **auto-picked**
on startup (`runtime.rs:61-85`) using the configured strategy
(`best_score` | `best_score_under_budget` | `cheapest_above_floor` |
`best_value`) and optional `min_score` / `max_cost_per_token` /
`min_context_window` constraints. The picked id overrides `llm.model`. If no
model satisfies the constraints, it falls back to `llm.model` and logs a
warning. See [subsystems](subsystems.md#model-catalog).

## Concurrency model

- **One Tokio multi-threaded runtime**, built in `main.rs`. The agent drives
  it with `runtime::run(cfg)` selected against `ctrl_c`.
- The chat client is `async-openai` (`Client<OpenAIConfig>`); the bridge is
  `tokio-tungstenite`.
- Shared state lives behind `Arc<...>` and locks (`tokio::sync::Mutex`,
  `DashMap`). Every registry/store is `Clone`-cheap.
- **No spawned task is required for correctness** except the scheduler ticker,
  the two hot-reload tickers (skills, catalog), the per-session writer and
  heartbeat, and the `Drop` of an A2A `AgentEntry` (which best-effort
  `destroy()`s the transport via a spawned task).

### Per-sender conversation history

`ChatAgent` holds a `DashMap<String, Mutex<Conversation>>` keyed by sender
id (`agent.rs:70`). Each `Conversation` is a `Vec<Stored>` (user/assistant
turns) capped to `history.max_messages`. The lock is held only while cloning
the snapshot for a request and while appending a recorded turn; the lock is
released before the (sync) message builders run. On `record_turn`, after the
turn is stored, an optional `Compactor` may summarize old turns while still
holding the per-sender lock (`agent.rs:485`).

## Runtime lifecycle

`run_ctx_with_shutdown` (`runtime.rs:204`) drives a **reconnect loop**. Each
iteration runs one `run_session` (`runtime.rs:314`) — a single WS connection
lifecycle:

1. **Connect** to `bridge.url`, log the `connect` span.
2. **Subscribe** to `channel:account_id` with a `subscribe` envelope.
3. **Spawn a writer task** that drains an `mpsc::channel<Envelope>(64)` of
   outbound replies into WS frames (`runtime.rs:345`).
4. **Spawn a heartbeat task** sending `ping` every `heartbeat_ms`
   (`runtime.rs:360`, 0 disables).
5. **Subscribe to HITL pending confirmations** (`hitl.pending_channel_async`,
   `runtime.rs:378`) — exactly one consumer per session.
6. **Pump** a `select!` over three sources (`runtime.rs:382`):
   - **fired scheduler job** → inject its prompt as a synthetic agent turn
     (`handle_fired_job`),
   - **pending HITL confirmation** → ask the human over the bridge
     (`ask_hitl`),
   - **bridge frame** → parsed as an `Envelope`, dispatched in `handle_text`.
   Reads time out after `heartbeat_ms*3` (min 60s) so a dead socket is
   detected.
7. On socket close/error, the session ends; the writer is awaited, the
   heartbeat aborted.

The outer loop then **sleeps** an exponential-backoff delay (capped at
`reconnect_max_ms`, base `reconnect_ms`) and reconnects — unless the shutdown
future fires first. `run_session` never returns an "unrecoverable" error
today; it always reconnects.

## Inbound dispatch

`handle_text` (`runtime.rs:505`) parses a text frame into an `Envelope` and
classifies it via `into_event()` (`bridge.rs:178`). Handled types:
`welcome`, `channel_status`, `send_ack`, `send_error`, `pong`, and the
important one — `inbound_message`. Non-text/markdown messages (image, voice)
are ignored. For a real user message:

- If a HITL confirmation is pending, the user's text is interpreted as the
  human's decision (parsed via `parse_hitl_reply`, fail-closed to `Denied`),
  the broker resolves it, and an acknowledgement is sent. Otherwise…
- The text is run through `ChatAgent::reply_with_tools` (the tool-calling
  loop) and the final reply is wrapped in a `send_text` envelope and pushed
  to the writer channel.

The outbound `to` target comes from the bridge-provided `replyTo` field when
present (channel-adapter-supplied, e.g. `user:<id>` for mattermost DMs), with
a fallback to `format_reply_target` that only rewrites bare ids to
`user:<id>` for the `mattermost` channel (`runtime.rs:666-691`). This keeps
the agent free of per-channel target-format knowledge — the bridge is the
single source of truth.

See [message flow](message-flow.md) for step-by-step traces.
