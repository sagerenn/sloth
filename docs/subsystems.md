# Subsystems

The feature modules and the patterns they share. References are into `src/`.

## Shared patterns

Most subsystems follow the same shape:

- A **registry/store** that is `#[derive(Clone)]` with all state behind
  `Arc<Mutex<...>>` (or `DashMap`), so it's cheap to hand around the
  `AgentContext` and share across the writer/heartbeat/scheduler tasks.
- **Eager, best-effort startup**: configured in `AgentContext::build`
  (`runtime.rs:56`); a connect/load failure is `tracing::warn!`-logged, not
  fatal. The agent still runs without that subsystem.
- **Hot reload**: a `reload()` that reconciles a desired list against the live
  set — drop removed entries, connect added ones — returning a `ReloadReport`
  `{ added, removed, failed }`. Driven either by an LLM tool (`mcp_reload`,
  `a2a_reload`, `skill_reload`) or by a background poll ticker.
- **Exposure flags**: each subsystem has an `expose_tools` config flag
  controlling whether its dynamic tools appear in `tool_definitions()`.
  Admin tools (`*_list_*`, `*_reload`) are always exposed.

## Scheduler + cron

A two-layer design: a pure cron parser and an in-process engine on top.

- **`cron.rs`** — a self-contained UNIX-cron parser with optional
  **second-level precision**. Accepts a 5-field expression
  (`minute hour dom month dow`; 0 and 7 both Sunday) for the classic
  minute-granular Vixie form, **or** a 6-field expression
  (`second minute hour dom month dow`) for sub-minute scheduling — what lets
  the agent honor "in 10 seconds" requests (`*/10 * * * * *` fires every 10s).
  Supports `*`, comma lists, `a-b` ranges, and `a-b/n` / `*/n` steps. No named
  ranges (Jan/Mon) — numeric only, dependency-free. Fields compile to per-field
  `u64` bitsets; the optional seconds field is stored separately. Day matching
  follows **Vixie cron semantics**: when both DOM and DOW are restricted, a
  match on *either* satisfies the day (`cron.rs`). `next_after(epoch_secs)`
  iterates second-by-second for 6-field expressions (minute-by-minute
  otherwise) to find the next match (capped at ~1 year to avoid pathological
  loops). Civil-date breakdown uses Howard Hinnant's days-from-civil algorithm,
  so no date crate is pulled in.

- **`scheduler.rs`** — the engine. Each `ScheduledJob` has a name, cron expr,
  prompt, `session_id`, and an optional `reply_to` (the outbound `to` target
  for the fired job's reply, set from the originating sender). `add_at(job, now)`
  parses the cron and computes the first fire time; `add()` uses the wall clock.
  `evaluate(now)` is the deterministic core: it advances every due job's
  `next_fire` and emits `FiredJob` events, catching up missed ticks (capped at
  60/tick, and it stops after the first future reschedule to avoid emitting
  dozens of events for one tick). `start()` spawns a ticker that calls
  `evaluate(wall_secs())` every `tick_secs` and emits on an unbounded channel.
  The runtime drains that channel in its `select!` and injects each
  `FiredJob`'s prompt as a synthetic agent
  turn under the job's session (`runtime.rs:459`).

  Exposing `evaluate` separately from `start` lets the scheduler be tested
  deterministically without a wall clock.

## Sessions

`SessionManager` (`session.rs:34`) holds named `Session`s (id, label,
optional `workspace` working directory, `created_at`) and tracks the active
session per sender. Seeded with a default session on startup. Operations:
`create`, `set_workspace` (creates the dir if missing), `switch` (sets the
sender's active session), `list`, `delete` (the default session is
protected; senders active in a deleted session are re-pointed to the
default, `session.rs:148`).

> Note: per-sender conversation history lives in `ChatAgent`, keyed by
> sender id (`agent.rs:70`), not in the session manager. Sessions add the
> concept of named contexts + workspaces; the scheduler and tools reference
> sessions by id.

## Persistent memory

`MemoryStore` (`memory.rs:51`) stores per-sender `key → value` facts as a
TOML file per sender under `[memory].dir` (`<sender>.toml`). Writes persist
immediately; reads are cached lazily in a `BTreeMap` behind a per-store
mutex. `recall(sender)` loads (from cache or disk) and returns a
`SenderMemory`. When `[memory].inject_into_prompt` is on, the agent appends
a `Known facts about this user:` snippet to the system prompt each request
(`agent.rs:468`). The model can also read/write facts via the `memory_set` /
`memory_recall` tools. The sender id is **sanitized** to `[-_.A-Za-z0-9]`
before becoming a filename, so a malicious sender id can't escape the
directory (`memory.rs:154`). An empty id becomes `anon`.

## Auto-compactor

`Compactor` (`compact.rs:29`) shares the chat client config. When a sender's
history crosses `[compact].threshold_messages`, the older turns (all but the
last `keep_recent`) are rendered as a transcript and summarized via a
second LLM call (`temperature: 0`, `max_tokens: 512`), and the history is
replaced by one `[summary of earlier conversation]` entry + the verbatim tail
(`compact.rs:127`). Triggered from `record_turn` under the per-sender lock
(`agent.rs:500`). If the summarization call fails, the full history is
retained (warn-logged) rather than losing data.

## Model catalog

`Catalog` (`model_catalog.rs:143`) loads YAML files from `[models].dir`; each
file is either a single model map or a `models:` list (`CatalogFile`,
`model_catalog.rs:107`). A `ModelInfo` has id, provider, `context_window`,
optional `max_output`, `pricing` (per-1M-token prompt/completion), and
free-form benchmark `scores`. `score()` prefers the special `average` key,
falls back to the mean of all scores, then 0 (`model_catalog.rs:85`).
`blended_cost_per_token()` weights prompt 0.75 / completion 0.25 for ranking
(`model_catalog.rs:98`).

`pick(opts)` (`model_catalog.rs:198`) filters by `min_score`,
`max_cost_per_token`, and `min_context_window`, then selects by `Strategy`:

| strategy | selection |
|----------|-----------|
| `best_score` (default) | highest `score()` |
| `best_score_under_budget` | highest score among those under cost cap |
| `cheapest_above_floor` | lowest blended cost among those ≥ `min_score` |
| `best_value` | max `score / blended_cost_per_token` |

Picked once at startup (`runtime.rs:72`) and used as the effective model id;
falls back to `llm.model` if nothing satisfies the constraints. The catalog
has a hot-reload ticker (`runtime.rs:249`), and the LLM can browse/re-pick
via `model_list` / `model_pick`. Entries without an id are dropped on load
(`model_catalog.rs:333`); malformed files are skipped with a warning, not
fatal.

## Remote registries (MCP, A2A, skills)

Three hot-reloading registries of *external* capabilities, each surfaced as
dynamic tools. They share the reload/diff shape but differ in transport.

### MCP (`mcp.rs`)

Backed by the official `rmcp` SDK over **Streamable HTTP**. `add_server`
initializes the rmcp client and lists its tools; each tool is converted to
an SDK-agnostic `McpTool` (`{name, description, input_schema}`) so the rest
of the crate doesn't depend on rmcp internals. Tools are surfaced as
`mcp_<server>__<tool>` (`qualify`, `mcp.rs:285`); calls route through
`call_qualified` → `call_tool`, which coerces args to a `JsonObject` and
joins the text content of the result (`extract_text`, `mcp.rs:64`).
`reload(desired)` diffs live vs desired: removes dropped names, connects
added ones; a per-server connect failure is recorded in the report, not
fatal (`mcp.rs:128`).

### A2A (`a2a.rs`)

Backed by the official `a2a-rs` SDK (vendored). `add_agent` fetches the
**Agent Card** from `{url}/.well-known/agent-card.json` via `reqwest`, then
the SDK's `A2AClientFactory` negotiates a transport from the card
(`a2a.rs:192`). A bearer `AuthInterceptor` is injected when a token is set
(`a2a.rs:203`). Each agent is surfaced as `a2a_<name>` whose single arg is a
`prompt`; `send` dispatches it and extracts text from the task's status
message or artifacts (`task_text`, `a2a.rs:48`). `AgentEntry::drop` spawns a
best-effort `client.destroy()` (can't await in `Drop`, `a2a.rs:37`). Same
reload/diff shape as MCP.

### Skills (`skill.rs`)

Markdown skill files (YAML-like frontmatter `name`/`description`/`arguments`
+ a body with `{{arg}}` placeholders), loaded from `[skills].dir`. Each
loaded skill becomes an invocable `skill_<name>` tool whose schema is
derived from the frontmatter arguments; invoking it renders the body with the
args substituted. Reload/diff shape as above, plus a **background poll
ticker** that rescans the directory every `skills.poll_secs` so added/edited/
removed files take effect without the LLM calling `skill_reload`
(`runtime.rs:228`). The model catalog has an analogous poll ticker
(`runtime.rs:249`).

## Multi-tenancy & access control (`tenant.rs`)

A **tenant** is an isolation scope derived from the bridge subscription:
`tenant_id = "{channel}:{account_id}"` (`tenant_id_from_subscription`). A
**principal** is the pair `(tenant_id, sender_id)` — the unit of isolation.
Every stateful subsystem is namespaced by `Principal::scope_key()`
(`"{tenant}/{sender}"`, sanitized), so two principals never share state:

| subsystem | namespacing |
|-----------|-------------|
| conversation history | keyed by scope key (`agent.rs` `reply_with_tools`) |
| persistent memory | file per scope key (`memory_set`/`memory_recall`) |
| sessions | active-session map keyed by scope key |
| scheduler jobs | `ScheduledJob.tenant_id`; `list_for`/`remove_for` filter by tenant |

**Access control is RBAC.** Each sender resolves to a `Role` (`admin` |
`member` | `guest` | `Custom`); each role grants a `BTreeSet<Permission>`;
each tool name maps to a required `Permission` (`required_permission`).
`ToolRouter::execute` calls `Tenants::authorize(principal, tool)` **before**
HITL and before dispatch — a denied call returns a tool error to the model
without executing or prompting the human. Builtin role → permission sets:

- **admin** — everything, including `Admin` (registry reloads,
  `tenant_list_members`).
- **member** — chat, schedule, sessions, MCP, skills, A2A, models, memory.
- **guest** — chat + models only.

Custom roles are defined in `[[tenancy.roles]]` as a name + permission list;
unknown custom roles fall back to guest (least privilege). Per-sender
overrides (`[[tenancy.members]]`) may be bare sender ids (apply in every
tenant) or `tenant/sender` scope keys (apply only in that tenant); both take
precedence over `default_role`.

When `[tenancy].enabled = false`, enforcement is off and every principal is
authorized — the default, preserving single-tenant behavior. The `tenant_whoami`
tool (baseline `chat` permission) reports the caller's tenant, sender, role,
and permissions; `tenant_list_members` (admin-only) lists configured members.
Env overrides: `SLOTH_TENANCY_ENABLED`, `SLOTH_TENANCY_DEFAULT_ROLE`.

## Configuration

`Config` (`config.rs:13`) loads `config.toml` (optional; falls back to
defaults on missing/parse error) then applies `SLOTH_*` env overrides on top
(`apply_env`, `config.rs:403`) — env always wins, enabling no-touch
deployment. `load_optional_explicit` surfaces file errors only when a path
was given via `--config` (`config.rs:477`). `validate()` checks the bridge
URL parses and `channel`/`base_url`/`model` are non-empty (`config.rs:457`).
Every config struct is `#[serde(default)]`, so a partial TOML works.
