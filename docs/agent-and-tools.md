# Agent & tools

The model-facing core: how `ChatAgent` produces replies, how it manages
conversation history, and how the `ToolRouter` dispatches function calls.

## ChatAgent

`ChatAgent` (`agent.rs:59`) wraps an `async-openai` `Client<OpenAIConfig>`
configured with a custom base URL (`llm.base_url`, default
`http://172.17.0.1:8317/v1`). It is cheaply cloneable — all state lives in
an `Arc<Inner>`.

### Auth quirk (important)

The local gateway at `172.17.0.1:8317` needs **no** API key and rejects
unknown bearer tokens. `async-openai` always sends an
`Authorization: Bearer {key}` header; with an empty key it sends `Bearer `
(empty), which the gateway accepts. So an unset `llm.api_key` is configured
as an empty string rather than suppressing the header (`agent.rs:137`). Set
`api_key` only when the gateway actually validates one.

### Two reply paths

- `reply()` (`agent.rs:163`) — single shot, no tools. Builds a request
  (system prompt + history + current turn), sends it, records the turn.
- `reply_with_tools()` (`agent.rs:220`) — the agentic loop, used by the
  runtime for all inbound messages. This is the path that matters.

### The tool-calling loop (`reply_with_tools`)

```
tools = router.tool_definitions()                          agent.rs:235
messages = system_prompt + history + [user turn]           agent.rs:236
loop {                                                     agent.rs:248
  steps += 1; if steps > max_steps → bail with cap message
  request = { model, messages, tools?, temperature?, max_tokens? }
  response = client.chat().create(request)
  accumulate usage across rounds
  if no tool_calls → final_text = content; break
  append assistant message (content + tool_calls)
  for each tool_call (Function variant only):
    outcome = run_tool_call(call, router, sender)          agent.rs:354
       └─ parse args defensively (empty → Null, bad JSON → Null)
       └─ router.execute(name, args, sender_id)  → ToolOutcome
    append tool result message { content, tool_call_id }
}
record_turn(sender, user_text, final_text)                 agent.rs:344
return Reply { final_text, total_usage, model }
```

`max_steps` is `AgentContext::max_tool_steps` (default **8**, `runtime.rs:170`).
Hitting the cap stops the loop and returns a fixed "reached max tool steps"
message rather than continuing. Token usage is **summed** across every round.

Only the **final** assistant text is recorded to history; intermediate tool
rounds (assistant-with-tool-calls and tool results) are not persisted
(`agent.rs:342-344`).

## Conversation history

`Inner.conversations: DashMap<String, Mutex<Conversation>>` keyed by sender
id (`agent.rs:70`). Each `Conversation` is a `Vec<Stored>` where `Stored` is
`User(String)` / `Assistant(String)`, capped to `history.max_messages`
(`Conversation::push`, `agent.rs:47`). The cap counts **roles** (one Stored
entry = one message), so `max_messages = 20` holds the 20 most recent
messages regardless of pairing.

`record_turn` (`agent.rs:485`) appends the `User` then `Assistant` turn, then
optionally auto-compacts (see below). The snapshot for a request clones the
`Vec` under the lock and drops the guard before building messages
(`agent.rs:382`).

### Memory injection into the prompt

`system_prompt_for(sender)` (`agent.rs:468`) returns the base `llm.system_prompt`
plus, when `[memory].inject_into_prompt` is on and the sender has recalled
facts, a `Known facts about this user:` snippet from `MemoryStore::recall`
(`memory.rs:108` → `SenderMemory::to_prompt_snippet`). So every request for a
sender carries that sender's stored facts, not just when the model calls
`memory_recall`.

### Auto-compaction

If `[compact].enabled`, `record_turn` checks `Compactor::should_compact` after
appending (`agent.rs:500`). When `history.len() >= threshold_messages` (and
`keep_recent < len`), the older turns (all but the last `keep_recent`) are
rendered as a transcript and summarized by a **second LLM call** at
`temperature: 0` (`compact.rs:62`). The history is then replaced *in place*
by a single `[summary of earlier conversation]` assistant entry followed by
the verbatim `keep_recent` tail (`compact.rs:127`). This keeps long-running
sessions inside the context window. Compaction runs under the per-sender
lock, so it can't race other turns for that sender.

## ToolRouter

`ToolRouter` (`tools.rs:83`) is the single dispatch point for function calls.
It is `Clone` with all collaborators behind `Arc`. It does two jobs:

### 1. `tool_definitions()` — what the model sees (`tools.rs:184`)

Builds the OpenAI tool list sent with each request. Always-present built-ins:

- **Scheduler**: `scheduler_add_job`, `scheduler_remove_job`, `scheduler_list_jobs`
- **Sessions**: `session_switch`, `session_set_workspace`, `session_list`
- **MCP admin**: `mcp_list_servers`, `mcp_reload`
- **Skill admin**: `skill_list`, `skill_reload`
- **A2A admin**: `a2a_list_agents`, `a2a_reload`

Plus **dynamic** tools, each gated by an `expose_*` config flag:

| flag on | dynamic tools added |
|---------|---------------------|
| `mcp.expose_tools` | one `mcp_<server>__<tool>` per remote MCP tool (`tools.rs:279`) |
| `skills.expose_tools` | one `skill_<name>` per loaded skill (`tools.rs:313`) |
| `a2a.expose_tools` | one `a2a_<name>` per connected agent (`tools.rs:354`) |
| `models.expose_tools` | `model_list`, `model_pick` (`tools.rs:374`) |
| `memory.expose_tools` | `memory_set`, `memory_recall` (`tools.rs:396`) |

Definitions are recomputed per request, so hot-reloaded tools/skills/agents
appear (or disappear) on the next turn without a restart.

### 2. `execute(tool, args, sender_id)` — run a call (`tools.rs:426`)

```
if hitl.requires_confirmation(tool):            HITL gate (see below)
    register + publish pending, await_decision (timeout = fail-closed deny)
    Denied/TimedOut → ToolOutcome::err(...)   (no dispatch)
dispatch(tool, args, sender_id)                 tools.rs:453
  match tool {
    scheduler_* / session_* / memory_* / model_*      → built-in handlers
    mcp_list_servers / mcp_reload                     → McpRegistry
    skill_list / skill_reload                         → SkillRegistry
    a2a_list_agents / a2a_reload                      → A2aRegistry
    "mcp_<srv>__<tool>"                               → mcp.call_qualified
    "a2a_<name>" (≠ list/reload)                      → a2a.send(prompt)
    "skill_<name>" (≠ list/reload)                    → skills.invoke(args)
    _                                                 → ToolOutcome::err("unknown tool")
  }
```

`ToolOutcome` is `{ content: String, is_error: bool }` — the `content`
becomes the OpenAI `tool` role message. Built-in handlers parse arguments
defensively (missing fields → a clear error string rather than a panic).
Dynamic MCP calls clone the args object before forwarding (`tools.rs:594`).

### Naming scheme for dynamic tools

The `dispatch` match arms discriminate dynamic tools **by prefix** in a
specific order, which is why the names are shaped the way they are:

- **MCP** tools are `mcp_<server>__<tool>` — server name with `-`→`_`, a
  literal `__` separator, then the tool name (`mcp.rs:285`). The dispatch
  guard is `starts_with("mcp_") && contains("__")` (`tools.rs:593`).
- **A2A** tools are `a2a_<name>`, guarded by `starts_with("a2a_")` but
  excluding the `a2a_list_agents`/`a2a_reload` built-ins (`tools.rs:600`).
- **Skill** tools are `skill_<name>`, guarded by `starts_with("skill_")`
  excluding `skill_list`/`skill_reload` (`tools.rs:623`).

The admin built-ins (`mcp_reload`, `a2a_reload`, `skill_reload`) let the
model **drive hot reloads at runtime**: e.g. `mcp_reload` accepts a desired
`servers` array (or omits it to reload from the cached config) and updates
the cached desired list so subsequent reloads are consistent (`tools.rs:537`).

## HITL gating

`hitl.requires_confirmation(tool)` (`hitl.rs:121`): if `[hitl].enabled` is
off, nothing is gated. If `confirm_tools` is **empty**, *every* tool is
gated. Otherwise only tools matching the glob patterns (`*` = any run, `?` =
one char, case-sensitive; `hitl.rs:186`) are gated.

The full request → human → resolve flow is traced in
[message flow §4](message-flow.md#4-human-in-the-loop-hitl-confirmation).
The essential design points:

- Gating is applied **inside** `ToolRouter::execute`, so it covers built-in,
  MCP, A2A, and skill tools uniformly.
- Denial/timeout produces an **error tool result** fed back to the model —
  the loop continues, the model just sees the call failed.
- Auto-deny timeout is `[hitl].timeout_secs` (default 120s). Fail-closed.
