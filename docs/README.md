# Sloth Agent — Design Docs

These documents describe the *design* of the Sloth agent: how it is put
together, how data flows through it, and how each subsystem works. They
complement the [README](../README.md), which is the user/operator guide
(build, configure, run, test).

- [**Architecture**](architecture.md) — the big picture: component map, the
  shared `AgentContext`, the concurrency model, and the runtime lifecycle
  (connect → subscribe → pump → reconnect).
- [**Message flow**](message-flow.md) — end-to-end traces of the paths data
  takes: an inbound user round trip, the reconnect loop, a scheduled-job
  firing, and a Human-in-the-Loop confirmation.
- [**Agent & tools**](agent-and-tools.md) — the `ChatAgent` completion loop,
  conversation history, auto-compaction and memory injection, the
  `ToolRouter`, HITL gating, and the dynamic-tool naming scheme.
- [**Subsystems**](subsystems.md) — the feature modules and their shared
  patterns: the cron engine + scheduler, sessions, persistent memory, the
  auto-compactor, the model catalog, and the hot-reloading remote registries
  (MCP, A2A, skills).

References like `runtime.rs:314` point into `src/` for the implementation
behind each design choice.
