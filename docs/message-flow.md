# Message flow

Step-by-step traces of the paths data takes through the runtime. Line
references are into `src/`.

## 1. Inbound user round trip (the main loop)

```
platform user sends "hello"
  │
  ▼
bridge ──WS text frame──► sloth  runtime.rs:399  (timeout-bounded stream.next())
  │
  ▼
handle_text  runtime.rs:505  → serde_json::from_str → Envelope → into_event()
  │   bridge.rs:178 classifies `inbound_message`
  ▼
Inbound::Message  runtime.rs:549
  ├── ignore if msg_type not in {text, markdown}        runtime.rs:551
  ├── ignore if trimmed text empty                      runtime.rs:560
  └── resolve_reply_target(channel, msg)                runtime.rs:579  (prefers `replyTo`)
  │
  ├── pending HITL?  peek hitl_rx.try_recv()            runtime.rs:580
  │     yes → parse user text as decision → hitl.resolve → ack → return
  │
  ▼
ChatAgent::reply_with_tools(sender_id, user_text, router, max_tool_steps)
  │   agent.rs:220
  │   (see "agent-and-tools" for the inner tool loop)
  ▼
Reply { text, usage, model }
  │
  ▼
build_send_text(channel, account_id, reply_target, reply.text, reply_to_msg_id, context_token)
  │   runtime.rs:643 / :694
  ▼
tx.send(Envelope)  →  writer task  runtime.rs:345  →  sink.send(Text frame) → bridge
```

Only the **final** assistant text is recorded into history (`agent.rs:344`);
intermediate tool rounds are transient scaffolding and are not stored.

## 2. Reconnect loop

```
run_ctx_with_shutdown  runtime.rs:204
  └── loop {
        select! {
          shutdown → return Ok
          outcome = run_session(&cfg, &ctx, channel, account, sched_rx) → log outcome
        }
        // backoff  runtime.rs:283
        backoff = clamp(reconnect_ms, min=1000, reconnect_max_ms)   // currently a flat max, not exponential-doubling
        select! {
          shutdown → return Ok
          sleep(backoff) → continue
        }
      }
```

`AgentContext` survives reconnects (built once), so tool/scheduler/session/
memory state outlives a dropped socket — only the WS session is re-created.
`run_session` never returns an unrecoverable error; every outcome is treated
as "reconnect". The read timeout (`heartbeat_ms*3`, min 60s, `runtime.rs:400`)
ensures a silently-dead socket is detected rather than hung.

## 3. Scheduled job firing

```
[scheduler]  Scheduler::start  runtime.rs  → background task  scheduler.rs
  every tick_secs: evaluate(wall_secs())
    for each job whose next_fire <= now:
      emit FiredJob { id, name, prompt, session_id, reply_to, fired_at }
      advance next_fire = cron.next_after(next_fire)   // catch-up capped at 60/tick
  → mpsc::UnboundedSender<FiredJob>

Note: the shutdown oneshot *sender* that stops the ticker is held in the outer
run scope (`_sched_handle`) for the whole run — it must NOT be dropped inside
the `if cfg.scheduler.enabled` block, or the ticker dies the instant it starts.

run_session select!  drains sched_rx
  ▼
handle_fired_job  runtime.rs
  prompt = "[scheduled task: {name}]\n{job.prompt}"
  ChatAgent::reply_with_tools(&job.session_id, prompt, router, max_tool_steps)
  to = job.reply_to (matters: user:<id>) else channel   ← replies to the scheduler
  ▼
build_send_text → tx → writer → bridge   (reply is visible to the channel user)
```

Jobs fire into the **named session** (`job.session_id`, default `"default"`),
so scheduled prompts use that session's history/context. Because jobs are
dispatched through the same `reply_with_tools` path, a fired job can itself
trigger tool calls (and HITL gating) just like a user turn.

## 4. Human-in-the-Loop (HITL) confirmation

```
ChatAgent runs a tool_call that the router flags for confirmation
  │
  ▼
ToolRouter::execute  tools.rs:426
  if hitl.requires_confirmation(tool)        hitl.rs:121  (glob match on confirm_tools; empty = all gated)
    pending = hitl.new_pending(tool, summary, "default", sender_id)
    rx = hitl.register(pending)              hitl.rs:138  → oneshot, keyed by pending.id
    hitl.publish(pending)                    → pending_tx mpsc
    outcome = hitl.await_decision(rx)        hitl.rs:146  → timeout(timeout_secs) auto-deny
      Approved → continue to dispatch
      Denied   → ToolOutcome::err("…denied…")
      TimedOut → ToolOutcome::err("…timed out…")       (fail-closed)
  ▼
dispatch(tool, args, sender_id)  tools.rs:453
```

The user-facing side, run by the session:

```
HitlBroker publishes pending              runtime.rs:378 subscribed hitl_rx
  ▼
run_session select!  runtime.rs:392   pending = hitl_rx.recv()
  ▼
ask_hitl  runtime.rs:489
  sends: "🔑 Approval needed for `{tool}` (id `{id}`):\n{summary}\nReply yes/no (auto-denies in {N}s)."
  → build_send_text → tx → writer → bridge → user

user replies "yes" / "no" / "approve" / "deny" / ...
  ▼
handle_text, Inbound::Message  runtime.rs:580
  hitl_rx.try_recv() → PendingConfirmation
  decision = parse_hitl_reply(user_text)   runtime.rs:299  (case-insensitive; unknown → Denied, fail-closed)
  hitl.resolve(&pending.id, decision)      → sends Outcome on the oneshot
  → ack ("✅ Approved." / "🚫 Denied." / "⏱️ Timed out.") sent to user
```

Key properties:
- **Fail-closed.** Timeout ⇒ `TimedOut` ⇒ treated as denial and the tool is
  not executed. An unrecognized reply word ⇒ `Denied`.
- **One pending confirmation at a time** is assumed by the `try_recv`-based
  reply matching; the most recent pending request is matched against the
  user's next text message.
- A denial resolves the tool call to an **error tool result**, which the
  agent then feeds back to the model as part of the tool loop.
