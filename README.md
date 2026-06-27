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

## Layout

```
src/
  config.rs      config.toml + SLOTH_* env overrides
  bridge.rs      typed OpenClaw WS envelope protocol (camelCase on the wire)
  agent.rs       async-openai chat completions + per-sender history
  runtime.rs     connect / subscribe / heartbeat / reconnect loop
  main.rs        thin binary: init tracing, drive runtime, signal handling
tests/
  agent_live.rs  live ChatAgent tests against the gateway
  bridge_e2e.rs  live end-to-end round trip through a mock bridge
```

## License

MIT
