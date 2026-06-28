//! Sloth agent library — bridges the OpenClaw WebSocket bridge to an
//! OpenAI-compatible chat completion backend.
//!
//! See the `agent`, `bridge`, `config`, and `runtime` modules for details.
//! The `sloth-agent` binary (`src/main.rs`) is a thin wrapper over [`run`].

pub mod a2a;
pub mod agent;
pub mod bridge;
pub mod compact;
pub mod config;
pub mod cron;
pub mod hitl;
pub mod mcp;
pub mod model_catalog;
pub mod memory;
pub mod runtime;
pub mod scheduler;
pub mod session;
pub mod skill;
pub mod tools;

pub use config::Config;
