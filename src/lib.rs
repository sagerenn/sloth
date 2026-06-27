//! Sloth agent library — bridges the OpenClaw WebSocket bridge to an
//! OpenAI-compatible chat completion backend.
//!
//! See the `agent`, `bridge`, `config`, and `runtime` modules for details.
//! The `sloth-agent` binary (`src/main.rs`) is a thin wrapper over [`run`].

pub mod agent;
pub mod bridge;
pub mod config;
pub mod runtime;

pub use config::Config;
