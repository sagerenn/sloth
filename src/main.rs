//! Sloth agent — a Rust AI agent that bridges the OpenClaw WebSocket bridge to
//! an OpenAI-compatible chat completion backend.
//!
//! Thin binary over the `sloth_agent` library. See `src/lib.rs` and the
//! `agent`, `bridge`, `config`, and `runtime` modules for the implementation.

use std::process::ExitCode;

use sloth_agent::config;
use sloth_agent::runtime;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn main() -> ExitCode {
    // CLI: optional `--config <path>`.
    let explicit_config = parse_config_arg();

    let cfg = match config::load_optional_explicit(explicit_config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load config: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = cfg.validate() {
        eprintln!("error: invalid config: {e:#}");
        return ExitCode::FAILURE;
    }

    init_observability(&cfg.observability);

    tracing::info!(
        bridge_url = %cfg.bridge.url,
        channel = %cfg.bridge.channel,
        account = %cfg.bridge.account_id,
        llm_base = %cfg.llm.base_url,
        model = %cfg.llm.model,
        has_api_key = cfg.llm.api_key.is_some(),
        "starting sloth agent"
    );

    // Build a runtime and drive it.
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to build tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = rt.block_on(async {
        tokio::select! {
            res = runtime::run(cfg) => res,
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("ctrl-c received; shutting down");
                Ok(())
            }
        }
    });

    match result {
        Ok(()) => {
            tracing::info!("sloth agent stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            tracing::error!(error = %e, "sloth agent exited with error");
            ExitCode::FAILURE
        }
    }
}

/// Parse a `--config <path>` argument from argv, if present.
fn parse_config_arg() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            return args.next();
        } else if let Some(rest) = arg.strip_prefix("--config=") {
            return Some(rest.to_string());
        }
    }
    None
}

/// Initialise the tracing subscriber from observability config.
fn init_observability(obs: &config::ObservabilityConfig) {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&obs.log_filter))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let registry = tracing_subscriber::registry().with(filter);
    match obs.log_format.as_str() {
        "json" => registry.with(fmt::layer().json()).init(),
        "pretty" => registry.with(fmt::layer().pretty()).init(),
        _ => registry.with(fmt::layer()).init(),
    }
}
