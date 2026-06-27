//! Validate that the example config parses and is valid.
//!
//! Run: `cargo run --example validate_cfg -- --config config.example.toml`

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("config.example.toml");

    let mut cfg = sloth_agent::config::Config::load_from_file(path);
    cfg.apply_env();
    if let Err(e) = cfg.validate() {
        eprintln!("invalid config {path}: {e}");
        return std::process::ExitCode::FAILURE;
    }
    println!(
        "config {path} OK: bridge={} -> {}/{}, model={}, history={}",
        cfg.bridge.url,
        cfg.bridge.channel,
        cfg.bridge.account_id,
        cfg.llm.model,
        cfg.history.max_messages
    );
    std::process::ExitCode::SUCCESS
}
