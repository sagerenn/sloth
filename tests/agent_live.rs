//! Live smoke test for the chat agent against the OpenAI-compatible endpoint.
//!
//! Requires the gateway at http://172.17.0.1:8317/v1 to be reachable. Skips
//! (rather than fails) when it is not, so this remains safe to run in CI
//! without the backend.
//!
//! Run with: `cargo test --test agent_live -- --nocapture --include-ignored`

use std::time::Duration;

use sloth_agent::agent::ChatAgent;
use sloth_agent::config::LlmConfig;

async fn endpoint_reachable() -> bool {
    let base = std::env::var("SLOTH_LLM_BASE_URL")
        .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".to_string());
    let url = format!("{}/models", base.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(&url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn test_llm_config() -> LlmConfig {
    LlmConfig {
        base_url: std::env::var("SLOTH_LLM_BASE_URL")
            .unwrap_or_else(|_| "http://172.17.0.1:8317/v1".to_string()),
        model: std::env::var("SLOTH_LLM_MODEL").unwrap_or_else(|_| "glm-5.2".to_string()),
        api_key: std::env::var("SLOTH_LLM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        system_prompt: "You are a helpful test assistant. Reply with only the answer.".to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
        timeout_secs: 60,
    }
}

#[tokio::test]
#[ignore = "requires live LLM gateway"]
async fn agent_replies_to_simple_prompt() {
    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }

    let agent = ChatAgent::new(&test_llm_config(), 10).expect("agent build");
    let reply = agent
        .reply("test-sender-1", "Reply with exactly: pong")
        .await
        .expect("reply");

    assert!(!reply.text.is_empty(), "reply text should not be empty");
    println!("reply: {:?}", reply.text);
    println!("usage: {:?}, model: {}", reply.usage, reply.model);
}

#[tokio::test]
#[ignore = "requires live LLM gateway"]
async fn agent_keeps_history_turns() {
    if !endpoint_reachable().await {
        eprintln!("skipping: LLM endpoint not reachable");
        return;
    }

    let agent = ChatAgent::new(&test_llm_config(), 10).expect("agent build");

    let first = agent
        .reply(
            "test-sender-hist",
            "My favorite color is blue. Remember it.",
        )
        .await
        .expect("first reply");
    println!("first: {:?}", first.text);

    let second = agent
        .reply(
            "test-sender-hist",
            "What is my favorite color? Answer in one word.",
        )
        .await
        .expect("second reply");
    println!("second: {:?}", second.text);

    // The model should recall "blue" from history.
    let lower = second.text.to_lowercase();
    assert!(
        lower.contains("blue"),
        "expected history recall of 'blue', got: {:?}",
        second.text
    );
}
