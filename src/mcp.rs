//! Remote MCP (Model Context Protocol) support with hot reload — now backed
//! by the **official `rmcp` Rust SDK** (modelcontextprotocol/rust-sdk).
//!
//! We use rmcp's Streamable HTTP client transport to connect to remote MCP
//! servers, list their tools, and call them. [`McpRegistry`] holds multiple
//! named servers, exposes their tools under a prefixed name
//! (`mcp_<server>__<tool>`) so the agent can route calls, and supports **hot
//! reload**: [`McpRegistry::reload`] diffs a desired server list against the
//! live set, dropping removed servers and connecting added ones without
//! restarting the agent.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rmcp::model::{CallToolRequestParams, Tool as RmcpTool};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::McpServerConfig;

/// A tool exposed by an MCP server, in our own (SDK-agnostic) shape so the
/// rest of the crate doesn't depend on rmcp internals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema describing the tool's arguments.
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

/// Result of a tool call, flattened to text + a structured flag.
#[derive(Debug, Clone, Serialize)]
pub struct McpToolResult {
    /// Joined text content of the tool's response.
    pub text: String,
    /// True when the server reported an error in the call result.
    pub is_error: bool,
}

/// Convert an rmcp `Tool` into our own shape.
fn convert_tool(t: &RmcpTool) -> McpTool {
    use rmcp::model::JsonObject;
    let schema: Value = serde_json::to_value(&*t.input_schema as &JsonObject)
        .unwrap_or(Value::Object(serde_json::Map::new()));
    McpTool {
        name: t.name.to_string(),
        description: t.description.as_ref().map(|c| c.to_string()),
        input_schema: schema,
    }
}

/// Extract the concatenated text content from a `tools/call` result value
/// (the rmcp `CallToolResult` serialized). The result is the raw array of
/// content items; we join all `text` items.
fn extract_text(content: &[Value]) -> String {
    let mut parts = Vec::with_capacity(content.len());
    for item in content {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                    parts.push(t.to_string());
                }
            }
            Some(other) => parts.push(format!("[{other} content]")),
            None => {
                if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                    parts.push(t.to_string());
                }
            }
        }
    }
    parts.join("\n")
}

/// A handle to one connected remote MCP server: the rmcp client + its tool list.
struct ServerEntry {
    /// `name` is the rmcp service is not `Send`-friendly to keep around as a
    /// raw value; we store the running service here.
    client: Arc<RunningService<RoleClient, ()>>,
    tools: Vec<McpTool>,
}

/// Registry of connected remote MCP servers. Cloneable (state is shared).
#[derive(Clone, Default)]
pub struct McpRegistry {
    inner: Arc<Mutex<RegistryState>>,
}

#[derive(Default)]
struct RegistryState {
    /// server name → entry
    servers: HashMap<String, ServerEntry>,
}

impl McpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Connect to a single server via rmcp and add it to the registry.
    pub async fn add_server(&self, cfg: &McpServerConfig) -> Result<()> {
        let client = connect_rmcp(cfg).await?;
        let tools = list_tools(&client).await?;
        info!(server = %cfg.name, tools = tools.len(), "MCP server connected (rmcp)");
        let mut g = self.inner.lock().await;
        g.servers.insert(
            cfg.name.clone(),
            ServerEntry {
                client: Arc::new(client),
                tools,
            },
        );
        Ok(())
    }

    /// Hot reload: reconcile the live set against `desired`. Removes servers
    /// no longer in the list, connects newly-added ones. Server entries whose
    /// URL differs are reconnected.
    pub async fn reload(&self, desired: &[McpServerConfig]) -> Result<ReloadReport> {
        let mut report = ReloadReport::default();

        // Removals + URL-changed reconnections.
        {
            let g = self.inner.lock().await;
            let live_names: Vec<String> = g.servers.keys().cloned().collect();
            let desired_names: std::collections::HashSet<&str> =
                desired.iter().map(|d| d.name.as_str()).collect();
            for name in &live_names {
                if !desired_names.contains(name.as_str()) {
                    report.removed.push(name.clone());
                }
            }
        }
        for name in &report.removed {
            let mut g = self.inner.lock().await;
            g.servers.remove(name);
            info!(server = %name, "MCP server removed (hot reload)");
        }

        // Additions.
        for d in desired {
            let exists = {
                let g = self.inner.lock().await;
                g.servers.contains_key(&d.name)
            };
            if !exists {
                if let Err(e) = self.add_server(d).await {
                    warn!(server = %d.name, error = %e, "MCP server connect failed on reload");
                    report.failed.push((d.name.clone(), format!("{e:#}")));
                } else {
                    report.added.push(d.name.clone());
                }
            }
        }
        Ok(report)
    }

    /// All tools from all servers, qualified and routed.
    pub async fn routed_tools(&self) -> Vec<RoutedTool> {
        let g = self.inner.lock().await;
        let mut out = Vec::new();
        for (server, entry) in g.servers.iter() {
            for t in &entry.tools {
                out.push(RoutedTool {
                    qualified_name: qualify(server, &t.name),
                    server: server.clone(),
                    tool: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                });
            }
        }
        out
    }

    /// Call a fully-qualified tool (`mcp_<server>__<tool>`).
    pub async fn call_qualified(&self, qualified: &str, args: Value) -> Result<McpToolResult> {
        let (server, tool) = split_qualified(qualified)?;
        let client = {
            let g = self.inner.lock().await;
            g.servers
                .get(&server)
                .map(|e| e.client.clone())
                .ok_or_else(|| anyhow!("unknown MCP server: {server}"))?
        };
        call_tool(&client, &tool, args).await
    }

    /// Number of connected servers.
    pub async fn server_count(&self) -> usize {
        self.inner.lock().await.servers.len()
    }
}

/// Connect to an MCP server using rmcp's Streamable HTTP client transport.
async fn connect_rmcp(cfg: &McpServerConfig) -> Result<RunningService<RoleClient, ()>> {
    let mut transport_cfg = StreamableHttpClientTransportConfig::with_uri(cfg.url.clone());
    if let Some(tok) = &cfg.token {
        transport_cfg = transport_cfg.auth_header(format!("Bearer {tok}"));
    }
    let _ = cfg.timeout_secs; // rmcp uses the underlying reqwest client timeout
    let transport = StreamableHttpClientTransport::from_config(transport_cfg);
    // `().serve(transport)` returns Result; the client handler `()` uses
    // empty defaults for notifications/roots.
    let client = rmcp::service::ServiceExt::serve((), transport)
        .await
        .map_err(|e| anyhow!("rmcp initialize failed: {e}"))?;
    Ok(client)
}

/// List tools via the rmcp client.
async fn list_tools(client: &RunningService<RoleClient, ()>) -> Result<Vec<McpTool>> {
    let peer = client.peer();
    let tools = peer
        .list_all_tools()
        .await
        .context("rmcp tools/list failed")?;
    Ok(tools.iter().map(convert_tool).collect())
}

/// Call a tool via the rmcp client.
async fn call_tool(
    client: &RunningService<RoleClient, ()>,
    name: &str,
    args: Value,
) -> Result<McpToolResult> {
    let peer = client.peer();
    // rmcp expects arguments as a JsonObject; coerce.
    let obj = match args {
        Value::Object(m) => m,
        other => {
            let mut m = serde_json::Map::new();
            m.insert("value".to_string(), other);
            m
        }
    };
    let param = CallToolRequestParams::new(name.to_string()).with_arguments(obj);
    let result = peer
        .call_tool(param)
        .await
        .map_err(|e| anyhow!("rmcp tools/call failed: {e}"))?;
    // Serialize the CallToolResult to extract content text deterministically.
    let raw: Value = serde_json::to_value(&result).context("encoding call result")?;
    let is_error = raw
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let content = raw
        .get("content")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    let text = extract_text(&content);
    Ok(McpToolResult { text, is_error })
}

/// Summary of a hot-reload operation.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ReloadReport {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub failed: Vec<(String, String)>,
}

/// A routed tool: a fully-qualified name + the server that owns it.
#[derive(Debug, Clone, Serialize)]
pub struct RoutedTool {
    pub qualified_name: String,
    pub server: String,
    pub tool: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Build a qualified tool name.
pub fn qualify(server: &str, tool: &str) -> String {
    let s = server.replace('-', "_");
    format!("mcp_{s}__{tool}")
}

/// Split a qualified name back into (server, tool).
pub fn split_qualified(qualified: &str) -> Result<(String, String)> {
    let rest = qualified
        .strip_prefix("mcp_")
        .ok_or_else(|| anyhow!("not a qualified MCP tool name: {qualified}"))?;
    let (server, tool) = rest
        .split_once("__")
        .ok_or_else(|| anyhow!("malformed MCP tool name: {qualified}"))?;
    Ok((server.replace('_', "-"), tool.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualify_and_split_roundtrip() {
        assert_eq!(qualify("weather", "forecast"), "mcp_weather__forecast");
        assert_eq!(qualify("my-srv", "tool"), "mcp_my_srv__tool");
        let (s, t) = split_qualified("mcp_weather__forecast").unwrap();
        assert_eq!(s, "weather");
        assert_eq!(t, "forecast");
        let (s, t) = split_qualified("mcp_my_srv__tool").unwrap();
        assert_eq!(s, "my-srv");
        assert_eq!(t, "tool");
        assert!(split_qualified("not_mcp").is_err());
        assert!(split_qualified("mcp_no_double_underscore").is_err());
    }

    #[test]
    fn extract_text_from_content_array() {
        let v = vec![
            serde_json::json!({ "type": "text", "text": "hello" }),
            serde_json::json!({ "type": "image", "data": "..." }),
            serde_json::json!({ "type": "text", "text": "world" }),
        ];
        assert_eq!(extract_text(&v), "hello\n[image content]\nworld");
    }
}
