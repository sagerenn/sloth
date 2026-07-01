//! Multi-tenancy and access control (RBAC).
//!
//! A **tenant** is an isolation scope. Sloth derives a [`Principal`] at the
//! inbound boundary from the bridge context. Every stateful subsystem is
//! namespaced by [`Principal::scope_key`], so two principals never share state.
//!
//! **Access control** is role-based: each sender maps to a [`Role`], each role
//! carries a set of [`Permission`]s, and each tool maps to a required
//! permission. The `ToolRouter` checks permissions before dispatch (and before
//! HITL). Unknown senders fall back to the configured `default_role`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

// ─── Principal & tenant identity ─────────────────────────────────────────────

/// The identity of an inbound actor: which tenant they belong to and their
/// sender id within that tenant. The pair is the unit of isolation: state
/// keys are derived from [`Principal::scope_key`], so two principals never
/// share state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Principal {
    /// Tenant id — typically `"{channel}:{account_id}"` from the subscription.
    pub tenant_id: String,
    /// Sender id within the tenant (the bridge's `senderId`).
    pub sender_id: String,
}

impl Principal {
    /// Build a principal from the bridge subscription context.
    pub fn new(tenant_id: impl Into<String>, sender_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            sender_id: sender_id.into(),
        }
    }

    /// The namespacing key for stateful subsystems. Two principals with
    /// different `tenant_id` or `sender_id` get disjoint keys.
    pub fn scope_key(&self) -> String {
        // Escape the separator so a sender id containing '/' can't collide.
        let t = sanitize(&self.tenant_id);
        let s = sanitize(&self.sender_id);
        format!("{t}/{s}")
    }
}

/// Replace any non-safe char with `_` so the key is a flat path segment.
fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("anon");
    }
    out
}

/// Derive the tenant id from the bridge subscription. The bridge subscribes
/// to `channel:account_id`, so the pair is the natural tenant boundary: two
/// accounts on the same channel are distinct tenants.
pub fn tenant_id_from_subscription(channel: &str, account_id: &str) -> String {
    format!("{}:{}", sanitize(channel), sanitize(account_id))
}

// ─── RBAC: permissions, roles, tool→permission map ───────────────────────────

/// A coarse-grained capability a role may hold. Tools require one of these;
/// the router checks `role.permissions ⊇ tool.required_permission`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Send a message / chat with the agent. The baseline capability.
    Chat,
    /// Use scheduling tools (add/remove/list cron jobs).
    Schedule,
    /// Manage sessions (switch / set workspace / list).
    ManageSessions,
    /// Invoke remote MCP tools (`mcp_*__*`).
    UseMcp,
    /// Invoke skills (`skill_*`).
    UseSkills,
    /// Invoke remote A2A agents (`a2a_*`).
    UseA2a,
    /// Browse the model catalog (`model_list` / `model_pick`).
    UseModels,
    /// Read/write persistent memory (`memory_set` / `memory_recall`).
    UseMemory,
    /// Admin: hot-reload registries (`mcp_reload`, `a2a_reload`, `skill_reload`),
    /// inspect tenant membership, mutate cross-user state.
    Admin,
}

impl Permission {
    /// The set of permissions granted to each builtin role.
    pub fn for_builtin(role: Role) -> &'static [Permission] {
        match role {
            Role::Admin => &[
                Permission::Chat,
                Permission::Schedule,
                Permission::ManageSessions,
                Permission::UseMcp,
                Permission::UseSkills,
                Permission::UseA2a,
                Permission::UseModels,
                Permission::UseMemory,
                Permission::Admin,
            ],
            Role::Member => &[
                Permission::Chat,
                Permission::Schedule,
                Permission::ManageSessions,
                Permission::UseMcp,
                Permission::UseSkills,
                Permission::UseA2a,
                Permission::UseModels,
                Permission::UseMemory,
            ],
            Role::Guest => &[Permission::Chat, Permission::UseModels],
            // Custom roles resolve their own permission set via `permissions_for_role`.
            Role::Custom(_) => &[],
        }
    }
}

/// A builtin role. Custom roles (defined in config as a named permission set)
/// are represented by [`Role::Custom`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Full access including admin tools. The default for unknown senders
    /// when RBAC is disabled.
    #[default]
    Admin,
    Member,
    Guest,
    /// A custom role name, resolved against the configured permission sets.
    Custom(String),
}

impl Role {
    /// Parse a role name (case-insensitive) into a builtin or custom role.
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "admin" => Role::Admin,
            "member" => Role::Member,
            "guest" => Role::Guest,
            other => Role::Custom(other.to_string()),
        }
    }

    /// The configured name of this role.
    pub fn name(&self) -> &str {
        match self {
            Role::Admin => "admin",
            Role::Member => "member",
            Role::Guest => "guest",
            Role::Custom(c) => c.as_str(),
        }
    }
}

/// The required permission for a tool name. Returns [`Permission::Admin`] for
/// admin/mutation tools, [`Permission::Chat`] as the fallback for unknown
/// tools (least-privilege: a guest can still chat), and the matching
/// capability for the rest.
pub fn required_permission(tool: &str) -> Permission {
    use crate::tools::names as n;
    match tool {
        // Admin / registry mutation tools.
        n::MCP_RELOAD | n::A2A_RELOAD | n::SKILL_RELOAD => Permission::Admin,
        // Scheduler.
        n::SCHED_ADD | n::SCHED_REMOVE | n::SCHED_LIST => Permission::Schedule,
        // Sessions.
        n::SESS_SWITCH | n::SESS_SET_WS | n::SESS_LIST => Permission::ManageSessions,
        // Model catalog.
        n::MODEL_LIST | n::MODEL_PICK => Permission::UseModels,
        // Memory.
        n::MEM_SET | n::MEM_RECALL => Permission::UseMemory,
        // MCP admin (read-only) needs Admin — these can mutate remote state.
        n::MCP_LIST_SERVERS => Permission::Admin,
        // A2A / skill admin list tools are read-only but restricted to members.
        n::A2A_LIST => Permission::UseA2a,
        n::SKILL_LIST => Permission::UseSkills,
        // Tenant tools: whoami is baseline chat; listing members is admin-only.
        n::TENANT_WHOAMI => Permission::Chat,
        n::TENANT_LIST_MEMBERS => Permission::Admin,
        // Prefixed dynamic tools.
        other if other.starts_with("mcp_") && other.contains("__") => Permission::UseMcp,
        other if other.starts_with(n::A2A_PREFIX) => Permission::UseA2a,
        other if other.starts_with(n::SKILL_PREFIX) => Permission::UseSkills,
        // Anything else (unknown / future builtin) — baseline chat.
        _ => Permission::Chat,
    }
}

// ─── Tenants registry ────────────────────────────────────────────────────────

/// Resolves a [`Principal`] to its [`Role`] and the role's permission set.
///
/// Built once from [`TenancyConfig`](crate::config::TenancyConfig) and cheaply
/// cloneable (state shared behind an `Arc`). Per-sender role assignments take
/// precedence over the configured `default_role`; senders with no assignment
/// get `default_role`.
#[derive(Clone, Default)]
pub struct Tenants {
    inner: Arc<Mutex<TenantState>>,
}

#[derive(Default)]
struct TenantState {
    /// Default role for unknown senders.
    default_role: Role,
    /// Custom roles: name → permission set (a superset/relaxation of builtin).
    custom_roles: BTreeMap<String, BTreeSet<Permission>>,
    /// Per-(tenant, sender) role overrides. Keyed by `Principal::scope_key()`
    /// so overrides are tenant-scoped.
    overrides: HashMap<String, Role>,
}

impl Tenants {
    /// Build a registry from config.
    pub fn from_config(cfg: &crate::config::TenancyConfig) -> Self {
        let default_role = Role::parse(&cfg.default_role);
        let mut custom_roles: BTreeMap<String, BTreeSet<Permission>> = BTreeMap::new();
        for cr in &cfg.roles {
            custom_roles.insert(cr.name.clone(), permissions_from_names(&cr.permissions));
        }
        let mut overrides = HashMap::new();
        for m in &cfg.members {
            // `m.sender` may be a bare sender id (applies in any tenant) or
            // a `tenant/sender` scope key (applies only in that tenant).
            overrides.insert(m.sender.clone(), Role::parse(&m.role));
        }
        Self {
            inner: Arc::new(Mutex::new(TenantState {
                default_role,
                custom_roles,
                overrides,
            })),
        }
    }

    /// Reload the registry from config (hot reload).
    pub async fn reload(&self, cfg: &crate::config::TenancyConfig) {
        let next = Tenants::from_config(cfg);
        let mut g = self.inner.lock().await;
        *g = next.inner.lock().await.clone_state();
    }

    /// The effective role for a principal. Checks tenant-scoped overrides
    /// (`tenant/sender`), then bare-sender overrides, then the default role.
    pub async fn role_for(&self, principal: &Principal) -> Role {
        let g = self.inner.lock().await;
        let key = principal.scope_key();
        if let Some(r) = g.overrides.get(&key) {
            return r.clone();
        }
        if let Some(r) = g.overrides.get(&principal.sender_id) {
            return r.clone();
        }
        g.default_role.clone()
    }

    /// The permission set held by `principal`'s role.
    pub async fn permissions_for(&self, principal: &Principal) -> BTreeSet<Permission> {
        let role = self.role_for(principal).await;
        self.permissions_for_role(&role).await
    }

    /// Resolve a role to its permission set (builtin defaults + custom).
    pub async fn permissions_for_role(&self, role: &Role) -> BTreeSet<Permission> {
        let g = self.inner.lock().await;
        match role {
            Role::Admin | Role::Member | Role::Guest => Permission::for_builtin(role.clone())
                .iter()
                .copied()
                .collect(),
            Role::Custom(name) => g
                .custom_roles
                .get(name)
                .cloned()
                // Unknown custom role: fall back to guest (least privilege).
                .unwrap_or_else(|| {
                    Permission::for_builtin(Role::Guest)
                        .iter()
                        .copied()
                        .collect()
                }),
        }
    }

    /// Does `principal` hold the permission `req`?
    pub async fn has_permission(&self, principal: &Principal, req: Permission) -> bool {
        self.permissions_for(principal).await.contains(&req)
    }

    /// Authorize a tool call. Returns the resolved role on success, or an
    /// error describing the denial (the router surfaces it to the model).
    pub async fn authorize(&self, principal: &Principal, tool: &str) -> Result<Role, AuthzError> {
        let role = self.role_for(principal).await;
        let req = required_permission(tool);
        let perms = self.permissions_for_role(&role).await;
        if perms.contains(&req) {
            Ok(role)
        } else {
            Err(AuthzError::Denied {
                tool: tool.to_string(),
                role: role.name().to_string(),
                required: req,
                sender: principal.sender_id.clone(),
            })
        }
    }

    /// List the configured members (for the `tenant_list_members` admin tool).
    pub async fn list_members(&self) -> Vec<MemberView> {
        let g = self.inner.lock().await;
        let mut out: Vec<MemberView> = g
            .overrides
            .iter()
            .map(|(k, r)| MemberView {
                identity: k.clone(),
                role: r.name().to_string(),
            })
            .collect();
        out.sort_by(|a, b| a.identity.cmp(&b.identity));
        out.push(MemberView {
            identity: "(default)".to_string(),
            role: g.default_role.name().to_string(),
        });
        out
    }

    /// The default role name (for diagnostics / `whoami`).
    pub async fn default_role_name(&self) -> String {
        self.inner.lock().await.default_role.name().to_string()
    }
}

impl TenantState {
    fn clone_state(&self) -> TenantState {
        TenantState {
            default_role: self.default_role.clone(),
            custom_roles: self.custom_roles.clone(),
            overrides: self.overrides.clone(),
        }
    }
}

/// Parse a list of permission-name strings into a set. Unknown names are
/// ignored (fail-open on config typos, logged by the caller).
fn permissions_from_names(names: &[String]) -> BTreeSet<Permission> {
    let mut set = BTreeSet::new();
    for n in names {
        if let Some(p) = parse_permission(n) {
            set.insert(p);
        }
    }
    set
}

/// Parse a single permission name (case-insensitive).
fn parse_permission(s: &str) -> Option<Permission> {
    match s.trim().to_ascii_lowercase().as_str() {
        "chat" => Some(Permission::Chat),
        "schedule" => Some(Permission::Schedule),
        "manage_sessions" => Some(Permission::ManageSessions),
        "use_mcp" => Some(Permission::UseMcp),
        "use_skills" => Some(Permission::UseSkills),
        "use_a2a" => Some(Permission::UseA2a),
        "use_models" => Some(Permission::UseModels),
        "use_memory" => Some(Permission::UseMemory),
        "admin" => Some(Permission::Admin),
        _ => None,
    }
}

/// Authorization denial (returned to the model as a tool error).
#[derive(Debug, Clone)]
pub enum AuthzError {
    Denied {
        tool: String,
        role: String,
        required: Permission,
        sender: String,
    },
}

impl std::fmt::Display for AuthzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthzError::Denied {
                tool,
                role,
                required,
                sender,
            } => write!(
                f,
                "access denied: sender '{sender}' (role '{role}') lacks permission '{required:?}' required by tool '{tool}'",
            ),
        }
    }
}

impl std::error::Error for AuthzError {}

/// A view of a configured member (returned by `tenant_list_members`).
#[derive(Debug, Clone, Serialize)]
pub struct MemberView {
    pub identity: String,
    pub role: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TenancyConfig, TenancyMemberConfig, TenancyRoleConfig};

    fn cfg_with(member_role: &str) -> TenancyConfig {
        TenancyConfig {
            enabled: true,
            default_role: "guest".to_string(),
            roles: Vec::new(),
            members: vec![TenancyMemberConfig {
                sender: "alice".to_string(),
                role: member_role.to_string(),
            }],
        }
    }

    #[tokio::test]
    async fn principal_scope_key_isolates_tenants_and_senders() {
        let a = Principal::new("t1", "alice");
        let b = Principal::new("t1", "bob");
        let c = Principal::new("t2", "alice");
        assert_ne!(a.scope_key(), b.scope_key());
        assert_ne!(a.scope_key(), c.scope_key());
        assert_eq!(a.scope_key(), Principal::new("t1", "alice").scope_key());
    }

    #[tokio::test]
    async fn override_and_default_roles_resolve() {
        let t = Tenants::from_config(&cfg_with("admin"));
        // alice has an override → admin.
        let alice = Principal::new("t1", "alice");
        assert_eq!(t.role_for(&alice).await, Role::Admin);
        // bob has no override → default guest.
        let bob = Principal::new("t1", "bob");
        assert_eq!(t.role_for(&bob).await, Role::Guest);
    }

    #[tokio::test]
    async fn authorize_allows_and_denies() {
        let t = Tenants::from_config(&cfg_with("member"));
        let alice = Principal::new("t1", "alice");
        // member can schedule.
        assert!(t.authorize(&alice, "scheduler_add_job").await.is_ok());
        // member cannot admin-reload.
        assert!(t.authorize(&alice, "mcp_reload").await.is_err());
        // guest cannot schedule.
        let t2 = Tenants::from_config(&TenancyConfig {
            enabled: true,
            default_role: "guest".to_string(),
            roles: Vec::new(),
            members: Vec::new(),
        });
        let guest = Principal::new("t1", "stranger");
        assert!(t2.authorize(&guest, "scheduler_add_job").await.is_err());
        // guest can chat (baseline).
        assert!(t2.authorize(&guest, "some_unknown_tool").await.is_ok());
    }

    #[tokio::test]
    async fn custom_role_permissions() {
        let cfg = TenancyConfig {
            enabled: true,
            default_role: "operator".to_string(),
            roles: vec![TenancyRoleConfig {
                name: "operator".to_string(),
                permissions: vec!["chat".into(), "schedule".into(), "use_mcp".into()],
            }],
            members: Vec::new(),
        };
        let t = Tenants::from_config(&cfg);
        let p = Principal::new("t1", "someone");
        assert_eq!(t.role_for(&p).await, Role::Custom("operator".into()));
        assert!(t.authorize(&p, "scheduler_add_job").await.is_ok());
        assert!(t.authorize(&p, "mcp_weather__echo").await.is_ok());
        assert!(t.authorize(&p, "memory_set").await.is_err());
    }

    #[tokio::test]
    async fn tenant_scoped_override_only_applies_in_its_tenant() {
        let cfg = TenancyConfig {
            enabled: true,
            default_role: "guest".to_string(),
            roles: Vec::new(),
            members: vec![TenancyMemberConfig {
                // tenant-scoped: only this exact pair.
                sender: "t1/alice".to_string(),
                role: "admin".to_string(),
            }],
        };
        let t = Tenants::from_config(&cfg);
        let in_tenant = Principal::new("t1", "alice");
        let other_tenant = Principal::new("t2", "alice");
        assert_eq!(t.role_for(&in_tenant).await, Role::Admin);
        assert_eq!(t.role_for(&other_tenant).await, Role::Guest);
    }

    #[test]
    fn required_permission_covers_admin_tools() {
        assert_eq!(required_permission("mcp_reload"), Permission::Admin);
        assert_eq!(required_permission("a2a_reload"), Permission::Admin);
        assert_eq!(required_permission("skill_reload"), Permission::Admin);
        assert_eq!(required_permission("mcp_weather__echo"), Permission::UseMcp);
        assert_eq!(required_permission("a2a_worker"), Permission::UseA2a);
        assert_eq!(
            required_permission("skill_summarize"),
            Permission::UseSkills
        );
        assert_eq!(
            required_permission("scheduler_add_job"),
            Permission::Schedule
        );
    }
}
