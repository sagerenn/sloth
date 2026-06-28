//! Skill system with hot reload.
//!
//! A **skill** is a markdown file with YAML frontmatter describing a named,
//! self-contained capability the agent can invoke. Each skill has:
//!
//! ```text
//! ---
//! name: my-skill            # stable id; must match a-z0-9_-
//! description: ...          # when/how to use it
//! arguments:                # optional JSON-schema-ish arg list
//!   - name: query
//!     description: ...
//!     required: true
//! ---
//! <body: instructions / template the agent fills>
//! ```
//!
//! Skills are surfaced to the LLM as invocable tools `skill_<name>`. Running a
//! skill returns its (template-expanded) body as the tool result, which the
//! model then acts on. The [`SkillRegistry`] hot-reloads: it re-scans the
//! skill directory on demand ([`SkillRegistry::reload`]) and the runtime
//! triggers rescans on a poll interval, so adding/editing/removing a skill
//! file takes effect without restarting the agent.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// One declared argument of a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillArg {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub required: Option<bool>,
}

/// Parsed skill frontmatter.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SkillMeta {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub arguments: Vec<SkillArg>,
}

/// A loaded skill.
#[derive(Debug, Clone, Serialize)]
pub struct Skill {
    pub name: String,
    pub description: Option<String>,
    /// The body markdown (template, with `{{arg}}` placeholders).
    pub body: String,
    /// Declared arguments.
    #[serde(skip_serializing)]
    pub arguments: Vec<SkillArg>,
    /// Source file path (for diagnostics).
    #[serde(skip_serializing)]
    pub source: Option<PathBuf>,
}

impl Skill {
    /// JSON Schema describing this skill's arguments (for the LLM tool def).
    pub fn input_schema(&self) -> Value {
        let mut props = Map::new();
        let mut required = Vec::new();
        for a in &self.arguments {
            let mut s = Map::new();
            s.insert("type".to_string(), Value::String("string".to_string()));
            if let Some(d) = &a.description {
                s.insert("description".to_string(), Value::String(d.clone()));
            }
            props.insert(a.name.clone(), Value::Object(s));
            if a.required.unwrap_or(false) {
                required.push(Value::String(a.name.clone()));
            }
        }
        json!({
            "type": "object",
            "properties": Value::Object(props),
            "required": required,
        })
    }

    /// Instantiate the skill body, substituting `{{arg}}` placeholders with
    /// the matching argument from `args`. Unknown placeholders are left as-is.
    pub fn instantiate(&self, args: &Value) -> String {
        let mut out = self.body.clone();
        for a in &self.arguments {
            let val = args.get(&a.name).and_then(|v| v.as_str()).unwrap_or("");
            let pat = format!("{{{{{}}}}}", a.name); // {{arg}}
            out = out.replace(&pat, val);
        }
        out
    }
}

/// Registry of loaded skills. Cloneable (state shared behind an `Arc`).
#[derive(Clone)]
pub struct SkillRegistry {
    inner: Arc<Mutex<SkillState>>,
}

#[derive(Default)]
struct SkillState {
    skills: Vec<Skill>,
    dir: Option<PathBuf>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SkillState::default())),
        }
    }

    /// Set the skill directory (does not load — call [`Self::reload`]).
    pub async fn set_dir(&self, dir: impl Into<PathBuf>) {
        self.inner.lock().await.dir = Some(dir.into());
    }

    /// Hot reload: re-scan the skill directory. Removes skills whose source
    /// file is gone, parses present files. Returns the count loaded.
    pub async fn reload(&self) -> Result<usize> {
        let dir = {
            let g = self.inner.lock().await;
            g.dir.clone()
        };
        let Some(dir) = dir else {
            // No directory configured → clear and return 0.
            let mut g = self.inner.lock().await;
            g.skills.clear();
            return Ok(0);
        };

        let mut skills = Vec::new();
        if dir.exists() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
                .with_context(|| format!("reading skills dir {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension().and_then(|e| e.to_str()) == Some("md")
                        || p.extension().and_then(|e| e.to_str()) == Some("markdown")
                })
                .collect();
            entries.sort();
            for path in entries {
                match parse_skill_file(&path) {
                    Ok(mut s) => {
                        s.source = Some(path.clone());
                        skills.push(s);
                    }
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "failed to parse skill file");
                    }
                }
            }
        }
        let count = skills.len();
        let mut g = self.inner.lock().await;
        g.skills = skills;
        info!(dir = %dir.display(), count, "skills hot-reloaded");
        Ok(count)
    }

    /// All loaded skills.
    pub async fn list(&self) -> Vec<Skill> {
        self.inner.lock().await.skills.clone()
    }

    /// Look up a skill by name.
    pub async fn get(&self, name: &str) -> Option<Skill> {
        self.inner
            .lock()
            .await
            .skills
            .iter()
            .find(|s| s.name == name)
            .cloned()
    }

    /// Instantiate a skill by name with the given arguments.
    pub async fn invoke(&self, name: &str, args: &Value) -> Result<String> {
        let skill = self
            .get(name)
            .await
            .ok_or_else(|| anyhow!("unknown skill: {name}"))?;
        Ok(skill.instantiate(args))
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a skill from a markdown file: `---\n<yaml frontmatter>\n---\n<body>`.
pub fn parse_skill_file(path: &Path) -> Result<Skill> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading skill file {}", path.display()))?;
    parse_skill_str(&text)
}

/// Parse a skill from its raw markdown text.
pub fn parse_skill_str(text: &str) -> Result<Skill> {
    let text = text.trim_start_matches('\u{feff}').trim_start();
    if let Some(rest) = text.strip_prefix("---") {
        // Find the closing `---` on its own line.
        let closer_rel = rest
            .find("\n---")
            .ok_or_else(|| anyhow!("skill frontmatter missing closing '---'"))?;
        let fm = rest[..closer_rel].trim_start_matches('\n');
        let body = &rest[closer_rel + "\n---".len()..];
        let body = body.trim_start_matches('\n');
        let meta: SkillMeta = serde_yamlish(fm).context("parsing skill frontmatter")?;
        return build_skill(meta, body);
    }
    // No frontmatter → nothing useful (name required). Error out.
    anyhow::bail!("skill missing frontmatter (expected leading '---')")
}

/// Very small frontmatter parser using the `toml`-style key parsing is risky;
/// instead we parse a constrained YAML subset by hand (key: value, lists with
/// `- name:` items). This avoids pulling a YAML dependency for the few fields
/// skills need.
fn serde_yamlish(fm: &str) -> Result<SkillMeta> {
    let mut meta = SkillMeta::default();
    let mut lines = fm.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim_end();
        if line.is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let key = k.trim();
        let val = v.trim();
        match key {
            "name" => meta.name = unquote(val).to_string(),
            "description" => meta.description = Some(unquote(val).to_string()),
            "arguments" => {
                // Subsequent indented `- name: ...` list items, each possibly
                // followed by further-indented `key: value` continuation lines.
                while let Some(next) = lines.peek() {
                    let trimmed = next.trim_start();
                    if !trimmed.starts_with("- ") {
                        break;
                    }
                    let item = lines.next().unwrap();
                    let item_body = item
                        .trim_start()
                        .strip_prefix("- ")
                        .unwrap_or_default()
                        .trim_start();
                    let mut arg = SkillArg {
                        name: String::new(),
                        description: None,
                        required: None,
                    };
                    // The item line may be `- name: x` (and nothing more) or
                    // carry inline fields separated by commas.
                    for field in item_body.split(',') {
                        if let Some((fk, fv)) = field.split_once(':') {
                            match fk.trim() {
                                "name" => arg.name = unquote(fv.trim()).to_string(),
                                "description" => {
                                    arg.description = Some(unquote(fv.trim()).to_string())
                                }
                                "required" => {
                                    arg.required =
                                        Some(matches!(unquote(fv.trim()), "true" | "yes" | "1"))
                                }
                                _ => {}
                            }
                        } else if arg.name.is_empty() {
                            arg.name = unquote(field.trim()).to_string();
                        }
                    }
                    // Continuation lines (deeper indent, not a new item).
                    while let Some(peek) = lines.peek() {
                        let p = peek.trim_start();
                        let indented = peek.starts_with(' ') || peek.starts_with('\t');
                        if !indented || p.starts_with("- ") || p.is_empty() {
                            break;
                        }
                        let cont = lines.next().unwrap();
                        if let Some((fk, fv)) = cont.trim().split_once(':') {
                            match fk.trim() {
                                "name" => arg.name = unquote(fv.trim()).to_string(),
                                "description" => {
                                    arg.description = Some(unquote(fv.trim()).to_string())
                                }
                                "required" => {
                                    arg.required =
                                        Some(matches!(unquote(fv.trim()), "true" | "yes" | "1"))
                                }
                                _ => {}
                            }
                        }
                    }
                    if !arg.name.is_empty() {
                        meta.arguments.push(arg);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(meta)
}

fn unquote(s: &str) -> &str {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn build_skill(meta: SkillMeta, body: &str) -> Result<Skill> {
    if meta.name.trim().is_empty() {
        anyhow::bail!("skill is missing a 'name' in its frontmatter");
    }
    Ok(Skill {
        name: meta.name,
        description: meta.description,
        body: body.to_string(),
        arguments: meta.arguments,
        source: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skill_with_args() {
        let md = "---\nname: summarizer\ndescription: Summarize text\narguments:\n  - name: text\n    description: text to summarize\n    required: true\n  - name: length\n    required: false\n---\nSummarize the following in {{length}} words:\n{{text}}\n";
        let s = parse_skill_str(md).unwrap();
        assert_eq!(s.name, "summarizer");
        assert_eq!(s.description.as_deref(), Some("Summarize text"));
        assert_eq!(s.arguments.len(), 2);
        assert!(s.arguments[0].required.unwrap_or(false));
        let out = s.instantiate(&json!({ "text": "hello world", "length": "10" }));
        assert!(out.contains("in 10 words:"));
        assert!(out.contains("hello world"));
    }

    #[test]
    fn parse_skill_missing_name_fails() {
        let md = "---\ndescription: no name\n---\nbody\n";
        assert!(parse_skill_str(md).is_err());
    }

    #[tokio::test]
    async fn registry_reload_from_dir() {
        let dir = std::env::temp_dir().join(format!("sloth-skills-{}", uuid_str()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.md"),
            "---\nname: alpha\ndescription: a\n---\nbody-a\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("b.md"),
            "---\nname: beta\ndescription: b\n---\nbody-b\n",
        )
        .unwrap();
        let reg = SkillRegistry::new();
        reg.set_dir(&dir).await;
        assert_eq!(reg.reload().await.unwrap(), 2);
        assert!(reg.get("alpha").await.is_some());
        assert!(reg.get("beta").await.is_some());

        // Hot reload: remove one, add one.
        std::fs::remove_file(dir.join("a.md")).unwrap();
        std::fs::write(
            dir.join("c.md"),
            "---\nname: gamma\ndescription: g\n---\nbody-c\n",
        )
        .unwrap();
        assert_eq!(reg.reload().await.unwrap(), 2);
        assert!(reg.get("alpha").await.is_none());
        assert!(reg.get("gamma").await.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn uuid_str() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    }
}
