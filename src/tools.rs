use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("path escapes the working directory: {0}")]
    PathEscape(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("command failed with status {status}: {stderr}")]
    CommandFailed { status: i32, stderr: String },
}

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn call(&self, arguments: Value) -> Result<String, ToolError>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(ReadFile));
        registry.register(Box::new(WriteFile));
        registry.register(Box::new(Shell));
        registry
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.spec().name;
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }
}

fn object_schema(properties: Value, required: &[&str]) -> Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn scoped_path(raw: &str) -> Result<PathBuf, ToolError> {
    let path = Path::new(raw);
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(ToolError::PathEscape(raw.to_string()));
    }
    Ok(path.to_path_buf())
}

fn parse_args<T: for<'de> Deserialize<'de>>(arguments: Value) -> Result<T, ToolError> {
    serde_json::from_value(arguments)
        .map_err(|err| ToolError::InvalidArguments(err.to_string()))
}

struct ReadFile;

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
}

impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a UTF-8 file relative to the process working directory.".into(),
            parameters: object_schema(
                serde_json::json!({
                    "path": { "type": "string", "description": "Path to read." }
                }),
                &["path"],
            ),
        }
    }

    fn call(&self, arguments: Value) -> Result<String, ToolError> {
        let args: ReadArgs = parse_args(arguments)?;
        let path = scoped_path(&args.path)?;
        Ok(std::fs::read_to_string(path)?)
    }
}

struct WriteFile;

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    contents: String,
}

impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Write a UTF-8 file relative to the process working directory.".into(),
            parameters: object_schema(
                serde_json::json!({
                    "path": { "type": "string", "description": "Path to write." },
                    "contents": { "type": "string", "description": "File contents." }
                }),
                &["path", "contents"],
            ),
        }
    }

    fn call(&self, arguments: Value) -> Result<String, ToolError> {
        let args: WriteArgs = parse_args(arguments)?;
        let path = scoped_path(&args.path)?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path, args.contents)?;
        Ok(format!("wrote {}", path.display()))
    }
}

struct Shell;

#[derive(Deserialize)]
struct ShellArgs {
    command: String,
}

impl Tool for Shell {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell".into(),
            description: "Run a shell command in the process working directory.".into(),
            parameters: object_schema(
                serde_json::json!({
                    "command": { "type": "string", "description": "Command passed to sh -c." }
                }),
                &["command"],
            ),
        }
    }

    fn call(&self, arguments: Value) -> Result<String, ToolError> {
        let args: ShellArgs = parse_args(arguments)?;
        let output = Command::new("sh").arg("-c").arg(&args.command).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            let status = output.status.code().unwrap_or(-1);
            return Err(ToolError::CommandFailed {
                status,
                stderr: stderr.into_owned(),
            });
        }
        let mut combined = stdout.into_owned();
        if !stderr.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(&stderr);
        }
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_parent_dir_on_read() {
        let tool = ReadFile;
        let err = tool
            .call(json!({ "path": "../secret" }))
            .expect_err("path escape should fail");
        assert!(matches!(err, ToolError::PathEscape(p) if p == "../secret"));
    }

    #[test]
    fn rejects_parent_dir_on_write() {
        let tool = WriteFile;
        let err = tool
            .call(json!({ "path": "../secret", "contents": "nope" }))
            .expect_err("path escape should fail");
        assert!(matches!(err, ToolError::PathEscape(p) if p == "../secret"));
    }
}
