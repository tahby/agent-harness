mod agent;
mod config;
mod message;
mod provider;
mod tools;

use anyhow::{Context, Result};
use clap::Parser;

use crate::agent::run;
use crate::config::Config;
use crate::message::Message;
use crate::provider::OpenAiProvider;
use crate::tools::ToolRegistry;

#[derive(Parser, Debug)]
#[command(
    name = "agent-harness",
    about = "Send a prompt through a small tool-using chat loop."
)]
struct Cli {
    /// User prompt
    prompt: String,
}

const SYSTEM: &str = "You are a small local agent. Use tools when they help answer the user. Prefer read_file, write_file, and shell for work in the current directory.";

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::from_env();
    let provider = OpenAiProvider::new(&config).context("failed to build OpenAI client")?;
    let registry = ToolRegistry::builtins();
    let mut messages = vec![
        Message::System {
            text: SYSTEM.to_string(),
        },
        Message::User { text: cli.prompt },
    ];

    let text = run(&provider, &registry, &mut messages).await?;
    println!("{text}");
    Ok(())
}
