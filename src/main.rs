mod agents;
mod provider;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use rig::completion::{Chat, Message};
use tracing_subscriber::EnvFilter;

use crate::agents::peer::build_peer_coach;
use crate::provider::config::GenerationConfig;
use crate::provider::{CandleProvider, ModelRegistry};

#[derive(Parser)]
#[command(name = "chiron")]
#[command(about = "MI peer support chatbot powered by local LLMs")]
struct Args {
    /// Path to the peer coach GGUF model file
    #[arg(long)]
    model: PathBuf,

    /// Path to the tokenizer.json file (e.g. from HuggingFace)
    #[arg(long)]
    tokenizer: PathBuf,

    /// Maximum tokens to generate per response
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy, higher = more random)
    #[arg(long, default_value = "0.6")]
    temperature: f64,

    /// Run a single benchmark inference and exit. Value is the prompt to send.
    #[arg(long)]
    bench: Option<String>,
}

const COACH_MODEL_ID: &str = "peer-coach";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Load model into registry
    let mut registry = ModelRegistry::new()?;
    registry.load_model(
        COACH_MODEL_ID,
        &args.model,
        &args.tokenizer,
        GenerationConfig {
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            ..Default::default()
        },
    )?;

    // Create provider and agent
    let provider = CandleProvider::new(registry);
    let model = provider.completion_model(COACH_MODEL_ID);
    let agent = build_peer_coach(model);

    // Bench mode: single prompt, print timing, exit
    if let Some(prompt) = args.bench {
        println!("=== Benchmark Mode ===");
        println!("Prompt: {prompt}");
        println!("---");

        let t_start = Instant::now();
        let response = agent
            .chat(&prompt, vec![])
            .await
            .context("Benchmark inference failed")?;
        let total = t_start.elapsed();

        println!("\nChiron: {response}");
        println!("\n=== Benchmark Results ===");
        println!("Total agent.chat() time: {}ms", total.as_millis());
        println!("(See tracing output above for prefill/decode breakdown)");
        return Ok(());
    }

    println!("Chiron MI Peer Support");
    println!("Type your message, or 'quit' to exit.");
    println!("---");

    // Chat loop
    let mut chat_history: Vec<Message> = Vec::new();

    loop {
        // Read user input
        print!("\nYou: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("Failed to read input")?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Take care of yourself. Goodbye.");
            break;
        }

        // Send to agent with chat history
        let t_start = Instant::now();
        let response = agent
            .chat(input, chat_history.clone())
            .await
            .context("Failed to get response from agent")?;
        let elapsed = t_start.elapsed();

        println!("\nChiron: {response}");
        tracing::info!(response_ms = elapsed.as_millis() as u64, "Chat turn complete");

        // Update chat history
        chat_history.push(Message::user(input));
        chat_history.push(Message::assistant(&response));
    }

    Ok(())
}
