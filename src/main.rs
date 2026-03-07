mod agents;
mod catalog;
mod memory;
mod orchestrator;
mod provider;
mod router;
mod supervision;

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use rig::completion::Chat;
use tracing_subscriber::EnvFilter;

use crate::agents::peer::build_peer_coach;
use crate::catalog::PromptCatalog;
use crate::orchestrator::Orchestrator;
use crate::provider::config::GenerationConfig;
use crate::provider::LlamaCppProvider;

#[derive(Parser)]
#[command(name = "chiron")]
#[command(about = "MI peer support chatbot powered by Plotinus (llama.cpp)")]
struct Args {
    /// Path to the GGUF model file
    #[arg(long, default_value = "models/plotinus.gguf")]
    model: PathBuf,

    /// Maximum tokens to generate per response
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy, higher = more random)
    #[arg(long, default_value = "0.7")]
    temperature: f64,

    /// Run a single benchmark inference and exit. Value is the prompt to send.
    #[arg(long)]
    bench: Option<String>,

    /// Path to SQLite database file for chat history + case notes
    #[arg(long, default_value = "chiron.db")]
    db_path: String,

    /// Path to coach prompt variants TOML
    #[arg(long, default_value = "prompts/coach.toml")]
    coach_variants: PathBuf,

    /// Coach variant ID to use (default: first variant in catalog)
    #[arg(long)]
    coach_variant: Option<String>,

    /// Number of GPU layers to offload (default: 99 = all)
    #[arg(long, default_value = "99")]
    n_gpu_layers: u32,

    /// Show the model's internal <think> block reasoning after each response (default: on)
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    show_thinking: bool,

    /// Enable verbose logging (tracing info/debug output)
    #[arg(long, short)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let default_level = if args.verbose { "info" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level)),
        )
        .init();

    // Load prompt catalog
    let coach_catalog = PromptCatalog::load(&args.coach_variants)
        .context("Failed to load coach prompt catalog")?;

    let coach_variant = match &args.coach_variant {
        Some(id) => coach_catalog.get_variant(id)?.clone(),
        None => coach_catalog
            .variants
            .first()
            .context("Coach catalog has no variants")?
            .clone(),
    };

    tracing::info!(coach = &coach_variant.id, "Selected prompt variant");

    // Resolve model path (symlinks)
    let model_path = args.model.canonicalize().with_context(|| {
        format!("Model file not found: {}", args.model.display())
    })?;

    // Initialize llama.cpp provider
    let provider = Arc::new(Mutex::new(
        LlamaCppProvider::new(&model_path, args.n_gpu_layers)
            .context("Failed to initialize llama.cpp provider")?,
    ));

    let config = GenerationConfig {
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        ..Default::default()
    };

    // Bench mode: single prompt, no DB, print timing, exit
    if let Some(prompt) = args.bench {
        let completion_model = crate::provider::completion_model(&provider, config.clone());
        let agent = build_peer_coach(
            completion_model,
            &coach_variant.preamble,
            coach_variant.temperature,
            coach_variant.max_tokens,
        );

        println!("=== Benchmark Mode ===");
        println!("Coach variant: {}", coach_variant.id);
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
        return Ok(());
    }

    // --- Interactive mode ---

    let chat_conn = memory::open_memory(&args.db_path).await?;

    let completion_model = crate::provider::completion_model(&provider, config);

    // Generate session ID
    let session_id = format!(
        "session_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs()
    );
    tracing::info!(session_id, "Starting interactive session");

    let mut orchestrator = Orchestrator::new(
        completion_model,
        coach_variant.clone(),
        session_id,
        chat_conn,
        args.show_thinking,
    );

    println!("Chiron MI Peer Support (Plotinus V18 + llama.cpp)");
    println!("Coach: {}", coach_variant.id);
    println!("Type your message, or 'quit' to exit. 'reset' clears conversation.");
    println!("---");

    // Chat loop
    loop {
        print!("\nYou: ");
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin()
            .read_line(&mut input)
            .context("Failed to read input")?;

        if bytes_read == 0 {
            println!("Take care of yourself. Goodbye.");
            break;
        }

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Take care of yourself. Goodbye.");
            break;
        }

        if input.eq_ignore_ascii_case("reset") {
            orchestrator.reset();
            println!("Conversation reset.");
            continue;
        }

        orchestrator
            .run_turn(input)
            .await
            .context("Turn failed")?;
    }

    Ok(())
}
