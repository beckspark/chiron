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
use crate::catalog::{ModeCatalog, PromptCatalog};
use crate::orchestrator::Orchestrator;
use crate::provider::config::GenerationConfig;
use crate::provider::LlamaCppProvider;

/// A scripted test conversation loaded from TOML.
#[derive(serde::Deserialize)]
struct TestScript {
    id: String,
    description: String,
    turns: Vec<TestTurn>,
}

/// A single turn in a test script.
#[derive(serde::Deserialize)]
struct TestTurn {
    input: String,
    notes: String,
    expected_mode: Option<String>,
}

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

    /// Run a scripted test conversation from a TOML file and output JSON results.
    #[arg(long)]
    script: Option<PathBuf>,

    /// Path to SQLite database file for chat history + case notes
    #[arg(long, default_value = "chiron.db")]
    db_path: String,

    /// Path to coach prompt variants TOML
    #[arg(long, default_value = "prompts/coach.toml")]
    coach_variants: PathBuf,

    /// Path to conversation modes TOML
    #[arg(long, default_value = "prompts/modes.toml")]
    modes: PathBuf,

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

    /// Path to LanceDB vector store directory
    #[arg(long, default_value = "chiron_vectors")]
    lance_db_path: String,

    /// Number of conversation turns to keep in the sliding window
    #[arg(long, default_value = "4")]
    history_turns: usize,

    /// Total context window size in tokens (for RAG budget calculation)
    #[arg(long, default_value = "4096")]
    context_size: usize,

    /// Default top-k results per RAG collection
    #[arg(long, default_value = "3")]
    rag_top_k: usize,
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
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());

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

    // Load mode catalog (optional — degrades gracefully if missing)
    let mode_catalog = ModeCatalog::load(&args.modes).ok();
    if mode_catalog.is_some() {
        tracing::info!("Loaded conversation modes from {}", args.modes.display());
    }

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

    // --- Script mode: run test conversation, output JSON ---
    if let Some(script_path) = &args.script {
        let script_content = std::fs::read_to_string(script_path)
            .with_context(|| format!("Failed to read script: {}", script_path.display()))?;
        let script: TestScript = toml::from_str(&script_content)
            .with_context(|| format!("Failed to parse script: {}", script_path.display()))?;

        let db_path = format!(":memory:"); // In-memory DB for scripted runs
        let chat_conn = memory::open_memory(&db_path).await?;
        let completion_model = crate::provider::completion_model(&provider, config);

        let session_id = format!("script_{}", script.id);
        let mut orchestrator = Orchestrator::new(
            completion_model,
            coach_variant.clone(),
            coach_catalog.think_instructions.clone(),
            mode_catalog,
            session_id,
            chat_conn,
            true, // always show thinking in script mode
            args.history_turns,
            None, // no vector store in script mode (for now)
            None, // no embedding model in script mode (for now)
            args.rag_top_k,
        );
        orchestrator.set_output_to_stderr(true);

        eprintln!("=== Script Mode: {} ===", script.id);
        eprintln!("Description: {}", script.description);
        eprintln!("Coach: {}", coach_variant.id);
        eprintln!("Turns: {}", script.turns.len());
        eprintln!("---");

        let run_start = Instant::now();
        let mut turn_results = Vec::new();

        for (i, turn) in script.turns.iter().enumerate() {
            eprintln!("\n--- Turn {} ---", i + 1);
            eprintln!("Input: {}", turn.input);
            if let Some(ref mode) = turn.expected_mode {
                eprintln!("Expected mode: {mode}");
            }
            eprintln!("Notes: {}", turn.notes);

            let result = orchestrator
                .run_turn_captured(&turn.input)
                .await
                .with_context(|| format!("Turn {} failed", i + 1))?;

            eprintln!("Case notes: {}", result.case_notes.as_deref().unwrap_or("none"));

            turn_results.push(serde_json::json!({
                "turn_number": result.turn_number,
                "input": result.input,
                "response": result.response,
                "think_content": result.think_content,
                "case_notes": result.case_notes,
                "expected_mode": turn.expected_mode,
                "script_notes": turn.notes,
                "duration_ms": result.duration_ms,
            }));
        }

        let output = serde_json::json!({
            "script_id": script.id,
            "description": script.description,
            "coach_variant": coach_variant.id,
            "total_duration_ms": run_start.elapsed().as_millis() as u64,
            "turns": turn_results,
        });

        // Write JSON to stdout (eprintln used for progress above)
        println!("{}", serde_json::to_string_pretty(&output)?);

        return Ok(());
    }

    // --- Interactive mode ---

    // Initialize vector store + embedding model
    let vector_conn = memory::vectors::open_vector_db(&args.lance_db_path).await?;
    memory::vectors::ensure_tables(&vector_conn).await?;
    let embedding_model = memory::embeddings::init_embedding_model();

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
        coach_catalog.think_instructions.clone(),
        mode_catalog,
        session_id,
        chat_conn,
        args.show_thinking,
        args.history_turns,
        Some(vector_conn),
        Some(embedding_model),
        args.rag_top_k,
    );

    println!("Chiron MI Peer Support (Plotinus V19 + llama.cpp)");
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
