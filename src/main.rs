mod agents;
mod knowledge;
mod memory;
mod orchestrator;
mod provider;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use rig::completion::Chat;
use rig::embeddings::{EmbeddingModel as _, EmbeddingsBuilder};
use tracing_subscriber::EnvFilter;

use crate::agents::case_notes::build_supervisor;
use crate::agents::peer::build_peer_coach;
use crate::orchestrator::Orchestrator;
use crate::provider::config::GenerationConfig;
use crate::provider::{CandleProvider, ModelRegistry, SMOLLM3_EOS_TOKEN_IDS};

#[derive(Parser)]
#[command(name = "chiron")]
#[command(about = "MI peer support chatbot powered by local LLMs")]
struct Args {
    /// Path to the GGUF model file (8B recommended)
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

    /// Path to SQLite database file for memory and RAG
    #[arg(long, default_value = "chiron.db")]
    db_path: String,

    /// Path to Plotinus knowledge_base/ directory. Seeds MI knowledge into the database.
    #[arg(long)]
    seed_knowledge: Option<PathBuf>,

    /// Number of RAG results to inject as context per query
    #[arg(long, default_value = "5")]
    rag_top_k: usize,

    /// Path to tokenizer_config.json or chat_template.jinja for HF chat template.
    /// When provided, uses the model's native Jinja2 template instead of built-in presets.
    #[arg(long)]
    chat_template: Option<PathBuf>,

    /// Maximum tokens for the peer coach response (default: 256).
    /// Separate from --max-tokens which controls the model-level default.
    #[arg(long, default_value = "256")]
    coach_max_tokens: usize,
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

    // Load single model into registry
    let mut registry = ModelRegistry::new()?;
    registry.load_model(
        COACH_MODEL_ID,
        &args.model,
        &args.tokenizer,
        GenerationConfig {
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            eos_token_ids: SMOLLM3_EOS_TOKEN_IDS.to_vec(),
            ..Default::default()
        },
        args.chat_template.as_deref(),
    )?;
    let provider = CandleProvider::new(registry);

    // Bench mode: single prompt, no RAG, no case notes, print timing, exit
    if let Some(prompt) = args.bench {
        let completion_model = provider.completion_model(COACH_MODEL_ID);
        let agent = build_peer_coach(completion_model);
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

    // --- Interactive mode: initialize RAG, memory, and case notes ---

    // Initialize sqlite-vec extension (must happen before any DB connections)
    memory::init_sqlite_vec();

    // Initialize fastembed for local embeddings (CPU, ONNX)
    tracing::info!("Initializing embedding model (first run downloads ~20MB)...");
    let fastembed_client = rig_fastembed::Client::new();
    let embedding_model =
        fastembed_client.embedding_model(&rig_fastembed::FastembedModel::BGESmallENV15);
    tracing::info!(
        dims = embedding_model.ndims(),
        "Embedding model ready"
    );

    // Open database: one vector store + chat connection
    let (knowledge_store, chat_conn) =
        memory::open_memory(&args.db_path, &embedding_model).await?;

    // Seed MI knowledge if requested
    if let Some(kb_path) = &args.seed_knowledge {
        let chunks = knowledge::seed::load_mi_principles(kb_path)?;
        tracing::info!(chunks = chunks.len(), "Embedding MI knowledge chunks...");

        let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
            .documents(chunks)?
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to embed MI knowledge: {e}"))?;

        let inserted = knowledge_store
            .add_rows(embeddings)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to insert MI knowledge: {e}"))?;
        tracing::info!(rows = inserted, "MI knowledge seeded into database");
    }

    // Build two completion model handles from the same provider
    let peer_coach_model = provider.completion_model(COACH_MODEL_ID);
    let supervisor_model = provider.completion_model(COACH_MODEL_ID);

    let supervisor = build_supervisor(supervisor_model);

    // Generate session ID
    let session_id = format!(
        "session_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs()
    );
    tracing::info!(session_id, "Starting interactive session");

    let mut orchestrator = Orchestrator::new(
        peer_coach_model,
        supervisor,
        knowledge_store,
        embedding_model,
        args.rag_top_k,
        args.coach_max_tokens,
        session_id,
        chat_conn,
    );

    println!("Chiron MI Peer Support (SmolLM3 + Case Notes)");
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

        // EOF — stdin closed (e.g., piped input exhausted)
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
