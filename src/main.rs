mod agents;
mod eval;
mod knowledge;
mod memory;
mod orchestrator;
mod provider;
mod router;

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
use crate::eval::catalog::{ModesCatalog, PromptCatalog};
use crate::orchestrator::Orchestrator;
use crate::provider::config::GenerationConfig;
use crate::provider::{CandleProvider, ModelArch, ModelRegistry};
use crate::router::ModeRouter;

#[derive(Parser)]
#[command(name = "chiron")]
#[command(about = "MI peer support chatbot powered by local LLMs")]
struct Args {
    /// Path to the GGUF model file
    #[arg(long, default_value = "models/smollm3-3b.gguf")]
    model: PathBuf,

    /// Path to the tokenizer.json file
    #[arg(long, default_value = "models/tokenizer-smollm3.json")]
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

    /// Run prompt evaluation matrix and exit
    #[arg(long)]
    eval: bool,

    /// Path to the test script TOML for eval mode
    #[arg(long, default_value = "prompts/test_scripts/standard_5turn.toml")]
    eval_script: PathBuf,

    /// Path to coach prompt variants TOML
    #[arg(long, default_value = "prompts/coach.toml")]
    coach_variants: PathBuf,

    /// Path to supervisor prompt variants TOML
    #[arg(long, default_value = "prompts/supervisor.toml")]
    supervisor_variants: PathBuf,

    /// Path to conversation modes TOML
    #[arg(long, default_value = "prompts/modes.toml")]
    modes_catalog: PathBuf,

    /// Output directory for eval results JSON
    #[arg(long, default_value = "evals")]
    eval_output: PathBuf,

    /// Coach variant ID to use for interactive/bench mode (default: first variant in catalog)
    #[arg(long)]
    coach_variant: Option<String>,

    /// Supervisor variant ID to use for interactive mode (default: first variant in catalog)
    #[arg(long)]
    supervisor_variant: Option<String>,

    /// Model architecture override. Auto-detected from GGUF metadata when not set.
    /// Supported: smollm3, qwen3
    #[arg(long)]
    model_arch: Option<String>,
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

    // Load prompt catalogs
    let coach_catalog = PromptCatalog::load(&args.coach_variants)
        .context("Failed to load coach prompt catalog")?;
    let supervisor_catalog = PromptCatalog::load(&args.supervisor_variants)
        .context("Failed to load supervisor prompt catalog")?;
    let modes_catalog = ModesCatalog::load(&args.modes_catalog)
        .context("Failed to load modes catalog")?;

    // Select variants for interactive/bench mode
    let coach_variant = match &args.coach_variant {
        Some(id) => coach_catalog.get_variant(id)?.clone(),
        None => coach_catalog.variants.first()
            .context("Coach catalog has no variants")?.clone(),
    };
    let supervisor_variant = match &args.supervisor_variant {
        Some(id) => supervisor_catalog.get_variant(id)?.clone(),
        None => supervisor_catalog.variants.first()
            .context("Supervisor catalog has no variants")?.clone(),
    };

    tracing::info!(
        coach = &coach_variant.id,
        supervisor = &supervisor_variant.id,
        "Selected prompt variants"
    );

    // Canonicalize model/tokenizer paths so symlinks resolve for the GGUF loader
    let model_path = args.model.canonicalize().with_context(|| {
        format!("Model file not found: {}", args.model.display())
    })?;
    let tokenizer_path = args.tokenizer.canonicalize().with_context(|| {
        format!("Tokenizer file not found: {}", args.tokenizer.display())
    })?;

    // Determine model architecture (explicit override or auto-detect from GGUF)
    let arch = match &args.model_arch {
        Some(s) => Some(s.parse::<ModelArch>()?),
        None => None,
    };

    // Use architecture-specific EOS tokens (auto-detected if arch is None)
    let detected_arch = match arch {
        Some(a) => a,
        None => ModelArch::detect_from_gguf(&model_path)?,
    };
    let eos_token_ids = detected_arch.default_eos_token_ids().to_vec();

    // Load single model into registry
    let mut registry = ModelRegistry::new()?;
    registry.load_model(
        COACH_MODEL_ID,
        &model_path,
        &tokenizer_path,
        arch,
        GenerationConfig {
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            eos_token_ids,
            ..Default::default()
        },
        args.chat_template.as_deref(),
    )?;
    let provider = CandleProvider::new(registry);

    // Bench mode: single prompt, no RAG, no case notes, print timing, exit
    if let Some(prompt) = args.bench {
        let completion_model = provider.completion_model(COACH_MODEL_ID);
        let preamble = crate::agents::peer::build_peer_coach_preamble(
            &coach_variant.preamble,
            None,
            None,
        );
        let agent = build_peer_coach(
            completion_model,
            &preamble,
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

    // Eval mode: run prompt evaluation matrix and exit
    if args.eval {
        let eval_model = provider.completion_model(COACH_MODEL_ID);
        return eval::run_eval(
            eval_model,
            knowledge_store,
            embedding_model,
            args.rag_top_k,
            chat_conn,
            &args.coach_variants,
            &args.supervisor_variants,
            &args.eval_script,
            &args.eval_output,
        )
        .await;
    }

    // Build semantic router from modes catalog
    let mode_router = ModeRouter::from_catalog(&modes_catalog, embedding_model.clone())
        .await
        .context("Failed to build mode router")?;

    // Build two completion model handles from the same provider
    let peer_coach_model = provider.completion_model(COACH_MODEL_ID);
    let supervisor_model = provider.completion_model(COACH_MODEL_ID);

    let supervisor = build_supervisor(
        supervisor_model,
        &supervisor_variant.preamble,
        supervisor_variant.temperature,
        supervisor_variant.max_tokens,
    );

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
        coach_variant.clone(),
        mode_router,
        modes_catalog,
        knowledge_store,
        embedding_model,
        args.rag_top_k,
        session_id,
        chat_conn,
    );

    println!("Chiron MI Peer Support (Mode-Routed + Case Notes)");
    println!("Coach: {} | Supervisor: {}", coach_variant.id, supervisor_variant.id);
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
