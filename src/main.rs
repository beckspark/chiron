mod agents;
mod knowledge;
mod memory;
mod provider;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::completion::{Chat, Message};
use rig::embeddings::{EmbeddingModel as _, EmbeddingsBuilder};
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use tracing_subscriber::EnvFilter;

use crate::agents::peer::{build_peer_coach, build_peer_coach_with_rag};
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

    /// Path to SQLite database file for memory and RAG
    #[arg(long, default_value = "chiron.db")]
    db_path: String,

    /// Path to Plotinus knowledge_base/ directory. Seeds MI knowledge into the database.
    #[arg(long)]
    seed_knowledge: Option<PathBuf>,

    /// Number of RAG results to inject as context per query
    #[arg(long, default_value = "5")]
    rag_top_k: usize,
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

    // Load LLM into registry
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
    let provider = CandleProvider::new(registry);
    let completion_model = provider.completion_model(COACH_MODEL_ID);

    // Bench mode: single prompt, no RAG, print timing, exit
    if let Some(prompt) = args.bench {
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

    // --- Interactive mode: initialize RAG and memory ---

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

    // Open database and create vector store
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

    // Create search index (consumes store) and build RAG-enabled agent
    let knowledge_index = knowledge_store.index(embedding_model);
    let agent = build_peer_coach_with_rag(completion_model, knowledge_index, args.rag_top_k);

    // Generate session ID
    let session_id = format!(
        "session_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs()
    );
    tracing::info!(session_id, "Starting interactive session");

    println!("Chiron MI Peer Support (with RAG)");
    println!("Type your message, or 'quit' to exit. 'reset' clears conversation.");
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

        if input.eq_ignore_ascii_case("reset") {
            chat_history.clear();
            println!("Conversation reset.");
            continue;
        }

        // Stream response token by token
        let t_start = Instant::now();
        print!("\nChiron: ");
        io::stdout().flush()?;

        let mut stream = agent.stream_chat(input, chat_history.clone()).await;

        let mut full_response = String::new();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::Text(text),
                )) => {
                    print!("{}", text.text);
                    io::stdout().flush()?;
                    full_response.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(final_resp)) => {
                    // FinalResponse contains the aggregated response
                    if full_response.is_empty() {
                        full_response = final_resp.response().to_string();
                        print!("{full_response}");
                        io::stdout().flush()?;
                    }
                }
                Err(e) => {
                    eprintln!("\nStreaming error: {e}");
                    break;
                }
                _ => {}
            }
        }

        println!();
        let elapsed = t_start.elapsed();
        tracing::info!(response_ms = elapsed.as_millis() as u64, "Chat turn complete");

        // Save turn to database
        memory::save_chat_turn(&chat_conn, &session_id, "user", input).await?;
        memory::save_chat_turn(&chat_conn, &session_id, "assistant", &full_response).await?;

        // Update chat history
        chat_history.push(Message::user(input));
        chat_history.push(Message::assistant(&full_response));
    }

    Ok(())
}
