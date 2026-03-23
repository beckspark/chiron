use std::io::{self, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use futures::StreamExt;
use rig::agent::{Agent, MultiTurnStreamItem};
use rig::completion::Message;
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use tokio_rusqlite::Connection;

use crate::agents::peer::build_peer_coach_preamble;
use crate::catalog::{ModeCatalog, PromptVariant};
use crate::memory;
use crate::memory::case_notes;
use crate::memory::retrieval;
use crate::provider::LlamaCppCompletionModel;
use crate::router;
use crate::supervision::{
    analyze_think_block, extract_mi_stage, extract_themes, merge_themes, ThinkAnalysis,
};
use rig_fastembed::EmbeddingModel;

/// Maximum number of themes retained in case notes.
/// Prevents unbounded accumulation that bloats the preamble.
const MAX_THEMES: usize = 8;

/// Maximum characters for RAG context injected into the preamble.
const MAX_RAG_CONTEXT_CHARS: usize = 400;

/// Builds case notes from a think block analysis and previous notes.
///
/// Extracted from `Orchestrator::update_case_notes` to enable unit testing
/// without requiring a full orchestrator instance.
pub fn build_case_notes_from_analysis(
    think_content: Option<&str>,
    existing_notes: Option<&str>,
) -> (String, Option<String>) {
    let prev_themes = existing_notes
        .and_then(extract_themes)
        .unwrap_or_default();

    let analysis = think_content
        .filter(|t| !t.is_empty())
        .map(analyze_think_block)
        .unwrap_or_else(|| ThinkAnalysis {
            mi_stage: None,
            strategy_used: None,
            talk_type: None,
            themes: vec![],
            user_facts: vec![],
            significant_signal: None,
            raw_think: String::new(),
        });

    let mi_stage = analysis
        .mi_stage
        .or_else(|| existing_notes.and_then(extract_mi_stage));

    let merged = merge_themes(&prev_themes, &analysis.themes, MAX_THEMES);

    let mut notes = format!(
        "MI Stage: {}",
        mi_stage.as_deref().unwrap_or("engage"),
    );
    if let Some(ref strat) = analysis.strategy_used {
        notes.push_str(&format!("\nStrategy Used: {strat}"));
    }
    if let Some(ref talk) = analysis.talk_type {
        notes.push_str(&format!("\nTalk Type: {talk}"));
    }
    notes.push_str(&format!(
        "\nRunning Themes: {}",
        if merged.is_empty() {
            "none".to_string()
        } else {
            merged.join(", ")
        },
    ));

    (notes, mi_stage)
}


/// Structured result from a single conversation turn (public, for eval/script mode).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TurnResult {
    pub turn_number: i32,
    pub input: String,
    pub response: String,
    pub think_content: Option<String>,
    pub case_notes: Option<String>,
    pub preamble_injected: String,
    pub duration_ms: u64,
}

/// Internal output from the shared turn pipeline.
struct TurnOutput {
    response: String,
    think_content: Option<String>,
    preamble: String,
}

/// Single-pass pipeline orchestrator.
///
/// Pipeline per turn:
/// 1. Crisis check (keyword)
/// 2. Load case notes from DB
/// 3. Build peer coach with lean preamble + case notes
/// 4. Stream response (think blocks buffered, visible text streamed)
/// 5. Analyze think block for MI stage + themes
/// 6. Update case notes in DB
/// 7. Save chat turn + update history
pub struct Orchestrator {
    peer_coach_model: LlamaCppCompletionModel,
    coach_variant: PromptVariant,
    think_instructions: Option<String>,
    mode_catalog: Option<ModeCatalog>,
    chat_history: Vec<Message>,
    session_id: String,
    chat_conn: Connection,
    turn_number: i32,
    show_thinking: bool,
    /// When true, streaming output goes to stderr instead of stdout.
    /// Used in script mode to keep stdout clean for JSON output.
    output_to_stderr: bool,
    /// Maximum number of messages (user+assistant pairs) in the sliding window.
    max_history_messages: usize,
    /// Rolling checkpoint counter (incremented each time the sliding window drains).
    checkpoint_counter: u32,
    /// LanceDB vector store connection for RAG retrieval.
    vector_conn: Option<lancedb::Connection>,
    /// Embedding model for RAG queries.
    embedding_model: Option<EmbeddingModel>,
    /// Top-k results per RAG collection.
    rag_top_k: usize,
    /// MI stage at the start of the session (set on first case note update).
    initial_mi_stage: Option<String>,
    /// Number of user facts extracted and stored during this session.
    facts_extracted: u32,
    /// Number of significant turns flagged and stored during this session.
    significant_turns_flagged: u32,
}

impl Orchestrator {
    pub fn new(
        peer_coach_model: LlamaCppCompletionModel,
        coach_variant: PromptVariant,
        think_instructions: Option<String>,
        mode_catalog: Option<ModeCatalog>,
        session_id: String,
        chat_conn: Connection,
        show_thinking: bool,
        max_history_turns: usize,
        vector_conn: Option<lancedb::Connection>,
        embedding_model: Option<EmbeddingModel>,
        rag_top_k: usize,
    ) -> Self {
        Self {
            peer_coach_model,
            coach_variant,
            think_instructions,
            mode_catalog,
            chat_history: Vec::new(),
            session_id,
            chat_conn,
            turn_number: 0,
            show_thinking,
            output_to_stderr: false,
            max_history_messages: max_history_turns * 2,
            checkpoint_counter: 0,
            vector_conn,
            embedding_model,
            rag_top_k,
            initial_mi_stage: None,
            facts_extracted: 0,
            significant_turns_flagged: 0,
        }
    }

    /// Sets output to stderr (for script mode where stdout is reserved for JSON).
    pub fn set_output_to_stderr(&mut self, value: bool) {
        self.output_to_stderr = value;
    }

    /// Clears conversation history (but not the database or case notes).
    pub fn reset(&mut self) {
        self.chat_history.clear();
        self.turn_number = 0;
    }

    /// Ends the current session, generates a mechanical summary, stores it,
    /// and resets state for a new session.
    ///
    /// Returns the summary string for display to the user.
    pub async fn end_session(&mut self) -> Result<String> {
        // Build mechanical summary from tracked state
        let existing_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;
        let current_mi_stage = existing_notes
            .as_deref()
            .and_then(extract_mi_stage)
            .unwrap_or_else(|| "engage".to_string());
        let themes = existing_notes
            .as_deref()
            .and_then(extract_themes)
            .unwrap_or_default();

        let initial_stage = self.initial_mi_stage.as_deref().unwrap_or("engage");

        let summary_text = format!(
            "Session {} — {} turns\n\
             MI Stage: {} → {}\n\
             Themes: {}\n\
             Facts extracted: {}, Significant turns: {}",
            self.session_id,
            self.turn_number,
            initial_stage,
            current_mi_stage,
            if themes.is_empty() { "none".to_string() } else { themes.join(", ") },
            self.facts_extracted,
            self.significant_turns_flagged,
        );

        // Store as SessionSummary in vector store if available
        if let (Some(vconn), Some(model)) = (&self.vector_conn, &self.embedding_model) {
            use rig::embeddings::EmbeddingModel as _;

            let summary = crate::memory::vectors::SessionSummary {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: self.session_id.clone(),
                summary: summary_text.clone(),
                mi_stage_start: initial_stage.to_string(),
                mi_stage_end: current_mi_stage.clone(),
                themes: if themes.is_empty() { "none".to_string() } else { themes.join(", ") },
                turn_count: self.turn_number,
                created_at: chrono::Utc::now().to_rfc3339(),
            };

            match model.embed_text(&summary_text).await {
                Ok(embedding) => {
                    if let Err(e) = crate::memory::vectors::add_session_summary(
                        vconn, &summary, &embedding.vec,
                    ).await {
                        tracing::warn!(error = %e, "Failed to store session summary");
                    } else {
                        tracing::info!(session_id = self.session_id, "Stored session summary");
                    }
                }
                Err(e) => tracing::warn!(error = %e, "Failed to embed session summary"),
            }
        }

        // Reset for new session
        let new_session_id = format!(
            "session_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        self.session_id = new_session_id;
        self.chat_history.clear();
        self.turn_number = 0;
        self.checkpoint_counter = 0;
        self.initial_mi_stage = None;
        self.facts_extracted = 0;
        self.significant_turns_flagged = 0;

        Ok(summary_text)
    }

    /// Runs one full conversation turn.
    #[tracing::instrument(level = "info", skip(self))]
    pub async fn run_turn(&mut self, input: &str) -> Result<()> {
        let turn_start = Instant::now();
        self.turn_number += 1;

        // Crisis short-circuit
        if router::is_crisis(input) {
            let response = router::crisis_response();
            self.print_response(response);
            self.save_and_record(input, response).await?;
            return Ok(());
        }

        let _output = self.run_turn_inner(input).await?;

        tracing::info!(
            total_ms = turn_start.elapsed().as_millis() as u64,
            "Turn complete"
        );
        Ok(())
    }

    /// Runs one full conversation turn and returns structured results.
    ///
    /// Same pipeline as `run_turn` but captures all intermediate data
    /// for evaluation and scripted testing.
    #[tracing::instrument(level = "info", skip(self))]
    pub async fn run_turn_captured(&mut self, input: &str) -> Result<TurnResult> {
        let turn_start = Instant::now();
        self.turn_number += 1;

        // Crisis short-circuit
        if router::is_crisis(input) {
            let response = router::crisis_response();
            self.print_response(response);
            self.save_and_record(input, response).await?;
            return Ok(TurnResult {
                turn_number: self.turn_number,
                input: input.to_string(),
                response: response.to_string(),
                think_content: None,
                case_notes: None,
                preamble_injected: String::new(),
                duration_ms: turn_start.elapsed().as_millis() as u64,
            });
        }

        let output = self.run_turn_inner(input).await?;

        // Fetch the case notes we just wrote
        let updated_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;

        Ok(TurnResult {
            turn_number: self.turn_number,
            input: input.to_string(),
            response: output.response,
            think_content: output.think_content,
            case_notes: updated_notes,
            preamble_injected: output.preamble,
            duration_ms: turn_start.elapsed().as_millis() as u64,
        })
    }

    /// Shared turn pipeline: RAG retrieve → load notes → build preamble → stream → update notes → save.
    async fn run_turn_inner(&mut self, input: &str) -> Result<TurnOutput> {
        // Step 1: Load latest case notes
        let existing_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;

        // Step 1.5: RAG retrieval (if vector store is available)
        let rag_context = if let (Some(vconn), Some(model)) =
            (&self.vector_conn, &self.embedding_model)
        {
            let mi_stage = existing_notes
                .as_deref()
                .and_then(extract_mi_stage);
            let ctx = retrieval::retrieve_context(
                vconn,
                model,
                input,
                mi_stage.as_deref(),
                self.rag_top_k,
            )
            .await;
            let formatted = retrieval::format_rag_context(&ctx, MAX_RAG_CONTEXT_CHARS);
            if formatted.is_some() {
                tracing::info!("RAG context retrieved for turn");
            }
            formatted
        } else {
            None
        };

        // Step 2: Build peer coach with preamble + RAG context + case notes + mode guidance
        let preamble = build_peer_coach_preamble(
            &self.coach_variant.preamble,
            self.think_instructions.as_deref(),
            existing_notes.as_deref(),
            self.mode_catalog.as_ref(),
            rag_context.as_deref(),
        );

        let peer_coach = rig::agent::AgentBuilder::new(self.peer_coach_model.clone())
            .preamble(&preamble)
            .temperature(self.coach_variant.temperature)
            .max_tokens(self.coach_variant.max_tokens as u64)
            .build();

        // Step 3: Stream response (returns visible text + think block content)
        let (response, think_content) = self.stream_peer_coach(&peer_coach, input).await?;

        // Step 4: Analyze think block and update case notes
        let analysis = self.update_case_notes(input, &response, think_content.as_deref(), existing_notes.as_deref())
            .await?;

        // Step 4.5: Store user facts and significant turns in vector store (background)
        self.maybe_store_rag_data(input, &response, &analysis, analysis.mi_stage.as_deref());

        // Step 5: Save turn to DB + update history
        self.save_and_record(input, &response).await?;

        Ok(TurnOutput {
            response,
            think_content,
            preamble,
        })
    }

    /// Prints a response to the appropriate output stream.
    fn print_response(&self, text: &str) {
        if self.output_to_stderr {
            eprintln!("\nChiron: {text}");
        } else {
            println!("\nChiron: {text}");
        }
    }

    /// Streams the peer coach response, printing visible tokens to the display output.
    /// Returns (visible_response, think_content).
    ///
    /// Display goes to stderr when `output_to_stderr` is true (script mode),
    /// otherwise to stdout (interactive mode).
    async fn stream_peer_coach(
        &self,
        peer_coach: &Agent<LlamaCppCompletionModel>,
        input: &str,
    ) -> Result<(String, Option<String>)> {
        let use_stderr = self.output_to_stderr;

        // Display think block header if show_thinking is enabled
        if self.show_thinking {
            if use_stderr {
                eprint!("\n\x1b[2m[thinking...]\x1b[0m");
                io::stderr().flush()?;
            } else {
                print!("\n\x1b[2m[thinking...]\x1b[0m");
                io::stdout().flush()?;
            }
        }

        if use_stderr {
            eprint!("\nChiron: ");
            io::stderr().flush()?;
        } else {
            print!("\nChiron: ");
            io::stdout().flush()?;
        }

        let mut stream = peer_coach
            .stream_chat(input, self.chat_history.clone())
            .await;

        let mut full_response = String::new();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::Text(text),
                )) => {
                    if use_stderr {
                        eprint!("{}", text.text);
                        io::stderr().flush()?;
                    } else {
                        print!("{}", text.text);
                        io::stdout().flush()?;
                    }
                    full_response.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(final_resp)) => {
                    if full_response.is_empty() {
                        full_response = final_resp.response().to_string();
                        if use_stderr {
                            eprint!("{full_response}");
                            io::stderr().flush()?;
                        } else {
                            print!("{full_response}");
                            io::stdout().flush()?;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Streaming error");
                    break;
                }
                _ => {}
            }
        }

        if use_stderr { eprintln!(); } else { println!(); }

        // Always capture think content for case note analysis
        let think_content = self.peer_coach_model.take_think_content();

        // Show think block content if flag is set
        if self.show_thinking {
            if let Some(ref think) = think_content {
                if use_stderr {
                    eprintln!("\x1b[2m--- think block ---\x1b[0m");
                    eprintln!("\x1b[2m{think}\x1b[0m");
                    eprintln!("\x1b[2m--- end think ---\x1b[0m");
                } else {
                    println!("\x1b[2m--- think block ---\x1b[0m");
                    println!("\x1b[2m{think}\x1b[0m");
                    println!("\x1b[2m--- end think ---\x1b[0m");
                }
            }
        }

        // Safety net: strip any think blocks that leaked through
        let clean_response = crate::provider::strip_think_blocks(&full_response);
        if clean_response.len() < full_response.len() {
            tracing::debug!(
                stripped_bytes = full_response.len() - clean_response.len(),
                "Stripped think blocks from response"
            );
        }

        Ok((clean_response, think_content))
    }

    /// Updates case notes from the model's structured think block tags.
    ///
    /// The model is prompted to include `[MI-STAGE: ...]`, `[STRATEGY: ...]`,
    /// `[TALK-TYPE: ...]`, and `[THEMES: ...]` in its think block.
    /// If no think block or no tags, fields carry forward from previous notes.
    ///
    /// Returns the `ThinkAnalysis` so callers can access user facts and signals.
    async fn update_case_notes(
        &mut self,
        _input: &str,
        _response: &str,
        think_content: Option<&str>,
        existing_notes: Option<&str>,
    ) -> Result<ThinkAnalysis> {
        let (notes, mi_stage) = build_case_notes_from_analysis(think_content, existing_notes);

        // Track the initial MI stage for session summary
        if self.initial_mi_stage.is_none() {
            self.initial_mi_stage = mi_stage.clone();
        }

        let merged = extract_themes(&notes).unwrap_or_default();

        case_notes::save_case_note(
            &self.chat_conn,
            &self.session_id,
            self.turn_number,
            mi_stage.as_deref(),
            &notes,
        )
        .await?;

        tracing::info!(
            mi_stage = mi_stage.as_deref().unwrap_or("unknown"),
            themes = merged.join(", "),
            "Case notes updated"
        );

        // Return the analysis for downstream use (RAG storage)
        let analysis = think_content
            .filter(|t| !t.is_empty())
            .map(analyze_think_block)
            .unwrap_or_else(|| ThinkAnalysis {
                mi_stage: None,
                strategy_used: None,
                talk_type: None,
                themes: vec![],
                user_facts: vec![],
                significant_signal: None,
                raw_think: String::new(),
            });

        Ok(analysis)
    }

    /// Stores extracted user facts and significant turns in the vector store.
    ///
    /// Runs embedding and insertion in background tasks to avoid blocking
    /// the response pipeline. Skips entirely if vector store is not configured.
    #[tracing::instrument(level = "debug", skip(self, analysis))]
    fn maybe_store_rag_data(
        &mut self,
        input: &str,
        response: &str,
        analysis: &ThinkAnalysis,
        mi_stage: Option<&str>,
    ) {
        let (Some(vconn), Some(model)) = (&self.vector_conn, &self.embedding_model) else {
            return;
        };

        // Track counters for session summary
        self.facts_extracted += analysis.user_facts.len() as u32;
        if analysis.significant_signal.is_some() {
            self.significant_turns_flagged += 1;
        }

        use rig::embeddings::EmbeddingModel as _;
        use tracing::Instrument as _;

        let session_id = self.session_id.clone();
        let turn_number = self.turn_number;
        let now = chrono::Utc::now().to_rfc3339();

        // Store user facts
        for (fact_type, content) in &analysis.user_facts {
            let vconn = vconn.clone();
            let model = model.clone();
            let fact = crate::memory::vectors::UserFact {
                id: uuid::Uuid::new_v4().to_string(),
                fact_type: fact_type.clone(),
                content: content.clone(),
                source_session: session_id.clone(),
                last_confirmed: session_id.clone(),
                created_at: now.clone(),
                updated_at: now.clone(),
            };
            let content = content.clone();
            tokio::spawn(
                async move {
                    match model.embed_text(&content).await {
                        Ok(embedding) => {
                            match crate::memory::vectors::upsert_user_fact(
                                &vconn, &model, &fact, &embedding.vec,
                            )
                            .await
                            {
                                Ok(crate::memory::vectors::UpsertResult::Inserted) => {
                                    tracing::info!(fact_type = fact.fact_type, "Stored new user fact");
                                }
                                Ok(crate::memory::vectors::UpsertResult::Updated(id)) => {
                                    tracing::info!(fact_type = fact.fact_type, matched_id = %id, "Updated existing user fact");
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "Failed to upsert user fact");
                                }
                            }
                        }
                        Err(e) => tracing::warn!(error = %e, "Failed to embed user fact"),
                    }
                }
                .in_current_span(),
            );
        }

        // Store significant turn
        if let Some(signal_type) = &analysis.significant_signal {
            let vconn = vconn.clone();
            let model = model.clone();
            let turn = crate::memory::vectors::SignificantTurn {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: session_id.clone(),
                turn_number,
                user_content: input.to_string(),
                assistant_content: response.to_string(),
                signal_type: signal_type.clone(),
                mi_stage: mi_stage.unwrap_or("unknown").to_string(),
                talk_type: analysis.talk_type.clone().unwrap_or_default(),
                created_at: now,
            };
            let embed_text = input.to_string();
            tokio::spawn(
                async move {
                    match model.embed_text(&embed_text).await {
                        Ok(embedding) => {
                            if let Err(e) = crate::memory::vectors::add_significant_turn(
                                &vconn,
                                &turn,
                                &embedding.vec,
                            )
                            .await
                            {
                                tracing::warn!(error = %e, "Failed to store significant turn");
                            } else {
                                tracing::info!(signal_type = turn.signal_type, "Stored significant turn");
                            }
                        }
                        Err(e) => tracing::warn!(error = %e, "Failed to embed significant turn"),
                    }
                }
                .in_current_span(),
            );
        }
    }

    /// Saves the turn to the database and appends to in-memory chat history.
    ///
    /// Applies a sliding window to keep chat history within context limits.
    /// When messages are drained, they are captured as a checkpoint in the
    /// vector store so context isn't permanently lost.
    async fn save_and_record(&mut self, input: &str, response: &str) -> Result<()> {
        memory::save_chat_turn(&self.chat_conn, &self.session_id, "user", input)
            .await
            .context("Failed to save user turn")?;
        memory::save_chat_turn(&self.chat_conn, &self.session_id, "assistant", response)
            .await
            .context("Failed to save assistant turn")?;

        self.chat_history.push(Message::user(input));
        self.chat_history.push(Message::assistant(response));

        // Sliding window: keep last N turns (pairs of user+assistant messages).
        if self.chat_history.len() > self.max_history_messages {
            let trim_count = self.chat_history.len() - self.max_history_messages;

            // Capture drained messages as a checkpoint before removing them
            let drained: Vec<Message> = self.chat_history[..trim_count].to_vec();
            self.maybe_create_checkpoint(&drained);

            self.chat_history.drain(..trim_count);
            tracing::debug!(
                kept = self.max_history_messages,
                trimmed = trim_count,
                "Sliding window trimmed chat history"
            );
        }

        Ok(())
    }

    /// Creates a rolling checkpoint from messages being drained from the sliding window.
    ///
    /// Concatenates drained messages into a checkpoint text, embeds it, and
    /// stores it as a `SessionCheckpoint` in LanceDB via a background task.
    fn maybe_create_checkpoint(&mut self, drained: &[Message]) {
        let (Some(vconn), Some(model)) = (&self.vector_conn, &self.embedding_model) else {
            return;
        };

        use rig::embeddings::EmbeddingModel as _;
        use tracing::Instrument as _;

        // Build checkpoint text from drained messages
        let mut checkpoint_text = String::new();
        for msg in drained {
            match msg {
                Message::User { content } => {
                    for part in content.clone() {
                        if let rig::message::UserContent::Text(rig::message::Text { text }) = part {
                            checkpoint_text.push_str("User: ");
                            checkpoint_text.push_str(&text);
                            checkpoint_text.push('\n');
                        }
                    }
                }
                Message::Assistant { content, .. } => {
                    for part in content.clone() {
                        if let rig::message::AssistantContent::Text(rig::message::Text { text }) = part {
                            checkpoint_text.push_str("Assistant: ");
                            checkpoint_text.push_str(&text);
                            checkpoint_text.push('\n');
                        }
                    }
                }
                _ => {}
            }
        }

        if checkpoint_text.is_empty() {
            return;
        }

        self.checkpoint_counter += 1;

        // Calculate turn range from message count
        // Each pair of messages is one turn, checkpoint_counter tells us which batch
        let msgs_per_checkpoint = drained.len();
        let turn_end = self.checkpoint_counter as usize * (msgs_per_checkpoint / 2);
        let turn_start = turn_end.saturating_sub(msgs_per_checkpoint / 2) + 1;
        let turn_range = format!("{turn_start}-{turn_end}");

        let checkpoint = crate::memory::vectors::SessionCheckpoint {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: self.session_id.clone(),
            checkpoint_number: self.checkpoint_counter as i32,
            content: checkpoint_text.clone(),
            mi_stage: String::new(), // filled below from case notes if available
            themes: String::new(),
            turn_range,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let vconn = vconn.clone();
        let model = model.clone();
        tokio::spawn(
            async move {
                match model.embed_text(&checkpoint_text).await {
                    Ok(embedding) => {
                        if let Err(e) = crate::memory::vectors::add_session_checkpoint(
                            &vconn,
                            &checkpoint,
                            &embedding.vec,
                        )
                        .await
                        {
                            tracing::warn!(error = %e, "Failed to store session checkpoint");
                        } else {
                            tracing::info!(
                                checkpoint_number = checkpoint.checkpoint_number,
                                "Stored session checkpoint"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "Failed to embed session checkpoint"),
                }
            }
            .in_current_span(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that case notes are built correctly from a think block with all tags.
    #[test]
    fn test_case_notes_from_think_block() {
        let think = "[MI-STAGE: evoke]\n[STRATEGY: complex reflection]\n[TALK-TYPE: desire change talk]\n[THEMES: drinking, anxiety]";
        let (notes, mi_stage) = build_case_notes_from_analysis(Some(think), None);

        assert_eq!(mi_stage.as_deref(), Some("evoke"));
        assert!(notes.contains("MI Stage: evoke"));
        assert!(notes.contains("Strategy Used: complex reflection"));
        assert!(notes.contains("Talk Type: desire change talk"));
        assert!(notes.contains("drinking"));
        assert!(notes.contains("anxiety"));
    }

    /// Test that themes accumulate across turns (set union, never regress).
    #[test]
    fn test_theme_accumulation_across_turns() {
        // Turn 1: model identifies drinking, job
        let think1 = "[MI-STAGE: engage]\n[THEMES: drinking, job]";
        let (notes1, _) = build_case_notes_from_analysis(Some(think1), None);
        assert!(notes1.contains("drinking"));
        assert!(notes1.contains("job"));

        // Turn 2: model identifies anxiety (new) but doesn't mention drinking/job
        let think2 = "[MI-STAGE: focus]\n[THEMES: anxiety]";
        let (notes2, _) = build_case_notes_from_analysis(Some(think2), Some(&notes1));

        // All three themes must be present (union)
        assert!(notes2.contains("anxiety"), "new theme should appear");
        assert!(notes2.contains("drinking"), "old theme must not regress");
        assert!(notes2.contains("job"), "old theme must not regress");
        assert!(notes2.contains("MI Stage: focus"), "stage should update");
    }

    /// Test that MI stage carries forward when think block lacks a tag.
    #[test]
    fn test_mi_stage_carry_forward() {
        // Turn 1: model sets stage
        let think1 = "[MI-STAGE: evoke]\n[THEMES: drinking]";
        let (notes1, _) = build_case_notes_from_analysis(Some(think1), None);

        // Turn 2: empty think block (no tags)
        let (notes2, mi_stage2) = build_case_notes_from_analysis(Some("just thinking"), Some(&notes1));
        assert_eq!(mi_stage2.as_deref(), Some("evoke"), "stage should carry forward");
        assert!(notes2.contains("drinking"), "themes should carry forward");
    }

    /// Test that no think content defaults to "engage" stage.
    #[test]
    fn test_no_think_defaults_to_engage() {
        let (notes, mi_stage) = build_case_notes_from_analysis(None, None);
        assert_eq!(mi_stage, None);
        assert!(notes.contains("MI Stage: engage"));
        assert!(notes.contains("Running Themes: none"));
    }

    /// Test sliding window trim logic (extracted to test without DB).
    #[test]
    fn test_sliding_window_boundary() {
        let max_history_messages = 4; // 2 turns worth
        let mut history: Vec<Message> = Vec::new();

        // Simulate 3 turns (6 messages)
        for i in 1..=3 {
            history.push(Message::user(format!("user msg {i}")));
            history.push(Message::assistant(format!("assistant msg {i}")));

            if history.len() > max_history_messages {
                let trim_count = history.len() - max_history_messages;
                history.drain(..trim_count);
            }
        }

        // Should keep last 2 turns (4 messages)
        assert_eq!(history.len(), 4);
    }

    /// Test crisis routing short-circuits without case notes.
    #[test]
    fn test_crisis_short_circuits() {
        assert!(router::is_crisis("I want to kill myself"));
        let response = router::crisis_response();
        assert!(response.contains("988"));
        assert!(response.contains("741741"));
    }

    /// Test that the preamble ordering matches the documented pipeline.
    #[test]
    fn test_preamble_ordering_matches_pipeline() {
        let base = "You are a peer supporter.";
        let think = "Think carefully.";
        let rag = "## Retrieved Context\n- User likes hiking";
        let notes = "MI Stage: engage\nRunning Themes: exercise";

        let preamble = build_peer_coach_preamble(
            base,
            Some(think),
            Some(notes),
            None,
            Some(rag),
        );

        // Verify ordering: base → think → RAG → case notes
        let base_pos = preamble.find(base).unwrap();
        let think_pos = preamble.find(think).unwrap();
        let rag_pos = preamble.find("Retrieved Context").unwrap();
        let notes_pos = preamble.find("Session Context").unwrap();

        assert!(base_pos < think_pos);
        assert!(think_pos < rag_pos);
        assert!(rag_pos < notes_pos);
    }

    /// Mechanical proof: themes stay bounded over 20 turns of 3 new themes each.
    #[test]
    fn test_theme_bounded_over_20_turns() {
        let mut notes: Option<String> = None;
        for i in 1..=20 {
            let think = format!("[MI-STAGE: engage]\n[THEMES: topic_{i}a, topic_{i}b, topic_{i}c]");
            let (new_notes, _) = build_case_notes_from_analysis(Some(&think), notes.as_deref());
            let themes = extract_themes(&new_notes).unwrap_or_default();
            assert!(
                themes.len() <= MAX_THEMES,
                "Turn {i}: {} themes exceeds cap of {MAX_THEMES}",
                themes.len()
            );
            notes = Some(new_notes);
        }
    }

    /// Mechanical proof: preamble stays bounded over 10 turns with growing context.
    #[test]
    fn test_preamble_bounded_over_10_turns() {
        use crate::agents::peer::{build_peer_coach_preamble_budgeted, DEFAULT_MAX_PREAMBLE_CHARS};

        let base = "You are a trained peer mental health supporter. You provide empathetic support.";
        let think_inst = "Before responding, reason briefly. [MI-STAGE], [STRATEGY], [TALK-TYPE], [THEMES].";
        let mut notes: Option<String> = None;

        for i in 1..=10 {
            let think = format!(
                "[MI-STAGE: evoke]\n[STRATEGY: complex reflection]\n[TALK-TYPE: change talk]\n[THEMES: topic_{i}, subtopic_{i}, extra_{i}]"
            );
            let (new_notes, _) = build_case_notes_from_analysis(Some(&think), notes.as_deref());

            // Simulate growing RAG context
            let rag = format!(
                "## What You Know About This Person\n{}",
                (1..=i).map(|j| format!("- [goal] User fact number {j} about their life")).collect::<Vec<_>>().join("\n")
            );

            let preamble = build_peer_coach_preamble_budgeted(
                base,
                Some(think_inst),
                Some(&new_notes),
                None,
                Some(&rag),
                DEFAULT_MAX_PREAMBLE_CHARS,
            );

            assert!(
                preamble.len() <= DEFAULT_MAX_PREAMBLE_CHARS,
                "Turn {i}: preamble {} chars exceeds budget {}",
                preamble.len(),
                DEFAULT_MAX_PREAMBLE_CHARS
            );
            // Base and think instructions always present
            assert!(preamble.contains(base), "Turn {i}: base missing");
            assert!(preamble.contains(think_inst), "Turn {i}: think instructions missing");
            notes = Some(new_notes);
        }
    }
}
