use std::io::{self, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use futures::StreamExt;
use rig::agent::{Agent, MultiTurnStreamItem};
use rig::completion::Message;
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use tokio_rusqlite::Connection;

use crate::agents::peer::build_peer_coach_preamble;
use crate::catalog::PromptVariant;
use crate::memory;
use crate::memory::case_notes;
use crate::provider::LlamaCppCompletionModel;
use crate::router;
use crate::supervision::{
    analyze_think_block, extract_mi_stage, extract_themes, merge_themes,
};

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
    chat_history: Vec<Message>,
    session_id: String,
    chat_conn: Connection,
    turn_number: i32,
    show_thinking: bool,
}

impl Orchestrator {
    pub fn new(
        peer_coach_model: LlamaCppCompletionModel,
        coach_variant: PromptVariant,
        session_id: String,
        chat_conn: Connection,
        show_thinking: bool,
    ) -> Self {
        Self {
            peer_coach_model,
            coach_variant,
            chat_history: Vec::new(),
            session_id,
            chat_conn,
            turn_number: 0,
            show_thinking,
        }
    }

    /// Clears conversation history (but not the database or case notes).
    pub fn reset(&mut self) {
        self.chat_history.clear();
        self.turn_number = 0;
    }

    /// Runs one full conversation turn.
    #[tracing::instrument(level = "info", skip(self))]
    pub async fn run_turn(&mut self, input: &str) -> Result<()> {
        let turn_start = Instant::now();
        self.turn_number += 1;

        // Step 1: Crisis check
        if router::is_crisis(input) {
            let response = router::crisis_response();
            println!("\nChiron: {response}");
            self.save_and_record(input, response).await?;
            return Ok(());
        }

        // Step 2: Load latest case notes
        let existing_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;

        // Step 3: Build peer coach with preamble + case notes
        let preamble = build_peer_coach_preamble(
            &self.coach_variant.preamble,
            existing_notes.as_deref(),
        );

        let peer_coach = rig::agent::AgentBuilder::new(self.peer_coach_model.clone())
            .preamble(&preamble)
            .temperature(self.coach_variant.temperature)
            .max_tokens(self.coach_variant.max_tokens as u64)
            .build();

        // Step 4: Stream response
        let response = self.stream_peer_coach(&peer_coach, input).await?;

        // Step 5: Analyze think block (if the response came from LlamaCppResponse)
        // The think content is extracted during streaming and logged.
        // For case notes, we do keyword heuristics on the response itself
        // since the think block was consumed during streaming.
        self.update_case_notes(input, &response, existing_notes.as_deref())
            .await?;

        // Step 6: Save turn to DB + update history
        self.save_and_record(input, &response).await?;

        tracing::info!(
            total_ms = turn_start.elapsed().as_millis() as u64,
            "Turn complete"
        );

        Ok(())
    }

    /// Streams the peer coach response, printing visible tokens to stdout.
    async fn stream_peer_coach(
        &self,
        peer_coach: &Agent<LlamaCppCompletionModel>,
        input: &str,
    ) -> Result<String> {
        // Display think block header if show_thinking is enabled
        if self.show_thinking {
            print!("\n\x1b[2m[thinking...]\x1b[0m");
            io::stdout().flush()?;
        }

        print!("\nChiron: ");
        io::stdout().flush()?;

        let mut stream = peer_coach
            .stream_chat(input, self.chat_history.clone())
            .await;

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
                    if full_response.is_empty() {
                        full_response = final_resp.response().to_string();
                        print!("{full_response}");
                        io::stdout().flush()?;
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Streaming error");
                    break;
                }
                _ => {}
            }
        }

        println!();

        // Show think block content if flag is set
        if self.show_thinking {
            if let Some(think_content) = self.peer_coach_model.take_think_content() {
                println!("\x1b[2m--- think block ---\x1b[0m");
                println!("\x1b[2m{think_content}\x1b[0m");
                println!("\x1b[2m--- end think ---\x1b[0m");
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

        Ok(clean_response)
    }

    /// Updates case notes based on the exchange.
    ///
    /// Uses the think parser for MI stage and theme extraction instead
    /// of a separate supervisor model pass.
    async fn update_case_notes(
        &self,
        input: &str,
        response: &str,
        existing_notes: Option<&str>,
    ) -> Result<()> {
        let prev_themes = existing_notes
            .and_then(extract_themes)
            .unwrap_or_default();

        // Build simple case notes from the exchange
        // The think parser analyzes any think content for MI signals
        let exchange = format!("Person: {input}\nCoach: {response}");
        let analysis = analyze_think_block(&exchange);

        let mi_stage = analysis
            .mi_stage
            .or_else(|| existing_notes.and_then(extract_mi_stage));

        // Merge themes from analysis with previous themes
        let new_themes = analysis.themes;
        let merged = merge_themes(&prev_themes, &new_themes);

        // Build updated notes
        let notes = format!(
            "MI Stage: {}\nRunning Themes: {}",
            mi_stage.as_deref().unwrap_or("engage"),
            if merged.is_empty() {
                "none".to_string()
            } else {
                merged.join(", ")
            },
        );

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

        Ok(())
    }

    /// Saves the turn to the database and appends to in-memory chat history.
    async fn save_and_record(&mut self, input: &str, response: &str) -> Result<()> {
        memory::save_chat_turn(&self.chat_conn, &self.session_id, "user", input)
            .await
            .context("Failed to save user turn")?;
        memory::save_chat_turn(&self.chat_conn, &self.session_id, "assistant", response)
            .await
            .context("Failed to save assistant turn")?;

        self.chat_history.push(Message::user(input));
        self.chat_history.push(Message::assistant(response));

        Ok(())
    }
}
