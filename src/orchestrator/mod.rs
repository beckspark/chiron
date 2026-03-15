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
use crate::provider::LlamaCppCompletionModel;
use crate::router;
use crate::supervision::{
    analyze_think_block, extract_mi_stage, extract_themes, merge_themes, ThinkAnalysis,
};


/// Structured result from a single conversation turn.
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

    /// Runs one full conversation turn.
    #[tracing::instrument(level = "info", skip(self))]
    pub async fn run_turn(&mut self, input: &str) -> Result<()> {
        let turn_start = Instant::now();
        self.turn_number += 1;

        // Step 1: Crisis check
        if router::is_crisis(input) {
            let response = router::crisis_response();
            if self.output_to_stderr {
                eprintln!("\nChiron: {response}");
            } else {
                println!("\nChiron: {response}");
            }
            self.save_and_record(input, response).await?;
            return Ok(());
        }

        // Step 2: Load latest case notes
        let existing_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;

        // Step 3: Build peer coach with preamble + case notes + mode guidance
        let preamble = build_peer_coach_preamble(
            &self.coach_variant.preamble,
            self.think_instructions.as_deref(),
            existing_notes.as_deref(),
            self.mode_catalog.as_ref(),
        );

        let peer_coach = rig::agent::AgentBuilder::new(self.peer_coach_model.clone())
            .preamble(&preamble)
            .temperature(self.coach_variant.temperature)
            .max_tokens(self.coach_variant.max_tokens as u64)
            .build();

        // Step 4: Stream response (returns visible text + think block content)
        let (response, think_content) = self.stream_peer_coach(&peer_coach, input).await?;

        // Step 5: Analyze think block for MI stage + themes + strategy
        // Uses think block content when available (rich model reasoning),
        // falls back to exchange text for keyword heuristics.
        self.update_case_notes(input, &response, think_content.as_deref(), existing_notes.as_deref())
            .await?;

        // Step 6: Save turn to DB + update history
        self.save_and_record(input, &response).await?;

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

        // Crisis check
        if router::is_crisis(input) {
            let response = router::crisis_response();
            if self.output_to_stderr {
                eprintln!("\nChiron: {response}");
            } else {
                println!("\nChiron: {response}");
            }
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

        let existing_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;

        let preamble = build_peer_coach_preamble(
            &self.coach_variant.preamble,
            self.think_instructions.as_deref(),
            existing_notes.as_deref(),
            self.mode_catalog.as_ref(),
        );

        let peer_coach = rig::agent::AgentBuilder::new(self.peer_coach_model.clone())
            .preamble(&preamble)
            .temperature(self.coach_variant.temperature)
            .max_tokens(self.coach_variant.max_tokens as u64)
            .build();

        let (response, think_content) = self.stream_peer_coach(&peer_coach, input).await?;

        self.update_case_notes(input, &response, think_content.as_deref(), existing_notes.as_deref())
            .await?;

        self.save_and_record(input, &response).await?;

        // Fetch the case notes we just wrote
        let updated_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;

        Ok(TurnResult {
            turn_number: self.turn_number,
            input: input.to_string(),
            response: response.clone(),
            think_content,
            case_notes: updated_notes,
            preamble_injected: preamble,
            duration_ms: turn_start.elapsed().as_millis() as u64,
        })
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
    async fn update_case_notes(
        &self,
        _input: &str,
        _response: &str,
        think_content: Option<&str>,
        existing_notes: Option<&str>,
    ) -> Result<()> {
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
                raw_think: String::new(),
            });

        // Use model's classification, fall back to previous notes if absent
        let mi_stage = analysis
            .mi_stage
            .or_else(|| existing_notes.and_then(extract_mi_stage));

        let merged = merge_themes(&prev_themes, &analysis.themes);

        // Build case notes from model's analysis
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
    ///
    /// Applies a sliding window to keep chat history within context limits.
    /// Case notes preserve accumulated themes/stage, so older context isn't lost.
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
        // With 2048 context, ~580 tokens for system prompt + case notes,
        // ~512 for generation, leaves ~950 for history. 4 turns is safe.
        const MAX_HISTORY_MESSAGES: usize = 8; // 4 turns = 8 messages
        if self.chat_history.len() > MAX_HISTORY_MESSAGES {
            let trim_count = self.chat_history.len() - MAX_HISTORY_MESSAGES;
            self.chat_history.drain(..trim_count);
            tracing::debug!(
                kept = MAX_HISTORY_MESSAGES,
                trimmed = trim_count,
                "Sliding window trimmed chat history"
            );
        }

        Ok(())
    }
}
