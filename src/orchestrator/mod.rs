use std::io::{self, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use futures::StreamExt;
use rig::agent::{Agent, MultiTurnStreamItem};
use rig::completion::{Chat, Message};
use rig::embeddings::EmbeddingModel;
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use rig_sqlite::SqliteVectorStore;
use tokio_rusqlite::Connection;

use crate::agents::peer::{build_peer_coach_preamble, build_peer_coach_with_rag};
use crate::memory;
use crate::memory::case_notes;
use crate::memory::store::MiKnowledge;
use crate::provider::completion::CandleCompletionModel;

/// Result of a single conversation turn through the orchestrator pipeline.
#[derive(Debug)]
pub struct TurnResult {
    pub response: String,
}

/// Two-agent reflection pipeline using SmolLM3 3B with persistent case notes.
///
/// Pipeline per turn:
/// 1. Load latest case notes from DB
/// 2. Build peer coach with case notes in system prompt (streaming, with RAG)
/// 3. After response: run supervisor to update case notes (non-streaming)
///
/// The reflection loop: coach reads notes → responds → supervisor critiques →
/// writes updated notes → coach reads notes → ...
pub struct Orchestrator<E>
where
    E: EmbeddingModel + Clone + 'static,
{
    /// Completion model handle (cheap Arc clone) for building peer coach each turn.
    peer_coach_model: CandleCompletionModel,
    /// Pre-built supervisor agent (preamble is static, no RAG needed).
    supervisor: Agent<CandleCompletionModel>,
    /// Cloneable vector store — cloned each turn to create a fresh index for the peer coach.
    knowledge_store: SqliteVectorStore<E, MiKnowledge>,
    /// Embedding model for creating search indexes from the knowledge store.
    embedding_model: E,
    /// Number of RAG results to inject per query.
    rag_top_k: usize,
    /// Maximum tokens for the peer coach response.
    coach_max_tokens: usize,
    chat_history: Vec<Message>,
    session_id: String,
    chat_conn: Connection,
    turn_number: i32,
}

impl<E> Orchestrator<E>
where
    E: EmbeddingModel + Clone + 'static,
{
    pub fn new(
        peer_coach_model: CandleCompletionModel,
        supervisor: Agent<CandleCompletionModel>,
        knowledge_store: SqliteVectorStore<E, MiKnowledge>,
        embedding_model: E,
        rag_top_k: usize,
        coach_max_tokens: usize,
        session_id: String,
        chat_conn: Connection,
    ) -> Self {
        Self {
            peer_coach_model,
            supervisor,
            knowledge_store,
            embedding_model,
            rag_top_k,
            coach_max_tokens,
            chat_history: Vec::new(),
            session_id,
            chat_conn,
            turn_number: 0,
        }
    }

    /// Clears conversation history (but not the database or case notes).
    pub fn reset(&mut self) {
        self.chat_history.clear();
        self.turn_number = 0;
    }

    /// Runs one full conversation turn through the two-agent pipeline.
    ///
    /// 1. Load latest case notes from DB
    /// 2. Build peer coach with case notes in system prompt + RAG
    /// 3. Stream peer coach response
    /// 4. Run supervisor (non-streaming, after response printed)
    /// 5. Save turn to DB + update history
    #[tracing::instrument(level = "info", skip(self))]
    pub async fn run_turn(&mut self, input: &str) -> Result<TurnResult> {
        let turn_start = Instant::now();
        self.turn_number += 1;

        // Step 1: Load latest case notes
        let existing_notes = case_notes::get_latest_case_note(&self.chat_conn).await?;
        tracing::info!(
            found = existing_notes.is_some(),
            "Loading latest case notes from DB"
        );

        // Step 2: Build peer coach with case notes in preamble + RAG
        let preamble = build_peer_coach_preamble(existing_notes.as_deref());
        let knowledge_index = self
            .knowledge_store
            .clone()
            .index(self.embedding_model.clone());
        let peer_coach = build_peer_coach_with_rag(
            self.peer_coach_model.clone(),
            &preamble,
            knowledge_index,
            self.rag_top_k,
            self.coach_max_tokens,
        );

        // Step 3: Stream peer coach response
        let response = self
            .stream_peer_coach(&peer_coach, input)
            .await?;

        // Step 4: Supervisor case notes update (runs after response is printed)
        self.run_supervisor(input, &response, existing_notes.as_deref())
            .await?;

        // Step 5: Save turn to DB + update history
        self.save_and_record(input, &response).await?;

        tracing::info!(
            total_ms = turn_start.elapsed().as_millis() as u64,
            "Turn complete"
        );

        Ok(TurnResult { response })
    }

    /// Streams the peer coach response, printing tokens to stdout as they arrive.
    ///
    /// Think blocks are stripped from the output as a safety net (the template
    /// suppresses thinking structurally, but this catches any leaks).
    async fn stream_peer_coach(
        &self,
        peer_coach: &Agent<CandleCompletionModel>,
        input: &str,
    ) -> Result<String> {
        let effective_input = input.to_string();

        print!("\nChiron: ");
        io::stdout().flush()?;

        let mut stream = peer_coach
            .stream_chat(&effective_input, self.chat_history.clone())
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

        // Strip any think blocks that leaked through despite /no_think
        let clean_response = crate::provider::strip_think_blocks(&full_response);
        if clean_response.len() < full_response.len() {
            tracing::debug!(
                stripped_bytes = full_response.len() - clean_response.len(),
                "Stripped think blocks from coach response"
            );
        }

        Ok(clean_response)
    }

    /// Runs the supervisor to analyze the exchange and update case notes.
    ///
    /// Runs after the response is printed — no perceived latency for the user.
    async fn run_supervisor(
        &self,
        input: &str,
        response: &str,
        existing_notes: Option<&str>,
    ) -> Result<()> {
        let t_start = Instant::now();

        let notes_context = existing_notes.unwrap_or("No prior case notes.");

        let prompt = format!(
            "PREVIOUS CASE NOTES:\n{}\n\nLATEST EXCHANGE:\nPerson: {}\nCoach: {}",
            notes_context, input, response
        );

        let raw_notes = self
            .supervisor
            .chat(&prompt, vec![])
            .await
            .map_err(|e| anyhow::anyhow!("Supervisor inference failed: {e}"))?;

        // Strip think blocks — supervisor may reason internally but we only want the structured notes
        let updated_notes = crate::provider::strip_think_blocks(&raw_notes);

        // Strip echoed exchange blocks — the 3B model often echoes the LATEST EXCHANGE
        // prompt back into its output, causing a snowball where notes grow each turn
        let updated_notes = strip_echoed_exchanges(&updated_notes);

        let mi_stage = extract_mi_stage(&updated_notes);

        case_notes::save_case_note(
            &self.chat_conn,
            &self.session_id,
            self.turn_number,
            mi_stage.as_deref(),
            &updated_notes,
        )
        .await?;

        tracing::info!(
            mi_stage = mi_stage.as_deref().unwrap_or("unknown"),
            supervisor_ms = t_start.elapsed().as_millis() as u64,
            "Supervisor case notes updated"
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

/// Strips echoed `LATEST EXCHANGE:` blocks from supervisor output.
///
/// The 3B model often copies the exchange prompt into its output. If left in,
/// these blocks accumulate across turns (snowball effect), consuming the
/// supervisor's token budget with repeated old exchanges instead of analysis.
fn strip_echoed_exchanges(notes: &str) -> String {
    match notes.find("LATEST EXCHANGE:") {
        Some(pos) => notes[..pos].trim_end().to_string(),
        None => notes.to_string(),
    }
}

/// Extracts the MI stage from case notes, handling markdown bold formatting.
///
/// Matches lines like `MI Stage: engage`, `**MI Stage:** evoke`, or
/// `**MI Stage: focus**`. Returns the stage as a lowercase trimmed string.
fn extract_mi_stage(notes: &str) -> Option<String> {
    notes
        .lines()
        .map(|l| l.replace("**", ""))
        .find(|l| l.trim().to_lowercase().starts_with("mi stage:"))
        .and_then(|l| {
            let (_, value) = l.split_once(':')?;
            Some(value.trim().to_lowercase())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_mi_stage_plain() {
        let notes = "MI Stage: engage\nChange Talk: none observed";
        assert_eq!(extract_mi_stage(notes), Some("engage".to_string()));
    }

    #[test]
    fn test_extract_mi_stage_bold() {
        let notes = "**MI Stage:** evoke\nChange Talk: \"I want to stop\"";
        assert_eq!(extract_mi_stage(notes), Some("evoke".to_string()));
    }

    #[test]
    fn test_extract_mi_stage_fully_bold() {
        let notes = "**MI Stage: focus**\nKey Themes: anxiety";
        assert_eq!(extract_mi_stage(notes), Some("focus".to_string()));
    }

    #[test]
    fn test_extract_mi_stage_case_insensitive() {
        let notes = "mi stage: Plan\nSession Summary: moving forward";
        assert_eq!(extract_mi_stage(notes), Some("plan".to_string()));
    }

    #[test]
    fn test_extract_mi_stage_missing() {
        let notes = "Change Talk: desire\nKey Themes: work stress";
        assert_eq!(extract_mi_stage(notes), None);
    }

    #[test]
    fn test_extract_mi_stage_with_leading_whitespace() {
        let notes = "  MI Stage: engage\nChange Talk: none";
        assert_eq!(extract_mi_stage(notes), Some("engage".to_string()));
    }

    #[test]
    fn test_strip_echoed_exchanges_present() {
        let notes = "MI Stage: engage\nChange Talk: none\n\nLATEST EXCHANGE:\nPerson: hello\nCoach: hi there";
        let stripped = strip_echoed_exchanges(notes);
        assert_eq!(stripped, "MI Stage: engage\nChange Talk: none");
    }

    #[test]
    fn test_strip_echoed_exchanges_multiple() {
        let notes = "MI Stage: focus\nKey Themes: drinking\n\nLATEST EXCHANGE:\nPerson: msg1\nCoach: resp1\n\nLATEST EXCHANGE:\nPerson: msg2\nCoach: resp2";
        let stripped = strip_echoed_exchanges(notes);
        assert_eq!(stripped, "MI Stage: focus\nKey Themes: drinking");
    }

    #[test]
    fn test_strip_echoed_exchanges_none() {
        let notes = "MI Stage: engage\nChange Talk: desire to quit";
        let stripped = strip_echoed_exchanges(notes);
        assert_eq!(stripped, notes);
    }
}
