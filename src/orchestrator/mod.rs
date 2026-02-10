use std::collections::HashSet;
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
    /// The prompt provides only the LATEST EXCHANGE, PREVIOUS MI STAGE, and
    /// PREVIOUS THEMES — not the full previous notes. This removes the "What
    /// Changed" and "Coach Effectiveness" lines from the input so the 3B model
    /// has nothing to copy and must extract from the exchange.
    ///
    /// After generation, themes are merged programmatically (set union with
    /// previous themes preserved) so themes can never regress across turns.
    ///
    /// Runs after the response is printed — no perceived latency for the user.
    async fn run_supervisor(
        &self,
        input: &str,
        response: &str,
        existing_notes: Option<&str>,
    ) -> Result<()> {
        let t_start = Instant::now();

        let prev_themes = existing_notes.and_then(extract_themes).unwrap_or_default();
        let prompt = build_supervisor_prompt(input, response, existing_notes);

        let raw_notes = self
            .supervisor
            .chat(&prompt, vec![])
            .await
            .map_err(|e| anyhow::anyhow!("Supervisor inference failed: {e}"))?;

        // Strip think blocks — supervisor may reason internally but we only want structured notes
        let updated_notes = crate::provider::strip_think_blocks(&raw_notes);

        // Strip echoed exchange blocks — safety net for 3B model echo behavior
        let updated_notes = strip_echoed_exchanges(&updated_notes);

        // Programmatic theme accumulation — themes can never regress
        let new_themes = extract_themes(&updated_notes).unwrap_or_default();
        let merged = merge_themes(&prev_themes, &new_themes);
        let updated_notes = if !merged.is_empty() {
            replace_themes_line(&updated_notes, &merged)
        } else {
            updated_notes
        };

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
            themes = merged.join(", "),
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

/// Extracts themes from the `Running Themes:` line in case notes.
///
/// Handles markdown bold formatting (`**Running Themes:**`). Returns themes
/// as a lowercased, trimmed vector preserving original order.
fn extract_themes(notes: &str) -> Option<Vec<String>> {
    notes
        .lines()
        .map(|l| l.replace("**", ""))
        .find(|l| l.trim().to_lowercase().starts_with("running themes:"))
        .and_then(|l| {
            let (_, value) = l.split_once(':')?;
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.to_lowercase() == "none" {
                return None;
            }
            let themes: Vec<String> = trimmed
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            if themes.is_empty() {
                None
            } else {
                Some(themes)
            }
        })
}

/// Merges previous and new themes with order-preserving set union.
///
/// Previous themes keep their original order. New themes not already present
/// are appended at the end. Comparison is case-insensitive (all values are
/// expected to be lowercased by `extract_themes`).
fn merge_themes(previous: &[String], new: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = previous.iter().cloned().collect();
    let mut merged: Vec<String> = previous.to_vec();

    for theme in new {
        if seen.insert(theme.clone()) {
            merged.push(theme.clone());
        }
    }

    merged
}

/// Replaces the `Running Themes:` line in supervisor output with merged themes.
///
/// If no `Running Themes:` line exists, appends one. Handles markdown bold
/// formatting in the existing line.
fn replace_themes_line(notes: &str, merged: &[String]) -> String {
    let new_line = format!("Running Themes: {}", merged.join(", "));

    let mut found = false;
    let replaced: Vec<String> = notes
        .lines()
        .map(|l| {
            let clean = l.replace("**", "");
            if clean.trim().to_lowercase().starts_with("running themes:") {
                found = true;
                new_line.clone()
            } else {
                l.to_string()
            }
        })
        .collect();

    if found {
        replaced.join("\n")
    } else {
        format!("{}\n{}", notes.trim_end(), new_line)
    }
}

/// Builds the supervisor prompt from the exchange and previous notes context.
///
/// Only passes PREVIOUS MI STAGE and PREVIOUS THEMES to the model — not the
/// full previous notes. This eliminates the copy source for "What Changed"
/// and "Coach Effectiveness", forcing extraction from the exchange.
fn build_supervisor_prompt(input: &str, response: &str, existing_notes: Option<&str>) -> String {
    let prev_stage = existing_notes
        .and_then(extract_mi_stage)
        .unwrap_or_else(|| "none".to_string());
    let prev_themes = existing_notes.and_then(extract_themes).unwrap_or_default();

    let prev_themes_str = if prev_themes.is_empty() {
        "none".to_string()
    } else {
        prev_themes.join(", ")
    };

    format!(
        "LATEST EXCHANGE:\nPerson: {input}\nCoach: {response}\n\n\
         PREVIOUS MI STAGE: {prev_stage}\n\
         PREVIOUS THEMES: {prev_themes_str}"
    )
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

    // --- extract_themes tests ---

    #[test]
    fn test_extract_themes_plain() {
        let notes = "MI Stage: evoke\nRunning Themes: drinking, breakup, sleep";
        assert_eq!(
            extract_themes(notes),
            Some(vec![
                "drinking".to_string(),
                "breakup".to_string(),
                "sleep".to_string()
            ])
        );
    }

    #[test]
    fn test_extract_themes_bold() {
        let notes = "**MI Stage:** evoke\n**Running Themes:** drinking, breakup";
        assert_eq!(
            extract_themes(notes),
            Some(vec!["drinking".to_string(), "breakup".to_string()])
        );
    }

    #[test]
    fn test_extract_themes_case_normalization() {
        let notes = "Running Themes: Drinking, BREAKUP, Sleep";
        assert_eq!(
            extract_themes(notes),
            Some(vec![
                "drinking".to_string(),
                "breakup".to_string(),
                "sleep".to_string()
            ])
        );
    }

    #[test]
    fn test_extract_themes_extra_spaces() {
        let notes = "Running Themes:  drinking ,  breakup  ,  sleep  ";
        assert_eq!(
            extract_themes(notes),
            Some(vec![
                "drinking".to_string(),
                "breakup".to_string(),
                "sleep".to_string()
            ])
        );
    }

    #[test]
    fn test_extract_themes_missing() {
        let notes = "MI Stage: engage\nWhat Changed: Person mentioned stress";
        assert_eq!(extract_themes(notes), None);
    }

    #[test]
    fn test_extract_themes_empty_value() {
        let notes = "Running Themes:";
        assert_eq!(extract_themes(notes), None);
    }

    #[test]
    fn test_extract_themes_none_value() {
        let notes = "Running Themes: none";
        assert_eq!(extract_themes(notes), None);
    }

    // --- merge_themes tests ---

    #[test]
    fn test_merge_themes_union() {
        let prev = vec!["drinking".to_string(), "breakup".to_string()];
        let new = vec!["breakup".to_string(), "sleep".to_string()];
        assert_eq!(
            merge_themes(&prev, &new),
            vec![
                "drinking".to_string(),
                "breakup".to_string(),
                "sleep".to_string()
            ]
        );
    }

    #[test]
    fn test_merge_themes_order_preservation() {
        let prev = vec![
            "drinking".to_string(),
            "breakup".to_string(),
            "sleep".to_string(),
        ];
        let new = vec!["sister".to_string(), "drinking".to_string()];
        assert_eq!(
            merge_themes(&prev, &new),
            vec![
                "drinking".to_string(),
                "breakup".to_string(),
                "sleep".to_string(),
                "sister".to_string(),
            ]
        );
    }

    #[test]
    fn test_merge_themes_empty_previous() {
        let prev: Vec<String> = vec![];
        let new = vec!["drinking".to_string(), "breakup".to_string()];
        assert_eq!(
            merge_themes(&prev, &new),
            vec!["drinking".to_string(), "breakup".to_string()]
        );
    }

    #[test]
    fn test_merge_themes_empty_new() {
        let prev = vec!["drinking".to_string(), "breakup".to_string()];
        let new: Vec<String> = vec![];
        assert_eq!(
            merge_themes(&prev, &new),
            vec!["drinking".to_string(), "breakup".to_string()]
        );
    }

    #[test]
    fn test_merge_themes_both_empty() {
        let prev: Vec<String> = vec![];
        let new: Vec<String> = vec![];
        let result: Vec<String> = vec![];
        assert_eq!(merge_themes(&prev, &new), result);
    }

    #[test]
    fn test_merge_themes_all_duplicates() {
        let prev = vec!["drinking".to_string(), "breakup".to_string()];
        let new = vec!["breakup".to_string(), "drinking".to_string()];
        assert_eq!(
            merge_themes(&prev, &new),
            vec!["drinking".to_string(), "breakup".to_string()]
        );
    }

    // --- replace_themes_line tests ---

    #[test]
    fn test_replace_themes_line_standard() {
        let notes = "MI Stage: evoke\nWhat Changed: Person mentioned stress.\nCoach Effectiveness: Good reflections.\nRunning Themes: drinking, breakup";
        let merged = vec![
            "drinking".to_string(),
            "breakup".to_string(),
            "sleep".to_string(),
        ];
        let result = replace_themes_line(notes, &merged);
        assert!(result.contains("Running Themes: drinking, breakup, sleep"));
        assert!(result.contains("MI Stage: evoke"));
    }

    #[test]
    fn test_replace_themes_line_missing_appends() {
        let notes = "MI Stage: evoke\nWhat Changed: Person mentioned stress.\nCoach Effectiveness: Good reflections.";
        let merged = vec!["drinking".to_string(), "breakup".to_string()];
        let result = replace_themes_line(notes, &merged);
        assert!(result.ends_with("Running Themes: drinking, breakup"));
        assert!(result.contains("Coach Effectiveness: Good reflections."));
    }

    #[test]
    fn test_replace_themes_line_bold_format() {
        let notes = "**MI Stage:** evoke\n**What Changed:** Stress.\n**Coach Effectiveness:** Good.\n**Running Themes:** drinking";
        let merged = vec![
            "drinking".to_string(),
            "breakup".to_string(),
            "sleep".to_string(),
        ];
        let result = replace_themes_line(notes, &merged);
        assert!(result.contains("Running Themes: drinking, breakup, sleep"));
        // Bold on other lines should be preserved
        assert!(result.contains("**MI Stage:** evoke"));
    }

    // --- build_supervisor_prompt tests ---

    #[test]
    fn test_prompt_first_turn_no_existing_notes() {
        let prompt = build_supervisor_prompt(
            "I've been drinking every night since the breakup",
            "That sounds really tough.",
            None,
        );
        assert!(prompt.contains("Person: I've been drinking every night since the breakup"));
        assert!(prompt.contains("Coach: That sounds really tough."));
        assert!(prompt.contains("PREVIOUS MI STAGE: none"));
        assert!(prompt.contains("PREVIOUS THEMES: none"));
        // Must NOT contain PREVIOUS CASE NOTES or What Changed
        assert!(!prompt.contains("PREVIOUS CASE NOTES"));
        assert!(!prompt.contains("What Changed"));
        assert!(!prompt.contains("Coach Effectiveness"));
    }

    #[test]
    fn test_prompt_subsequent_turn_with_notes() {
        let existing = "MI Stage: engage\nWhat Changed: Person disclosed nightly drinking.\nCoach Effectiveness: Good open question.\nRunning Themes: drinking, breakup";
        let prompt = build_supervisor_prompt(
            "Drinking is the only way I can sleep",
            "It sounds like sleep is a big part of this.",
            Some(existing),
        );
        assert!(prompt.contains("Person: Drinking is the only way I can sleep"));
        assert!(prompt.contains("PREVIOUS MI STAGE: engage"));
        assert!(prompt.contains("PREVIOUS THEMES: drinking, breakup"));
        // Must NOT contain the full previous notes content
        assert!(!prompt.contains("What Changed: Person disclosed"));
        assert!(!prompt.contains("Coach Effectiveness: Good open"));
    }

    #[test]
    fn test_multi_turn_theme_accumulation() {
        // Simulate 3-turn accumulation entirely through helpers
        // Turn 1: no previous themes
        let t1_output = "MI Stage: engage\nWhat Changed: Person disclosed drinking.\nCoach Effectiveness: Open question.\nRunning Themes: drinking, breakup";
        let prev_themes_1: Vec<String> = vec![];
        let new_themes_1 = extract_themes(t1_output).unwrap();
        let merged_1 = merge_themes(&prev_themes_1, &new_themes_1);
        assert_eq!(merged_1, vec!["drinking", "breakup"]);

        // Turn 2: previous themes from turn 1
        let t2_output = "MI Stage: focus\nWhat Changed: Person links drinking to sleep.\nCoach Effectiveness: Good reflection.\nRunning Themes: drinking, sleep";
        let new_themes_2 = extract_themes(t2_output).unwrap();
        let merged_2 = merge_themes(&merged_1, &new_themes_2);
        // "breakup" must be preserved even though model dropped it
        assert_eq!(merged_2, vec!["drinking", "breakup", "sleep"]);

        // Turn 3: model only outputs new theme, drops all old ones
        let t3_output = "MI Stage: evoke\nWhat Changed: Person tried sobriety.\nCoach Effectiveness: Affirming attempt.\nRunning Themes: sobriety attempt";
        let new_themes_3 = extract_themes(t3_output).unwrap();
        let merged_3 = merge_themes(&merged_2, &new_themes_3);
        // ALL previous themes preserved + new one added
        assert_eq!(
            merged_3,
            vec!["drinking", "breakup", "sleep", "sobriety attempt"]
        );

        // Verify replace_themes_line produces correct output
        let final_notes = replace_themes_line(t3_output, &merged_3);
        assert!(final_notes
            .contains("Running Themes: drinking, breakup, sleep, sobriety attempt"));
    }
}
