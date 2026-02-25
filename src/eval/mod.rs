pub mod catalog;
pub mod results;
pub mod script;

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use rig::agent::AgentBuilder;
use rig::completion::{Chat, Message};
use rig::embeddings::EmbeddingModel;
use rig_sqlite::SqliteVectorStore;
use tokio_rusqlite::Connection;

use crate::agents::peer::build_peer_coach_preamble;
use crate::memory::case_notes;
use crate::memory::store::MiKnowledge;
use crate::orchestrator::{
    build_supervisor_prompt, extract_mi_stage, extract_themes, merge_themes, replace_themes_line,
    strip_echoed_exchanges,
};
use crate::provider::completion::CandleCompletionModel;

use self::catalog::{PromptCatalog, PromptVariant};
use self::results::{CombinationResult, EvalRun, EvalTurnResult};
use self::script::TestScript;

/// Runs the full evaluation matrix: all (coach x supervisor) combinations across
/// every turn in the test script.
///
/// For each combination, case notes are wiped clean so state doesn't leak between
/// runs. Results are written to `<output_dir>/eval_<timestamp>.json`.
///
/// # Errors
///
/// Returns an error if catalog/script files can't be read or parsed, if the
/// database operations fail, or if agent inference fails.
#[tracing::instrument(level = "info", skip(provider_model, knowledge_store, embedding_model, chat_conn))]
pub async fn run_eval<E>(
    provider_model: CandleCompletionModel,
    knowledge_store: SqliteVectorStore<E, MiKnowledge>,
    embedding_model: E,
    rag_top_k: usize,
    chat_conn: Connection,
    coach_catalog_path: &Path,
    supervisor_catalog_path: &Path,
    script_path: &Path,
    output_dir: &Path,
) -> Result<()>
where
    E: EmbeddingModel + Clone + 'static,
{
    // Load catalogs and script
    let coach_catalog = PromptCatalog::load(coach_catalog_path)?;
    let supervisor_catalog = PromptCatalog::load(supervisor_catalog_path)?;

    let script: TestScript = toml::from_str(
        &fs::read_to_string(script_path)
            .with_context(|| format!("Failed to read {}", script_path.display()))?,
    )
    .with_context(|| format!("Failed to parse {}", script_path.display()))?;

    let total_combos = coach_catalog.variants.len() * supervisor_catalog.variants.len();
    let total_turns = script.turns.len();
    println!(
        "=== Eval: {} combinations x {} turns ===",
        total_combos, total_turns
    );

    let run_start = Instant::now();
    let mut combinations = Vec::new();

    for coach_variant in &coach_catalog.variants {
        for supervisor_variant in &supervisor_catalog.variants {
            let combo_result = run_combination(
                &provider_model,
                &knowledge_store,
                &embedding_model,
                rag_top_k,
                &chat_conn,
                coach_variant,
                supervisor_variant,
                &script,
            )
            .await?;

            // Print summary line
            let stages: Vec<&str> = combo_result
                .turns
                .iter()
                .map(|t| {
                    t.mi_stage
                        .as_deref()
                        .map(|s| match s {
                            "engage" => "E",
                            "focus" => "F",
                            "evoke" => "V",
                            "plan" => "P",
                            _ => "?",
                        })
                        .unwrap_or("?")
                })
                .collect();
            let last_themes = combo_result
                .turns
                .last()
                .map(|t| t.themes.len())
                .unwrap_or(0);
            println!(
                "  {} + {}:  {:.1}s  themes: {}  stages: {}",
                coach_variant.id,
                supervisor_variant.id,
                combo_result.total_duration_ms as f64 / 1000.0,
                last_themes,
                stages.join("→"),
            );

            combinations.push(combo_result);
        }
    }

    let eval_run = EvalRun {
        run_id: chrono_timestamp(),
        model: "smollm3-3b".to_string(),
        test_script: script.id.clone(),
        seed: 42,
        architecture: "two-agent".to_string(),
        total_duration_ms: run_start.elapsed().as_millis() as u64,
        combinations,
    };

    // Write results
    fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create {}", output_dir.display()))?;
    let output_path = output_dir.join(format!("eval_{}.json", eval_run.run_id));
    let json = serde_json::to_string_pretty(&eval_run).context("Failed to serialize eval run")?;
    fs::write(&output_path, &json)
        .with_context(|| format!("Failed to write {}", output_path.display()))?;

    println!(
        "\n=== Eval Complete ({} combinations x {} turns) in {:.1}s ===",
        total_combos,
        total_turns,
        eval_run.total_duration_ms as f64 / 1000.0,
    );
    println!("Results written to: {}", output_path.display());

    Ok(())
}

/// Runs a single (coach, supervisor) combination across all script turns.
///
/// Wipes case notes before starting so each combination gets a clean slate.
async fn run_combination<E>(
    provider_model: &CandleCompletionModel,
    knowledge_store: &SqliteVectorStore<E, MiKnowledge>,
    embedding_model: &E,
    rag_top_k: usize,
    chat_conn: &Connection,
    coach_variant: &PromptVariant,
    supervisor_variant: &PromptVariant,
    script: &TestScript,
) -> Result<CombinationResult>
where
    E: EmbeddingModel + Clone + 'static,
{
    let combo_start = Instant::now();

    // Clean slate: delete all case notes so this combination starts fresh
    delete_all_case_notes(chat_conn).await?;

    let mut chat_history: Vec<Message> = Vec::new();
    let mut turns = Vec::new();

    for (i, test_turn) in script.turns.iter().enumerate() {
        let turn_number = i + 1;
        tracing::info!(
            turn = turn_number,
            coach = &coach_variant.id,
            supervisor = &supervisor_variant.id,
            "Running eval turn"
        );

        // 1. Load case notes (same as production)
        let existing_notes = case_notes::get_latest_case_note(chat_conn).await?;

        // 2. Build coach with variant preamble + case notes + RAG
        let preamble =
            build_peer_coach_preamble(&coach_variant.preamble, None, existing_notes.as_deref());
        let knowledge_index = knowledge_store.clone().index(embedding_model.clone());
        let peer_coach = AgentBuilder::new(provider_model.clone())
            .preamble(&preamble)
            .dynamic_context(rag_top_k, knowledge_index)
            .temperature(coach_variant.temperature)
            .max_tokens(coach_variant.max_tokens as u64)
            .build();

        // 3. Run coach (non-streaming for eval)
        let coach_start = Instant::now();
        let raw_coach_response = peer_coach
            .chat(&test_turn.input, chat_history.clone())
            .await
            .map_err(|e| anyhow::anyhow!("Coach inference failed: {e}"))?;
        let coach_response = crate::provider::strip_think_blocks(&raw_coach_response);
        let coach_ms = coach_start.elapsed().as_millis() as u64;

        // 4. Run supervisor with variant preamble
        let supervisor = AgentBuilder::new(provider_model.clone())
            .preamble(&supervisor_variant.preamble)
            .temperature(supervisor_variant.temperature)
            .max_tokens(supervisor_variant.max_tokens as u64)
            .build();

        let prev_themes = existing_notes
            .as_deref()
            .and_then(extract_themes)
            .unwrap_or_default();
        let prompt =
            build_supervisor_prompt(&test_turn.input, &coach_response, existing_notes.as_deref());

        let supervisor_start = Instant::now();
        let raw_notes = supervisor
            .chat(&prompt, vec![])
            .await
            .map_err(|e| anyhow::anyhow!("Supervisor inference failed: {e}"))?;
        let supervisor_ms = supervisor_start.elapsed().as_millis() as u64;

        // 5. Post-process (same as production)
        let updated_notes = crate::provider::strip_think_blocks(&raw_notes);
        let updated_notes = strip_echoed_exchanges(&updated_notes);
        let new_themes = extract_themes(&updated_notes).unwrap_or_default();
        let merged = merge_themes(&prev_themes, &new_themes);
        let updated_notes = if !merged.is_empty() {
            replace_themes_line(&updated_notes, &merged)
        } else {
            updated_notes
        };
        let mi_stage = extract_mi_stage(&updated_notes);

        // 6. Save case notes + update chat history
        case_notes::save_case_note(
            chat_conn,
            "eval",
            turn_number as i32,
            mi_stage.as_deref(),
            &updated_notes,
        )
        .await?;

        chat_history.push(Message::user(&test_turn.input));
        chat_history.push(Message::assistant(&coach_response));

        // 7. Record turn result
        turns.push(EvalTurnResult {
            turn_number,
            input: test_turn.input.clone(),
            coach_response,
            case_notes: updated_notes,
            mi_stage,
            themes: merged,
            detected_mode: None,
            route_confidence: None,
            coach_ms,
            supervisor_ms,
        });
    }

    Ok(CombinationResult {
        coach_variant: coach_variant.id.clone(),
        supervisor_variant: supervisor_variant.id.clone(),
        total_duration_ms: combo_start.elapsed().as_millis() as u64,
        turns,
    })
}

/// Deletes all case notes from the database. Used between eval combinations
/// to prevent state leakage.
async fn delete_all_case_notes(conn: &Connection) -> Result<()> {
    conn.call(|conn| {
        conn.execute("DELETE FROM case_notes", [])?;
        Ok(())
    })
    .await
    .context("Failed to delete case notes for eval reset")?;
    Ok(())
}

/// Generates a filesystem-safe timestamp string for output filenames.
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Convert to a basic ISO-ish format without external dependencies
    // Format: YYYY-MM-DDThh-mm-ss (approximate from unix timestamp)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Simplified date from days since epoch (good enough for filenames)
    // Using a basic algorithm for year/month/day from days since 1970-01-01
    let (year, month, day) = days_to_date(days);

    format!(
        "{year:04}-{month:02}-{day:02}T{hours:02}-{minutes:02}-{seconds:02}"
    )
}

/// Converts days since Unix epoch to (year, month, day).
fn days_to_date(days_since_epoch: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm (Howard Hinnant)
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
