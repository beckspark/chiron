use anyhow::Result;
use rig::vector_store::VectorStoreIndex;
use rig_lancedb::LanceDBFilter;

use super::vectors::{self, MiKnowledge, SessionSummary, SignificantTurn, UserFact};

/// Collected RAG context from all vector store tables.
#[derive(Debug, Default)]
pub struct RetrievalContext {
    pub user_facts: Vec<UserFact>,
    pub session_summaries: Vec<SessionSummary>,
    pub significant_turns: Vec<SignificantTurn>,
    pub mi_knowledge: Vec<MiKnowledge>,
}

/// Retrieves relevant context from all vector store tables for a given query.
///
/// Queries run in parallel via `tokio::join!`. Tables that are empty or fail
/// to query degrade gracefully (empty results, logged warning).
pub async fn retrieve_context(
    conn: &lancedb::Connection,
    embedding_model: &rig_fastembed::EmbeddingModel,
    query: &str,
    mi_stage: Option<&str>,
    top_k: usize,
) -> RetrievalContext {
    let (user_facts, session_summaries, significant_turns, mi_knowledge) = tokio::join!(
        query_user_facts(conn, embedding_model, query, top_k),
        query_session_summaries(conn, embedding_model, query, top_k),
        query_significant_turns(conn, embedding_model, query, top_k),
        query_mi_knowledge(conn, embedding_model, query, mi_stage, top_k),
    );

    RetrievalContext {
        user_facts: log_retrieval_err("user_knowledge", user_facts),
        session_summaries: log_retrieval_err("session_summaries", session_summaries),
        significant_turns: log_retrieval_err("significant_turns", significant_turns),
        mi_knowledge: log_retrieval_err("mi_knowledge", mi_knowledge),
    }
}

/// Formats retrieved context into structured preamble sections.
///
/// Sections are built in priority order (user facts > significant turns >
/// session summaries > MI knowledge). If `max_chars` is exceeded, remaining
/// sections are omitted. Returns `None` if all sections are empty.
pub fn format_rag_context(ctx: &RetrievalContext, max_chars: usize) -> Option<String> {
    let mut sections = Vec::new();
    let mut total_len = 0;
    let separator_len = 2; // "\n\n" between sections

    // Build sections in priority order, stop when budget exhausted
    let candidate_sections = [
        build_user_facts_section(&ctx.user_facts),
        build_significant_turns_section(&ctx.significant_turns),
        build_session_summaries_section(&ctx.session_summaries),
        build_mi_knowledge_section(&ctx.mi_knowledge),
    ];

    for section in candidate_sections.into_iter().flatten() {
        let added_len = section.len() + if sections.is_empty() { 0 } else { separator_len };
        if total_len + added_len > max_chars {
            break;
        }
        total_len += added_len;
        sections.push(section);
    }

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

fn build_user_facts_section(facts: &[UserFact]) -> Option<String> {
    if facts.is_empty() { return None; }
    let mut s = String::from("## What You Know About This Person");
    for fact in facts {
        s.push_str(&format!("\n- [{}] {}", fact.fact_type, fact.content));
    }
    Some(s)
}

fn build_significant_turns_section(turns: &[SignificantTurn]) -> Option<String> {
    if turns.is_empty() { return None; }
    let mut s = String::from("## Relevant Past Moments");
    for turn in turns {
        s.push_str(&format!("\n- \"{}\" ({}, {})", turn.user_content, turn.signal_type, turn.talk_type));
    }
    Some(s)
}

fn build_session_summaries_section(summaries: &[SessionSummary]) -> Option<String> {
    if summaries.is_empty() { return None; }
    let mut s = String::from("## Previous Sessions");
    for summary in summaries {
        s.push_str(&format!(
            "\n- [{}] {} (MI stage: {} -> {})",
            summary.created_at, summary.summary, summary.mi_stage_start, summary.mi_stage_end
        ));
    }
    Some(s)
}

fn build_mi_knowledge_section(knowledge: &[MiKnowledge]) -> Option<String> {
    if knowledge.is_empty() { return None; }
    let mut s = String::from("## MI Technique Guidance");
    for k in knowledge {
        s.push_str(&format!("\n- [{}] {}", k.category, k.content));
    }
    Some(s)
}

fn log_retrieval_err<T: Default>(table: &str, result: Result<T>) -> T {
    match result {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(table, error = %e, "RAG retrieval failed, skipping");
            T::default()
        }
    }
}

// ─── Per-table query helpers ────────────────────────────────────────────────

async fn query_user_facts(
    conn: &lancedb::Connection,
    model: &rig_fastembed::EmbeddingModel,
    query: &str,
    top_k: usize,
) -> Result<Vec<UserFact>> {
    let index = vectors::vector_index(conn, "user_knowledge", model.clone()).await?;
    let request = build_request(query, top_k)?;
    let results: Vec<(f64, String, UserFact)> = index.top_n(request).await?;
    Ok(results.into_iter().map(|(_, _, fact)| fact).collect())
}

async fn query_session_summaries(
    conn: &lancedb::Connection,
    model: &rig_fastembed::EmbeddingModel,
    query: &str,
    top_k: usize,
) -> Result<Vec<SessionSummary>> {
    let index = vectors::vector_index(conn, "session_summaries", model.clone()).await?;
    let request = build_request(query, top_k)?;
    let results: Vec<(f64, String, SessionSummary)> = index.top_n(request).await?;
    Ok(results.into_iter().map(|(_, _, s)| s).collect())
}

async fn query_significant_turns(
    conn: &lancedb::Connection,
    model: &rig_fastembed::EmbeddingModel,
    query: &str,
    top_k: usize,
) -> Result<Vec<SignificantTurn>> {
    let index = vectors::vector_index(conn, "significant_turns", model.clone()).await?;
    let request = build_request(query, top_k)?;
    let results: Vec<(f64, String, SignificantTurn)> = index.top_n(request).await?;
    Ok(results.into_iter().map(|(_, _, t)| t).collect())
}

async fn query_mi_knowledge(
    conn: &lancedb::Connection,
    model: &rig_fastembed::EmbeddingModel,
    query: &str,
    mi_stage: Option<&str>,
    top_k: usize,
) -> Result<Vec<MiKnowledge>> {
    let index = vectors::vector_index(conn, "mi_knowledge", model.clone()).await?;
    let request = match mi_stage {
        Some(stage) => build_filtered_request(query, top_k, stage)?,
        None => build_request(query, top_k)?,
    };
    let results: Vec<(f64, String, MiKnowledge)> = index.top_n(request).await?;
    Ok(results.into_iter().map(|(_, _, k)| k).collect())
}

fn build_request(
    query: &str,
    top_k: usize,
) -> Result<rig::vector_store::request::VectorSearchRequest<LanceDBFilter>> {
    rig::vector_store::request::VectorSearchRequest::<LanceDBFilter>::builder()
        .query(query)
        .samples(top_k as u64)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build search request: {e}"))
}

fn build_filtered_request(
    query: &str,
    top_k: usize,
    mi_stage: &str,
) -> Result<rig::vector_store::request::VectorSearchRequest<LanceDBFilter>> {
    // Use LIKE to match stage within comma-separated mi_stages field
    let filter = LanceDBFilter::like("mi_stages".to_string(), format!("%{mi_stage}%"));
    rig::vector_store::request::VectorSearchRequest::<LanceDBFilter>::builder()
        .query(query)
        .samples(top_k as u64)
        .filter(filter)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build filtered search request: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_empty_context() {
        let ctx = RetrievalContext::default();
        assert!(format_rag_context(&ctx, 4000).is_none());
    }

    /// Verifies that retrieve_context on empty tables returns empty results without errors.
    #[tokio::test]
    async fn test_retrieve_from_empty_tables() {
        use crate::memory::embeddings::init_embedding_model;
        use crate::memory::vectors;

        let dir = tempfile::tempdir().unwrap();
        let conn = vectors::open_vector_db(dir.path().to_str().unwrap())
            .await
            .unwrap();
        vectors::ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();
        let ctx = retrieve_context(&conn, &model, "how are you feeling today", None, 3).await;

        assert!(ctx.user_facts.is_empty());
        assert!(ctx.session_summaries.is_empty());
        assert!(ctx.significant_turns.is_empty());
        assert!(ctx.mi_knowledge.is_empty());
        assert!(format_rag_context(&ctx, 4000).is_none());
    }

    #[test]
    fn test_format_user_facts_only() {
        let ctx = RetrievalContext {
            user_facts: vec![UserFact {
                id: "1".into(),
                fact_type: "goal".into(),
                content: "reduce drinking".into(),
                source_session: "s1".into(),
                last_confirmed: "s1".into(),
                created_at: "2026-03-22".into(),
                updated_at: "2026-03-22".into(),
            }],
            ..Default::default()
        };
        let formatted = format_rag_context(&ctx, 4000).unwrap();
        assert!(formatted.contains("## What You Know About This Person"));
        assert!(formatted.contains("[goal] reduce drinking"));
        assert!(!formatted.contains("## Previous Sessions"));
    }

    /// Full round-trip: embed → insert → retrieve → format → preamble assembly.
    /// Tests the complete data path from vector store to the model's system prompt.
    #[tokio::test]
    async fn test_rag_context_round_trip() {
        use crate::agents::peer::build_peer_coach_preamble;
        use crate::memory::embeddings::init_embedding_model;
        use crate::memory::vectors;
        use rig::embeddings::EmbeddingModel as _;

        let dir = tempfile::tempdir().unwrap();
        let conn = vectors::open_vector_db(dir.path().to_str().unwrap())
            .await
            .unwrap();
        vectors::ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();

        // Build a UserFact and embed its content
        let fact_content = "wants to cut back on drinking to weekends only";
        let fact = vectors::UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "goal".into(),
            content: fact_content.into(),
            source_session: "session-001".into(),
            last_confirmed: "session-001".into(),
            created_at: "2026-03-22".into(),
            updated_at: "2026-03-22".into(),
        };

        let embedding = model.embed_text(fact_content).await.unwrap();
        vectors::add_user_fact(&conn, &fact, &embedding.vec).await.unwrap();

        // Retrieve with a relevant query
        let ctx = retrieve_context(&conn, &model, "tell me about your drinking habits", None, 3).await;
        assert!(!ctx.user_facts.is_empty(), "should retrieve the inserted fact");
        assert!(
            ctx.user_facts[0].content.contains("drinking"),
            "retrieved fact should be about drinking"
        );

        // Format into RAG context
        let formatted = format_rag_context(&ctx, 4000).expect("should produce formatted context");
        assert!(formatted.contains("## What You Know About This Person"));
        assert!(formatted.contains("cut back on drinking"));

        // Pass through preamble builder and verify placement
        let preamble = build_peer_coach_preamble(
            "You are a peer supporter.",
            Some("Think carefully."),
            Some("MI Stage: evoke\nRunning Themes: drinking"),
            None,
            Some(&formatted),
        );

        // RAG context present and in correct position (before case notes)
        let rag_pos = preamble.find("What You Know About This Person").unwrap();
        let notes_pos = preamble.find("Session Context").unwrap();
        assert!(rag_pos < notes_pos, "RAG context should precede case notes in preamble");
        assert!(preamble.contains("cut back on drinking"), "fact content in preamble");
    }

    #[test]
    fn test_format_all_sections() {
        let ctx = RetrievalContext {
            user_facts: vec![UserFact {
                id: "1".into(),
                fact_type: "value".into(),
                content: "independence".into(),
                source_session: "s1".into(),
                last_confirmed: "s1".into(),
                created_at: "2026-03-22".into(),
                updated_at: "2026-03-22".into(),
            }],
            session_summaries: vec![SessionSummary {
                id: "2".into(),
                session_id: "s1".into(),
                summary: "Discussed anxiety".into(),
                mi_stage_start: "engage".into(),
                mi_stage_end: "focus".into(),
                themes: "anxiety".into(),
                turn_count: 5,
                created_at: "2 days ago".into(),
            }],
            significant_turns: vec![SignificantTurn {
                id: "3".into(),
                session_id: "s1".into(),
                turn_number: 3,
                user_content: "I went two days without a drink".into(),
                assistant_content: "That sounds important".into(),
                signal_type: "change_talk".into(),
                mi_stage: "evoke".into(),
                talk_type: "taking_steps".into(),
                created_at: "2026-03-20".into(),
            }],
            mi_knowledge: vec![MiKnowledge {
                id: "4".into(),
                category: "oars".into(),
                subcategory: "complex_reflection".into(),
                content: "Add meaning beyond what was stated".into(),
                mi_stages: "evoke,plan".into(),
                source: "mi_text_summary.md".into(),
            }],
        };
        let formatted = format_rag_context(&ctx, 4000).unwrap();
        assert!(formatted.contains("## What You Know About This Person"));
        assert!(formatted.contains("## Previous Sessions"));
        assert!(formatted.contains("## Relevant Past Moments"));
        assert!(formatted.contains("## MI Technique Guidance"));
        assert!(formatted.contains("I went two days without a drink"));
        assert!(formatted.contains("[oars] Add meaning beyond"));
    }
}
