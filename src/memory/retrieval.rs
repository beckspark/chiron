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
        user_facts: user_facts.unwrap_or_default(),
        session_summaries: session_summaries.unwrap_or_default(),
        significant_turns: significant_turns.unwrap_or_default(),
        mi_knowledge: mi_knowledge.unwrap_or_default(),
    }
}

/// Formats retrieved context into structured preamble sections.
///
/// Returns `None` if all sections are empty.
pub fn format_rag_context(ctx: &RetrievalContext) -> Option<String> {
    let mut sections = Vec::new();

    if !ctx.user_facts.is_empty() {
        let mut s = String::from("## What You Know About This Person");
        for fact in &ctx.user_facts {
            s.push_str(&format!("\n- [{}] {}", fact.fact_type, fact.content));
        }
        sections.push(s);
    }

    if !ctx.session_summaries.is_empty() {
        let mut s = String::from("## Previous Sessions");
        for summary in &ctx.session_summaries {
            s.push_str(&format!(
                "\n- [{}] {} (MI stage: {} -> {})",
                summary.created_at, summary.summary, summary.mi_stage_start, summary.mi_stage_end
            ));
        }
        sections.push(s);
    }

    if !ctx.significant_turns.is_empty() {
        let mut s = String::from("## Relevant Past Moments");
        for turn in &ctx.significant_turns {
            s.push_str(&format!(
                "\n- \"{}\" ({}, {})",
                turn.user_content, turn.signal_type, turn.talk_type
            ));
        }
        sections.push(s);
    }

    if !ctx.mi_knowledge.is_empty() {
        let mut s = String::from("## MI Technique Guidance");
        for k in &ctx.mi_knowledge {
            s.push_str(&format!("\n- [{}] {}", k.category, k.content));
        }
        sections.push(s);
    }

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
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
        assert!(format_rag_context(&ctx).is_none());
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
        let formatted = format_rag_context(&ctx).unwrap();
        assert!(formatted.contains("## What You Know About This Person"));
        assert!(formatted.contains("[goal] reduce drinking"));
        assert!(!formatted.contains("## Previous Sessions"));
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
        let formatted = format_rag_context(&ctx).unwrap();
        assert!(formatted.contains("## What You Know About This Person"));
        assert!(formatted.contains("## Previous Sessions"));
        assert!(formatted.contains("## Relevant Past Moments"));
        assert!(formatted.contains("## MI Technique Guidance"));
        assert!(formatted.contains("I went two days without a drink"));
        assert!(formatted.contains("[oars] Add meaning beyond"));
    }
}
