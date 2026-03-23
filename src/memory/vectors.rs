use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float64Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::Connection;
use rig_fastembed::EmbeddingModel;
use rig_lancedb::{LanceDbVectorIndex, SearchParams};
use serde::{Deserialize, Serialize};

/// Embedding dimensionality for BGE-Small-EN-V1.5.
const EMBEDDING_DIM: usize = 384;

/// Opens or creates a LanceDB database at the given path.
pub async fn open_vector_db(path: &str) -> Result<Connection> {
    lancedb::connect(path)
        .execute()
        .await
        .context("Failed to open LanceDB")
}

/// Creates all vector tables if they don't already exist.
pub async fn ensure_tables(conn: &Connection) -> Result<()> {
    let existing = conn
        .table_names()
        .execute()
        .await
        .context("Failed to list tables")?;

    if !existing.contains(&"user_knowledge".to_string()) {
        conn.create_empty_table("user_knowledge", user_knowledge_schema())
            .execute()
            .await
            .context("Failed to create user_knowledge table")?;
    }

    if !existing.contains(&"session_summaries".to_string()) {
        conn.create_empty_table("session_summaries", session_summaries_schema())
            .execute()
            .await
            .context("Failed to create session_summaries table")?;
    }

    if !existing.contains(&"session_checkpoints".to_string()) {
        conn.create_empty_table("session_checkpoints", session_checkpoints_schema())
            .execute()
            .await
            .context("Failed to create session_checkpoints table")?;
    }

    if !existing.contains(&"significant_turns".to_string()) {
        conn.create_empty_table("significant_turns", significant_turns_schema())
            .execute()
            .await
            .context("Failed to create significant_turns table")?;
    }

    if !existing.contains(&"mi_knowledge".to_string()) {
        conn.create_empty_table("mi_knowledge", mi_knowledge_schema())
            .execute()
            .await
            .context("Failed to create mi_knowledge table")?;
    }

    tracing::info!("Vector store tables verified");
    Ok(())
}

// ─── Table schemas ──────────────────────────────────────────────────────────

fn embedding_field() -> Field {
    Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, true)),
            EMBEDDING_DIM as i32,
        ),
        false,
    )
}

fn user_knowledge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("fact_type", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("source_session", DataType::Utf8, false),
        Field::new("last_confirmed", DataType::Utf8, false),
        Field::new("created_at", DataType::Utf8, false),
        Field::new("updated_at", DataType::Utf8, false),
        embedding_field(),
    ]))
}

fn session_summaries_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("mi_stage_start", DataType::Utf8, false),
        Field::new("mi_stage_end", DataType::Utf8, false),
        Field::new("themes", DataType::Utf8, false),
        Field::new("turn_count", DataType::Int32, false),
        Field::new("created_at", DataType::Utf8, false),
        embedding_field(),
    ]))
}

fn session_checkpoints_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("checkpoint_number", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("mi_stage", DataType::Utf8, false),
        Field::new("themes", DataType::Utf8, false),
        Field::new("turn_range", DataType::Utf8, false),
        Field::new("created_at", DataType::Utf8, false),
        embedding_field(),
    ]))
}

fn significant_turns_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("turn_number", DataType::Int32, false),
        Field::new("user_content", DataType::Utf8, false),
        Field::new("assistant_content", DataType::Utf8, false),
        Field::new("signal_type", DataType::Utf8, false),
        Field::new("mi_stage", DataType::Utf8, false),
        Field::new("talk_type", DataType::Utf8, false),
        Field::new("created_at", DataType::Utf8, false),
        embedding_field(),
    ]))
}

fn mi_knowledge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("subcategory", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("mi_stages", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, false),
        embedding_field(),
    ]))
}

// ─── Data types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFact {
    pub id: String,
    pub fact_type: String,
    pub content: String,
    pub source_session: String,
    pub last_confirmed: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub session_id: String,
    pub summary: String,
    pub mi_stage_start: String,
    pub mi_stage_end: String,
    pub themes: String,
    pub turn_count: i32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    pub id: String,
    pub session_id: String,
    pub checkpoint_number: i32,
    pub content: String,
    pub mi_stage: String,
    pub themes: String,
    pub turn_range: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificantTurn {
    pub id: String,
    pub session_id: String,
    pub turn_number: i32,
    pub user_content: String,
    pub assistant_content: String,
    pub signal_type: String,
    pub mi_stage: String,
    pub talk_type: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiKnowledge {
    pub id: String,
    pub category: String,
    pub subcategory: String,
    pub content: String,
    pub mi_stages: String,
    pub source: String,
}

// ─── Insert helpers ─────────────────────────────────────────────────────────

fn make_embedding_array(embedding: &[f64]) -> FixedSizeListArray {
    let values: ArrayRef = Arc::new(Float64Array::from(embedding.to_vec()));
    let field = Arc::new(Field::new("item", DataType::Float64, true));
    FixedSizeListArray::try_new(field, EMBEDDING_DIM as i32, values, None)
        .expect("embedding dimension mismatch")
}

/// Adds a user fact with its embedding to the user_knowledge table.
pub async fn add_user_fact(
    conn: &Connection,
    fact: &UserFact,
    embedding: &[f64],
) -> Result<()> {
    let schema = user_knowledge_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![fact.id.as_str()])),
            Arc::new(StringArray::from(vec![fact.fact_type.as_str()])),
            Arc::new(StringArray::from(vec![fact.content.as_str()])),
            Arc::new(StringArray::from(vec![fact.source_session.as_str()])),
            Arc::new(StringArray::from(vec![fact.last_confirmed.as_str()])),
            Arc::new(StringArray::from(vec![fact.created_at.as_str()])),
            Arc::new(StringArray::from(vec![fact.updated_at.as_str()])),
            Arc::new(make_embedding_array(embedding)),
        ],
    )
    .context("Failed to create user_fact RecordBatch")?;

    let table = conn.open_table("user_knowledge").execute().await?;
    table
        .add(RecordBatchIterator::new(vec![Ok(batch)], schema))
        .execute()
        .await
        .context("Failed to insert user fact")?;
    Ok(())
}

/// Result of an upsert operation on user facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpsertResult {
    /// A new fact was inserted.
    Inserted,
    /// An existing fact was updated (id of the matched fact).
    Updated(String),
}

/// Cosine distance threshold for considering two user facts as duplicates.
/// 0.20 corresponds to ~0.90 cosine similarity.
const DEDUP_DISTANCE_THRESHOLD: f64 = 0.20;

/// Upserts a user fact: inserts if no semantically similar fact with the same
/// `fact_type` exists, otherwise updates `last_confirmed` and `updated_at` on
/// the existing match.
///
/// Uses vector similarity search to find duplicates. A match is considered
/// a duplicate if it has the same `fact_type` and cosine distance < threshold.
pub async fn upsert_user_fact(
    conn: &Connection,
    model: &EmbeddingModel,
    fact: &UserFact,
    embedding: &[f64],
) -> Result<UpsertResult> {
    use rig::vector_store::VectorStoreIndex;

    // Search for a similar existing fact
    let index = vector_index(conn, "user_knowledge", model.clone()).await?;
    let request =
        rig::vector_store::request::VectorSearchRequest::<rig_lancedb::LanceDBFilter>::builder()
            .query(&fact.content)
            .samples(1)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build dedup search request: {e}"))?;

    let results: Vec<(f64, String, UserFact)> = index.top_n(request).await?;

    // Check if top result is a close match with same fact_type
    if let Some((distance, _, existing)) = results.first() {
        if *distance < DEDUP_DISTANCE_THRESHOLD && existing.fact_type == fact.fact_type {
            // Update existing fact's confirmation timestamps
            let table = conn.open_table("user_knowledge").execute().await?;
            table
                .update()
                .only_if(format!("id = '{}'", existing.id))
                .column("last_confirmed", format!("'{}'", fact.last_confirmed))
                .column("updated_at", format!("'{}'", fact.updated_at))
                .execute()
                .await
                .context("Failed to update existing user fact")?;

            return Ok(UpsertResult::Updated(existing.id.clone()));
        }
    }

    // No close match — insert as new
    add_user_fact(conn, fact, embedding).await?;
    Ok(UpsertResult::Inserted)
}

/// Adds a session summary with its embedding.
pub async fn add_session_summary(
    conn: &Connection,
    summary: &SessionSummary,
    embedding: &[f64],
) -> Result<()> {
    let schema = session_summaries_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![summary.id.as_str()])),
            Arc::new(StringArray::from(vec![summary.session_id.as_str()])),
            Arc::new(StringArray::from(vec![summary.summary.as_str()])),
            Arc::new(StringArray::from(vec![summary.mi_stage_start.as_str()])),
            Arc::new(StringArray::from(vec![summary.mi_stage_end.as_str()])),
            Arc::new(StringArray::from(vec![summary.themes.as_str()])),
            Arc::new(Int32Array::from(vec![summary.turn_count])),
            Arc::new(StringArray::from(vec![summary.created_at.as_str()])),
            Arc::new(make_embedding_array(embedding)),
        ],
    )
    .context("Failed to create session_summary RecordBatch")?;

    let table = conn.open_table("session_summaries").execute().await?;
    table
        .add(RecordBatchIterator::new(vec![Ok(batch)], schema))
        .execute()
        .await
        .context("Failed to insert session summary")?;
    Ok(())
}

/// Adds a session checkpoint with its embedding.
pub async fn add_session_checkpoint(
    conn: &Connection,
    checkpoint: &SessionCheckpoint,
    embedding: &[f64],
) -> Result<()> {
    let schema = session_checkpoints_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![checkpoint.id.as_str()])),
            Arc::new(StringArray::from(vec![checkpoint.session_id.as_str()])),
            Arc::new(Int32Array::from(vec![checkpoint.checkpoint_number])),
            Arc::new(StringArray::from(vec![checkpoint.content.as_str()])),
            Arc::new(StringArray::from(vec![checkpoint.mi_stage.as_str()])),
            Arc::new(StringArray::from(vec![checkpoint.themes.as_str()])),
            Arc::new(StringArray::from(vec![checkpoint.turn_range.as_str()])),
            Arc::new(StringArray::from(vec![checkpoint.created_at.as_str()])),
            Arc::new(make_embedding_array(embedding)),
        ],
    )
    .context("Failed to create session_checkpoint RecordBatch")?;

    let table = conn.open_table("session_checkpoints").execute().await?;
    table
        .add(RecordBatchIterator::new(vec![Ok(batch)], schema))
        .execute()
        .await
        .context("Failed to insert session checkpoint")?;
    Ok(())
}

/// Adds a significant turn with its embedding.
pub async fn add_significant_turn(
    conn: &Connection,
    turn: &SignificantTurn,
    embedding: &[f64],
) -> Result<()> {
    let schema = significant_turns_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![turn.id.as_str()])),
            Arc::new(StringArray::from(vec![turn.session_id.as_str()])),
            Arc::new(Int32Array::from(vec![turn.turn_number])),
            Arc::new(StringArray::from(vec![turn.user_content.as_str()])),
            Arc::new(StringArray::from(vec![turn.assistant_content.as_str()])),
            Arc::new(StringArray::from(vec![turn.signal_type.as_str()])),
            Arc::new(StringArray::from(vec![turn.mi_stage.as_str()])),
            Arc::new(StringArray::from(vec![turn.talk_type.as_str()])),
            Arc::new(StringArray::from(vec![turn.created_at.as_str()])),
            Arc::new(make_embedding_array(embedding)),
        ],
    )
    .context("Failed to create significant_turn RecordBatch")?;

    let table = conn.open_table("significant_turns").execute().await?;
    table
        .add(RecordBatchIterator::new(vec![Ok(batch)], schema))
        .execute()
        .await
        .context("Failed to insert significant turn")?;
    Ok(())
}

/// Adds an MI knowledge entry with its embedding.
pub async fn add_mi_knowledge(
    conn: &Connection,
    knowledge: &MiKnowledge,
    embedding: &[f64],
) -> Result<()> {
    let schema = mi_knowledge_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![knowledge.id.as_str()])),
            Arc::new(StringArray::from(vec![knowledge.category.as_str()])),
            Arc::new(StringArray::from(vec![knowledge.subcategory.as_str()])),
            Arc::new(StringArray::from(vec![knowledge.content.as_str()])),
            Arc::new(StringArray::from(vec![knowledge.mi_stages.as_str()])),
            Arc::new(StringArray::from(vec![knowledge.source.as_str()])),
            Arc::new(make_embedding_array(embedding)),
        ],
    )
    .context("Failed to create mi_knowledge RecordBatch")?;

    let table = conn.open_table("mi_knowledge").execute().await?;
    table
        .add(RecordBatchIterator::new(vec![Ok(batch)], schema))
        .execute()
        .await
        .context("Failed to insert MI knowledge")?;
    Ok(())
}

// ─── Search helpers (rig VectorStoreIndex) ──────────────────────────────────

/// Creates a `LanceDbVectorIndex` for the given table.
pub async fn vector_index(
    conn: &Connection,
    table_name: &str,
    model: EmbeddingModel,
) -> Result<LanceDbVectorIndex<EmbeddingModel>> {
    let table = conn
        .open_table(table_name)
        .execute()
        .await
        .with_context(|| format!("Failed to open table {table_name}"))?;

    LanceDbVectorIndex::new(
        table,
        model,
        "id",
        SearchParams::default()
            .distance_type(lancedb::DistanceType::Cosine)
            .column("vector"),
    )
    .await
    .map_err(|e| anyhow::anyhow!("Failed to create vector index for {table_name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_tables() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let tables = conn.table_names().execute().await.unwrap();
        assert!(tables.contains(&"user_knowledge".to_string()));
        assert!(tables.contains(&"session_summaries".to_string()));
        assert!(tables.contains(&"session_checkpoints".to_string()));
        assert!(tables.contains(&"significant_turns".to_string()));
        assert!(tables.contains(&"mi_knowledge".to_string()));
    }

    #[tokio::test]
    async fn test_create_tables_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();
        ensure_tables(&conn).await.unwrap(); // Should not error
    }

    #[tokio::test]
    async fn test_insert_and_count_user_fact() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let fact = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "goal".to_string(),
            content: "wants to reduce drinking".to_string(),
            source_session: "session_1".to_string(),
            last_confirmed: "session_1".to_string(),
            created_at: "2026-03-22T00:00:00Z".to_string(),
            updated_at: "2026-03-22T00:00:00Z".to_string(),
        };

        // Fake embedding (correct dimension)
        let embedding = vec![0.0f64; EMBEDDING_DIM];
        add_user_fact(&conn, &fact, &embedding).await.unwrap();

        let table = conn.open_table("user_knowledge").execute().await.unwrap();
        let count = table.count_rows(None).await.unwrap();
        assert_eq!(count, 1);
    }

    /// Full round-trip: embed with fastembed, store in LanceDB, retrieve via rig VectorStoreIndex.
    /// Downloads BGE-Small-EN model on first run (~130MB, cached).
    #[tokio::test]
    async fn test_embed_store_retrieve_round_trip() {
        use crate::memory::embeddings::init_embedding_model;
        use rig::embeddings::EmbeddingModel as _;
        use rig::vector_store::VectorStoreIndex;

        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();

        // Embed and store 3 distinct facts
        let facts = vec![
            ("goal", "wants to reduce drinking to weekends only"),
            ("relationship", "close with sister Val who is supportive"),
            ("trigger", "work stress on Monday mornings causes anxiety"),
        ];

        for (fact_type, content) in &facts {
            let embedding = model.embed_text(content).await.unwrap();
            let fact = UserFact {
                id: uuid::Uuid::new_v4().to_string(),
                fact_type: fact_type.to_string(),
                content: content.to_string(),
                source_session: "session_test".to_string(),
                last_confirmed: "session_test".to_string(),
                created_at: "2026-03-22T00:00:00Z".to_string(),
                updated_at: "2026-03-22T00:00:00Z".to_string(),
            };
            add_user_fact(&conn, &fact, &embedding.vec).await.unwrap();
        }

        // Verify row count
        let table = conn.open_table("user_knowledge").execute().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 3);

        // Semantic search: query about alcohol should match the drinking fact
        let index = vector_index(&conn, "user_knowledge", model).await.unwrap();
        let request = rig::vector_store::request::VectorSearchRequest::<rig_lancedb::LanceDBFilter>::builder()
            .query("how much alcohol do they drink")
            .samples(2)
            .build()
            .unwrap();
        let results: Vec<(f64, String, UserFact)> = index.top_n(request).await.unwrap();

        assert!(!results.is_empty(), "Should have search results");
        let top_result = &results[0].2;
        assert_eq!(top_result.fact_type, "goal");
        assert!(
            top_result.content.contains("drinking"),
            "Top result should be about drinking, got: {}",
            top_result.content
        );
    }

    #[tokio::test]
    async fn test_upsert_insert_when_empty() {
        use crate::memory::embeddings::init_embedding_model;
        use rig::embeddings::EmbeddingModel as _;

        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();
        let embedding = model.embed_text("wants to quit drinking").await.unwrap();
        let fact = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "goal".to_string(),
            content: "wants to quit drinking".to_string(),
            source_session: "session_1".to_string(),
            last_confirmed: "session_1".to_string(),
            created_at: "2026-03-23T00:00:00Z".to_string(),
            updated_at: "2026-03-23T00:00:00Z".to_string(),
        };

        let result = upsert_user_fact(&conn, &model, &fact, &embedding.vec).await.unwrap();
        assert_eq!(result, UpsertResult::Inserted);

        let table = conn.open_table("user_knowledge").execute().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_upsert_dedup_similar_content() {
        use crate::memory::embeddings::init_embedding_model;
        use rig::embeddings::EmbeddingModel as _;

        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();

        // Insert first fact
        let emb1 = model.embed_text("wants to quit drinking alcohol").await.unwrap();
        let fact1 = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "goal".to_string(),
            content: "wants to quit drinking alcohol".to_string(),
            source_session: "session_1".to_string(),
            last_confirmed: "session_1".to_string(),
            created_at: "2026-03-23T00:00:00Z".to_string(),
            updated_at: "2026-03-23T00:00:00Z".to_string(),
        };
        let r1 = upsert_user_fact(&conn, &model, &fact1, &emb1.vec).await.unwrap();
        assert_eq!(r1, UpsertResult::Inserted);

        // Upsert similar fact — should dedup
        let emb2 = model.embed_text("wants to stop drinking alcohol").await.unwrap();
        let fact2 = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "goal".to_string(),
            content: "wants to stop drinking alcohol".to_string(),
            source_session: "session_2".to_string(),
            last_confirmed: "session_2".to_string(),
            created_at: "2026-03-23T01:00:00Z".to_string(),
            updated_at: "2026-03-23T01:00:00Z".to_string(),
        };
        let r2 = upsert_user_fact(&conn, &model, &fact2, &emb2.vec).await.unwrap();
        assert!(matches!(r2, UpsertResult::Updated(_)), "Expected Update, got {r2:?}");

        // Still only 1 row
        let table = conn.open_table("user_knowledge").execute().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_upsert_no_dedup_different_fact_type() {
        use crate::memory::embeddings::init_embedding_model;
        use rig::embeddings::EmbeddingModel as _;

        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();

        // Insert as goal
        let emb1 = model.embed_text("drinking is a big concern").await.unwrap();
        let fact1 = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "goal".to_string(),
            content: "drinking is a big concern".to_string(),
            source_session: "session_1".to_string(),
            last_confirmed: "session_1".to_string(),
            created_at: "2026-03-23T00:00:00Z".to_string(),
            updated_at: "2026-03-23T00:00:00Z".to_string(),
        };
        upsert_user_fact(&conn, &model, &fact1, &emb1.vec).await.unwrap();

        // Upsert similar content but different fact_type — should NOT dedup
        let emb2 = model.embed_text("drinking causes problems at work").await.unwrap();
        let fact2 = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "trigger".to_string(),
            content: "drinking causes problems at work".to_string(),
            source_session: "session_2".to_string(),
            last_confirmed: "session_2".to_string(),
            created_at: "2026-03-23T01:00:00Z".to_string(),
            updated_at: "2026-03-23T01:00:00Z".to_string(),
        };
        let r2 = upsert_user_fact(&conn, &model, &fact2, &emb2.vec).await.unwrap();
        assert_eq!(r2, UpsertResult::Inserted);

        let table = conn.open_table("user_knowledge").execute().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_upsert_no_dedup_dissimilar_content() {
        use crate::memory::embeddings::init_embedding_model;
        use rig::embeddings::EmbeddingModel as _;

        let dir = tempfile::tempdir().unwrap();
        let conn = open_vector_db(dir.path().to_str().unwrap()).await.unwrap();
        ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();

        let emb1 = model.embed_text("loves hiking in the mountains").await.unwrap();
        let fact1 = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "interest".to_string(),
            content: "loves hiking in the mountains".to_string(),
            source_session: "session_1".to_string(),
            last_confirmed: "session_1".to_string(),
            created_at: "2026-03-23T00:00:00Z".to_string(),
            updated_at: "2026-03-23T00:00:00Z".to_string(),
        };
        upsert_user_fact(&conn, &model, &fact1, &emb1.vec).await.unwrap();

        let emb2 = model.embed_text("hates Monday morning meetings").await.unwrap();
        let fact2 = UserFact {
            id: uuid::Uuid::new_v4().to_string(),
            fact_type: "interest".to_string(),
            content: "hates Monday morning meetings".to_string(),
            source_session: "session_2".to_string(),
            last_confirmed: "session_2".to_string(),
            created_at: "2026-03-23T01:00:00Z".to_string(),
            updated_at: "2026-03-23T01:00:00Z".to_string(),
        };
        let r2 = upsert_user_fact(&conn, &model, &fact2, &emb2.vec).await.unwrap();
        assert_eq!(r2, UpsertResult::Inserted);

        let table = conn.open_table("user_knowledge").execute().await.unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 2);
    }
}
