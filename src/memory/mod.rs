pub mod case_notes;
pub mod store;

use anyhow::{Context, Result};
use rig::embeddings::EmbeddingModel;
use rig_sqlite::SqliteVectorStore;
use tokio_rusqlite::Connection;

use self::store::MiKnowledge;

/// Registers the sqlite-vec extension globally. Must be called before opening any connections.
pub fn init_sqlite_vec() {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
}

/// Opens the database and creates an MI knowledge vector store plus a chat connection.
///
/// Returns a vector store for RAG and a separate connection for chat turn
/// and case note persistence. Both point at the same database file;
/// SQLite WAL handles concurrent reads.
#[tracing::instrument(level = "info", skip(embedding_model))]
pub async fn open_memory<E>(
    db_path: &str,
    embedding_model: &E,
) -> Result<(SqliteVectorStore<E, MiKnowledge>, Connection)>
where
    E: EmbeddingModel + Clone + 'static,
{
    // Connection for MI knowledge vector store
    let knowledge_conn = Connection::open(db_path)
        .await
        .context("Failed to open knowledge DB connection")?;

    // Separate connection for chat turn and case note operations
    let chat_conn = Connection::open(db_path)
        .await
        .context("Failed to open chat DB connection")?;

    // Enable WAL mode for concurrent reads across connections
    chat_conn
        .call(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL")?;
            Ok(())
        })
        .await
        .context("Failed to enable WAL mode")?;

    // Create chat_turns table
    chat_conn
        .call(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_chat_turns_session
                    ON chat_turns(session_id, created_at);",
            )?;
            Ok(())
        })
        .await
        .context("Failed to create chat_turns table")?;

    // Create case_notes table
    case_notes::create_case_notes_table(&chat_conn).await?;

    // Create MI knowledge vector store
    let knowledge_store = SqliteVectorStore::new(knowledge_conn, embedding_model)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create knowledge vector store: {e}"))?;

    tracing::info!("Memory initialized (vector store + case notes)");
    Ok((knowledge_store, chat_conn))
}

/// Saves a single chat turn to the database.
pub async fn save_chat_turn(
    conn: &Connection,
    session_id: &str,
    role: &str,
    content: &str,
) -> Result<()> {
    let session_id = session_id.to_string();
    let role = role.to_string();
    let content = content.to_string();

    conn.call(move |conn| {
        conn.execute(
            "INSERT INTO chat_turns (session_id, role, content) VALUES (?1, ?2, ?3)",
            rusqlite::params![session_id, role, content],
        )?;
        Ok(())
    })
    .await
    .context("Failed to save chat turn")?;

    Ok(())
}
