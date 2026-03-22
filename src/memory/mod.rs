pub mod case_notes;
pub mod embeddings;
pub mod retrieval;
pub mod vectors;

use anyhow::{Context, Result};
use tokio_rusqlite::Connection;

/// Opens the database and creates tables for chat history and case notes.
#[tracing::instrument(level = "info")]
pub async fn open_memory(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)
        .await
        .context("Failed to open database")?;

    // Enable WAL mode for concurrent reads
    conn.call(|conn| {
        conn.execute_batch("PRAGMA journal_mode=WAL")?;
        Ok(())
    })
    .await
    .context("Failed to enable WAL mode")?;

    // Create chat_turns table
    conn.call(|conn| {
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
    case_notes::create_case_notes_table(&conn).await?;

    tracing::info!("Memory initialized (chat history + case notes)");
    Ok(conn)
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
