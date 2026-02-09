use anyhow::{Context, Result};
use tokio_rusqlite::Connection;

/// Creates the case_notes table and index if they don't exist.
///
/// Case notes track the user's MI journey across sessions: stage progression,
/// change talk observations, strategy assessments, and key themes.
pub async fn create_case_notes_table(conn: &Connection) -> Result<()> {
    conn.call(|conn| {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS case_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                mi_stage TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_case_notes_latest
                ON case_notes(created_at DESC);",
        )?;
        Ok(())
    })
    .await
    .context("Failed to create case_notes table")?;

    Ok(())
}

/// Saves a new case note to the database.
pub async fn save_case_note(
    conn: &Connection,
    session_id: &str,
    turn_number: i32,
    mi_stage: Option<&str>,
    content: &str,
) -> Result<()> {
    let session_id = session_id.to_string();
    let mi_stage = mi_stage.map(|s| s.to_string());
    let content = content.to_string();

    conn.call(move |conn| {
        conn.execute(
            "INSERT INTO case_notes (session_id, turn_number, mi_stage, content) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![session_id, turn_number, mi_stage, content],
        )?;
        Ok(())
    })
    .await
    .context("Failed to save case note")?;

    Ok(())
}

/// Loads the most recent case note content from any session.
///
/// Returns `None` if no case notes exist (first turn of the first session).
/// Cross-session persistence: the latest note carries forward to new sessions.
pub async fn get_latest_case_note(conn: &Connection) -> Result<Option<String>> {
    let result = conn
        .call(|conn| {
            let mut stmt = conn.prepare(
                "SELECT content FROM case_notes ORDER BY id DESC LIMIT 1",
            )?;
            let content = stmt
                .query_row([], |row| row.get::<_, String>(0))
                .optional()?;
            Ok(content)
        })
        .await
        .context("Failed to load latest case note")?;

    Ok(result)
}

// Re-export optional for the query
use rusqlite::OptionalExtension;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_table_and_insert() {
        crate::memory::init_sqlite_vec();
        let conn = Connection::open(":memory:").await.unwrap();
        create_case_notes_table(&conn).await.unwrap();

        // No notes initially
        let latest = get_latest_case_note(&conn).await.unwrap();
        assert!(latest.is_none());

        // Save a note
        save_case_note(&conn, "session_1", 1, Some("engage"), "Initial contact.")
            .await
            .unwrap();

        let latest = get_latest_case_note(&conn).await.unwrap();
        assert_eq!(latest.unwrap(), "Initial contact.");
    }

    #[tokio::test]
    async fn test_latest_note_returns_most_recent() {
        crate::memory::init_sqlite_vec();
        let conn = Connection::open(":memory:").await.unwrap();
        create_case_notes_table(&conn).await.unwrap();

        save_case_note(&conn, "session_1", 1, Some("engage"), "First note.")
            .await
            .unwrap();
        save_case_note(&conn, "session_1", 2, Some("focus"), "Second note.")
            .await
            .unwrap();
        save_case_note(&conn, "session_2", 1, Some("evoke"), "Cross-session note.")
            .await
            .unwrap();

        let latest = get_latest_case_note(&conn).await.unwrap();
        assert_eq!(latest.unwrap(), "Cross-session note.");
    }
}
