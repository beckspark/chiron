use anyhow::{Context, Result};
use rig::embeddings::EmbeddingModel as _;
use rig_fastembed::EmbeddingModel;

use super::vectors::{self, MiKnowledge};

/// Parsed MI knowledge entry before embedding.
#[derive(Debug, Clone)]
pub struct ParsedKnowledge {
    pub category: String,
    pub subcategory: String,
    pub content: String,
    pub mi_stages: String,
}

/// Parses a markdown file into MI knowledge entries.
///
/// Expected format:
/// ```markdown
/// # Category Name
/// Stages: engage, focus
///
/// ## Subcategory Name
/// - Knowledge entry one
/// - Knowledge entry two
///
/// ## Another Subcategory
/// - Another entry
///
/// # Another Category
/// Stages: evoke, plan
/// ...
/// ```
///
/// Top-level `#` headers define categories. An optional `Stages:` line after
/// the header defines which MI stages these entries apply to. `##` headers
/// define subcategories. Bullet points (`-`) are individual knowledge entries.
pub fn parse_markdown(content: &str) -> Vec<ParsedKnowledge> {
    let mut entries = Vec::new();
    let mut category = String::new();
    let mut subcategory = String::new();
    let mut mi_stages = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if let Some(heading) = trimmed.strip_prefix("# ") {
            if !heading.starts_with('#') {
                category = heading.trim().to_string();
                subcategory.clear();
                mi_stages.clear();
            }
        }

        if let Some(heading) = trimmed.strip_prefix("## ") {
            subcategory = heading.trim().to_string();
        }

        if let Some(rest) = trimmed.strip_prefix("Stages:") {
            mi_stages = rest.trim().to_string();
        }

        if let Some(rest) = trimmed.strip_prefix("- ") {
            let text = rest.trim();
            if !text.is_empty() && !category.is_empty() {
                entries.push(ParsedKnowledge {
                    category: category.clone(),
                    subcategory: if subcategory.is_empty() {
                        "general".to_string()
                    } else {
                        subcategory.clone()
                    },
                    content: text.to_string(),
                    mi_stages: if mi_stages.is_empty() {
                        "all".to_string()
                    } else {
                        mi_stages.clone()
                    },
                });
            }
        }
    }

    entries
}

/// Seeds the mi_knowledge table from parsed entries.
///
/// Embeds each entry's content and inserts it into LanceDB.
/// Returns the number of entries successfully seeded.
pub async fn seed_knowledge(
    conn: &lancedb::Connection,
    model: &EmbeddingModel,
    entries: &[ParsedKnowledge],
    source: &str,
) -> Result<usize> {
    let mut count = 0;

    for entry in entries {
        let embedding = model
            .embed_text(&entry.content)
            .await
            .with_context(|| format!("Failed to embed: {}", &entry.content[..40.min(entry.content.len())]))?;

        let knowledge = MiKnowledge {
            id: uuid::Uuid::new_v4().to_string(),
            category: entry.category.clone(),
            subcategory: entry.subcategory.clone(),
            content: entry.content.clone(),
            mi_stages: entry.mi_stages.clone(),
            source: source.to_string(),
        };

        vectors::add_mi_knowledge(conn, &knowledge, &embedding.vec)
            .await
            .with_context(|| format!("Failed to insert knowledge entry: {}", &entry.content[..40.min(entry.content.len())]))?;

        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MD: &str = r#"# OARS
Stages: engage, focus, evoke, plan

## Open Questions
- Ask questions that cannot be answered with yes or no
- Begin with "what", "how", "tell me about" to invite exploration
- Avoid "why" questions that may trigger defensiveness

## Affirmations
- Acknowledge specific strengths and efforts
- Focus on behaviors and character, not just outcomes

# Change Talk
Stages: evoke, plan

## DARN Questions
- Desire: "What would you like to see different?"
- Ability: "What makes you think you could do this?"
- Reasons: "What are the best reasons for making this change?"
- Need: "How important is this change to you?"
"#;

    #[test]
    fn test_parse_markdown_knowledge() {
        let entries = parse_markdown(SAMPLE_MD);

        assert_eq!(entries.len(), 9);

        // First entry
        assert_eq!(entries[0].category, "OARS");
        assert_eq!(entries[0].subcategory, "Open Questions");
        assert!(entries[0].content.contains("cannot be answered with yes or no"));
        assert_eq!(entries[0].mi_stages, "engage, focus, evoke, plan");

        // Affirmations subcategory
        let affirm = entries.iter().find(|e| e.subcategory == "Affirmations").unwrap();
        assert_eq!(affirm.category, "OARS");

        // Change Talk category with different stages
        let darn = entries.iter().find(|e| e.subcategory == "DARN Questions").unwrap();
        assert_eq!(darn.category, "Change Talk");
        assert_eq!(darn.mi_stages, "evoke, plan");
    }

    #[test]
    fn test_parse_empty_markdown() {
        assert!(parse_markdown("").is_empty());
        assert!(parse_markdown("Just some text without headers").is_empty());
    }

    #[test]
    fn test_parse_no_stages_defaults_to_all() {
        let md = "# Basics\n## Core\n- Listen actively\n";
        let entries = parse_markdown(md);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].mi_stages, "all");
    }

    #[test]
    fn test_mi_knowledge_file_parses() {
        let content = include_str!("../../data/mi_knowledge.md");
        let entries = parse_markdown(content);

        assert!(
            entries.len() >= 60,
            "Expected at least 60 entries, got {}",
            entries.len()
        );

        for (i, entry) in entries.iter().enumerate() {
            assert!(!entry.content.is_empty(), "Entry {i} has empty content");
            assert!(!entry.category.is_empty(), "Entry {i} has empty category");
        }

        // Verify key categories are present
        let categories: std::collections::HashSet<&str> =
            entries.iter().map(|e| e.category.as_str()).collect();
        assert!(categories.contains("MI Spirit"), "Missing MI Spirit category");
        assert!(categories.contains("OARS"), "Missing OARS category");
        assert!(categories.contains("Change Talk"), "Missing Change Talk category");
        assert!(categories.contains("Discord"), "Missing Discord category");
        assert!(categories.contains("Peer Boundaries"), "Missing Peer Boundaries category");
        assert!(categories.contains("Common Mistakes"), "Missing Common Mistakes category");
    }

    #[tokio::test]
    async fn test_seed_and_retrieve_round_trip() {
        use crate::memory::embeddings::init_embedding_model;
        use crate::memory::retrieval;

        let dir = tempfile::tempdir().unwrap();
        let conn = vectors::open_vector_db(dir.path().to_str().unwrap())
            .await
            .unwrap();
        vectors::ensure_tables(&conn).await.unwrap();

        let model = init_embedding_model();
        let entries = parse_markdown(SAMPLE_MD);
        let count = seed_knowledge(&conn, &model, &entries, "test").await.unwrap();
        assert_eq!(count, 9);

        // Retrieve with a relevant query
        let ctx = retrieval::retrieve_context(&conn, &model, "how to ask open questions", None, 3).await;
        assert!(!ctx.mi_knowledge.is_empty(), "should retrieve seeded knowledge");
        assert!(
            ctx.mi_knowledge.iter().any(|k| k.content.contains("cannot be answered")),
            "should find the open questions entry"
        );
    }
}
