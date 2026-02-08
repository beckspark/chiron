use rig::embeddings::{Embed, EmbedError, TextEmbedder};
use rig_sqlite::{Column, ColumnValue, SqliteVectorStoreTable};
use serde::{Deserialize, Serialize};

/// A chunk of MI knowledge stored in the vector database.
///
/// Each chunk represents a self-contained piece of motivational interviewing
/// knowledge (e.g., one OARS skill, one MI spirit component) with metadata
/// for filtering and the embeddable content text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiKnowledge {
    pub id: String,
    pub category: String,
    pub topic: String,
    pub content: String,
}

impl Embed for MiKnowledge {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.content.clone());
        Ok(())
    }
}

impl SqliteVectorStoreTable for MiKnowledge {
    fn name() -> &'static str {
        "mi_knowledge"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("category", "TEXT").indexed(),
            Column::new("topic", "TEXT"),
            Column::new("content", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("category", Box::new(self.category.clone())),
            ("topic", Box::new(self.topic.clone())),
            ("content", Box::new(self.content.clone())),
        ]
    }
}
