use anyhow::Result;
use rig::embeddings::{Embed, EmbedError, EmbeddingModel, EmbeddingsBuilder, TextEmbedder};
use rig::vector_store::in_memory_store::{InMemoryVectorIndex, InMemoryVectorStore};
use rig::vector_store::{VectorSearchRequest, VectorStoreIndex};
use serde::{Deserialize, Serialize};

use crate::eval::catalog::ModesCatalog;

/// Default confidence threshold below which classification falls back to "engagement".
const DEFAULT_THRESHOLD: f64 = 0.75;

/// Result of classifying a user message into a conversation mode.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Mode ID matching a `ModeDefinition.id` from `modes.toml`.
    pub mode_id: String,
    /// Cosine similarity score from the semantic search (0.0–1.0).
    pub confidence: f64,
}

/// A single utterance mapped to a conversation mode, stored in the vector index.
///
/// Each mode has 10–15 representative utterances. When the router classifies
/// a user message, it finds the nearest utterance and returns its mode.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct RouteUtterance {
    /// The mode this utterance belongs to (e.g., "crisis", "resistance").
    pub mode_id: String,
    /// The representative utterance text (used for embedding).
    pub text: String,
}

impl Embed for RouteUtterance {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

/// Semantic router that classifies user messages into conversation modes.
///
/// Built from `modes.toml` utterances embedded via fastembed into an in-memory
/// vector store. Classification is a single nearest-neighbor lookup: embed the
/// user message, find the closest utterance, return its mode if above threshold.
pub struct ModeRouter<M: EmbeddingModel> {
    index: InMemoryVectorIndex<M, RouteUtterance>,
    threshold: f64,
    fallback_mode: String,
}

impl<M: EmbeddingModel + Clone + Sync> ModeRouter<M> {
    /// Builds a router from a modes catalog by embedding all utterances.
    ///
    /// Each mode's utterances are embedded and stored in an `InMemoryVectorStore`.
    /// Classification queries this store for the nearest utterance.
    #[tracing::instrument(level = "info", skip_all)]
    pub async fn from_catalog(
        catalog: &ModesCatalog,
        embedding_model: M,
    ) -> Result<Self> {
        let utterances: Vec<RouteUtterance> = catalog
            .modes
            .iter()
            .flat_map(|mode| {
                mode.utterances.iter().map(move |text| RouteUtterance {
                    mode_id: mode.id.clone(),
                    text: text.clone(),
                })
            })
            .collect();

        let total = utterances.len();
        tracing::info!(utterances = total, modes = catalog.modes.len(), "Embedding route utterances");

        let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
            .documents(utterances)
            .map_err(|e| anyhow::anyhow!("Failed to prepare utterance embeddings: {e}"))?
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to embed route utterances: {e}"))?;

        let store = InMemoryVectorStore::from_documents(embeddings);
        let index = store.index(embedding_model);

        tracing::info!(total, "Mode router ready");

        Ok(Self {
            index,
            threshold: DEFAULT_THRESHOLD,
            fallback_mode: "engagement".to_string(),
        })
    }

    /// Sets the confidence threshold for classification.
    ///
    /// Messages with similarity below this threshold fall back to the default mode.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Classifies a user message into a conversation mode.
    ///
    /// Returns the closest mode and confidence score. Falls back to "engagement"
    /// when no utterance exceeds the confidence threshold.
    pub async fn classify(&self, input: &str) -> Result<RouteResult> {
        let request = VectorSearchRequest::builder()
            .query(input)
            .samples(1)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build search request: {e}"))?;

        let results: Vec<(f64, String, RouteUtterance)> = self
            .index
            .top_n::<RouteUtterance>(request)
            .await
            .map_err(|e| anyhow::anyhow!("Route classification failed: {e}"))?;

        match results.first() {
            Some((score, _id, utterance)) if *score >= self.threshold => {
                tracing::debug!(
                    mode = %utterance.mode_id,
                    confidence = %score,
                    matched = %utterance.text,
                    "Route classified"
                );
                Ok(RouteResult {
                    mode_id: utterance.mode_id.clone(),
                    confidence: *score,
                })
            }
            Some((score, _id, utterance)) => {
                tracing::debug!(
                    best_mode = %utterance.mode_id,
                    confidence = %score,
                    threshold = %self.threshold,
                    fallback = %self.fallback_mode,
                    "Below threshold, using fallback"
                );
                Ok(RouteResult {
                    mode_id: self.fallback_mode.clone(),
                    confidence: *score,
                })
            }
            None => {
                tracing::warn!("No route results returned, using fallback");
                Ok(RouteResult {
                    mode_id: self.fallback_mode.clone(),
                    confidence: 0.0,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_utterance_embed() {
        let utterance = RouteUtterance {
            mode_id: "crisis".to_string(),
            text: "I don't want to be here anymore".to_string(),
        };
        let mut embedder = TextEmbedder::default();
        // Embed should succeed without error
        utterance.embed(&mut embedder).unwrap();
    }

    #[test]
    fn test_route_utterance_serialization() {
        let utterance = RouteUtterance {
            mode_id: "resistance".to_string(),
            text: "I don't need help".to_string(),
        };
        let json = serde_json::to_string(&utterance).unwrap();
        let deserialized: RouteUtterance = serde_json::from_str(&json).unwrap();
        assert_eq!(utterance, deserialized);
    }
}
