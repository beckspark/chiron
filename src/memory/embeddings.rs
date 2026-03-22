use rig_fastembed::{Client as FastembedClient, EmbeddingModel, FastembedModel};

/// Initializes the fastembed embedding model (BAAI/bge-small-en-v1.5).
///
/// Downloads the model on first run (~130MB, cached in `~/.cache/fastembed/`).
/// Returns a rig-compatible `EmbeddingModel` for use with `LanceDbVectorIndex`.
pub fn init_embedding_model() -> EmbeddingModel {
    let client = FastembedClient::new();
    client.embedding_model(&FastembedModel::BGESmallENV15)
}
