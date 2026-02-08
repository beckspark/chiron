pub mod completion;
pub mod config;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

use rig::completion::CompletionModel;

use self::completion::CandleCompletionModel;
use self::config::GenerationConfig;

/// A loaded GGUF model with its tokenizer and generation config.
pub struct LoadedModel {
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    pub config: GenerationConfig,
}

/// Registry holding all loaded models, keyed by model ID.
///
/// Thread-safe via `Arc<Mutex<_>>` since Candle inference is synchronous
/// and must be serialized per-model anyway.
pub struct ModelRegistry {
    models: HashMap<String, LoadedModel>,
    device: Device,
}

impl ModelRegistry {
    /// Creates a new empty registry, selecting CUDA device 0 if available.
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        tracing::info!("ModelRegistry using device: {:?}", device);
        Ok(Self {
            models: HashMap::new(),
            device,
        })
    }

    /// Loads a GGUF model from disk and registers it under `model_id`.
    ///
    /// `model_path` points to the GGUF weights file, `tokenizer_path` to a
    /// HuggingFace `tokenizer.json`.
    pub fn load_model(
        &mut self,
        model_id: &str,
        model_path: &Path,
        tokenizer_path: &Path,
        config: GenerationConfig,
    ) -> Result<()> {
        tracing::info!(model_id, model = %model_path.display(), tokenizer = %tokenizer_path.display(), "Loading model");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("Failed to open model file {}", model_path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to read GGUF content")?;
        let weights = ModelWeights::from_gguf(content, &mut file, &self.device)
            .context("Failed to load model weights")?;

        tracing::info!(model_id, "Model loaded successfully");
        self.models.insert(
            model_id.to_string(),
            LoadedModel {
                weights,
                tokenizer,
                config,
            },
        );
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// The Candle provider client, holding a shared reference to the model registry.
///
/// This is the `Client` type for `CandleCompletionModel`.
#[derive(Clone)]
pub struct CandleProvider {
    pub registry: Arc<Mutex<ModelRegistry>>,
}

impl CandleProvider {
    /// Creates a new provider wrapping an existing registry.
    pub fn new(registry: ModelRegistry) -> Self {
        Self {
            registry: Arc::new(Mutex::new(registry)),
        }
    }

    /// Creates a completion model handle for the given model ID.
    pub fn completion_model(&self, model_id: &str) -> CandleCompletionModel {
        CandleCompletionModel::make(self, model_id)
    }
}
