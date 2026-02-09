pub mod chat_template;
pub mod completion;
pub mod config;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::smol::quantized_smollm3;
use tokenizers::Tokenizer;

use rig::completion::CompletionModel;

use self::chat_template::ChatTemplate;
use self::completion::CandleCompletionModel;
use self::config::GenerationConfig;

/// SmolLM3 end-of-sequence token IDs: `<|im_end|>` and `<|end_of_text|>`.
pub const SMOLLM3_EOS_TOKEN_IDS: &[u32] = &[128012, 128001];

/// Quantized SmolLM3 model weights wrapper.
///
/// Handles KV cache clearing on prefill and squeezes the extra sequence
/// dimension from SmolLM3's `[batch, 1, vocab_size]` logit output.
pub struct ModelWeights {
    inner: quantized_smollm3::QuantizedModelForCausalLM,
}

impl ModelWeights {
    /// Runs a forward pass through the model, returning logits.
    ///
    /// Automatically clears the KV cache when `index_pos == 0` (prefill),
    /// since all agents share one model and stale cache causes shape mismatches.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        if index_pos == 0 {
            self.inner.clear_kv_cache();
        }
        let logits = self.inner.forward(x, index_pos)?;
        if logits.rank() == 3 {
            logits.squeeze(1)
        } else {
            Ok(logits)
        }
    }
}

/// A loaded GGUF model with its tokenizer, generation config, and chat template.
pub struct LoadedModel {
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    pub config: GenerationConfig,
    /// HF Jinja2 chat template for prompt formatting.
    pub template: ChatTemplate,
}

/// Registry holding all loaded models, keyed by model ID.
///
/// Thread-safe via `Arc<Mutex<_>>` since Candle inference is synchronous
/// and must be serialized per-model anyway.
pub struct ModelRegistry {
    pub(crate) models: HashMap<String, LoadedModel>,
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

    /// Loads a SmolLM3 GGUF model from disk and registers it under `model_id`.
    ///
    /// `model_path` points to the GGUF weights file, `tokenizer_path` to a
    /// HuggingFace `tokenizer.json`.
    ///
    /// `chat_template_path` optionally points to a `tokenizer_config.json` or
    /// `.jinja` file containing the model's native chat template. When `None`,
    /// falls back to the built-in ChatML-with-thinking preset.
    pub fn load_model(
        &mut self,
        model_id: &str,
        model_path: &Path,
        tokenizer_path: &Path,
        config: GenerationConfig,
        chat_template_path: Option<&Path>,
    ) -> Result<()> {
        tracing::info!(
            model_id,
            model = %model_path.display(),
            tokenizer = %tokenizer_path.display(),
            chat_template = ?chat_template_path,
            "Loading SmolLM3 model"
        );

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let weights = ModelWeights {
            inner: quantized_smollm3::QuantizedModelForCausalLM::from_gguf(
                model_path,
                &self.device,
            )
            .context("Failed to load SmolLM3 weights")?,
        };

        let template = match chat_template_path {
            Some(p) if p.extension().is_some_and(|e| e == "jinja") => {
                ChatTemplate::from_jinja_file(p)
                    .map_err(|e| anyhow::anyhow!("Failed to load .jinja chat template: {e}"))?
            }
            Some(p) => ChatTemplate::from_tokenizer_config(p).map_err(|e| {
                anyhow::anyhow!("Failed to load chat template from tokenizer_config.json: {e}")
            })?,
            None => ChatTemplate::chatml_with_thinking(),
        };

        tracing::info!(
            model_id,
            template_source = if chat_template_path.is_some() { "file" } else { "preset" },
            "Model loaded successfully"
        );
        self.models.insert(
            model_id.to_string(),
            LoadedModel {
                weights,
                tokenizer,
                config,
                template,
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

/// Strips `<think>...</think>` blocks from model output.
///
/// SmolLM3 produces internal reasoning wrapped in think tags.
/// This removes them from the final response, along with any leading whitespace
/// after the block.
pub fn strip_think_blocks(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            let block_end = end + "</think>".len();
            result = format!(
                "{}{}",
                &result[..start],
                result[block_end..].trim_start()
            );
        } else {
            // Unclosed think block — strip from <think> to end
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}
