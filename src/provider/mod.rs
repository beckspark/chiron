pub mod chat_template;
pub mod completion;
pub mod config;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3;
use candle_transformers::models::smol::quantized_smollm3;
use tokenizers::Tokenizer;

use rig::completion::CompletionModel;

use self::chat_template::ChatTemplate;
use self::completion::CandleCompletionModel;
use self::config::GenerationConfig;

/// SmolLM3 end-of-sequence token IDs: `<|im_end|>` and `<|end_of_text|>`.
pub const SMOLLM3_EOS_TOKEN_IDS: &[u32] = &[128012, 128001];

/// Qwen3 end-of-sequence token IDs: `<|im_end|>` and `<|endoftext|>`.
pub const QWEN3_EOS_TOKEN_IDS: &[u32] = &[151645, 151643];

/// Supported model architectures for GGUF loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    SmolLM3,
    Qwen3,
}

impl ModelArch {
    /// Returns the default EOS token IDs for this architecture.
    pub fn default_eos_token_ids(&self) -> &'static [u32] {
        match self {
            ModelArch::SmolLM3 => SMOLLM3_EOS_TOKEN_IDS,
            ModelArch::Qwen3 => QWEN3_EOS_TOKEN_IDS,
        }
    }

    /// Attempts to detect architecture from GGUF metadata keys.
    ///
    /// Reads the GGUF file header and checks for architecture-specific metadata
    /// prefixes (`qwen3.*`, `smollm3.*`, etc.).
    pub fn detect_from_gguf(path: &Path) -> Result<Self> {
        use candle_core::quantized::gguf_file;
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF metadata: {e}"))?;

        let general_arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.as_str());

        match general_arch {
            Some("qwen3") => Ok(ModelArch::Qwen3),
            Some("smollm3") => Ok(ModelArch::SmolLM3),
            Some(arch) => {
                // Try matching by metadata key prefixes
                let has_qwen3 = content.metadata.keys().any(|k| k.starts_with("qwen3."));
                let has_smollm3 = content.metadata.keys().any(|k| k.starts_with("smollm3."));
                if has_qwen3 {
                    tracing::warn!(
                        detected = "qwen3",
                        general_architecture = arch,
                        "Architecture mismatch in metadata, using detected qwen3"
                    );
                    Ok(ModelArch::Qwen3)
                } else if has_smollm3 {
                    Ok(ModelArch::SmolLM3)
                } else {
                    anyhow::bail!(
                        "Unknown model architecture '{}' in GGUF metadata. \
                        Supported: qwen3, smollm3. Use --model-arch to override.",
                        arch
                    )
                }
            }
            None => {
                // No general.architecture key — try key prefix heuristic
                let has_qwen3 = content.metadata.keys().any(|k| k.starts_with("qwen3."));
                if has_qwen3 {
                    Ok(ModelArch::Qwen3)
                } else {
                    // Default to SmolLM3 for backward compat
                    tracing::warn!("No architecture detected in GGUF metadata, defaulting to SmolLM3");
                    Ok(ModelArch::SmolLM3)
                }
            }
        }
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArch::SmolLM3 => write!(f, "smollm3"),
            ModelArch::Qwen3 => write!(f, "qwen3"),
        }
    }
}

impl std::str::FromStr for ModelArch {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "smollm3" | "smol-lm3" | "smol_lm3" => Ok(ModelArch::SmolLM3),
            "qwen3" | "qwen-3" | "qwen_3" | "qwen3-4b" => Ok(ModelArch::Qwen3),
            other => anyhow::bail!("Unknown model architecture: '{}'. Supported: smollm3, qwen3", other),
        }
    }
}

/// Quantized model weights wrapper supporting multiple architectures.
///
/// Handles KV cache clearing on prefill and normalizes logit output shapes
/// across architectures (SmolLM3 outputs `[batch, 1, vocab_size]`, Qwen3
/// outputs `[batch, vocab_size]`).
pub struct ModelWeights {
    inner: ModelWeightsInner,
}

enum ModelWeightsInner {
    SmolLM3(quantized_smollm3::QuantizedModelForCausalLM),
    Qwen3(quantized_qwen3::ModelWeights),
}

impl ModelWeights {
    /// Runs a forward pass through the model, returning logits.
    ///
    /// Automatically clears the KV cache when `index_pos == 0` (prefill),
    /// since all agents share one model and stale cache causes shape mismatches.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        if index_pos == 0 {
            self.clear_kv_cache();
        }

        let logits = match &mut self.inner {
            ModelWeightsInner::SmolLM3(m) => m.forward(x, index_pos)?,
            ModelWeightsInner::Qwen3(m) => m.forward(x, index_pos)?,
        };

        // Normalize: SmolLM3 outputs [batch, 1, vocab_size], squeeze to [batch, vocab_size]
        if logits.rank() == 3 {
            logits.squeeze(1)
        } else {
            Ok(logits)
        }
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.inner {
            ModelWeightsInner::SmolLM3(m) => m.clear_kv_cache(),
            ModelWeightsInner::Qwen3(m) => m.clear_kv_cache(),
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
    /// Pre-computed logit mask for token suppression.
    /// Shape: `[vocab_size]` with `0.0` at allowed positions and `-inf` at suppressed
    /// positions (CJK tokens for multilingual models). Added to logits before sampling.
    pub suppress_mask: Option<Tensor>,
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

    /// Loads a GGUF model from disk and registers it under `model_id`.
    ///
    /// `model_path` points to the GGUF weights file, `tokenizer_path` to a
    /// HuggingFace `tokenizer.json`.
    ///
    /// `arch` selects the model architecture. When `None`, auto-detects from
    /// GGUF metadata.
    ///
    /// `chat_template_path` optionally points to a `tokenizer_config.json` or
    /// `.jinja` file containing the model's native chat template. When `None`,
    /// falls back to the built-in ChatML-with-thinking preset.
    pub fn load_model(
        &mut self,
        model_id: &str,
        model_path: &Path,
        tokenizer_path: &Path,
        arch: Option<ModelArch>,
        config: GenerationConfig,
        chat_template_path: Option<&Path>,
    ) -> Result<()> {
        let arch = match arch {
            Some(a) => a,
            None => ModelArch::detect_from_gguf(model_path)?,
        };

        tracing::info!(
            model_id,
            arch = %arch,
            model = %model_path.display(),
            tokenizer = %tokenizer_path.display(),
            chat_template = ?chat_template_path,
            "Loading model"
        );

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let weights = match arch {
            ModelArch::SmolLM3 => {
                let inner = quantized_smollm3::QuantizedModelForCausalLM::from_gguf(
                    model_path,
                    &self.device,
                )
                .context("Failed to load SmolLM3 weights")?;
                ModelWeights {
                    inner: ModelWeightsInner::SmolLM3(inner),
                }
            }
            ModelArch::Qwen3 => {
                let inner = load_qwen3_gguf(model_path, &self.device)?;
                ModelWeights {
                    inner: ModelWeightsInner::Qwen3(inner),
                }
            }
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

        // For multilingual models, pre-compute a logit mask to suppress CJK tokens
        let suppress_mask = if arch == ModelArch::Qwen3 {
            let ids = find_cjk_token_ids(&tokenizer);
            let vocab_size = tokenizer.get_vocab_size(true);
            let mask = build_suppress_mask(&ids, vocab_size, &self.device)?;
            tracing::info!(
                model_id,
                suppressed_tokens = ids.len(),
                vocab_size,
                "CJK token suppression enabled"
            );
            Some(mask)
        } else {
            None
        };

        tracing::info!(
            model_id,
            arch = %arch,
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
                suppress_mask,
            },
        );
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Returns true if the character is in a CJK unicode range.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

/// Scans the tokenizer vocabulary for tokens that decode to CJK characters.
///
/// Byte-level BPE tokenizers (like Qwen3's) store tokens as byte-mapped characters,
/// not actual Unicode. This function decodes each token ID to get the real text,
/// then checks for CJK characters.
fn find_cjk_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let vocab_size = tokenizer.get_vocab_size(true);
    let mut ids = Vec::new();
    for id in 0..vocab_size as u32 {
        if let Ok(decoded) = tokenizer.decode(&[id], true) {
            if decoded.chars().any(is_cjk) {
                ids.push(id);
            }
        }
    }
    ids
}

/// Builds a logit mask tensor for token suppression.
///
/// Returns a `[vocab_size]` tensor with `0.0` at allowed positions and `f32::NEG_INFINITY`
/// at suppressed positions. This tensor can be added to logits before sampling.
fn build_suppress_mask(suppress_ids: &[u32], vocab_size: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0.0f32; vocab_size];
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    Tensor::new(mask, device).context("Failed to create suppress mask tensor")
}

/// Loads a Qwen3 GGUF model from disk.
///
/// Opens the GGUF file, reads metadata + tensor info, then constructs the
/// quantized model weights.
fn load_qwen3_gguf(
    path: &Path,
    device: &Device,
) -> Result<quantized_qwen3::ModelWeights> {
    use candle_core::quantized::gguf_file;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {e}"))?;
    quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen3 weights: {e}"))
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
/// Models with thinking support produce internal reasoning wrapped in think tags.
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
