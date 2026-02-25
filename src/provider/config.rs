use serde::{Deserialize, Serialize};

/// Configuration for text generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Temperature for sampling (0.0 = greedy, higher = more random).
    pub temperature: f64,
    /// Top-p nucleus sampling threshold.
    pub top_p: f64,
    /// Top-k sampling (0 = disabled). Used by Qwen3 (recommended: 20).
    pub top_k: usize,
    /// Maximum tokens to generate per response.
    pub max_tokens: usize,
    /// Token IDs that signal end of generation.
    pub eos_token_ids: Vec<u32>,
    /// Random seed for reproducibility. `None` for random.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 20,
            max_tokens: 512,
            eos_token_ids: vec![],
            seed: Some(42),
        }
    }
}
