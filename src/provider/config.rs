use serde::{Deserialize, Serialize};

/// Configuration for text generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Temperature for sampling (0.0 = greedy, higher = more random).
    pub temperature: f64,
    /// Top-p nucleus sampling threshold.
    pub top_p: f64,
    /// Maximum tokens to generate per response.
    pub max_tokens: usize,
    /// Token IDs that signal end of generation.
    ///
    /// For Llama 3.2 Instruct: `[128009]` (`<|eot_id|>`)
    /// Also include `128001` (`<|end_of_text|>`) as fallback.
    pub eos_token_ids: Vec<u32>,
    /// Random seed for reproducibility. `None` for random.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            // Llama 3.2 Instruct EOS tokens
            eos_token_ids: vec![128009, 128001],
            seed: Some(42),
        }
    }
}
