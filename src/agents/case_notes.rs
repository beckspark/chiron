use rig::agent::{Agent, AgentBuilder};

use crate::provider::completion::CandleCompletionModel;

/// Builds the peer coach supervisor agent (no RAG).
///
/// Takes the preamble, temperature, and max_tokens from the prompt catalog
/// variant so different supervisor strategies can be evaluated.
pub fn build_supervisor(
    model: CandleCompletionModel,
    preamble: &str,
    temperature: f64,
    max_tokens: usize,
) -> Agent<CandleCompletionModel> {
    AgentBuilder::new(model)
        .preamble(preamble)
        .temperature(temperature)
        .max_tokens(max_tokens as u64)
        .build()
}
