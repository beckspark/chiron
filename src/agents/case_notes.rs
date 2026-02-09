use rig::agent::{Agent, AgentBuilder};

use crate::provider::completion::CandleCompletionModel;

/// System prompt for the peer coach supervisor (case notes updater).
///
/// After each exchange between the peer coach and a person, the supervisor
/// analyzes the interaction and writes concise 4-field case notes tracking
/// MI stage, what changed, coach effectiveness, and running themes.
///
/// Design choices for 3B model reliability:
/// - Concrete example output instead of abstract `[X/Y/Z]` notation (prevents literal copying)
/// - "Exactly 4 lines" hard constraint (prevents snowball repetition of prior notes)
/// - Explicit carry-forward instruction for Running Themes (prevents per-turn regression)
pub const SUPERVISOR_PREAMBLE: &str = r#"You update case notes after each exchange in a peer support conversation.

Write exactly 4 lines:

MI Stage: (one of engage, focus, evoke, plan)
What Changed: (one sentence, latest exchange only)
Coach Effectiveness: (one sentence, what worked or try next)
Running Themes: (all topics so far, comma-separated)

Example output:
MI Stage: engage
What Changed: Person described drinking nightly to cope with stress.
Coach Effectiveness: Good reflection, could explore readiness to change.
Running Themes: drinking, stress, coping

Rules:
- Write EXACTLY 4 lines. No extra text before or after.
- Do NOT repeat or copy the previous notes. Write fresh lines based on the new exchange.
- Running Themes must include all topics from previous notes plus any new ones.
- Keep each line under 20 words."#;

/// Builds the peer coach supervisor agent (no RAG).
///
/// Uses low temperature (0.2) for deterministic structured output and
/// max_tokens 150 for the concise 4-field format.
pub fn build_supervisor(model: CandleCompletionModel) -> Agent<CandleCompletionModel> {
    AgentBuilder::new(model)
        .preamble(SUPERVISOR_PREAMBLE)
        .temperature(0.2)
        .max_tokens(150)
        .build()
}
