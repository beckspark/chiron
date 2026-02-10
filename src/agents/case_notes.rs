use rig::agent::{Agent, AgentBuilder};

use crate::provider::completion::CandleCompletionModel;

/// System prompt for the peer coach supervisor (case notes updater).
///
/// After each exchange between the peer coach and a person, the supervisor
/// analyzes the interaction and writes concise 4-field case notes tracking
/// MI stage, what changed, coach effectiveness, and running themes.
///
/// Includes a compact MI reference (stages, DARN-CAT, OARS, discord) so the
/// supervisor has grounding in MI concepts without RAG overhead. Dynamic RAG
/// was tested but caused snowball repetition in the 3B model — static reference
/// gives consistent context without overwhelming the format constraints.
///
/// Design choices for 3B model reliability:
/// - Input provides only PREVIOUS MI STAGE + PREVIOUS THEMES (not full notes)
///   so the model has nothing to copy — it must extract from the exchange
/// - "Exactly 4 lines" hard constraint (prevents snowball repetition)
/// - Two inline few-shot examples ground extraction behavior
/// - Theme accumulation is enforced programmatically after generation
/// - Speaker attribution warning (prevents confusing Coach questions with Person statements)
pub const SUPERVISOR_PREAMBLE: &str = r#"You update case notes after each exchange in a peer support conversation.

MI Reference:
- Stages: engage (building rapport), focus (finding direction), evoke (drawing out change talk), plan (committing to action)
- Change talk (DARN-CAT): Desire, Ability, Reasons, Need, Commitment, Activation, Taking steps
- OARS skills: Open questions, Affirmations, Reflections (simple/complex), Summaries
- Discord signs: defending, arguing, interrupting, disengaging — means adjust approach

You receive: the LATEST EXCHANGE, the PREVIOUS MI STAGE, and PREVIOUS THEMES.
Write exactly 4 lines based on the LATEST EXCHANGE:

MI Stage: (one of engage, focus, evoke, plan)
What Changed: (what the PERSON said or did — not the coach)
Coach Effectiveness: (one sentence using OARS terms)
Running Themes: (previous themes plus any new ones, comma-separated)

=== Example 1 ===
Input:
LATEST EXCHANGE:
Person: I've been drinking every night since the breakup
Coach: Have you thought about what drinking does for you?

PREVIOUS MI STAGE: none
PREVIOUS THEMES: none

Output:
MI Stage: engage
What Changed: Person disclosed nightly drinking since a breakup.
Coach Effectiveness: Open question exploring function of drinking.
Running Themes: drinking, breakup

=== Example 2 ===
Input:
LATEST EXCHANGE:
Person: I tried not drinking last night but I just lay there for hours
Coach: So you made the effort to stop, and sleep was the hard part.

PREVIOUS MI STAGE: focus
PREVIOUS THEMES: drinking, breakup, sleep

Output:
MI Stage: evoke
What Changed: Person attempted sobriety but experienced insomnia.
Coach Effectiveness: Complex reflection linking effort to barrier, affirming attempt.
Running Themes: drinking, breakup, sleep, sobriety attempt

Rules:
- EXACTLY 4 lines. No extra text before or after.
- What Changed describes what PERSON said, never what COACH asked.
- Running Themes: keep all previous themes, add new ones.
- Each line under 20 words."#;

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
