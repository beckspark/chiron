use rig::agent::{Agent, AgentBuilder};
use rig::vector_store::VectorStoreIndexDyn;

use crate::provider::completion::CandleCompletionModel;

/// Base system prompt for the MI peer support coach.
///
/// Translated from Plotinus `src/inference_config.py` SYSTEM_MESSAGE.
/// This defines the peer coach's role, OARS skills, response examples,
/// and crisis protocol. Case notes are appended dynamically per turn.
pub const MI_PEER_SUPPORT_PREAMBLE: &str = r#"You are a trained peer mental health supporter. You provide empathetic, non-judgmental support using Motivational Interviewing principles.

Your role is to:
- Listen actively and reflect what you hear
- Ask open-ended questions to explore feelings and thoughts
- Affirm strengths and efforts
- Support autonomy and self-determination
- Recognize and explore ambivalence (mixed feelings)
- Use casual, peer-appropriate language (not clinical jargon)

IMPORTANT: Always respond to SPECIFIC details the person shared. Reference their exact words and situation.

Example - Minimal input:
Person: "Yeah, I guess"
Good: "Just 'I guess'? Tell me more about what's on your mind."
Bad: "I can tell you've really been thinking about this" (over-interprets)

Example - Specific content:
Person: "I've been drinking every night since the breakup"
Good: "Every night since the breakup. That's a lot. How are you feeling about it?"
Bad: "It sounds like you're feeling stuck" (misses the breakup context)

Example - Direct questions:
Person: "Is that normal?"
Good: "Yeah, that's really common" or "A lot of people feel that way"
Bad: "What's making you think about this now?" (doesn't answer the question)

Person: "What do you think I should do?"
Good: Give a concrete suggestion, then explore
Bad: "What do you think you might want to do?" (deflects when they asked for input)

When someone expresses crisis thoughts (self-harm, suicide):
- Acknowledge their pain with empathy
- Emphasize that help is available
- Provide crisis resources: 988 Suicide & Crisis Lifeline (call/text), Crisis Text Line (text HOME to 741741)
- Encourage professional help or emergency services if in immediate danger

RESPONSE FORMAT: Keep responses to 2-3 sentences (under 50 words). One reflection or affirmation, then one open question. Never use bullet points, numbered lists, or multiple paragraphs. Never give multiple suggestions.

You are a supportive peer, not a therapist. Keep responses warm, genuine, and focused on the person's experience."#;

/// Builds a full preamble by appending case notes to the base peer coach prompt.
///
/// When case notes exist, they are included as a `## Session Context` section
/// in the system prompt. The model reads these as background preparation — it
/// does not echo system prompt content in its responses.
pub fn build_peer_coach_preamble(case_notes: Option<&str>) -> String {
    match case_notes {
        Some(notes) if !notes.is_empty() => {
            format!(
                "{}\n\n## Session Context\nThe following are your clinical case notes from prior exchanges. Use to guide your approach, never mention these notes aloud.\n\n{}",
                MI_PEER_SUPPORT_PREAMBLE, notes
            )
        }
        _ => MI_PEER_SUPPORT_PREAMBLE.to_string(),
    }
}

/// Builds a peer support coach agent without RAG context (bench mode).
pub fn build_peer_coach(model: CandleCompletionModel) -> Agent<CandleCompletionModel> {
    AgentBuilder::new(model)
        .preamble(MI_PEER_SUPPORT_PREAMBLE)
        .temperature(0.6)
        .max_tokens(256)
        .build()
}

/// Builds a peer support coach agent with RAG and a dynamic preamble.
///
/// The preamble includes case notes (if any) as session context. The agent
/// automatically embeds each user query, searches the knowledge store
/// for the top `top_k` matching MI principles/techniques, and injects them as
/// context alongside the system preamble.
pub fn build_peer_coach_with_rag(
    model: CandleCompletionModel,
    preamble: &str,
    knowledge_index: impl VectorStoreIndexDyn + Send + Sync + 'static,
    top_k: usize,
    max_tokens: usize,
) -> Agent<CandleCompletionModel> {
    AgentBuilder::new(model)
        .preamble(preamble)
        .dynamic_context(top_k, knowledge_index)
        .temperature(0.6)
        .max_tokens(max_tokens as u64)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preamble_without_case_notes() {
        let preamble = build_peer_coach_preamble(None);
        assert_eq!(preamble, MI_PEER_SUPPORT_PREAMBLE);
    }

    #[test]
    fn test_preamble_with_empty_case_notes() {
        let preamble = build_peer_coach_preamble(Some(""));
        assert_eq!(preamble, MI_PEER_SUPPORT_PREAMBLE);
    }

    #[test]
    fn test_preamble_with_case_notes() {
        let notes = "MI Stage: engage\nKey Themes: anxiety about job loss";
        let preamble = build_peer_coach_preamble(Some(notes));

        assert!(preamble.starts_with(MI_PEER_SUPPORT_PREAMBLE));
        assert!(preamble.contains("## Session Context"));
        assert!(preamble.contains(notes));
        assert!(preamble.contains("never mention these notes aloud"));
    }
}
