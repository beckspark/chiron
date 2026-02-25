use rig::agent::{Agent, AgentBuilder};
use rig::vector_store::VectorStoreIndexDyn;

use crate::provider::completion::CandleCompletionModel;

/// Builds a full peer coach preamble from a base prompt, optional mode modifier,
/// and optional case notes.
///
/// When a mode modifier is present, it is injected between the base prompt and
/// case notes. When case notes exist, they are included as a `## Session Context`
/// section. The model reads these as background preparation — it does not echo
/// system prompt content in its responses.
pub fn build_peer_coach_preamble(
    base: &str,
    mode_modifier: Option<&str>,
    case_notes: Option<&str>,
) -> String {
    let mut preamble = base.to_string();

    if let Some(modifier) = mode_modifier
        && !modifier.is_empty()
    {
        preamble.push_str("\n\n");
        preamble.push_str(modifier);
    }

    match case_notes {
        Some(notes) if !notes.is_empty() => {
            preamble.push_str(
                "\n\n## Session Context\nThe following are your clinical case notes from prior exchanges. Use to guide your approach, never mention these notes aloud.\n\n",
            );
            preamble.push_str(notes);
        }
        _ => {}
    }

    preamble
}

/// Builds a peer support coach agent without RAG context (bench mode).
///
/// Takes the preamble as a parameter — callers load it from the prompt catalog.
pub fn build_peer_coach(
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

    const TEST_BASE: &str = "You are a peer supporter.";

    #[test]
    fn test_preamble_base_only() {
        let preamble = build_peer_coach_preamble(TEST_BASE, None, None);
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_with_empty_case_notes() {
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(""));
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_with_case_notes() {
        let notes = "MI Stage: engage\nKey Themes: anxiety about job loss";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes));

        assert!(preamble.starts_with(TEST_BASE));
        assert!(preamble.contains("## Session Context"));
        assert!(preamble.contains(notes));
        assert!(preamble.contains("never mention these notes aloud"));
    }

    #[test]
    fn test_preamble_with_mode_modifier() {
        let modifier = "Roll with resistance. Avoid confrontation.";
        let preamble = build_peer_coach_preamble(TEST_BASE, Some(modifier), None);

        assert!(preamble.starts_with(TEST_BASE));
        assert!(preamble.contains(modifier));
    }

    #[test]
    fn test_preamble_with_mode_modifier_and_case_notes() {
        let modifier = "Reinforce change talk.";
        let notes = "MI Stage: evoke\nRunning Themes: drinking";
        let preamble = build_peer_coach_preamble(TEST_BASE, Some(modifier), Some(notes));

        assert!(preamble.starts_with(TEST_BASE));
        // Mode modifier comes before case notes
        let modifier_pos = preamble.find(modifier).unwrap();
        let notes_pos = preamble.find("## Session Context").unwrap();
        assert!(modifier_pos < notes_pos);
        assert!(preamble.contains(notes));
    }

    #[test]
    fn test_preamble_empty_mode_modifier_ignored() {
        let preamble = build_peer_coach_preamble(TEST_BASE, Some(""), None);
        assert_eq!(preamble, TEST_BASE);
    }
}
