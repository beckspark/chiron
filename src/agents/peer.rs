use rig::agent::{Agent, AgentBuilder};

use crate::provider::LlamaCppCompletionModel;

/// Builds a full peer coach preamble from a base prompt and optional case notes.
pub fn build_peer_coach_preamble(base: &str, case_notes: Option<&str>) -> String {
    let mut preamble = base.to_string();

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

/// Builds a peer support coach agent (bench mode, no case notes).
pub fn build_peer_coach(
    model: LlamaCppCompletionModel,
    preamble: &str,
    temperature: f64,
    max_tokens: usize,
) -> Agent<LlamaCppCompletionModel> {
    AgentBuilder::new(model)
        .preamble(preamble)
        .temperature(temperature)
        .max_tokens(max_tokens as u64)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BASE: &str = "You are a peer supporter.";

    #[test]
    fn test_preamble_base_only() {
        let preamble = build_peer_coach_preamble(TEST_BASE, None);
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_with_empty_case_notes() {
        let preamble = build_peer_coach_preamble(TEST_BASE, Some(""));
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_with_case_notes() {
        let notes = "MI Stage: engage\nKey Themes: anxiety about job loss";
        let preamble = build_peer_coach_preamble(TEST_BASE, Some(notes));

        assert!(preamble.starts_with(TEST_BASE));
        assert!(preamble.contains("## Session Context"));
        assert!(preamble.contains(notes));
        assert!(preamble.contains("never mention these notes aloud"));
    }
}
