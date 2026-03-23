use rig::agent::{Agent, AgentBuilder};

use crate::catalog::ModeCatalog;
use crate::provider::LlamaCppCompletionModel;

/// Builds a full peer coach preamble from a base prompt and optional case notes.
///
/// Injects think block structure, case notes, stage-matched MI technique guidance,
/// and mode-specific coaching modifiers into the system prompt.
pub fn build_peer_coach_preamble(
    base: &str,
    think_instructions: Option<&str>,
    case_notes: Option<&str>,
    mode_catalog: Option<&ModeCatalog>,
    rag_context: Option<&str>,
) -> String {
    let mut preamble = base.to_string();

    if let Some(instructions) = think_instructions {
        preamble.push_str("\n\n");
        preamble.push_str(instructions);
    }

    // RAG context injected between think instructions and case notes
    if let Some(context) = rag_context.filter(|c| !c.is_empty()) {
        preamble.push_str("\n\n");
        preamble.push_str(context);
    }

    match case_notes {
        Some(notes) if !notes.is_empty() => {
            preamble.push_str(
                "\n\n## Session Context\nThe following are your clinical case notes from prior exchanges. Use to guide your approach, never mention these notes aloud.\n\n",
            );
            preamble.push_str(notes);

            // Inject stage-matched technique guidance
            if let Some(guidance) = stage_guidance(notes) {
                preamble.push_str("\n\n## Technique Guidance\n");
                preamble.push_str(guidance);
            }

            // Inject mode-specific coaching modifier based on detected strategy
            if let Some(catalog) = mode_catalog {
                if let Some(modifier) = detect_mode_modifier(notes, catalog) {
                    preamble.push_str("\n\n## Current Mode\n");
                    preamble.push_str(modifier);
                }
            }
        }
        _ => {}
    }

    preamble
}

/// Detects the conversation mode from case notes and returns the coach modifier.
///
/// Maps strategy keywords in case notes to mode IDs in the catalog.
fn detect_mode_modifier<'a>(case_notes: &str, catalog: &'a ModeCatalog) -> Option<&'a str> {
    let lower = case_notes.to_lowercase();

    // Check for strategy signals that map to modes
    let mode_id = if lower.contains("rolling with resistance") || lower.contains("resistance") {
        "resistance"
    } else if lower.contains("reinforce") || lower.contains("change talk") {
        "change-talk"
    } else if lower.contains("discrepancy") || lower.contains("ambivalence") || lower.contains("double-sided") {
        "ambivalence"
    } else if lower.contains("mi stage: engage") {
        "engagement"
    } else {
        return None;
    };

    catalog.get_mode(mode_id).map(|m| m.coach_modifier.as_str())
}

/// Returns MI technique guidance appropriate for the detected stage.
fn stage_guidance(case_notes: &str) -> Option<&'static str> {
    let lower = case_notes.to_lowercase();
    let stage = lower
        .lines()
        .find(|l| l.trim().starts_with("mi stage:"))?;
    let (_, value) = stage.split_once(':')?;
    let stage = value.trim();

    Some(match stage {
        "engage" => "Focus on rapport: open questions, simple reflections. Don't push for change yet.",
        "focus" => "Direction is emerging. Clarify the target behavior with open questions. Reflect what matters to them.",
        "evoke" => "Draw out change talk: DARN questions (desire, ability, reasons, need). Use elaboration, values discrepancy, and selective summary of change talk.",
        "plan" => "They're showing commitment. Explore concrete steps. Support their plan with affirmations. Ask what they'll do first.",
        _ => return None,
    })
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
        let preamble = build_peer_coach_preamble(TEST_BASE, None, None, None, None);
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_with_empty_case_notes() {
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(""), None, None);
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_with_case_notes() {
        let notes = "MI Stage: engage\nKey Themes: anxiety about job loss";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes), None, None);

        assert!(preamble.starts_with(TEST_BASE));
        assert!(preamble.contains("## Session Context"));
        assert!(preamble.contains(notes));
        assert!(preamble.contains("never mention these notes aloud"));
    }

    #[test]
    fn test_preamble_with_stage_guidance() {
        let notes = "MI Stage: evoke\nRunning Themes: drinking";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes), None, None);

        assert!(preamble.contains("## Technique Guidance"));
        assert!(preamble.contains("DARN questions"));
    }

    #[test]
    fn test_preamble_engage_guidance() {
        let notes = "MI Stage: engage\nRunning Themes: anxiety";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes), None, None);

        assert!(preamble.contains("## Technique Guidance"));
        assert!(preamble.contains("rapport"));
    }

    #[test]
    fn test_preamble_with_rag_context() {
        let rag = "## What You Know About This Person\n- Goal: reduce drinking to weekends";
        let notes = "MI Stage: evoke\nRunning Themes: drinking";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes), None, Some(rag));

        assert!(preamble.contains("## What You Know About This Person"));
        assert!(preamble.contains("reduce drinking"));
        // RAG context should appear before case notes
        let rag_pos = preamble.find("What You Know").unwrap();
        let notes_pos = preamble.find("Session Context").unwrap();
        assert!(rag_pos < notes_pos, "RAG context should precede case notes");
    }

    #[test]
    fn test_preamble_empty_rag_context_ignored() {
        let preamble = build_peer_coach_preamble(TEST_BASE, None, None, None, Some(""));
        assert_eq!(preamble, TEST_BASE);
    }

    #[test]
    fn test_preamble_full_stack_assembly() {
        // Load real mode catalog from prompts/modes.toml
        let catalog = crate::catalog::ModeCatalog::load(std::path::Path::new("prompts/modes.toml"))
            .expect("modes.toml should be loadable in tests");

        let think_instructions = "Think carefully. Include [MI-STAGE], [STRATEGY], [TALK-TYPE], [THEMES].";
        let rag_context = "## What You Know About This Person\n- Lost job 3 months ago\n- Drinking more since then";
        // Case notes with resistance keyword to trigger mode detection
        let case_notes = "MI Stage: evoke\nStrategy Used: rolling with resistance\nRunning Themes: drinking, job, anxiety";

        let preamble = build_peer_coach_preamble(
            TEST_BASE,
            Some(think_instructions),
            Some(case_notes),
            Some(&catalog),
            Some(rag_context),
        );

        // All 5 sections present
        assert!(preamble.starts_with(TEST_BASE), "base preamble first");
        assert!(preamble.contains(think_instructions), "think instructions present");
        assert!(preamble.contains("What You Know About This Person"), "RAG context present");
        assert!(preamble.contains("## Session Context"), "case notes section present");
        assert!(preamble.contains("## Technique Guidance"), "technique guidance present");
        assert!(preamble.contains("## Current Mode"), "mode modifier present");

        // Ordering: think → RAG → case notes → technique → mode
        let think_pos = preamble.find(think_instructions).unwrap();
        let rag_pos = preamble.find("What You Know").unwrap();
        let notes_pos = preamble.find("## Session Context").unwrap();
        let technique_pos = preamble.find("## Technique Guidance").unwrap();
        let mode_pos = preamble.find("## Current Mode").unwrap();

        assert!(think_pos < rag_pos, "think instructions before RAG");
        assert!(rag_pos < notes_pos, "RAG before case notes");
        assert!(notes_pos < technique_pos, "case notes before technique");
        assert!(technique_pos < mode_pos, "technique before mode");
    }

    #[test]
    fn test_mode_detection_keywords() {
        let catalog = crate::catalog::ModeCatalog::load(std::path::Path::new("prompts/modes.toml"))
            .expect("modes.toml should be loadable in tests");

        // Resistance mode triggered by keyword
        let notes_resistance = "MI Stage: evoke\nStrategy Used: rolling with resistance";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes_resistance), Some(&catalog), None);
        assert!(preamble.contains("## Current Mode"), "resistance mode should trigger");

        // Change-talk mode triggered by keyword
        let notes_change = "MI Stage: evoke\nTalk Type: change talk";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes_change), Some(&catalog), None);
        assert!(preamble.contains("## Current Mode"), "change-talk mode should trigger");

        // No mode when notes have no trigger keywords
        let notes_plain = "MI Stage: focus\nRunning Themes: work";
        let preamble = build_peer_coach_preamble(TEST_BASE, None, Some(notes_plain), Some(&catalog), None);
        assert!(!preamble.contains("## Current Mode"), "no mode modifier for plain focus notes");
    }
}
