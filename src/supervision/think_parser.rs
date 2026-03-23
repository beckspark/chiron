use std::collections::HashSet;

/// Analysis extracted from a model's `<think>` block.
#[derive(Debug, Clone)]
pub struct ThinkAnalysis {
    /// Detected MI stage (engage, focus, evoke, plan).
    pub mi_stage: Option<String>,
    /// Strategy the model chose (e.g., "complex reflection", "open question").
    pub strategy_used: Option<String>,
    /// Talk type classification (e.g., "desire change talk", "sustain talk").
    pub talk_type: Option<String>,
    /// Themes mentioned in the think block.
    pub themes: Vec<String>,
    /// User facts extracted via `[USER-FACT: type | content]` tags.
    pub user_facts: Vec<(String, String)>,
    /// Significant turn signal detected via `[SIGNIFICANT: signal_type]` tag.
    pub significant_signal: Option<String>,
    /// Raw think block content for logging.
    pub raw_think: String,
}

/// Analyzes think block content by parsing structured tags the model produces.
///
/// Expects the model to include tags like:
/// - `[MI-STAGE: evoke]`
/// - `[STRATEGY: complex reflection]`
/// - `[TALK-TYPE: desire change talk]`
/// - `[THEMES: drinking, anxiety, sleep]`
pub fn analyze_think_block(think_content: &str) -> ThinkAnalysis {
    let mi_stage = parse_tag(think_content, "MI-STAGE");
    let strategy_used = parse_tag(think_content, "STRATEGY");
    let talk_type = parse_tag(think_content, "TALK-TYPE");
    let themes = parse_tag(think_content, "THEMES")
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let user_facts = parse_user_facts(think_content);
    let significant_signal = parse_tag(think_content, "SIGNIFICANT");

    ThinkAnalysis {
        mi_stage,
        strategy_used,
        talk_type,
        themes,
        user_facts,
        significant_signal,
        raw_think: think_content.to_string(),
    }
}

/// Parses a `[TAG: value]` from think block text. Case-insensitive tag match.
fn parse_tag(text: &str, tag: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let pattern = format!("[{}:", tag.to_lowercase());
    let start = lower.find(&pattern)?;
    let after_tag = start + pattern.len();
    let end = text[after_tag..].find(']')?;
    let value = text[after_tag..after_tag + end].trim();
    if value.is_empty() || value.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(value.to_lowercase())
    }
}

/// Parses all `[USER-FACT: type | content]` tags from think block text.
///
/// Unlike other tags, there can be multiple USER-FACT tags per think block.
/// Each is split on `|` to extract `(fact_type, content)`.
fn parse_user_facts(text: &str) -> Vec<(String, String)> {
    let lower = text.to_lowercase();
    let pattern = "[user-fact:";
    let mut facts = Vec::new();
    let mut search_start = 0;

    while let Some(pos) = lower[search_start..].find(pattern) {
        let abs_pos = search_start + pos;
        let after_tag = abs_pos + pattern.len();
        if let Some(end) = text[after_tag..].find(']') {
            let value = text[after_tag..after_tag + end].trim();
            if let Some((fact_type, content)) = value.split_once('|') {
                let ft = fact_type.trim().to_lowercase();
                let ct = content.trim().to_string();
                if !ft.is_empty() && !ct.is_empty() {
                    facts.push((ft, ct));
                }
            }
            search_start = after_tag + end + 1;
        } else {
            break;
        }
    }

    facts
}

/// Extracts themes from the `Running Themes:` line in case notes.
pub fn extract_themes(notes: &str) -> Option<Vec<String>> {
    notes
        .lines()
        .map(|l| l.replace("**", ""))
        .find(|l| l.trim().to_lowercase().starts_with("running themes:"))
        .and_then(|l| {
            let (_, value) = l.split_once(':')?;
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.to_lowercase() == "none" {
                return None;
            }
            let themes: Vec<String> = trimmed
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty() && t != "none" && t != "n/a")
                .collect();
            if themes.is_empty() {
                None
            } else {
                Some(themes)
            }
        })
}

/// Extracts the MI stage from case notes.
pub fn extract_mi_stage(notes: &str) -> Option<String> {
    notes
        .lines()
        .map(|l| l.replace("**", ""))
        .find(|l| l.trim().to_lowercase().starts_with("mi stage:"))
        .and_then(|l| {
            let (_, value) = l.split_once(':')?;
            Some(value.trim().to_lowercase())
        })
}

/// Merges previous and new themes with recency-biased capping.
///
/// New themes get priority (most recently observed). Previous themes fill
/// remaining slots up to `max_themes`. This prevents unbounded accumulation
/// that would bloat case notes and squeeze the token budget.
pub fn merge_themes(previous: &[String], new: &[String], max_themes: usize) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut merged: Vec<String> = Vec::new();

    // New themes first (highest priority — most recently observed)
    for theme in new {
        if seen.insert(theme.clone()) {
            merged.push(theme.clone());
        }
    }

    // Previous themes fill remaining slots
    for theme in previous {
        if seen.insert(theme.clone()) {
            merged.push(theme.clone());
        }
    }

    merged.truncate(max_themes);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structured_tags() {
        let think = "They're opening up about drinking.\n[MI-STAGE: evoke]\n[STRATEGY: complex reflection]\n[TALK-TYPE: desire change talk]\n[THEMES: drinking, anxiety]";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.mi_stage, Some("evoke".to_string()));
        assert_eq!(analysis.strategy_used, Some("complex reflection".to_string()));
        assert_eq!(analysis.talk_type, Some("desire change talk".to_string()));
        assert_eq!(analysis.themes, vec!["drinking", "anxiety"]);
    }

    #[test]
    fn test_parse_tags_case_insensitive() {
        let think = "[mi-stage: Focus]\n[strategy: Open Question]";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.mi_stage, Some("focus".to_string()));
        assert_eq!(analysis.strategy_used, Some("open question".to_string()));
    }

    #[test]
    fn test_no_tags_returns_none() {
        let think = "Just some reasoning without any structured tags.";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.mi_stage, None);
        assert_eq!(analysis.strategy_used, None);
        assert_eq!(analysis.talk_type, None);
        assert!(analysis.themes.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let analysis = analyze_think_block("");
        assert_eq!(analysis.mi_stage, None);
        assert_eq!(analysis.strategy_used, None);
        assert!(analysis.themes.is_empty());
    }

    #[test]
    fn test_none_values_treated_as_absent() {
        let think = "[MI-STAGE: none]\n[THEMES: none]";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.mi_stage, None);
        assert!(analysis.themes.is_empty());
    }

    #[test]
    fn test_extract_mi_stage_from_case_notes() {
        assert_eq!(extract_mi_stage("MI Stage: engage"), Some("engage".to_string()));
        assert_eq!(extract_mi_stage("**MI Stage:** evoke"), Some("evoke".to_string()));
        assert_eq!(extract_mi_stage("No stage here"), None);
    }

    #[test]
    fn test_parse_single_user_fact() {
        let think = "They mentioned wanting to cut back.\n[MI-STAGE: evoke]\n[USER-FACT: goal | reduce drinking to weekends only]";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.user_facts.len(), 1);
        assert_eq!(analysis.user_facts[0].0, "goal");
        assert_eq!(analysis.user_facts[0].1, "reduce drinking to weekends only");
    }

    #[test]
    fn test_parse_multiple_user_facts() {
        let think = "[USER-FACT: goal | quit smoking]\nSome reasoning here.\n[USER-FACT: barrier | lives with smokers]\n[USER-FACT: strength | ran a 5k last year]";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.user_facts.len(), 3);
        assert_eq!(analysis.user_facts[0], ("goal".into(), "quit smoking".into()));
        assert_eq!(analysis.user_facts[1], ("barrier".into(), "lives with smokers".into()));
        assert_eq!(analysis.user_facts[2], ("strength".into(), "ran a 5k last year".into()));
    }

    #[test]
    fn test_parse_no_user_facts() {
        let think = "[MI-STAGE: engage]\nJust regular reasoning.";
        let analysis = analyze_think_block(think);
        assert!(analysis.user_facts.is_empty());
    }

    #[test]
    fn test_parse_malformed_user_fact() {
        // Missing pipe separator — should be skipped
        let think = "[USER-FACT: goal without pipe]\n[USER-FACT: | empty type]";
        let analysis = analyze_think_block(think);
        assert!(analysis.user_facts.is_empty());
    }

    #[test]
    fn test_parse_significant_signal() {
        let think = "[MI-STAGE: evoke]\n[SIGNIFICANT: change_talk]\nThey expressed desire to change.";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.significant_signal, Some("change_talk".into()));
    }

    #[test]
    fn test_parse_no_significant() {
        let think = "[MI-STAGE: engage]\nNormal conversation.";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.significant_signal, None);
    }

    #[test]
    fn test_merge_themes() {
        let prev = vec!["drinking".to_string(), "breakup".to_string()];
        let new = vec!["breakup".to_string(), "sleep".to_string()];
        // New themes first, then previous (deduped)
        assert_eq!(
            merge_themes(&prev, &new, 8),
            vec!["breakup", "sleep", "drinking"]
        );
    }

    #[test]
    fn test_merge_themes_respects_cap() {
        let prev: Vec<String> = (1..=10).map(|i| format!("old_{i}")).collect();
        let new = vec!["new_a".to_string(), "new_b".to_string()];
        let merged = merge_themes(&prev, &new, 8);
        assert_eq!(merged.len(), 8);
        // New themes always retained
        assert_eq!(merged[0], "new_a");
        assert_eq!(merged[1], "new_b");
        // Oldest previous themes get evicted
        assert!(!merged.contains(&"old_9".to_string()));
        assert!(!merged.contains(&"old_10".to_string()));
    }

    #[test]
    fn test_merge_themes_cap_preserves_recent() {
        let prev = vec!["old1".into(), "old2".into(), "old3".into()];
        let new = vec!["new1".into(), "new2".into()];
        let merged = merge_themes(&prev, &new, 3);
        assert_eq!(merged, vec!["new1", "new2", "old1"]);
    }
}
