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

    ThinkAnalysis {
        mi_stage,
        strategy_used,
        talk_type,
        themes,
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

/// Merges previous and new themes with order-preserving set union.
pub fn merge_themes(previous: &[String], new: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = previous.iter().cloned().collect();
    let mut merged: Vec<String> = previous.to_vec();

    for theme in new {
        if seen.insert(theme.clone()) {
            merged.push(theme.clone());
        }
    }

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
    fn test_merge_themes() {
        let prev = vec!["drinking".to_string(), "breakup".to_string()];
        let new = vec!["breakup".to_string(), "sleep".to_string()];
        assert_eq!(
            merge_themes(&prev, &new),
            vec!["drinking", "breakup", "sleep"]
        );
    }
}
