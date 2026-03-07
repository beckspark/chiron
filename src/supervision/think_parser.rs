use std::collections::HashSet;

/// Analysis extracted from a model's `<think>` block.
#[derive(Debug, Clone)]
pub struct ThinkAnalysis {
    /// Detected MI stage (engage, focus, evoke, plan).
    pub mi_stage: Option<String>,
    /// Strategy the model chose (e.g., "complex reflection", "open question").
    pub strategy_used: Option<String>,
    /// Themes mentioned in the think block.
    pub themes: Vec<String>,
    /// Raw think block content for logging.
    pub raw_think: String,
}

/// Analyzes think block content using keyword heuristics.
pub fn analyze_think_block(think_content: &str) -> ThinkAnalysis {
    let lower = think_content.to_lowercase();

    let mi_stage = detect_mi_stage(&lower);
    let strategy_used = detect_strategy(&lower);
    let themes = extract_themes_from_think(&lower);

    ThinkAnalysis {
        mi_stage,
        strategy_used,
        themes,
        raw_think: think_content.to_string(),
    }
}

/// Detects MI stage from think block keywords.
fn detect_mi_stage(text: &str) -> Option<String> {
    // Order matters — check more specific stages first
    let stages = [
        ("plan", &["planning", "action plan", "next step", "concrete step", "plan stage"][..]),
        ("evoke", &["evoke", "evoking", "change talk", "elicit", "motivation", "ambivalence", "discrepancy"][..]),
        ("focus", &["focus", "focusing", "agenda", "direction", "priorit"][..]),
        ("engage", &["engage", "engaging", "rapport", "trust", "relationship", "build connection"][..]),
    ];

    for (stage, keywords) in &stages {
        if keywords.iter().any(|k| text.contains(k)) {
            return Some(stage.to_string());
        }
    }

    None
}

/// Detects MI strategy from think block keywords.
fn detect_strategy(text: &str) -> Option<String> {
    let strategies = [
        ("complex reflection", &["complex reflection", "deeper reflection", "meaning behind", "feels like"][..]),
        ("simple reflection", &["simple reflection", "reflect back", "paraphrase"][..]),
        ("affirmation", &["affirm", "strength", "acknowledge effort", "recognize"][..]),
        ("open question", &["open question", "open-ended", "explore", "what do you", "how do you"][..]),
        ("summarizing", &["summarize", "summary", "pulling together"][..]),
        ("rolling with resistance", &["roll with", "resistance", "not arguing", "avoid confrontation"][..]),
        ("developing discrepancy", &["discrepancy", "values", "gap between"][..]),
        ("autonomy support", &["autonomy", "their choice", "up to them", "self-determination"][..]),
    ];

    for (strategy, keywords) in &strategies {
        if keywords.iter().any(|k| text.contains(k)) {
            return Some(strategy.to_string());
        }
    }

    None
}

/// Extracts potential themes from think block text.
fn extract_themes_from_think(text: &str) -> Vec<String> {
    let theme_indicators = [
        "drinking", "alcohol", "substance", "drug",
        "breakup", "relationship", "partner", "divorce",
        "anxiety", "anxious", "worried", "stress",
        "depression", "depressed", "sad", "hopeless",
        "sleep", "insomnia",
        "work", "job", "career", "employment",
        "family", "parent", "child", "sibling",
        "grief", "loss", "death",
        "anger", "frustrated", "resentment",
        "isolation", "lonely", "alone",
        "self-esteem", "confidence", "worth",
        "trauma", "abuse",
        "health", "pain", "chronic",
        "financial", "money", "debt",
    ];

    let mut found: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for theme in &theme_indicators {
        if text.contains(theme) && seen.insert(theme.to_string()) {
            found.push(theme.to_string());
        }
    }

    found
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

/// Replaces the `Running Themes:` line in notes with merged themes.
pub fn replace_themes_line(notes: &str, merged: &[String]) -> String {
    let new_line = format!("Running Themes: {}", merged.join(", "));

    let mut found = false;
    let replaced: Vec<String> = notes
        .lines()
        .map(|l| {
            let clean = l.replace("**", "");
            if clean.trim().to_lowercase().starts_with("running themes:") {
                found = true;
                new_line.clone()
            } else {
                l.to_string()
            }
        })
        .collect();

    if found {
        replaced.join("\n")
    } else {
        format!("{}\n{}", notes.trim_end(), new_line)
    }
}

/// Strips echoed `LATEST EXCHANGE:` blocks from text.
pub fn strip_echoed_exchanges(notes: &str) -> String {
    match notes.find("LATEST EXCHANGE:") {
        Some(pos) => notes[..pos].trim_end().to_string(),
        None => notes.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_think_block_evoke() {
        let think = "The person is showing change talk about drinking. I should use a complex reflection to evoke more motivation.";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.mi_stage, Some("evoke".to_string()));
        assert_eq!(analysis.strategy_used, Some("complex reflection".to_string()));
        assert!(analysis.themes.contains(&"drinking".to_string()));
    }

    #[test]
    fn test_analyze_think_block_engage() {
        let think = "Building rapport with this person. They seem anxious about opening up.";
        let analysis = analyze_think_block(think);
        assert_eq!(analysis.mi_stage, Some("engage".to_string()));
        assert!(analysis.themes.contains(&"anxious".to_string()));
    }

    #[test]
    fn test_analyze_think_block_empty() {
        let analysis = analyze_think_block("");
        assert_eq!(analysis.mi_stage, None);
        assert_eq!(analysis.strategy_used, None);
        assert!(analysis.themes.is_empty());
    }

    #[test]
    fn test_extract_mi_stage() {
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

    #[test]
    fn test_strip_echoed_exchanges() {
        let notes = "MI Stage: engage\n\nLATEST EXCHANGE:\nPerson: hello";
        assert_eq!(strip_echoed_exchanges(notes), "MI Stage: engage");
    }
}
