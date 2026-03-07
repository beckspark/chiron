/// Checks if user input contains crisis indicators.
///
/// Simple keyword matching — the fine-tuned model handles nuanced crisis
/// detection in its think block, but this catches obvious cases for
/// immediate hardcoded response before model inference.
pub fn is_crisis(input: &str) -> bool {
    const PATTERNS: &[&str] = &[
        "kill myself",
        "suicide",
        "end my life",
        "want to die",
        "better off dead",
        "no reason to live",
        "ending it all",
        "take my own life",
    ];
    let lower = input.to_lowercase();
    PATTERNS.iter().any(|p| lower.contains(p))
}

/// Returns a hardcoded crisis response with resource information.
pub fn crisis_response() -> &'static str {
    "I hear you, and I'm really glad you told me. What you're feeling matters. \
     Please reach out to the 988 Suicide & Crisis Lifeline (call or text 988) \
     or text HOME to 741741 for the Crisis Text Line. You don't have to go through this alone."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crisis_detection() {
        assert!(is_crisis("I want to kill myself"));
        assert!(is_crisis("thinking about suicide"));
        assert!(is_crisis("I'd be better off dead"));
        assert!(is_crisis("WANT TO DIE"));
    }

    #[test]
    fn test_non_crisis() {
        assert!(!is_crisis("I've been feeling down lately"));
        assert!(!is_crisis("I'm stressed about work"));
        assert!(!is_crisis("I don't know what to do"));
    }
}
