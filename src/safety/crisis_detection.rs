use crate::Result;

pub struct CrisisDetector;

impl CrisisDetector {
    pub fn new() -> Self {
        Self
    }

    pub fn detect_crisis(&self, input: &str) -> Result<bool> {
        // TODO: Implement crisis detection logic
        // Look for keywords like "suicide", "hurt myself", "end it all", etc.
        let crisis_keywords = ["suicide", "kill myself", "hurt myself", "end it all"];

        let input_lower = input.to_lowercase();
        for keyword in &crisis_keywords {
            if input_lower.contains(keyword) {
                return Ok(true);
            }
        }

        Ok(false)
    }
}