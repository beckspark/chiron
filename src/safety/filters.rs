use crate::Result;

pub struct SafetyFilters;

impl SafetyFilters {
    pub fn new() -> Self {
        Self
    }

    pub fn filter_input(&self, input: &str) -> Result<String> {
        // TODO: Implement input safety filtering
        Ok(input.to_string())
    }

    pub fn filter_output(&self, output: &str) -> Result<String> {
        let mut filtered = output.to_string();

        // Add disclaimer if medical advice patterns are detected
        let medical_keywords = ["diagnosis", "prescribe", "medication", "disorder"];
        let output_lower = output.to_lowercase();

        for keyword in &medical_keywords {
            if output_lower.contains(keyword) {
                filtered = format!(
                    "{}\n\n⚠️  Reminder: I cannot provide medical advice or diagnoses. Please consult a qualified mental health professional for clinical guidance.",
                    filtered
                );
                break;
            }
        }

        Ok(filtered)
    }
}