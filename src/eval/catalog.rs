use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// A collection of prompt variants loaded from a TOML catalog file.
///
/// Each catalog (coach or supervisor) contains one or more prompt variants
/// that can be combined during eval runs to test different prompt strategies.
#[derive(Deserialize)]
pub struct PromptCatalog {
    pub variants: Vec<PromptVariant>,
}

impl PromptCatalog {
    /// Loads a prompt catalog from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        toml::from_str(&content)
            .with_context(|| format!("Failed to parse {}", path.display()))
    }

    /// Returns the variant with the given ID, or an error if not found.
    pub fn get_variant(&self, id: &str) -> Result<&PromptVariant> {
        self.variants
            .iter()
            .find(|v| v.id == id)
            .with_context(|| {
                let available: Vec<&str> = self.variants.iter().map(|v| v.id.as_str()).collect();
                format!("Variant '{}' not found. Available: {:?}", id, available)
            })
    }
}

/// A single prompt variant with generation parameters.
///
/// Includes the full preamble text plus model parameters (temperature, max_tokens)
/// so each variant can tune both the prompt content and generation behavior.
#[derive(Debug, Deserialize, Clone)]
pub struct PromptVariant {
    pub id: String,
    pub description: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub preamble: String,
}

/// A collection of conversation mode definitions loaded from `modes.toml`.
///
/// Each mode shapes how the coach responds to a particular type of user
/// utterance (crisis, resistance, change talk, ambivalence, engagement).
#[derive(Deserialize)]
pub struct ModesCatalog {
    pub modes: Vec<ModeDefinition>,
}

impl ModesCatalog {
    /// Loads a modes catalog from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        toml::from_str(&content)
            .with_context(|| format!("Failed to parse {}", path.display()))
    }

    /// Returns the mode definition with the given ID, or `None` if not found.
    pub fn get_mode(&self, id: &str) -> Option<&ModeDefinition> {
        self.modes.iter().find(|m| m.id == id)
    }
}

/// A single conversation mode with its description, coach behavior modifier,
/// and representative utterances for semantic routing.
#[derive(Deserialize, Clone)]
pub struct ModeDefinition {
    /// Short identifier (e.g., "crisis", "resistance", "change-talk").
    pub id: String,
    /// Human-readable description of when this mode applies.
    pub description: String,
    /// Instruction appended to the coach preamble when this mode is active.
    pub coach_modifier: String,
    /// Representative utterances for building a semantic router.
    pub utterances: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn prompts_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("prompts")
    }

    #[test]
    fn test_load_coach_catalog() {
        let catalog = PromptCatalog::load(&prompts_dir().join("coach.toml")).unwrap();
        assert!(catalog.variants.len() >= 3, "Expected at least 3 coach variants");
        assert!(catalog.get_variant("coach-v1-baseline").is_ok());
    }

    #[test]
    fn test_load_supervisor_catalog() {
        let catalog = PromptCatalog::load(&prompts_dir().join("supervisor.toml")).unwrap();
        assert!(catalog.variants.len() >= 3, "Expected at least 3 supervisor variants");
        assert!(catalog.get_variant("supervisor-v1-baseline").is_ok());
    }

    #[test]
    fn test_load_modes_catalog() {
        let catalog = ModesCatalog::load(&prompts_dir().join("modes.toml")).unwrap();
        assert_eq!(catalog.modes.len(), 5, "Expected 5 conversation modes");

        let mode_ids: Vec<&str> = catalog.modes.iter().map(|m| m.id.as_str()).collect();
        assert!(mode_ids.contains(&"crisis"));
        assert!(mode_ids.contains(&"resistance"));
        assert!(mode_ids.contains(&"change-talk"));
        assert!(mode_ids.contains(&"ambivalence"));
        assert!(mode_ids.contains(&"engagement"));

        // Each mode should have utterances
        for mode in &catalog.modes {
            assert!(
                mode.utterances.len() >= 10,
                "Mode '{}' should have at least 10 utterances, has {}",
                mode.id,
                mode.utterances.len()
            );
            assert!(!mode.coach_modifier.is_empty(), "Mode '{}' should have a coach_modifier", mode.id);
        }
    }

    #[test]
    fn test_get_mode_by_id() {
        let catalog = ModesCatalog::load(&prompts_dir().join("modes.toml")).unwrap();
        let crisis = catalog.get_mode("crisis").expect("crisis mode should exist");
        assert!(crisis.coach_modifier.contains("988"));
    }

    #[test]
    fn test_get_variant_not_found() {
        let catalog = PromptCatalog::load(&prompts_dir().join("coach.toml")).unwrap();
        let result = catalog.get_variant("nonexistent-variant");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"), "Error should say not found: {}", err);
    }
}
