use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// A collection of prompt variants loaded from a TOML catalog file.
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
#[derive(Debug, Deserialize, Clone)]
pub struct PromptVariant {
    pub id: String,
    pub description: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub preamble: String,
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
        assert!(!catalog.variants.is_empty());
        assert!(catalog.get_variant("v5-finetuned").is_ok());
    }

    #[test]
    fn test_get_variant_not_found() {
        let catalog = PromptCatalog::load(&prompts_dir().join("coach.toml")).unwrap();
        let result = catalog.get_variant("nonexistent");
        assert!(result.is_err());
    }
}
