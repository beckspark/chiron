use serde::Serialize;

/// Top-level result of a full evaluation run across all prompt combinations.
#[derive(Serialize)]
pub struct EvalRun {
    pub run_id: String,
    pub model: String,
    pub test_script: String,
    pub seed: u64,
    /// Pipeline architecture used: "two-agent" or "single-context".
    pub architecture: String,
    pub total_duration_ms: u64,
    pub combinations: Vec<CombinationResult>,
}

/// Results for a single (coach_variant, supervisor_variant) combination.
#[derive(Serialize)]
pub struct CombinationResult {
    pub coach_variant: String,
    pub supervisor_variant: String,
    pub total_duration_ms: u64,
    pub turns: Vec<EvalTurnResult>,
}

/// Per-turn output capturing both agents' responses and extracted metadata.
#[derive(Serialize)]
pub struct EvalTurnResult {
    pub turn_number: usize,
    pub input: String,
    pub coach_response: String,
    pub case_notes: String,
    pub mi_stage: Option<String>,
    pub themes: Vec<String>,
    /// Conversation mode detected by the semantic router (e.g., "crisis", "engagement").
    pub detected_mode: Option<String>,
    /// Cosine similarity confidence of the route classification (0.0–1.0).
    pub route_confidence: Option<f64>,
    pub coach_ms: u64,
    pub supervisor_ms: u64,
}
