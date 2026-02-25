use serde::Deserialize;

/// A fixed test conversation for evaluation runs.
///
/// Each script defines a sequence of user inputs that exercise specific MI
/// conversation dynamics (engagement, ambivalence, change talk, etc.).
#[derive(Deserialize)]
pub struct TestScript {
    pub id: String,
    pub description: String,
    pub turns: Vec<TestTurn>,
}

/// A single turn in a test script.
///
/// `notes` is a human annotation describing what the turn exercises —
/// it is not used by the eval runner, only for documentation.
/// `expected_mode` is the ground-truth conversation mode for route accuracy
/// measurement (optional — only present in mode-exercise scripts).
#[derive(Deserialize)]
pub struct TestTurn {
    pub input: String,
    pub notes: String,
    pub expected_mode: Option<String>,
}
