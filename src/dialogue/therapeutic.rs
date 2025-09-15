use crate::Result;

#[derive(Debug)]
pub enum TherapyPhase {
    Assessment,
    Initial,
    Middle,
    Termination,
}

pub struct TherapeuticContext {
    pub phase: TherapyPhase,
    pub session_count: u32,
    pub goals: Vec<String>,
}

impl TherapeuticContext {
    pub fn new() -> Self {
        Self {
            phase: TherapyPhase::Assessment,
            session_count: 0,
            goals: Vec::new(),
        }
    }

    pub fn advance_phase(&mut self) -> Result<()> {
        self.phase = match self.phase {
            TherapyPhase::Assessment => TherapyPhase::Initial,
            TherapyPhase::Initial => TherapyPhase::Middle,
            TherapyPhase::Middle => TherapyPhase::Termination,
            TherapyPhase::Termination => TherapyPhase::Termination,
        };
        Ok(())
    }

    pub fn increment_session(&mut self) {
        self.session_count += 1;
    }

    pub fn add_goal(&mut self, goal: String) {
        self.goals.push(goal);
    }
}
