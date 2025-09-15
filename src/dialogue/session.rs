use crate::Result;
use uuid::Uuid;

#[derive(Debug)]
pub struct DialogueSession {
    pub id: Uuid,
    pub messages: Vec<Message>,
}

#[derive(Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl DialogueSession {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            messages: Vec::new(),
        }
    }

    pub fn add_message(&mut self, role: Role, content: String) {
        let message = Message {
            role,
            content,
            timestamp: std::time::SystemTime::now(),
        };
        self.messages.push(message);
    }

    pub fn get_context(&self) -> Result<String> {
        // TODO: Build conversation context for LLM
        let context = self
            .messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(context)
    }
}