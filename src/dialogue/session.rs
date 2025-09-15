use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DialogueSession {
    pub id: Uuid,
    pub user_id: Option<String>, // For future multi-user support
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub messages: Vec<Message>,
    pub therapeutic_metadata: TherapeuticMetadata,
    pub session_quality: SessionQuality,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub id: Uuid,
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub embedding_metadata: Option<EmbeddingMetadata>, // For future RAG indexing
    pub therapeutic_tags: Vec<String>, // e.g., ["anxiety", "coping_strategy", "breakthrough"]
    pub sentiment_score: Option<f32>,  // -1.0 to 1.0
    pub crisis_indicators: Vec<String>, // Any detected crisis patterns
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TherapeuticMetadata {
    pub primary_concerns: Vec<String>,        // Main therapeutic themes
    pub intervention_techniques: Vec<String>, // CBT, mindfulness, etc.
    pub progress_indicators: Vec<ProgressIndicator>,
    pub therapy_phase: String, // assessment, initial, middle, termination
    pub session_count: u32,
    pub total_duration_minutes: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProgressIndicator {
    pub metric: String, // e.g., "mood_improvement", "coping_skills"
    pub baseline_score: f32,
    pub current_score: f32,
    pub trend: String, // "improving", "stable", "declining"
    pub last_assessed: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SessionQuality {
    pub therapeutic_alliance_score: Option<f32>, // 1-10 scale
    pub conversation_coherence: Option<f32>,     // How well the conversation flows
    pub safety_compliance: bool,                 // All safety protocols followed
    pub user_engagement_level: Option<f32>,      // Response length, emotion, etc.
    pub ai_response_quality: Option<f32>,        // Empathy, appropriateness
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingMetadata {
    pub vector_id: Option<String>,      // Reference to vector DB
    pub semantic_tags: Vec<String>,     // For semantic search
    pub clinical_concepts: Vec<String>, // Extracted therapeutic concepts
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl DialogueSession {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            user_id: None,
            created_at: now,
            last_updated: now,
            messages: Vec::new(),
            therapeutic_metadata: TherapeuticMetadata {
                primary_concerns: Vec::new(),
                intervention_techniques: Vec::new(),
                progress_indicators: Vec::new(),
                therapy_phase: "assessment".to_string(),
                session_count: 0,
                total_duration_minutes: None,
            },
            session_quality: SessionQuality {
                therapeutic_alliance_score: None,
                conversation_coherence: None,
                safety_compliance: true,
                user_engagement_level: None,
                ai_response_quality: None,
            },
        }
    }

    pub fn add_message(&mut self, role: Role, content: String) {
        let message = Message {
            id: Uuid::new_v4(),
            role,
            content,
            timestamp: Utc::now(),
            embedding_metadata: None,
            therapeutic_tags: Vec::new(),
            sentiment_score: None,
            crisis_indicators: Vec::new(),
        };
        self.messages.push(message);
        self.last_updated = Utc::now();
    }

    pub fn add_message_with_metadata(
        &mut self,
        role: Role,
        content: String,
        therapeutic_tags: Vec<String>,
        sentiment_score: Option<f32>,
        crisis_indicators: Vec<String>,
    ) {
        let message = Message {
            id: Uuid::new_v4(),
            role,
            content,
            timestamp: Utc::now(),
            embedding_metadata: None,
            therapeutic_tags,
            sentiment_score,
            crisis_indicators,
        };
        self.messages.push(message);
        self.last_updated = Utc::now();
    }

    pub fn get_context(&self) -> Result<String> {
        // Build therapeutic context for LLM with recent conversation history
        let recent_messages = self.messages.iter().rev().take(10).rev();
        let context = recent_messages
            .map(|m| {
                let role_str = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::System => "System",
                };
                format!("{}: {}", role_str, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(context)
    }

    pub fn get_therapeutic_summary(&self) -> String {
        format!(
            "Session {} - Phase: {} | Concerns: {} | Techniques: {}",
            self.id,
            self.therapeutic_metadata.therapy_phase,
            self.therapeutic_metadata.primary_concerns.join(", "),
            self.therapeutic_metadata.intervention_techniques.join(", ")
        )
    }

    pub fn extract_training_data(&self) -> Vec<TrainingExample> {
        // Extract conversation pairs suitable for model training/RAG
        let mut training_examples = Vec::new();

        for window in self.messages.windows(2) {
            if let [user_msg, assistant_msg] = window {
                if matches!(user_msg.role, Role::User)
                    && matches!(assistant_msg.role, Role::Assistant)
                {
                    training_examples.push(TrainingExample {
                        id: Uuid::new_v4(),
                        session_id: self.id,
                        user_input: user_msg.content.clone(),
                        assistant_response: assistant_msg.content.clone(),
                        therapeutic_context: self.therapeutic_metadata.clone(),
                        quality_score: self.session_quality.ai_response_quality,
                        therapeutic_tags: assistant_msg.therapeutic_tags.clone(),
                        timestamp: assistant_msg.timestamp,
                    });
                }
            }
        }

        training_examples
    }

    pub fn update_progress(&mut self, metric: String, score: f32) {
        if let Some(indicator) = self
            .therapeutic_metadata
            .progress_indicators
            .iter_mut()
            .find(|p| p.metric == metric)
        {
            indicator.current_score = score;
            indicator.last_assessed = Utc::now();
        } else {
            self.therapeutic_metadata
                .progress_indicators
                .push(ProgressIndicator {
                    metric,
                    baseline_score: score,
                    current_score: score,
                    trend: "stable".to_string(),
                    last_assessed: Utc::now(),
                });
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingExample {
    pub id: Uuid,
    pub session_id: Uuid,
    pub user_input: String,
    pub assistant_response: String,
    pub therapeutic_context: TherapeuticMetadata,
    pub quality_score: Option<f32>,
    pub therapeutic_tags: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

// Session storage and retrieval functionality
use std::fs;
use std::path::PathBuf;

pub struct SessionStorage {
    storage_dir: PathBuf,
}

impl SessionStorage {
    pub fn new() -> Result<Self> {
        let storage_dir = dirs::data_local_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find local data directory"))?
            .join("chiron")
            .join("sessions");

        fs::create_dir_all(&storage_dir)?;

        Ok(Self { storage_dir })
    }

    pub async fn save_session(&self, session: &DialogueSession) -> Result<()> {
        let file_path = self.storage_dir.join(format!("{}.json", session.id));
        let json_data = serde_json::to_string_pretty(session)?;
        tokio::fs::write(file_path, json_data).await?;
        Ok(())
    }

    pub async fn load_session(&self, session_id: Uuid) -> Result<DialogueSession> {
        let file_path = self.storage_dir.join(format!("{}.json", session_id));
        let json_data = tokio::fs::read_to_string(file_path).await?;
        let session: DialogueSession = serde_json::from_str(&json_data)?;
        Ok(session)
    }

    pub async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let mut sessions = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.storage_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            if let Some(ext) = entry.path().extension() {
                if ext == "json" {
                    if let Ok(session) = self.load_session_summary(&entry.path()).await {
                        sessions.push(session);
                    }
                }
            }
        }

        sessions.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
        Ok(sessions)
    }

    async fn load_session_summary(&self, path: &PathBuf) -> Result<SessionSummary> {
        let json_data = tokio::fs::read_to_string(path).await?;
        let session: DialogueSession = serde_json::from_str(&json_data)?;

        Ok(SessionSummary {
            id: session.id,
            created_at: session.created_at,
            last_updated: session.last_updated,
            message_count: session.messages.len(),
            therapy_phase: session.therapeutic_metadata.therapy_phase,
            primary_concerns: session.therapeutic_metadata.primary_concerns,
            preview: session
                .messages
                .first()
                .map(|m| m.content.chars().take(100).collect::<String>())
                .unwrap_or_else(|| "Empty session".to_string()),
        })
    }

    pub async fn export_training_data(&self) -> Result<Vec<TrainingExample>> {
        let sessions = self.list_sessions().await?;
        let mut all_training_data = Vec::new();

        for session_summary in sessions {
            if let Ok(session) = self.load_session(session_summary.id).await {
                all_training_data.extend(session.extract_training_data());
            }
        }

        Ok(all_training_data)
    }

    pub async fn export_training_jsonl(&self, output_path: &PathBuf) -> Result<()> {
        let training_data = self.export_training_data().await?;
        let mut jsonl_content = String::new();

        for example in training_data {
            let line = serde_json::to_string(&example)?;
            jsonl_content.push_str(&line);
            jsonl_content.push('\n');
        }

        tokio::fs::write(output_path, jsonl_content).await?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub message_count: usize,
    pub therapy_phase: String,
    pub primary_concerns: Vec<String>,
    pub preview: String,
}
