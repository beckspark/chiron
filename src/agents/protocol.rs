use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Agent capability definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub description: String,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
}

/// Request to an agent
#[derive(Debug, Clone)]
pub struct AgentRequest {
    pub input: String,
    pub context: AgentContext,
    pub parameters: HashMap<String, String>,
}

/// Response from an agent
#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub content: String,
    pub metadata: AgentMetadata,
    pub resources_used: Vec<String>,
}

/// Metadata about the agent response
#[derive(Debug, Clone)]
pub struct AgentMetadata {
    pub agent_name: String,
    pub confidence: f32, // 0.0 to 1.0
    pub processing_time_ms: u64,
    pub sources: Vec<String>,
    pub content_type: String, // "text", "json", "markdown", etc.
}

/// Shared context between agents
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub user_input: String,
    pub session_id: String,
    pub therapeutic_phase: String,
    pub session_count: u32,
    pub conversation_history: Vec<String>,
    pub shared_resources: HashMap<String, serde_json::Value>,
    pub ollama_client: Arc<crate::inference::OllamaClient>,
    pub current_model: String,
}

/// MCP-style agent trait
#[async_trait::async_trait]
pub trait Agent: Send + Sync {
    /// Get the agent's name
    fn name(&self) -> &str;

    /// Get the agent's capabilities
    async fn capabilities(&self) -> Vec<Capability>;

    /// Check if the agent can handle this request (0.0 = can't handle, 1.0 = perfect match)
    async fn can_handle(&self, request: &AgentRequest) -> f32;

    /// Execute the agent's functionality
    async fn execute(&self, request: AgentRequest) -> Result<AgentResponse>;

    /// Called when the agent should clean up resources
    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }
}

/// Agent registry for managing multiple agents
pub struct AgentRegistry {
    agents: HashMap<String, Box<dyn Agent>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Register a new agent
    pub fn register(&mut self, agent: Box<dyn Agent>) {
        let name = agent.name().to_string();
        self.agents.insert(name, agent);
    }

    /// Find the best agent to handle a request
    pub async fn find_best_agent(&self, request: &AgentRequest) -> Option<&dyn Agent> {
        let mut best_agent = None;
        let mut best_score = 0.0;

        for agent in self.agents.values() {
            let score = agent.can_handle(request).await;
            if score > best_score {
                best_score = score;
                best_agent = Some(agent.as_ref());
            }
        }

        if best_score > 0.5 { // Minimum confidence threshold
            best_agent
        } else {
            None
        }
    }

    /// Get all available agents
    pub fn get_agents(&self) -> &HashMap<String, Box<dyn Agent>> {
        &self.agents
    }

    /// Get agent by name
    pub fn get_agent(&self, name: &str) -> Option<&dyn Agent> {
        self.agents.get(name).map(|a| a.as_ref())
    }
}

/// Agent coordinator for orchestrating multiple agents
pub struct AgentCoordinator {
    registry: AgentRegistry,
    context: AgentContext,
}

impl AgentCoordinator {
    pub fn new(context: AgentContext) -> Self {
        Self {
            registry: AgentRegistry::new(),
            context,
        }
    }

    /// Register an agent with the coordinator
    pub fn register_agent(&mut self, agent: Box<dyn Agent>) {
        self.registry.register(agent);
    }

    /// Process a user input through the agent system
    pub async fn process_input(&mut self, input: &str) -> Result<CoordinatorResponse> {
        // Update context with new input
        self.context.user_input = input.to_string();

        // Create agent request
        let request = AgentRequest {
            input: input.to_string(),
            context: self.context.clone(),
            parameters: HashMap::new(),
        };

        // Find best agent to handle the request
        if let Some(agent) = self.registry.find_best_agent(&request).await {
            let start_time = std::time::Instant::now();
            let response = agent.execute(request).await?;
            let processing_time = start_time.elapsed().as_millis() as u64;

            // Update shared resources
            if !response.resources_used.is_empty() {
                for resource in &response.resources_used {
                    self.context.shared_resources.insert(
                        resource.clone(),
                        serde_json::Value::String(response.content.clone())
                    );
                }
            }

            Ok(CoordinatorResponse {
                content: response.content,
                agent_used: response.metadata.agent_name,
                confidence: response.metadata.confidence,
                processing_time_ms: processing_time,
                sources: response.metadata.sources,
                has_additional_context: !response.resources_used.is_empty(),
            })
        } else {
            Ok(CoordinatorResponse {
                content: "No agent available to handle this request".to_string(),
                agent_used: "none".to_string(),
                confidence: 0.0,
                processing_time_ms: 0,
                sources: vec![],
                has_additional_context: false,
            })
        }
    }

    /// Get all capabilities from registered agents
    pub async fn get_all_capabilities(&self) -> HashMap<String, Vec<Capability>> {
        let mut all_capabilities = HashMap::new();

        for (name, agent) in self.registry.get_agents() {
            let capabilities = agent.capabilities().await;
            all_capabilities.insert(name.clone(), capabilities);
        }

        all_capabilities
    }

    /// Update the shared context
    pub fn update_context(&mut self, updates: HashMap<String, serde_json::Value>) {
        for (key, value) in updates {
            self.context.shared_resources.insert(key, value);
        }
    }
}

/// Response from the agent coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorResponse {
    pub content: String,
    pub agent_used: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub sources: Vec<String>,
    pub has_additional_context: bool,
}