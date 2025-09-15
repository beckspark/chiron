# Chiron Mental Health SLM Project

## Bash Commands
- `cargo build`: Build the project
- `cargo run`: Run the main application
- `cargo test`: Run all tests
- `cargo clippy`: Run linter
- `cargo fmt`: Format code
- `ollama serve`: Start Ollama server (required for inference)
- `ollama pull gemma3n:e4b`: Download Gemma 3N E4B model

## Architecture
- Local SLM inference via Ollama (gemma3n:e4b or llama3.2:1b)
- Modular agent system: Intake → Assessment → Intervention → Monitoring
- Safety-first design with crisis detection at every layer
- No cloud dependencies, fully local processing

## Project Structure
```
src/
  agents/     - Agent implementations (intake, assessment, intervention, monitoring)
  inference/  - Ollama client and model interface
  safety/     - Crisis detection and safety filters
  dialogue/   - Therapeutic conversation management
  main.rs     - CLI application entry point
```

## Technical Stack
- Ollama for local SLM inference
- reqwest for HTTP client to Ollama API
- tokio for async runtime
- serde for JSON serialization
- clap for CLI interface

## Development Status

### Completed
- Mental health SLM requirements analysis
- Core architecture design
- Ollama integration architecture

### In Progress
- Project tracking documentation

### Pending
- Foundational Rust dependencies setup
- Ollama client implementation
- Safety architecture with crisis detection
- Therapeutic dialogue management

## Safety Requirements
IMPORTANT: All implementations must prioritize safety over functionality
- Crisis detection at all agent levels
- Human handoff protocols
- Explicit AI limitation communication
- Never provide medical advice or diagnoses

## Code Style
- Use standard Rust formatting (cargo fmt)
- Implement proper error handling with Result types
- Use async/await for all I/O operations
- Document all public APIs