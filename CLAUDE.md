# Chiron Mental Health SLM Project

IMPORTANT: Review Claude Code best practices at https://www.anthropic.com/engineering/claude-code-best-practices before making changes to this project.

## Bash Commands
- `cargo build`: Build the project
- `cargo run`: Run the main application
- `cargo test`: Run all tests
- `cargo clippy`: Run linter
- `cargo fmt`: Format code
- `cargo run -- --mock`: Test without Ollama
- `cargo run -- --no-save`: Run without saving session data (temporary session)
- `cargo run -- --list-sessions`: View saved sessions
- `cargo run -- --resume <SESSION_ID>`: Continue previous session
- `cargo run -- --export-training <FILE>`: Export training data

## Architecture
- Local SLM inference via Ollama (gemma3n:e4b or llama3.2:1b)
- Real-time streaming responses with progress indicators
- Modular agent system: Intake → Assessment → Intervention → Monitoring
- Safety-first design with crisis detection
- Session persistence with training data export for model fine-tuning

## Development Status

### Completed
- Ollama client implementation with async interface
- Session management with persistence and training data export
- Basic safety systems with crisis detection
- Mock mode for testing

### In Progress
- Enhanced safety systems with sophisticated crisis detection
- Modular agent system (intake, assessment, intervention, monitoring)

### Pending
- Comprehensive testing framework
- Logging and monitoring

## Code Style
- Use standard Rust formatting (cargo fmt)
- Proper error handling with Result types
- Async/await for all I/O operations