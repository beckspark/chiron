use chiron::dialogue::therapeutic::TherapyPhase;
use chiron::Result;
use clap::{Arg, Command};
use std::io::Write;
use std::sync::Arc;
use tokio::signal;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Only initialize tracing if RUST_LOG is set (for debugging)
    if std::env::var("RUST_LOG").is_ok() {
        tracing_subscriber::fmt::init();
    }

    let matches = Command::new("chiron")
        .version("0.1.0")
        .about("Mental Health SLM Coaching System")
        .arg(
            Arg::new("model")
                .long("model")
                .value_name("MODEL")
                .help("Ollama model to use (default: gemma3n:e4b)")
                .default_value("gemma3n:e4b"),
        )
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("Ollama server host")
                .default_value("http://localhost:11434"),
        )
        .arg(
            Arg::new("resume")
                .long("resume")
                .value_name("SESSION_ID")
                .help("Resume a previous session by session ID"),
        )
        .arg(
            Arg::new("list-sessions")
                .long("list-sessions")
                .help("List all previous sessions")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("export-training")
                .long("export-training")
                .value_name("OUTPUT_FILE")
                .help("Export all sessions as training data in JSONL format"),
        )
        .arg(
            Arg::new("mock")
                .long("mock")
                .help("Use mock responses instead of Ollama (for testing)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no-save")
                .long("no-save")
                .help("Don't save or load session data (temporary session)")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let model = matches.get_one::<String>("model").unwrap();
    let host = matches.get_one::<String>("host").unwrap();
    let no_save = matches.get_flag("no-save");

    // Initialize session storage (skip if no-save flag is set)
    let session_storage = if no_save {
        None
    } else {
        Some(chiron::dialogue::session::SessionStorage::new()?)
    };

    // Handle special commands
    if matches.get_flag("list-sessions") {
        if no_save {
            println!("Cannot list sessions when --no-save flag is used");
            return Ok(());
        }
        return list_sessions(session_storage.as_ref().unwrap()).await;
    }

    if let Some(output_file) = matches.get_one::<String>("export-training") {
        if no_save {
            println!("Cannot export training data when --no-save flag is used");
            return Ok(());
        }
        return export_training_data(session_storage.as_ref().unwrap(), output_file).await;
    }

    println!("Chiron Mental Health SLM System");
    println!("Type 'quit' to exit\n");

    // Initialize Ollama client
    let ollama_client = Arc::new(chiron::inference::OllamaClient::new(host.clone()));
    let use_mock = matches.get_flag("mock");

    // Test connection (skip if using mock)
    if !use_mock {
        print!("🔌 Connecting to Ollama...");
        std::io::stdout().flush().unwrap();

        match test_ollama_connection(&ollama_client, model).await {
            Ok(_) => {
                println!(" ✅");
            }
            Err(e) => {
                println!(" ❌");
                eprintln!("Failed to connect to Ollama: {}", e);
                eprintln!(
                    "Make sure Ollama is running and the model '{}' is available",
                    model
                );
                eprintln!("Alternatively, use --mock flag for testing without Ollama");
                return Err(e);
            }
        }
    } else {
        println!("🤖 Using mock mode for testing\n");
    }

    // Handle session resumption or create new session
    let mut session = if let Some(session_id_str) = matches.get_one::<String>("resume") {
        if no_save {
            println!("Cannot resume sessions when --no-save flag is used");
            println!("Starting new temporary session instead...");
            chiron::dialogue::DialogueSession::new()
        } else {
            match session_id_str.parse::<uuid::Uuid>() {
                Ok(session_id) => match session_storage.as_ref().unwrap().load_session(session_id).await {
                    Ok(session) => {
                        println!("📂 Resuming session: {}", session.get_therapeutic_summary());
                        session
                    }
                    Err(e) => {
                        eprintln!("Failed to load session {}: {}", session_id, e);
                        println!("Starting new session instead...");
                        chiron::dialogue::DialogueSession::new()
                    }
                },
                Err(_) => {
                    eprintln!("Invalid session ID format: {}", session_id_str);
                    println!("Starting new session instead...");
                    chiron::dialogue::DialogueSession::new()
                }
            }
        }
    } else {
        chiron::dialogue::DialogueSession::new()
    };

    if no_save {
        println!("🆔 Session ID: {} (temporary session - not saved)", session.id);
    } else {
        println!(
            "🆔 Session ID: {} (use --resume {} to continue later)",
            session.id, session.id
        );
    }

    // Setup cleanup handler for graceful exits
    let cleanup_client = ollama_client.clone();
    let cleanup_model = model.to_string();
    let cleanup_use_mock = use_mock;

    // Start interactive chat loop with cleanup
    let chat_result = tokio::select! {
        result = start_chat_loop(
            ollama_client.clone(),
            model,
            &mut session,
            session_storage.as_ref(),
            use_mock,
        ) => result,
        _ = signal::ctrl_c() => {
            println!("\n🛑 Received interrupt signal...");
            cleanup_on_exit(&cleanup_client, &cleanup_model, cleanup_use_mock).await;
            Ok(())
        }
    };

    // Always attempt cleanup on normal exit
    cleanup_on_exit(&cleanup_client, &cleanup_model, use_mock).await;

    chat_result?;

    Ok(())
}

async fn cleanup_on_exit(
    client: &Arc<chiron::inference::OllamaClient>,
    model: &str,
    use_mock: bool,
) {
    if !use_mock {
        print!("🧹 Cleaning up model resources...");
        std::io::stdout().flush().unwrap();

        if let Err(_) = client.unload_model(model).await {
            // Silently ignore cleanup errors
        }

        println!(" ✅");
    }
}

async fn test_ollama_connection(
    client: &chiron::inference::OllamaClient,
    model: &str,
) -> Result<()> {
    let _response = client.generate(model, "Hello").await?;
    Ok(())
}

async fn start_chat_loop(
    client: Arc<chiron::inference::OllamaClient>,
    model: &str,
    session: &mut chiron::dialogue::DialogueSession,
    storage: Option<&chiron::dialogue::session::SessionStorage>,
    use_mock: bool,
) -> Result<()> {
    use std::io::{self, Write};

    // Initialize safety systems
    let crisis_detector = chiron::safety::CrisisDetector::new();
    let safety_filters = chiron::safety::SafetyFilters::new();

    // Get therapeutic context from session
    let mut therapeutic_context = chiron::dialogue::TherapeuticContext::new();
    therapeutic_context.phase = match session.therapeutic_metadata.therapy_phase.as_str() {
        "assessment" => TherapyPhase::Assessment,
        "initial" => TherapyPhase::Initial,
        "middle" => TherapyPhase::Middle,
        "termination" => TherapyPhase::Termination,
        _ => TherapyPhase::Assessment,
    };
    therapeutic_context.session_count = session.therapeutic_metadata.session_count;

    println!("⚠️  IMPORTANT: I am an AI assistant, not a mental health professional.");
    println!("For immediate crisis support, contact:");
    println!("• National Suicide Prevention Lifeline: 988");
    println!("• Crisis Text Line: Text HOME to 741741");
    println!("• Emergency Services: 911\n");

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                // EOF reached (e.g., Ctrl+D or timeout), exit gracefully
                println!("\nSession ended.");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.to_lowercase() == "quit" {
            println!("Goodbye! Take care of yourself.");
            break;
        }

        // Crisis detection check
        if crisis_detector.detect_crisis(input)? {
            println!("\n🚨 I'm concerned about what you've shared. Your safety is important.");
            println!("Please reach out for immediate help:");
            println!("• National Suicide Prevention Lifeline: 988");
            println!("• Crisis Text Line: Text HOME to 741741");
            println!("• Emergency Services: 911");
            println!("• Or go to your nearest emergency room");
            println!("\nWould you like to continue our conversation? (yes/no)");

            let mut crisis_response = String::new();
            match io::stdin().read_line(&mut crisis_response) {
                Ok(0) => {
                    // EOF reached, treat as exit
                    println!("\nPlease prioritize getting professional help. Take care.");
                    break;
                }
                Ok(_) => {
                    if crisis_response.trim().to_lowercase().starts_with('n') {
                        println!("Please prioritize getting professional help. Take care.");
                        break;
                    }
                }
                Err(_) => {
                    println!("Please prioritize getting professional help. Take care.");
                    break;
                }
            }
            continue;
        }

        // Filter and process input
        let filtered_input = safety_filters.filter_input(input)?;

        // Add user message to session with metadata
        let crisis_indicators = if crisis_detector.detect_crisis(input)? {
            vec!["crisis_detected".to_string()]
        } else {
            vec![]
        };

        session.add_message_with_metadata(
            chiron::dialogue::session::Role::User,
            filtered_input.clone(),
            vec![], // TODO: Add therapeutic tagging
            None,   // TODO: Add sentiment analysis
            crisis_indicators,
        );

        // Build therapeutic prompt with context
        let context = session.get_context()?;
        let therapeutic_prompt = format!(
            "You are Chiron, a supportive AI companion focused on mental wellness.
            You provide empathetic listening and gentle guidance but never give medical advice or diagnoses.
            Always remind users you're not a replacement for professional mental health care.

            Current therapy phase: {:?}
            Session count: {}

            Conversation context:
            {}

            Respond empathetically to the most recent user message.",
            therapeutic_context.phase,
            therapeutic_context.session_count,
            context
        );

        let (response, already_printed) = if use_mock {
            print!("Chiron: ");
            io::stdout().flush().unwrap();
            (generate_mock_response(&filtered_input, &therapeutic_context), false)
        } else {
            // Use streaming for real-time progress (already prints)
            (client.generate_with_progress(model, &therapeutic_prompt, true).await?, true)
        };

        let filtered_response = safety_filters.filter_output(&response)?;

        if !already_printed {
            // Format response with proper line wrapping and indentation for mock mode
            let wrapped_response = wrap_text(&filtered_response, 80, "");
            println!("{}\n", wrapped_response);
        } else {
            // Just add spacing after streaming response
            println!();
        }

        // Add assistant response to session
        session.add_message_with_metadata(
            chiron::dialogue::session::Role::Assistant,
            filtered_response,
            vec![], // TODO: Add therapeutic tagging for AI responses
            None,   // TODO: Add quality scoring
            vec![],
        );

        // Save session periodically (skip if no storage)
        if let Some(storage) = storage {
            if session.messages.len() % 4 == 0 {
                if let Err(e) = storage.save_session(session).await {
                    eprintln!("Warning: Failed to save session: {}", e);
                }
            }
        }

        // Update therapeutic context
        session.therapeutic_metadata.session_count = therapeutic_context.session_count;
        session.therapeutic_metadata.therapy_phase = format!("{:?}", therapeutic_context.phase);

        // Increment session count periodically
        if session.messages.len() % 10 == 0 {
            therapeutic_context.increment_session();
        }
    }

    // Final session save (skip if no storage)
    if let Some(storage) = storage {
        if let Err(e) = storage.save_session(session).await {
            eprintln!("Warning: Failed to save final session: {}", e);
        } else {
            println!(
                "💾 Session saved. Use --resume {} to continue later.",
                session.id
            );
        }
    } else {
        println!("🗑️  Session not saved (temporary session)");
    }

    Ok(())
}

async fn list_sessions(storage: &chiron::dialogue::session::SessionStorage) -> Result<()> {
    let sessions = storage.list_sessions().await?;

    if sessions.is_empty() {
        println!("No previous sessions found.");
        return Ok(());
    }

    println!("📋 Previous Sessions:");
    println!(
        "{:<38} {:<12} {:<8} {:<15} {}",
        "Session ID", "Phase", "Messages", "Last Updated", "Preview"
    );
    println!("{}", "-".repeat(100));

    for session in sessions {
        println!(
            "{:<38} {:<12} {:<8} {:<15} {}",
            session.id.to_string(),
            session.therapy_phase,
            session.message_count,
            session.last_updated.format("%Y-%m-%d %H:%M"),
            session.preview
        );
    }

    Ok(())
}

async fn export_training_data(
    storage: &chiron::dialogue::session::SessionStorage,
    output_file: &str,
) -> Result<()> {
    let output_path = std::path::PathBuf::from(output_file);

    storage.export_training_jsonl(&output_path).await?;

    let training_data = storage.export_training_data().await?;
    println!(
        "📊 Exported {} training examples to {}",
        training_data.len(),
        output_file
    );

    Ok(())
}

fn wrap_text(text: &str, width: usize, prefix: &str) -> String {
    let mut wrapped = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            wrapped.push('\n');
            continue;
        }

        let mut current_line = String::from(prefix);
        let words: Vec<&str> = line.split_whitespace().collect();

        for word in words {
            if current_line.len() + word.len() + 1 > width && !current_line.trim_end().is_empty() {
                wrapped.push_str(&current_line);
                wrapped.push('\n');
                current_line = String::from(prefix);
            }

            if !current_line.trim_end().is_empty() {
                current_line.push(' ');
            }
            current_line.push_str(word);
        }

        if !current_line.trim().is_empty() {
            wrapped.push_str(&current_line);
            wrapped.push('\n');
        }
    }

    wrapped.trim_end().to_string()
}

fn generate_mock_response(input: &str, context: &chiron::dialogue::TherapeuticContext) -> String {
    let input_lower = input.to_lowercase();

    // Simple therapeutic responses based on input patterns
    if input_lower.contains("anxious") || input_lower.contains("anxiety") {
        "I hear that you're feeling anxious. That's a very common experience, and it's important to acknowledge these feelings. Can you tell me more about what might be contributing to your anxiety right now? Sometimes talking through our worries can help us understand them better.".to_string()
    } else if input_lower.contains("sad")
        || input_lower.contains("depressed")
        || input_lower.contains("down")
    {
        "Thank you for sharing that you're feeling down. It takes courage to acknowledge difficult emotions. Remember that these feelings are temporary, even when they feel overwhelming. What has helped you cope with similar feelings in the past?".to_string()
    } else if input_lower.contains("stress") || input_lower.contains("overwhelmed") {
        "Feeling stressed and overwhelmed is something many people experience. Let's take a moment to acknowledge that you're dealing with a lot right now. What's one small thing that might help you feel a bit more in control today?".to_string()
    } else if input_lower.contains("sleep") || input_lower.contains("tired") {
        "Sleep difficulties can really impact how we feel during the day. Good sleep hygiene is an important part of mental wellness. Have you noticed any patterns in what helps or hinders your sleep?".to_string()
    } else if input_lower.contains("thanks")
        || input_lower.contains("thank")
        || input_lower.contains("helpful")
    {
        "I'm glad you're finding our conversation helpful. Remember, the work you're doing to understand yourself and develop coping strategies is really important. How are you feeling about continuing to explore these topics?".to_string()
    } else if input_lower.contains("work") || input_lower.contains("job") {
        "Work-related stress is very common. It sounds like your professional life is weighing on you. Can you help me understand what specific aspects of work are most challenging for you right now?".to_string()
    } else if input_lower.contains("family") || input_lower.contains("relationship") {
        "Relationships can be both a source of support and stress. It sounds like there's something important happening in your relationships. Would you like to share more about what's on your mind?".to_string()
    } else if context.session_count == 0 {
        "Hello, and welcome to our session. I'm here to listen and support you as we explore what's on your mind. This is a safe space where you can share your thoughts and feelings. What would you like to talk about today?".to_string()
    } else {
        "I appreciate you sharing that with me. It sounds like there's a lot on your mind. Can you help me understand more about what you're experiencing? Remember, there's no rush - we can take this at whatever pace feels comfortable for you.".to_string()
    }
}
