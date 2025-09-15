use chiron::Result;
use clap::{Arg, Command};
use tracing::info;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

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
        .get_matches();

    let model = matches.get_one::<String>("model").unwrap();
    let host = matches.get_one::<String>("host").unwrap();

    info!("Starting Chiron Mental Health SLM System");
    info!("Model: {}, Host: {}", model, host);

    println!("Chiron Mental Health SLM System");
    println!("Type 'quit' to exit\n");

    // Initialize Ollama client
    let ollama_client = chiron::inference::OllamaClient::new(host.clone());

    // Test connection
    match test_ollama_connection(&ollama_client, model).await {
        Ok(_) => info!("Successfully connected to Ollama"),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}", e);
            eprintln!("Make sure Ollama is running and the model '{}' is available", model);
            return Err(e);
        }
    }

    // Start interactive chat loop
    start_chat_loop(ollama_client, model).await?;

    Ok(())
}

async fn test_ollama_connection(client: &chiron::inference::OllamaClient, model: &str) -> Result<()> {
    let response = client.generate(model, "Hello").await?;
    info!("Ollama test response: {}", response.chars().take(50).collect::<String>());
    Ok(())
}

async fn start_chat_loop(client: chiron::inference::OllamaClient, model: &str) -> Result<()> {
    use std::io::{self, Write};

    // Initialize safety systems
    let crisis_detector = chiron::safety::CrisisDetector::new();
    let safety_filters = chiron::safety::SafetyFilters::new();

    // Initialize dialogue session and therapeutic context
    let mut session = chiron::dialogue::DialogueSession::new();
    let mut therapeutic_context = chiron::dialogue::TherapeuticContext::new();

    println!("⚠️  IMPORTANT: I am an AI assistant, not a mental health professional.");
    println!("For immediate crisis support, contact:");
    println!("• National Suicide Prevention Lifeline: 988");
    println!("• Crisis Text Line: Text HOME to 741741");
    println!("• Emergency Services: 911\n");

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
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
            io::stdin().read_line(&mut crisis_response).unwrap();
            if crisis_response.trim().to_lowercase().starts_with('n') {
                println!("Please prioritize getting professional help. Take care.");
                break;
            }
            continue;
        }

        // Filter and process input
        let filtered_input = safety_filters.filter_input(input)?;

        // Add user message to session
        session.add_message(chiron::dialogue::session::Role::User, filtered_input.clone());

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

        print!("Chiron: ");
        io::stdout().flush().unwrap();

        match client.generate(model, &therapeutic_prompt).await {
            Ok(response) => {
                let filtered_response = safety_filters.filter_output(&response)?;
                println!("{}\n", filtered_response);

                // Add assistant response to session
                session.add_message(chiron::dialogue::session::Role::Assistant, filtered_response);

                // Increment session count periodically
                if session.messages.len() % 10 == 0 {
                    therapeutic_context.increment_session();
                }
            },
            Err(e) => eprintln!("Error: {}\n", e),
        }
    }

    Ok(())
}
