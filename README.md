# Chiron: Mental Health SLM Experiment

Playing around with Small Language Models (SLMs) for mental health and life coaching stuff.

## What's an SLM?

Small Language Models are trendy in late 2025. Check out [Microsoft's explanation](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-are-small-language-models), or a [recent Economist article](https://archive.ph/eL10C) that got me thinking about this, if you want the technical details.

## The Idea

I'm experimenting with training small models on mental health materials like:

- Motivational Interviewing techniques
- Psychology session transcripts (when I can find ethically sourced ones)
- CBT/DBT frameworks
- I'm a fan of Jungian stuff, so Analytical/archetypal/mythological content as well (hence the name, referring to the Greek "wounded healer" archetype)

Everything runs locally via Ollama - no sending your thoughts to the cloud and big tech to turn into ads/spy on you!

## Current Features

This thing actually works now! Here's what you get:

- **Real-time streaming responses** - See text appear as the AI generates it, just like ChatGPT
- **Session persistence** - Pick up conversations where you left off
- **Training data export** - Build datasets for fine-tuning your own models
- **Safety systems** - Crisis detection and content filtering
- **Clean interface** - No log spam, proper text formatting, progress indicators
- **Memory management** - Automatically cleans up Ollama resources when you exit

## Usage

```bash
# Basic usage:
ollama serve
ollama pull gemma3n:e4b  # or llama3.2:1b
cargo run

# Testing without Ollama:
cargo run -- --mock

# Temporary sessions (don't save anything):
cargo run -- --no-save

# Resume a previous conversation:
cargo run -- --list-sessions
cargo run -- --resume <SESSION_ID>

# Export training data:
cargo run -- --export-training training_data.jsonl
```

## Architecture

Simple and modular:

- **Ollama client** for local SLM inference
- **Session storage** with SQLite-like persistence
- **Safety systems** for crisis detection and content filtering
- **Therapeutic context** tracking for better conversations
- **Training data export** for model fine-tuning

## Disclaimer

This is an experiment! Don't use it for actual mental health crises or as a replacement for real therapy.
