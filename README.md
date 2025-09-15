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

## Current State

Just getting started. Have some Rust boilerplate and ideas about connecting to Ollama models.

```bash
# If you want to mess around with it:
ollama serve
ollama pull gemma3n:e4b
cargo run
```

## Disclaimer

This is just an experiment! Don't use it for actual mental health crises or as a replacement for real therapy.