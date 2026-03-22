# Chiron

Chiron is an open-source [Motivational Interviewing](https://en.wikipedia.org/wiki/Motivational_interviewing) peer support chatbot that runs entirely on your machine. No cloud API, no external inference service. The model compiles directly into the binary via [llama.cpp](https://llama-cpp.com/), so your conversations never leave your device.

The model *plotinus* is a [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B-Base) fine-tuned on MI techniques: reflective listening, open questions, affirmations, and the stages-of-change model. Chiron wraps it with a supervision layer that parses the model's internal reasoning, tracks stage of change state across turns, and adapts its coaching strategy in real time. It uses [Rig](https://github.com/0xPlaygrounds/rig) as its agent framework to orchestrate multi-turn conversations.

This project is experimental. The *plotinus* model is not publicly available yet. It needs more evaluation before it's appropriate for general use. That said, I'm actively looking to collaborate, especially with practitioners or students in the MI and coaching space. If you'd like to test it or contribute, reach out: <stephenpbeck@gmail.com>! :)

## Architecture

```
User Input
    |
    v
[Crisis Router] ──crisis──> Hardcoded safety response (988, Crisis Text Line)
    |
    | (non-crisis)
    v
[Case Notes DB] ──load──> Previous MI stage, strategy, running themes
    |
    v
[Preamble Builder] ── Assembles system prompt from:
    |                   - Base MI coaching prompt (coach.toml variant)
    |                   - Think block tag instructions
    |                   - Session context (case notes from prior turns)
    |                   - Stage-matched technique guidance (engage/focus/evoke/plan)
    |                   - Mode modifier (resistance/change-talk/ambivalence/crisis)
    |
    v
[Rig Agent] ── Streams completion via llama.cpp (Q4_K_M GGUF)
    |           Think blocks buffered; visible text streamed to terminal
    |
    v
[Think Parser] ── Extracts structured tags from <think> block:
    |               [MI-STAGE: focus]
    |               [STRATEGY: complex reflection]
    |               [TALK-TYPE: sustain talk]
    |               [THEMES: drinking, anxiety, sleep]
    |
    v
[Case Notes Update] ── Merges new themes with running theme set
    |                    Persists MI stage + strategy to SQLite
    |
    v
[Sliding Window] ── Trims chat history to 4 turns (context budget)
                     Case notes preserve accumulated context beyond window
```

### Model / Coach Supervision

The model produces structured metadata tags inside its `<think>` block before every response. Chiron parses these tags to build case notes, which feed back into the system prompt on the next turn. This creates a feedback loop:

1. Model reasons about MI stage, talk type, and themes
2. Chiron extracts that reasoning into persistent case notes
3. Next turn's system prompt includes those notes + stage-appropriate technique guidance
4. Model adapts its strategy based on the accumulated clinical picture

This lets a small local model (4B parameters, 2.5GB quantized) maintain coherent multi-turn coaching direction without external orchestration.

### Routing and mode detection

The `router` module handles crisis detection via keyword matching before any model inference runs -- immediate safety responses bypass the LLM entirely.

For non-crisis turns, the `peer` agent detects conversation modes (resistance, change-talk, ambivalence, engagement) from the case notes and injects mode-specific coaching modifiers into the preamble. This means the model gets different MI technique guidance depending on what it detected in the previous turn's think block.

## Modules

| Module | Purpose |
|--------|---------|
| `orchestrator` | Turn pipeline: crisis check -> case notes -> inference -> parse -> update |
| `agents/peer` | Preamble builder with stage guidance and mode detection |
| `provider/llamacpp` | Rig `CompletionModel` impl wrapping llama-cpp-2 |
| `supervision/think_parser` | Parses `[MI-STAGE]`, `[STRATEGY]`, `[TALK-TYPE]`, `[THEMES]` from think blocks |
| `memory/case_notes` | SQLite persistence for clinical state across turns |
| `router` | Pre-inference crisis keyword detection + safety responses |
| `catalog` | TOML-driven prompt variants and conversation mode definitions |

## Building

Default is CPU-only so `cargo build` works (slowly) everywhere. For best results, build with the proper feature flag for your machine's GPU:

```bash
# CPU-only (default)
cargo build --release

# NVIDIA CUDA
cargo build --release --features cuda

# Apple Metal
brew install cmake
brew install rust
cargo build --release --features metal

# AMD ROCm
cargo build --release --features rocm

# Vulkan
cargo build --release --features vulkan
```

## Usage

```bash
# Interactive conversation
cargo run --release --features cuda

# With a specific coach variant
cargo run --release --features cuda -- --coach-variant v7-unified

# Scripted test (outputs JSON)
cargo run --release --features cuda -- --script prompts/test_scripts/standard_5turn.toml --coach-variant v7-unified

# Benchmark single prompt
cargo run --release --features cuda -- --bench "I've been feeling really down lately"
```

Requires a GGUF model at `models/plotinus.gguf` (symlink to quantized export from Plotinus).

## Prompt Configuration

`prompts/coach.toml` defines prompt variants (base preamble + think block instructions). `prompts/modes.toml` defines conversation modes with coaching modifiers. Both are loaded at startup and can be swapped without rebuilding.

## Requirements

- Rust 2024 edition
- GPU toolkit matching your chosen feature flag (optional — CPU works without any)
- GGUF model file (exported from Plotinus training pipeline)
