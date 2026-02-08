# Chiron: Multi-Agent MI Peer Support Chatbot

## Context

Chiron is a Rust inference runtime for an MI (Motivational Interviewing) peer support chatbot. Built on Candle (local GGUF inference) + Rig (agent framework). Pairs with Plotinus (Python training pipeline, separate repo).

Single Rust binary, everything in-process -- no HTTP, no Ollama. Rig's `CompletionModel` calls `ModelWeights::forward()` directly. sqlite-vec embedded, fastembed on CPU via ONNX.

**Completed**: Phase 1 (CandleProvider + Rig agent + CLI, `5cdcdb7`) and Phase 2 (true token streaming via mpsc, `a9fa3b4`).

**Goal**: Multi-agent orchestration in Rust via Rig.

---

## Architecture: 4-Step Sequential Pipeline

```
User Message
    │
    ▼
┌─────────────────────────────────────────────┐
│  STEP 1: RAG MEMORY (deterministic)         │
│                                              │
│  sqlite-vec semantic search:                 │
│  - MI principles & techniques                │
│  - Past session history                      │
│  - User goals & plans                        │
│  - Psychoeducation content                   │
│  - Crisis resources                          │
│                                              │
│  Feeds BOTH supervisors and peer coach       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 2: COACH SUPERVISOR (1B Llama)         │
│                                              │
│  Input:                                      │
│  - User message                              │
│  - Retrieved MI context (from RAG)           │
│  - Past session context (from RAG)           │
│  - Recent conversation history               │
│                                              │
│  Output:                                     │
│  1. Safety gate (crisis/ethics/off-topic)    │
│  2. MI state classification                  │
│     (engage/focus/evoke/plan)                │
│  3. Strategy guidance for the coach          │
│     ("User showing change talk about         │
│      exercise -- affirm and use complex      │
│      reflection to deepen")                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼ (if safe)
┌─────────────────────────────────────────────┐
│  STEP 3: PEER SUPPORT COACH (3B Llama)       │
│                                              │
│  Input:                                      │
│  - User message                              │
│  - Coach supervisor guidance                 │
│  - Retrieved context (MI principles,         │
│    session history, user goals)              │
│  - Conversation history                      │
│                                              │
│  Output:                                     │
│  - MI peer support response (OARS)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 4: SYSTEM SUPERVISOR                   │
│  (FunctionGemma 270M -- tool calling)        │
│                                              │
│  Input:                                      │
│  - User message + coach response             │
│  - Coach supervisor's classification         │
│  - Retrieved context (from RAG)              │
│                                              │
│  Tool calls (via Rig Tool trait):            │
│  - save_session_summary                      │
│  - update_user_goals                         │
│  - update_stage_of_change                    │
│  - save_coaching_plan                        │
│  - flag_for_followup                         │
│                                              │
│  Decides WHICH tools to call (if any)        │
│  based on conversation content               │
└──────────────────────────────────────────────┘
```

**Sequential**: RAG → Coach Supervisor → Peer Coach → System Supervisor. The coach supervisor needs RAG context to classify accurately. The system supervisor runs after the turn to decide what to persist.

---

## Agents In Detail

### 1. Coach Supervisor (1B -- Llama 3.2 1B Instruct)

Reviews every user message before the peer coach sees it. Outputs structured guidance, not conversation:

```
Input:  User message + RAG context + recent turns
Output: Structured (parsed from model output):
  - safety:    safe | crisis | off-topic | boundary-violation
  - mi_state:  engage | focus | evoke | plan
  - strategy:  "User expressed desire to change (DARN-D). Affirm
               their courage, use complex reflection to deepen."
  - crisis_action: (if crisis) specific referral text
```

Does NOT talk to the user. Talks to the peer coach. 1B Instruct can do this because output is constrained classification + short generation.

### 2. RAG Memory System (deterministic -- no LLM)

Embed user message via fastembed → search sqlite-vec across tables → return top-k results → format as context. Purely mechanical.

### 3. Peer Support Coach (3B -- Llama 3.2 3B Instruct → llam-mi fine-tune)

User-facing conversational agent. Receives:
- System preamble (MI peer support role, OARS techniques)
- Coach supervisor's strategy guidance (injected as coaching notes)
- Retrieved context (MI principles, session history, goals)
- Conversation history + current user message

Focuses on empathetic MI conversation. All meta-cognitive work offloaded to supervisor.

### 4. System Supervisor (FunctionGemma 270M)

Runs AFTER the peer coach responds. Sees full turn and decides what structured actions to take via tool calling.

**Why separate from coach supervisor:**
- Peer coach (MI fine-tune) can't do function calling
- Coach supervisor's job is MI expertise, not system operations
- FunctionGemma is purpose-built for function calling at 270M params (~150MB Q4_K_M)
- Clean separation: coaching intelligence vs system operations

**Tools (via Rig `Tool` trait):**
- `save_session_summary`, `update_user_goals`, `update_stage_of_change`, `save_coaching_plan`, `flag_for_followup`

Most turns trigger no tool calls. Only meaningful events (goal identified, commitment language, stage change) trigger persistence.

**Risk**: FunctionGemma uses Gemma architecture. Need to verify `candle-transformers` supports quantized Gemma GGUF (`quantized_gemma` module). Fallback: 1B Llama Instruct with tool-calling prompts.

---

## Model Setup

```
~/code/llms/gguf/                              # Shared model directory
├── Llama-3.2-1B-Instruct-Q4_K_M.gguf         # Coach Supervisor
├── Llama-3.2-3B-Instruct-Q4_K_M.gguf         # Peer Coach
└── functiongemma-270m-it-Q4_K_M.gguf         # System Supervisor

~/code/rust/chiron/
├── models/                                     # .gitignore'd, symlinks only
│   ├── coach-supervisor.gguf → ~/code/llms/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf
│   ├── coach.gguf            → ~/code/llms/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
│   └── system-supervisor.gguf → ~/code/llms/gguf/functiongemma-270m-it-Q4_K_M.gguf
```

**VRAM budget (all Q4_K_M):**
| Model | Params | VRAM |
|-------|--------|------|
| Peer Coach (3B) | 3B | ~2.0 GB |
| Coach Supervisor (1B) | 1B | ~0.8 GB |
| System Supervisor (270M) | 270M | ~0.15 GB |
| **Total** | | **~3.0 GB** |

Baseline with foundation Instruct models. Swap to llam-mi fine-tune via symlink when ready.

---

## Module Structure

```
src/
├── main.rs                  # CLI entry point, tokio runtime, chat loop
├── provider/
│   ├── mod.rs               # CandleProvider client + model registry
│   ├── completion.rs        # CompletionModel impl for Candle GGUF
│   └── config.rs            # Model configs (paths, generation params, EOS tokens)
├── agents/
│   ├── mod.rs               # Agent construction helpers
│   ├── peer.rs              # Peer Coach (3B) -- MI preamble, OARS
│   ├── coach_supervisor.rs  # Coach Supervisor (1B) -- safety/classify/strategy
│   └── system_supervisor.rs # System Supervisor (FunctionGemma 270M) -- tool calling
├── tools/
│   ├── mod.rs               # Rig Tool trait implementations
│   ├── goals.rs             # save/update user goals
│   ├── stage.rs             # update stage of change
│   ├── plans.rs             # save coaching plans
│   └── followup.rs          # flag topics for followup
├── orchestrator/
│   ├── mod.rs               # Turn pipeline: RAG → coach sup → peer → sys sup
│   └── session.rs           # Session state, conversation history management
├── memory/
│   ├── mod.rs               # sqlite-vec setup, schema migrations
│   ├── store.rs             # Vector search, CRUD operations
│   └── schema.sql           # Table definitions
└── knowledge/
    └── seed.rs              # Seed MI principles, resources from Plotinus data/
```

---

## Critical Design Decisions

### 1. Split Supervisor Architecture
- **Coach Supervisor** (1B Llama Instruct): MI domain -- classification, strategy, safety
- **Peer Coach** (3B → llam-mi): Conversation -- receives guidance as natural language
- **System Supervisor** (FunctionGemma 270M): Structured operations -- tool calling for persistence
- **RAG**: Deterministic (no LLM for retrieval)

### 2. Custom Candle Provider for Rig
`CompletionModel` trait wrapping Candle GGUF. Sync inference via `tokio::spawn_blocking`. Multiple models in shared `ModelRegistry`. Llama 3.2 Instruct chat template with `<|eot_id|>` (token 128009) as EOS -- wrong EOS causes multi-turn hallucination.

### 3. fastembed for Embeddings
`rig-fastembed` (ONNX, CPU) for embeddings. BAAI/bge-small-en-v1.5. Doesn't compete for GPU VRAM.

### 4. sqlite-vec for All Storage
`rig-sqlite` wraps sqlite-vec + rusqlite. Single embedded DB for vectors + relational data.

---

## Phased Implementation

### Phase 1: Candle Provider + Single Agent CLI Chat [DONE]
CandleCompletionModel, ModelRegistry, MI Chat Agent via AgentBuilder, multi-turn CLI.

### Phase 2: Token Streaming [DONE]
True token-by-token streaming via mpsc channel, diff-decode, KV cache.

### Phase 3: sqlite-vec Memory + RAG
1. sqlite-vec schema (chat history + MI knowledge + user state)
2. `rig-fastembed` for local embeddings
3. Seed MI knowledge from Plotinus `data/knowledge_base/`
4. RAG pipeline: embed → search → inject context into peer coach prompt
5. Save each turn with embedding

### Phase 4: Coach Supervisor + Orchestrator Pipeline
1. Load 1B Llama Instruct alongside 3B in `ModelRegistry`
2. Coach Supervisor with structured-output prompt (SAFETY/STATE/STRATEGY)
3. Pipeline: RAG → coach supervisor → peer coach
4. Parse into `CoachingGuidance` struct, inject into peer coach prompt
5. Session state management

### Phase 5: System Supervisor + Tool Calling
1. Verify Candle supports quantized Gemma GGUF (fallback: 1B Llama with tool prompts)
2. Load FunctionGemma 270M
3. Implement Rig `Tool` trait: SaveSessionSummary, UpdateUserGoals, UpdateStageOfChange, SaveCoachingPlan, FlagForFollowup
4. Wire into orchestrator post-response step
5. System supervisor sees full turn context, decides which tools to invoke

### Phase 6: Interface & Evaluation (Future)
- TUI with `ratatui` or web with axum + WebSocket
- MITI 4.2.1 evaluation integration

---

## Key Files from Plotinus

| File | Use In Chiron |
|------|---------------|
| `data/knowledge_base/mi_principles.json` | Seed `mi_knowledge` table |
| `data/knowledge_base/annomi_selected.json` | Seed `mi_knowledge` + exemplars |
| `data/raw/mi_text_summary.md` | Agent preamble content |
| `src/inference_config.py` | Translate system prompt to Rust |
| `INVARIANTS.md` | EOS token config (critical) |

---

## Verification

```bash
# Phase 3
cargo run -- --model models/coach.gguf --seed-knowledge ~/code/python/plotinus/data/knowledge_base/

# Phase 4
cargo run -- --coach-model models/coach.gguf --coach-supervisor-model models/coach-supervisor.gguf

# Phase 5
cargo run -- --coach-model models/coach.gguf \
             --coach-supervisor-model models/coach-supervisor.gguf \
             --system-supervisor-model models/system-supervisor.gguf
```

---

## Known Issues
- FunctionGemma Candle support needs verification (`quantized_gemma` module)
