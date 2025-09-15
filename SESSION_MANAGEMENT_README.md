# Chiron Session Management & Training Data System

## Overview

Chiron now includes comprehensive session management designed for therapeutic conversations with built-in support for future model training and RAG (Retrieval Augmented Generation) capabilities.

## Key Features

### 🗄️ **Session Persistence**
- **Structured Storage**: Sessions saved as JSON with rich metadata
- **Automatic Backups**: Sessions saved every 4 messages
- **Local Storage**: Uses system data directory (`~/.local/share/chiron/sessions/`)
- **Privacy First**: All data stays local

### 🔄 **Session Resumption**
```bash
# Start a new session
cargo run -- --mock

# Resume a specific session
cargo run -- --resume <SESSION_ID> --mock

# List all previous sessions
cargo run -- --list-sessions
```

### 📊 **Training Data Export**
```bash
# Export all conversations as JSONL for model training
cargo run -- --export-training training_data.jsonl
```

### 🧠 **Therapeutic Metadata**
Each session captures:
- **Primary Concerns**: Main therapeutic themes (anxiety, depression, etc.)
- **Intervention Techniques**: CBT, mindfulness, coping strategies
- **Progress Indicators**: Quantified therapeutic progress over time
- **Session Quality**: Alliance scores, conversation coherence
- **Crisis Indicators**: Safety-relevant patterns detected

### 🤖 **Mock Mode for Testing**
Test the complete system without Ollama:
```bash
cargo run -- --mock
```

## Session Data Structure

### Core Session
```json
{
  "id": "uuid-v4",
  "created_at": "2024-12-15T10:30:00Z",
  "last_updated": "2024-12-15T11:15:00Z",
  "messages": [...],
  "therapeutic_metadata": {...},
  "session_quality": {...}
}
```

### Training Example Format
```json
{
  "id": "uuid-v4",
  "session_id": "parent-session-uuid",
  "user_input": "I've been feeling really anxious lately",
  "assistant_response": "I hear that you're feeling anxious...",
  "therapeutic_context": {
    "primary_concerns": ["anxiety"],
    "intervention_techniques": ["validation", "exploration"],
    "therapy_phase": "assessment"
  },
  "quality_score": 8.5,
  "therapeutic_tags": ["anxiety", "validation"],
  "timestamp": "2024-12-15T10:35:00Z"
}
```

## Testing the System

### 1. **Basic Session Test**
```bash
# Start a mock session
cargo run -- --mock

# Sample conversation:
You: I've been feeling really anxious about work lately
Chiron: I hear that you're feeling anxious. That's a very common experience...

You: It's been affecting my sleep too
Chiron: Sleep difficulties can really impact how we feel during the day...

You: quit
# Session automatically saved
```

### 2. **Session Management Test**
```bash
# List sessions (after the above conversation)
cargo run -- --list-sessions
# Shows: Session ID, phase, message count, last updated, preview

# Resume the session
cargo run -- --resume <SESSION_ID> --mock
# Continues from where you left off
```

### 3. **Training Data Export Test**
```bash
# Export conversation data for model training
cargo run -- --export-training my_training_data.jsonl

# Check the output
cat my_training_data.jsonl
# Each line is a JSON training example
```

### 4. **Crisis Detection Test**
```bash
cargo run -- --mock

You: I've been thinking about hurting myself
# Triggers crisis protocol:
# 🚨 I'm concerned about what you've shared...
# • National Suicide Prevention Lifeline: 988
# etc.
```

## Future Training & RAG Capabilities

### Model Fine-tuning
The exported JSONL format is ready for:
- **OpenAI fine-tuning**: Direct compatibility with their format
- **Hugging Face training**: Standard conversation pairs
- **Custom training loops**: Structured therapeutic context

### RAG Integration
The session metadata enables:
- **Semantic Search**: Find similar therapeutic scenarios
- **Context Retrieval**: Pull relevant past conversations
- **Pattern Recognition**: Identify successful intervention techniques
- **Progress Tracking**: Compare therapeutic outcomes over time

### Example Training Pipeline
```bash
# 1. Collect sessions over time
cargo run -- --mock  # Multiple sessions

# 2. Export training data
cargo run -- --export-training therapeutic_conversations.jsonl

# 3. Use with fine-tuning
# - Upload to OpenAI for fine-tuning
# - Use with Hugging Face transformers
# - Feed into custom training pipeline

# 4. Evaluate improvements
# - Compare response quality scores
# - Track therapeutic alliance metrics
# - Measure crisis detection accuracy
```

## Privacy & Safety

### Privacy Protection
- **Local Only**: No cloud storage or transmission
- **Anonymized**: No personal identifiers stored
- **User Control**: Sessions can be deleted manually
- **Secure Storage**: JSON files in user data directory

### Safety Features
- **Crisis Detection**: Built-in keyword and pattern detection
- **Quality Scoring**: Track conversation quality over time
- **Safety Compliance**: Flag conversations that don't meet safety standards
- **Human Handoff**: Clear escalation protocols

## Development Notes

### Adding New Therapeutic Tags
Extend the `add_message_with_metadata` calls to include:
- **Therapeutic techniques**: "CBT", "mindfulness", "exposure_therapy"
- **Emotional states**: "anxious", "depressed", "hopeful", "breakthrough"
- **Session markers**: "first_session", "resistance", "engagement"

### Quality Metrics
Implement scoring for:
- **Therapeutic Alliance**: 1-10 scale based on user engagement
- **Response Appropriateness**: Contextual relevance
- **Safety Compliance**: Adherence to ethical guidelines
- **Conversation Flow**: Natural dialogue progression

### Future Enhancements
- **Embedding Storage**: Add vector embeddings for semantic search
- **Progress Visualization**: Charts showing therapeutic progress
- **Outcome Tracking**: Long-term follow-up and success metrics
- **Multi-modal Support**: Voice, text, and behavioral data integration