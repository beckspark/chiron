use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use rig::OneOrMany;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_template::{self, ChatTemplate};
use super::{CandleProvider, ModelRegistry};

/// Response type for non-streaming completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleResponse {
    pub text: String,
    pub tokens_generated: usize,
}

/// Response type for streaming completions (final response object).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleStreamingResponse {
    pub tokens_generated: usize,
}

impl rig::completion::GetTokenUsage for CandleStreamingResponse {
    fn token_usage(&self) -> Option<Usage> {
        Some(Usage {
            input_tokens: 0,
            output_tokens: self.tokens_generated as u64,
            total_tokens: self.tokens_generated as u64,
            cached_input_tokens: 0,
        })
    }
}

/// A completion model backed by Candle GGUF inference.
///
/// Implements Rig's `CompletionModel` trait to run SmolLM3 3B via
/// quantized GGUF inference on CUDA (or CPU fallback).
#[derive(Clone)]
pub struct CandleCompletionModel {
    registry: Arc<Mutex<ModelRegistry>>,
    model_id: String,
}

impl CandleCompletionModel {
    /// Formats a completion request using the model's chat template.
    ///
    /// Acquires the registry lock briefly to access the template. Always
    /// suppresses thinking via `enable_thinking=false` in template options.
    fn format_with_template(&self, request: &CompletionRequest) -> String {
        let registry = self.registry.lock().expect("Registry lock poisoned");
        let model = registry
            .models
            .get(&self.model_id)
            .expect("Model not found in registry");
        format_request(request, &model.template)
    }
}

impl CompletionModel for CandleCompletionModel {
    type Response = CandleResponse;
    type StreamingResponse = CandleStreamingResponse;
    type Client = CandleProvider;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self {
            registry: client.registry.clone(),
            model_id: model.into(),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let registry = self.registry.clone();
        let model_id = self.model_id.clone();
        let prompt_text = self.format_with_template(&request);
        let max_tokens_override = request.max_tokens.map(|t| t as usize);

        let result = tokio::task::spawn_blocking(move || {
            run_inference(&registry, &model_id, &prompt_text, max_tokens_override)
        })
        .await
        .map_err(|e| CompletionError::ProviderError(format!("Task join error: {e}")))?
        .map_err(|e| CompletionError::ProviderError(format!("{e}")))?;

        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(&result.text)),
            usage: Usage {
                input_tokens: 0,
                output_tokens: result.tokens_generated as u64,
                total_tokens: result.tokens_generated as u64,
                cached_input_tokens: 0,
            },
            raw_response: result,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let registry = self.registry.clone();
        let model_id = self.model_id.clone();
        let prompt_text = self.format_with_template(&request);
        let max_tokens_override = request.max_tokens.map(|t| t as usize);

        let (tx, rx) = mpsc::channel::<Result<RawStreamingChoice<CandleStreamingResponse>, CompletionError>>(32);

        tokio::task::spawn_blocking(move || {
            let result = run_inference_streaming(&registry, &model_id, &prompt_text, &tx, max_tokens_override);
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(CompletionError::ProviderError(format!("{e}"))));
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
}

/// Formats a `CompletionRequest` into a prompt string using the model's chat template.
///
/// Combines the preamble (with RAG documents appended) and chat history into
/// template `Message`s, then renders via the HF Jinja2 template.
///
/// Always uses `enable_thinking=false` to structurally suppress reasoning,
/// causing the template to prefill an empty `<think>\n\n</think>` block.
fn format_request(request: &CompletionRequest, template: &ChatTemplate) -> String {
    let mut messages = Vec::new();

    // System message: preamble + RAG documents
    if let Some(preamble) = &request.preamble {
        let mut system_content = preamble.clone();
        if !request.documents.is_empty() {
            tracing::debug!(
                document_count = request.documents.len(),
                document_ids = ?request.documents.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
                "Injecting RAG context into prompt"
            );
            for doc in &request.documents {
                tracing::trace!(
                    doc_id = doc.id,
                    doc_text_len = doc.text.len(),
                    doc_text_preview = &doc.text[..doc.text.len().min(120)],
                    "RAG document"
                );
            }
            system_content.push_str("\n\n# Reference Context\n");
            for doc in &request.documents {
                system_content.push_str(&format!("{doc}\n"));
            }
        }
        messages.push(chat_template::Message::system(system_content));
    }

    // Chat history
    for message in request.chat_history.iter() {
        match message {
            Message::User { content } => {
                let text: String = content
                    .iter()
                    .filter_map(|c| match c {
                        rig::message::UserContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                messages.push(chat_template::Message::user(text));
            }
            Message::Assistant { content, .. } => {
                let text: String = content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                messages.push(chat_template::Message::assistant(text));
            }
        }
    }

    // Render with thinking suppressed (enable_thinking defaults to false)
    let options = chat_template::ChatTemplateOptions::for_generation();

    template
        .apply(&messages, &options)
        .expect("Chat template rendering failed")
}

/// Runs streaming inference, sending each decoded token through the channel.
///
/// Uses the diff-decode strategy: decodes all generated tokens so far and diffs
/// against the previous decode to get incremental text, avoiding UTF-8 boundary issues.
fn run_inference_streaming(
    registry: &Arc<Mutex<ModelRegistry>>,
    model_id: &str,
    prompt_text: &str,
    tx: &mpsc::Sender<Result<RawStreamingChoice<CandleStreamingResponse>, CompletionError>>,
    max_tokens_override: Option<usize>,
) -> Result<(), anyhow::Error> {
    let t0 = Instant::now();

    let mut registry = registry.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;

    let device = registry.device().clone();

    let model = registry
        .models
        .get_mut(model_id)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", model_id))?;

    let encoding = model
        .tokenizer
        .encode(prompt_text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let t1 = Instant::now();
    let prompt_token_count = prompt_tokens.len();
    tracing::info!(
        prompt_tokens = prompt_token_count,
        tokenization_ms = t1.duration_since(t0).as_millis() as u64,
        "Tokenized prompt"
    );

    let max_tokens = max_tokens_override.unwrap_or(model.config.max_tokens);
    let eos_token_ids = model.config.eos_token_ids.clone();
    let seed = model.config.seed.unwrap_or(42);
    let temperature = model.config.temperature;
    let top_p = model.config.top_p;

    let mut logits_processor = LogitsProcessor::new(
        seed,
        if temperature > 0.0 {
            Some(temperature)
        } else {
            None
        },
        if top_p < 1.0 { Some(top_p) } else { None },
    );

    // Full prefill — always reset KV cache and process all tokens in one batch.
    // Candle's causal mask doesn't support multi-token forward with existing cache,
    // and GPU batch prefill is fast enough that cache reuse offers no benefit.
    let prompt_tensor = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let logits = model.weights.forward(&prompt_tensor, 0)?;
    let logits = logits.squeeze(0)?;

    let t2 = Instant::now();
    let prefill_ms = t2.duration_since(t1).as_millis() as u64;
    let prefill_tok_per_sec = if prefill_ms > 0 {
        (prompt_token_count as f64 / prefill_ms as f64) * 1000.0
    } else {
        0.0
    };
    tracing::info!(
        prefill_ms,
        tokens_processed = prompt_token_count,
        prefill_tok_per_sec = format!("{prefill_tok_per_sec:.1}"),
        "Prefill complete"
    );

    // Sample first token
    let mut next_token = logits_processor.sample(&logits)?;
    let mut generated_tokens: Vec<u32> = vec![next_token];

    // Check if first token is already EOS
    if eos_token_ids.contains(&next_token) {
        tracing::info!("First token was EOS, no generation needed");
        let _ = tx.blocking_send(Ok(RawStreamingChoice::FinalResponse(
            CandleStreamingResponse {
                tokens_generated: 0,
            },
        )));
        return Ok(());
    }

    // Decode and send first token
    let mut prev_text = String::new();
    let decoded = model
        .tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    let incremental = &decoded[prev_text.len()..];
    if !incremental.is_empty() {
        if tx
            .blocking_send(Ok(RawStreamingChoice::Message(incremental.to_string())))
            .is_err()
        {
            // Receiver dropped, abort generation
            return Ok(());
        }
    }
    prev_text = decoded;

    // Autoregressive generation loop
    let decode_start = Instant::now();
    let total_prompt_len = prompt_tokens.len();
    for i in 1..max_tokens {
        let token_start = Instant::now();
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model
            .weights
            .forward(&input, total_prompt_len + i)?;
        let logits = logits.squeeze(0)?;

        next_token = logits_processor.sample(&logits)?;

        let token_ms = token_start.elapsed().as_millis() as u64;
        if i <= 5 {
            tracing::info!(token_index = i, token_ms, "Decoded token");
        }

        if eos_token_ids.contains(&next_token) {
            break;
        }

        generated_tokens.push(next_token);

        // Diff-decode: decode all tokens, diff against previous
        let decoded = model
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
        let incremental = &decoded[prev_text.len()..];
        if !incremental.is_empty() {
            if tx
                .blocking_send(Ok(RawStreamingChoice::Message(incremental.to_string())))
                .is_err()
            {
                return Ok(());
            }
        }
        prev_text = decoded;
    }

    let decode_elapsed = decode_start.elapsed();
    let total_elapsed = t0.elapsed();
    let tokens_generated = generated_tokens.len();
    let decode_tok_per_sec = if decode_elapsed.as_millis() > 0 {
        (tokens_generated as f64 / decode_elapsed.as_millis() as f64) * 1000.0
    } else {
        0.0
    };

    tracing::info!(
        tokens_generated,
        decode_ms = decode_elapsed.as_millis() as u64,
        decode_tok_per_sec = format!("{decode_tok_per_sec:.1}"),
        total_ms = total_elapsed.as_millis() as u64,
        prefill_ms,
        "Generation complete"
    );

    // Send final response
    let _ = tx.blocking_send(Ok(RawStreamingChoice::FinalResponse(
        CandleStreamingResponse { tokens_generated },
    )));

    Ok(())
}

/// Runs synchronous Candle inference (non-streaming). Called inside `spawn_blocking`.
///
/// Logs detailed timing information at each phase: tokenization, prefill,
/// per-token decode, and total generation summary.
fn run_inference(
    registry: &Arc<Mutex<ModelRegistry>>,
    model_id: &str,
    prompt_text: &str,
    max_tokens_override: Option<usize>,
) -> Result<CandleResponse, anyhow::Error> {
    let t0 = Instant::now();

    let mut registry = registry.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;

    // Clone device before mutable borrow of models
    let device = registry.device().clone();

    let model = registry
        .models
        .get_mut(model_id)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", model_id))?;

    let encoding = model
        .tokenizer
        .encode(prompt_text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let prompt_tokens = encoding.get_ids();

    let t1 = Instant::now();
    let prompt_token_count = prompt_tokens.len();
    tracing::info!(
        prompt_tokens = prompt_token_count,
        tokenization_ms = t1.duration_since(t0).as_millis() as u64,
        "Tokenized prompt"
    );

    let max_tokens = max_tokens_override.unwrap_or(model.config.max_tokens);
    let eos_token_ids = model.config.eos_token_ids.clone();
    let seed = model.config.seed.unwrap_or(42);
    let temperature = model.config.temperature;
    let top_p = model.config.top_p;

    let mut logits_processor = LogitsProcessor::new(
        seed,
        if temperature > 0.0 {
            Some(temperature)
        } else {
            None
        },
        if top_p < 1.0 { Some(top_p) } else { None },
    );

    // Process all prompt tokens at once (prefill)
    let prompt_tensor = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.weights.forward(&prompt_tensor, 0)?;
    let logits = logits.squeeze(0)?;

    let t2 = Instant::now();
    let prefill_ms = t2.duration_since(t1).as_millis() as u64;
    let prefill_tok_per_sec = if prefill_ms > 0 {
        (prompt_token_count as f64 / prefill_ms as f64) * 1000.0
    } else {
        0.0
    };
    tracing::info!(
        prefill_ms,
        prefill_tok_per_sec = format!("{prefill_tok_per_sec:.1}"),
        "Prefill complete"
    );

    // Sample first token
    let mut next_token = logits_processor.sample(&logits)?;
    let mut generated_tokens: Vec<u32> = vec![next_token];

    // Check if first token is already EOS
    if eos_token_ids.contains(&next_token) {
        let text = model
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
        tracing::info!("First token was EOS, no generation needed");
        return Ok(CandleResponse {
            text,
            tokens_generated: 0,
        });
    }

    // Autoregressive generation loop
    let decode_start = Instant::now();
    for i in 1..max_tokens {
        let token_start = Instant::now();
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model
            .weights
            .forward(&input, prompt_tokens.len() + i)?;
        let logits = logits.squeeze(0)?;

        next_token = logits_processor.sample(&logits)?;

        let token_ms = token_start.elapsed().as_millis() as u64;
        if i <= 5 {
            tracing::info!(token_index = i, token_ms, "Decoded token");
        }

        if eos_token_ids.contains(&next_token) {
            break;
        }

        generated_tokens.push(next_token);
    }

    let decode_elapsed = decode_start.elapsed();
    let total_elapsed = t0.elapsed();
    let tokens_generated = generated_tokens.len();
    let decode_tok_per_sec = if decode_elapsed.as_millis() > 0 {
        (tokens_generated as f64 / decode_elapsed.as_millis() as f64) * 1000.0
    } else {
        0.0
    };

    tracing::info!(
        tokens_generated,
        decode_ms = decode_elapsed.as_millis() as u64,
        decode_tok_per_sec = format!("{decode_tok_per_sec:.1}"),
        total_ms = total_elapsed.as_millis() as u64,
        prefill_ms,
        "Generation complete"
    );

    let text = model
        .tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;

    Ok(CandleResponse {
        text,
        tokens_generated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::completion::Document;
    use rig::OneOrMany;

    #[test]
    fn test_format_request_simple() {
        let template = ChatTemplate::chatml_with_thinking();
        let request = CompletionRequest {
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(Message::user("Hello")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let formatted = format_request(&request, &template);

        assert!(formatted.contains("<|im_start|>system\n"));
        assert!(formatted.contains("You are a helpful assistant."));
        assert!(formatted.contains("<|im_end|>"));
        assert!(formatted.contains("<|im_start|>user\n"));
        assert!(formatted.contains("Hello"));
        // Always suppresses thinking
        assert!(
            formatted.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "Should prefill empty think block, got: ...{}",
            &formatted[formatted.len().saturating_sub(80)..],
        );
    }

    #[test]
    fn test_format_request_with_history() {
        let template = ChatTemplate::chatml_with_thinking();
        let request = CompletionRequest {
            preamble: Some("System prompt".to_string()),
            chat_history: OneOrMany::many(vec![
                Message::user("First message"),
                Message::assistant("First response"),
                Message::user("Second message"),
            ])
            .unwrap(),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let formatted = format_request(&request, &template);

        assert!(formatted.contains("First message"));
        assert!(formatted.contains("First response"));
        assert!(formatted.contains("Second message"));
        assert!(formatted.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_format_request_with_rag_documents() {
        let template = ChatTemplate::chatml_with_thinking();
        let request = CompletionRequest {
            preamble: Some("You are a peer coach.".to_string()),
            chat_history: OneOrMany::one(Message::user("I want to stop drinking")),
            documents: vec![
                Document {
                    id: "oars_reflections".into(),
                    text: "OARS - Reflections\nSimple reflections repeat back what the person said."
                        .into(),
                    additional_props: [("category".into(), "oars".into())].into(),
                },
                Document {
                    id: "change_talk_prep_desire".into(),
                    text: "Change Talk - Desire\nStatements about wanting change.".into(),
                    additional_props: [("category".into(), "change_talk".into())].into(),
                },
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let formatted = format_request(&request, &template);

        // RAG context inside system block
        assert!(formatted.contains("# Reference Context"));
        assert!(formatted.contains("oars_reflections"));
        assert!(formatted.contains("OARS - Reflections"));
        assert!(formatted.contains("change_talk_prep_desire"));
        assert!(formatted.contains("Change Talk - Desire"));

        // Context appears between system im_start and im_end
        let system_start = formatted.find("<|im_start|>system").unwrap();
        let system_end = formatted[system_start..].find("<|im_end|>").unwrap() + system_start;
        let ref_context_pos = formatted.find("# Reference Context").unwrap();
        assert!(
            ref_context_pos > system_start && ref_context_pos < system_end,
            "RAG context must be inside system block"
        );

        assert!(formatted.contains("I want to stop drinking"));
    }

    #[test]
    fn test_format_request_no_documents_no_context_section() {
        let template = ChatTemplate::chatml_with_thinking();
        let request = CompletionRequest {
            preamble: Some("System prompt".to_string()),
            chat_history: OneOrMany::one(Message::user("Hello")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let formatted = format_request(&request, &template);
        assert!(
            !formatted.contains("# Reference Context"),
            "No context section when documents are empty"
        );
    }

    #[test]
    fn test_format_request_no_preamble() {
        let template = ChatTemplate::chatml_with_thinking();
        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(Message::user("Hello")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let formatted = format_request(&request, &template);

        assert!(!formatted.contains("<|im_start|>system"));
        assert!(formatted.contains("<|im_start|>user\nHello<|im_end|>"));
    }
}
