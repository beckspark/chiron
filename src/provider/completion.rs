use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use futures::stream;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use rig::OneOrMany;
use serde::{Deserialize, Serialize};

use super::ModelRegistry;
use super::CandleProvider;

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
/// Implements Rig's `CompletionModel` trait to run local Llama models
/// via Candle on CUDA (or CPU fallback).
#[derive(Clone)]
pub struct CandleCompletionModel {
    registry: Arc<Mutex<ModelRegistry>>,
    model_id: String,
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
        let prompt_text = format_request(&request);

        let result = tokio::task::spawn_blocking(move || {
            run_inference(&registry, &model_id, &prompt_text)
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
        // For local models, we don't have true token-by-token streaming from the model
        // since Candle's forward pass is synchronous. We generate the full response
        // then emit it as a single streaming chunk.
        let response = self.completion(request).await?;
        let text = response.raw_response.text.clone();
        let tokens = response.raw_response.tokens_generated;

        let chunks = vec![
            Ok(RawStreamingChoice::Message(text)),
            Ok(RawStreamingChoice::FinalResponse(CandleStreamingResponse {
                tokens_generated: tokens,
            })),
        ];

        Ok(StreamingCompletionResponse::stream(Box::pin(
            stream::iter(chunks),
        )))
    }
}

/// Formats a `CompletionRequest` into a Llama 3.2 Instruct chat template string.
///
/// Uses the format:
/// ```text
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
///
/// {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
///
/// {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
///
/// ```
fn format_request(request: &CompletionRequest) -> String {
    let mut prompt = String::from("<|begin_of_text|>");

    // System preamble
    if let Some(preamble) = &request.preamble {
        prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        prompt.push_str(preamble);

        // Append any context documents to the system message
        if !request.documents.is_empty() {
            prompt.push_str("\n\n# Reference Context\n");
            for doc in &request.documents {
                prompt.push_str(&format!("{}\n", doc));
            }
        }

        prompt.push_str("<|eot_id|>");
    }

    // Chat history (all messages including the final prompt)
    for message in request.chat_history.iter() {
        match message {
            Message::User { content } => {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                for item in content.iter() {
                    if let rig::message::UserContent::Text(text) = item {
                        prompt.push_str(&text.text);
                    }
                }
                prompt.push_str("<|eot_id|>");
            }
            Message::Assistant { content, .. } => {
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                for item in content.iter() {
                    if let AssistantContent::Text(text) = item {
                        prompt.push_str(&text.text);
                    }
                }
                prompt.push_str("<|eot_id|>");
            }
        }
    }

    // Generation prompt -- signal the model to generate an assistant response
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    prompt
}

/// Runs synchronous Candle inference. Called inside `spawn_blocking`.
///
/// Logs detailed timing information at each phase: tokenization, prefill,
/// per-token decode, and total generation summary.
fn run_inference(
    registry: &Arc<Mutex<ModelRegistry>>,
    model_id: &str,
    prompt_text: &str,
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

    let max_tokens = model.config.max_tokens;
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
    use rig::OneOrMany;

    #[test]
    fn test_format_request_simple() {
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

        let formatted = format_request(&request);

        assert!(formatted.starts_with("<|begin_of_text|>"));
        assert!(formatted.contains("system"));
        assert!(formatted.contains("You are a helpful assistant."));
        assert!(formatted.contains("Hello"));
        assert!(formatted.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_format_request_with_history() {
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

        let formatted = format_request(&request);

        assert!(formatted.contains("First message"));
        assert!(formatted.contains("First response"));
        assert!(formatted.contains("Second message"));
        // Should end with assistant generation prompt
        assert!(formatted.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }
}
