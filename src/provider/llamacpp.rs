use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Message, Usage,
};
use rig::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use rig::OneOrMany;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::config::GenerationConfig;

/// Holds the llama.cpp backend and model.
pub struct LlamaCppProvider {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl LlamaCppProvider {
    /// Loads a GGUF model with the given GPU layer count.
    pub fn new(model_path: &Path, n_gpu_layers: u32) -> Result<Self> {
        let backend = LlamaBackend::init().context("Failed to init llama backend")?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))?;

        tracing::info!(
            path = %model_path.display(),
            n_gpu_layers,
            vocab_size = model.n_vocab(),
            "Model loaded"
        );

        Ok(Self {
            backend,
            model,
        })
    }

    /// Applies the model's chat template to format messages.
    fn apply_chat_template(
        &self,
        messages: &[(String, String)],
        add_ass: bool,
    ) -> Result<String> {
        let chat_messages: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|(role, content)| {
                LlamaChatMessage::new(role.clone(), content.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to create chat message: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let template = self
            .model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {e}"))?;

        self.model
            .apply_chat_template(&template, &chat_messages, add_ass)
            .map_err(|e| anyhow::anyhow!("Failed to apply chat template: {e}"))
    }

    /// Tokenizes a string using the model's tokenizer.
    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
        let bos = if add_bos {
            AddBos::Always
        } else {
            AddBos::Never
        };
        self.model
            .str_to_token(text, bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))
    }

    /// Creates a new context for inference.
    fn new_context(&self, n_ctx: u32) -> Result<LlamaContext<'_>> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx));
        self.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {e}"))
    }
}

/// Response type for non-streaming completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppResponse {
    pub text: String,
    pub think_content: Option<String>,
    pub tokens_generated: usize,
}

/// Response type for streaming completions (final response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppStreamingResponse {
    pub tokens_generated: usize,
}

impl rig::completion::GetTokenUsage for LlamaCppStreamingResponse {
    fn token_usage(&self) -> Option<Usage> {
        Some(Usage {
            input_tokens: 0,
            output_tokens: self.tokens_generated as u64,
            total_tokens: self.tokens_generated as u64,
            cached_input_tokens: 0,
        })
    }
}

/// Completion model backed by llama.cpp via llama-cpp-2.
#[derive(Clone)]
pub struct LlamaCppCompletionModel {
    provider: Arc<Mutex<LlamaCppProvider>>,
    config: GenerationConfig,
    /// Shared buffer where streaming inference deposits think block content.
    /// Read by the orchestrator after streaming completes.
    think_buffer: Arc<Mutex<Option<String>>>,
}

/// The client type for LlamaCppCompletionModel.
#[derive(Clone)]
pub struct LlamaCppClient {
    pub provider: Arc<Mutex<LlamaCppProvider>>,
    pub config: GenerationConfig,
}

/// Creates a completion model from a shared provider and config.
pub fn completion_model(
    provider: &Arc<Mutex<LlamaCppProvider>>,
    config: GenerationConfig,
) -> LlamaCppCompletionModel {
    LlamaCppCompletionModel {
        provider: provider.clone(),
        config,
        think_buffer: Arc::new(Mutex::new(None)),
    }
}

impl LlamaCppCompletionModel {
    /// Returns and clears the think block content captured during the last streaming call.
    pub fn take_think_content(&self) -> Option<String> {
        self.think_buffer.lock().ok()?.take()
    }
}

impl CompletionModel for LlamaCppCompletionModel {
    type Response = LlamaCppResponse;
    type StreamingResponse = LlamaCppStreamingResponse;
    type Client = LlamaCppClient;

    fn make(client: &Self::Client, _model: impl Into<String>) -> Self {
        Self {
            provider: client.provider.clone(),
            config: client.config.clone(),
            think_buffer: Arc::new(Mutex::new(None)),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider = self.provider.clone();
        let config = self.config.clone();
        let max_tokens_override = request.max_tokens.map(|t| t as usize);

        let prompt_text = format_request(&provider, &request)?;

        let result = tokio::task::spawn_blocking(move || {
            run_inference(&provider, &prompt_text, &config, max_tokens_override)
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
        let provider = self.provider.clone();
        let config = self.config.clone();
        let max_tokens_override = request.max_tokens.map(|t| t as usize);
        let think_buffer = self.think_buffer.clone();

        let prompt_text = format_request(&provider, &request)?;

        let (tx, rx) = mpsc::channel::<
            Result<RawStreamingChoice<LlamaCppStreamingResponse>, CompletionError>,
        >(32);

        tokio::task::spawn_blocking(move || {
            let result = run_inference_streaming(
                &provider,
                &prompt_text,
                &config,
                &tx,
                max_tokens_override,
                &think_buffer,
            );
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(CompletionError::ProviderError(format!("{e}"))));
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
}

/// Formats a CompletionRequest into a prompt string using the model's chat template.
fn format_request(
    provider: &Arc<Mutex<LlamaCppProvider>>,
    request: &CompletionRequest,
) -> Result<String, CompletionError> {
    let provider = provider
        .lock()
        .map_err(|e| CompletionError::ProviderError(format!("Lock poisoned: {e}")))?;

    let mut messages: Vec<(String, String)> = Vec::new();

    // System message: preamble + RAG documents (if any)
    if let Some(preamble) = &request.preamble {
        let mut system_content = preamble.clone();
        if !request.documents.is_empty() {
            system_content.push_str("\n\n# Reference Context\n");
            for doc in &request.documents {
                system_content.push_str(&format!("{doc}\n"));
            }
        }
        messages.push(("system".to_string(), system_content));
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
                messages.push(("user".to_string(), text));
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
                messages.push(("assistant".to_string(), text));
            }
        }
    }

    provider
        .apply_chat_template(&messages, true)
        .map_err(|e| CompletionError::ProviderError(format!("{e}")))
}

/// Builds a sampler chain with temperature, top-k, top-p.
fn build_sampler(config: &GenerationConfig) -> LlamaSampler {
    let samplers = if config.temperature <= 0.0 {
        vec![LlamaSampler::greedy()]
    } else {
        let mut s = Vec::new();
        if config.top_k > 0 {
            s.push(LlamaSampler::top_k(config.top_k as i32));
        }
        s.push(LlamaSampler::top_p(config.top_p as f32, 1));
        s.push(LlamaSampler::temp(config.temperature as f32));
        s.push(LlamaSampler::dist(config.seed.unwrap_or(42) as u32));
        s
    };

    LlamaSampler::chain(samplers, false)
}

/// Parses think blocks from generated text.
/// Returns (visible_text, think_content).
fn parse_think_blocks(text: &str) -> (String, Option<String>) {
    if let Some(start) = text.find("<think>") {
        if let Some(end) = text.find("</think>") {
            let think_content = text[start + "<think>".len()..end].trim().to_string();
            let visible = format!(
                "{}{}",
                &text[..start],
                text[end + "</think>".len()..].trim_start()
            );
            let think = if think_content.is_empty() {
                None
            } else {
                Some(think_content)
            };
            return (visible.trim().to_string(), think);
        }
        // Unclosed think block
        let visible = text[..start].trim().to_string();
        let think_content = text[start + "<think>".len()..].trim().to_string();
        let think = if think_content.is_empty() {
            None
        } else {
            Some(think_content)
        };
        return (visible, think);
    }
    (text.to_string(), None)
}

/// Runs synchronous inference (non-streaming).
fn run_inference(
    provider: &Arc<Mutex<LlamaCppProvider>>,
    prompt_text: &str,
    config: &GenerationConfig,
    max_tokens_override: Option<usize>,
) -> Result<LlamaCppResponse> {
    let t0 = Instant::now();
    let provider = provider
        .lock()
        .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;

    let tokens = provider.tokenize(prompt_text, false)?;
    let prompt_token_count = tokens.len();
    let max_tokens = max_tokens_override.unwrap_or(config.max_tokens);

    let t1 = Instant::now();
    tracing::info!(
        prompt_tokens = prompt_token_count,
        tokenization_ms = t1.duration_since(t0).as_millis() as u64,
        "Tokenized prompt"
    );

    // Create context large enough for prompt + generation
    let n_ctx = (prompt_token_count + max_tokens + 64) as u32;
    let mut ctx = provider.new_context(n_ctx)?;

    // Prefill: process all prompt tokens
    let mut batch = LlamaBatch::new(prompt_token_count + max_tokens, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|_| anyhow::anyhow!("Failed to add token to batch"))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("Prefill decode failed: {e}"))?;

    let t2 = Instant::now();
    let prefill_ms = t2.duration_since(t1).as_millis() as u64;
    tracing::info!(prefill_ms, prompt_tokens = prompt_token_count, "Prefill complete");

    // Generation loop
    let mut sampler = build_sampler(config);
    let mut generated_tokens: Vec<LlamaToken> = Vec::new();
    let mut n_decoded = prompt_token_count;
    let eos_token = provider.model.token_eos();

    let decode_start = Instant::now();
    for _ in 0..max_tokens {
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);

        if token == eos_token {
            break;
        }

        generated_tokens.push(token);
        n_decoded += 1;

        batch.clear();
        batch
            .add(token, n_decoded as i32 - 1, &[0], true)
            .map_err(|_| anyhow::anyhow!("Failed to add token to batch"))?;

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    }

    let decode_elapsed = decode_start.elapsed();
    let tokens_generated = generated_tokens.len();
    let decode_tok_per_sec = if decode_elapsed.as_millis() > 0 {
        (tokens_generated as f64 / decode_elapsed.as_millis() as f64) * 1000.0
    } else {
        0.0
    };

    // Detokenize — drop ctx first to release immutable borrow on provider
    drop(ctx);
    let mut full_text = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    for &token in &generated_tokens {
        if let Ok(piece) = provider.model.token_to_piece(token, &mut decoder, true, None) {
            full_text.push_str(&piece);
        }
    }

    tracing::info!(
        tokens_generated,
        decode_ms = decode_elapsed.as_millis() as u64,
        decode_tok_per_sec = format!("{decode_tok_per_sec:.1}"),
        total_ms = t0.elapsed().as_millis() as u64,
        prefill_ms,
        "Generation complete"
    );

    let (visible_text, think_content) = parse_think_blocks(&full_text);

    if let Some(ref think) = think_content {
        tracing::debug!(think_len = think.len(), "Think block captured");
    }

    Ok(LlamaCppResponse {
        text: visible_text,
        think_content,
        tokens_generated,
    })
}

/// Runs streaming inference, sending visible tokens through the channel.
/// Think blocks are buffered and NOT streamed to the user.
fn run_inference_streaming(
    provider: &Arc<Mutex<LlamaCppProvider>>,
    prompt_text: &str,
    config: &GenerationConfig,
    tx: &mpsc::Sender<Result<RawStreamingChoice<LlamaCppStreamingResponse>, CompletionError>>,
    max_tokens_override: Option<usize>,
    think_output: &Arc<Mutex<Option<String>>>,
) -> Result<()> {
    let t0 = Instant::now();
    let provider = provider
        .lock()
        .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;

    let tokens = provider.tokenize(prompt_text, false)?;
    let prompt_token_count = tokens.len();
    let max_tokens = max_tokens_override.unwrap_or(config.max_tokens);

    let t1 = Instant::now();
    tracing::info!(
        prompt_tokens = prompt_token_count,
        tokenization_ms = t1.duration_since(t0).as_millis() as u64,
        "Tokenized prompt"
    );

    let n_ctx = (prompt_token_count + max_tokens + 64) as u32;
    let mut ctx = provider.new_context(n_ctx)?;

    // Prefill
    let mut batch = LlamaBatch::new(prompt_token_count + max_tokens, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|_| anyhow::anyhow!("Failed to add token to batch"))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("Prefill decode failed: {e}"))?;

    let t2 = Instant::now();
    tracing::info!(
        prefill_ms = t2.duration_since(t1).as_millis() as u64,
        prompt_tokens = prompt_token_count,
        "Prefill complete"
    );

    // Generation loop with think block buffering
    let mut sampler = build_sampler(config);
    let mut n_decoded = prompt_token_count;
    let mut tokens_generated = 0usize;
    let eos_token = provider.model.token_eos();

    // State machine for think block detection
    let mut in_think_block = false;
    let mut think_buffer = String::new();
    let mut text_buffer = String::new();
    let mut think_closed = false;

    let decode_start = Instant::now();
    for _ in 0..max_tokens {
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);

        if token == eos_token {
            break;
        }

        tokens_generated += 1;
        n_decoded += 1;

        // Use token_to_piece_bytes to avoid borrow conflict with ctx
        let piece = provider
            .model
            .token_to_piece_bytes(token, 64, true, None)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .unwrap_or_default();

        // State machine: detect and buffer think blocks
        if !think_closed {
            if !in_think_block {
                text_buffer.push_str(&piece);
                if text_buffer.contains("<think>") {
                    in_think_block = true;
                    if let Some(pos) = text_buffer.find("<think>") {
                        let before = &text_buffer[..pos];
                        if !before.is_empty()
                            && tx
                                .blocking_send(Ok(RawStreamingChoice::Message(
                                    before.to_string(),
                                )))
                                .is_err()
                        {
                            return Ok(());
                        }
                        think_buffer = text_buffer[pos + "<think>".len()..].to_string();
                    }
                    text_buffer.clear();
                }
            } else {
                // Inside think block — buffer everything
                think_buffer.push_str(&piece);

                if think_buffer.contains("</think>") {
                    in_think_block = false;
                    think_closed = true;
                    if let Some(pos) = think_buffer.find("</think>") {
                        let after = think_buffer[pos + "</think>".len()..].to_string();
                        think_buffer = think_buffer[..pos].to_string();
                        let trimmed = after.trim_start().to_string();
                        if !trimmed.is_empty()
                            && tx
                                .blocking_send(Ok(RawStreamingChoice::Message(trimmed)))
                                .is_err()
                        {
                            return Ok(());
                        }
                    }
                }
            }
        } else {
            // Outside think block — stream directly
            if !piece.is_empty()
                && tx
                    .blocking_send(Ok(RawStreamingChoice::Message(piece)))
                    .is_err()
            {
                return Ok(());
            }
        }

        // Decode next token
        batch.clear();
        batch
            .add(token, n_decoded as i32 - 1, &[0], true)
            .map_err(|_| anyhow::anyhow!("Failed to add token to batch"))?;

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
    }

    // If we never entered/closed a think block, flush text_buffer
    if !think_closed && !in_think_block && !text_buffer.is_empty() {
        let _ = tx.blocking_send(Ok(RawStreamingChoice::Message(text_buffer)));
    }

    let decode_elapsed = decode_start.elapsed();
    let decode_tok_per_sec = if decode_elapsed.as_millis() > 0 {
        (tokens_generated as f64 / decode_elapsed.as_millis() as f64) * 1000.0
    } else {
        0.0
    };

    if !think_buffer.is_empty() {
        let trimmed = think_buffer.trim().to_string();
        tracing::debug!(
            think_len = trimmed.len(),
            "Think block captured (streaming)"
        );
        if let Ok(mut buf) = think_output.lock() {
            *buf = Some(trimmed);
        }
    }

    tracing::info!(
        tokens_generated,
        decode_ms = decode_elapsed.as_millis() as u64,
        decode_tok_per_sec = format!("{decode_tok_per_sec:.1}"),
        total_ms = t0.elapsed().as_millis() as u64,
        "Generation complete"
    );

    let _ = tx.blocking_send(Ok(RawStreamingChoice::FinalResponse(
        LlamaCppStreamingResponse { tokens_generated },
    )));

    Ok(())
}
