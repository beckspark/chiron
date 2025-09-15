use crate::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use tokio::time::{sleep, Duration};

#[derive(Debug, Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
    pub done: bool,
}

pub struct OllamaClient {
    client: Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }

    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        self.generate_with_progress(model, prompt, false).await
    }

    pub async fn generate_with_progress(&self, model: &str, prompt: &str, show_progress: bool) -> Result<String> {
        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: show_progress,
        };

        if show_progress {
            self.generate_streaming(&request).await
        } else {
            self.generate_non_streaming(&request).await
        }
    }

    /// Unload the model from memory immediately
    pub async fn unload_model(&self, model: &str) -> Result<()> {
        let unload_request = serde_json::json!({
            "model": model,
            "keep_alive": 0
        });

        let response = self
            .client
            .post(&format!("{}/api/generate", self.base_url))
            .json(&unload_request)
            .send()
            .await;

        // Ignore errors during cleanup - model might already be unloaded
        match response {
            Ok(_) => Ok(()),
            Err(_) => Ok(()), // Graceful failure for cleanup
        }
    }

    async fn generate_non_streaming(&self, request: &GenerateRequest) -> Result<String> {
        let response = self
            .client
            .post(&format!("{}/api/generate", self.base_url))
            .json(request)
            .send()
            .await?;

        let generate_response: GenerateResponse = response.json().await?;
        Ok(generate_response.response)
    }

    async fn generate_streaming(&self, request: &GenerateRequest) -> Result<String> {
        use futures_util::StreamExt;

        let response = self
            .client
            .post(&format!("{}/api/generate", self.base_url))
            .json(request)
            .send()
            .await?;

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();
        let mut spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'].iter().cycle();

        print!("Chiron: ");
        io::stdout().flush().unwrap();

        let mut buffer = Vec::new();
        let mut spinner_counter = 0;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            buffer.extend_from_slice(&chunk);

            // Try to parse complete JSON lines from buffer
            let buffer_str = String::from_utf8_lossy(&buffer);
            let mut last_complete_index = 0;

            for line in buffer_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                if let Ok(response) = serde_json::from_str::<GenerateResponse>(line) {
                    if !response.response.is_empty() {
                        print!("{}", response.response);
                        io::stdout().flush().unwrap();
                        full_response.push_str(&response.response);
                    }

                    if response.done {
                        println!(); // New line after completion
                        return Ok(full_response);
                    }

                    last_complete_index = line.len() + 1; // +1 for newline
                } else {
                    // Show spinner while waiting for more data
                    spinner_counter += 1;
                    if spinner_counter % 10 == 0 {
                        print!("\r{} ", spinner_chars.next().unwrap());
                        io::stdout().flush().unwrap();
                        sleep(Duration::from_millis(50)).await;
                        print!("\rChiron: {}", full_response);
                        io::stdout().flush().unwrap();
                    }
                }
            }

            // Remove processed lines from buffer
            if last_complete_index > 0 {
                buffer.drain(..last_complete_index);
            }
        }

        println!(); // Ensure we end with a newline
        Ok(full_response)
    }
}
