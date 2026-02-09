//! Chat template support for LLM inference.
//!
//! Vendored from `candle-examples` (MIT license) with modifications:
//! - Fixed `chatml_with_thinking()` to prefill empty `<think>\n\n</think>` block
//!   when `enable_thinking=false`, matching Qwen3/SmolLM3 template behavior
//! - Added `from_jinja_file()` for loading standalone `.jinja` template files
//! - Removed unused presets and `Conversation` type

use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A chat message with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Options for applying a chat template.
#[derive(Debug, Clone, Default)]
pub struct ChatTemplateOptions {
    /// Add tokens that prompt the model to generate an assistant response.
    pub add_generation_prompt: bool,
    /// Enable thinking/reasoning mode (adds `<think>` tags).
    pub enable_thinking: bool,
}

impl ChatTemplateOptions {
    pub fn for_generation() -> Self {
        Self {
            add_generation_prompt: true,
            ..Default::default()
        }
    }

    pub fn with_thinking(mut self) -> Self {
        self.enable_thinking = true;
        self
    }
}

/// Token configuration loaded from `tokenizer_config.json`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenConfig {
    #[serde(default)]
    pub bos_token: Option<StringOrToken>,
    #[serde(default)]
    pub eos_token: Option<StringOrToken>,
    #[serde(default)]
    pub chat_template: Option<ChatTemplateConfig>,
}

/// Handles both string and object token formats in `tokenizer_config.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrToken {
    String(String),
    Token { content: String },
}

impl StringOrToken {
    pub fn as_str(&self) -> &str {
        match self {
            StringOrToken::String(s) => s,
            StringOrToken::Token { content } => content,
        }
    }
}

impl Default for StringOrToken {
    fn default() -> Self {
        StringOrToken::String(String::new())
    }
}

/// Chat template can be a single string or multiple named templates.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateConfig {
    Single(String),
    Multiple(Vec<NamedTemplate>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct NamedTemplate {
    pub name: String,
    pub template: String,
}

/// Chat template renderer using MiniJinja.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Creates from a Jinja template string with explicit BOS/EOS tokens.
    pub fn new(
        template: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
    ) -> Result<Self, ChatTemplateError> {
        let mut env = Environment::new();
        // Add the raise_exception function that HF templates use
        env.add_function("raise_exception", |msg: String| -> Result<String, _> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });

        env.add_template_owned("chat".to_string(), template.into())
            .map_err(|e| ChatTemplateError::TemplateError(e.to_string()))?;

        Ok(Self {
            env,
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        })
    }

    /// Loads chat template from a `tokenizer_config.json` file.
    pub fn from_tokenizer_config(path: impl AsRef<Path>) -> Result<Self, ChatTemplateError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ChatTemplateError::IoError(e.to_string()))?;

        Self::from_tokenizer_config_str(&content)
    }

    /// Loads chat template from `tokenizer_config.json` content.
    pub fn from_tokenizer_config_str(json: &str) -> Result<Self, ChatTemplateError> {
        let config: TokenConfig =
            serde_json::from_str(json).map_err(|e| ChatTemplateError::ParseError(e.to_string()))?;

        let template = match config.chat_template {
            Some(ChatTemplateConfig::Single(t)) => t,
            Some(ChatTemplateConfig::Multiple(templates)) => {
                // Use "default" template if available, otherwise first one
                templates
                    .iter()
                    .find(|t| t.name == "default")
                    .or_else(|| templates.first())
                    .map(|t| t.template.clone())
                    .ok_or(ChatTemplateError::NoTemplate)?
            }
            None => return Err(ChatTemplateError::NoTemplate),
        };

        let bos = config
            .bos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();
        let eos = config
            .eos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();

        Self::new(template, bos, eos)
    }

    /// Loads a standalone `.jinja` template file, reading BOS/EOS from a sibling
    /// `tokenizer_config.json` if present.
    pub fn from_jinja_file(path: impl AsRef<Path>) -> Result<Self, ChatTemplateError> {
        let path = path.as_ref();
        let template = std::fs::read_to_string(path)
            .map_err(|e| ChatTemplateError::IoError(e.to_string()))?;

        let config_path = path.with_file_name("tokenizer_config.json");
        let (bos, eos) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| ChatTemplateError::IoError(e.to_string()))?;
            let config: TokenConfig = serde_json::from_str(&config_str)
                .map_err(|e| ChatTemplateError::ParseError(e.to_string()))?;
            (
                config
                    .bos_token
                    .map(|t| t.as_str().to_string())
                    .unwrap_or_default(),
                config
                    .eos_token
                    .map(|t| t.as_str().to_string())
                    .unwrap_or_default(),
            )
        } else {
            (String::new(), String::new())
        };

        Self::new(template, bos, eos)
    }

    /// ChatML template with thinking/reasoning support (SmolLM3 default).
    ///
    /// When `enable_thinking=true`: prefills `<|im_start|>assistant\n<think>\n`
    /// When `enable_thinking=false`: prefills `<|im_start|>assistant\n<think>\n\n</think>\n\n`
    /// to structurally suppress thinking.
    pub fn chatml_with_thinking() -> Self {
        let template = r#"
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content | trim + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{%- if enable_thinking %}
{{- '<|im_start|>assistant\n<think>\n' }}
{%- else %}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}
"#;
        Self::new(template, "", "<|im_end|>").unwrap()
    }

    /// Applies the chat template to messages with the given options.
    pub fn apply(
        &self,
        messages: &[Message],
        options: &ChatTemplateOptions,
    ) -> Result<String, ChatTemplateError> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| ChatTemplateError::TemplateError(e.to_string()))?;

        let result = template
            .render(context! {
                messages => messages,
                add_generation_prompt => options.add_generation_prompt,
                enable_thinking => options.enable_thinking,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .map_err(|e| ChatTemplateError::RenderError(e.to_string()))?;

        Ok(result.trim_start().to_string())
    }

    /// Convenience: apply with `add_generation_prompt=true`.
    pub fn apply_for_generation(&self, messages: &[Message]) -> Result<String, ChatTemplateError> {
        self.apply(messages, &ChatTemplateOptions::for_generation())
    }
}

/// Errors that can occur with chat templates.
#[derive(Debug)]
pub enum ChatTemplateError {
    IoError(String),
    ParseError(String),
    TemplateError(String),
    RenderError(String),
    NoTemplate,
}

impl std::fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::ParseError(e) => write!(f, "Parse error: {e}"),
            Self::TemplateError(e) => write!(f, "Template error: {e}"),
            Self::RenderError(e) => write!(f, "Render error: {e}"),
            Self::NoTemplate => write!(f, "No chat_template found in config"),
        }
    }
}

impl std::error::Error for ChatTemplateError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_with_thinking_enabled() {
        let template = ChatTemplate::chatml_with_thinking();
        let messages = vec![Message::user("Think about this")];

        let result = template
            .apply(
                &messages,
                &ChatTemplateOptions::for_generation().with_thinking(),
            )
            .unwrap();

        assert!(result.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn test_chatml_with_thinking_disabled() {
        let template = ChatTemplate::chatml_with_thinking();
        let messages = vec![Message::user("Just answer")];

        let result = template
            .apply(&messages, &ChatTemplateOptions::for_generation())
            .unwrap();

        assert!(result.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_from_json_config() {
        let json = r#"{
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        }"#;

        let template = ChatTemplate::from_tokenizer_config_str(json).unwrap();
        let messages = vec![Message::user("test")];
        let result = template.apply_for_generation(&messages).unwrap();

        assert!(result.contains("user: test"));
    }

    #[test]
    fn test_from_json_config_multiple_templates() {
        let json = r#"{
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": [
                {"name": "tool_use", "template": "tool template"},
                {"name": "default", "template": "{% for m in messages %}{{ m.content }}{% endfor %}"}
            ]
        }"#;

        let template = ChatTemplate::from_tokenizer_config_str(json).unwrap();
        let messages = vec![Message::user("hello")];
        let result = template.apply_for_generation(&messages).unwrap();

        assert!(result.contains("hello"));
    }
}
