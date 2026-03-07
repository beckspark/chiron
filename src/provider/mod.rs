pub mod config;
pub mod llamacpp;

pub use llamacpp::{completion_model, LlamaCppCompletionModel, LlamaCppProvider};

/// Strips `<think>...</think>` blocks from model output.
///
/// Safety net for cases where think blocks leak into visible text.
pub fn strip_think_blocks(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            let block_end = end + "</think>".len();
            result = format!(
                "{}{}",
                &result[..start],
                result[block_end..].trim_start()
            );
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}
