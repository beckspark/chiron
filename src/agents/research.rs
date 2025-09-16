use crate::Result;
use super::protocol::{Agent, AgentRequest, AgentResponse, AgentMetadata, Capability};
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use url::Url;


/// Intent detection for research requests
pub struct IntentDetector {
    url_patterns: Vec<Regex>,
    research_keywords: HashSet<String>,
}

impl IntentDetector {
    pub fn new() -> Self {
        let url_patterns = vec![
            // HTTP/HTTPS URLs
            Regex::new(r"https?://[^\s)]+").unwrap(),
            // Markdown links - capture just the URL part
            Regex::new(r"\[.*?\]\((https?://[^\s)]+)\)").unwrap(),
        ];

        let research_keywords = [
            "research", "tell me about", "what is", "explain", "look up",
            "find information", "search for", "more about", "definition of",
            "can we research", "let's research", "research this", "research further",
        ]
        .iter()
        .map(|s| s.to_lowercase())
        .collect();

        Self {
            url_patterns,
            research_keywords,
        }
    }

    /// Detect research intent from user input (fast pattern-based detection)
    pub fn detect_intent(&self, input: &str) -> ResearchIntent {
        let input_lower = input.to_lowercase();

        // Debug output (comment out for production)
        // eprintln!("🔍 Intent Detection Debug:");
        // eprintln!("  Input: '{}'", input);
        // eprintln!("  Lowercase: '{}'", input_lower);

        // Check for direct URLs
        if let Some(url) = self.extract_url(input) {
            eprintln!("  -> Direct URL detected: {}", url);
            return ResearchIntent::DirectUrl(url);
        }

        // Check for explicit research requests
        let has_research_keyword = self.research_keywords.iter()
            .any(|keyword| input_lower.contains(keyword));

        // If explicit research keywords are used, extract topic and let main LLM decide
        if has_research_keyword {
            let topic = extract_research_topic(&input_lower, &self.research_keywords);
            return ResearchIntent::ExplicitResearch(vec![topic]);
        }

        // Check for question patterns - let main LLM decide if relevant
        if input_lower.starts_with("what") || input_lower.starts_with("how") ||
           input_lower.starts_with("why") || input_lower.contains("?") {
            let topic = extract_question_topic(&input_lower);
            if !topic.is_empty() {
                return ResearchIntent::SuggestedResearch(vec![topic]);
            }
        }

        ResearchIntent::None
    }

    /// Extract URL from user input
    fn extract_url(&self, input: &str) -> Option<String> {
        for pattern in &self.url_patterns {
            if let Some(captures) = pattern.captures(input) {
                // If it's a markdown link with capture group, use the captured URL
                if captures.len() > 1 {
                    return captures.get(1).map(|m| m.as_str().to_string());
                }
                // Otherwise use the full match
                return captures.get(0).map(|m| m.as_str().to_string());
            }
        }
        None
    }
}

/// Extract research topic from input after removing research keywords
fn extract_research_topic(input: &str, research_keywords: &HashSet<String>) -> String {
    let mut topic = input.to_string();

    // Remove research keywords from the input
    for keyword in research_keywords {
        if input.contains(keyword) {
            topic = input.replace(keyword, "").trim().to_string();
            break;
        }
    }

    // Clean up the topic
    topic = topic.trim_start_matches("the ").trim().to_string();

    // If topic is empty or too short, use a generic fallback
    if topic.is_empty() || topic.len() < 3 {
        topic = "general topic".to_string();
    }

    topic
}

/// Extract topic from a question (what is X?, how does Y work?, etc.)
fn extract_question_topic(input: &str) -> String {
    // Simple patterns for extracting topics from questions
    let patterns = [
        r"what is (.+?)(?:\?|$)",
        r"what are (.+?)(?:\?|$)",
        r"how does (.+?) work(?:\?|$)",
        r"how do (.+?) work(?:\?|$)",
        r"tell me about (.+?)(?:\?|$)",
        r"explain (.+?)(?:\?|$)",
    ];

    for pattern in &patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(captures) = re.captures(input) {
                if let Some(topic) = captures.get(1) {
                    let cleaned = topic.as_str()
                        .trim()
                        .trim_start_matches("the ")
                        .trim()
                        .to_string();
                    if !cleaned.is_empty() && cleaned.len() > 2 {
                        return cleaned;
                    }
                }
            }
        }
    }

    // Fallback: return empty string if no pattern matches
    String::new()
}

/// Types of research intent detected
#[derive(Debug, Clone)]
pub enum ResearchIntent {
    /// Direct URL provided - automatic research
    DirectUrl(String),
    /// Explicit research request - automatic research
    ExplicitResearch(Vec<String>),
    /// Suggested research based on topic - ask user
    SuggestedResearch(Vec<String>),
    /// No research needed
    None,
}

/// Result of web content extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub url: String,
    pub title: String,
    pub content: String,
    pub source_domain: String,
    pub extracted_at: chrono::DateTime<chrono::Utc>,
}

/// Processed research content after LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedResearch {
    pub summary: String,
    pub key_facts: Vec<String>,
    pub relevant_sections: Vec<String>,
    pub therapeutic_relevance: String,
}

/// URL whitelist validator
pub struct UrlValidator {
    whitelisted_domains: HashSet<String>,
}

impl UrlValidator {
    pub fn new() -> Self {
        let whitelisted_domains = [
            "en.wikipedia.org",
            "www.psychologytoday.com",
            "psychologytoday.com",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self { whitelisted_domains }
    }

    /// Validate if URL is from whitelisted domain
    pub fn is_whitelisted(&self, url_str: &str) -> bool {
        if let Ok(url) = Url::parse(url_str) {
            if let Some(domain) = url.domain() {
                return self.whitelisted_domains.contains(domain);
            }
        }
        false
    }

    /// Get the domain from a URL for logging
    pub fn get_domain(&self, url_str: &str) -> Option<String> {
        if let Ok(url) = Url::parse(url_str) {
            url.domain().map(|d| d.to_string())
        } else {
            None
        }
    }
}

/// Research agent for fetching mental health information
pub struct ResearchAgent {
    client: reqwest::Client,
    ollama_client: Arc<crate::inference::OllamaClient>,
    intent_detector: IntentDetector,
    url_validator: UrlValidator,
}

impl ResearchAgent {
    pub fn new(ollama_client: Arc<crate::inference::OllamaClient>) -> Self {
        Self {
            client: reqwest::Client::new(),
            ollama_client,
            intent_detector: IntentDetector::new(),
            url_validator: UrlValidator::new(),
        }
    }

    /// Analyze user input for research intent (fast pattern-based)
    pub fn analyze_intent(&self, input: &str) -> ResearchIntent {
        self.intent_detector.detect_intent(input)
    }

    /// Check if URL is whitelisted
    pub fn is_url_whitelisted(&self, url: &str) -> bool {
        self.url_validator.is_whitelisted(url)
    }

    /// Research a topic using Wikipedia API
    pub async fn research_topic(&self, topic: &str, request: &AgentRequest) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        use std::io::{self, Write};

        println!("🔍 Searching Wikipedia for '{}'...", topic);
        io::stdout().flush().unwrap();

        // Create Wikipedia API client
        let api = match mediawiki::api::Api::new("https://en.wikipedia.org/w/api.php").await {
            Ok(api) => api,
            Err(e) => {
                return Ok(AgentResponse {
                    content: format!("❌ Failed to connect to Wikipedia API: {}", e),
                    metadata: AgentMetadata {
                        agent_name: "research".to_string(),
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                        confidence: 0.0,
                        sources: vec![],
                        content_type: "text".to_string(),
                    },
                    resources_used: vec![],
                });
            }
        };

        // Search for the topic
        let search_params = api.params_into(&[
            ("action", "query"),
            ("list", "search"),
            ("srsearch", topic),
            ("srlimit", "3"),
            ("format", "json"),
        ]);

        let search_result = match api.get_query_api_json(&search_params).await {
            Ok(result) => result,
            Err(e) => {
                return Ok(AgentResponse {
                    content: format!("❌ Wikipedia search failed: {}", e),
                    metadata: AgentMetadata {
                        agent_name: "research".to_string(),
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                        confidence: 0.0,
                        sources: vec![],
                        content_type: "text".to_string(),
                    },
                    resources_used: vec![],
                });
            }
        };

        // Extract search results
        let search_results = search_result["query"]["search"].as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|item| {
                Some((
                    item["title"].as_str()?.to_string(),
                    item["snippet"].as_str().unwrap_or("").to_string(),
                ))
            })
            .collect::<Vec<_>>();

        if search_results.is_empty() {
            return Ok(AgentResponse {
                content: format!("❌ No Wikipedia articles found for '{}'", topic),
                metadata: AgentMetadata {
                    agent_name: "research".to_string(),
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    confidence: 0.0,
                    sources: vec![],
                    content_type: "text".to_string(),
                },
                resources_used: vec![],
            });
        }

        // Get the first (most relevant) result
        let (title, _snippet) = &search_results[0];
        println!("📄 Found '{}', fetching content...", title);
        io::stdout().flush().unwrap();

        // Get page content
        let content_params = api.params_into(&[
            ("action", "query"),
            ("prop", "extracts"),
            ("titles", title),
            ("exintro", "1"),
            ("explaintext", "1"),
            ("exsectionformat", "plain"),
            ("format", "json"),
        ]);

        let content_result = match api.get_query_api_json(&content_params).await {
            Ok(result) => result,
            Err(e) => {
                return Ok(AgentResponse {
                    content: format!("❌ Failed to fetch Wikipedia content: {}", e),
                    metadata: AgentMetadata {
                        agent_name: "research".to_string(),
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                        confidence: 0.0,
                        sources: vec![],
                        content_type: "text".to_string(),
                    },
                    resources_used: vec![],
                });
            }
        };

        // Extract the content
        let pages = content_result["query"]["pages"].as_object().unwrap();
        let page_content = pages.values().next()
            .and_then(|page| page["extract"].as_str())
            .unwrap_or("No content available");

        if page_content.is_empty() || page_content == "No content available" {
            return Ok(AgentResponse {
                content: format!("❌ No content found for Wikipedia article '{}'", title),
                metadata: AgentMetadata {
                    agent_name: "research".to_string(),
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    confidence: 0.0,
                    sources: vec![],
                    content_type: "text".to_string(),
                },
                resources_used: vec![],
            });
        }

        println!("🤖 Analyzing content with {}...", request.context.current_model);
        io::stdout().flush().unwrap();

        // Process content with LLM for therapeutic relevance
        let processed = self.process_with_llm(
            page_content,
            &format!("Research topic: {}", topic),
            &request.context.current_model
        ).await?;

        let wikipedia_url = format!("https://en.wikipedia.org/wiki/{}", title.replace(" ", "_"));

        let result = format!(
            "📚 **Wikipedia Research: {}**\n\n{}\n\n**Key Facts:**\n{}\n\n**Therapeutic Relevance:** {}\n\n*Source: {}*",
            title,
            processed.summary,
            processed.key_facts.iter()
                .map(|fact| format!("• {}", fact))
                .collect::<Vec<_>>()
                .join("\n"),
            processed.therapeutic_relevance,
            wikipedia_url
        );

        println!("✅ Research complete!");

        Ok(AgentResponse {
            content: result,
            metadata: AgentMetadata {
                agent_name: "research".to_string(),
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                confidence: 0.8,
                sources: vec![wikipedia_url.clone()],
                content_type: "markdown".to_string(),
            },
            resources_used: vec![wikipedia_url],
        })
    }

    /// Fetch and process content from a URL
    pub async fn fetch_url(&self, url: &str) -> Result<ResearchResult> {
        use std::io::{self, Write};

        // Validate URL is whitelisted
        if !self.is_url_whitelisted(url) {
            return Err(anyhow::anyhow!("URL not whitelisted: {}", url));
        }

        // Show progress indicator
        print!("🌐 Fetching content from {}...", self.url_validator.get_domain(url).unwrap_or_else(|| "unknown".to_string()));
        io::stdout().flush().unwrap();

        // Fetch the webpage
        let response = self.client
            .get(url)
            .header("User-Agent", "Chiron Mental Health Research Agent/1.0")
            .send()
            .await?;

        print!(" ✅\n📄 Extracting content...");
        io::stdout().flush().unwrap();

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error {}: {}", response.status(), url));
        }

        let html_content = response.text().await?;

        // Parse and extract content
        let extracted_content = self.extract_main_content(&html_content, url)?;

        // Convert to markdown for better LLM processing
        print!(" ✅\n📝 Converting to markdown...");
        io::stdout().flush().unwrap();

        let markdown_content = html2text::from_read(extracted_content.as_bytes(), 120);

        println!(" ✅");

        Ok(ResearchResult {
            url: url.to_string(),
            title: self.extract_title(&html_content).unwrap_or_else(|| "Untitled".to_string()),
            content: markdown_content,
            source_domain: self.url_validator.get_domain(url).unwrap_or_else(|| "unknown".to_string()),
            extracted_at: chrono::Utc::now(),
        })
    }

    /// Extract main content from HTML
    fn extract_main_content(&self, html: &str, url: &str) -> Result<String> {
        let document = Html::parse_document(html);

        // Try different selectors based on the domain
        let content_selectors = if url.contains("wikipedia.org") {
            vec![
                "#mw-content-text",
                "#bodyContent",
                ".mw-parser-output",
            ]
        } else if url.contains("psychologytoday.com") {
            vec![
                ".entry-content",
                ".article-content",
                ".post-content",
                "main article",
                "article",
            ]
        } else {
            vec![
                "main",
                "article",
                ".content",
                "#content",
                ".post-content",
                ".entry-content",
            ]
        };

        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let mut content = element.text().collect::<Vec<_>>().join(" ");

                    // Clean up the content
                    content = content
                        .lines()
                        .map(|line| line.trim())
                        .filter(|line| !line.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n");

                    if content.len() > 100 { // Ensure we got substantial content
                        return Ok(content);
                    }
                }
            }
        }

        // Fallback: extract from body
        if let Ok(body_selector) = Selector::parse("body") {
            if let Some(body) = document.select(&body_selector).next() {
                let content = body.text().collect::<Vec<_>>().join(" ");
                return Ok(content);
            }
        }

        Err(anyhow::anyhow!("Could not extract content from HTML"))
    }

    /// Extract title from HTML
    fn extract_title(&self, html: &str) -> Option<String> {
        let document = Html::parse_document(html);

        if let Ok(title_selector) = Selector::parse("title") {
            if let Some(title_element) = document.select(&title_selector).next() {
                let title = title_element.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return Some(title);
                }
            }
        }

        // Try h1 as fallback
        if let Ok(h1_selector) = Selector::parse("h1") {
            if let Some(h1_element) = document.select(&h1_selector).next() {
                let title = h1_element.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return Some(title);
                }
            }
        }

        None
    }

    /// Search Wikipedia for a mental health topic
    pub async fn search_wikipedia(&self, query: &str) -> Result<ResearchResult> {
        let search_url = format!(
            "https://en.wikipedia.org/wiki/{}",
            query.replace(" ", "_")
        );

        self.fetch_url(&search_url).await
    }

    /// Process research content with LLM for extraction
    pub async fn process_with_llm(&self, content: &str, query: &str, model: &str) -> Result<ProcessedResearch> {
        use std::io::{self, Write};

        print!("🤖 Analyzing content with {}...", model);
        io::stdout().flush().unwrap();

        let prompt = format!(
            r#"You are a mental health research assistant. Extract key information from this content about: {}

Content:
{}

IMPORTANT: Respond with ONLY a valid JSON object, no markdown formatting, no explanation. Use this exact structure:

{{
    "summary": "Write a 2-3 sentence summary of the main points",
    "key_facts": ["Write 3-5 important facts as separate strings"],
    "relevant_sections": ["List 2-3 main topic areas covered"],
    "therapeutic_relevance": "Explain how this information helps with mental health treatment"
}}

JSON response:"#,
            query, content
        );

        let response = self.ollama_client.generate(model, &prompt).await?;

        print!(" ✅\n🔍 Processing results...");
        io::stdout().flush().unwrap();

        // Clean up the response - remove markdown code blocks and extra formatting
        let cleaned_response = response
            .trim()
            .strip_prefix("```json")
            .unwrap_or(&response)
            .strip_suffix("```")
            .unwrap_or(&response)
            .trim();

        // Try to parse the JSON response
        match serde_json::from_str::<ProcessedResearch>(cleaned_response) {
            Ok(processed) => {
                println!(" ✅");
                Ok(processed)
            },
            Err(e) => {
                println!(" ⚠️");
                // Enhanced fallback with better error info
                eprintln!("JSON parsing failed: {}", e);
                eprintln!("Raw response: {}", response);
                eprintln!("Cleaned response: {}", cleaned_response);

                Ok(ProcessedResearch {
                    summary: cleaned_response.chars().take(300).collect::<String>() + "...",
                    key_facts: vec![
                        "JSON parsing failed - showing raw content".to_string(),
                        format!("Error: {}", e).chars().take(100).collect::<String>(),
                    ],
                    relevant_sections: vec![],
                    therapeutic_relevance: "Content available but needs manual processing".to_string(),
                })
            }
        }
    }
}

#[async_trait::async_trait]
impl Agent for ResearchAgent {
    fn name(&self) -> &str {
        "research"
    }

    async fn capabilities(&self) -> Vec<Capability> {
        vec![
            Capability {
                name: "url_research".to_string(),
                description: "Fetch and analyze content from whitelisted URLs".to_string(),
                input_types: vec!["url".to_string(), "text_with_url".to_string()],
                output_types: vec!["research_result".to_string(), "processed_research".to_string()],
            },
            Capability {
                name: "wikipedia_search".to_string(),
                description: "Search Wikipedia for mental health topics".to_string(),
                input_types: vec!["mental_health_query".to_string()],
                output_types: vec!["research_result".to_string()],
            },
            Capability {
                name: "intent_detection".to_string(),
                description: "Detect research intent in user messages".to_string(),
                input_types: vec!["text".to_string()],
                output_types: vec!["research_intent".to_string()],
            },
        ]
    }

    async fn can_handle(&self, request: &AgentRequest) -> f32 {
        let intent = self.analyze_intent(&request.input);

        match intent {
            ResearchIntent::DirectUrl(_) => 1.0, // Perfect match for direct URLs
            ResearchIntent::ExplicitResearch(_) => 0.9, // High confidence for explicit research
            ResearchIntent::SuggestedResearch(_) => 0.7, // Good confidence for suggested research
            ResearchIntent::None => 0.0, // Can't handle
        }
    }

    async fn execute(&self, request: AgentRequest) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        let intent = self.analyze_intent(&request.input);

        let result = match intent {
            ResearchIntent::DirectUrl(url) => {
                if self.is_url_whitelisted(&url) {
                    match self.fetch_url(&url).await {
                        Ok(research_result) => {
                            let processed = self.process_with_llm(
                                &research_result.content,
                                &request.input,
                                &request.context.current_model
                            ).await?;

                            format!(
                                "📚 **Research from {}**\n\n**{}**\n\n{}\n\n**Key Facts:**\n{}\n\n**Therapeutic Relevance:** {}\n\n*Source: {}*",
                                research_result.source_domain,
                                research_result.title,
                                processed.summary,
                                processed.key_facts.iter()
                                    .map(|fact| format!("• {}", fact))
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                processed.therapeutic_relevance,
                                research_result.url
                            )
                        }
                        Err(e) => format!("❌ Research failed: {}", e),
                    }
                } else {
                    "❌ URL not whitelisted. Only Wikipedia and Psychology Today are supported.".to_string()
                }
            }

            ResearchIntent::ExplicitResearch(terms) => {
                let main_term = terms.first().unwrap_or(&"mental health".to_string()).clone();
                match self.search_wikipedia(&main_term).await {
                    Ok(research_result) => {
                        let processed = self.process_with_llm(
                            &research_result.content,
                            &main_term,
                            &request.context.current_model
                        ).await?;

                        format!(
                            "📚 **Research: {}**\n\n**{}**\n\n{}\n\n**Key Facts:**\n{}\n\n**Therapeutic Relevance:** {}\n\n*Source: {}*",
                            main_term,
                            research_result.title,
                            processed.summary,
                            processed.key_facts.iter()
                                .map(|fact| format!("• {}", fact))
                                .collect::<Vec<_>>()
                                .join("\n"),
                            processed.therapeutic_relevance,
                            research_result.url
                        )
                    }
                    Err(e) => format!("❌ Wikipedia search failed: {}", e),
                }
            }

            ResearchIntent::SuggestedResearch(terms) => {
                let main_term = terms.first().unwrap_or(&"mental health".to_string()).clone();
                format!(
                    "🔍 I noticed you mentioned '{}'. Would you like me to research this topic for you? I can search Wikipedia for evidence-based information.",
                    main_term
                )
            }

            ResearchIntent::None => {
                "I don't see any research requests in your message.".to_string()
            }
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(AgentResponse {
            content: result,
            metadata: AgentMetadata {
                agent_name: "research".to_string(),
                confidence: self.can_handle(&request).await,
                processing_time_ms: processing_time,
                sources: vec!["wikipedia.org".to_string(), "psychologytoday.com".to_string()],
                content_type: "markdown".to_string(),
            },
            resources_used: vec!["research_cache".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_url_detection() {
        let ollama_client = Arc::new(crate::inference::OllamaClient::new("http://localhost:11434".to_string()));
        let detector = IntentDetector::new(ollama_client, "gemma3:12b".to_string());

        // Test direct URL
        let input = "Can you read https://en.wikipedia.org/wiki/Depression";
        if let ResearchIntent::DirectUrl(url) = detector.detect_intent(input).await {
            assert_eq!(url, "https://en.wikipedia.org/wiki/Depression");
        } else {
            panic!("Expected DirectUrl intent");
        }

        // Test markdown link
        let input = "Check out [this article](https://www.psychologytoday.com/anxiety)";
        if let ResearchIntent::DirectUrl(url) = detector.detect_intent(input).await {
            assert_eq!(url, "https://www.psychologytoday.com/anxiety");
        } else {
            panic!("Expected DirectUrl intent");
        }
    }

    // TODO: Update these tests for new LLM-based classification
    // #[tokio::test]
    // async fn test_explicit_research() {
    //     // Test will need to mock LLM calls or use real Ollama
    // }

    // #[tokio::test]
    // async fn test_suggested_research() {
    //     // Test will need to mock LLM calls or use real Ollama
    // }

    #[test]
    fn test_url_whitelist() {
        let validator = UrlValidator::new();

        assert!(validator.is_whitelisted("https://en.wikipedia.org/wiki/Anxiety"));
        assert!(validator.is_whitelisted("https://www.psychologytoday.com/article"));
        assert!(!validator.is_whitelisted("https://malicious-site.com/page"));
        assert!(!validator.is_whitelisted("https://google.com"));
    }
}