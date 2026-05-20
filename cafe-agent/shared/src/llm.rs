use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct ClaudeClient {
    api_key: String,
    base_url: String,
    model: String,
    http: reqwest::Client,
}

/// Build `{base}/v1/messages` for the Anthropic Messages API.
pub fn anthropic_messages_endpoint(base_url: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    format!("{base}/v1/messages")
}

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

impl ClaudeClient {
    pub fn new(api_key: String, base_url: String, model: String) -> Self {
        Self {
            api_key,
            base_url: base_url.trim().trim_end_matches('/').to_string(),
            model,
            http: reqwest::Client::new(),
        }
    }

    pub async fn complete(&self, system: &str, user: &str) -> Result<String> {
        self.complete_with_limit(system, user, 1024).await
    }

    pub async fn complete_with_limit(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<String> {
        let body = MessagesRequest {
            model: self.model.clone(),
            max_tokens,
            messages: vec![
                Message {
                    role: "user".into(),
                    content: format!("{system}\n\n{user}"),
                },
            ],
        };

        let url = anthropic_messages_endpoint(&self.base_url);
        let resp = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("anthropic request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("anthropic HTTP {status}: {text}");
        }

        let parsed: MessagesResponse = resp.json().await.context("parse anthropic response")?;
        let text = parsed
            .content
            .into_iter()
            .find(|b| b.block_type == "text")
            .and_then(|b| b.text)
            .unwrap_or_default();
        Ok(text.trim().to_string())
    }
}

pub fn stub_summary(agent: &str, detail: &str) -> String {
    format!("{agent}: {detail}")
}
