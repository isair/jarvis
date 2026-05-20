use anyhow::{Context, Result};
use figment::{providers::Format, Figment};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub anthropic: AnthropicConfig,
    pub weather: WeatherConfig,
    pub database: DatabaseConfig,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct AnthropicConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default = "default_claude_model")]
    pub model: String,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct WeatherConfig {
    #[serde(default = "default_lat")]
    pub latitude: f64,
    #[serde(default = "default_lon")]
    pub longitude: f64,
    #[serde(default = "default_city")]
    pub city: String,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct DatabaseConfig {
    #[serde(default = "default_db_path")]
    pub path: String,
}

fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    8787
}
fn default_claude_model() -> String {
    "claude-sonnet-4-20250514".into()
}
fn default_anthropic_base_url() -> String {
    "https://api.anthropic.com".into()
}
fn default_lat() -> f64 {
    56.9496
}
fn default_lon() -> f64 {
    24.1052
}
fn default_city() -> String {
    "Riga".into()
}
fn default_db_path() -> String {
    "./cafe_agent.db".into()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: String::new(),
            model: default_claude_model(),
        }
    }
}

impl Default for WeatherConfig {
    fn default() -> Self {
        Self {
            latitude: default_lat(),
            longitude: default_lon(),
            city: default_city(),
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: default_db_path(),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            anthropic: AnthropicConfig::default(),
            weather: WeatherConfig::default(),
            database: DatabaseConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let mut figment =
            Figment::from(figment::providers::Serialized::defaults(AppConfig::default()));

        if std::path::Path::new("config.toml").exists() {
            figment = figment.merge(figment::providers::Toml::file("config.toml"));
        } else if std::path::Path::new("cafe-agent/config.toml").exists() {
            figment = figment.merge(figment::providers::Toml::file("cafe-agent/config.toml"));
        }

        figment = figment.merge(figment::providers::Env::prefixed("CAFE_AGENT_").split("_"));

        figment
            .extract()
            .context("failed to load cafe-agent configuration")
    }

    pub fn resolved_anthropic_key(&self) -> Option<String> {
        if !self.anthropic.api_key.trim().is_empty() {
            return Some(self.anthropic.api_key.clone());
        }
        std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
    }

    /// Anthropic API origin (no path). Config `[anthropic].base_url`, then
    /// `ANTHROPIC_BASE_URL`, then `https://api.anthropic.com`.
    pub fn resolved_anthropic_base_url(&self) -> String {
        if !self.anthropic.base_url.trim().is_empty() {
            return self.anthropic.base_url.trim().trim_end_matches('/').to_string();
        }
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            let trimmed = url.trim();
            if !trimmed.is_empty() {
                return trimmed.trim_end_matches('/').to_string();
            }
        }
        default_anthropic_base_url()
            .trim_end_matches('/')
            .to_string()
    }
}
