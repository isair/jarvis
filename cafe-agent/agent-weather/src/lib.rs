use anyhow::{Context, Result};
use serde::Deserialize;
use shared::{db::Database, llm::ClaudeClient, task::AgentResult, AppConfig, WeatherTask};

#[derive(Debug, Deserialize)]
struct ForecastResponse {
    daily: DailySeries,
}

#[derive(Debug, Deserialize)]
struct DailySeries {
    time: Vec<String>,
    temperature_2m_max: Vec<f64>,
    temperature_2m_min: Vec<f64>,
    precipitation_sum: Vec<f64>,
}

pub async fn run(
    cfg: &AppConfig,
    db: &Database,
    task: WeatherTask,
    claude: Option<&ClaudeClient>,
) -> Result<AgentResult> {
    let days = task.days.unwrap_or(7).min(14) as usize;
    let url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FRiga&forecast_days={days}",
        cfg.weather.latitude,
        cfg.weather.longitude,
    );

    let forecast: ForecastResponse = reqwest::get(&url)
        .await
        .context("open-meteo request")?
        .json()
        .await
        .context("parse open-meteo json")?;

    let mut rows = Vec::new();
    for i in 0..forecast.daily.time.len().min(days) {
        let date = &forecast.daily.time[i];
        let tmax = forecast.daily.temperature_2m_max[i];
        let tmin = forecast.daily.temperature_2m_min[i];
        let precip = forecast.daily.precipitation_sum[i];
        db.upsert_weather_day(date, tmax, tmin, precip).await?;
        rows.push(serde_json::json!({
            "date": date,
            "temp_max_c": tmax,
            "temp_min_c": tmin,
            "precipitation_mm": precip,
        }));
    }

    let mut summary = format!(
        "{}: {}-day forecast loaded ({} days).",
        cfg.weather.city,
        days,
        rows.len()
    );

    if let Some(client) = claude {
        let sales_n = db.sales_row_count().await.unwrap_or(0);
        let prompt = format!(
            "You help a café owner. Weather forecast JSON:\n{}\n\nHistorical sales rows in DB: {}. \
             Give 3 short bullet insights in plain English (rain vs sales correlation only if data exists).",
            serde_json::to_string_pretty(&rows)?,
            sales_n
        );
        match client
            .complete(
                "Be concise. No markdown headers.",
                &prompt,
            )
            .await
        {
            Ok(text) if !text.is_empty() => summary = text,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("claude weather analysis skipped: {e}");
            }
        }
    }

    Ok(AgentResult {
        ok: true,
        summary,
        data: serde_json::json!({
            "city": cfg.weather.city,
            "latitude": cfg.weather.latitude,
            "longitude": cfg.weather.longitude,
            "days": rows,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::AppConfig;

    #[tokio::test]
    async fn fetch_open_meteo_riga() {
        let cfg = AppConfig {
            server: shared::config::ServerConfig::default(),
            anthropic: shared::config::AnthropicConfig::default(),
            weather: shared::config::WeatherConfig {
                latitude: 56.9496,
                longitude: 24.1052,
                city: "Riga".into(),
            },
            database: shared::config::DatabaseConfig {
                path: ":memory:".into(),
            },
        };
        let db = Database::connect(":memory:").await.expect("db");
        let result = run(&cfg, &db, WeatherTask { days: Some(3) }, None)
            .await
            .expect("weather");
        assert!(result.ok);
        assert!(result.data.get("days").and_then(|d| d.as_array()).map(|a| !a.is_empty()).unwrap_or(false));
    }
}
