use shared::{task::AgentResult, EmailTask, WhatsAppTask};

const DEFAULT_DASHBOARD_PORT: u16 = 5050;

pub async fn bridge_email(task: EmailTask) -> AgentResult {
    let action = task.action.as_deref();
    post_bridge("email", action).await
}

pub async fn bridge_whatsapp(task: WhatsAppTask) -> AgentResult {
    let action = task.action.as_deref();
    post_bridge("whatsapp", action).await
}

async fn post_bridge(channel: &str, action: Option<&str>) -> AgentResult {
    let port = std::env::var("JARVIS_DASHBOARD_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_DASHBOARD_PORT);
    let url = format!("http://127.0.0.1:{port}/api/cafe-agent/jarvis-bridge");

    let body = serde_json::json!({
        "channel": channel,
        "action": action.unwrap_or("sync"),
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build();

    let client = match client {
        Ok(c) => c,
        Err(e) => {
            return AgentResult {
                ok: false,
                summary: format!("HTTP client error: {e}"),
                data: serde_json::json!({}),
            };
        }
    };

    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let parsed: serde_json::Value =
                serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({ "raw": text }));
            if !status.is_success() {
                return AgentResult {
                    ok: false,
                    summary: parsed
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("jarvis bridge failed")
                        .to_string(),
                    data: parsed,
                };
            }
            let summary = parsed
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("Queued via Jarvis")
                .to_string();
            AgentResult {
                ok: parsed.get("ok").and_then(|v| v.as_bool()).unwrap_or(true),
                summary,
                data: parsed,
            }
        }
        Err(e) => AgentResult {
            ok: false,
            summary: format!(
                "Dashboard not reachable on port {port} ({e}). Start memory_viewer / Jarvis shell first."
            ),
            data: serde_json::json!({ "channel": channel }),
        },
    }
}
