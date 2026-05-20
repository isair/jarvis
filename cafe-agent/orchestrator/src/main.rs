mod jarvis_bridge;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{extract::State, routing::{get, post}, Json, Router};
use shared::{
    llm::ClaudeClient, AgentTask, AppConfig, Database, TaskRequest, TaskResponse,
};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    cfg: Arc<AppConfig>,
    db: Arc<Database>,
    claude: Option<Arc<ClaudeClient>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cfg = Arc::new(AppConfig::load()?);
    let db = Arc::new(Database::connect(&cfg.database.path).await?);
    if agent_schedule::seed_demo_if_empty(&db).await? {
        tracing::info!("☕ seeded demo employees and shifts for café payroll");
    }

    let claude = cfg.resolved_anthropic_key().map(|key| {
        Arc::new(ClaudeClient::new(
            key,
            cfg.resolved_anthropic_base_url(),
            cfg.anthropic.model.clone(),
        ))
    });

    let bind_host = cfg.server.host.clone();
    let bind_port = cfg.server.port;
    let state = AppState { cfg, db, claude };

    let app = Router::new()
        .route("/health", get(health))
        .route("/task", post(handle_task))
        .with_state(state);

    let addr: SocketAddr = format!("{bind_host}:{bind_port}").parse()?;
    tracing::info!("☕ cafe-orchestrator listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(st): State<AppState>) -> Json<serde_json::Value> {
    let sales = st.db.sales_row_count().await.unwrap_or(0);
    Json(serde_json::json!({
        "status": "ok",
        "service": "cafe-orchestrator",
        "sales_rows": sales,
        "claude_configured": st.claude.is_some(),
    }))
}

async fn handle_task(
    State(st): State<AppState>,
    Json(body): Json<TaskRequest>,
) -> Json<TaskResponse> {
    let task_id = Uuid::new_v4();
    let agent = body.task.kind();
    let payload = serde_json::to_value(&body.task).unwrap_or_default();

    let result = dispatch(&st, body.task).await;
    let _ = st
        .db
        .log_task(task_id, agent, &payload, &result)
        .await;

    Json(TaskResponse {
        task_id,
        agent,
        result,
    })
}

async fn dispatch(st: &AppState, task: AgentTask) -> shared::AgentResult {
    match task {
        AgentTask::WeatherCheck(t) => {
            match agent_weather::run(&st.cfg, &st.db, t, st.claude.as_deref())
                .await
            {
                Ok(r) => r,
                Err(e) => shared::AgentResult {
                    ok: false,
                    summary: format!("weather failed: {e}"),
                    data: serde_json::json!({}),
                },
            }
        }
        AgentTask::SalesAnalysis(t) => match agent_sales::run(&st.db, t).await {
            Ok(r) => r,
            Err(e) => shared::AgentResult {
                ok: false,
                summary: format!("sales failed: {e}"),
                data: serde_json::json!({}),
            },
        },
        AgentTask::SchedulePlan(t) => {
            match agent_schedule::run_schedule(&st.db, t, st.claude.as_deref()).await {
                Ok(r) => r,
                Err(e) => shared::AgentResult {
                    ok: false,
                    summary: format!("schedule failed: {e}"),
                    data: serde_json::json!({}),
                },
            }
        }
        AgentTask::PayrollCalc(t) => match agent_schedule::run_payroll(&st.db, t).await {
            Ok(r) => r,
            Err(e) => shared::AgentResult {
                ok: false,
                summary: format!("payroll failed: {e}"),
                data: serde_json::json!({}),
            },
        },
        AgentTask::Email(t) => jarvis_bridge::bridge_email(t).await,
        AgentTask::WhatsApp(t) => jarvis_bridge::bridge_whatsapp(t).await,
    }
}
