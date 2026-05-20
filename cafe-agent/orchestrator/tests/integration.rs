use std::time::Duration;

#[tokio::test]
async fn health_and_weather_task() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("test.db");
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
[server]
host = "127.0.0.1"
port = 18787

[anthropic]
api_key = ""

[weather]
latitude = 56.9496
longitude = 24.1052
city = "Riga"

[database]
path = "{}"
"#,
            db_path.to_string_lossy().replace('\\', "/")
        ),
    )
    .expect("write config");

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_cafe-orchestrator"))
        .current_dir(&dir)
        .env("RUST_LOG", "error")
        .spawn()
        .expect("spawn orchestrator");

    tokio::time::sleep(Duration::from_millis(800)).await;

    let client = reqwest::Client::new();
    let health: serde_json::Value = client
        .get("http://127.0.0.1:18787/health")
        .send()
        .await
        .expect("health")
        .json()
        .await
        .expect("health json");
    assert_eq!(health.get("status").and_then(|v| v.as_str()), Some("ok"));

    let task_body = serde_json::json!({
        "task": { "type": "weather_check", "days": 2 }
    });
    let resp: serde_json::Value = client
        .post("http://127.0.0.1:18787/task")
        .json(&task_body)
        .send()
        .await
        .expect("task")
        .json()
        .await
        .expect("task json");
    assert!(resp.get("task_id").is_some());
    assert_eq!(
        resp.pointer("/result/ok").and_then(|v| v.as_bool()),
        Some(true)
    );

    child.kill().await.ok();
}

#[tokio::test]
async fn schedule_persist_and_payroll_task() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("payroll.db");
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
[server]
host = "127.0.0.1"
port = 18788

[anthropic]
api_key = ""

[weather]
latitude = 56.9496
longitude = 24.1052
city = "Riga"

[database]
path = "{}"
"#,
            db_path.to_string_lossy().replace('\\', "/")
        ),
    )
    .expect("write config");

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_cafe-orchestrator"))
        .current_dir(&dir)
        .env("RUST_LOG", "error")
        .spawn()
        .expect("spawn orchestrator");

    tokio::time::sleep(Duration::from_millis(1200)).await;

    let client = reqwest::Client::new();
    let base = "http://127.0.0.1:18788";

    let schedule = serde_json::json!({
        "task": { "type": "schedule_plan", "week_start": "2026-05-12", "persist": true }
    });
    let sched_resp: serde_json::Value = client
        .post(format!("{base}/task"))
        .json(&schedule)
        .send()
        .await
        .expect("schedule")
        .json()
        .await
        .expect("schedule json");
    assert_eq!(
        sched_resp.pointer("/result/ok").and_then(|v| v.as_bool()),
        Some(true)
    );
    let written = sched_resp
        .pointer("/result/data/shifts_written")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(written > 0);

    let payroll = serde_json::json!({
        "task": { "type": "payroll_calc", "month": "2026-05" }
    });
    let pay_resp: serde_json::Value = client
        .post(format!("{base}/task"))
        .json(&payroll)
        .send()
        .await
        .expect("payroll")
        .json()
        .await
        .expect("payroll json");
    assert_eq!(
        pay_resp.pointer("/result/ok").and_then(|v| v.as_bool()),
        Some(true)
    );
    let employees = pay_resp
        .pointer("/result/data/employees")
        .and_then(|v| v.as_array());
    assert!(employees.map(|a| !a.is_empty()).unwrap_or(false));

    child.kill().await.ok();
}
