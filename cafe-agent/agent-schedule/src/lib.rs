mod claude_plan;
mod holidays;
mod payroll;
mod schedule;

use anyhow::Result;
use chrono::{Datelike, Utc};
use shared::{db::Database, llm::ClaudeClient, task::AgentResult, PayrollTask, ScheduleTask};

pub use holidays::{is_lv_public_holiday, lv_public_holidays_for_year};

pub async fn run_schedule(
    db: &Database,
    task: ScheduleTask,
    claude: Option<&ClaudeClient>,
) -> Result<AgentResult> {
    schedule::run(db, task, claude).await
}

pub async fn run_payroll(db: &Database, task: PayrollTask) -> Result<AgentResult> {
    payroll::run(db, task).await
}

/// Insert demo staff + shifts for the current month when the DB is empty.
pub async fn seed_demo_if_empty(db: &Database) -> Result<bool> {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM employees")
        .fetch_one(db.pool())
        .await?;
    if count > 0 {
        return Ok(false);
    }

    sqlx::query(
        "INSERT INTO employees (name, hourly_rate, phone, email) VALUES
         ('Anna Bērziņa', 8.50, NULL, NULL),
         ('Roberts Kalns', 9.00, NULL, NULL)",
    )
    .execute(db.pool())
    .await?;

    let now = Utc::now().date_naive();
    let y = now.year();
    let m = now.month();
    let Some(mut day) = chrono::NaiveDate::from_ymd_opt(y, m, 1) else {
        return Ok(true);
    };
    while day.month() == m {
        let date = day.format("%Y-%m-%d").to_string();
        if !holidays::is_lv_public_holiday(&date) && !matches!(day.weekday(), chrono::Weekday::Sun) {
            let _ = sqlx::query(
                "INSERT INTO shifts (employee_id, date, start_time, end_time, hours_worked) VALUES
                 (1, ?, '08:00', '16:00', 8.0),
                 (2, ?, '12:00', '20:00', 8.0)",
            )
            .bind(&date)
            .bind(&date)
            .execute(db.pool())
            .await;
        }
        day += chrono::Duration::days(1);
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::Database;

    #[tokio::test]
    async fn payroll_after_seed() {
        let db = Database::connect(":memory:").await.expect("db");
        seed_demo_if_empty(&db).await.expect("seed");
        let result = run_payroll(&db, PayrollTask { month: None })
            .await
            .expect("payroll");
        assert!(result.ok);
        let emps = result.data.get("employees").and_then(|v| v.as_array());
        assert!(emps.map(|a| !a.is_empty()).unwrap_or(false));
    }

    #[tokio::test]
    async fn heuristic_plan_covers_seven_days() {
        let db = Database::connect(":memory:").await.expect("db");
        sqlx::query("INSERT INTO employees (name, hourly_rate) VALUES ('A', 8.0), ('B', 9.0)")
            .execute(db.pool())
            .await
            .expect("employees");

        let result = run_schedule(
            &db,
            ScheduleTask {
                week_start: Some("2026-05-12".into()),
                persist: Some(false),
            },
            None,
        )
        .await
        .expect("schedule");
        assert!(result.ok);
        assert_eq!(
            result.data.get("planner").and_then(|v| v.as_str()),
            Some("heuristic")
        );
        let days = result
            .data
            .get("days")
            .and_then(|v| v.as_array())
            .expect("days array");
        assert_eq!(days.len(), 7);
        let first = days[0].get("date").and_then(|v| v.as_str()).unwrap();
        assert_eq!(first, "2026-05-12");
    }

    #[tokio::test]
    async fn heuristic_weekend_single_shift() {
        let db = Database::connect(":memory:").await.expect("db");
        sqlx::query("INSERT INTO employees (name, hourly_rate) VALUES ('A', 8.0), ('B', 9.0)")
            .execute(db.pool())
            .await
            .expect("employees");

        let result = run_schedule(
            &db,
            ScheduleTask {
                week_start: Some("2026-05-12".into()),
                persist: Some(false),
            },
            None,
        )
        .await
        .expect("schedule");
        let days = result
            .data
            .get("days")
            .and_then(|v| v.as_array())
            .expect("days");
        let sat = days
            .iter()
            .find(|d| d.get("date").and_then(|v| v.as_str()) == Some("2026-05-16"))
            .expect("saturday");
        let shifts = sat.get("shifts").and_then(|v| v.as_array()).expect("shifts");
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].get("employee_id").and_then(|v| v.as_i64()), Some(1));
    }

    #[tokio::test]
    async fn schedule_persist_writes_shifts() {
        let db = Database::connect(":memory:").await.expect("db");
        sqlx::query(
            "INSERT INTO employees (name, hourly_rate) VALUES ('A', 8.0), ('B', 9.0)",
        )
        .execute(db.pool())
        .await
        .expect("employees");

        let result = run_schedule(
            &db,
            ScheduleTask {
                week_start: Some("2026-05-12".into()),
                persist: Some(true),
            },
            None,
        )
        .await
        .expect("schedule");
        assert!(result.ok);
        let written = result
            .data
            .get("shifts_written")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        assert!(written > 0);

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM shifts WHERE date >= '2026-05-12' AND date <= '2026-05-18'",
        )
        .fetch_one(db.pool())
        .await
        .expect("count");
        assert_eq!(count as u64, written);
    }
}
