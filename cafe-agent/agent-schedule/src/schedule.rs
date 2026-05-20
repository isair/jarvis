use anyhow::{Context, Result};
use chrono::{Datelike, Duration, NaiveDate, Weekday};
use shared::{db::Database, llm::ClaudeClient, task::AgentResult, ScheduleTask};
use sqlx::FromRow;

use crate::claude_plan::try_claude_plan;

#[derive(FromRow)]
pub(crate) struct EmployeeRow {
    id: i64,
    name: String,
    hourly_rate: f64,
}

#[derive(Clone)]
pub(crate) struct DraftShift {
    pub employee_id: i64,
    pub name: String,
    pub date: String,
    pub start: String,
    pub end: String,
    pub hours: f64,
}

pub async fn run(
    db: &Database,
    task: ScheduleTask,
    claude: Option<&ClaudeClient>,
) -> Result<AgentResult> {
    let week_start = parse_week_start(task.week_start.as_deref())?;
    let week_end = week_start + Duration::days(6);
    let employees = sqlx::query_as::<_, EmployeeRow>(
        "SELECT id, name, hourly_rate FROM employees ORDER BY name",
    )
    .fetch_all(db.pool())
    .await
    .context("load employees")?;

    if employees.is_empty() {
        return Ok(AgentResult {
            ok: false,
            summary: "No employees in database. Seed demo data or add staff first.".into(),
            data: serde_json::json!({}),
        });
    }

    let mut planner = "heuristic";
    let (days, draft_rows) = if let Some(client) = claude {
        if let Some((d, rows)) = try_claude_plan(db, client, week_start, week_end).await? {
            planner = "claude";
            (d, rows)
        } else {
            build_heuristic_days(&employees, week_start)
        }
    } else {
        build_heuristic_days(&employees, week_start)
    };

    let persist = task.persist.unwrap_or(false);
    let mut persisted = 0u32;
    if persist {
        persisted = persist_week_shifts(db, week_start, week_end, &draft_rows).await?;
    }

    let summary = if persist {
        format!(
            "Week {} – {} ({planner}): saved {persisted} shift(s).",
            week_start, week_end
        )
    } else {
        format!(
            "Draft week {} – {} ({planner}, {} day(s)). Set persist=true to write shifts.",
            week_start,
            week_end,
            days.len()
        )
    };

    Ok(AgentResult {
        ok: true,
        summary,
        data: serde_json::json!({
            "week_start": week_start.to_string(),
            "week_end": week_end.to_string(),
            "planner": planner,
            "persisted": persist,
            "shifts_written": persisted,
            "days": days,
        }),
    })
}

pub(crate) fn build_heuristic_days(
    employees: &[EmployeeRow],
    week_start: NaiveDate,
) -> (Vec<serde_json::Value>, Vec<DraftShift>) {
    let mut days = Vec::new();
    let mut draft_rows = Vec::new();

    for offset in 0..7 {
        let day = week_start + Duration::days(offset);
        let iso = day.format("%Y-%m-%d").to_string();
        let is_weekend = matches!(day.weekday(), Weekday::Sat | Weekday::Sun);
        let mut shifts = Vec::new();
        for (idx, emp) in employees.iter().enumerate() {
            if is_weekend && idx > 0 {
                continue;
            }
            let (start, end, hours) = if is_weekend {
                ("10:00", "16:00", 6.0)
            } else if idx % 2 == 0 {
                ("08:00", "16:00", 8.0)
            } else {
                ("12:00", "20:00", 8.0)
            };
            draft_rows.push(DraftShift {
                employee_id: emp.id,
                name: emp.name.clone(),
                date: iso.clone(),
                start: start.into(),
                end: end.into(),
                hours,
            });
            shifts.push(serde_json::json!({
                "employee_id": emp.id,
                "name": emp.name,
                "start": start,
                "end": end,
                "hours": hours,
            }));
        }
        days.push(serde_json::json!({
            "date": iso,
            "weekday": day.format("%a").to_string(),
            "shifts": shifts,
        }));
    }

    (days, draft_rows)
}

pub(crate) async fn persist_week_shifts(
    db: &Database,
    week_start: NaiveDate,
    week_end: NaiveDate,
    rows: &[DraftShift],
) -> Result<u32> {
    let start_s = week_start.format("%Y-%m-%d").to_string();
    let end_s = week_end.format("%Y-%m-%d").to_string();

    let mut tx = db.pool().begin().await.context("begin tx")?;
    sqlx::query("DELETE FROM shifts WHERE date >= ? AND date <= ?")
        .bind(&start_s)
        .bind(&end_s)
        .execute(&mut *tx)
        .await
        .context("clear week shifts")?;

    let mut count = 0u32;
    for row in rows {
        sqlx::query(
            "INSERT INTO shifts (employee_id, date, start_time, end_time, hours_worked) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(row.employee_id)
        .bind(&row.date)
        .bind(&row.start)
        .bind(&row.end)
        .bind(row.hours)
        .execute(&mut *tx)
        .await
        .context("insert shift")?;
        count += 1;
    }
    tx.commit().await.context("commit shifts")?;
    Ok(count)
}

fn parse_week_start(raw: Option<&str>) -> Result<NaiveDate> {
    if let Some(s) = raw {
        let s = s.trim();
        if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            return Ok(d);
        }
    }
    let today = chrono::Utc::now().date_naive();
    let weekday = today.weekday().num_days_from_monday();
    Ok(today - Duration::days(weekday as i64))
}
