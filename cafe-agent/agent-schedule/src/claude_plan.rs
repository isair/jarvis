use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;
use shared::{json_extract::parse_json_payload, llm::ClaudeClient, Database};

use crate::schedule::DraftShift;

#[derive(Debug, Deserialize)]
struct ClaudeDay {
    date: String,
    shifts: Vec<ClaudeShift>,
}

#[derive(Debug, Deserialize)]
struct ClaudeShift {
    employee_id: i64,
    start: String,
    end: String,
    hours: f64,
}

#[derive(Debug, Deserialize)]
struct ClaudePlan {
    days: Vec<ClaudeDay>,
}

pub async fn try_claude_plan(
    db: &Database,
    claude: &ClaudeClient,
    week_start: NaiveDate,
    week_end: NaiveDate,
) -> Result<Option<(Vec<serde_json::Value>, Vec<DraftShift>)>> {
    let employees = sqlx::query_as::<_, (i64, String, f64)>(
        "SELECT id, name, hourly_rate FROM employees ORDER BY name",
    )
    .fetch_all(db.pool())
    .await
    .context("load employees for claude plan")?;

    if employees.is_empty() {
        return Ok(None);
    }

    let staff_json = serde_json::to_string(
        &employees
            .iter()
            .map(|(id, name, rate)| {
                serde_json::json!({ "employee_id": id, "name": name, "hourly_rate": rate })
            })
            .collect::<Vec<_>>(),
    )?;

    let system = "You are a café shift planner. Reply with a single JSON object only — no markdown, no commentary outside JSON.";
    let user = format!(
        r#"Plan shifts for {week_start} through {week_end} (inclusive).
Staff (use employee_id exactly as given):
{staff_json}

Rules:
- Cover each weekday with reasonable café hours (typically 08:00-20:00 window).
- Weekends: lighter coverage, one person is enough unless busy.
- Each shift needs start, end, hours (numeric).
- Max 2 overlapping shifts per day.

JSON schema:
{{
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "shifts": [
        {{ "employee_id": 1, "start": "08:00", "end": "16:00", "hours": 8.0 }}
      ]
    }}
  ]
}}"#,
        week_start = week_start,
        week_end = week_end,
        staff_json = staff_json,
    );

    let raw = claude
        .complete_with_limit(system, &user, 2048)
        .await
        .context("claude schedule plan")?;
    let plan: ClaudePlan = parse_json_payload(&raw).context("parse claude schedule JSON")?;

    let mut days_json = Vec::new();
    let mut draft_rows = Vec::new();

    for day in plan.days {
        let mut shifts = Vec::new();
        for s in day.shifts {
            let name = employees
                .iter()
                .find(|(id, _, _)| *id == s.employee_id)
                .map(|(_, n, _)| n.clone())
                .unwrap_or_else(|| format!("Employee {}", s.employee_id));
            draft_rows.push(DraftShift {
                employee_id: s.employee_id,
                name: name.clone(),
                date: day.date.clone(),
                start: s.start.clone(),
                end: s.end.clone(),
                hours: s.hours,
            });
            shifts.push(serde_json::json!({
                "employee_id": s.employee_id,
                "name": name,
                "start": s.start,
                "end": s.end,
                "hours": s.hours,
            }));
        }
        let weekday = NaiveDate::parse_from_str(&day.date, "%Y-%m-%d")
            .ok()
            .map(|d| d.format("%a").to_string())
            .unwrap_or_else(|| "?".into());
        days_json.push(serde_json::json!({
            "date": day.date,
            "weekday": weekday,
            "shifts": shifts,
        }));
    }

    if days_json.is_empty() {
        return Ok(None);
    }

    Ok(Some((days_json, draft_rows)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::json_extract::parse_json_payload;

    #[test]
    fn parses_schedule_plan_json_fixture() {
        let raw = r#"```json
{
  "days": [
    {
      "date": "2026-05-12",
      "shifts": [
        { "employee_id": 1, "start": "08:00", "end": "16:00", "hours": 8.0 }
      ]
    }
  ]
}
```"#;
        let plan: ClaudePlan = parse_json_payload(raw).expect("parse plan");
        assert_eq!(plan.days.len(), 1);
        assert_eq!(plan.days[0].shifts[0].employee_id, 1);
        assert!((plan.days[0].shifts[0].hours - 8.0).abs() < f64::EPSILON);
    }
}
