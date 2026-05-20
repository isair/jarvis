use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use shared::{db::Database, task::AgentResult, PayrollTask};
use sqlx::FromRow;

const IIN_RATE: f64 = 0.23;
const VSAOI_EMPLOYEE_RATE: f64 = 0.105;
const VSAOI_EMPLOYER_RATE: f64 = 0.2359;

#[derive(FromRow)]
struct ShiftRow {
    employee_id: i64,
    name: String,
    hourly_rate: f64,
    date: String,
    hours_worked: f64,
}

#[derive(serde::Serialize)]
struct EmployeePayroll {
    employee_id: i64,
    name: String,
    hours: f64,
    gross_eur: f64,
    iin_eur: f64,
    vsaoi_employee_eur: f64,
    net_eur: f64,
    employer_vsaoi_eur: f64,
    total_employer_cost_eur: f64,
}

pub async fn run(db: &Database, task: PayrollTask) -> Result<AgentResult> {
    let (year, month) = parse_month(task.month.as_deref())?;
    let month_prefix = format!("{year}-{month:02}");

    let rows = sqlx::query_as::<_, ShiftRow>(
        r#"
        SELECT s.employee_id, e.name, e.hourly_rate, s.date, COALESCE(s.hours_worked, 0) AS hours_worked
        FROM shifts s
        JOIN employees e ON e.id = s.employee_id
        WHERE s.date LIKE ? || '%'
        "#,
    )
    .bind(&month_prefix)
    .fetch_all(db.pool())
    .await
    .context("load shifts for payroll")?;

    let mut by_employee: std::collections::HashMap<i64, EmployeePayroll> =
        std::collections::HashMap::new();

    for row in rows {
        let hours = row.hours_worked;
        let gross = hours * row.hourly_rate;
        let entry = by_employee.entry(row.employee_id).or_insert(EmployeePayroll {
            employee_id: row.employee_id,
            name: row.name.clone(),
            hours: 0.0,
            gross_eur: 0.0,
            iin_eur: 0.0,
            vsaoi_employee_eur: 0.0,
            net_eur: 0.0,
            employer_vsaoi_eur: 0.0,
            total_employer_cost_eur: 0.0,
        });
        entry.hours += hours;
        entry.gross_eur += gross;
    }

    let mut lines: Vec<EmployeePayroll> = Vec::new();
    for mut ep in by_employee.into_values() {
        ep.iin_eur = round2(ep.gross_eur * IIN_RATE);
        ep.vsaoi_employee_eur = round2(ep.gross_eur * VSAOI_EMPLOYEE_RATE);
        ep.net_eur = round2(ep.gross_eur - ep.iin_eur - ep.vsaoi_employee_eur);
        ep.employer_vsaoi_eur = round2(ep.gross_eur * VSAOI_EMPLOYER_RATE);
        ep.total_employer_cost_eur = round2(ep.gross_eur + ep.employer_vsaoi_eur);
        lines.push(ep);
    }
    lines.sort_by(|a, b| a.name.cmp(&b.name));

    let summary = if lines.is_empty() {
        format!(
            "No shifts recorded for {month_prefix}. Add employees/shifts or run demo seed."
        )
    } else {
        let total_net: f64 = lines.iter().map(|l| l.net_eur).sum();
        format!(
            "Payroll {month_prefix}: {} employee(s), total net {:.2} EUR (IIN {:.0}%, VSAOI {:.1}%).",
            lines.len(),
            total_net,
            IIN_RATE * 100.0,
            VSAOI_EMPLOYEE_RATE * 100.0
        )
    };

    Ok(AgentResult {
        ok: true,
        summary,
        data: serde_json::json!({
            "month": month_prefix,
            "rates": {
                "iin": IIN_RATE,
                "vsaoi_employee": VSAOI_EMPLOYEE_RATE,
                "vsaoi_employer": VSAOI_EMPLOYER_RATE
            },
            "employees": lines,
        }),
    })
}

fn parse_month(raw: Option<&str>) -> Result<(i32, u32)> {
    if let Some(s) = raw {
        let s = s.trim();
        if let Ok(nd) = NaiveDate::parse_from_str(&format!("{s}-01"), "%Y-%m-%d") {
            return Ok((nd.year(), nd.month()));
        }
        if s.len() == 7 {
            let parts: Vec<_> = s.split('-').collect();
            if parts.len() == 2 {
                let y: i32 = parts[0].parse().context("year")?;
                let m: u32 = parts[1].parse().context("month")?;
                return Ok((y, m));
            }
        }
    }
    let now = chrono::Utc::now().date_naive();
    Ok((now.year(), now.month()))
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
