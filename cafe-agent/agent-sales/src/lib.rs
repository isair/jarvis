use anyhow::{Context, Result};
use shared::{db::Database, task::AgentResult, SalesTask};
use std::path::Path;

pub async fn run(db: &Database, task: SalesTask) -> Result<AgentResult> {
    if let Some(path) = task.csv_path.as_deref() {
        import_csv(db, path).await?;
    }

    let days = task.days.unwrap_or(7) as i64;
    let rows = sqlx::query_as::<_, (String, String, i64, f64)>(
        r#"
        SELECT date, product, COALESCE(quantity, 0), COALESCE(amount, 0.0)
        FROM sales
        WHERE date >= date('now', printf('-%d days', ?))
        ORDER BY amount DESC
        LIMIT 20
        "#,
    )
    .bind(days)
    .fetch_all(db.pool())
    .await
    .context("query sales")?;

    let top: Vec<_> = rows
        .iter()
        .map(|(date, product, qty, amount)| {
            serde_json::json!({
                "date": date,
                "product": product,
                "quantity": qty,
                "amount": amount,
            })
        })
        .collect();

    let total: f64 = rows.iter().map(|r| r.3).sum();
    let summary = if rows.is_empty() {
        "No sales rows in range. Import a CSV with csv_path.".into()
    } else {
        format!(
            "Top {} lines over {} days; total amount {:.2}.",
            rows.len(),
            days,
            total
        )
    };

    Ok(AgentResult {
        ok: true,
        summary,
        data: serde_json::json!({ "days": days, "top": top, "total_amount": total }),
    })
}

async fn import_csv(db: &Database, path: &str) -> Result<u32> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("read csv {}", Path::new(path).display()))?;
    let mut count = 0u32;
    for (i, line) in content.lines().enumerate() {
        if i == 0 && line.to_lowercase().contains("product") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let date = parts[0];
        let product = parts[1];
        let quantity: i64 = parts[2].parse().unwrap_or(0);
        let amount: f64 = parts[3].parse().unwrap_or(0.0);
        sqlx::query(
            "INSERT INTO sales (date, product, quantity, amount, source) VALUES (?, ?, ?, ?, 'csv')",
        )
        .bind(date)
        .bind(product)
        .bind(quantity)
        .bind(amount)
        .execute(db.pool())
        .await?;
        count += 1;
    }
    Ok(count)
}
