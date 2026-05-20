use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use sqlx::Row;
use std::str::FromStr;
use uuid::Uuid;

use crate::task::{AgentKind, AgentResult};

pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn connect(path: &str) -> Result<Self> {
        let owned;
        let dsn = if path == ":memory:" {
            "sqlite::memory:"
        } else if path.starts_with("sqlite:") {
            path
        } else {
            owned = format!("sqlite:{path}");
            owned.as_str()
        };
        let opts = SqliteConnectOptions::from_str(dsn)?
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts)
            .await
            .with_context(|| format!("connect sqlite at {path}"))?;
        let db = Self { pool };
        db.migrate().await?;
        Ok(db)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                product TEXT NOT NULL,
                quantity INTEGER,
                amount REAL,
                source TEXT
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                hourly_rate REAL NOT NULL,
                phone TEXT,
                email TEXT
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS shifts (
                id INTEGER PRIMARY KEY,
                employee_id INTEGER REFERENCES employees(id),
                date TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                hours_worked REAL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS weather_cache (
                date TEXT PRIMARY KEY,
                temp_max REAL,
                temp_min REAL,
                precipitation REAL,
                fetched_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS task_log (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                payload TEXT NOT NULL,
                result_summary TEXT NOT NULL,
                result_json TEXT NOT NULL,
                ok INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn log_task(
        &self,
        task_id: Uuid,
        agent: AgentKind,
        payload: &serde_json::Value,
        result: &AgentResult,
    ) -> Result<()> {
        let now: DateTime<Utc> = Utc::now();
        sqlx::query(
            r#"
            INSERT INTO task_log (id, agent, payload, result_summary, result_json, ok, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(task_id.to_string())
        .bind(agent.as_str())
        .bind(payload.to_string())
        .bind(&result.summary)
        .bind(result.data.to_string())
        .bind(if result.ok { 1 } else { 0 })
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn upsert_weather_day(
        &self,
        date: &str,
        temp_max: f64,
        temp_min: f64,
        precipitation: f64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"
            INSERT INTO weather_cache (date, temp_max, temp_min, precipitation, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                temp_max = excluded.temp_max,
                temp_min = excluded.temp_min,
                precipitation = excluded.precipitation,
                fetched_at = excluded.fetched_at
            "#,
        )
        .bind(date)
        .bind(temp_max)
        .bind(temp_min)
        .bind(precipitation)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn sales_row_count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as c FROM sales")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get::<i64, _>("c"))
    }
}
