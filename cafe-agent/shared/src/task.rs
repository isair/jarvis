use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Email,
    WhatsApp,
    SalesAnalysis,
    WeatherCheck,
    SchedulePlan,
    PayrollCalc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentTask {
    Email(EmailTask),
    WhatsApp(WhatsAppTask),
    SalesAnalysis(SalesTask),
    WeatherCheck(WeatherTask),
    SchedulePlan(ScheduleTask),
    PayrollCalc(PayrollTask),
}

impl AgentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::WhatsApp => "whatsapp",
            Self::SalesAnalysis => "sales_analysis",
            Self::WeatherCheck => "weather_check",
            Self::SchedulePlan => "schedule_plan",
            Self::PayrollCalc => "payroll_calc",
        }
    }
}

impl AgentTask {
    pub fn kind(&self) -> AgentKind {
        match self {
            Self::Email(_) => AgentKind::Email,
            Self::WhatsApp(_) => AgentKind::WhatsApp,
            Self::SalesAnalysis(_) => AgentKind::SalesAnalysis,
            Self::WeatherCheck(_) => AgentKind::WeatherCheck,
            Self::SchedulePlan(_) => AgentKind::SchedulePlan,
            Self::PayrollCalc(_) => AgentKind::PayrollCalc,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmailTask {
    pub action: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhatsAppTask {
    pub action: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SalesTask {
    pub csv_path: Option<String>,
    pub days: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeatherTask {
    pub days: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleTask {
    pub week_start: Option<String>,
    /// When true, replace shifts in the planned week with the draft rows.
    pub persist: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PayrollTask {
    pub month: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task: AgentTask,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub ok: bool,
    pub summary: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    pub task_id: Uuid,
    pub agent: AgentKind,
    pub result: AgentResult,
}
