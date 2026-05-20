use shared::{AgentTask, WeatherTask};

#[test]
fn weather_task_serialises_with_tag() {
    let task = AgentTask::WeatherCheck(WeatherTask { days: Some(5) });
    let json = serde_json::to_string(&task).expect("serialize");
    assert!(json.contains("weather_check"));
    let back: AgentTask = serde_json::from_str(&json).expect("deserialize");
    assert!(matches!(back, AgentTask::WeatherCheck(_)));
}
