use shared::AppConfig;

#[test]
fn resolved_base_url_defaults_to_anthropic_api() {
    const KEY: &str = "ANTHROPIC_BASE_URL";
    let prior = std::env::var(KEY).ok();
    std::env::remove_var(KEY);
    let cfg = AppConfig::default();
    assert_eq!(
        cfg.resolved_anthropic_base_url(),
        "https://api.anthropic.com"
    );
    match prior {
        Some(v) => std::env::set_var(KEY, v),
        None => std::env::remove_var(KEY),
    }
}

#[test]
fn resolved_base_url_prefers_config_toml_value() {
    let mut cfg = AppConfig::default();
    cfg.anthropic.base_url = "http://localhost:4000".into();
    assert_eq!(
        cfg.resolved_anthropic_base_url(),
        "http://localhost:4000"
    );
}

#[test]
fn resolved_base_url_uses_anthropic_base_url_env() {
    const KEY: &str = "ANTHROPIC_BASE_URL";
    let prior = std::env::var(KEY).ok();
    std::env::set_var(KEY, "http://localhost:4000");
    let cfg = AppConfig::default();
    assert_eq!(
        cfg.resolved_anthropic_base_url(),
        "http://localhost:4000"
    );
    match prior {
        Some(v) => std::env::set_var(KEY, v),
        None => std::env::remove_var(KEY),
    }
}

#[test]
fn anthropic_messages_endpoint_joins_v1_messages() {
    assert_eq!(
        shared::llm::anthropic_messages_endpoint("http://localhost:4000/"),
        "http://localhost:4000/v1/messages"
    );
}
