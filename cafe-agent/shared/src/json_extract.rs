use anyhow::{Context, Result};
use serde::de::DeserializeOwned;

/// Parse JSON from a model reply (raw object or ```json fenced block).
pub fn parse_json_payload<T: DeserializeOwned>(text: &str) -> Result<T> {
    let trimmed = text.trim();
    if let Ok(v) = serde_json::from_str::<T>(trimmed) {
        return Ok(v);
    }
    if let Some(block) = extract_fenced_json(trimmed) {
        return serde_json::from_str(&block).context("parse fenced JSON");
    }
    if let Some(slice) = find_object_slice(trimmed) {
        return serde_json::from_str(slice).context("parse embedded JSON object");
    }
    anyhow::bail!("no JSON object found in model output")
}

fn extract_fenced_json(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    for marker in ["```json", "```"] {
        if let Some(start) = lower.find(marker) {
            let content_start = start + marker.len();
            let rest = &text[content_start..];
            if let Some(end) = rest.find("```") {
                return Some(rest[..end].trim().to_string());
            }
        }
    }
    None
}

fn find_object_slice(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0i32;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Demo {
        days: Vec<serde_json::Value>,
    }

    #[test]
    fn parses_fenced_json() {
        let raw = r#"Here is the plan:
```json
{"days": []}
```"#;
        let d: Demo = parse_json_payload(raw).expect("parse");
        assert!(d.days.is_empty());
    }
}
