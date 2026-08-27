#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Config {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
}

impl Config {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
    pub const DEFAULT_MODEL: &'static str = "gpt-4.1-mini";

    pub fn from_env() -> Self {
        Self::from_get(|key| std::env::var(key).ok())
    }

    pub fn from_get<F>(mut get: F) -> Self
    where
        F: FnMut(&str) -> Option<String>,
    {
        Self {
            api_key: nonempty(get("OPENAI_API_KEY")),
            base_url: nonempty(get("OPENAI_BASE_URL"))
                .unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_string()),
            model: nonempty(get("OPENAI_MODEL")).unwrap_or_else(|| Self::DEFAULT_MODEL.to_string()),
        }
    }
}

fn nonempty(value: Option<String>) -> Option<String> {
    value.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn from_map(pairs: &[(&str, &str)]) -> Config {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        Config::from_get(|key| map.get(key).map(|s| (*s).to_string()))
    }

    #[test]
    fn defaults_when_env_is_empty() {
        let cfg = from_map(&[]);
        assert_eq!(
            cfg,
            Config {
                api_key: None,
                base_url: Config::DEFAULT_BASE_URL.to_string(),
                model: Config::DEFAULT_MODEL.to_string(),
            }
        );
    }

    #[test]
    fn overrides_from_env() {
        let cfg = from_map(&[
            ("OPENAI_API_KEY", "sk-test"),
            ("OPENAI_BASE_URL", "http://localhost:8080/v1"),
            ("OPENAI_MODEL", "local-model"),
        ]);
        assert_eq!(
            cfg,
            Config {
                api_key: Some("sk-test".into()),
                base_url: "http://localhost:8080/v1".into(),
                model: "local-model".into(),
            }
        );
    }
}
