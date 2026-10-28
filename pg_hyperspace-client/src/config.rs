//! Configuration for connecting to a local or remote HyperspaceDB engine.

use std::time::Duration;

/// Default local HyperspaceDB HTTP endpoint.
pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:50050";

/// Default request timeout in milliseconds.
pub const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Default retry attempts for transient network or 5xx/429 errors.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Configuration for [`crate::Client`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Base URL of the HyperspaceDB HTTP server (e.g., `http://127.0.0.1:50050` or cloud URL).
    pub endpoint: String,
    /// Optional API key sent via `x-api-key` header.
    pub api_key: Option<String>,
    /// Optional multi-tenant user ID sent via `x-hyperspace-user-id` header.
    pub user_id: Option<String>,
    /// Per-request global timeout.
    pub timeout: Duration,
    /// Maximum retry attempts on transient transport or HTTP 429/502/503/504 errors.
    pub max_retries: u32,
    /// Custom User-Agent header.
    pub user_agent: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_string(),
            api_key: None,
            user_id: None,
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            max_retries: DEFAULT_MAX_RETRIES,
            user_agent: None,
        }
    }
}

impl Config {
    /// Creates a new configuration pointing at the specified `endpoint`.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        let raw = endpoint.into();
        let trimmed = raw.trim().trim_end_matches('/').to_string();
        Self {
            endpoint: if trimmed.is_empty() {
                DEFAULT_ENDPOINT.to_string()
            } else {
                trimmed
            },
            ..Self::default()
        }
    }

    /// Sets an optional API key. Empty strings are normalized to `None`.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        let k = key.into();
        self.api_key = if k.trim().is_empty() { None } else { Some(k) };
        self
    }

    /// Sets an optional multi-tenant user ID. Empty strings are normalized to `None`.
    #[must_use]
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        let u = user_id.into();
        self.user_id = if u.trim().is_empty() { None } else { Some(u) };
        self
    }

    /// Sets the per-request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the maximum retry attempts for transient errors.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}
