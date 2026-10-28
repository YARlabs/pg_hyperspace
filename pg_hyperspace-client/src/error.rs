//! Structured error types with transient vs. permanent classification for outbox backoff.

use thiserror::Error;

/// Error returned by [`crate::Client`] operations and spatial validators.
#[derive(Debug, Error)]
pub enum Error {
    /// HTTP status error returned by HyperspaceDB.
    #[error("HyperspaceDB HTTP {status}: {message}")]
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Server response body or status text.
        message: String,
    },

    /// Transport-level error (timeout, connection refused, DNS, TLS).
    #[error("HyperspaceDB transport error: {0}")]
    Transport(String),

    /// JSON serialization or deserialization error.
    #[error("HyperspaceDB serialization error: {0}")]
    Serialization(String),

    /// Invalid input parameter or mathematical domain violation.
    #[error("HyperspaceDB invalid argument: {0}")]
    InvalidArgument(String),

    /// Malformed PostgreSQL TID encoding.
    #[error("HyperspaceDB malformed TID: {0}")]
    MalformedTid(String),
}

impl Error {
    /// Returns `true` if the error is transient (network hiccup, timeout, HTTP 429/502/503/504)
    /// and should be retried without counting against permanent failure limits.
    #[must_use]
    pub const fn is_transient(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::HttpStatus { status, .. } => {
                matches!(*status, 408 | 429 | 500 | 502 | 503 | 504)
            }
            Self::Serialization(_) | Self::InvalidArgument(_) | Self::MalformedTid(_) => false,
        }
    }

    /// Returns `true` if the error represents a `404 Not Found`.
    #[must_use]
    pub const fn is_not_found(&self) -> bool {
        matches!(self, Self::HttpStatus { status: 404, .. })
    }
}
