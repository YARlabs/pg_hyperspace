//! Thread-local connection-pooled HyperspaceDB client management.
//!
//! ## Performance Optimization over `pg_pinecone`
//! `pg_pinecone` constructs a fresh `Client` (`ureq::Agent`, TLS config, DNS resolver, and
//! connection pool) on every single query and outbox drain step. Because PostgreSQL uses a
//! process-per-connection model, caching a reusable `Arc<Client>` inside a `thread_local!` slot
//! keyed by [`Config`] eliminates per-query agent construction and keeps TCP connections warm
//! across sequential queries and nested-loop index scans.

use std::{cell::RefCell, ffi::CString, sync::Arc, time::Duration};

use pg_hyperspace_client::{Client, Config, Error};
use pgrx::{guc::GucSetting, prelude::*};

use crate::{API_KEY, ENDPOINT_URL, MAX_RETRIES, TIMEOUT_MS, USER_ID};

thread_local! {
    static CLIENT_CACHE: RefCell<Option<(Config, Arc<Client>)>> = const { RefCell::new(None) };
}

/// Reads a `GucSetting<Option<CString>>` into a trimmed `String`.
#[inline]
pub(crate) fn guc_string(guc: &GucSetting<Option<CString>>) -> String {
    guc.get()
        .map(|c| c.to_string_lossy().trim().to_string())
        .unwrap_or_default()
}

/// Builds the effective [`Config`] for the current session or table override.
pub(crate) fn effective_config(
    endpoint_override: Option<&str>,
    api_key_override: Option<&str>,
) -> Config {
    let endpoint = endpoint_override
        .filter(|s| !s.trim().is_empty())
        .map_or_else(|| guc_string(&ENDPOINT_URL), str::to_string);

    let api_key = api_key_override
        .filter(|s| !s.trim().is_empty())
        .map_or_else(|| guc_string(&API_KEY), str::to_string);

    let user_id = guc_string(&USER_ID);
    let timeout_ms = TIMEOUT_MS.get().max(100) as u64;
    let max_retries = MAX_RETRIES.get().max(0) as u32;

    Config::new(endpoint)
        .with_api_key(api_key)
        .with_user_id(user_id)
        .with_timeout(Duration::from_millis(timeout_ms))
        .with_max_retries(max_retries)
}

/// Returns a thread-local cached `Arc<Client>` matching `cfg`, creating a new instance only if
/// the configuration changed.
pub(crate) fn client_for_config(cfg: Config) -> Arc<Client> {
    CLIENT_CACHE.with_borrow_mut(|slot| {
        if let Some((cached_cfg, client)) = slot.as_ref()
            && *cached_cfg == cfg
        {
            return Arc::clone(client);
        }
        let client = Arc::new(Client::new(cfg.clone()));
        *slot = Some((cfg, Arc::clone(&client)));
        client
    })
}

/// Returns the cached [`Client`] for the current session's GUCs and optional table overrides.
pub(crate) fn client_for_table(
    endpoint_override: Option<&str>,
    api_key_override: Option<&str>,
) -> Arc<Client> {
    let cfg = effective_config(endpoint_override, api_key_override);
    client_for_config(cfg)
}

/// Raises a [`pg_hyperspace_client::Error`] as a PostgreSQL `ERROR` with actionable hints.
pub(crate) fn raise_client_error(err: Error) -> ! {
    match err {
        Error::HttpStatus {
            status: 401 | 403,
            message,
        } => {
            ereport!(
                ERROR,
                PgSqlErrorCode::ERRCODE_INVALID_AUTHORIZATION_SPECIFICATION,
                format!("pg_hyperspace: authentication rejected by HyperspaceDB ({message})"),
                "Set hyperspace.api_key via SET hyperspace.api_key = 'hs_...' or hyperspace.set_endpoint()"
            );
        }
        Error::Transport(msg) => {
            ereport!(
                ERROR,
                PgSqlErrorCode::ERRCODE_CONNECTION_FAILURE,
                format!("pg_hyperspace: cannot reach HyperspaceDB engine ({msg})"),
                "Ensure hyperspace-server is running at hyperspace.endpoint_url (default http://127.0.0.1:50050)"
            );
        }
        other => {
            error!("pg_hyperspace: {other}");
        }
    }
}
