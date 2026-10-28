//! `pg_hyperspace`: The Universal Multi-Metric Vector, Hyperbolic Geometry, and Native KAG
//! GraphRAG Extension for PostgreSQL.
//!
//! Backs `pgvector` columns with HyperspaceDB's Thread-per-Core engine, SuperKMeans clustering,
//! 32-Stripe Sharded Payload Index, 33D Lorentz Light-Cones, Poincaré Ball, 1D Wasserstein
//! Optimal Transport, Native CSR Graph (KAG), and Tribunal Hallucination Router.

#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]

use std::ffi::CString;

use pgrx::{
    guc::{GucContext, GucFlags, GucRegistry, GucSetting, PostgresGucEnum},
    prelude::*,
};

mod am;
mod am_filter;
mod backend;
mod catalog;
mod outbox;
mod sql_api;
mod vector_datum;
mod worker;

::pgrx::pg_module_magic!(name, version);

// ---------------------------------------------------------------------------
// GUC Configuration Settings
// ---------------------------------------------------------------------------

/// Base URL of the local or remote HyperspaceDB engine (`hyperspace.endpoint_url`).
pub(crate) static ENDPOINT_URL: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"http://127.0.0.1:50050"));

/// Session or database API key (`hyperspace.api_key`).
pub(crate) static API_KEY: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c""));

/// Optional multi-tenant user ID (`hyperspace.user_id`).
pub(crate) static USER_ID: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c""));

/// Request timeout in milliseconds (`hyperspace.timeout_ms`).
pub(crate) static TIMEOUT_MS: GucSetting<i32> = GucSetting::<i32>::new(5_000);

/// Maximum transient retry attempts (`hyperspace.max_retries`).
pub(crate) static MAX_RETRIES: GucSetting<i32> = GucSetting::<i32>::new(3);

/// Write synchronization mode for new indexes (`direct` vs `async` transactional outbox).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PostgresGucEnum)]
pub(crate) enum SyncMode {
    #[name = c"direct"]
    Direct,
    #[name = c"async"]
    Async,
}

impl SyncMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Async => "async",
        }
    }
}

pub(crate) static SYNC_MODE: GucSetting<SyncMode> = GucSetting::<SyncMode>::new(SyncMode::Direct);

/// Iterative scan behavior when a selective `WHERE` filter or large `LIMIT` drains the initial batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PostgresGucEnum)]
pub(crate) enum IterativeScan {
    #[name = c"auto"]
    Auto,
    #[name = c"off"]
    Off,
    #[name = c"relaxed_order"]
    RelaxedOrder,
    #[name = c"strict_order"]
    StrictOrder,
}

pub(crate) static ITERATIVE_SCAN: GucSetting<IterativeScan> =
    GucSetting::<IterativeScan>::new(IterativeScan::Auto);

/// Planner kill-switch for `USING hyperspace` indexes (`hyperspace.am_enabled`).
pub(crate) static AM_ENABLED: GucSetting<bool> = GucSetting::<bool>::new(true);

/// Initial `top_k` requested on the first index scan batch (`hyperspace.am_min_topk`).
pub(crate) static AM_MIN_TOPK: GucSetting<i32> = GucSetting::<i32>::new(100);

/// Maximum `top_k` an iterative scan may grow to (`hyperspace.am_max_topk`).
pub(crate) static AM_MAX_TOPK: GucSetting<i32> = GucSetting::<i32>::new(10_000);

/// Optional Matryoshka (MRL) cutoff dimension (`hyperspace.mrl_dimension`, `0` = disabled).
pub(crate) static MRL_DIMENSION: GucSetting<i32> = GucSetting::<i32>::new(0);

/// Whether to enable 1D Wasserstein optimal transport reranking (`hyperspace.use_wasserstein`).
pub(crate) static USE_WASSERSTEIN: GucSetting<bool> = GucSetting::<bool>::new(false);

/// Session-level JSON array of complex/spatial filters (`InCone`, `InBox`, `InBall`, `Range`)
/// pushed directly into HyperspaceDB during index scans (`hyperspace.query_filters`).
pub(crate) static QUERY_FILTERS_JSON: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c""));

/// Session-level JSON object of multi-component / hybrid manifold weights
/// (`hyperspace.component_weights`, e.g. `'{"lorentz": 0.7, "euclidean": 0.3}'`).
pub(crate) static COMPONENT_WEIGHTS_JSON: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c""));

/// Consecutive permanent failure limit for outbox entries (`hyperspace.outbox_max_attempts`).
pub(crate) static OUTBOX_MAX_ATTEMPTS: GucSetting<i32> = GucSetting::<i32>::new(10);

/// Whether the background outbox drain worker is enabled (`hyperspace.worker_enabled`).
pub(crate) static WORKER_ENABLED: GucSetting<bool> = GucSetting::<bool>::new(true);

/// Naptime in milliseconds between background worker rounds (`hyperspace.worker_naptime_ms`).
pub(crate) static WORKER_NAPTIME_MS: GucSetting<i32> = GucSetting::<i32>::new(1_000);

/// Target database for the background worker (`hyperspace.worker_database`).
pub(crate) static WORKER_DATABASE: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"postgres"));

// ---------------------------------------------------------------------------
// Extension Initialization (_PG_init)
// ---------------------------------------------------------------------------

#[pg_guard]
#[allow(clippy::too_many_lines)]
pub extern "C-unwind" fn _PG_init() {
    GucRegistry::define_string_guc(
        c"hyperspace.endpoint_url",
        c"Base URL of the local or remote HyperspaceDB engine.",
        c"Defaults to http://127.0.0.1:50050.",
        &ENDPOINT_URL,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"hyperspace.api_key",
        c"API key for authenticating with HyperspaceDB.",
        c"",
        &API_KEY,
        GucContext::Userset,
        GucFlags::SUPERUSER_ONLY,
    );
    GucRegistry::define_string_guc(
        c"hyperspace.user_id",
        c"Optional tenant user ID sent via x-hyperspace-user-id.",
        c"",
        &USER_ID,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.timeout_ms",
        c"HTTP timeout in milliseconds for HyperspaceDB requests.",
        c"",
        &TIMEOUT_MS,
        100,
        600_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.max_retries",
        c"Maximum retry attempts for transient network or 5xx/429 errors.",
        c"",
        &MAX_RETRIES,
        0,
        20,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"hyperspace.sync_mode",
        c"Default write synchronization mode: direct (immediate WriteBuffer) or async (transactional outbox).",
        c"",
        &SYNC_MODE,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"hyperspace.iterative_scan",
        c"Iterative scan behavior when more tuples are needed past the first batch.",
        c"",
        &ITERATIVE_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"hyperspace.am_enabled",
        c"Whether the planner may select a USING hyperspace index.",
        c"",
        &AM_ENABLED,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.am_min_topk",
        c"Initial top_k requested on the first batch of an index scan.",
        c"",
        &AM_MIN_TOPK,
        1,
        100_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.am_max_topk",
        c"Maximum top_k an iterative index scan may grow to.",
        c"",
        &AM_MAX_TOPK,
        1,
        1_000_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.mrl_dimension",
        c"Optional Matryoshka (MRL) fast-tier cutoff dimension (0 = disabled).",
        c"",
        &MRL_DIMENSION,
        0,
        65_536,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"hyperspace.use_wasserstein",
        c"Enable 1D Wasserstein Optimal Transport distance evaluation during search.",
        c"",
        &USE_WASSERSTEIN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"hyperspace.query_filters",
        c"Optional JSON array of spatial/geometric filter objects (in_cone, in_box, in_ball, range).",
        c"",
        &QUERY_FILTERS_JSON,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"hyperspace.component_weights",
        c"Optional JSON object of hybrid component weights (e.g. '{\"lorentz\": 0.7, \"euclidean\": 0.3}').",
        c"",
        &COMPONENT_WEIGHTS_JSON,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.outbox_max_attempts",
        c"Maximum permanent failure attempts before an outbox entry is parked.",
        c"",
        &OUTBOX_MAX_ATTEMPTS,
        1,
        1_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"hyperspace.worker_enabled",
        c"Enable background outbox drain worker.",
        c"",
        &WORKER_ENABLED,
        GucContext::Sighup,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"hyperspace.worker_naptime_ms",
        c"Background worker sleep interval in milliseconds.",
        c"",
        &WORKER_NAPTIME_MS,
        100,
        600_000,
        GucContext::Sighup,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"hyperspace.worker_database",
        c"Database name the background worker connects to.",
        c"",
        &WORKER_DATABASE,
        GucContext::Sighup,
        GucFlags::default(),
    );

    worker::register_bgworker_if_preloaded();
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_non_euclidean_distance_functions() {
        let d_poincare = Spi::get_one::<f64>(
            "SELECT hyperspace.poincare_distance(ARRAY[0.1, 0.2], ARRAY[0.2, 0.3])",
        )
        .expect("SPI")
        .expect("value");
        assert!(d_poincare > 0.0);

        let d_lorentz = Spi::get_one::<f64>(
            "SELECT hyperspace.lorentz_distance( \
                hyperspace.lift_to_lorentz(ARRAY[0.1, 0.2]), \
                hyperspace.lift_to_lorentz(ARRAY[0.2, 0.4]) \
             )",
        )
        .expect("SPI")
        .expect("value");
        assert!(d_lorentz > 0.0);

        let in_cone = Spi::get_one::<bool>(
            "SELECT hyperspace.in_lorentz_cone( \
                hyperspace.lift_to_lorentz(ARRAY[0.1, 0.1]), \
                hyperspace.lift_to_lorentz(ARRAY[0.2, 0.2]), \
                2.0 \
             )",
        )
        .expect("SPI")
        .expect("value");
        assert!(in_cone);
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["shared_preload_libraries = 'pg_hyperspace'"]
    }
}
