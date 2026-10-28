//! PostgreSQL Background Worker for asynchronous outbox draining (`hyperspace.worker_enabled`).
//!
//! When `pg_hyperspace` is loaded via `shared_preload_libraries = 'pg_hyperspace'`, registers a
//! lightweight background worker that periodically wakes up on its latch and drains queued batches
//! from `hyperspace.outbox` and `hyperspace.edges_outbox` into HyperspaceDB.

use std::time::Duration;

use pgrx::{
    bgworkers::{BackgroundWorker, BackgroundWorkerBuilder, SignalWakeFlags},
    prelude::*,
};

use crate::{WORKER_DATABASE, WORKER_ENABLED, WORKER_NAPTIME_MS, backend, outbox};

/// Registers the static background worker during `_PG_init` when preloaded.
pub(crate) fn register_bgworker_if_preloaded() {
    let preloading = unsafe { pg_sys::process_shared_preload_libraries_in_progress };
    if !preloading {
        return;
    }

    BackgroundWorkerBuilder::new("pg_hyperspace outbox worker")
        .set_function("hyperspace_bgworker_main")
        .set_library("pg_hyperspace")
        .set_start_time(pgrx::bgworkers::BgWorkerStartTime::ConsistentState)
        .enable_spi_access()
        .set_restart_time(Some(Duration::from_secs(5)))
        .load();
}

/// Entrypoint for the `pg_hyperspace` background worker.
#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn hyperspace_bgworker_main(_arg: pg_sys::Datum) {
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);

    let db_name = {
        let configured = backend::guc_string(&WORKER_DATABASE);
        if configured.is_empty() {
            "postgres".to_string()
        } else {
            configured
        }
    };

    BackgroundWorker::connect_worker_to_spi(Some(db_name.as_str()), None);

    while BackgroundWorker::wait_latch(Some(Duration::from_millis(
        WORKER_NAPTIME_MS.get().max(100) as u64,
    ))) {
        if BackgroundWorker::sighup_received() {
            // Reload configuration on SIGHUP
        }
        if !WORKER_ENABLED.get() {
            continue;
        }

        BackgroundWorker::transaction(|| {
            let ext_installed = Spi::get_one::<bool>(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pg_hyperspace')",
            )
            .unwrap_or(Some(false))
            .unwrap_or(false);

            if ext_installed {
                let _ = outbox::drain_all_tables();
            }
        });
    }
}
