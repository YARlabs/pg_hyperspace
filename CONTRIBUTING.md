# Contributing to `pg_hyperspace`

Thank you for your interest in contributing to **`pg_hyperspace`**! By submitting a pull request or contribution, you agree that your work will be licensed under the repository's [MIT License](LICENSE).

---

## Workspace Architecture

The repository is organized as a Cargo workspace (`edition = "2024"`, `resolver = "3"`) split into two crates to keep pure Rust client logic testable without requiring a running PostgreSQL backend for every unit test:

1. **`pg_hyperspace-client`**:
   - Pure Rust HTTP transport (`ureq 3.3` + `rustls`), `EINTR`-safe socket reader (`transport.rs`), zero-allocation 31-bit `(block, offset)` <-> `u32` `PgTid` codec (`tid.rs`), and 4-way unrolled SIMD-friendly distance kernels (`metric.rs`) for all 7 metrics (`l2`, `cosine`, `dot`, `poincare`, `lorentz`, `wasserstein`, `hybrid`).
2. **`pg_hyperspace`**:
   - PostgreSQL extension built on `pgrx = "=0.19.3"` (`pg16`, `pg17`, `pg18` features).
   - Implements the `USING hyperspace` Index Access Method (`am.rs`), multi-column B-tree strategy 1..5 filter pushdown (`am_filter.rs`), zero-leak TOAST vector datum parser (`vector_datum.rs`), thread-local connection pool (`backend.rs`), transactional outbox (`outbox.rs`), background worker (`worker.rs`), and SQL API (`sql_api.rs`).

---

## Development Setup

### Prerequisites
- **Rust Nightly** (automatically selected via `rust-toolchain.toml`).
- **PostgreSQL 16, 17, or 18** development headers (or `cargo-pgrx` managed installations).
- **HyperspaceDB** (`hyperspace-server` v4.1+) running locally on `http://127.0.0.1:50050` for live integration tests.

### Running Tests & Static Analysis

Every pull request must compile with **zero errors and zero Clippy warnings** under `clippy::pedantic`:

```bash
# 1. Check formatting
cargo fmt --all -- --check

# 2. Run pedantic Clippy across all workspace targets (zero warnings allowed)
DOCS_RS=1 cargo clippy --workspace --all-targets -- -D warnings

# 3. Run unit & live HyperspaceDB integration tests
cargo test -p pg_hyperspace-client

# 4. Run pgrx PostgreSQL extension test suite (when cargo-pgrx is initialized)
cargo pgrx test pg17
```

---

## Engineering Standards & Safety Rules

When modifying the PostgreSQL extension crate (`pg_hyperspace`), strictly adhere to these rules:

1. **Zero TOAST Memory Leaks:**
   - Never call `pg_sys::pg_detoast_datum` without checking whether a new buffer was allocated (`detoasted != raw_ptr`) and freeing it via `pg_sys::pfree`. Always use `vector_datum::parse_vector_f64`.
2. **PostgreSQL `ERROR` / `longjmp` Safety:**
   - Every `extern "C-unwind"` callback registered in `IndexAmRoutine` or `_PG_init` must be annotated with `#[pg_guard]`.
   - Any heap state attached to `(*scan).opaque` must be allocated via `PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(...)` so `Drop` runs deterministically even if PostgreSQL aborts the query mid-scan.
3. **Signal Resilience (`EINTR`):**
   - PostgreSQL backends receive frequent `SIGUSR1` and `SIGALRM` signals from latch wakeups and deadlock timers. All blocking I/O in `pg_hyperspace-client` must retry `ErrorKind::Interrupted` within the configured deadline (`transport.rs`).
4. **English-Only Code & Documentation:**
   - All code comments, docstrings, commit messages, and documentation must be written in clear English.

