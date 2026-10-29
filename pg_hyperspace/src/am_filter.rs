//! Multi-column & range filter pushdown + session-level spatial/KAG filter injection.
//!
//! ## Architectural Advantage over `pg_pinecone`
//! `pg_pinecone` supports at most **one** filter column and **only** strategy 1 (`=`).
//! `pg_hyperspace` supports:
//! - Arbitrary composite filter columns (`CREATE INDEX ON t USING hyperspace (emb, tenant_id, price, category)`)
//! - All 5 B-tree comparison strategies (`<` = 1, `<=` = 2, `=` = 3, `>=` = 4, `>` = 5)
//! - Direct translation into HyperspaceDB's 32-Stripe Sharded Payload Index (`Match` & `Range`)
//! - Session-level JSONB filter injection (`SET hyperspace.query_filters = '[{"type":"in_cone",...}]'`)
//!   so spatial AABB (`InBox`), Euclidean ball (`InBall`), and Hyperbolic cone (`InCone`) filters
//!   execute inside the HNSW/SuperKMeans engine during index scans.

use std::collections::HashMap;

use pg_hyperspace_client::HttpFilter;
use pgrx::{PgTupleDesc, prelude::*};

use crate::{QUERY_FILTERS_JSON, backend};

/// Standard PostgreSQL B-tree strategy numbers used by `pg_hyperspace` filter operator classes.
pub(crate) const STRATEGY_LT: u16 = 1;
pub(crate) const STRATEGY_LE: u16 = 2;
pub(crate) const STRATEGY_EQ: u16 = 3;
pub(crate) const STRATEGY_GE: u16 = 4;
pub(crate) const STRATEGY_GT: u16 = 5;

/// Decoded scan keys for a single index scan.
#[derive(Debug, Clone, Default)]
pub(crate) struct DecodedScanFilters {
    /// If `true`, a `WHERE col = NULL` or contradictory bound makes the scan trivially empty.
    pub impossible: bool,
    /// Exact equality matches (`col = literal`).
    pub exact_matches: HashMap<String, String>,
    /// Complex/range/geometric filters (`Range`, `InCone`, `InBox`, `InBall`, etc.).
    pub complex_filters: Vec<HttpFilter>,
    /// Whether the executor should recheck returned tuples (`xs_recheck = true`) — required for
    /// strict `<` / `>` comparisons or float precision boundaries.
    pub needs_recheck: bool,
}

/// Extracts all index scan keys (`(*scan).keyData[0..numberOfKeys]`) and merges any session-level
/// `hyperspace.query_filters` JSONB predicates.
///
/// # Safety
/// `scan` must be a valid `pg_sys::IndexScanDesc` with initialized `keyData`.
pub(crate) unsafe fn extract_scan_filters(
    scan: pg_sys::IndexScanDesc,
    filter_columns: &[String],
) -> DecodedScanFilters {
    let mut out = DecodedScanFilters::default();

    // 1. Parse session-level spatial/geometric filters from `hyperspace.query_filters`, if set.
    let raw_json = backend::guc_string(&QUERY_FILTERS_JSON);
    if !raw_json.is_empty() {
        match serde_json::from_str::<Vec<HttpFilter>>(&raw_json) {
            Ok(extra) => {
                out.complex_filters.extend(extra);
            }
            Err(e) => {
                error!(
                    "pg_hyperspace: invalid JSON in hyperspace.query_filters: {e} \
                     (expected a JSON array of filter objects, e.g. '[{{\"type\":\"in_box\",...}}]')"
                );
            }
        }
    }

    let nkeys = unsafe { (*scan).numberOfKeys }.max(0) as usize;
    if nkeys == 0 || unsafe { (*scan).keyData.is_null() } {
        return out;
    }

    // Track range bounds per column name so `WHERE price >= 10 AND price <= 50` merges into a single Range filter.
    let mut range_bounds: HashMap<String, (Option<f64>, Option<f64>)> = HashMap::new();

    for i in 0..nkeys {
        let sk = unsafe { &*(*scan).keyData.add(i) };

        // A scan key with SK_ISNULL on standard comparison operators never matches any row.
        if (sk.sk_flags & (pg_sys::SK_ISNULL as i32)) != 0 {
            out.impossible = true;
            return out;
        }

        // Index attribute 1 is the vector column; attribute 2..N map to `filter_columns[0..N-1]`.
        let attno = sk.sk_attno;
        if attno < 2 {
            continue;
        }
        let filter_idx = (attno - 2) as usize;
        let Some(col_name) = filter_columns.get(filter_idx) else {
            out.needs_recheck = true;
            continue;
        };

        let strategy = sk.sk_strategy;
        let subtype = sk.sk_subtype;
        let datum = sk.sk_argument;

        match strategy {
            STRATEGY_EQ => {
                if let Some((str_val, inexact)) = unsafe { datum_to_string(datum, subtype) } {
                    out.exact_matches.insert(col_name.clone(), str_val);
                    if inexact {
                        out.needs_recheck = true;
                    }
                } else {
                    out.needs_recheck = true;
                }
            }
            STRATEGY_LT | STRATEGY_LE | STRATEGY_GT | STRATEGY_GE => {
                if let Some(num) = unsafe { datum_to_f64(datum, subtype) } {
                    if !num.is_finite() {
                        out.needs_recheck = true;
                        continue;
                    }
                    let entry = range_bounds.entry(col_name.clone()).or_insert((None, None));
                    match strategy {
                        STRATEGY_GE => {
                            entry.0 = Some(entry.0.map_or(num, |prev| prev.max(num)));
                        }
                        STRATEGY_GT => {
                            entry.0 = Some(entry.0.map_or(num, |prev| prev.max(num)));
                            // HyperspaceDB Range is inclusive [gte, lte], so Postgres rechecks strict `>`.
                            out.needs_recheck = true;
                        }
                        STRATEGY_LE => {
                            entry.1 = Some(entry.1.map_or(num, |prev| prev.min(num)));
                        }
                        STRATEGY_LT => {
                            entry.1 = Some(entry.1.map_or(num, |prev| prev.min(num)));
                            out.needs_recheck = true;
                        }
                        _ => {}
                    }
                } else {
                    out.needs_recheck = true;
                }
            }
            _ => {
                out.needs_recheck = true;
            }
        }
    }

    for (col, (gte, lte)) in range_bounds {
        if let (Some(lo), Some(hi)) = (gte, lte)
            && lo > hi
        {
            out.impossible = true;
            return out;
        }
        out.complex_filters.push(HttpFilter::range(col, gte, lte));
    }

    out
}

/// Extracts all filter column values from a heap tuple or `aminsert` `values`/`isnull` array into
/// a metadata `HashMap<String, String>` ready for HyperspaceDB ingestion.
///
/// For numeric columns, also populates HyperspaceDB's typed metadata shadow entry
/// (`__hs_typed__<col> = {"v": <number>}`) so numeric `Range` queries work at full speed.
///
/// # Safety
/// `values` and `isnull` must be valid arrays of length `>= 1 + filter_columns.len()`, and
/// `index_tupdesc` must match the index relation's tuple descriptor.
pub(crate) unsafe fn extract_insert_metadata(
    index_tupdesc: pg_sys::TupleDesc,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    filter_columns: &[String],
) -> HashMap<String, String> {
    let mut meta = HashMap::with_capacity(filter_columns.len() * 2 + 1);
    let tupdesc = unsafe { PgTupleDesc::from_pg_is_copy(index_tupdesc) };
    let natts = tupdesc.len();

    for (idx, col_name) in filter_columns.iter().enumerate() {
        let att_pos = idx + 1; // 0 is the vector attribute
        if att_pos >= natts {
            break;
        }
        let attr_null = unsafe { *isnull.add(att_pos) };
        if attr_null {
            continue;
        }
        let datum = unsafe { *values.add(att_pos) };
        let Some(attr) = tupdesc.get(att_pos) else {
            continue;
        };
        let typoid = attr.atttypid;

        if let Some((str_val, _)) = unsafe { datum_to_string(datum, typoid) } {
            meta.insert(col_name.clone(), str_val);
        }
        if let Some(num_val) = unsafe { datum_to_f64(datum, typoid) }
            && num_val.is_finite()
        {
            meta.insert(
                format!("__hs_typed__{col_name}"),
                format!("{{\"v\":{num_val}}}"),
            );
        }
    }
    meta
}

/// Converts a scalar PostgreSQL `Datum` into `(String, inexact_float_representation)`.
unsafe fn datum_to_string(datum: pg_sys::Datum, typoid: pg_sys::Oid) -> Option<(String, bool)> {
    unsafe {
        match typoid {
            pg_sys::BOOLOID => {
                let b = bool::from_datum(datum, false)?;
                Some((b.to_string(), false))
            }
            pg_sys::INT2OID => {
                let v = i16::from_datum(datum, false)?;
                Some((v.to_string(), false))
            }
            pg_sys::INT4OID => {
                let v = i32::from_datum(datum, false)?;
                Some((v.to_string(), false))
            }
            pg_sys::INT8OID => {
                let v = i64::from_datum(datum, false)?;
                // JSON numbers beyond 2^53 may lose precision in generic JSON parsers; recheck if > 2^53.
                let inexact = v.unsigned_abs() > (1_u64 << 53);
                Some((v.to_string(), inexact))
            }
            pg_sys::FLOAT4OID => {
                let v = f32::from_datum(datum, false)?;
                Some((v.to_string(), true))
            }
            pg_sys::FLOAT8OID => {
                let v = f64::from_datum(datum, false)?;
                Some((v.to_string(), true))
            }
            pg_sys::TEXTOID | pg_sys::VARCHAROID | pg_sys::BPCHAROID => {
                let s = String::from_datum(datum, false)?;
                Some((s, false))
            }
            _ => None,
        }
    }
}

/// Converts a numeric PostgreSQL `Datum` into `f64` for range comparisons.
unsafe fn datum_to_f64(datum: pg_sys::Datum, typoid: pg_sys::Oid) -> Option<f64> {
    unsafe {
        match typoid {
            pg_sys::INT2OID => i16::from_datum(datum, false).map(f64::from),
            pg_sys::INT4OID => i32::from_datum(datum, false).map(f64::from),
            pg_sys::INT8OID => i64::from_datum(datum, false).map(|v| v as f64),
            pg_sys::FLOAT4OID => f32::from_datum(datum, false).map(f64::from),
            pg_sys::FLOAT8OID => f64::from_datum(datum, false),
            _ => None,
        }
    }
}
