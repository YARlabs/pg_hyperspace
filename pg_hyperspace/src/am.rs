//! PostgreSQL Index Access Method (`CREATE INDEX ... USING hyperspace`) supporting 7 vector metrics
//! and multi-column range/equality filter pushdown.

use std::{collections::HashSet, ffi::c_void};

use pg_hyperspace_client::{
    CreateCollectionRequest, InsertPayload, META_PG_TID_KEY, MetricKind, PgTid, SearchRequest,
};
use pgrx::{
    PgBox, PgMemoryContexts, PgTupleDesc,
    itemptr::{item_pointer_get_both, item_pointer_set_all},
    prelude::*,
    spi::{Spi, quote_identifier, quote_qualified_identifier},
};

use crate::{
    AM_ENABLED, AM_MAX_TOPK, AM_MIN_TOPK, COMPONENT_WEIGHTS_JSON, ITERATIVE_SCAN, IterativeScan,
    MRL_DIMENSION, SYNC_MODE, USE_WASSERSTEIN,
    am_filter::{self, DecodedScanFilters},
    backend::{self, client_for_table},
    catalog::{self, TableRegistryRow},
    outbox, vector_datum,
};

// ---------------------------------------------------------------------------
// Access Method Handler Registration
// ---------------------------------------------------------------------------

#[pg_extern(
    sql = r#"
CREATE FUNCTION hyperspace._am_handler(internal) RETURNS index_am_handler
PARALLEL SAFE IMMUTABLE STRICT LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
CREATE ACCESS METHOD hyperspace TYPE INDEX HANDLER hyperspace._am_handler;
"#,
    requires = ["pg_hyperspace_catalog_schema"]
)]
fn am_handler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut r =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    r.amstrategies = 5;
    r.amsupport = 0;
    r.amoptsprocnum = 0;
    r.amcanorder = false;
    r.amcanorderbyop = true;
    r.amcanbackward = false;
    r.amcanunique = false;
    r.amcanmulticol = true;
    r.amoptionalkey = true;
    r.amsearcharray = false;
    r.amsearchnulls = false;
    r.amstorage = false;
    r.amclusterable = false;
    r.ampredlocks = false;
    r.amcanparallel = false;
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    {
        r.amcanbuildparallel = false;
    }
    r.amcaninclude = false;
    r.amusemaintenanceworkmem = false;
    r.amsummarizing = false;
    r.amparallelvacuumoptions = 0;
    r.amkeytype = pg_sys::InvalidOid;

    r.ambuild = Some(ambuild);
    r.ambuildempty = Some(ambuildempty);
    r.aminsert = Some(aminsert);
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    {
        r.aminsertcleanup = None;
    }
    r.ambulkdelete = Some(ambulkdelete);
    r.amvacuumcleanup = Some(amvacuumcleanup);
    r.amcanreturn = None;
    r.amcostestimate = Some(amcostestimate);
    r.amoptions = None;
    r.amproperty = None;
    r.ambuildphasename = None;
    r.amvalidate = Some(amvalidate);
    r.amadjustmembers = None;
    r.ambeginscan = Some(ambeginscan);
    r.amrescan = Some(amrescan);
    r.amgettuple = Some(amgettuple);
    r.amgetbitmap = None;
    r.amendscan = Some(amendscan);
    r.ammarkpos = None;
    r.amrestrpos = None;
    r.amestimateparallelscan = None;
    r.aminitparallelscan = None;
    r.amparallelrescan = None;

    r.into_pg_boxed()
}

/// Installs operator classes for `USING hyperspace`:
/// - Standard `pgvector` distance operators (`vector_cosine_ops` `<=>`, `vector_l2_ops` `<->`, `vector_ip_ops` `<#>`)
/// - Non-Euclidean `pg_hyperspace` operators (`vector_poincare_ops` `<~>`, `vector_lorentz_ops` `<@>`, `vector_wasserstein_ops` `<|>`)
/// - Multi-column filter operator classes with all 5 B-tree comparison strategies (`<`, `<=`, `=`, `>=`, `>`)
///   for `int2`, `int4`, `int8`, `float4`, `float8`, `text`, and `bool`.
#[pg_extern(
    sql = "CREATE FUNCTION hyperspace._install_opclasses() RETURNS void LANGUAGE c \
           AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';",
    requires = [am_handler]
)]
#[allow(clippy::too_many_lines)]
fn install_opclasses() {
    let Ok(ns) = catalog::pgvector_namespace() else {
        warning!(
            "pg_hyperspace: pgvector ('vector') is not installed yet — run CREATE EXTENSION vector; \
             SELECT hyperspace._install_opclasses(); before creating USING hyperspace indexes"
        );
        return;
    };
    let vector_type = quote_qualified_identifier(ns.as_str(), "vector");
    let ns_ident = quote_identifier(&ns);

    // 1. Standard pgvector operators
    for (opclass, op) in [
        ("vector_cosine_ops", "<=>"),
        ("vector_l2_ops", "<->"),
        ("vector_ip_ops", "<#>"),
    ] {
        if !opclass_exists(opclass) {
            Spi::run(&format!(
                "CREATE OPERATOR CLASS {opclass} FOR TYPE {vector_type} USING hyperspace AS \
                 OPERATOR 1 {ns_ident}.{op} ({vector_type}, {vector_type}) FOR ORDER BY float_ops"
            ))
            .unwrap_or_else(|e| {
                error!("pg_hyperspace: failed to create operator class {opclass}: {e}");
            });
        }
    }

    // 2. Non-Euclidean & Product-Manifold distance operators
    //    (<~> Poincare, <@> Lorentz, <|> Wasserstein, <+> Hybrid 33D Lorentz + Euclidean)
    for (opclass, op_sym, fn_name, f32_fn) in [
        (
            "vector_poincare_ops",
            "<~>",
            "poincare_distance_vec",
            "poincare_distance_f32",
        ),
        (
            "vector_lorentz_ops",
            "<@>",
            "lorentz_distance_vec",
            "lorentz_distance_f32",
        ),
        (
            "vector_wasserstein_ops",
            "<|>",
            "wasserstein_distance_vec",
            "wasserstein_distance_f32",
        ),
        (
            "vector_hybrid_ops",
            "<+>",
            "hybrid_distance_vec",
            "hybrid_distance_f32",
        ),
    ] {
        let _ = Spi::run(&format!(
            "CREATE OR REPLACE FUNCTION hyperspace.{fn_name}({vector_type}, {vector_type}) \
             RETURNS float8 LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT AS \
             $$SELECT hyperspace.{f32_fn}($1::real[], $2::real[])$$"
        ));
        let _ = Spi::run(&format!(
            "DO $$ BEGIN \
                IF NOT EXISTS ( \
                    SELECT 1 FROM pg_catalog.pg_operator o \
                    JOIN pg_catalog.pg_namespace n ON n.oid = o.oprnamespace \
                    WHERE o.oprname = '{op_sym}' AND n.nspname = 'public' \
                ) THEN \
                    EXECUTE 'CREATE OPERATOR public.{op_sym} ( \
                        LEFTARG = {vector_type}, \
                        RIGHTARG = {vector_type}, \
                        FUNCTION = hyperspace.{fn_name} \
                    )'; \
                END IF; \
             END $$;"
        ));
        if !opclass_exists(opclass) {
            let _ = Spi::run(&format!(
                "CREATE OPERATOR CLASS {opclass} FOR TYPE {vector_type} USING hyperspace AS \
                 OPERATOR 1 public.{op_sym} ({vector_type}, {vector_type}) FOR ORDER BY float_ops"
            ));
        }
    }

    // 3. Multi-column filter operator classes (Strategies 1..5: <, <=, =, >=, >)
    for (opclass, ty) in [
        ("int2_ops", "int2"),
        ("int4_ops", "int4"),
        ("int8_ops", "int8"),
        ("float4_ops", "float4"),
        ("float8_ops", "float8"),
        ("text_ops", "text"),
    ] {
        if !opclass_exists(opclass) {
            Spi::run(&format!(
                "CREATE OPERATOR CLASS {opclass} DEFAULT FOR TYPE {ty} USING hyperspace AS \
                 OPERATOR 1 < ({ty}, {ty}), \
                 OPERATOR 2 <= ({ty}, {ty}), \
                 OPERATOR 3 = ({ty}, {ty}), \
                 OPERATOR 4 >= ({ty}, {ty}), \
                 OPERATOR 5 > ({ty}, {ty})"
            ))
            .unwrap_or_else(|e| {
                error!("pg_hyperspace: failed to create filter operator class {opclass}: {e}");
            });
        }
    }

    if !opclass_exists("bool_ops") {
        Spi::run(
            "CREATE OPERATOR CLASS bool_ops DEFAULT FOR TYPE bool USING hyperspace AS \
             OPERATOR 3 = (bool, bool)",
        )
        .unwrap_or_else(|e| {
            error!("pg_hyperspace: failed to create filter operator class bool_ops: {e}");
        });
    }
}

fn opclass_exists(name: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS ( \
            SELECT 1 FROM pg_catalog.pg_opclass c \
            JOIN pg_catalog.pg_am a ON a.oid = c.opcmethod \
            WHERE a.amname = 'hyperspace' AND c.opcname = $1 \
        )",
        &[name.into()],
    )
    .unwrap_or(Some(false))
    .unwrap_or(false)
}

// Event trigger for automatic catalog & outbox cleanup on DROP INDEX / DROP TABLE.
extension_sql!(
    r"
    CREATE FUNCTION hyperspace._on_sql_drop() RETURNS event_trigger
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
    AS $$
    DECLARE
        obj record;
        v_relid oid;
    BEGIN
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'table' LOOP
            DELETE FROM hyperspace.outbox WHERE relid = obj.objid;
            DELETE FROM hyperspace.edges_outbox WHERE relid = obj.objid;
            DELETE FROM hyperspace.tables WHERE relid = obj.objid;
        END LOOP;
        FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() WHERE object_type = 'index' LOOP
            SELECT relid INTO v_relid FROM hyperspace.tables WHERE am_index_relid = obj.objid;
            IF FOUND THEN
                DELETE FROM hyperspace.outbox WHERE relid = v_relid;
                DELETE FROM hyperspace.edges_outbox WHERE relid = v_relid;
                DELETE FROM hyperspace.tables WHERE am_index_relid = obj.objid;
            END IF;
        END LOOP;
    END;
    $$;
    REVOKE EXECUTE ON FUNCTION hyperspace._on_sql_drop() FROM PUBLIC;
    CREATE EVENT TRIGGER pg_hyperspace_on_sql_drop ON sql_drop
        EXECUTE FUNCTION hyperspace._on_sql_drop();
    ",
    name = "pg_hyperspace_am_sql_drop",
    requires = ["pg_hyperspace_catalog_schema", am_handler],
);

// ---------------------------------------------------------------------------
// Index Build & Insert
// ---------------------------------------------------------------------------

#[pg_guard]
unsafe extern "C-unwind" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    _index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    unsafe {
        let heap_relid = (*heap).rd_id;
        let index_relid = (*index).rd_id;
        let index_tupdesc = (*index).rd_att;
        let natts = (*index_tupdesc).natts.max(0) as usize;
        if natts == 0 {
            error!("pg_hyperspace: index must have at least one vector column");
        }

        let indkey = &(*(*index).rd_index).indkey;
        let indkey_slice = indkey.values.as_slice(natts);
        let vec_heap_attnum = indkey_slice[0];
        if vec_heap_attnum <= 0 {
            error!(
                "pg_hyperspace: expression indexes on the vector column are not supported; index a table column directly"
            );
        }

        let heap_tupdesc = PgTupleDesc::from_pg_is_copy((*heap).rd_att);
        let Some(vec_attr) = heap_tupdesc.get((vec_heap_attnum - 1) as usize) else {
            error!("pg_hyperspace: invalid vector column attribute number {vec_heap_attnum}");
        };
        let vector_column = pgrx::name_data_to_str(&vec_attr.attname).to_string();

        // In pgvector, atttypmod stores the exact dimension (> 0).
        let dimension = vec_attr.atttypmod;
        if dimension <= 0 {
            error!(
                "pg_hyperspace: column \"{vector_column}\" must specify an explicit dimension, e.g. vector(768)"
            );
        }

        let mut filter_columns = Vec::with_capacity(natts.saturating_sub(1));
        let mut filter_attnums = Vec::with_capacity(natts.saturating_sub(1));
        for &attnum in &indkey_slice[1..] {
            if attnum <= 0 {
                error!("pg_hyperspace: filter columns must be plain table columns");
            }
            let Some(attr) = heap_tupdesc.get((attnum - 1) as usize) else {
                error!("pg_hyperspace: invalid filter column attribute number {attnum}");
            };
            filter_columns.push(pgrx::name_data_to_str(&attr.attname).to_string());
            filter_attnums.push(attnum);
        }

        let metric = resolve_index_metric(index_relid);
        let collection_name = default_collection_name(heap_relid, index_relid);
        let mrl_cutoff = {
            let v = MRL_DIMENSION.get();
            if v > 0 { Some(v) } else { None }
        };
        let sync_mode = SYNC_MODE.get().as_str().to_string();

        let existing = catalog::lookup_table(heap_relid);
        let endpoint_override = existing.as_ref().and_then(|r| r.endpoint_url.clone());
        let api_key_override = existing.as_ref().and_then(|r| r.api_key.clone());

        let row = TableRegistryRow {
            relid: heap_relid,
            am_index_relid: Some(index_relid),
            collection_name: collection_name.clone(),
            vector_column: vector_column.clone(),
            vector_attnum: vec_heap_attnum,
            dimension,
            metric: metric.as_str().to_string(),
            filter_columns: filter_columns.clone(),
            filter_attnums,
            mrl_cutoff_dimension: mrl_cutoff,
            quantization: None,
            sync_mode,
            endpoint_url: endpoint_override.clone(),
            api_key: api_key_override.clone(),
        };

        let client = client_for_table(endpoint_override.as_deref(), api_key_override.as_deref());
        // Recreate collection cleanly on CREATE INDEX / REINDEX.
        let _ = client.delete_collection(&collection_name);
        client
            .create_collection(&CreateCollectionRequest {
                name: collection_name.clone(),
                dimension: dimension as u32,
                metric: metric.collection_metric_str().to_string(),
                mrl_cutoff_dimension: mrl_cutoff.map(|d| d as u32),
                mrl_rerank_top_k: Some(100),
                quantization: None,
                encrypted: None,
            })
            .unwrap_or_else(|e| backend::raise_client_error(e));

        catalog::upsert_table_registry(&row);

        // Backfill existing rows from heap via SPI in streaming chunks.
        let indexed_tuples = backfill_heap_to_hyperspace(heap, &row, &client);

        let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
            .cast::<pg_sys::IndexBuildResult>();
        (*result).heap_tuples = indexed_tuples as f64;
        (*result).index_tuples = indexed_tuples as f64;
        result
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn ambuildempty(_index: pg_sys::Relation) {}

#[pg_guard]
#[allow(clippy::too_many_arguments)]
unsafe extern "C-unwind" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    unsafe {
        // Skip rows where the vector column itself is NULL.
        if *isnull {
            return false;
        }
        let vector = vector_datum::parse_vector_f64(*values);
        vector_datum::validate_finite_f64(&vector, "INSERT");

        let (block, offset) = item_pointer_get_both(*heap_tid);
        let tid =
            PgTid::try_from_parts(block, offset).unwrap_or_else(|e| error!("pg_hyperspace: {e}"));

        let heap_relid = (*(*index).rd_index).indrelid;
        let Some(row) = catalog::lookup_table(heap_relid) else {
            error!("pg_hyperspace: no hyperspace.tables entry found for relid {heap_relid:?}");
        };

        let mut meta = am_filter::extract_insert_metadata(
            (*index).rd_att,
            values,
            isnull,
            &row.filter_columns,
        );
        meta.insert(META_PG_TID_KEY.to_string(), tid.to_str_repr());

        if row.sync_mode == "async" {
            outbox::enqueue_vector_upsert(heap_relid, tid, &vector, &meta);
        } else {
            let client = client_for_table(row.endpoint_url.as_deref(), row.api_key.as_deref());
            let payload = InsertPayload {
                vector: vector.clone(),
                id: tid.to_point_id(),
                metadata: Some(meta.clone()),
            };
            if let Err(err) = client.insert_point(&row.collection_name, &payload) {
                if err.is_transient() {
                    // Graceful degradation: queue into transactional outbox if local daemon is restarting.
                    outbox::enqueue_vector_upsert(heap_relid, tid, &vector, &meta);
                } else {
                    backend::raise_client_error(err);
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Scan Lifecycle: ambeginscan / amrescan / amgettuple / amendscan
// ---------------------------------------------------------------------------

struct HyperspaceScanState {
    /// Cached registry row resolved once per scan (eliminates per-query SPI lookups).
    registry_row: Option<TableRegistryRow>,
    /// Decoded multi-column & geometric scan filters.
    filters: DecodedScanFilters,
    /// Current batch of `(ItemPointerData, distance)` results.
    matches: Option<Vec<(pg_sys::ItemPointerData, f64)>>,
    /// Index into `matches`.
    cursor: usize,
    /// Most recent `top_k` requested from HyperspaceDB.
    current_topk: usize,
    /// `true` if HyperspaceDB returned fewer results than `current_topk`.
    exhausted: bool,
    /// Deduplication set of `(block, offset)` already returned to Postgres across regrow rounds.
    seen: HashSet<(u32, u16)>,
    /// Last distance served (for `strict_order` iterative scans).
    last_distance: f64,
    /// Active iterative scan mode.
    iterative_mode: IterativeScan,
}

#[pg_guard]
unsafe extern "C-unwind" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    unsafe {
        if norderbys != 1 {
            error!(
                "pg_hyperspace: index scan requires exactly one ORDER BY <vector_col> <op> <query_vec> clause (got {norderbys})"
            );
        }
        let scan = pg_sys::RelationGetIndexScan(index, nkeys, norderbys);
        let heap_relid = (*(*index).rd_index).indrelid;
        let registry_row = catalog::lookup_table(heap_relid);

        let state = HyperspaceScanState {
            registry_row,
            filters: DecodedScanFilters::default(),
            matches: None,
            cursor: 0,
            current_topk: 0,
            exhausted: false,
            seen: HashSet::with_capacity(128),
            last_distance: f64::NEG_INFINITY,
            iterative_mode: ITERATIVE_SCAN.get(),
        };

        // Allocate inside CurrentMemoryContext with automatic Drop on context reset/delete,
        // guaranteeing zero memory leaks even if a mid-scan Postgres ERROR longjmps out.
        (*scan).opaque = PgMemoryContexts::CurrentMemoryContext
            .leak_and_drop_on_delete(state)
            .cast::<c_void>();
        scan
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    _nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    _norderbys: std::ffi::c_int,
) {
    unsafe {
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as usize);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(
                orderbys,
                (*scan).orderByData,
                (*scan).numberOfOrderBys as usize,
            );
        }

        let st = &mut *(*scan).opaque.cast::<HyperspaceScanState>();
        st.matches = None;
        st.cursor = 0;
        st.current_topk = 0;
        st.exhausted = false;
        st.seen.clear();
        st.last_distance = f64::NEG_INFINITY;
        st.iterative_mode = ITERATIVE_SCAN.get();

        let filter_cols = st
            .registry_row
            .as_ref()
            .map_or(&[][..], |r| r.filter_columns.as_slice());
        st.filters = am_filter::extract_scan_filters(scan, filter_cols);
    }
}

#[pg_guard]
#[allow(clippy::too_many_lines)]
unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    unsafe {
        let st = &mut *(*scan).opaque.cast::<HyperspaceScanState>();
        if st.filters.impossible {
            return false;
        }

        loop {
            if let Some(ref matches) = st.matches {
                while st.cursor < matches.len() {
                    let (tid, dist) = matches[st.cursor];
                    st.cursor += 1;
                    let key_pair = item_pointer_get_both(tid);

                    if st.iterative_mode == IterativeScan::StrictOrder {
                        if dist < st.last_distance {
                            st.seen.insert(key_pair);
                            continue;
                        }
                        st.last_distance = dist;
                    }

                    st.seen.insert(key_pair);
                    (*scan).xs_heaptid = tid;
                    (*scan).xs_recheck = st.filters.needs_recheck;
                    (*scan).xs_recheckorderby = false;
                    return true;
                }

                if st.exhausted {
                    return false;
                }

                let max_topk = AM_MAX_TOPK.get().max(1) as usize;
                if st.current_topk >= max_topk {
                    if st.iterative_mode == IterativeScan::Off {
                        error!(
                            "pg_hyperspace: scan exhausted hyperspace.am_max_topk ({max_topk}) — \
                             increase hyperspace.am_max_topk or set hyperspace.iterative_scan = 'relaxed_order'"
                        );
                    }
                    return false;
                }
                st.current_topk = (st.current_topk.saturating_mul(4)).min(max_topk);
            } else {
                let min_topk = AM_MIN_TOPK.get().max(1) as usize;
                let max_topk = AM_MAX_TOPK.get().max(1) as usize;
                st.current_topk = min_topk.min(max_topk);
            }

            let ob = &*(*scan).orderByData;
            if (ob.sk_flags & (pg_sys::SK_ISNULL as i32)) != 0 {
                return false;
            }
            let query_vector = vector_datum::parse_vector_f64(ob.sk_argument);
            vector_datum::validate_finite_f64(&query_vector, "ORDER BY query");

            let Some(ref row) = st.registry_row else {
                let heap_relid = (*(*scan).heapRelation).rd_id;
                error!(
                    "pg_hyperspace: index scan found no hyperspace.tables entry for relid {heap_relid:?}"
                );
            };

            let client = client_for_table(row.endpoint_url.as_deref(), row.api_key.as_deref());
            let mrl_dim = {
                let guc_mrl = MRL_DIMENSION.get();
                if guc_mrl > 0 {
                    Some(guc_mrl as usize)
                } else {
                    row.mrl_cutoff_dimension.map(|d| d as usize)
                }
            };
            let use_wasserstein = USE_WASSERSTEIN.get() || row.metric == "wasserstein";
            let weights = {
                let raw_w = backend::guc_string(&COMPONENT_WEIGHTS_JSON);
                if raw_w.is_empty() {
                    None
                } else {
                    Some(
                        serde_json::from_str::<std::collections::HashMap<String, f32>>(&raw_w)
                            .unwrap_or_else(|e| {
                                error!(
                                    "pg_hyperspace: invalid JSON in hyperspace.component_weights: {e}"
                                );
                            }),
                    )
                }
            };

            let req = SearchRequest {
                vector: query_vector,
                top_k: Some(st.current_topk),
                filter: if st.filters.exact_matches.is_empty() {
                    None
                } else {
                    Some(st.filters.exact_matches.clone())
                },
                filters: if st.filters.complex_filters.is_empty() {
                    None
                } else {
                    Some(st.filters.complex_filters.clone())
                },
                use_wasserstein: if use_wasserstein { Some(true) } else { None },
                mrl_dimension: mrl_dim,
                include_payload: Some(false),
                weights,
            };

            let hits = client
                .search(&row.collection_name, &req)
                .unwrap_or_else(|e| backend::raise_client_error(e));

            st.exhausted = hits.len() < st.current_topk;

            let mut fresh = Vec::with_capacity(hits.len());
            for hit in hits {
                let pg_tid = PgTid::from_point_id_and_meta(hit.id, Some(&hit.metadata))
                    .unwrap_or_else(|e| {
                        error!(
                            "pg_hyperspace: corrupted TID in collection '{}': {e}",
                            row.collection_name
                        )
                    });
                if st.seen.contains(&(pg_tid.block, pg_tid.offset)) {
                    continue;
                }
                let mut item_ptr = pg_sys::ItemPointerData {
                    ip_blkid: pg_sys::BlockIdData { bi_hi: 0, bi_lo: 0 },
                    ip_posid: 0,
                };
                item_pointer_set_all(&mut item_ptr, pg_tid.block, pg_tid.offset);
                fresh.push((item_ptr, hit.distance));
            }

            st.matches = Some(fresh);
            st.cursor = 0;
        }
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {
    // Memory is automatically reclaimed by `PgMemoryContexts::leak_and_drop_on_delete`.
}

// ---------------------------------------------------------------------------
// VACUUM: ambulkdelete & amvacuumcleanup
// ---------------------------------------------------------------------------

#[pg_guard]
unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    unsafe {
        let stats = ensure_vacuum_stats(stats);
        let Some(callback_fn) = callback else {
            return stats;
        };

        let heap_relid = (*(*info).heaprel).rd_id;
        let Some(row) = catalog::lookup_table(heap_relid) else {
            return stats;
        };

        let client = client_for_table(row.endpoint_url.as_deref(), row.api_key.as_deref());
        let remote_points = client
            .list_all_points(&row.collection_name)
            .unwrap_or_default();
        let queued_points = outbox::queued_upsert_point_ids(heap_relid);

        let mut dead_outbox_ids = Vec::new();
        for (pid, tid_str) in queued_points {
            let mut meta = std::collections::HashMap::new();
            if let Some(ts) = tid_str {
                meta.insert(META_PG_TID_KEY.to_string(), ts);
            }
            if let Ok(pg_tid) = PgTid::from_point_id_and_meta(pid, Some(&meta)) {
                let mut item_ptr = pg_sys::ItemPointerData {
                    ip_blkid: pg_sys::BlockIdData { bi_hi: 0, bi_lo: 0 },
                    ip_posid: 0,
                };
                item_pointer_set_all(&mut item_ptr, pg_tid.block, pg_tid.offset);
                if callback_fn(&raw mut item_ptr, callback_state) {
                    dead_outbox_ids.push(i64::from(pid));
                }
            }
        }
        outbox::purge_dead_outbox_rows(heap_relid, &dead_outbox_ids);

        let total_remote = remote_points.len();
        let mut dead_remote_ids = Vec::new();
        for (pid, meta) in remote_points {
            if let Ok(pg_tid) = PgTid::from_point_id_and_meta(pid, Some(&meta)) {
                let mut item_ptr = pg_sys::ItemPointerData {
                    ip_blkid: pg_sys::BlockIdData { bi_hi: 0, bi_lo: 0 },
                    ip_posid: 0,
                };
                item_pointer_set_all(&mut item_ptr, pg_tid.block, pg_tid.offset);
                if callback_fn(&raw mut item_ptr, callback_state) {
                    dead_remote_ids.push(pid);
                }
            }
        }

        if !dead_remote_ids.is_empty() {
            let _ = client.delete_points_batch(&row.collection_name, &dead_remote_ids);
        }

        (*stats).num_index_tuples = (total_remote.saturating_sub(dead_remote_ids.len())) as f64;
        (*stats).tuples_removed = dead_remote_ids.len() as f64;
        stats
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    unsafe { ensure_vacuum_stats(stats) }
}

unsafe fn ensure_vacuum_stats(
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if stats.is_null() {
        unsafe {
            pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
                .cast::<pg_sys::IndexBulkDeleteResult>()
        }
    } else {
        stats
    }
}

// ---------------------------------------------------------------------------
// Planner Cost Estimation & Validation
// ---------------------------------------------------------------------------

#[pg_guard]
#[allow(clippy::too_many_arguments)]
unsafe extern "C-unwind" fn amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        *index_selectivity = 0.01;
        *index_correlation = 0.0;
        *index_pages = 1.0;

        if !AM_ENABLED.get() || (*path).indexorderbys.is_null() {
            *index_startup_cost = pg_sys::disable_cost;
            *index_total_cost = pg_sys::disable_cost;
            #[cfg(feature = "pg18")]
            {
                (*path).path.disabled_nodes += 1;
            }
            return;
        }

        let min_topk = f64::from(AM_MIN_TOPK.get().max(1));
        *index_startup_cost = 2.5;
        *index_total_cost = 2.5 + min_topk * 0.01;
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Internal Build Helpers
// ---------------------------------------------------------------------------

fn default_collection_name(heap_relid: pg_sys::Oid, index_relid: pg_sys::Oid) -> String {
    format!("pg_hs_{}_{}", heap_relid.to_u32(), index_relid.to_u32())
}

fn resolve_index_metric(index_relid: pg_sys::Oid) -> MetricKind {
    let opclass_name = Spi::get_one_with_args::<String>(
        "SELECT c.opcname::text \
         FROM pg_catalog.pg_index i \
         JOIN pg_catalog.pg_opclass c ON c.oid = i.indclass[0] \
         WHERE i.indexrelid = $1",
        &[index_relid.into()],
    )
    .ok()
    .flatten()
    .unwrap_or_else(|| "vector_cosine_ops".to_string());

    MetricKind::parse(&opclass_name).unwrap_or(MetricKind::Cosine)
}

unsafe fn backfill_heap_to_hyperspace(
    heap: pg_sys::Relation,
    row: &TableRegistryRow,
    client: &pg_hyperspace_client::Client,
) -> usize {
    let relid = unsafe { (*heap).rd_id };
    let schema_table = Spi::get_one_with_args::<String>(
        "SELECT pg_catalog.Format('%I.%I', n.nspname, c.relname) \
         FROM pg_catalog.pg_class c \
         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
         WHERE c.oid = $1",
        &[relid.into()],
    )
    .ok()
    .flatten()
    .unwrap_or_default();

    if schema_table.is_empty() {
        return 0;
    }

    let vec_col = quote_identifier(&row.vector_column);
    let query = format!(
        "SELECT ctid::text AS tid_text, {vec_col}::text AS vec_text FROM {schema_table} WHERE {vec_col} IS NOT NULL"
    );

    let payloads: Vec<InsertPayload> = Spi::connect(|spi| {
        let Ok(table) = spi.select(&query, None, &[]) else {
            return Vec::new();
        };
        let mut batch = Vec::new();
        for tuple in table {
            let Some(tid_text) = tuple.get_by_name::<String, _>("tid_text").ok().flatten() else {
                continue;
            };
            let Some(vec_text) = tuple.get_by_name::<String, _>("vec_text").ok().flatten() else {
                continue;
            };
            let Some(tid) = parse_ctid_text(&tid_text) else {
                continue;
            };
            let Some(vector) = parse_pgvector_text(&vec_text) else {
                continue;
            };
            let mut meta = std::collections::HashMap::with_capacity(1);
            meta.insert(META_PG_TID_KEY.to_string(), tid.to_str_repr());
            batch.push(InsertPayload {
                vector,
                id: tid.to_point_id(),
                metadata: Some(meta),
            });
        }
        batch
    });

    if payloads.is_empty() {
        return 0;
    }
    client
        .insert_batch(&row.collection_name, &payloads)
        .unwrap_or_else(|e| backend::raise_client_error(e))
}

fn parse_ctid_text(ctid: &str) -> Option<PgTid> {
    let trimmed = ctid.trim().trim_start_matches('(').trim_end_matches(')');
    let (b_str, o_str) = trimmed.split_once(',')?;
    let block: u32 = b_str.trim().parse().ok()?;
    let offset: u16 = o_str.trim().parse().ok()?;
    PgTid::try_from_parts(block, offset).ok()
}

pub(crate) fn parse_pgvector_text(raw: &str) -> Option<Vec<f64>> {
    let inner = raw.trim().trim_start_matches('[').trim_end_matches(']');
    if inner.is_empty() {
        return None;
    }
    inner
        .split(',')
        .map(|s| s.trim().parse::<f64>().ok())
        .collect()
}
