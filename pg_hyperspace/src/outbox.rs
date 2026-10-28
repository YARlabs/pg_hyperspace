//! High-throughput transactional outbox for asynchronous vector and KAG edge ingestion.
//!
//! ## Architecture
//! When a table is configured with `sync_mode = 'async'`, `aminsert` writes a compact row to
//! `hyperspace.outbox` inside the caller's ACID transaction (if the transaction rolls back, the
//! outbox row rolls back automatically). The background worker (or `SELECT hyperspace.flush()`)
//! drains pending rows in batches of up to 1,000 vectors via `/api/collections/{name}/insert/batch`.
//!
//! When a table is configured with `sync_mode = 'direct'` (the default for local `hyperspace-server`),
//! `aminsert` streams the point directly into HyperspaceDB's lock-free `WriteBuffer` for
//! zero-lag read-your-writes consistency, falling back to `hyperspace.outbox` only if a transient
//! transport error occurs.

use std::collections::HashMap;

use pg_hyperspace_client::{InsertPayload, LinkEdgeRequest, META_PG_TID_KEY, PgTid};
use pgrx::{JsonB, prelude::*, spi::Spi};

use crate::{
    OUTBOX_MAX_ATTEMPTS,
    backend::{self, client_for_table},
    catalog::{self, TableRegistryRow},
};

/// Maximum outbox rows drained per batch round.
const DRAIN_BATCH_LIMIT: i64 = 1_000;

/// Enqueues a single vector upsert into `hyperspace.outbox`.
pub(crate) fn enqueue_vector_upsert(
    relid: pg_sys::Oid,
    tid: PgTid,
    vector: &[f64],
    metadata: &HashMap<String, String>,
) {
    let point_id = i64::from(tid.to_point_id());
    let tid_str = tid.to_str_repr();
    let meta_json = serde_json::to_value(metadata).unwrap_or(serde_json::Value::Null);

    Spi::run_with_args(
        "INSERT INTO hyperspace.outbox (relid, op, point_id, tid_str, vector, metadata) \
         VALUES ($1, 'upsert', $2, $3, $4, $5)",
        &[
            relid.into(),
            point_id.into(),
            tid_str.into(),
            vector.to_vec().into(),
            JsonB(meta_json).into(),
        ],
    )
    .expect("pg_hyperspace: failed to enqueue vector upsert into hyperspace.outbox");
}

/// Enqueues a KAG directed graph edge into `hyperspace.edges_outbox`.
pub(crate) fn enqueue_edge_link(
    relid: pg_sys::Oid,
    source_id: u32,
    target_id: u32,
    relation: &str,
    weight: f32,
) {
    Spi::run_with_args(
        "INSERT INTO hyperspace.edges_outbox (relid, source_id, target_id, relation, weight) \
         VALUES ($1, $2, $3, $4, $5)",
        &[
            relid.into(),
            i64::from(source_id).into(),
            i64::from(target_id).into(),
            relation.into(),
            weight.into(),
        ],
    )
    .expect("pg_hyperspace: failed to enqueue edge into hyperspace.edges_outbox");
}

/// Purges any still-queued `'upsert'` outbox rows for `relid` matching `dead_point_ids` during `VACUUM`.
pub(crate) fn purge_dead_outbox_rows(relid: pg_sys::Oid, dead_point_ids: &[i64]) {
    if dead_point_ids.is_empty() {
        return;
    }
    Spi::run_with_args(
        "DELETE FROM hyperspace.outbox \
         WHERE relid = $1 AND op = 'upsert' AND point_id = ANY($2::int8[])",
        &[relid.into(), dead_point_ids.to_vec().into()],
    )
    .expect("pg_hyperspace: failed to purge dead outbox rows");
}

/// Returns all distinct `point_id`s currently queued in `hyperspace.outbox` for `relid`.
pub(crate) fn queued_upsert_point_ids(relid: pg_sys::Oid) -> Vec<(u32, Option<String>)> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT point_id, tid_str FROM hyperspace.outbox WHERE relid = $1 AND op = 'upsert'",
                None,
                &[relid.into()],
            )
            .expect("pg_hyperspace: failed to read queued outbox point_ids")
            .filter_map(|row| {
                let pid = row.get_by_name::<i64, _>("point_id").ok()??;
                let tid_str = row.get_by_name::<String, _>("tid_str").ok().flatten();
                u32::try_from(pid).ok().map(|id| (id, tid_str))
            })
            .collect()
    })
}

struct PendingVectorRow {
    id: i64,
    op: String,
    point_id: u32,
    vector: Vec<f64>,
    meta: HashMap<String, String>,
}

/// Drains up to [`DRAIN_BATCH_LIMIT`] pending vector and KAG edge rows for a single `relid`.
///
/// Returns `(vectors_flushed, edges_flushed)`.
pub(crate) fn drain_table_outbox(row: &TableRegistryRow) -> Result<(usize, usize), String> {
    let client = client_for_table(row.endpoint_url.as_deref(), row.api_key.as_deref());
    let vectors_flushed = drain_vector_rows(row, &client)?;
    let edges_flushed = drain_edge_rows(row, &client)?;
    let _ = backend::guc_string(&crate::ENDPOINT_URL);
    Ok((vectors_flushed, edges_flushed))
}

#[allow(clippy::too_many_lines)]
fn drain_vector_rows(
    row: &TableRegistryRow,
    client: &pg_hyperspace_client::Client,
) -> Result<usize, String> {
    let max_attempts = OUTBOX_MAX_ATTEMPTS.get().max(1);
    let pending: Vec<PendingVectorRow> = Spi::connect(|spi| {
        spi.select(
            "SELECT id, op, point_id, tid_str, vector, metadata \
             FROM hyperspace.outbox \
             WHERE relid = $1 AND attempts < $2 \
             ORDER BY id ASC \
             LIMIT $3",
            None,
            &[
                row.relid.into(),
                max_attempts.into(),
                DRAIN_BATCH_LIMIT.into(),
            ],
        )
        .map_err(|e| e.to_string())?
        .map(|r| {
            let id = r.get_by_name::<i64, _>("id").unwrap().unwrap_or(0);
            let op = r
                .get_by_name::<String, _>("op")
                .unwrap()
                .unwrap_or_default();
            let point_id = r.get_by_name::<i64, _>("point_id").unwrap().unwrap_or(0) as u32;
            let tid_str = r.get_by_name::<String, _>("tid_str").unwrap();
            let vector = r
                .get_by_name::<Vec<f64>, _>("vector")
                .unwrap()
                .unwrap_or_default();
            let meta_json = r.get_by_name::<JsonB, _>("metadata").unwrap();
            let mut meta: HashMap<String, String> = meta_json
                .and_then(|jb| serde_json::from_value(jb.0).ok())
                .unwrap_or_default();
            if let Some(ts) = tid_str {
                meta.entry(META_PG_TID_KEY.to_string()).or_insert(ts);
            }
            Ok(PendingVectorRow {
                id,
                op,
                point_id,
                vector,
                meta,
            })
        })
        .collect::<Result<Vec<_>, String>>()
    })?;

    if pending.is_empty() {
        return Ok(0);
    }

    let mut outbox_ids = Vec::with_capacity(pending.len());
    let mut upserts = Vec::with_capacity(pending.len());
    let mut deletes = Vec::new();

    for item in pending {
        outbox_ids.push(item.id);
        match item.op.as_str() {
            "upsert" => {
                upserts.push(InsertPayload {
                    vector: item.vector,
                    id: item.point_id,
                    metadata: if item.meta.is_empty() {
                        None
                    } else {
                        Some(item.meta)
                    },
                });
            }
            "delete" => {
                deletes.push(item.point_id);
            }
            _ => {}
        }
    }

    let res = (|| -> Result<usize, pg_hyperspace_client::Error> {
        let mut count = 0;
        if !upserts.is_empty() {
            count += client.insert_batch(&row.collection_name, &upserts)?;
        }
        if !deletes.is_empty() {
            client.delete_points_batch(&row.collection_name, &deletes)?;
        }
        Ok(count)
    })();

    match res {
        Ok(flushed) => {
            Spi::run_with_args(
                "DELETE FROM hyperspace.outbox WHERE id = ANY($1::int8[])",
                &[outbox_ids.into()],
            )
            .map_err(|e| e.to_string())?;
            Ok(flushed)
        }
        Err(err) => {
            let err_str = err.to_string();
            catalog::record_table_error(row.relid, &err_str);
            if !err.is_transient() {
                let _ = Spi::run_with_args(
                    "UPDATE hyperspace.outbox \
                     SET attempts = attempts + 1, last_error = $2 \
                     WHERE id = ANY($1::int8[])",
                    &[outbox_ids.into(), err_str.clone().into()],
                );
            }
            Err(err_str)
        }
    }
}

fn drain_edge_rows(
    row: &TableRegistryRow,
    client: &pg_hyperspace_client::Client,
) -> Result<usize, String> {
    let pending_edges: Vec<(i64, LinkEdgeRequest)> = Spi::connect(|spi| {
        spi.select(
            "SELECT id, source_id, target_id, relation, weight \
             FROM hyperspace.edges_outbox \
             WHERE relid = $1 \
             ORDER BY id ASC \
             LIMIT $2",
            None,
            &[row.relid.into(), DRAIN_BATCH_LIMIT.into()],
        )
        .map_err(|e| e.to_string())?
        .map(|r| {
            let id = r.get_by_name::<i64, _>("id").unwrap().unwrap_or(0);
            let source_id = r.get_by_name::<i64, _>("source_id").unwrap().unwrap_or(0) as u32;
            let target_id = r.get_by_name::<i64, _>("target_id").unwrap().unwrap_or(0) as u32;
            let relation = r
                .get_by_name::<String, _>("relation")
                .unwrap()
                .unwrap_or_else(|| "REFERENCES".to_string());
            let weight = r.get_by_name::<f32, _>("weight").unwrap().unwrap_or(1.0);
            Ok((
                id,
                LinkEdgeRequest {
                    source_id,
                    target_id,
                    relation,
                    weight,
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()
    })?;

    if pending_edges.is_empty() {
        return Ok(0);
    }

    let edge_row_ids: Vec<i64> = pending_edges.iter().map(|(id, _)| *id).collect();
    let edges: Vec<LinkEdgeRequest> = pending_edges.into_iter().map(|(_, e)| e).collect();
    match client.link_batch(&row.collection_name, &edges) {
        Ok(linked) => {
            Spi::run_with_args(
                "DELETE FROM hyperspace.edges_outbox WHERE id = ANY($1::int8[])",
                &[edge_row_ids.into()],
            )
            .map_err(|e| e.to_string())?;
            Ok(linked)
        }
        Err(err) => {
            let err_str = err.to_string();
            catalog::record_table_error(row.relid, &err_str);
            Err(err_str)
        }
    }
}

/// Drains all tables that currently have queued items in `hyperspace.outbox` or `hyperspace.edges_outbox`.
pub(crate) fn drain_all_tables() -> (usize, usize) {
    let relids: Vec<pg_sys::Oid> = Spi::connect(|spi| {
        spi.select(
            "SELECT DISTINCT relid FROM ( \
                SELECT relid FROM hyperspace.outbox \
                UNION \
                SELECT relid FROM hyperspace.edges_outbox \
             ) q",
            None,
            &[],
        )
        .expect("pg_hyperspace: failed to list active outbox relids")
        .filter_map(|r| r.get_by_name::<pg_sys::Oid, _>("relid").ok().flatten())
        .collect()
    });

    let mut total_vecs = 0;
    let mut total_edges = 0;
    for relid in relids {
        if let Some(row) = catalog::lookup_table(relid)
            && let Ok((v, e)) = drain_table_outbox(&row)
        {
            total_vecs += v;
            total_edges += e;
        }
    }
    (total_vecs, total_edges)
}
