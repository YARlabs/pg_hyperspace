//! Catalog DDL (`hyperspace.tables`, `hyperspace.outbox`, `hyperspace.edges_outbox`) and
//! `pgvector` schema discovery.

use pgrx::{prelude::*, spi::Spi};

// Create the `hyperspace` schema and core registry / outbox tables during `CREATE EXTENSION`.
extension_sql!(
    r"
    CREATE SCHEMA IF NOT EXISTS hyperspace;
    REVOKE ALL ON SCHEMA hyperspace FROM PUBLIC;
    GRANT USAGE ON SCHEMA hyperspace TO PUBLIC;

    CREATE TABLE IF NOT EXISTS hyperspace.tables (
        relid                oid PRIMARY KEY,
        am_index_relid       oid UNIQUE,
        collection_name      text NOT NULL,
        vector_column        text NOT NULL,
        vector_attnum        int2 NOT NULL,
        dimension            int4 NOT NULL,
        metric               text NOT NULL DEFAULT 'cosine',
        filter_columns       text[] NOT NULL DEFAULT '{}',
        filter_attnums       int2[] NOT NULL DEFAULT '{}',
        mrl_cutoff_dimension int4,
        quantization         text,
        sync_mode            text NOT NULL DEFAULT 'direct',
        endpoint_url         text,
        api_key              text,
        last_error           text,
        updated_at           timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    REVOKE ALL ON TABLE hyperspace.tables FROM PUBLIC;

    CREATE TABLE IF NOT EXISTS hyperspace.outbox (
        id         bigserial PRIMARY KEY,
        relid      oid NOT NULL,
        op         text NOT NULL,
        point_id   int8 NOT NULL DEFAULT 0,
        tid_str    text,
        vector     float8[],
        metadata   jsonb,
        attempts   int4 NOT NULL DEFAULT 0,
        last_error text,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    CREATE INDEX IF NOT EXISTS hyperspace_outbox_relid_id_idx
        ON hyperspace.outbox (relid, id);
    REVOKE ALL ON TABLE hyperspace.outbox FROM PUBLIC;
    REVOKE ALL ON SEQUENCE hyperspace.outbox_id_seq FROM PUBLIC;

    CREATE TABLE IF NOT EXISTS hyperspace.edges_outbox (
        id         bigserial PRIMARY KEY,
        relid      oid NOT NULL,
        source_id  int8 NOT NULL,
        target_id  int8 NOT NULL,
        relation   text NOT NULL,
        weight     float4 NOT NULL DEFAULT 1.0,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    CREATE INDEX IF NOT EXISTS hyperspace_edges_outbox_relid_id_idx
        ON hyperspace.edges_outbox (relid, id);
    REVOKE ALL ON TABLE hyperspace.edges_outbox FROM PUBLIC;
    REVOKE ALL ON SEQUENCE hyperspace.edges_outbox_id_seq FROM PUBLIC;
    ",
    name = "pg_hyperspace_catalog_schema",
    bootstrap,
);

/// Metadata row from `hyperspace.tables` describing an indexed PostgreSQL table.
#[derive(Debug, Clone)]
pub(crate) struct TableRegistryRow {
    pub relid: pg_sys::Oid,
    pub am_index_relid: Option<pg_sys::Oid>,
    pub collection_name: String,
    pub vector_column: String,
    pub vector_attnum: i16,
    pub dimension: i32,
    pub metric: String,
    pub filter_columns: Vec<String>,
    pub filter_attnums: Vec<i16>,
    pub mrl_cutoff_dimension: Option<i32>,
    pub quantization: Option<String>,
    pub sync_mode: String,
    pub endpoint_url: Option<String>,
    pub api_key: Option<String>,
}

/// Resolves the schema where `pgvector` (`CREATE EXTENSION vector`) is installed.
pub(crate) fn pgvector_namespace() -> Result<String, String> {
    let ns = Spi::get_one::<String>(
        "SELECT n.nspname::text \
         FROM pg_catalog.pg_extension e \
         JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace \
         WHERE e.extname = 'vector'",
    )
    .map_err(|e| format!("SPI error resolving pgvector extension namespace: {e}"))?;

    ns.ok_or_else(|| "pgvector extension ('vector') is not installed in this database".to_string())
}

/// Looks up a registered table in `hyperspace.tables` by heap `relid`.
pub(crate) fn lookup_table(relid: pg_sys::Oid) -> Option<TableRegistryRow> {
    Spi::connect(|client| {
        let mut rows = client
            .select(
                "SELECT relid, am_index_relid, collection_name, vector_column, vector_attnum, \
                        dimension, metric, filter_columns, filter_attnums, mrl_cutoff_dimension, \
                        quantization, sync_mode, endpoint_url, api_key \
                 FROM hyperspace.tables \
                 WHERE relid = $1",
                Some(1),
                &[relid.into()],
            )
            .expect("pg_hyperspace: failed to query hyperspace.tables");

        rows.next().map(|row| TableRegistryRow {
            relid: row
                .get_by_name::<pg_sys::Oid, _>("relid")
                .expect("relid")
                .expect("relid not null"),
            am_index_relid: row
                .get_by_name::<pg_sys::Oid, _>("am_index_relid")
                .expect("am_index_relid"),
            collection_name: row
                .get_by_name::<String, _>("collection_name")
                .expect("collection_name")
                .expect("collection_name not null"),
            vector_column: row
                .get_by_name::<String, _>("vector_column")
                .expect("vector_column")
                .expect("vector_column not null"),
            vector_attnum: row
                .get_by_name::<i16, _>("vector_attnum")
                .expect("vector_attnum")
                .expect("vector_attnum not null"),
            dimension: row
                .get_by_name::<i32, _>("dimension")
                .expect("dimension")
                .expect("dimension not null"),
            metric: row
                .get_by_name::<String, _>("metric")
                .expect("metric")
                .expect("metric not null"),
            filter_columns: row
                .get_by_name::<Vec<String>, _>("filter_columns")
                .expect("filter_columns")
                .unwrap_or_default(),
            filter_attnums: row
                .get_by_name::<Vec<i16>, _>("filter_attnums")
                .expect("filter_attnums")
                .unwrap_or_default(),
            mrl_cutoff_dimension: row
                .get_by_name::<i32, _>("mrl_cutoff_dimension")
                .expect("mrl_cutoff_dimension"),
            quantization: row
                .get_by_name::<String, _>("quantization")
                .expect("quantization"),
            sync_mode: row
                .get_by_name::<String, _>("sync_mode")
                .expect("sync_mode")
                .unwrap_or_else(|| "direct".to_string()),
            endpoint_url: row
                .get_by_name::<String, _>("endpoint_url")
                .expect("endpoint_url"),
            api_key: row.get_by_name::<String, _>("api_key").expect("api_key"),
        })
    })
}

/// Inserts or updates a row in `hyperspace.tables`.
pub(crate) fn upsert_table_registry(row: &TableRegistryRow) {
    Spi::run_with_args(
        "INSERT INTO hyperspace.tables ( \
            relid, am_index_relid, collection_name, vector_column, vector_attnum, \
            dimension, metric, filter_columns, filter_attnums, mrl_cutoff_dimension, \
            quantization, sync_mode, endpoint_url, api_key, updated_at \
         ) VALUES ( \
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, clock_timestamp() \
         ) ON CONFLICT (relid) DO UPDATE SET \
            am_index_relid = EXCLUDED.am_index_relid, \
            collection_name = EXCLUDED.collection_name, \
            vector_column = EXCLUDED.vector_column, \
            vector_attnum = EXCLUDED.vector_attnum, \
            dimension = EXCLUDED.dimension, \
            metric = EXCLUDED.metric, \
            filter_columns = EXCLUDED.filter_columns, \
            filter_attnums = EXCLUDED.filter_attnums, \
            mrl_cutoff_dimension = EXCLUDED.mrl_cutoff_dimension, \
            quantization = EXCLUDED.quantization, \
            sync_mode = EXCLUDED.sync_mode, \
            endpoint_url = COALESCE(EXCLUDED.endpoint_url, hyperspace.tables.endpoint_url), \
            api_key = COALESCE(EXCLUDED.api_key, hyperspace.tables.api_key), \
            last_error = NULL, \
            updated_at = clock_timestamp()",
        &[
            row.relid.into(),
            row.am_index_relid.into(),
            row.collection_name.clone().into(),
            row.vector_column.clone().into(),
            row.vector_attnum.into(),
            row.dimension.into(),
            row.metric.clone().into(),
            row.filter_columns.clone().into(),
            row.filter_attnums.clone().into(),
            row.mrl_cutoff_dimension.into(),
            row.quantization.clone().into(),
            row.sync_mode.clone().into(),
            row.endpoint_url.clone().into(),
            row.api_key.clone().into(),
        ],
    )
    .expect("pg_hyperspace: failed to upsert hyperspace.tables row");
}

/// Records an error message on `hyperspace.tables` for operator observability (`hyperspace.status()`).
pub(crate) fn record_table_error(relid: pg_sys::Oid, err_msg: &str) {
    let _ = Spi::run_with_args(
        "UPDATE hyperspace.tables SET last_error = $2, updated_at = clock_timestamp() WHERE relid = $1",
        &[relid.into(), err_msg.into()],
    );
}
