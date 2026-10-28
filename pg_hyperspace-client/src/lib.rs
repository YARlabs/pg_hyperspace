//! `pg_hyperspace-client`: High-performance synchronous client and multi-metric spatial kernel
//! library for the `pg_hyperspace` PostgreSQL extension.
//!
//! Designed with zero PostgreSQL FFI dependencies so all wire protocols, TID bit-packing codecs,
//! and non-Euclidean distance kernels (Poincaré, Lorentz, Wasserstein, Cosine, L2, Inner Product)
//! can be unit- and integration-tested independently.

#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::type_complexity)]

pub mod client;
pub mod config;
pub mod error;
pub mod metric;
pub mod tid;
pub(crate) mod transport;
pub mod types;

pub use client::{Client, MAX_UPSERT_BATCH_SIZE, VERSION};
pub use config::{Config, DEFAULT_ENDPOINT, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_MS};
pub use error::Error;
pub use metric::MetricKind;
pub use tid::{META_PG_TID_KEY, PgTid};
pub use types::{
    BatchInsertRequest, CountRequest, CountResponse, CreateCollectionRequest, GraphNode,
    GraphTraverseRequest, HttpFilter, InduceGraphRequest, InsertPayload, LinkBatchRequest,
    LinkEdgeRequest, MetadataInductionRule, PointRecord, ScrollRequest, SearchRequest,
    SearchResultHit, UnlinkEdgeRequest, VerifyGenerationRequest, VerifyGenerationResponse,
};
