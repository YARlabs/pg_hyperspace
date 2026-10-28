//! Wire-format request and response types matching `hyperspace-server` v4.1.1 HTTP endpoints.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request body for `POST /api/collections`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    /// Collection name.
    pub name: String,
    /// Vector dimension.
    pub dimension: u32,
    /// Metric name (`"l2"`, `"cosine"`, `"dot"`, `"poincare"`, `"lorentz"`, `"wasserstein"`, `"hybrid"`).
    pub metric: String,
    /// Optional Matryoshka (MRL) fast-tier cutoff dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrl_cutoff_dimension: Option<u32>,
    /// Optional MRL rerank pool size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrl_rerank_top_k: Option<u32>,
    /// Optional quantization mode (`"none"`, `"scalar_i8"`, `"binary"`, `"turbo"`, `"pq"`, `"opq"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Optional client-side encryption flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted: Option<bool>,
}

/// Single vector payload for `POST /api/collections/{name}/insert` and batch insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertPayload {
    /// Vector coordinates (`f64` on the wire).
    pub vector: Vec<f64>,
    /// 32-bit point identifier (encodes `PgTid`).
    pub id: u32,
    /// Optional key-value metadata map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

/// Request body for `POST /api/collections/{name}/insert/batch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertRequest {
    /// Batch of vectors to upsert.
    pub vectors: Vec<InsertPayload>,
}

/// Structured filter AST matching `hyperspace-server`'s `HttpFilter`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpFilter {
    /// Filter kind: `"match"`, `"prefix"`, `"range"`, `"in_cone"`, `"in_box"`, `"in_ball"`, `"and"`, `"or"`, `"not"`.
    #[serde(rename = "type")]
    pub filter_type: String,
    /// Metadata key for `"match"`, `"prefix"`, and `"range"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Exact match string for `"match"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Prefix string for `"prefix"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    /// Inclusive lower bound for `"range"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gte: Option<f64>,
    /// Inclusive upper bound for `"range"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lte: Option<f64>,
    /// Cone axes for `"in_cone"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axes: Option<Vec<f64>>,
    /// Cone apertures for `"in_cone"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apertures: Option<Vec<f64>>,
    /// Cone tolerance factor for `"in_cone"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cen: Option<f64>,
    /// Lower corner for `"in_box"` (AABB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_bounds: Option<Vec<f64>>,
    /// Upper corner for `"in_box"` (AABB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_bounds: Option<Vec<f64>>,
    /// Center point for `"in_ball"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center: Option<Vec<f64>>,
    /// Radius for `"in_ball"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// Child filters for `"and"` / `"or"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditions: Option<Vec<HttpFilter>>,
    /// Negated child filter for `"not"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Box<HttpFilter>>,
}

impl HttpFilter {
    /// Creates an exact metadata equality filter (`key == value`).
    #[must_use]
    pub fn eq_match(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            filter_type: "match".to_string(),
            key: Some(key.into()),
            value: Some(value.into()),
            prefix: None,
            gte: None,
            lte: None,
            axes: None,
            apertures: None,
            cen: None,
            min_bounds: None,
            max_bounds: None,
            center: None,
            radius: None,
            conditions: None,
            condition: None,
        }
    }

    /// Creates a metadata prefix filter (`key LIKE 'prefix%'`).
    #[must_use]
    pub fn prefix(key: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            filter_type: "prefix".to_string(),
            key: Some(key.into()),
            value: None,
            prefix: Some(prefix.into()),
            gte: None,
            lte: None,
            axes: None,
            apertures: None,
            cen: None,
            min_bounds: None,
            max_bounds: None,
            center: None,
            radius: None,
            conditions: None,
            condition: None,
        }
    }

    /// Creates a numeric range filter (`gte <= key <= lte`).
    #[must_use]
    pub fn range(key: impl Into<String>, gte: Option<f64>, lte: Option<f64>) -> Self {
        Self {
            filter_type: "range".to_string(),
            key: Some(key.into()),
            value: None,
            prefix: None,
            gte,
            lte,
            axes: None,
            apertures: None,
            cen: None,
            min_bounds: None,
            max_bounds: None,
            center: None,
            radius: None,
            conditions: None,
            condition: None,
        }
    }

    /// Creates a geometric `in_cone` filter.
    #[must_use]
    pub fn in_cone(axes: Vec<f64>, apertures: Vec<f64>, cen: f64) -> Self {
        Self {
            filter_type: "in_cone".to_string(),
            key: None,
            value: None,
            prefix: None,
            gte: None,
            lte: None,
            axes: Some(axes),
            apertures: Some(apertures),
            cen: Some(cen),
            min_bounds: None,
            max_bounds: None,
            center: None,
            radius: None,
            conditions: None,
            condition: None,
        }
    }

    /// Creates a spatial AABB `in_box` filter.
    #[must_use]
    pub fn in_box(min_bounds: Vec<f64>, max_bounds: Vec<f64>) -> Self {
        Self {
            filter_type: "in_box".to_string(),
            key: None,
            value: None,
            prefix: None,
            gte: None,
            lte: None,
            axes: None,
            apertures: None,
            cen: None,
            min_bounds: Some(min_bounds),
            max_bounds: Some(max_bounds),
            center: None,
            radius: None,
            conditions: None,
            condition: None,
        }
    }

    /// Creates a spatial `in_ball` filter.
    #[must_use]
    pub fn in_ball(center: Vec<f64>, radius: f64) -> Self {
        Self {
            filter_type: "in_ball".to_string(),
            key: None,
            value: None,
            prefix: None,
            gte: None,
            lte: None,
            axes: None,
            apertures: None,
            cen: None,
            min_bounds: None,
            max_bounds: None,
            center: Some(center),
            radius: Some(radius),
            conditions: None,
            condition: None,
        }
    }
}

/// Request body for `POST /api/collections/{name}/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query vector coordinates.
    pub vector: Vec<f64>,
    /// Number of nearest neighbors to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Exact string equality filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<HashMap<String, String>>,
    /// Complex/geometric filter ASTs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Vec<HttpFilter>>,
    /// Whether to rerank/evaluate with Wasserstein distance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_wasserstein: Option<bool>,
    /// Optional Matryoshka cutoff dimension override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrl_dimension: Option<usize>,
    /// Whether to fetch disk sidecar payloads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_payload: Option<bool>,
    /// Optional multi-component weights.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<HashMap<String, f32>>,
}

/// Single hit returned by `POST /api/collections/{name}/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultHit {
    /// 32-bit point ID.
    pub id: u32,
    /// Computed distance.
    pub distance: f64,
    /// Plain string metadata map (contains `_pg_tid` when present).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Typed JSON metadata map.
    #[serde(default)]
    pub typed_metadata: HashMap<String, Value>,
    /// Optional base64-encoded sidecar payload.
    #[serde(default)]
    pub payload: Option<String>,
}

/// Point record returned by `POST /api/collections/{name}/scroll` or `GET /points`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointRecord {
    /// 32-bit point ID.
    pub id: u32,
    /// Vector coordinates.
    #[serde(default)]
    pub vector: Vec<f64>,
    /// Plain string metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Typed JSON metadata.
    #[serde(default)]
    pub typed_metadata: HashMap<String, Value>,
}

/// Request body for `POST /api/collections/{name}/scroll`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScrollRequest {
    /// Maximum points per page (up to 1000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Page offset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    /// Optional filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Vec<HttpFilter>>,
}

/// Request body for `POST /api/collections/{name}/count`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CountRequest {
    /// Optional filter list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Vec<HttpFilter>>,
}

/// Response from `POST /api/collections/{name}/count`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountResponse {
    /// Total matching points.
    pub count: usize,
}

/// Directed edge payload for `POST /api/collections/{name}/graph/link`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkEdgeRequest {
    /// Source point ID.
    pub source_id: u32,
    /// Target point ID.
    pub target_id: u32,
    /// Relation label (`"OWNS"`, `"DEPENDS_ON"`, `"SUPERSEDES"`, `"AMENDS"`, `"CONTAINS"`, `"REFERENCES"`).
    pub relation: String,
    /// Edge weight (default `1.0`).
    pub weight: f32,
}

/// Request body for `POST /api/collections/{name}/graph/links`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkBatchRequest {
    /// Edges to insert into the collection's CSR/Dynamic graph.
    pub edges: Vec<LinkEdgeRequest>,
}

/// Request body for `DELETE /api/collections/{name}/graph/link`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlinkEdgeRequest {
    /// Source point ID.
    pub source_id: u32,
    /// Target point ID.
    pub target_id: u32,
    /// Relation label.
    pub relation: String,
}

/// Declarative metadata edge-induction rule for `POST /api/collections/{name}/graph/induce`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataInductionRule {
    /// Metadata field to inspect.
    pub field: String,
    /// Rule type (`"path_hierarchy"`, `"foreign_key"`, `"exact_match"`).
    pub rule_type: String,
    /// Optional target metadata field for `"foreign_key"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_field: Option<String>,
    /// Relation label to assign.
    pub relation: String,
    /// Optional edge weight.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
}

/// Request body for `POST /api/collections/{name}/graph/induce`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InduceGraphRequest {
    /// Declarative induction rules.
    pub rules: Vec<MetadataInductionRule>,
}

/// Request body for `POST /api/collections/{name}/graph/traverse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraverseRequest {
    /// Seed node ID.
    pub start_id: u32,
    /// HNSW/KAG layer (default `0`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    /// Maximum BFS hop depth (1..=8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    /// Maximum nodes to visit (1..=10,000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_nodes: Option<usize>,
    /// Optional exact metadata filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<HashMap<String, String>>,
    /// Optional complex filter list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Vec<HttpFilter>>,
}

/// Node returned by `POST /api/collections/{name}/graph/traverse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Point ID.
    pub id: u32,
    /// Graph layer.
    pub layer: usize,
    /// Optional vector coordinates.
    #[serde(default)]
    pub vector: Option<Vec<f64>>,
    /// Adjacent neighbor IDs.
    #[serde(default)]
    pub neighbors: Vec<u32>,
    /// Plain string metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Typed JSON metadata.
    #[serde(default)]
    pub typed_metadata: HashMap<String, Value>,
}

/// Request body for `POST /api/collections/{name}/verify-generation` (Tribunal Router).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyGenerationRequest {
    /// LLM-generated answer text.
    pub llm_answer: String,
    /// Context point IDs or text chunks.
    #[serde(default)]
    pub context_ids: Vec<String>,
    /// Optional collection name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// Optional maximum geometric distance threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance_threshold: Option<f64>,
    /// Whether to verify multi-hop geodesic paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub check_geodesic: Option<bool>,
}

/// Response from `POST /api/collections/{name}/verify-generation` (Tribunal Router).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyGenerationResponse {
    /// Geometric trust score in `[0.0, 1.0]`.
    pub trust_score: f64,
    /// `true` if the generation satisfies the geometric & light-cone bounds.
    pub is_valid: bool,
    /// Minimum distance to context nodes.
    pub min_distance: f64,
    /// Average distance to context nodes.
    pub avg_distance: f64,
    /// Geodesic path length across the KAG graph.
    pub geodesic_distance: f64,
    /// Number of hops along the geodesic path.
    pub geodesic_hops: u32,
    /// Whether the Lorentz light-cone causal bound is satisfied.
    pub cone_satisfied: bool,
    /// Diagnostic explanation.
    pub details: String,
    /// Resolved integer point IDs.
    #[serde(default)]
    pub resolved_context_ids: Vec<u32>,
}
