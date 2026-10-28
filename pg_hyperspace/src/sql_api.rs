//! User-facing SQL functions for Multi-Metric Distances, Hyperbolic Cones, Spatial Regions,
//! Native Vector-Graph (KAG) Traversals, Tribunal Hallucination Verification, and Engine Control.

use pg_hyperspace_client::{
    GraphTraverseRequest, LinkEdgeRequest, MetadataInductionRule, UnlinkEdgeRequest,
    VerifyGenerationRequest,
    metric::{
        HYBRID_LORENTZ_HEAD_DIM, angular_cone_contains, compose_hybrid_vector, cosine_distance_f64,
        hybrid_distance_f64, hybrid_weighted_distance_f64, in_ball_contains, in_box_contains,
        l2_distance_f64, lift_euclidean_to_lorentz, lorentz_distance_f64,
        lorentz_light_cone_contains, poincare_distance_f64, project_euclidean_to_poincare,
        wasserstein_1d_distance_f64,
    },
};
use pgrx::{JsonB, prelude::*, spi::Spi};
use serde_json::json;

use crate::{
    backend::{self, client_for_table},
    catalog, outbox,
};

// ---------------------------------------------------------------------------
// 1. Multi-Metric Non-Euclidean Distance & Manifold Projection Functions
// ---------------------------------------------------------------------------

/// Computes Poincaré Ball geodesic distance between two vectors.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn poincare_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        error!(
            "pg_hyperspace: poincare_distance dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
    }
    poincare_distance_f64(&a, &b)
}

/// Computes Lorentz hyperboloid Minkowski geodesic distance between two vectors.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn lorentz_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        error!(
            "pg_hyperspace: lorentz_distance requires matching dimensions >= 2 (got {} and {})",
            a.len(),
            b.len()
        );
    }
    lorentz_distance_f64(&a, &b)
}

/// Computes 1D Wasserstein ($W_1$ Earth Mover's) Optimal Transport distance between two vectors.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn wasserstein_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        error!(
            "pg_hyperspace: wasserstein_distance dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
    }
    wasserstein_1d_distance_f64(&a, &b)
}

/// Computes Poincaré Ball geodesic distance between two `real[]` (`float4[]`) vectors (used by `<~>` operator on `vector`).
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn poincare_distance_f32(a: Vec<f32>, b: Vec<f32>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        error!(
            "pg_hyperspace: poincare_distance_f32 dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
    }
    let a64: Vec<f64> = a.into_iter().map(f64::from).collect();
    let b64: Vec<f64> = b.into_iter().map(f64::from).collect();
    poincare_distance_f64(&a64, &b64)
}

/// Computes Lorentz hyperboloid geodesic distance between two `real[]` (`float4[]`) vectors (used by `<@>` operator on `vector`).
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn lorentz_distance_f32(a: Vec<f32>, b: Vec<f32>) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        error!(
            "pg_hyperspace: lorentz_distance_f32 requires matching dimensions >= 2 (got {} and {})",
            a.len(),
            b.len()
        );
    }
    let a64: Vec<f64> = a.into_iter().map(f64::from).collect();
    let b64: Vec<f64> = b.into_iter().map(f64::from).collect();
    lorentz_distance_f64(&a64, &b64)
}

/// Computes 1D Wasserstein Optimal Transport distance between two `real[]` (`float4[]`) vectors (used by `<|>` operator on `vector`).
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn wasserstein_distance_f32(a: Vec<f32>, b: Vec<f32>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        error!(
            "pg_hyperspace: wasserstein_distance_f32 dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
    }
    let a64: Vec<f64> = a.into_iter().map(f64::from).collect();
    let b64: Vec<f64> = b.into_iter().map(f64::from).collect();
    wasserstein_1d_distance_f64(&a64, &b64)
}

/// Computes Product-Manifold Hybrid distance between two vectors:
/// first 33 dimensions are evaluated on the Lorentz hyperboloid ($\mathbb{H}^{32}$),
/// and remaining $D - 33$ dimensions are evaluated in Euclidean space ($\mathbb{R}^{D-33}$).
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn hybrid_distance(
    a: Vec<f64>,
    b: Vec<f64>,
    lorentz_weight: default!(f64, 1.0),
    euclidean_weight: default!(f64, 1.0),
) -> f64 {
    if a.len() != b.len() || a.len() < HYBRID_LORENTZ_HEAD_DIM {
        error!(
            "pg_hyperspace: hybrid_distance requires matching dimensions >= {HYBRID_LORENTZ_HEAD_DIM} (33D Lorentz head + Euclidean tail), got {} and {}",
            a.len(),
            b.len()
        );
    }
    hybrid_weighted_distance_f64(&a, &b, lorentz_weight, euclidean_weight)
}

/// Computes Product-Manifold Hybrid distance between two `real[]` (`float4[]`) vectors (used by `<+>` operator on `vector`).
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn hybrid_distance_f32(a: Vec<f32>, b: Vec<f32>) -> f64 {
    if a.len() != b.len() || a.len() < HYBRID_LORENTZ_HEAD_DIM {
        error!(
            "pg_hyperspace: hybrid_distance_f32 requires matching dimensions >= {HYBRID_LORENTZ_HEAD_DIM} (33D Lorentz head + Euclidean tail), got {} and {}",
            a.len(),
            b.len()
        );
    }
    let a64: Vec<f64> = a.into_iter().map(f64::from).collect();
    let b64: Vec<f64> = b.into_iter().map(f64::from).collect();
    hybrid_distance_f64(&a64, &b64)
}

/// Composes a Product-Manifold Hybrid vector (`33 + D` dimensions, e.g., 801D `AsymmetricHybrid801`)
/// by lifting `hierarchy` onto the 33D Lorentz hyperboloid $\mathbb{H}^{32}$ ($u_0 = \sqrt{1 + \|x\|_2^2}$)
/// and appending `semantic_tail`.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn compose_hybrid(hierarchy: Vec<f64>, semantic_tail: Vec<f64>) -> Vec<f64> {
    compose_hybrid_vector(&hierarchy, &semantic_tail)
}

/// Extracts the 33-dimensional Lorentz hyperboloid head ($\mathbb{H}^{32}$) from a Product-Manifold Hybrid vector.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn hybrid_lorentz_head(hybrid_vec: Vec<f64>) -> Vec<f64> {
    if hybrid_vec.len() < HYBRID_LORENTZ_HEAD_DIM {
        error!(
            "pg_hyperspace: hybrid_lorentz_head requires a vector of length >= {HYBRID_LORENTZ_HEAD_DIM}, got {}",
            hybrid_vec.len()
        );
    }
    hybrid_vec[..HYBRID_LORENTZ_HEAD_DIM].to_vec()
}

/// Extracts the $(D - 33)$-dimensional Euclidean semantic tail ($\mathbb{R}^{D-33}$) from a Product-Manifold Hybrid vector.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn hybrid_semantic_tail(hybrid_vec: Vec<f64>) -> Vec<f64> {
    if hybrid_vec.len() <= HYBRID_LORENTZ_HEAD_DIM {
        error!(
            "pg_hyperspace: hybrid_semantic_tail requires a vector of length > {HYBRID_LORENTZ_HEAD_DIM}, got {}",
            hybrid_vec.len()
        );
    }
    hybrid_vec[HYBRID_LORENTZ_HEAD_DIM..].to_vec()
}

/// Computes 4-way unrolled Cosine distance between two vectors.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn cosine_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        error!(
            "pg_hyperspace: cosine_distance dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
    }
    cosine_distance_f64(&a, &b)
}

/// Computes 4-way unrolled Euclidean $L_2$ distance between two vectors.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn l2_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        error!(
            "pg_hyperspace: l2_distance dimension mismatch ({} vs {})",
            a.len(),
            b.len()
        );
    }
    l2_distance_f64(&a, &b)
}

/// Lifts a $d$-dimensional Euclidean vector onto the upper sheet of the $(d+1)$-dimensional
/// unit Lorentz hyperboloid $\mathbb{H}^d$: $u_0 = \sqrt{1 + \|x\|_2^2}$.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn lift_to_lorentz(spatial: Vec<f64>) -> Vec<f64> {
    lift_euclidean_to_lorentz(&spatial)
}

/// Conformal radial projection of an arbitrary Euclidean vector into the open Poincaré unit ball.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn project_to_poincare(euclidean: Vec<f64>) -> Vec<f64> {
    project_euclidean_to_poincare(&euclidean)
}

// ---------------------------------------------------------------------------
// 2. Hyperbolic Light-Cone & Spatial Region Predicates
// ---------------------------------------------------------------------------

/// Checks whether `child` lies inside the future Lorentz light-cone of `parent` within
/// hyperbolic radius `max_radius` ($\langle \text{parent}, \text{child} \rangle_{\mathcal{L}} \ge -\cosh(r) \wedge \text{child}_0 \ge \text{parent}_0$).
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn in_lorentz_cone(parent: Vec<f64>, child: Vec<f64>, max_radius: default!(f64, 1.5)) -> bool {
    lorentz_light_cone_contains(&parent, &child, max_radius)
}

/// Checks whether `coords` lies inside the ConE angular cone `(axes, apertures, cen)`.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn in_angular_cone(
    axes: Vec<f64>,
    apertures: Vec<f64>,
    cen: default!(f64, 0.02),
    coords: Vec<f64>,
) -> bool {
    angular_cone_contains(&axes, &apertures, cen, &coords)
}

/// Checks whether `coords` lies inside the Axis-Aligned Bounding Box `[min_bounds, max_bounds]`.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn in_box(min_bounds: Vec<f64>, max_bounds: Vec<f64>, coords: Vec<f64>) -> bool {
    in_box_contains(&min_bounds, &max_bounds, &coords)
}

/// Checks whether `coords` lies inside the Euclidean ball `B(center, radius)`.
#[pg_extern(schema = "hyperspace", immutable, parallel_safe)]
fn in_ball(center: Vec<f64>, radius: f64, coords: Vec<f64>) -> bool {
    in_ball_contains(&center, radius, &coords)
}

// ---------------------------------------------------------------------------
// 3. Native Vector-Graph (KAG) SQL Functions
// ---------------------------------------------------------------------------

/// Links two nodes with a directed labeled edge in the collection's KAG CSR/Dynamic graph.
#[pg_extern(schema = "hyperspace")]
fn link(
    table_or_collection: &str,
    source_id: i64,
    target_id: i64,
    relation: default!(&str, "'DEPENDS_ON'"),
    weight: default!(f32, 1.0),
) -> bool {
    let (collection, endpoint, api_key, sync_mode, relid_opt) =
        resolve_collection_target(table_or_collection);
    let src = u32::try_from(source_id)
        .unwrap_or_else(|_| error!("pg_hyperspace: source_id {source_id} out of u32 range"));
    let tgt = u32::try_from(target_id)
        .unwrap_or_else(|_| error!("pg_hyperspace: target_id {target_id} out of u32 range"));

    if sync_mode == "async"
        && let Some(relid) = relid_opt
    {
        outbox::enqueue_edge_link(relid, src, tgt, relation, weight);
        return true;
    }

    let client = client_for_table(endpoint.as_deref(), api_key.as_deref());
    client
        .link_edge(
            &collection,
            &LinkEdgeRequest {
                source_id: src,
                target_id: tgt,
                relation: relation.to_string(),
                weight,
            },
        )
        .unwrap_or_else(|e| backend::raise_client_error(e));
    true
}

/// Removes a directed labeled edge from the collection's KAG graph.
#[pg_extern(schema = "hyperspace")]
fn unlink(
    table_or_collection: &str,
    source_id: i64,
    target_id: i64,
    relation: default!(&str, "'DEPENDS_ON'"),
) -> bool {
    let (collection, endpoint, api_key, _, _) = resolve_collection_target(table_or_collection);
    let src = u32::try_from(source_id)
        .unwrap_or_else(|_| error!("pg_hyperspace: source_id {source_id} out of u32 range"));
    let tgt = u32::try_from(target_id)
        .unwrap_or_else(|_| error!("pg_hyperspace: target_id {target_id} out of u32 range"));

    let client = client_for_table(endpoint.as_deref(), api_key.as_deref());
    client
        .unlink_edge(
            &collection,
            &UnlinkEdgeRequest {
                source_id: src,
                target_id: tgt,
                relation: relation.to_string(),
            },
        )
        .unwrap_or_else(|e| backend::raise_client_error(e))
}

/// Induces KAG graph edges declaratively from metadata rules (`path_hierarchy`, `foreign_key`, `exact_match`).
#[pg_extern(schema = "hyperspace")]
fn induce_graph(table_or_collection: &str, rules: JsonB) -> i64 {
    let (collection, endpoint, api_key, _, _) = resolve_collection_target(table_or_collection);
    let parsed_rules: Vec<MetadataInductionRule> = serde_json::from_value(rules.0)
        .unwrap_or_else(|e| error!("pg_hyperspace: invalid induction rules JSONB: {e}"));

    let client = client_for_table(endpoint.as_deref(), api_key.as_deref());
    let created = client
        .induce_graph(&collection, &parsed_rules)
        .unwrap_or_else(|e| backend::raise_client_error(e));
    created as i64
}

/// Executes sub-millisecond KAG BFS graph traversal starting from `start_id` and returns all
/// reachable node IDs (`int8[]`), replacing slow SQL `WITH RECURSIVE` CTEs.
#[pg_extern(schema = "hyperspace")]
fn reachable_nodes(
    table_or_collection: &str,
    start_id: i64,
    max_depth: default!(i32, 2),
    max_nodes: default!(i32, 256),
) -> Vec<i64> {
    let (collection, endpoint, api_key, _, _) = resolve_collection_target(table_or_collection);
    let seed = u32::try_from(start_id)
        .unwrap_or_else(|_| error!("pg_hyperspace: start_id {start_id} out of u32 range"));

    let client = client_for_table(endpoint.as_deref(), api_key.as_deref());
    let nodes = client
        .traverse_graph(
            &collection,
            &GraphTraverseRequest {
                start_id: seed,
                layer: Some(0),
                max_depth: Some(max_depth.clamp(1, 8) as usize),
                max_nodes: Some(max_nodes.clamp(1, 10_000) as usize),
                filter: None,
                filters: None,
            },
        )
        .unwrap_or_else(|e| backend::raise_client_error(e));

    nodes.into_iter().map(|n| i64::from(n.id)).collect()
}

// ---------------------------------------------------------------------------
// 4. Tribunal Router (Geometric Hallucination Prevention)
// ---------------------------------------------------------------------------

/// Verifies an LLM-generated answer against source context chunks using HyperspaceDB's
/// Tribunal Router (`trust_score`, `is_valid`, `geodesic_distance`, `cone_satisfied`).
#[pg_extern(schema = "hyperspace")]
fn verify_generation(
    table_or_collection: &str,
    llm_answer: &str,
    context_ids: Vec<String>,
    max_distance_threshold: default!(Option<f64>, "NULL"),
    check_geodesic: default!(bool, true),
) -> JsonB {
    let (collection, endpoint, api_key, _, _) = resolve_collection_target(table_or_collection);
    let client = client_for_table(endpoint.as_deref(), api_key.as_deref());
    let resp = client
        .verify_generation(
            &collection,
            &VerifyGenerationRequest {
                llm_answer: llm_answer.to_string(),
                context_ids,
                collection: Some(collection.clone()),
                max_distance_threshold,
                check_geodesic: Some(check_geodesic),
            },
        )
        .unwrap_or_else(|e| backend::raise_client_error(e));

    JsonB(serde_json::to_value(resp).unwrap_or_else(|_| json!({})))
}

// ---------------------------------------------------------------------------
// 5. Engine Lifecycle & Observability Functions
// ---------------------------------------------------------------------------

/// Configures the HyperspaceDB endpoint, API key, and sync mode (`'direct'` or `'async'`)
/// for a registered table. Enables instant switching between a local `hyperspace-server`
/// daemon and a remote HyperspaceDB Cloud cluster without changing SQL queries.
#[pg_extern(schema = "hyperspace")]
fn set_endpoint(
    table_name: &str,
    endpoint_url: &str,
    api_key: default!(Option<&str>, "NULL"),
    sync_mode: default!(&str, "'direct'"),
) -> bool {
    let mode = sync_mode.trim().to_ascii_lowercase();
    if mode != "direct" && mode != "async" {
        error!("pg_hyperspace: sync_mode must be 'direct' or 'async' (got '{sync_mode}')");
    }
    let Some(relid) = try_resolve_regclass_oid(table_name) else {
        error!("pg_hyperspace: relation '{table_name}' does not exist");
    };
    Spi::run_with_args(
        "UPDATE hyperspace.tables \
         SET endpoint_url = $2, api_key = $3, sync_mode = $4, updated_at = clock_timestamp() \
         WHERE relid = $1",
        &[
            relid.into(),
            endpoint_url.trim().into(),
            api_key.into(),
            mode.into(),
        ],
    )
    .expect("pg_hyperspace: failed to update hyperspace.tables endpoint");
    true
}

/// Synchronously drains all pending vector upserts and KAG graph edges from `hyperspace.outbox`
/// and `hyperspace.edges_outbox` into HyperspaceDB.
#[pg_extern(schema = "hyperspace")]
fn flush() -> JsonB {
    let (vectors_flushed, edges_flushed) = outbox::drain_all_tables();
    JsonB(json!({
        "status": "OK",
        "vectors_flushed": vectors_flushed,
        "edges_flushed": edges_flushed
    }))
}

/// Checks whether the configured HyperspaceDB server is online (`GET /health`).
#[pg_extern(schema = "hyperspace")]
fn health() -> bool {
    let client = client_for_table(None, None);
    client.health().unwrap_or(false)
}

/// Returns a JSONB diagnostic summary of all indexed tables, pending outbox queues, and engine health.
#[pg_extern(schema = "hyperspace")]
fn status() -> JsonB {
    let engine_online = health();
    let pending_vectors = Spi::get_one::<i64>("SELECT count(*) FROM hyperspace.outbox")
        .ok()
        .flatten()
        .unwrap_or(0);
    let pending_edges = Spi::get_one::<i64>("SELECT count(*) FROM hyperspace.edges_outbox")
        .ok()
        .flatten()
        .unwrap_or(0);

    let tables: Vec<serde_json::Value> = Spi::connect(|spi| {
        spi.select(
            "SELECT relid::regclass::text AS table_name, collection_name, vector_column, \
                    dimension, metric, filter_columns, sync_mode, \
                    COALESCE(endpoint_url, current_setting('hyperspace.endpoint_url', true)) AS endpoint, \
                    last_error \
             FROM hyperspace.tables ORDER BY collection_name",
            None,
            &[],
        )
        .expect("pg_hyperspace: failed to query hyperspace.tables")
        .map(|r| {
            json!({
                "table": r.get_by_name::<String, _>("table_name").ok().flatten(),
                "collection": r.get_by_name::<String, _>("collection_name").ok().flatten(),
                "vector_column": r.get_by_name::<String, _>("vector_column").ok().flatten(),
                "dimension": r.get_by_name::<i32, _>("dimension").ok().flatten(),
                "metric": r.get_by_name::<String, _>("metric").ok().flatten(),
                "filter_columns": r.get_by_name::<Vec<String>, _>("filter_columns").ok().flatten().unwrap_or_default(),
                "sync_mode": r.get_by_name::<String, _>("sync_mode").ok().flatten(),
                "endpoint": r.get_by_name::<String, _>("endpoint").ok().flatten(),
                "last_error": r.get_by_name::<String, _>("last_error").ok().flatten(),
            })
        })
        .collect()
    });

    JsonB(json!({
        "engine_online": engine_online,
        "pending_outbox_vectors": pending_vectors,
        "pending_outbox_edges": pending_edges,
        "tables": tables,
    }))
}

// ---------------------------------------------------------------------------
// Internal Resolution Helpers
// ---------------------------------------------------------------------------

fn try_resolve_regclass_oid(name: &str) -> Option<pg_sys::Oid> {
    Spi::get_one_with_args::<pg_sys::Oid>("SELECT to_regclass($1)::oid", &[name.into()])
        .ok()
        .flatten()
}

fn resolve_collection_target(
    table_or_collection: &str,
) -> (
    String,
    Option<String>,
    Option<String>,
    String,
    Option<pg_sys::Oid>,
) {
    if let Some(relid) = try_resolve_regclass_oid(table_or_collection)
        && let Some(row) = catalog::lookup_table(relid)
    {
        return (
            row.collection_name,
            row.endpoint_url,
            row.api_key,
            row.sync_mode,
            Some(relid),
        );
    }
    (
        table_or_collection.trim().to_string(),
        None,
        None,
        "direct".to_string(),
        None,
    )
}
