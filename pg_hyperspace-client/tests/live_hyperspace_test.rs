//! End-to-end integration tests for `pg_hyperspace-client` against `wiremock` and live
//! local `hyperspace-server` (`http://127.0.0.1:50050`).

use std::collections::HashMap;

use pg_hyperspace_client::{
    Client, Config, CreateCollectionRequest, GraphTraverseRequest, HttpFilter, InsertPayload,
    LinkEdgeRequest, META_PG_TID_KEY, MetricKind, PgTid, SearchRequest,
    metric::{
        compose_hybrid_vector, hybrid_distance_f64, lift_euclidean_to_lorentz,
        project_euclidean_to_poincare,
    },
};

#[test]
fn test_live_hyperspace_multi_metric_and_kag_pipeline() {
    let api_key =
        std::env::var("HYPERSPACE_API_KEY").unwrap_or_else(|_| "I_LOVE_HYPERSPACEDB".to_string());
    let client = Client::new(Config::default().with_api_key(api_key));
    // If local hyperspace-server is running on 127.0.0.1:50050, run full live verification.
    if !client.health().unwrap_or(false) {
        eprintln!("Skipping live HyperspaceDB test: server not reachable at 127.0.0.1:50050");
        return;
    }

    // 1. Test Cosine + Multi-Column Range & Equality Filter + KAG Graph Traversal
    let col_cosine = "pg_hs_test_cosine_kag";
    let _ = client.delete_collection(col_cosine);
    client
        .create_collection(&CreateCollectionRequest {
            name: col_cosine.to_string(),
            dimension: 4,
            metric: MetricKind::Cosine.as_str().to_string(),
            mrl_cutoff_dimension: None,
            mrl_rerank_top_k: None,
            quantization: None,
            encrypted: None,
        })
        .expect("create cosine collection");

    let tid1 = PgTid::try_from_parts(10, 1).unwrap();
    let tid2 = PgTid::try_from_parts(10, 2).unwrap();
    let tid3 = PgTid::try_from_parts(10, 3).unwrap();

    let make_meta = |tid: PgTid, tenant: &str, price: f64| {
        let mut m = HashMap::new();
        m.insert(META_PG_TID_KEY.to_string(), tid.to_str_repr());
        m.insert("tenant".to_string(), tenant.to_string());
        m.insert("price".to_string(), price.to_string());
        m.insert(
            "__hs_typed__price".to_string(),
            format!("{{\"v\":{price}}}"),
        );
        m
    };

    let inserted = client
        .insert_batch(
            col_cosine,
            &[
                InsertPayload {
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    id: tid1.to_point_id(),
                    metadata: Some(make_meta(tid1, "acme", 25.0)),
                },
                InsertPayload {
                    vector: vec![0.9, 0.1, 0.0, 0.0],
                    id: tid2.to_point_id(),
                    metadata: Some(make_meta(tid2, "acme", 90.0)),
                },
                InsertPayload {
                    vector: vec![0.0, 1.0, 0.0, 0.0],
                    id: tid3.to_point_id(),
                    metadata: Some(make_meta(tid3, "other", 15.0)),
                },
            ],
        )
        .expect("batch insert");
    assert_eq!(inserted, 3);

    // Filtered search: tenant = 'acme' AND price <= 50.0
    let mut exact = HashMap::new();
    exact.insert("tenant".to_string(), "acme".to_string());
    let hits = client
        .search(
            col_cosine,
            &SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                top_k: Some(10),
                filter: Some(exact),
                filters: Some(vec![HttpFilter::range("price", Some(10.0), Some(50.0))]),
                use_wasserstein: None,
                mrl_dimension: None,
                include_payload: Some(false),
                weights: None,
            },
        )
        .expect("filtered search");
    assert_eq!(hits.len(), 1);
    let decoded_tid =
        PgTid::from_point_id_and_meta(hits[0].id, Some(&hits[0].metadata)).expect("decode tid");
    assert_eq!(decoded_tid, tid1);

    // KAG Graph Link & Multi-Hop Traversal: tid1 -> tid2 -> tid3
    let edges_linked = client
        .link_batch(
            col_cosine,
            &[
                LinkEdgeRequest {
                    source_id: tid1.to_point_id(),
                    target_id: tid2.to_point_id(),
                    relation: "DEPENDS_ON".to_string(),
                    weight: 1.0,
                },
                LinkEdgeRequest {
                    source_id: tid2.to_point_id(),
                    target_id: tid3.to_point_id(),
                    relation: "DEPENDS_ON".to_string(),
                    weight: 0.9,
                },
            ],
        )
        .expect("link batch");
    assert_eq!(edges_linked, 2);

    let reachable = client
        .traverse_graph(
            col_cosine,
            &GraphTraverseRequest {
                start_id: tid1.to_point_id(),
                layer: Some(0),
                max_depth: Some(2),
                max_nodes: Some(50),
                filter: None,
                filters: None,
            },
        )
        .expect("traverse graph");
    assert!(!reachable.is_empty());

    let _ = client.delete_collection(col_cosine);

    // 2. Test Hyperbolic Poincaré & Lorentz Collections
    for (metric, col_name, dim) in [
        (MetricKind::Poincare, "pg_hs_test_poincare", 4),
        (MetricKind::Lorentz, "pg_hs_test_lorentz", 5),
        (MetricKind::Wasserstein, "pg_hs_test_wasserstein", 4),
    ] {
        let _ = client.delete_collection(col_name);
        client
            .create_collection(&CreateCollectionRequest {
                name: col_name.to_string(),
                dimension: dim,
                metric: metric.collection_metric_str().to_string(),
                mrl_cutoff_dimension: None,
                mrl_rerank_top_k: None,
                quantization: None,
                encrypted: None,
            })
            .expect("create metric collection");

        let raw_a = [0.2, 0.1, 0.05, 0.02];
        let raw_b = [0.4, 0.2, 0.10, 0.04];
        let (vec_a, vec_b) = match metric {
            MetricKind::Poincare => (
                project_euclidean_to_poincare(&raw_a),
                project_euclidean_to_poincare(&raw_b),
            ),
            MetricKind::Lorentz => (
                lift_euclidean_to_lorentz(&raw_a),
                lift_euclidean_to_lorentz(&raw_b),
            ),
            _ => (raw_a.to_vec(), raw_b.to_vec()),
        };

        let t_a = PgTid::try_from_parts(1, 1).unwrap();
        let t_b = PgTid::try_from_parts(1, 2).unwrap();
        client
            .insert_batch(
                col_name,
                &[
                    InsertPayload {
                        vector: vec_a.clone(),
                        id: t_a.to_point_id(),
                        metadata: None,
                    },
                    InsertPayload {
                        vector: vec_b,
                        id: t_b.to_point_id(),
                        metadata: None,
                    },
                ],
            )
            .expect("insert non-euclidean vectors");

        let res = client
            .search(
                col_name,
                &SearchRequest {
                    vector: vec_a,
                    top_k: Some(2),
                    filter: None,
                    filters: None,
                    use_wasserstein: Some(metric == MetricKind::Wasserstein),
                    mrl_dimension: None,
                    include_payload: Some(false),
                    weights: None,
                },
            )
            .expect("search non-euclidean");
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, t_a.to_point_id());

        let _ = client.delete_collection(col_name);
    }

    // 3. Test Product-Manifold Hybrid Collection (33D Lorentz Head + Euclidean Tail, e.g. 37D & 801D)
    let col_hybrid = "pg_hs_test_hybrid_product_manifold";
    let _ = client.delete_collection(col_hybrid);
    client
        .create_collection(&CreateCollectionRequest {
            name: col_hybrid.to_string(),
            dimension: 37,
            metric: MetricKind::Hybrid.collection_metric_str().to_string(),
            mrl_cutoff_dimension: None,
            mrl_rerank_top_k: None,
            quantization: None,
            encrypted: None,
        })
        .expect("create hybrid collection");

    let mut h_root = [0.0_f64; 32];
    h_root[0] = 0.05;
    let mut h_child = [0.0_f64; 32];
    h_child[0] = 0.35;

    let hyb_a = compose_hybrid_vector(&h_root, &[1.0, 0.0, 0.0, 0.0]);
    let hyb_b = compose_hybrid_vector(&h_child, &[0.0, 1.0, 0.0, 0.0]);
    assert_eq!(hyb_a.len(), 37);
    assert!(hybrid_distance_f64(&hyb_a, &hyb_b) > 0.0);

    let th_a = PgTid::try_from_parts(2, 1).unwrap();
    let th_b = PgTid::try_from_parts(2, 2).unwrap();
    client
        .insert_batch(
            col_hybrid,
            &[
                InsertPayload {
                    vector: hyb_a.clone(),
                    id: th_a.to_point_id(),
                    metadata: None,
                },
                InsertPayload {
                    vector: hyb_b,
                    id: th_b.to_point_id(),
                    metadata: None,
                },
            ],
        )
        .expect("insert hybrid vectors");

    let hyb_hits = client
        .search(
            col_hybrid,
            &SearchRequest {
                vector: hyb_a,
                top_k: Some(2),
                filter: None,
                filters: None,
                use_wasserstein: None,
                mrl_dimension: None,
                include_payload: Some(false),
                weights: None,
            },
        )
        .expect("search hybrid collection");
    assert_eq!(hyb_hits.len(), 2);
    assert_eq!(hyb_hits[0].id, th_a.to_point_id());

    let _ = client.delete_collection(col_hybrid);
}
