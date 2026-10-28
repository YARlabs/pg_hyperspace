# Changelog

All notable changes to `pg_hyperspace` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-28

### Added
- **PostgreSQL Index Access Method (`USING hyperspace`)**:
  - Seamless drop-in support for `pgvector` `vector(D)` columns across PostgreSQL 16, 17, and 18 via `pgrx = "0.19.3"`.
  - **7 Geometric Operator Classes**:
    - `vector_cosine_ops` (`<=>`) — Cosine angular distance.
    - `vector_l2_ops` (`<->`) — Euclidean $L_2$ distance.
    - `vector_ip_ops` (`<#>`) — Negative Inner Product (MIPS).
    - `vector_poincare_ops` (`<~>`) — Poincaré Ball hyperbolic geodesic distance ($\mathbb{B}^d$).
    - `vector_lorentz_ops` (`<@>`) — Lorentz Hyperboloid Minkowski geodesic distance ($\mathbb{H}^d$).
    - `vector_wasserstein_ops` (`<|>`) — 1D Wasserstein ($W_1$ Earth Mover's) Optimal Transport distance.
    - `vector_hybrid_ops` (`<+>`) — Product-Manifold Hybrid distance (`33D Lorentz Hyperboloid Head` $\mathbb{H}^{32}$ + $(D - 33)$-dimensional Euclidean Semantic Tail $\mathbb{R}^{D-33}$, including 801D `AsymmetricHybrid801`).
- **Multi-Column & Range Filter Pushdown (`am_filter.rs`)**:
  - Composite index support for an arbitrary number of scalar filter columns (`int2`, `int4`, `int8`, `float4`, `float8`, `text`, `varchar`, `bpchar`, `bool`).
  - Pushdown of all 5 B-tree comparison strategies (`<`, `<=`, `=`, `>=`, `>`) directly into HyperspaceDB's 32-Stripe Sharded Payload Index (`Match` & `Range`), plus automatic `__hs_typed__<col>` numeric shadow indexing.
  - Session-level spatial and hyperbolic filter injection via `hyperspace.query_filters` (`InCone`, `InBox`, `InBall`, `Range`, `And`, `Or`, `Not`) and multi-manifold weighting via `hyperspace.component_weights`.
- **Product-Manifold Hybrid Vector & Non-Euclidean SQL Functions (`sql_api.rs`)**:
  - `hyperspace.compose_hybrid(hierarchy, semantic_tail)` — lifts a 32D hierarchy vector onto the 33D Lorentz hyperboloid sheet ($u_0 = \sqrt{1 + \|x\|_2^2}$) and concatenates the semantic tail.
  - `hyperspace.hybrid_distance(a, b, lorentz_weight, euclidean_weight)`, `hyperspace.hybrid_lorentz_head(v)`, and `hyperspace.hybrid_semantic_tail(v)`.
  - `hyperspace.poincare_distance`, `hyperspace.lorentz_distance`, `hyperspace.wasserstein_distance`, `hyperspace.lift_to_lorentz`, `hyperspace.project_to_poincare`.
  - Geometric containment predicates: `hyperspace.in_lorentz_cone`, `hyperspace.in_angular_cone`, `hyperspace.in_box`, `hyperspace.in_ball`.
- **Native Vector-Graph (KAG) SQL Engine**:
  - `hyperspace.link`, `hyperspace.unlink`, `hyperspace.induce_graph`, and `hyperspace.reachable_nodes` for sub-millisecond multi-hop CSR graph traversal inside SQL queries, replacing slow `WITH RECURSIVE` CTEs.
- **Tribunal Hallucination Router**:
  - `hyperspace.verify_generation` for geometric hallucination detection and causal entailment verification directly from SQL.
- **Zero-Leak Memory & High-Throughput Architecture**:
  - Zero-leak `pgvector` TOAST datum decoding (`vector_datum.rs`) with immediate `pg_sys::pfree` of detoasted `varlena` buffers.
  - Zero-allocation 31-bit `(block, offset)` <-> `u32` `PgTid` bit-packing codec (`tid.rs`) with `_pg_tid` metadata fallback for multi-terabyte tables.
  - Thread-local `Arc<Client>` HTTP connection pooling (`backend.rs`) and `EINTR`-resilient socket reader (`transport.rs`).
  - Dual write path (`direct` immediate WriteBuffer streaming + `async` ACID transactional outbox drained by a PostgreSQL background worker).

