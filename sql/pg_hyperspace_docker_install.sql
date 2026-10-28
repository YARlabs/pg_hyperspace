-- =============================================================================
-- pg_hyperspace: Pure-SQL / PL/pgSQL Multi-Metric Spatial Kernel & Outbox Layer
-- =============================================================================
-- Enables immediate installation into any running PostgreSQL container (including
-- Alpine-based containers without native C/Rust build toolchains) while remaining
-- 100% schema-compatible with the compiled pgrx `pg_hyperspace` C-extension.

CREATE SCHEMA IF NOT EXISTS hyperspace;

CREATE TABLE IF NOT EXISTS hyperspace.tables (
    relid                oid PRIMARY KEY,
    am_index_relid       oid UNIQUE,
    collection_name      text NOT NULL,
    vector_column        text NOT NULL,
    vector_attnum        int2 NOT NULL DEFAULT 1,
    dimension            int4 NOT NULL,
    metric               text NOT NULL DEFAULT 'cosine',
    filter_columns       text[] NOT NULL DEFAULT '{}',
    filter_attnums       int2[] NOT NULL DEFAULT '{}',
    mrl_cutoff_dimension int4,
    quantization         text,
    sync_mode            text NOT NULL DEFAULT 'direct',
    endpoint_url         text DEFAULT 'http://127.0.0.1:50050',
    api_key              text,
    last_error           text,
    updated_at           timestamptz NOT NULL DEFAULT clock_timestamp()
);

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

-- -----------------------------------------------------------------------------
-- 1. Euclidean L2 & Cosine Distance Functions
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hyperspace.l2_distance(a float8[], b float8[])
RETURNS float8
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(a, 1), array_length(b, 1));
    acc float8 := 0.0;
    d float8;
BEGIN
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) OR n IS NULL OR n = 0 THEN
        RAISE EXCEPTION 'pg_hyperspace: l2_distance dimension mismatch';
    END IF;
    FOR i IN 1..n LOOP
        d := a[i] - b[i];
        acc := acc + d * d;
    END LOOP;
    RETURN sqrt(acc);
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.cosine_distance(a float8[], b float8[])
RETURNS float8
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(a, 1), array_length(b, 1));
    dot float8 := 0.0;
    na float8 := 0.0;
    nb float8 := 0.0;
    denom float8;
BEGIN
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) OR n IS NULL OR n = 0 THEN
        RAISE EXCEPTION 'pg_hyperspace: cosine_distance dimension mismatch';
    END IF;
    FOR i IN 1..n LOOP
        dot := dot + a[i] * b[i];
        na := na + a[i] * a[i];
        nb := nb + b[i] * b[i];
    END LOOP;
    denom := sqrt(na * nb);
    IF denom <= 1e-15 THEN
        RETURN 1.0;
    END IF;
    RETURN GREATEST(0.0, 1.0 - LEAST(1.0, GREATEST(-1.0, dot / denom)));
END;
$$;

-- -----------------------------------------------------------------------------
-- 2. Hyperbolic Poincaré Ball & Lorentz Hyperboloid Geodesic Kernels
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hyperspace.poincare_distance(a float8[], b float8[])
RETURNS float8
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(a, 1), array_length(b, 1));
    norm_u_sq float8 := 0.0;
    norm_v_sq float8 := 0.0;
    diff_sq float8 := 0.0;
    d float8;
    denom float8;
    arg float8;
BEGIN
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) OR n IS NULL OR n = 0 THEN
        RAISE EXCEPTION 'pg_hyperspace: poincare_distance dimension mismatch';
    END IF;
    FOR i IN 1..n LOOP
        norm_u_sq := norm_u_sq + a[i] * a[i];
        norm_v_sq := norm_v_sq + b[i] * b[i];
        d := a[i] - b[i];
        diff_sq := diff_sq + d * d;
    END LOOP;
    denom := GREATEST(1e-9, (1.0 - norm_u_sq)) * GREATEST(1e-9, (1.0 - norm_v_sq));
    arg := GREATEST(1.0, 1.0 + 2.0 * diff_sq / GREATEST(1e-9, denom));
    RETURN acosh(arg);
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.lorentz_distance(a float8[], b float8[])
RETURNS float8
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(a, 1), array_length(b, 1));
    spatial_dot float8 := 0.0;
    inner_prod float8;
    arg float8;
BEGIN
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) OR n IS NULL OR n < 2 THEN
        RAISE EXCEPTION 'pg_hyperspace: lorentz_distance requires matching dimensions >= 2';
    END IF;
    FOR i IN 2..n LOOP
        spatial_dot := spatial_dot + a[i] * b[i];
    END LOOP;
    inner_prod := -a[1] * b[1] + spatial_dot;
    arg := GREATEST(1.0 + 1e-12, -inner_prod);
    RETURN acosh(arg);
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.lift_to_lorentz(spatial float8[])
RETURNS float8[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := array_length(spatial, 1);
    norm_sq float8 := 0.0;
    t0 float8;
BEGIN
    IF n IS NULL OR n = 0 THEN
        RETURN ARRAY[1.0]::float8[];
    END IF;
    FOR i IN 1..n LOOP
        norm_sq := norm_sq + spatial[i] * spatial[i];
    END LOOP;
    t0 := sqrt(1.0 + norm_sq);
    RETURN array_prepend(t0, spatial);
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.project_to_poincare(euclidean float8[])
RETURNS float8[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := array_length(euclidean, 1);
    norm_sq float8 := 0.0;
    norm float8;
    scale float8;
    res float8[] := '{}';
BEGIN
    IF n IS NULL OR n = 0 THEN
        RETURN res;
    END IF;
    FOR i IN 1..n LOOP
        norm_sq := norm_sq + euclidean[i] * euclidean[i];
    END LOOP;
    norm := sqrt(norm_sq);
    IF norm <= 1e-12 THEN
        RETURN euclidean;
    END IF;
    scale := LEAST(tanh(norm) / norm, (1.0 - 1e-6) / norm);
    FOR i IN 1..n LOOP
        res := array_append(res, euclidean[i] * scale);
    END LOOP;
    RETURN res;
END;
$$;

-- -----------------------------------------------------------------------------
-- 3. 1D Wasserstein (Optimal Transport / Earth Mover's) Distance
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hyperspace.wasserstein_distance(a float8[], b float8[])
RETURNS float8
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(a, 1), array_length(b, 1));
    sum_a float8 := 0.0;
    sum_b float8 := 0.0;
    cdf_a float8 := 0.0;
    cdf_b float8 := 0.0;
    dist float8 := 0.0;
    va float8;
    vb float8;
BEGIN
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) OR n IS NULL OR n = 0 THEN
        RAISE EXCEPTION 'pg_hyperspace: wasserstein_distance dimension mismatch';
    END IF;
    FOR i IN 1..n LOOP
        IF a[i] > 0.0 THEN sum_a := sum_a + a[i]; END IF;
        IF b[i] > 0.0 THEN sum_b := sum_b + b[i]; END IF;
    END LOOP;
    IF sum_a <= 1e-9 OR sum_b <= 1e-9 THEN
        RETURN 1e300;
    END IF;
    FOR i IN 1..n LOOP
        va := CASE WHEN a[i] > 0.0 THEN a[i] / sum_a ELSE 0.0 END;
        vb := CASE WHEN b[i] > 0.0 THEN b[i] / sum_b ELSE 0.0 END;
        cdf_a := cdf_a + va;
        cdf_b := cdf_b + vb;
        dist := dist + abs(cdf_a - cdf_b);
    END LOOP;
    RETURN dist;
END;
$$;

-- -----------------------------------------------------------------------------
-- 3b. Product-Manifold Hybrid Vectors (33D Lorentz Head + Euclidean Semantic Tail)
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hyperspace.compose_hybrid(
    hierarchy float8[],
    semantic_tail float8[]
) RETURNS float8[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    h_len int := COALESCE(array_length(hierarchy, 1), 0);
    spatial float8[] := ARRAY[]::float8[];
    norm_sq float8 := 0.0;
    u0 float8;
    val float8;
BEGIN
    FOR i IN 1..32 LOOP
        IF h_len = 33 THEN
            val := hierarchy[i + 1];
        ELSIF i <= h_len THEN
            val := hierarchy[i];
        ELSE
            val := 0.0;
        END IF;
        spatial := array_append(spatial, val);
        norm_sq := norm_sq + val * val;
    END LOOP;
    u0 := sqrt(1.0 + norm_sq);
    RETURN array_cat(array_prepend(u0, spatial), semantic_tail);
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.hybrid_distance(
    a float8[],
    b float8[],
    lorentz_weight float8 DEFAULT 1.0,
    euclidean_weight float8 DEFAULT 1.0
) RETURNS float8
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(a, 1), array_length(b, 1));
    d_lor float8;
    d_euc float8 := 0.0;
BEGIN
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) OR n IS NULL OR n < 33 THEN
        RAISE EXCEPTION 'pg_hyperspace: hybrid_distance requires matching dimensions >= 33 (33D Lorentz head + Euclidean tail)';
    END IF;
    d_lor := hyperspace.lorentz_distance(a[1:33], b[1:33]);
    IF n > 33 THEN
        d_euc := hyperspace.l2_distance(a[34:n], b[34:n]);
    END IF;
    RETURN lorentz_weight * d_lor + euclidean_weight * d_euc;
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.hybrid_lorentz_head(hybrid_vec float8[])
RETURNS float8[]
LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT hybrid_vec[1:33];
$$;

CREATE OR REPLACE FUNCTION hyperspace.hybrid_semantic_tail(hybrid_vec float8[])
RETURNS float8[]
LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT AS $$
    SELECT hybrid_vec[34:array_length(hybrid_vec, 1)];
$$;

-- -----------------------------------------------------------------------------
-- 4. Hyperbolic Light-Cone & Spatial Region Predicates (InCone, InBox, InBall)
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hyperspace.in_lorentz_cone(
    parent float8[],
    child float8[],
    max_radius float8 DEFAULT 1.5
) RETURNS bool
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(parent, 1), array_length(child, 1));
    spatial_dot float8 := 0.0;
    minkowski_dot float8;
BEGIN
    IF n IS NULL OR n < 2 OR max_radius < 0.0 THEN
        RETURN false;
    END IF;
    FOR i IN 2..n LOOP
        spatial_dot := spatial_dot + parent[i] * child[i];
    END LOOP;
    minkowski_dot := -parent[1] * child[1] + spatial_dot;
    RETURN (minkowski_dot >= -cosh(max_radius)) AND (child[1] >= parent[1]);
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.in_box(
    min_bounds float8[],
    max_bounds float8[],
    coords float8[]
) RETURNS bool
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
DECLARE
    n int := LEAST(array_length(min_bounds, 1), array_length(max_bounds, 1), array_length(coords, 1));
BEGIN
    IF n IS NULL OR n = 0 THEN
        RETURN false;
    END IF;
    FOR i IN 1..n LOOP
        IF coords[i] < min_bounds[i] OR coords[i] > max_bounds[i] THEN
            RETURN false;
        END IF;
    END LOOP;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION hyperspace.in_ball(
    center float8[],
    radius float8,
    coords float8[]
) RETURNS bool
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT AS $$
BEGIN
    IF radius < 0.0 THEN
        RETURN false;
    END IF;
    RETURN hyperspace.l2_distance(center, coords) <= radius;
END;
$$;

