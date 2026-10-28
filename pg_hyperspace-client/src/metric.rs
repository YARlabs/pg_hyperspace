//! Universal Multi-Metric & Non-Euclidean Spatial Kernels for PostgreSQL and HyperspaceDB.
//!
//! Supports all 7 native HyperspaceDB metrics:
//! 1. `l2` — Euclidean $L_2$ distance
//! 2. `cosine` — Cosine angular distance ($1 - \cos(\theta)$)
//! 3. `dot` — Negative Inner Product ($-\langle u, v \rangle$, matching `pgvector` `<#>` ordering)
//! 4. `poincare` — Hyperbolic Poincaré Ball geodesic distance
//! 5. `lorentz` — Hyperbolic Lorentz (Minkowski hyperboloid $\mathbb{H}^n$) geodesic distance
//! 6. `wasserstein` — 1D Earth Mover's / Optimal Transport ($W_1$) distance
//! 7. `hybrid` — Multi-geometry / Lexical + Semantic RRF fusion
//!
//! All kernels use 4-way accumulator unrolling so LLVM emits AVX2/AVX-512 (x86_64) or NEON
//! (Apple Silicon / ARM64) vector instructions without data-dependency stalls.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Supported distance metrics in HyperspaceDB and `pg_hyperspace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricKind {
    /// Euclidean $L_2$ distance (`vector_l2_ops`, `<->`).
    L2,
    /// Cosine distance (`vector_cosine_ops`, `<=>`).
    Cosine,
    /// Inner product (`vector_ip_ops`, `<#>`).
    Dot,
    /// Poincaré Ball hyperbolic distance (`vector_poincare_ops`, `<~>`).
    Poincare,
    /// Lorentz hyperboloid Minkowski distance (`vector_lorentz_ops`, `<@>`).
    Lorentz,
    /// 1D Wasserstein Optimal Transport distance (`vector_wasserstein_ops`, `<|>`).
    Wasserstein,
    /// Product-manifold Hybrid distance (`vector_hybrid_ops`, `<+>`):
    /// 33D Lorentz Hyperboloid Head ($\mathbb{H}^{32}$) + Euclidean Semantic Tail ($\mathbb{R}^{D-33}$).
    Hybrid,
}

/// Dimension of the Lorentz hyperboloid head ($\mathbb{H}^{32} \subset \mathbb{R}^{1,32}$) in
/// HyperspaceDB Product-Manifold Hybrid vectors (e.g. `801D = 33D Lorentz + 768D Euclidean`).
pub const HYBRID_LORENTZ_HEAD_DIM: usize = 33;

impl MetricKind {
    /// Canonical string identifier for this metric.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::Dot => "dot",
            Self::Poincare => "poincare",
            Self::Lorentz => "lorentz",
            Self::Wasserstein => "wasserstein",
            Self::Hybrid => "hybrid",
        }
    }

    /// Metric string passed to `POST /api/collections` when creating a `ShardedCollection` in
    /// HyperspaceDB v4 (`Wasserstein` uses an `l2` base index and activates optimal-transport
    /// evaluation via `SearchRequest::use_wasserstein = Some(true)`).
    #[must_use]
    pub const fn collection_metric_str(self) -> &'static str {
        match self {
            Self::L2 | Self::Wasserstein => "l2",
            Self::Cosine => "cosine",
            Self::Dot => "dot",
            Self::Poincare => "poincare",
            Self::Lorentz => "lorentz",
            Self::Hybrid => "hybrid",
        }
    }

    /// Parses a metric name or operator class name into [`MetricKind`].
    pub fn parse(raw: &str) -> Result<Self, Error> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "l2" | "euclidean" | "vector_l2_ops" => Ok(Self::L2),
            "cosine" | "cos" | "vector_cosine_ops" => Ok(Self::Cosine),
            "dot" | "ip" | "inner_product" | "innerproduct" | "vector_ip_ops" => Ok(Self::Dot),
            "poincare" | "poincaré" | "vector_poincare_ops" => Ok(Self::Poincare),
            "lorentz" | "minkowski" | "hyperboloid" | "vector_lorentz_ops" => Ok(Self::Lorentz),
            "wasserstein" | "emd" | "earth_mover" | "vector_wasserstein_ops" => {
                Ok(Self::Wasserstein)
            }
            "hybrid" | "vector_hybrid_ops" => Ok(Self::Hybrid),
            other => Err(Error::InvalidArgument(format!(
                "unsupported metric '{other}'; expected one of: l2, cosine, dot, poincare, lorentz, wasserstein, hybrid"
            ))),
        }
    }

    /// Computes exact distance between two `f64` slices using this metric.
    #[inline]
    pub fn distance_f64(self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != b.len() || a.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "vector dimension mismatch or empty: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        Ok(match self {
            Self::L2 => l2_distance_f64(a, b),
            Self::Cosine => cosine_distance_f64(a, b),
            Self::Dot => -dot_product_f64(a, b),
            Self::Poincare => poincare_distance_f64(a, b),
            Self::Lorentz => {
                if a.len() < 2 {
                    return Err(Error::InvalidArgument(
                        "Lorentz metric requires at least 2 dimensions".to_string(),
                    ));
                }
                lorentz_distance_f64(a, b)
            }
            Self::Wasserstein => wasserstein_1d_distance_f64(a, b),
            Self::Hybrid => {
                if a.len() < HYBRID_LORENTZ_HEAD_DIM {
                    return Err(Error::InvalidArgument(format!(
                        "Hybrid metric requires at least {HYBRID_LORENTZ_HEAD_DIM} dimensions (33D Lorentz head + Euclidean tail), got {}",
                        a.len()
                    )));
                }
                hybrid_distance_f64(a, b)
            }
        })
    }
}

/// 4-way unrolled Euclidean $L_2$ distance: $\sqrt{\sum_i (a_i - b_i)^2}$.
#[inline]
#[must_use]
pub fn l2_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut acc = [0.0_f64; 4];
    let (ca, ra) = a[..n].as_chunks::<4>();
    let (cb, rb) = b[..n].as_chunks::<4>();

    for (va, vb) in ca.iter().zip(cb.iter()) {
        let d0 = va[0] - vb[0];
        let d1 = va[1] - vb[1];
        let d2 = va[2] - vb[2];
        let d3 = va[3] - vb[3];
        acc[0] = d0.mul_add(d0, acc[0]);
        acc[1] = d1.mul_add(d1, acc[1]);
        acc[2] = d2.mul_add(d2, acc[2]);
        acc[3] = d3.mul_add(d3, acc[3]);
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (&x, &y) in ra.iter().zip(rb.iter()) {
        let d = x - y;
        sum = d.mul_add(d, sum);
    }
    sum.sqrt()
}

/// 4-way unrolled Euclidean inner product: $\sum_i a_i b_i$.
#[inline]
#[must_use]
pub fn dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut acc = [0.0_f64; 4];
    let (ca, ra) = a[..n].as_chunks::<4>();
    let (cb, rb) = b[..n].as_chunks::<4>();

    for (va, vb) in ca.iter().zip(cb.iter()) {
        acc[0] = va[0].mul_add(vb[0], acc[0]);
        acc[1] = va[1].mul_add(vb[1], acc[1]);
        acc[2] = va[2].mul_add(vb[2], acc[2]);
        acc[3] = va[3].mul_add(vb[3], acc[3]);
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (&x, &y) in ra.iter().zip(rb.iter()) {
        sum = x.mul_add(y, sum);
    }
    sum
}

/// 4-way unrolled Cosine distance: $1 - \frac{\langle a, b \rangle}{\|a\|_2 \|b\|_2}$.
#[inline]
#[must_use]
pub fn cosine_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = [0.0_f64; 4];
    let mut na = [0.0_f64; 4];
    let mut nb = [0.0_f64; 4];
    let (ca, ra) = a[..n].as_chunks::<4>();
    let (cb, rb) = b[..n].as_chunks::<4>();

    for (va, vb) in ca.iter().zip(cb.iter()) {
        for l in 0..4 {
            dot[l] = va[l].mul_add(vb[l], dot[l]);
            na[l] = va[l].mul_add(va[l], na[l]);
            nb[l] = vb[l].mul_add(vb[l], nb[l]);
        }
    }
    let mut d_sum = (dot[0] + dot[1]) + (dot[2] + dot[3]);
    let mut a_sum = (na[0] + na[1]) + (na[2] + na[3]);
    let mut b_sum = (nb[0] + nb[1]) + (nb[2] + nb[3]);
    for (&x, &y) in ra.iter().zip(rb.iter()) {
        d_sum = x.mul_add(y, d_sum);
        a_sum = x.mul_add(x, a_sum);
        b_sum = y.mul_add(y, b_sum);
    }
    let denom = (a_sum * b_sum).sqrt();
    if denom <= 1e-15 {
        return 1.0;
    }
    (1.0 - (d_sum / denom).clamp(-1.0, 1.0)).max(0.0)
}

/// Poincaré Ball geodesic distance (matches `hyperspace_core::PoincareMetric::distance`):
///
/// $$d_{\mathbb{B}}(u, v) = \operatorname{arcosh}\!\left(1 + \frac{2\|u - v\|_2^2}{(1 - \|u\|_2^2)(1 - \|v\|_2^2)}\right)$$
#[inline]
#[must_use]
pub fn poincare_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut acc_u = [0.0_f64; 4];
    let mut acc_v = [0.0_f64; 4];
    let mut acc_d = [0.0_f64; 4];
    let (ca, ra) = a[..n].as_chunks::<4>();
    let (cb, rb) = b[..n].as_chunks::<4>();

    for (va, vb) in ca.iter().zip(cb.iter()) {
        for l in 0..4 {
            let u = va[l];
            let v = vb[l];
            let d = u - v;
            acc_u[l] = u.mul_add(u, acc_u[l]);
            acc_v[l] = v.mul_add(v, acc_v[l]);
            acc_d[l] = d.mul_add(d, acc_d[l]);
        }
    }
    let mut norm_u_sq = (acc_u[0] + acc_u[1]) + (acc_u[2] + acc_u[3]);
    let mut norm_v_sq = (acc_v[0] + acc_v[1]) + (acc_v[2] + acc_v[3]);
    let mut diff_sq = (acc_d[0] + acc_d[1]) + (acc_d[2] + acc_d[3]);
    for (&u, &v) in ra.iter().zip(rb.iter()) {
        let d = u - v;
        norm_u_sq = u.mul_add(u, norm_u_sq);
        norm_v_sq = v.mul_add(v, norm_v_sq);
        diff_sq = d.mul_add(d, diff_sq);
    }

    // Clamp norms inside the open unit ball (1 - 1e-7) so arbitrary embeddings never NaN.
    let u_factor = (1.0 - norm_u_sq).max(1e-9);
    let v_factor = (1.0 - norm_v_sq).max(1e-9);
    let denom = (u_factor * v_factor).max(1e-9);
    let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
    arg.acosh()
}

/// Minkowski bilinear form on $\mathbb{R}^{1, n}$:
/// $\langle u, v \rangle_{\mathcal{L}} = -u_0 v_0 + \sum_{i=1}^{n} u_i v_i$.
#[inline]
#[must_use]
pub fn minkowski_inner_product_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let spatial = if n > 1 {
        dot_product_f64(&a[1..n], &b[1..n])
    } else {
        0.0
    };
    -a[0] * b[0] + spatial
}

/// Lorentz hyperboloid geodesic distance (matches `hyperspace_core::LorentzMetric::distance`):
///
/// $$d_{\mathcal{L}}(u, v) = \operatorname{arcosh}\!\left(\max(1 + 10^{-12}, -\langle u, v \rangle_{\mathcal{L}})\right)$$
#[inline]
#[must_use]
pub fn lorentz_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let inner_prod = minkowski_inner_product_f64(a, b);
    let arg = (-inner_prod).max(1.0 + 1e-12);
    arg.acosh()
}

/// Lifts a $d$-dimensional Euclidean vector $x \in \mathbb{R}^d$ onto the upper sheet of the
/// $(d+1)$-dimensional unit Lorentz hyperboloid $\mathbb{H}^d$:
/// $u_0 = \sqrt{1 + \|x\|_2^2}, \quad u_{1..=d} = x$.
#[inline]
#[must_use]
pub fn lift_euclidean_to_lorentz(spatial: &[f64]) -> Vec<f64> {
    let norm_sq = dot_product_f64(spatial, spatial);
    let mut out = Vec::with_capacity(spatial.len() + 1);
    out.push((1.0 + norm_sq).sqrt());
    out.extend_from_slice(spatial);
    out
}

/// Projects an arbitrary Euclidean vector into the open Poincaré ball of radius `< 1 - eps`
/// via conformal radial scaling: $x \mapsto \tanh(\|x\|) \frac{x}{\|x\|}$.
#[inline]
#[must_use]
pub fn project_euclidean_to_poincare(vec: &[f64]) -> Vec<f64> {
    let norm = dot_product_f64(vec, vec).sqrt();
    if norm <= 1e-12 {
        return vec![0.0; vec.len()];
    }
    let scale = (norm.tanh() / norm).min((1.0 - 1e-6) / norm);
    vec.iter().map(|&v| v * scale).collect()
}

/// Product-Manifold Hybrid distance matching `hyperspace_core::hybrid::HybridMetric::distance`:
///
/// Splits each vector into:
/// - **Lorentz Head (`0..33`)**: 33D Minkowski hyperboloid ($\mathbb{H}^{32}$) capturing hierarchical
///   and causal structure via $d_{\mathcal{L}}(a_{0..33}, b_{0..33})$.
/// - **Euclidean Semantic Tail (`33..D`)**: Dense semantic representation (e.g., 768D in `AsymmetricHybrid801`)
///   evaluated via $d_{L_2}(a_{33..D}, b_{33..D})$.
///
/// $$d_{\text{hybrid}}(a, b) = d_{\mathcal{L}}(a_{0..33}, b_{0..33}) + \|a_{33..D} - b_{33..D}\|_2$$
#[inline]
#[must_use]
pub fn hybrid_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    hybrid_weighted_distance_f64(a, b, 1.0, 1.0)
}

/// Weighted Product-Manifold Hybrid distance allowing custom weighting between the 33D Lorentz
/// hyperboloid head and the $(D - 33)$-dimensional Euclidean semantic tail:
///
/// $$d_{\text{hybrid}}(a, b) = w_{\mathcal{L}} \cdot d_{\mathcal{L}}(a_{0..33}, b_{0..33}) + w_{E} \cdot \|a_{33..D} - b_{33..D}\|_2$$
#[inline]
#[must_use]
pub fn hybrid_weighted_distance_f64(
    a: &[f64],
    b: &[f64],
    lorentz_weight: f64,
    euclidean_weight: f64,
) -> f64 {
    let n = a.len().min(b.len());
    if n < HYBRID_LORENTZ_HEAD_DIM {
        return 0.0;
    }
    let d_lor = lorentz_distance_f64(&a[..HYBRID_LORENTZ_HEAD_DIM], &b[..HYBRID_LORENTZ_HEAD_DIM]);
    let d_euc = if n > HYBRID_LORENTZ_HEAD_DIM {
        l2_distance_f64(
            &a[HYBRID_LORENTZ_HEAD_DIM..n],
            &b[HYBRID_LORENTZ_HEAD_DIM..n],
        )
    } else {
        0.0
    };
    lorentz_weight.mul_add(d_lor, euclidean_weight * d_euc)
}

/// Composes a Product-Manifold Hybrid vector from a hierarchical spatial vector and a dense
/// Euclidean semantic vector:
/// - If `hierarchy` has 32 dimensions, lifts it onto the 33D unit Lorentz hyperboloid $\mathbb{H}^{32}$
///   by prepending $u_0 = \sqrt{1 + \|x\|_2^2}$.
/// - If `hierarchy` already has 33 dimensions, validates/repairs $u_0 = \sqrt{1 + \sum_{i=1}^{32} u_i^2}$
///   so the Lorentz head lies strictly on the upper hyperboloid sheet.
/// - Appends `semantic_tail` (e.g., 768D -> 801D `AsymmetricHybrid801` vector).
#[inline]
#[must_use]
pub fn compose_hybrid_vector(hierarchy: &[f64], semantic_tail: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(HYBRID_LORENTZ_HEAD_DIM + semantic_tail.len());
    if hierarchy.len() == HYBRID_LORENTZ_HEAD_DIM {
        let spatial = &hierarchy[1..HYBRID_LORENTZ_HEAD_DIM];
        let u0 = (1.0 + dot_product_f64(spatial, spatial)).sqrt();
        out.push(u0);
        out.extend_from_slice(spatial);
    } else {
        let mut spatial = [0.0_f64; HYBRID_LORENTZ_HEAD_DIM - 1];
        let copy_len = hierarchy.len().min(HYBRID_LORENTZ_HEAD_DIM - 1);
        spatial[..copy_len].copy_from_slice(&hierarchy[..copy_len]);
        let u0 = (1.0 + dot_product_f64(&spatial, &spatial)).sqrt();
        out.push(u0);
        out.extend_from_slice(&spatial);
    }
    out.extend_from_slice(semantic_tail);
    out
}

/// 1D Wasserstein ($W_1$ Earth Mover's) distance over normalized non-negative histograms
/// (matches `hyperspace_core::wasserstein::WassersteinDistance::compute`).
#[inline]
#[must_use]
pub fn wasserstein_1d_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    for &v in &a[..n] {
        if v > 0.0 {
            sum_a += v;
        }
    }
    for &v in &b[..n] {
        if v > 0.0 {
            sum_b += v;
        }
    }
    if sum_a <= 1e-9 || sum_b <= 1e-9 {
        return f64::MAX;
    }

    let inv_a = 1.0 / sum_a;
    let inv_b = 1.0 / sum_b;
    let mut cdf_a = 0.0_f64;
    let mut cdf_b = 0.0_f64;
    let mut dist = 0.0_f64;

    for i in 0..n {
        let va = if a[i] > 0.0 { a[i] * inv_a } else { 0.0 };
        let vb = if b[i] > 0.0 { b[i] * inv_b } else { 0.0 };
        cdf_a += va;
        cdf_b += vb;
        dist += (cdf_a - cdf_b).abs();
    }
    dist
}

/// Checks if `child` lies within the Lorentz future light-cone of `parent` within hyperbolic
/// radius `max_radius` (matches `LorentzTreeProjector::is_in_light_cone`):
///
/// $$\langle \text{parent}, \text{child} \rangle_{\mathcal{L}} \ge -\cosh(r_{\max}) \quad \wedge \quad \text{child}_0 \ge \text{parent}_0$$
#[inline]
#[must_use]
pub fn lorentz_light_cone_contains(parent: &[f64], child: &[f64], max_radius: f64) -> bool {
    if parent.len() < 2 || child.len() < 2 || max_radius < 0.0 {
        return false;
    }
    let dot = minkowski_inner_product_f64(parent, child);
    let cosh_r = max_radius.cosh();
    dot >= -cosh_r && child[0] >= parent[0]
}

/// Checks if `coords` lies inside the ConE angular cone `(axes, apertures, cen)`
/// (matches `hyperspace_core::region::ConeRegion::check_contains`).
#[inline]
#[must_use]
pub fn angular_cone_contains(axes: &[f64], apertures: &[f64], cen: f64, coords: &[f64]) -> bool {
    let n = axes.len().min(apertures.len()).min(coords.len());
    for d in 0..n {
        let threshold = (apertures[d] * 0.5).sin().abs() + cen;
        let dist_to_axis = ((coords[d] - axes[d]) * 0.5).sin().abs();
        if dist_to_axis > threshold {
            return false;
        }
    }
    true
}

/// Checks if `coords` lies within the Axis-Aligned Bounding Box `[min_bounds, max_bounds]`
/// (matches `hyperspace_core::region::BoxRegion::check_contains`).
#[inline]
#[must_use]
pub fn in_box_contains(min_bounds: &[f64], max_bounds: &[f64], coords: &[f64]) -> bool {
    let n = min_bounds.len().min(max_bounds.len()).min(coords.len());
    for d in 0..n {
        if coords[d] < min_bounds[d] || coords[d] > max_bounds[d] {
            return false;
        }
    }
    true
}

/// Checks if `coords` lies within the Euclidean ball `B(center, radius)`
/// (matches `hyperspace_core::region::BallRegion::check_contains`).
#[inline]
#[must_use]
pub fn in_ball_contains(center: &[f64], radius: f64, coords: &[f64]) -> bool {
    if radius < 0.0 {
        return false;
    }
    let n = center.len().min(coords.len());
    let dist = l2_distance_f64(&center[..n], &coords[..n]);
    dist <= radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_parsing_and_distances() {
        assert_eq!(MetricKind::parse("vector_l2_ops").unwrap(), MetricKind::L2);
        assert_eq!(
            MetricKind::parse("vector_cosine_ops").unwrap(),
            MetricKind::Cosine
        );
        assert_eq!(MetricKind::parse("vector_ip_ops").unwrap(), MetricKind::Dot);
        assert_eq!(MetricKind::parse("poincare").unwrap(), MetricKind::Poincare);
        assert_eq!(MetricKind::parse("lorentz").unwrap(), MetricKind::Lorentz);
        assert_eq!(
            MetricKind::parse("wasserstein").unwrap(),
            MetricKind::Wasserstein
        );

        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert!((l2_distance_f64(&a, &b) - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!((cosine_distance_f64(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn poincare_and_lorentz_identities() {
        let u_euc = [0.1, 0.2, -0.1, 0.05];
        let v_euc = [0.2, 0.4, -0.2, 0.10];

        let u_p = project_euclidean_to_poincare(&u_euc);
        let v_p = project_euclidean_to_poincare(&v_euc);
        assert!(poincare_distance_f64(&u_p, &u_p) < 1e-7);
        assert!(poincare_distance_f64(&u_p, &v_p) > 0.0);

        let u_l = lift_euclidean_to_lorentz(&u_euc);
        let v_l = lift_euclidean_to_lorentz(&v_euc);
        let d_self = lorentz_distance_f64(&u_l, &u_l);
        assert!(d_self < 1e-5);
        let d_uv = lorentz_distance_f64(&u_l, &v_l);
        assert!(d_uv > 0.0);
        assert!(lorentz_light_cone_contains(&u_l, &v_l, d_uv + 1e-4));
    }

    #[test]
    fn wasserstein_1d_exact() {
        let p = [1.0, 0.0, 0.0];
        let q = [0.0, 0.0, 1.0];
        assert!((wasserstein_1d_distance_f64(&p, &q) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn product_manifold_hybrid_vector_exact() {
        let mut h1 = [0.0_f64; 32];
        h1[0] = 0.1;
        h1[1] = 0.2;
        let mut h2 = [0.0_f64; 32];
        h2[0] = 0.2;
        h2[1] = 0.4;

        let tail1 = [1.0, 0.0, 0.0, 0.0];
        let tail2 = [0.0, 1.0, 0.0, 0.0];

        let vec1 = compose_hybrid_vector(&h1, &tail1);
        let vec2 = compose_hybrid_vector(&h2, &tail2);
        assert_eq!(vec1.len(), 37);
        assert_eq!(vec2.len(), 37);

        let expected_lor = lorentz_distance_f64(&vec1[..33], &vec2[..33]);
        let expected_euc = l2_distance_f64(&vec1[33..], &vec2[33..]);
        let d_hybrid = MetricKind::Hybrid.distance_f64(&vec1, &vec2).unwrap();
        assert!((d_hybrid - (expected_lor + expected_euc)).abs() < 1e-12);
        assert!(lorentz_light_cone_contains(
            &vec1[..33],
            &vec2[..33],
            expected_lor + 1e-4
        ));
    }
}
