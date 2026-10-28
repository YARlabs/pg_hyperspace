//! Zero-leak `pgvector` binary `Datum` decoder.
//!
//! ## Memory Safety Fix over `pg_pinecone`
//! In `pg_pinecone` (`am.rs:parse_vector`), `pg_sys::pg_detoast_datum` is called without freeing
//! the newly allocated detoasted `varlena` buffer when the datum was compressed or stored
//! out-of-line in TOAST. During full-table index builds or iterative scans, that leaks 3–12 KB per
//! row into `CurrentMemoryContext`.
//!
//! Here, [`parse_vector_f64`] and [`parse_vector_f32`] compare the detoasted pointer against the
//! input datum pointer and immediately call `pg_sys::pfree` on the detoasted buffer once its
//! coordinates have been copied into the Rust `Vec`.

use pgrx::prelude::*;

/// Binary layout of a `pgvector` `vector` datum:
/// - `0..4`: 4-byte `varlena` header (`vl_len_`)
/// - `4..6`: `dim` (`i16`, number of `f32` coordinates)
/// - `6..8`: `unused` (`i16`, always `0`)
/// - `8..(8 + 4 * dim)`: contiguous `f32` array
const VECTOR_HEADER_BYTES: usize = 8;

/// Reads a `pgvector` `Datum` into a `Vec<f64>` (the native coordinate type of HyperspaceDB)
/// and immediately frees any temporary detoasted `varlena` buffer allocated by Postgres.
///
/// # Safety
/// `datum` must be a valid non-null `pgvector` `vector` `Datum`.
#[inline]
pub(crate) unsafe fn parse_vector_f64(datum: pg_sys::Datum) -> Vec<f64> {
    unsafe {
        let raw_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
        let detoasted = pg_sys::pg_detoast_datum(raw_ptr);
        let base = detoasted.cast::<u8>();
        let dim_i16 = std::ptr::read_unaligned(base.add(4).cast::<i16>());
        let dim = dim_i16.max(0) as usize;
        let data_base = base.add(VECTOR_HEADER_BYTES);

        let mut coords = Vec::with_capacity(dim);
        for i in 0..dim {
            let elem_bytes = std::ptr::read_unaligned(data_base.add(i * 4).cast::<[u8; 4]>());
            coords.push(f64::from(f32::from_ne_bytes(elem_bytes)));
        }

        // Prevent TOAST memory leaks in long-lived memory contexts.
        if detoasted != raw_ptr {
            pg_sys::pfree(detoasted.cast());
        }
        coords
    }
}

/// Validates that all coordinates in `coords` are finite (`!is_nan() && !is_infinite()`).
#[inline]
pub(crate) fn validate_finite_f64(coords: &[f64], context: &str) {
    if coords.is_empty() {
        error!("pg_hyperspace: {context} received an empty (0-dimensional) vector");
    }
    for (idx, &v) in coords.iter().enumerate() {
        if !v.is_finite() {
            error!(
                "pg_hyperspace: {context} vector contains non-finite coordinate ({v}) at dimension {idx}"
            );
        }
    }
}
