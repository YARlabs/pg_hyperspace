//! Zero-allocation PostgreSQL `ItemPointerData` (TID) <-> HyperspaceDB `u32` ID codec.
//!
//! ## Encoding Design
//! A PostgreSQL heap tuple identifier (`ItemPointerData`) consists of:
//! - `block`: 32-bit `BlockNumber` (`0..0xFFFF_FFFE`; `0xFFFF_FFFF` is `InvalidBlockNumber`)
//! - `offset`: 16-bit `OffsetNumber` (`1..=MaxOffsetNumber`, where `MaxHeapTuplesPerPage` = 291
//!   on standard 8 KB pages, and `< 512` on all standard heap pages).
//!
//! To eliminate heap string allocations on the hot search and insert paths while supporting
//! arbitrary multi-terabyte tables:
//! 1. **Direct Bit-Packed Mode (`block < 2^22`, up to 32 GB per table, `offset < 512`):**
//!    The `u32` point ID has bit 31 = `0`, bits `9..31` = `block` (22 bits), and bits `0..9` =
//!    `offset` (9 bits). Decoding a `u32` with bit 31 == `0` takes a single CPU cycle (bitshift +
//!    mask) with zero string parsing or hash lookups.
//! 2. **Extended Hash Mode (`block >= 2^22` or `offset >= 512`):**
//!    Bit 31 is set to `1`, bits `0..31` store a deterministic 31-bit FNV-1a/Murmur3 mix of
//!    `(block, offset)`, and the exact `"block:offset"` string is read from the `_pg_tid`
//!    metadata field.
//!
//! In both modes, `_pg_tid` can optionally be attached to point metadata so full recovery and
//! debugging tools always have human-readable tuple coordinates.

use std::collections::HashMap;

use crate::error::Error;

/// Metadata key used to store exact `"block:offset"` coordinates in HyperspaceDB points.
pub const META_PG_TID_KEY: &str = "_pg_tid";

/// PostgreSQL's `InvalidBlockNumber` (`0xFFFF_FFFF`).
pub const INVALID_BLOCK_NUMBER: u32 = 0xFFFF_FFFF;

/// PostgreSQL's `InvalidOffsetNumber` (`0`).
pub const INVALID_OFFSET_NUMBER: u16 = 0;

/// Maximum valid `OffsetNumber` on an 8 KB PostgreSQL page (`BLCKSZ / sizeof(ItemIdData)` = 2048).
pub const MAX_OFFSET_NUMBER: u16 = 2048;

/// Threshold for direct lossless 31-bit packing (`2^22` blocks = 4,194,304 pages = 32 GiB).
const DIRECT_MAX_BLOCK: u32 = (1 << 22) - 1;

/// Threshold for direct lossless offset packing (`2^9 - 1` = 511; standard PG max is 291).
const DIRECT_MAX_OFFSET: u16 = (1 << 9) - 1;

/// High-bit tag indicating extended hash mode.
const EXTENDED_TAG_BIT: u32 = 1 << 31;

/// Validated PostgreSQL Heap Tuple Identifier (`(block, offset)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PgTid {
    /// Heap block number (`0..0xFFFF_FFFE`).
    pub block: u32,
    /// Page line pointer offset (`1..=2048`).
    pub offset: u16,
}

impl PgTid {
    /// Constructs and validates a `PgTid`.
    ///
    /// Rejects `InvalidBlockNumber` (`0xFFFF_FFFF` / `P_NEW`), `InvalidOffsetNumber` (`0`),
    /// and offsets above `MAX_OFFSET_NUMBER`.
    #[inline]
    pub const fn new(block: u32, offset: u16) -> Result<Self, Error> {
        if block == INVALID_BLOCK_NUMBER {
            return Err(Error::MalformedTid(String::new()));
        }
        if offset == INVALID_OFFSET_NUMBER || offset > MAX_OFFSET_NUMBER {
            return Err(Error::MalformedTid(String::new()));
        }
        Ok(Self { block, offset })
    }

    /// Validated constructor with detailed error messages.
    pub fn try_from_parts(block: u32, offset: u16) -> Result<Self, Error> {
        if block == INVALID_BLOCK_NUMBER {
            return Err(Error::MalformedTid(format!(
                "invalid block number {block:#X} (P_NEW/InvalidBlockNumber)"
            )));
        }
        if offset == INVALID_OFFSET_NUMBER || offset > MAX_OFFSET_NUMBER {
            return Err(Error::MalformedTid(format!(
                "invalid offset number {offset} (expected 1..={MAX_OFFSET_NUMBER})"
            )));
        }
        Ok(Self { block, offset })
    }

    /// Encodes this `PgTid` into a HyperspaceDB `u32` point ID.
    #[inline]
    #[must_use]
    pub const fn to_point_id(self) -> u32 {
        if self.block <= DIRECT_MAX_BLOCK && self.offset <= DIRECT_MAX_OFFSET {
            (self.block << 9) | (self.offset as u32)
        } else {
            // Deterministic 64->31 bit MurmurHash3 finalizer mix with MSB set
            let combined = ((self.block as u64) << 16) | (self.offset as u64);
            let mut h = combined ^ (combined >> 33);
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^= h >> 33;
            EXTENDED_TAG_BIT | ((h as u32) & 0x7FFF_FFFF)
        }
    }

    /// Encodes this `PgTid` into a 64-bit integer (`(block as u64) << 16 | offset as u64`)
    /// for outbox storage and SQL functions.
    #[inline]
    #[must_use]
    pub const fn to_u64(self) -> u64 {
        ((self.block as u64) << 16) | (self.offset as u64)
    }

    /// Decodes a `PgTid` from its lossless 64-bit integer representation.
    #[inline]
    pub fn from_u64(encoded: u64) -> Result<Self, Error> {
        let block = (encoded >> 16) as u32;
        let offset = (encoded & 0xFFFF) as u16;
        Self::try_from_parts(block, offset)
    }

    /// Fast-path decode from a HyperspaceDB `u32` point ID and optional point metadata.
    ///
    /// - If bit 31 of `point_id` is `0`, decodes `(block, offset)` in $O(1)$ using bitshifts
    ///   without touching `metadata`.
    /// - If bit 31 is `1`, parses the `"_pg_tid"` entry from `metadata`.
    #[inline]
    pub fn from_point_id_and_meta(
        point_id: u32,
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<Self, Error> {
        if (point_id & EXTENDED_TAG_BIT) == 0 {
            let block = point_id >> 9;
            let offset = (point_id & 0x1FF) as u16;
            return Self::try_from_parts(block, offset);
        }

        if let Some(meta) = metadata
            && let Some(raw) = meta.get(META_PG_TID_KEY)
        {
            return Self::from_str_repr(raw);
        }

        Err(Error::MalformedTid(format!(
            "extended point_id {point_id:#X} requires '{META_PG_TID_KEY}' in metadata"
        )))
    }

    /// Formats the TID as `"block:offset"`.
    #[must_use]
    pub fn to_str_repr(self) -> String {
        format!("{}:{}", self.block, self.offset)
    }

    /// Parses `"block:offset"` into a validated `PgTid`.
    pub fn from_str_repr(s: &str) -> Result<Self, Error> {
        let (b_str, o_str) = s
            .split_once(':')
            .ok_or_else(|| Error::MalformedTid(format!("expected 'block:offset', got '{s}'")))?;
        let block: u32 = b_str
            .parse()
            .map_err(|_| Error::MalformedTid(format!("invalid block number in '{s}'")))?;
        let offset: u16 = o_str
            .parse()
            .map_err(|_| Error::MalformedTid(format!("invalid offset number in '{s}'")))?;
        Self::try_from_parts(block, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_mode_roundtrips_without_metadata() {
        for (block, offset) in [(0, 1), (1, 42), (123_456, 291), (DIRECT_MAX_BLOCK, 511)] {
            let tid = PgTid::try_from_parts(block, offset).expect("valid tid");
            let id = tid.to_point_id();
            assert_eq!(id & EXTENDED_TAG_BIT, 0);
            let decoded = PgTid::from_point_id_and_meta(id, None).expect("decode");
            assert_eq!(decoded, tid);
            assert_eq!(PgTid::from_u64(tid.to_u64()).expect("u64"), tid);
        }
    }

    #[test]
    fn extended_mode_uses_metadata_fallback() {
        let tid = PgTid::try_from_parts(DIRECT_MAX_BLOCK + 100, 17).expect("valid tid");
        let id = tid.to_point_id();
        assert_ne!(id & EXTENDED_TAG_BIT, 0);

        let mut meta = HashMap::new();
        meta.insert(META_PG_TID_KEY.to_string(), tid.to_str_repr());
        let decoded = PgTid::from_point_id_and_meta(id, Some(&meta)).expect("decode via meta");
        assert_eq!(decoded, tid);
    }

    #[test]
    fn rejects_invalid_postgres_sentinels() {
        assert!(PgTid::try_from_parts(INVALID_BLOCK_NUMBER, 1).is_err());
        assert!(PgTid::try_from_parts(0, INVALID_OFFSET_NUMBER).is_err());
        assert!(PgTid::try_from_parts(0, MAX_OFFSET_NUMBER + 1).is_err());
        assert!(PgTid::from_str_repr("4294967295:1").is_err());
        assert!(PgTid::from_str_repr("0:0").is_err());
    }
}
