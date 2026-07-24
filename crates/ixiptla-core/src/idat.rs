/// Accumulates a PNG file's `IDAT` chunk payloads into the single
/// contiguous byte stream `zlib` decompression needs (spec §5.6: "the
/// IDAT chunks... are treated as a single stream").
///
/// The overwhelmingly common case -- every encoder this crate has been
/// benchmarked against (`png`, `image`, and this crate's own
/// [`crate::Png`] test fixtures) -- emits exactly *one* `IDAT` chunk per
/// file; splitting compressed data across multiple `IDAT`s is a legal but
/// rare accommodation for streaming encoders with a bounded chunk size.
/// Unconditionally copying every chunk's payload into a freshly allocated
/// `Vec` (this type's predecessor) paid for a whole image-sized
/// allocation + memcpy on *every* decode, even though for a single-chunk
/// file that data is already sitting in `input` as one contiguous slice
/// with nothing to concatenate -- `Png::decode` could just borrow it.
/// Measured via `ixiptla-bench`'s `end_to_end.rs` `AllocProfiler` output:
/// this redundant copy was the largest single contributor to `ixiptla`
/// decoding with roughly double the peer `png`/`image` crates' peak
/// allocation at 1024×1024 (8.4MB vs ~4.2MB) -- one whole extra
/// image-sized buffer that those decoders don't pay for.
///
/// This type defers the allocate-and-concatenate decision until it's
/// actually forced by a *second* `push`: the first pushed chunk is kept
/// as a plain borrow, and only promoted to an owned, growing `Vec` if a
/// second chunk shows up, at which point it behaves like the old
/// eager-concatenation code (still bounded by `total_len_hint` to avoid
/// `Vec`'s doubling reallocations on the rare multi-chunk file).
#[derive(Debug)]
pub enum IdatAccumulator<'a> {
    /// No `IDAT` chunk seen yet.
    Empty,
    /// Exactly one `IDAT` chunk seen so far: a zero-copy borrow of its
    /// payload, straight out of the original file bytes.
    Single(&'a [u8]),
    /// Two or more `IDAT` chunks seen: genuinely concatenated into an
    /// owned buffer, since there's no way to hand back a single
    /// contiguous borrow across non-adjacent chunk payloads.
    Multi(Vec<u8>),
}

impl<'a> IdatAccumulator<'a> {
    /// Start with no `IDAT` data accumulated yet.
    #[must_use]
    pub const fn new() -> Self {
        Self::Empty
    }

    /// Fold in one more `IDAT` chunk's payload, in file order.
    ///
    /// `total_len_hint` upper-bounds how many bytes of `IDAT` payload
    /// could possibly exist in this file (e.g. `input.len()`, the same
    /// conservative bound the old eager-concatenation code reserved
    /// against) -- used only if/when this promotes to `Multi`, so a
    /// multi-chunk file still gets one right-sized allocation instead of
    /// `Vec`'s repeated grow-and-copy doubling.
    pub fn push(&mut self, data: &'a [u8], total_len_hint: usize) {
        match self {
            Self::Empty => *self = Self::Single(data),
            Self::Single(first) => {
                let mut combined = Vec::with_capacity(total_len_hint);
                combined.extend_from_slice(first);
                combined.extend_from_slice(data);
                *self = Self::Multi(combined);
            }
            Self::Multi(combined) => combined.extend_from_slice(data),
        }
    }

    /// `true` if no `IDAT` chunk was ever pushed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// The accumulated `IDAT` payload as one contiguous slice -- a
    /// zero-copy borrow of `input` for the single-chunk case, or a borrow
    /// of the owned concatenation buffer for the multi-chunk case.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Single(data) => data,
            Self::Multi(combined) => combined,
        }
    }
}

impl Default for IdatAccumulator<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::IdatAccumulator;

    #[test]
    fn empty_accumulator_has_no_data() {
        let acc = IdatAccumulator::new();
        assert!(acc.is_empty());
        assert_eq!(acc.as_slice(), &[] as &[u8]);
    }

    #[test]
    fn single_chunk_is_a_zero_copy_borrow() {
        let payload = [1u8, 2, 3, 4];
        let mut acc = IdatAccumulator::new();
        acc.push(&payload, 4);
        assert!(!acc.is_empty());
        // Not just equal in content -- literally the same underlying
        // bytes, i.e. `Single`, not promoted to an owned `Multi` copy.
        assert!(matches!(acc, IdatAccumulator::Single(_)));
        assert_eq!(acc.as_slice(), &payload);
    }

    #[test]
    fn two_chunks_concatenate_in_order() {
        let a = [1u8, 2, 3];
        let b = [4u8, 5];
        let mut acc = IdatAccumulator::new();
        acc.push(&a, 5);
        acc.push(&b, 5);
        assert!(matches!(acc, IdatAccumulator::Multi(_)));
        assert_eq!(acc.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn three_chunks_concatenate_in_order() {
        let a = [1u8];
        let b = [2u8, 3];
        let c = [4u8, 5, 6];
        let mut acc = IdatAccumulator::new();
        acc.push(&a, 6);
        acc.push(&b, 6);
        acc.push(&c, 6);
        assert_eq!(acc.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn default_matches_new() {
        assert!(IdatAccumulator::default().is_empty());
    }
}
