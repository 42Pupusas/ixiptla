//! SIMD-accelerated wide-lane byte addition for defiltering ops that have
//! no true per-byte data dependency across the row.
//!
//! `Up` (`Recon(x) = Filt(x) + Recon(above)`) is the motivating case: every
//! output byte only depends on the byte directly above it in the previous
//! row, never on another byte in the *same* row, so all bytes in a row can
//! be reconstructed in parallel -- unlike `Sub`/`Average`/`Paeth`, which
//! have a genuine sequential dependency at `bpp` stride within the row
//! itself. That makes `Up` the natural first (and easiest, and fully
//! correct with zero cross-lane bookkeeping) SIMD target.
//!
//! Only `x86_64` gets a real SIMD implementation here, per this crate's
//! current target. SSE2 is part of the `x86_64` baseline ABI (every `x86_64`
//! CPU has it -- it's not an optional extension the way AVX2/AVX-512 are),
//! so no runtime feature detection is needed to use it unconditionally.
//! Non-x86_64 targets fall back to the plain scalar loop so the crate still
//! builds and behaves correctly everywhere; the two implementations live
//! behind a single `#[cfg(target_arch = "x86_64")]` split *inside this one
//! method*, rather than sprinkling architecture `#[cfg]`s through
//! `Defilter`'s business logic -- callers just see one `WrappingByteAdder::
//! add_wrapping` regardless of target.

/// Wrapping (mod-256) elementwise byte addition, `dst[i] += src[i]`.
#[derive(Debug)]
pub struct WrappingByteAdder;

impl WrappingByteAdder {
    /// Adds `src[i]` into `dst[i]` in place for every `i`, wrapping on
    /// overflow -- exactly PNG's `u8::wrapping_add` semantics.
    ///
    /// # Panics
    ///
    /// If `dst.len() != src.len()`.
    pub fn add_wrapping(dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len(), "add_wrapping: length mismatch");
        #[cfg(target_arch = "x86_64")]
        {
            Self::add_wrapping_sse2(dst, src);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::add_wrapping_scalar(dst, src);
        }
    }

    /// SSE2 byte-lane add (`paddb`), 16 bytes per instruction.
    //
    // `_mm_add_epi8` operates on 8-bit lanes with silent wraparound on
    // overflow (there's no signed/unsigned distinction at the instruction
    // level for plain addition, and no saturation -- that's `paddsb`/
    // `paddusb`, different instructions), which is precisely `wrapping_add`
    // -- no masking or correction pass needed after the SIMD op.
    #[cfg(target_arch = "x86_64")]
    fn add_wrapping_sse2(dst: &mut [u8], src: &[u8]) {
        use std::arch::x86_64::{_mm_add_epi8, _mm_loadu_si128, _mm_storeu_si128};

        const LANES: usize = 16; // one __m128i = 16 bytes.
        let len = dst.len();
        let chunks = len / LANES;

        // SAFETY: SSE2 is guaranteed present on every x86_64 target (it's
        // part of the baseline ABI, unlike AVX2/AVX-512), so this isn't
        // gated behind runtime `is_x86_feature_detected!` the way an
        // optional extension would need. `_mm_loadu_si128`/
        // `_mm_storeu_si128` are the *unaligned* load/store forms
        // specifically so `dst`/`src`'s alignment (arbitrary -- they're
        // slices into a `Vec<u8>` sliced at row boundaries, not aligned
        // allocations) never matters for correctness, only for speed.
        // Each `offset` stays within `0..chunks*LANES <= len`, so every
        // pointer + 16-byte read/write is in bounds for both slices (equal
        // length, asserted above).
        unsafe {
            for i in 0..chunks {
                let offset = i * LANES;
                let d = _mm_loadu_si128(dst.as_ptr().add(offset).cast());
                let s = _mm_loadu_si128(src.as_ptr().add(offset).cast());
                let sum = _mm_add_epi8(d, s);
                _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast(), sum);
            }
        }

        Self::add_wrapping_scalar(&mut dst[chunks * LANES..], &src[chunks * LANES..]);
    }

    /// Plain byte-at-a-time fallback: used for the tail remainder past the
    /// last full SIMD chunk on `x86_64`, and as the entire implementation on
    /// any other architecture.
    fn add_wrapping_scalar(dst: &mut [u8], src: &[u8]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = d.wrapping_add(s);
        }
    }
}

/// SIMD-accelerated wrapping prefix sum, the operation `Sub`/`Average`/
/// `Paeth` all reduce to once you deinterleave a row into its `bpp`
/// mutually-independent lanes.
///
/// `Sub`'s reconstruction is `Recon(x) = Filt(x) + Recon(x - bpp)`: within
/// *one* of the `bpp` interleaved lanes (all bytes at a fixed `i % bpp`),
/// that's exactly `lane[i] = lane[i] + lane[i-1] + lane[i-2] + ... +
/// lane[0]` mod 256 -- an inclusive running ("prefix") sum. The lanes
/// themselves have no dependency on each other, only elements *within* a
/// lane depend on earlier elements of that same lane; that's the
/// structural reason `Sub` can't be vectorized directly the way `Up` was
/// (adjacent bytes in the row belong to different lanes and don't share a
/// dependency, but a byte and the one `bpp` positions before it, in the
/// *same* lane, is a true chain), but *within* one deinterleaved lane, a
/// prefix sum is a textbook parallel ("SWAR"/Hillis-Steele scan)
/// primitive.
#[derive(Debug)]
pub struct WrappingPrefixSum;

impl WrappingPrefixSum {
    /// Computes an in-place inclusive wrapping (mod-256) prefix sum:
    /// afterwards `data[i] == data_before[0] + data_before[1] + ... +
    /// data_before[i]` (mod 256), for every `i`.
    pub fn scan(data: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            Self::scan_sse2(data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::scan_scalar(data);
        }
    }

    /// SSE2 Hillis-Steele scan: within each 16-byte chunk, four
    /// `shift-then-add` rounds (shifting by 1, 2, 4, then 8 bytes) compute
    /// the chunk's own inclusive prefix sum in `log2(16) = 4` dependent
    /// SIMD instructions instead of 16 dependent scalar `add`s. Chunks
    /// then need their running total carried in from all chunks before
    /// them -- that carry is still an inherently serial per-chunk step,
    /// but there are only `len / 16` chunks, not `len` individual bytes,
    /// so the serial part of the work shrinks by ~16x.
    //
    // `_mm_slli_si128(x, n)` is a *byte* shift of the whole 128-bit lane
    // (not a per-element shift -- there's no such per-byte-element shift
    // instruction), shifting bytes toward the high end and filling the
    // low `n` bytes with zero. Adding a byte-shifted copy of `x` into
    // itself is exactly one round of a prefix-sum reduction tree: after
    // shifting by 1 and adding, `x[i]` holds `orig[i] + orig[i-1]`; doing
    // that again shifted by 2 combines pairs of pairs into runs of 4; by
    // 4 into runs of 8; by 8 into the full 16-byte prefix sum. `paddb`
    // (`_mm_add_epi8`) wraps mod 256 on overflow with no extra masking,
    // which is exactly PNG's `wrapping_add` semantics.
    #[cfg(target_arch = "x86_64")]
    fn scan_sse2(data: &mut [u8]) {
        use std::arch::x86_64::{
            _mm_add_epi8, _mm_loadu_si128, _mm_set1_epi8, _mm_setzero_si128, _mm_slli_si128,
            _mm_storeu_si128,
        };

        const LANES: usize = 16;
        let len = data.len();
        let chunks = len / LANES;

        // Running total carried out of the previous chunk, broadcast to
        // all 16 lanes so it can just be added into the next chunk's own
        // (already-computed) internal prefix sum. Starts at 0 (there's no
        // carry into the first chunk).
        //
        // SAFETY: `offset` ranges over `0..chunks*LANES <= len`, so every
        // 16-byte unaligned load/store stays in bounds; `_mm_loadu_si128`/
        // `_mm_storeu_si128` don't require any particular alignment.
        unsafe {
            let mut carry = _mm_setzero_si128();
            for i in 0..chunks {
                let offset = i * LANES;
                let mut x = _mm_loadu_si128(data.as_ptr().add(offset).cast());
                x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
                x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
                x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
                x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
                x = _mm_add_epi8(x, carry);
                _mm_storeu_si128(data.as_mut_ptr().add(offset).cast(), x);

                // The new carry for the *next* chunk is this chunk's last
                // (highest-index) lane, i.e. its own total -- broadcast
                // across all 16 lanes via a byte array round-trip (no
                // dedicated "broadcast last lane" SSE2 instruction; that's
                // `pshufb`, SSSE3+, one step above this baseline).
                let mut last = [0u8; 16];
                _mm_storeu_si128(last.as_mut_ptr().cast(), x);
                #[allow(clippy::cast_possible_wrap)] // bit pattern only: paddb doesn't care about signedness.
                let carry_byte = last[15] as i8;
                carry = _mm_set1_epi8(carry_byte);
            }
        }

        let tail_start = chunks * LANES;
        // The scalar tail continues the running sum from wherever the
        // vectorized part left off: that's just the last byte the SIMD
        // loop wrote (its own inclusive prefix sum already folds in every
        // earlier chunk's carry), or 0 if there were no full chunks at
        // all.
        let carry_in = tail_start.checked_sub(1).map_or(0, |i| data[i]);
        Self::scan_scalar_from(&mut data[tail_start..], carry_in);
    }

    /// Plain scalar running sum, starting from `carry_in` instead of 0 --
    /// used both as the entire implementation on non-`x86_64` targets and
    /// as the tail continuation after the SSE2 chunks on `x86_64`.
    fn scan_scalar_from(data: &mut [u8], carry_in: u8) {
        let mut running = carry_in;
        for byte in data {
            running = running.wrapping_add(*byte);
            *byte = running;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn scan_scalar(data: &mut [u8]) {
        Self::scan_scalar_from(data, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::{WrappingByteAdder, WrappingPrefixSum};

    #[test]
    fn matches_scalar_wrapping_add_elementwise() {
        let mut dst: Vec<u8> = (0..250u16).map(|n| (n % 256) as u8).collect();
        let src: Vec<u8> = (0..250u16).map(|n| ((n * 7 + 3) % 256) as u8).collect();
        let expected: Vec<u8> = dst
            .iter()
            .zip(&src)
            .map(|(&d, &s)| d.wrapping_add(s))
            .collect();
        WrappingByteAdder::add_wrapping(&mut dst, &src);
        assert_eq!(dst, expected);
    }

    #[test]
    fn handles_lengths_not_a_multiple_of_simd_width() {
        for len in 0..40 {
            let mut dst: Vec<u8> = (0..len as u16).map(|n| (n * 3) as u8).collect();
            let src: Vec<u8> = (0..len as u16).map(|n| (n * 11) as u8).collect();
            let expected: Vec<u8> = dst
                .iter()
                .zip(&src)
                .map(|(&d, &s)| d.wrapping_add(s))
                .collect();
            WrappingByteAdder::add_wrapping(&mut dst, &src);
            assert_eq!(dst, expected, "len={len}");
        }
    }

    #[test]
    fn wraps_on_overflow() {
        let mut dst = vec![250u8, 255, 200];
        let src = vec![10u8, 1, 100];
        WrappingByteAdder::add_wrapping(&mut dst, &src);
        assert_eq!(dst, vec![4u8, 0, 44]); // 260%256=4, 256%256=0, 300%256=44
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn panics_on_length_mismatch() {
        let mut dst = vec![1u8, 2, 3];
        let src = vec![1u8, 2];
        WrappingByteAdder::add_wrapping(&mut dst, &src);
    }

    /// Reference scalar prefix sum, mirroring `Sub`'s own reconstruction
    /// formula at `bpp == 1` -- used to check `WrappingPrefixSum::scan`
    /// against, independent of its own internal SIMD/scalar split.
    fn scalar_prefix_sum(data: &[u8]) -> Vec<u8> {
        let mut running = 0u8;
        data.iter()
            .map(|&b| {
                running = running.wrapping_add(b);
                running
            })
            .collect()
    }

    #[test]
    fn matches_scalar_prefix_sum_for_various_lengths() {
        // Cover both sides of every SSE2 chunk boundary (0, 15, 16, 17,
        // ..., a couple of full chunks, and past several chunks) so both
        // the vectorized loop and its scalar tail continuation get
        // exercised.
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 100, 257] {
            let data: Vec<u8> = (0..len as u32).map(|n| ((n * 37 + 5) % 256) as u8).collect();
            let expected = scalar_prefix_sum(&data);
            let mut actual = data.clone();
            WrappingPrefixSum::scan(&mut actual);
            assert_eq!(actual, expected, "len={len}");
        }
    }

    #[test]
    fn prefix_sum_wraps_on_overflow() {
        let mut data = vec![200u8, 100, 50, 10];
        // running: 200, 300%256=44, 44+50=94, 94+10=104
        WrappingPrefixSum::scan(&mut data);
        assert_eq!(data, vec![200u8, 44, 94, 104]);
    }

    #[test]
    fn prefix_sum_of_empty_slice_is_a_no_op() {
        let mut data: Vec<u8> = vec![];
        WrappingPrefixSum::scan(&mut data);
        assert_eq!(data, Vec::<u8>::new());
    }
}
