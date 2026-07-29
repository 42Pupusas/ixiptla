//! Shared SSE2 kernels for PNG's Average and Paeth predictors (spec §9.3,
//! §9.4), sixteen byte positions at once.
//!
//! # Why this is its own module
//!
//! [`crate::filter_score_simd`] (scoring, for choosing which filter to
//! apply) and [`crate::filter_row_simd`] (application, for actually
//! producing the filtered bytes) both need the *exact same* predictor
//! value at every byte position -- scoring subtracts it and accumulates
//! the magnitude, application subtracts it and stores the residual.
//!
//! A four-line scalar formula is cheap enough to duplicate and eyeball
//! for drift (see [`crate::Filter`] and [`crate::FilterScorer`]'s own
//! independently-tested copies of the scalar Paeth predictor). A ~30-line
//! `unsafe` SIMD kernel is not: two independent copies of branchless
//! mask-and-blend logic are exactly the kind of thing that can silently
//! diverge under an unrelated edit to one of them. So the vector kernels
//! live here once, and both consumers call in rather than each keeping
//! their own copy.

use std::arch::x86_64::__m128i;

/// Sixteen-lanes-at-once PNG predictor arithmetic.
#[derive(Debug)]
pub struct PredictorSimd;

impl PredictorSimd {
    /// `floor((a + b) / 2)`, one byte per lane, sixteen lanes.
    ///
    /// The spec wants the floored mean, but `_mm_avg_epu8` computes the
    /// rounded-up `(a + b + 1) / 2`. They differ by exactly one when `a`
    /// and `b` disagree in their low bit, so subtracting `(a ^ b) & 1`
    /// converts the rounded-up mean into the floored one -- no widening
    /// to 16 bits needed.
    ///
    /// # Safety
    ///
    /// Requires SSE2, guaranteed on every `x86_64` target (part of the
    /// baseline ABI), so any `x86_64` caller may call this unconditionally.
    #[inline]
    pub unsafe fn average(a: __m128i, b: __m128i) -> __m128i {
        use std::arch::x86_64::{
            _mm_and_si128, _mm_avg_epu8, _mm_set1_epi8, _mm_sub_epi8, _mm_xor_si128,
        };
        // SAFETY: pure register operations, no memory access; the caller
        // guarantees SSE2 per this function's own safety contract.
        unsafe {
            let low_bit = _mm_set1_epi8(1);
            _mm_sub_epi8(
                _mm_avg_epu8(a, b),
                _mm_and_si128(_mm_xor_si128(a, b), low_bit),
            )
        }
    }

    /// The Paeth predictor (spec §9.4), one byte per lane, sixteen lanes.
    ///
    /// Paeth needs 16-bit intermediates (the spec's `p = a + b - c` can
    /// briefly fall outside a `u8` before the nearest-of-three comparison
    /// picks a winner), so each 16-lane `u8` input is split into two
    /// 8-lane `u16` halves, predicted independently, and repacked.
    /// `packus` saturates on overflow, but every result is exactly one of
    /// `a`/`b`/`c` and so already fits in `0..=255`.
    ///
    /// # Safety
    ///
    /// Requires SSE2, guaranteed on every `x86_64` target (part of the
    /// baseline ABI), so any `x86_64` caller may call this unconditionally.
    #[inline]
    pub unsafe fn paeth(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
        use std::arch::x86_64::{
            _mm_packus_epi16, _mm_setzero_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
        };
        // SAFETY: pure register operations, no memory access; the caller
        // guarantees SSE2 per this function's own safety contract.
        unsafe {
            let zero = _mm_setzero_si128();
            _mm_packus_epi16(
                Self::paeth_half(
                    _mm_unpacklo_epi8(a, zero),
                    _mm_unpacklo_epi8(b, zero),
                    _mm_unpacklo_epi8(c, zero),
                ),
                Self::paeth_half(
                    _mm_unpackhi_epi8(a, zero),
                    _mm_unpackhi_epi8(b, zero),
                    _mm_unpackhi_epi8(c, zero),
                ),
            )
        }
    }

    /// The Paeth predictor (spec §9.4) for eight 16-bit lanes.
    ///
    /// Selection is branchless: build an all-ones mask per condition and
    /// blend, which is why the scalar `if/else if/else` chain becomes
    /// three masks that partition the lanes exactly the way the spec's
    /// priority order does (a, then b, then c).
    ///
    /// # Safety
    ///
    /// Requires SSE2, guaranteed on every `x86_64` target (part of the
    /// baseline ABI), so any `x86_64` caller may call this unconditionally.
    #[inline]
    unsafe fn paeth_half(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
        use std::arch::x86_64::{
            _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi16, _mm_cmpgt_epi16,
            _mm_max_epi16, _mm_or_si128, _mm_setzero_si128, _mm_sub_epi16,
        };
        // SAFETY: pure register operations, no memory access; the caller
        // guarantees SSE2 per this function's own safety contract.
        unsafe {
            let zero = _mm_setzero_si128();
            let abs = |v: __m128i| _mm_max_epi16(v, _mm_sub_epi16(zero, v));

            // With p = a + b - c, the spec's distances simplify: no `p`
            // is ever formed, so everything stays inside 16 bits.
            let da = _mm_sub_epi16(a, c);
            let db = _mm_sub_epi16(b, c);
            let pa = abs(db); // |p - a| = |b - c|
            let pb = abs(da); // |p - b| = |a - c|
            let pc = abs(_mm_add_epi16(da, db)); // |p - c|

            // Inputs are bytes, so distances are far below i16::MAX and
            // signed compares are safe. `x <= y` is `!(x > y)`.
            let ones = _mm_cmpeq_epi16(zero, zero);
            let a_loses_to_b = _mm_cmpgt_epi16(pa, pb);
            let a_loses_to_c = _mm_cmpgt_epi16(pa, pc);
            let b_loses_to_c = _mm_cmpgt_epi16(pb, pc);

            // a wins when pa <= pb && pa <= pc; otherwise b wins when
            // pb <= pc; otherwise c. The masks are disjoint by
            // construction, so a plain OR blends them.
            let choose_a = _mm_andnot_si128(_mm_or_si128(a_loses_to_b, a_loses_to_c), ones);
            let not_a = _mm_andnot_si128(choose_a, ones);
            let choose_b = _mm_andnot_si128(b_loses_to_c, not_a);
            let choose_c = _mm_and_si128(b_loses_to_c, not_a);

            _mm_or_si128(
                _mm_and_si128(a, choose_a),
                _mm_or_si128(_mm_and_si128(b, choose_b), _mm_and_si128(c, choose_c)),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PredictorSimd;
    use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128};

    fn load(bytes: &[u8; 16]) -> std::arch::x86_64::__m128i {
        // SAFETY: `bytes` is a full 16-byte local array; the unaligned
        // load has no alignment requirement.
        unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
    }

    fn store(v: std::arch::x86_64::__m128i) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: `out` is a full 16-byte local array.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), v) };
        out
    }

    fn scalar_average(a: u8, b: u8) -> u8 {
        u8::midpoint(a, b)
    }

    fn scalar_paeth(a: u8, b: u8, c: u8) -> u8 {
        let p = i16::from(a) + i16::from(b) - i16::from(c);
        let pa = (p - i16::from(a)).abs();
        let pb = (p - i16::from(b)).abs();
        let pc = (p - i16::from(c)).abs();
        if pa <= pb && pa <= pc {
            a
        } else if pb <= pc {
            b
        } else {
            c
        }
    }

    /// Sixteen deterministic pseudo-random bytes, varied by `seed`.
    fn lane_bytes(seed: u32) -> [u8; 16] {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            u8::try_from(state & 0xFF).expect("masked into u8 range")
        })
    }

    #[test]
    fn average_matches_the_scalar_floor_of_the_mean() {
        for seed in [1u32, 2, 3, 4, 5] {
            let a = lane_bytes(seed);
            let b = lane_bytes(seed.wrapping_mul(7));
            // SAFETY: SSE2 is baseline on x86_64, the arch this test runs on.
            let result = store(unsafe { PredictorSimd::average(load(&a), load(&b)) });
            for lane in 0..16 {
                assert_eq!(
                    result[lane],
                    scalar_average(a[lane], b[lane]),
                    "lane {lane} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn average_handles_extreme_values() {
        let a = [0xFFu8; 16];
        let b = [0x00u8; 16];
        // SAFETY: SSE2 is baseline on x86_64, the arch this test runs on.
        let result = store(unsafe { PredictorSimd::average(load(&a), load(&b)) });
        assert_eq!(result, [127u8; 16]);
    }

    #[test]
    fn paeth_matches_the_scalar_predictor() {
        for seed in [1u32, 2, 3, 4, 5, 6] {
            let a = lane_bytes(seed);
            let b = lane_bytes(seed.wrapping_mul(11));
            let c = lane_bytes(seed.wrapping_mul(19));
            // SAFETY: SSE2 is baseline on x86_64, the arch this test runs on.
            let result = store(unsafe { PredictorSimd::paeth(load(&a), load(&b), load(&c)) });
            for lane in 0..16 {
                assert_eq!(
                    result[lane],
                    scalar_paeth(a[lane], b[lane], c[lane]),
                    "lane {lane} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn paeth_handles_extreme_and_tied_values() {
        let cases: [(u8, u8, u8); 5] = [
            (0, 0, 0),
            (0xFF, 0xFF, 0xFF),
            (0xFF, 0x00, 0x00),
            (10, 20, 10),
            (200, 5, 100),
        ];
        for (a_byte, b_byte, c_byte) in cases {
            let a = [a_byte; 16];
            let b = [b_byte; 16];
            let c = [c_byte; 16];
            // SAFETY: SSE2 is baseline on x86_64, the arch this test runs on.
            let result = store(unsafe { PredictorSimd::paeth(load(&a), load(&b), load(&c)) });
            assert_eq!(result, [scalar_paeth(a_byte, b_byte, c_byte); 16]);
        }
    }
}
