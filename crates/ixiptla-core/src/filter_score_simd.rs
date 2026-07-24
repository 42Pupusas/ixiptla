//! SIMD scoring of all five PNG filter candidates over one scanline.
//!
//! # Why filtering vectorizes when defiltering doesn't
//!
//! [`crate::simd`] can only accelerate `Up` on the decode side, because
//! reconstruction is a serial dependency chain: `Recon(a)` is the byte the
//! previous iteration just wrote, so `Sub`/`Average`/`Paeth` cannot be
//! parallelized across a row at all.
//!
//! Encoding has no such chain. Every filter reads only the *original*
//! bytes — the current byte `x`, its left neighbour `a`, the byte above
//! `b`, the byte above-left `c` — all known up front. Sixteen byte
//! positions can therefore be scored at once, and all five candidates
//! derived from the same four loaded vectors. This is the one place the
//! encoder gets a speedup the decoder structurally cannot.
//!
//! # The three tricks that make this cheap
//!
//! Scoring needs `sum of |residual as i8|`, and each has an exact SSE2
//! form:
//!
//! * **Magnitude**: `|v as i8|` is `min_epu8(v, 0 - v)`. One of `v` and its
//!   two's-complement negation is the magnitude and the other is at least
//!   as large, so the unsigned minimum is the magnitude — including the
//!   `0x80` case, where both sides are `0x80` and the answer is 128,
//!   matching `(-128i8).unsigned_abs()`.
//! * **Summation**: adding 16 magnitudes is one `psadbw` against zero,
//!   giving two 64-bit partials that accumulate without overflow.
//! * **Average**: the spec wants `floor((a + b) / 2)` but `_mm_avg_epu8`
//!   computes the rounded-up `(a + b + 1) / 2`. They differ by exactly one
//!   when `a` and `b` disagree in their low bit, so subtracting `(a ^ b) &
//!   1` converts one into the other — no widening to 16 bits needed.
//!
//! Only Paeth needs 16-bit intermediates, and even there the spec's
//! `p = a + b - c` never has to be formed: the three distances reduce to
//! `pa = |b - c|`, `pb = |a - c|`, `pc = |(a - c) + (b - c)|`.
//!
//! Only `x86_64` gets a real implementation, matching [`crate::simd`]'s
//! reasoning: SSE2 is part of the `x86_64` baseline ABI, so it needs no
//! runtime feature detection, while other targets fall back to the scalar
//! loop. The split lives inside this module so [`crate::FilterScorer`]
//! never sees an architecture `#[cfg]`.

/// Sum-of-absolute-residual totals for the five candidate filters over
/// some span of a scanline.
///
/// Crate-visible so the scalar and SIMD paths can exchange partial results
/// and add them: the SIMD kernel handles whole 16-byte chunks and hands
/// the ragged head and tail back to the caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterScores {
    pub none: u64,
    pub sub: u64,
    pub up: u64,
    pub average: u64,
    pub paeth: u64,
}

impl FilterScores {
    /// Adds another span's totals into this one.
    pub const fn add(&mut self, other: Self) {
        self.none += other.none;
        self.sub += other.sub;
        self.up += other.up;
        self.average += other.average;
        self.paeth += other.paeth;
    }
}

/// SSE2 scoring of all five filters across the whole-vector portion of a
/// scanline.
#[derive(Debug)]
pub struct FilterScoreSimd;

impl FilterScoreSimd {
    /// Bytes scored per vector iteration.
    pub const LANES: usize = 16;

    /// Scores `row[start..]` in whole `LANES`-byte chunks, returning the
    /// totals and the index where scoring stopped.
    ///
    /// `start` must be at least `bpp` so `a`/`c` (which read `bpp` bytes
    /// back) stay in bounds. The caller scores the first `bpp` bytes
    /// scalar-ly, where the spec's zero-neighbour convention applies and
    /// vectorizing would be wrong anyway.
    ///
    /// Returns `(scores, next_index)`. On non-`x86_64` targets it scores
    /// nothing and returns `start`, leaving the caller's scalar loop to do
    /// all the work.
    pub fn score_chunks(
        row: &[u8],
        previous_row: &[u8],
        bpp: usize,
        start: usize,
    ) -> (FilterScores, usize) {
        debug_assert!(start >= bpp, "a and c would read out of bounds");
        debug_assert_eq!(row.len(), previous_row.len());

        #[cfg(target_arch = "x86_64")]
        {
            Self::score_chunks_sse2(row, previous_row, bpp, start)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (row, previous_row, bpp);
            (FilterScores::default(), start)
        }
    }

    /// The SSE2 kernel: sixteen byte positions, five candidates, per pass.
    #[cfg(target_arch = "x86_64")]
    fn score_chunks_sse2(
        row: &[u8],
        previous_row: &[u8],
        bpp: usize,
        start: usize,
    ) -> (FilterScores, usize) {
        use std::arch::x86_64::{
            __m128i, _mm_add_epi64, _mm_and_si128, _mm_avg_epu8, _mm_loadu_si128, _mm_packus_epi16,
            _mm_set1_epi8, _mm_setzero_si128, _mm_storeu_si128, _mm_sub_epi8, _mm_unpackhi_epi8,
            _mm_unpacklo_epi8, _mm_xor_si128,
        };

        let len = row.len();
        if start + Self::LANES > len {
            return (FilterScores::default(), start);
        }

        // SAFETY: SSE2 is guaranteed on every x86_64 target (baseline ABI,
        // unlike AVX2), so no runtime detection is needed. Every load is
        // the *unaligned* form, so the slices' arbitrary alignment (they
        // are row-boundary subslices of a Vec<u8>) never matters. The loop
        // condition keeps `index + LANES <= len`, and `index >= bpp` holds
        // because `start >= bpp` and `index` only grows, so the
        // `index - bpp` reads are in bounds for both equal-length slices.
        unsafe {
            let zero = _mm_setzero_si128();
            let low_bit = _mm_set1_epi8(1);

            let mut acc_none = zero;
            let mut acc_sub = zero;
            let mut acc_up = zero;
            let mut acc_avg = zero;
            let mut acc_paeth = zero;

            let mut index = start;
            while index + Self::LANES <= len {
                let x = _mm_loadu_si128(row.as_ptr().add(index).cast());
                let a = _mm_loadu_si128(row.as_ptr().add(index - bpp).cast());
                let b = _mm_loadu_si128(previous_row.as_ptr().add(index).cast());
                let c = _mm_loadu_si128(previous_row.as_ptr().add(index - bpp).cast());

                // None, Sub, Up: a single byte subtract, wrapping exactly
                // like PNG's `u8::wrapping_sub`.
                acc_none = _mm_add_epi64(acc_none, Self::magnitude_sum(x, zero));
                acc_sub = _mm_add_epi64(acc_sub, Self::magnitude_sum(_mm_sub_epi8(x, a), zero));
                acc_up = _mm_add_epi64(acc_up, Self::magnitude_sum(_mm_sub_epi8(x, b), zero));

                // Average: round-up mean minus the disagreeing low bit.
                let mean = _mm_sub_epi8(
                    _mm_avg_epu8(a, b),
                    _mm_and_si128(_mm_xor_si128(a, b), low_bit),
                );
                acc_avg = _mm_add_epi64(acc_avg, Self::magnitude_sum(_mm_sub_epi8(x, mean), zero));

                // Paeth needs 16-bit lanes, so split into halves and
                // repack. `packus` saturates, but every result is one of
                // a/b/c and so already in 0..=255.
                let predictor = _mm_packus_epi16(
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
                );
                acc_paeth = _mm_add_epi64(
                    acc_paeth,
                    Self::magnitude_sum(_mm_sub_epi8(x, predictor), zero),
                );

                index += Self::LANES;
            }

            // `psadbw` leaves one partial sum in each 64-bit half.
            let horizontal = |v: __m128i| -> u64 {
                let mut lanes = [0u64; 2];
                _mm_storeu_si128(lanes.as_mut_ptr().cast(), v);
                lanes[0] + lanes[1]
            };

            let scores = FilterScores {
                none: horizontal(acc_none),
                sub: horizontal(acc_sub),
                up: horizontal(acc_up),
                average: horizontal(acc_avg),
                paeth: horizontal(acc_paeth),
            };
            (scores, index)
        }
    }

    /// Sum of `|v as i8|` over all 16 lanes, as two 64-bit partials.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn magnitude_sum(
        v: std::arch::x86_64::__m128i,
        zero: std::arch::x86_64::__m128i,
    ) -> std::arch::x86_64::__m128i {
        use std::arch::x86_64::{_mm_min_epu8, _mm_sad_epu8, _mm_sub_epi8};
        // SAFETY: pure register operations, no memory access; the caller
        // is already inside the SSE2-guaranteed block.
        unsafe { _mm_sad_epu8(_mm_min_epu8(v, _mm_sub_epi8(zero, v)), zero) }
    }

    /// The Paeth predictor (spec §9.4) for eight 16-bit lanes.
    ///
    /// Selection is branchless: build an all-ones mask per condition and
    /// blend, which is why the scalar `if/else if/else` chain becomes three
    /// masks that partition the lanes exactly the way the spec's priority
    /// order does (a, then b, then c).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn paeth_half(
        a: std::arch::x86_64::__m128i,
        b: std::arch::x86_64::__m128i,
        c: std::arch::x86_64::__m128i,
    ) -> std::arch::x86_64::__m128i {
        use std::arch::x86_64::{
            __m128i, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi16,
            _mm_cmpgt_epi16, _mm_max_epi16, _mm_or_si128, _mm_setzero_si128, _mm_sub_epi16,
        };
        // SAFETY: pure register operations, no memory access; the caller
        // is already inside the SSE2-guaranteed block.
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
    use super::{FilterScoreSimd, FilterScores};

    /// Scalar reference for the same span the SIMD kernel covers, written
    /// straight from the spec so the vector code is checked against an
    /// obviously-correct oracle rather than against itself.
    struct ScalarReference {
        bpp: usize,
    }

    impl ScalarReference {
        const fn new(bpp: usize) -> Self {
            Self { bpp }
        }

        fn paeth(a: u8, b: u8, c: u8) -> u8 {
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

        fn magnitude(v: u8) -> u64 {
            u64::from(v.cast_signed().unsigned_abs())
        }

        /// Scores exactly `start..end`, matching the SIMD kernel's span.
        fn score(&self, row: &[u8], previous_row: &[u8], start: usize, end: usize) -> FilterScores {
            let mut scores = FilterScores::default();
            for index in start..end {
                let x = row[index];
                let a = row[index - self.bpp];
                let b = previous_row[index];
                let c = previous_row[index - self.bpp];
                scores.none += Self::magnitude(x);
                scores.sub += Self::magnitude(x.wrapping_sub(a));
                scores.up += Self::magnitude(x.wrapping_sub(b));
                scores.average += Self::magnitude(x.wrapping_sub(u8::midpoint(a, b)));
                scores.paeth += Self::magnitude(x.wrapping_sub(Self::paeth(a, b, c)));
            }
            scores
        }
    }

    /// Deterministic pseudo-random bytes, varied by `step`/`offset`.
    fn bytes(len: usize, step: usize, offset: usize) -> Vec<u8> {
        (0..len)
            .map(|i| u8::try_from((i * step + offset) % 256).expect("masked into u8 range"))
            .collect()
    }

    #[test]
    fn matches_the_scalar_reference_across_lengths_and_bpps() {
        // The core property: for every pixel width and row length, the
        // vector kernel's totals equal the spec-derived scalar ones.
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            for len in [16usize, 17, 31, 32, 33, 64, 96, 129, 256] {
                if len <= bpp {
                    continue;
                }
                let row = bytes(len, 37, 11);
                let previous = bytes(len, 91, 5);
                let (simd, end) = FilterScoreSimd::score_chunks(&row, &previous, bpp, bpp);
                let expected = ScalarReference::new(bpp).score(&row, &previous, bpp, end);
                assert_eq!(simd, expected, "bpp {bpp}, len {len}");
            }
        }
    }

    #[test]
    fn matches_the_scalar_reference_on_saturating_and_extreme_content() {
        // Exercises the 0x80 magnitude edge case, the Average carry
        // correction, and Paeth's tie-breaking, none of which the gentle
        // ramps above reach reliably.
        let bpp = 4usize;
        let cases: [(Vec<u8>, Vec<u8>); 5] = [
            (vec![0x80; 64], vec![0x00; 64]),
            (vec![0x00; 64], vec![0x80; 64]),
            (vec![0xFF; 64], vec![0x01; 64]),
            (bytes(64, 1, 0), vec![0xFF; 64]),
            (vec![0x7F; 64], vec![0x81; 64]),
        ];
        for (row, previous) in cases {
            let (simd, end) = FilterScoreSimd::score_chunks(&row, &previous, bpp, bpp);
            let expected = ScalarReference::new(bpp).score(&row, &previous, bpp, end);
            assert_eq!(simd, expected);
        }
    }

    #[test]
    fn scores_nothing_when_no_whole_chunk_fits() {
        // A row shorter than one vector must be left entirely to the
        // caller's scalar loop rather than partially consumed.
        let row = bytes(20, 3, 1);
        let previous = bytes(20, 7, 2);
        let (scores, end) = FilterScoreSimd::score_chunks(&row, &previous, 8, 8);
        assert_eq!(scores, FilterScores::default());
        assert_eq!(end, 8);
    }

    #[test]
    fn add_accumulates_every_field() {
        let mut total = FilterScores {
            none: 1,
            sub: 2,
            up: 3,
            average: 4,
            paeth: 5,
        };
        total.add(FilterScores {
            none: 10,
            sub: 20,
            up: 30,
            average: 40,
            paeth: 50,
        });
        assert_eq!(
            total,
            FilterScores {
                none: 11,
                sub: 22,
                up: 33,
                average: 44,
                paeth: 55,
            }
        );
    }
}
