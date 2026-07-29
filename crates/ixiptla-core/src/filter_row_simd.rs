//! SIMD application of the PNG scanline filters, sixteen byte positions
//! at once.
//!
//! # Why filtering vectorizes as cleanly as scoring
//!
//! [`crate::filter_score_simd`] already established that filter *scoring*
//! is embarrassingly parallel per byte: every candidate reads only the
//! current byte `x`, its left neighbour `a`, the byte above `b`, and the
//! byte above-left `c` -- all original, already-known bytes, unlike
//! defiltering's serial reconstruction chain. Filter *application* reads
//! exactly the same four inputs and just skips the "sum the magnitude"
//! step in favour of "store the residual" -- so the exact same chunked
//! structure applies, and the two hot loops ([`crate::Filter::
//! filter_row_into`]'s scalar `for index in split..row.len()` loops)
//! become this module's vector kernels.
//!
//! # Why this doesn't share `filter_score_simd`'s function bodies
//!
//! Scoring accumulates five running sums and never writes scanline bytes;
//! application writes exactly one filter's residuals and accumulates
//! nothing. Different outputs, so different functions -- but the
//! non-trivial part, the Average/Paeth predictor math itself, is shared
//! via [`crate::predictor_simd::PredictorSimd`] so both consumers can
//! never disagree on what the predictor actually computes.
//!
//! Only `x86_64` gets a real implementation, matching every other SIMD
//! module in this crate: SSE2 is part of the baseline ABI there, so no
//! runtime feature detection is needed, while other targets fall back to
//! [`crate::Filter`]'s plain scalar loop for the entire row.

/// Sixteen-lanes-at-once PNG filter *application* (as opposed to
/// [`crate::filter_score_simd::FilterScoreSimd`], which only scores).
#[derive(Debug)]
pub struct FilterRowSimd;

impl FilterRowSimd {
    /// Bytes filtered per vector iteration.
    pub const LANES: usize = 16;

    /// `Sub`: `Filt(x) = Orig(x) - Orig(a)`, `a` = the `bpp`-back left
    /// neighbour -- a plain byte subtract, wrapping like `u8::wrapping_sub`.
    ///
    /// Appends filtered bytes for `row[start..]` to `out` in whole
    /// `LANES`-byte chunks, returning the index scoring stopped at. On
    /// non-`x86_64` this appends nothing and returns `start`.
    ///
    /// `start` must be at least `bpp` so the `a` read stays in bounds;
    /// the caller handles the zero-neighbour prefix scalar-ly, exactly as
    /// [`crate::filter_score_simd::FilterScoreSimd::score_chunks`] does.
    pub fn sub_chunks(row: &[u8], bpp: usize, start: usize, out: &mut Vec<u8>) -> usize {
        debug_assert!(start >= bpp, "a would read out of bounds");
        #[cfg(target_arch = "x86_64")]
        {
            Self::sub_chunks_sse2(row, bpp, start, out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (row, bpp, out);
            start
        }
    }

    /// `Up`: `Filt(x) = Orig(x) - Orig(b)`, `b` = the byte directly above.
    ///
    /// Unlike `Sub`/`Average`/`Paeth`, `Up` has no left-neighbour
    /// dependency at all, so `start` may be `0`.
    pub fn up_chunks(row: &[u8], previous_row: &[u8], start: usize, out: &mut Vec<u8>) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            Self::up_chunks_sse2(row, previous_row, start, out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (row, previous_row, out);
            start
        }
    }

    /// `Average`: `Filt(x) = Orig(x) - floor((Orig(a) + Orig(b)) / 2)`.
    pub fn average_chunks(
        row: &[u8],
        previous_row: &[u8],
        bpp: usize,
        start: usize,
        out: &mut Vec<u8>,
    ) -> usize {
        debug_assert!(start >= bpp, "a would read out of bounds");
        #[cfg(target_arch = "x86_64")]
        {
            Self::average_chunks_sse2(row, previous_row, bpp, start, out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (row, previous_row, bpp, out);
            start
        }
    }

    /// `Paeth`: `Filt(x) = Orig(x) - PaethPredictor(Orig(a), Orig(b), Orig(c))`.
    pub fn paeth_chunks(
        row: &[u8],
        previous_row: &[u8],
        bpp: usize,
        start: usize,
        out: &mut Vec<u8>,
    ) -> usize {
        debug_assert!(start >= bpp, "a and c would read out of bounds");
        #[cfg(target_arch = "x86_64")]
        {
            Self::paeth_chunks_sse2(row, previous_row, bpp, start, out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (row, previous_row, bpp, out);
            start
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn sub_chunks_sse2(row: &[u8], bpp: usize, start: usize, out: &mut Vec<u8>) -> usize {
        use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128, _mm_sub_epi8};

        let len = row.len();
        let mut index = start;
        let mut scratch = [0u8; Self::LANES];
        while index + Self::LANES <= len {
            // SAFETY: `index + LANES <= len` keeps the `x` load in bounds;
            // `index >= bpp` (the caller's precondition) keeps `index -
            // bpp` non-negative, and it stays `<= len - LANES` too since
            // `index` only grows, so the `a` load is in bounds as well.
            // Every load is the unaligned form, so `row`'s arbitrary
            // alignment never matters.
            unsafe {
                let x = _mm_loadu_si128(row.as_ptr().add(index).cast());
                let a = _mm_loadu_si128(row.as_ptr().add(index - bpp).cast());
                _mm_storeu_si128(scratch.as_mut_ptr().cast(), _mm_sub_epi8(x, a));
            }
            out.extend_from_slice(&scratch);
            index += Self::LANES;
        }
        index
    }

    #[cfg(target_arch = "x86_64")]
    fn up_chunks_sse2(row: &[u8], previous_row: &[u8], start: usize, out: &mut Vec<u8>) -> usize {
        use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128, _mm_sub_epi8};

        let len = row.len();
        let mut index = start;
        let mut scratch = [0u8; Self::LANES];
        while index + Self::LANES <= len {
            // SAFETY: `index + LANES <= len` keeps both loads in bounds
            // (`row` and `previous_row` are equal length, asserted by the
            // caller); unaligned loads tolerate any alignment.
            unsafe {
                let x = _mm_loadu_si128(row.as_ptr().add(index).cast());
                let b = _mm_loadu_si128(previous_row.as_ptr().add(index).cast());
                _mm_storeu_si128(scratch.as_mut_ptr().cast(), _mm_sub_epi8(x, b));
            }
            out.extend_from_slice(&scratch);
            index += Self::LANES;
        }
        index
    }

    #[cfg(target_arch = "x86_64")]
    fn average_chunks_sse2(
        row: &[u8],
        previous_row: &[u8],
        bpp: usize,
        start: usize,
        out: &mut Vec<u8>,
    ) -> usize {
        use crate::predictor_simd::PredictorSimd;
        use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128, _mm_sub_epi8};

        let len = row.len();
        let mut index = start;
        let mut scratch = [0u8; Self::LANES];
        while index + Self::LANES <= len {
            // SAFETY: same bounds argument as `sub_chunks_sse2` for the
            // `a` load; `b` only ever indexes `index`, in bounds by the
            // loop condition alone.
            unsafe {
                let x = _mm_loadu_si128(row.as_ptr().add(index).cast());
                let a = _mm_loadu_si128(row.as_ptr().add(index - bpp).cast());
                let b = _mm_loadu_si128(previous_row.as_ptr().add(index).cast());
                let mean = PredictorSimd::average(a, b);
                _mm_storeu_si128(scratch.as_mut_ptr().cast(), _mm_sub_epi8(x, mean));
            }
            out.extend_from_slice(&scratch);
            index += Self::LANES;
        }
        index
    }

    #[cfg(target_arch = "x86_64")]
    fn paeth_chunks_sse2(
        row: &[u8],
        previous_row: &[u8],
        bpp: usize,
        start: usize,
        out: &mut Vec<u8>,
    ) -> usize {
        use crate::predictor_simd::PredictorSimd;
        use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128, _mm_sub_epi8};

        let len = row.len();
        let mut index = start;
        let mut scratch = [0u8; Self::LANES];
        while index + Self::LANES <= len {
            // SAFETY: same bounds argument as `sub_chunks_sse2`, for both
            // the `a` and `c` (`previous_row[index - bpp]`) loads.
            unsafe {
                let x = _mm_loadu_si128(row.as_ptr().add(index).cast());
                let a = _mm_loadu_si128(row.as_ptr().add(index - bpp).cast());
                let b = _mm_loadu_si128(previous_row.as_ptr().add(index).cast());
                let c = _mm_loadu_si128(previous_row.as_ptr().add(index - bpp).cast());
                let predictor = PredictorSimd::paeth(a, b, c);
                _mm_storeu_si128(scratch.as_mut_ptr().cast(), _mm_sub_epi8(x, predictor));
            }
            out.extend_from_slice(&scratch);
            index += Self::LANES;
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::FilterRowSimd;

    /// Deterministic pseudo-random bytes, varied by `step`/`offset`.
    fn bytes(len: usize, step: usize, offset: usize) -> Vec<u8> {
        (0..len)
            .map(|i| u8::try_from((i * step + offset) % 256).expect("masked into u8 range"))
            .collect()
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

    #[test]
    fn sub_chunks_matches_scalar_reference_across_lengths_and_bpps() {
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            for len in [16usize, 17, 31, 32, 33, 64, 96] {
                let row = bytes(len, 37, 11);
                let mut out = Vec::new();
                let end = FilterRowSimd::sub_chunks(&row, bpp, bpp, &mut out);
                let expected: Vec<u8> = (bpp..end)
                    .map(|i| row[i].wrapping_sub(row[i - bpp]))
                    .collect();
                assert_eq!(out, expected, "bpp {bpp} len {len}");
            }
        }
    }

    #[test]
    fn up_chunks_matches_scalar_reference() {
        let row = bytes(64, 37, 11);
        let previous = bytes(64, 91, 5);
        let mut out = Vec::new();
        let end = FilterRowSimd::up_chunks(&row, &previous, 0, &mut out);
        let expected: Vec<u8> = (0..end).map(|i| row[i].wrapping_sub(previous[i])).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn average_chunks_matches_scalar_reference_across_bpps() {
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            let row = bytes(64, 37, 11);
            let previous = bytes(64, 91, 5);
            let mut out = Vec::new();
            let end = FilterRowSimd::average_chunks(&row, &previous, bpp, bpp, &mut out);
            let expected: Vec<u8> = (bpp..end)
                .map(|i| row[i].wrapping_sub(u8::midpoint(row[i - bpp], previous[i])))
                .collect();
            assert_eq!(out, expected, "bpp {bpp}");
        }
    }

    #[test]
    fn paeth_chunks_matches_scalar_reference_across_bpps() {
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            let row = bytes(64, 37, 11);
            let previous = bytes(64, 91, 5);
            let mut out = Vec::new();
            let end = FilterRowSimd::paeth_chunks(&row, &previous, bpp, bpp, &mut out);
            let expected: Vec<u8> = (bpp..end)
                .map(|i| {
                    let a = row[i - bpp];
                    let b = previous[i];
                    let c = previous[i - bpp];
                    row[i].wrapping_sub(scalar_paeth(a, b, c))
                })
                .collect();
            assert_eq!(out, expected, "bpp {bpp}");
        }
    }

    #[test]
    fn scores_nothing_when_no_whole_chunk_fits() {
        let row = bytes(20, 3, 1);
        let mut out = Vec::new();
        let end = FilterRowSimd::sub_chunks(&row, 8, 8, &mut out);
        assert!(out.is_empty());
        assert_eq!(end, 8);
    }

    #[test]
    fn handles_extreme_values_without_overflow_panics() {
        let row = vec![0xFFu8; 32];
        let previous = vec![0x00u8; 32];
        let mut sub_out = Vec::new();
        FilterRowSimd::sub_chunks(&row, 4, 4, &mut sub_out);
        assert!(sub_out.iter().all(|&b| b == 0));

        let mut paeth_out = Vec::new();
        FilterRowSimd::paeth_chunks(&row, &previous, 4, 4, &mut paeth_out);
        // a = 0xFF (row), b = c = 0x00 (previous): PaethPredictor picks a
        // (p = a, pa = 0 always wins), so residual is 0xFF - 0xFF = 0.
        assert!(paeth_out.iter().all(|&b| b == 0), "{paeth_out:?}");
    }
}
