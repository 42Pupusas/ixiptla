//! Choosing *which* filter to apply to a scanline, separated from
//! actually applying one ([`crate::Filter`]).
//!
//! # Why this is its own module
//!
//! Selection and application are genuinely different jobs. Application is
//! fixed by the spec — filter type `n` means formula `n`, no freedom.
//! Selection is a heuristic that trades encode time against output size,
//! and it is where an encoder's quality lives. Keeping it separate means
//! the heuristic can be replaced (or SIMD-accelerated) without touching
//! the spec-mandated arithmetic, and `Filter` stays a pure, obviously
//! correct transcription of §9.
//!
//! # Why one pass instead of five
//!
//! The obvious implementation of "pick the cheapest filter" is to filter
//! the row five times into a scratch buffer, score each result, keep the
//! best, then filter once more with the winner — six passes over the row,
//! five of which write bytes nobody ever reads.
//!
//! But all five filters are functions of the *same four inputs* at each
//! byte position: the current byte `x`, its left neighbour `a`, the byte
//! above `b`, and the byte above-left `c`. So a single pass can load those
//! four once and update five running score accumulators from them. That
//! turns 5 filter passes + 5 score passes into 1 combined pass, and skips
//! materializing four losing candidates entirely — the winner is then
//! written exactly once by [`crate::Filter`].
//!
//! The scores produced are bit-for-bit identical to scoring the
//! materialized rows, because the per-byte residual arithmetic is the
//! same; this is a pure strength reduction, not an approximation.

use crate::FilterType;
use crate::filter_score_simd::{FilterScoreSimd, FilterScores};

/// Tie-breaking and winner selection over accumulated candidate scores.
///
/// The totals themselves live in [`FilterScores`] so the scalar and SIMD
/// paths can both produce them; this trait adds the choosing.
trait BestFilter {
    /// The cheapest filter, ties broken in spec order.
    fn best(self) -> FilterType;
}

impl BestFilter for FilterScores {
    /// Ties break in spec order (None, Sub, Up, Average, Paeth).
    ///
    /// The order matters for reproducibility, not correctness: any of a
    /// tied set decodes identically, but a fixed order keeps encoder
    /// output deterministic across runs and platforms. It matches the
    /// order the five-pass implementation used (first strict improvement
    /// wins), so neither this nor the SIMD path can change which filter
    /// is chosen.
    fn best(self) -> FilterType {
        let mut best = FilterType::None;
        let mut best_score = self.none;
        for (score, filter) in [
            (self.sub, FilterType::Sub),
            (self.up, FilterType::Up),
            (self.average, FilterType::Average),
            (self.paeth, FilterType::Paeth),
        ] {
            if score < best_score {
                best_score = score;
                best = filter;
            }
        }
        best
    }
}

/// Scores every candidate filter for a scanline in a single pass and
/// reports the cheapest under libpng's minimum-sum-of-absolute-differences
/// heuristic.
#[derive(Debug)]
pub struct FilterScorer {
    /// Bytes per complete pixel — the left-neighbour stride, matching
    /// [`crate::Filter`]'s `bpp`.
    bpp: usize,
}

impl FilterScorer {
    /// Build a scorer for scanlines whose pixels are `bpp` bytes wide.
    #[must_use]
    pub const fn new(bpp: usize) -> Self {
        Self { bpp }
    }

    /// The filter that scores cheapest for `row` given the raw scanline
    /// `previous_row` above it.
    ///
    /// # Panics
    ///
    /// If `previous_row` isn't the same length as `row`.
    #[must_use]
    pub fn best_filter(&self, row: &[u8], previous_row: &[u8]) -> FilterType {
        assert_eq!(
            row.len(),
            previous_row.len(),
            "current and previous scanlines must have equal length"
        );

        // Row 0 of an image/pass has an all-zero row above it, so Up
        // degenerates to None and Average/Paeth to weaker forms of Sub.
        // Scoring them is pure waste; skipping them is an optimization
        // only, never a correctness requirement.
        if previous_row.iter().all(|&b| b == 0) {
            return self.best_without_previous_row(row);
        }

        let mut scores = FilterScores::default();

        // The first `bpp` bytes have no left neighbour, so `a` and `c` are
        // zero there (the convention `Filter` uses). That is a different
        // recurrence from the rest of the row, so it stays scalar rather
        // than complicating the kernel with a masked first iteration.
        let head = self.bpp.min(row.len());
        for index in 0..head {
            scores.add(self.score_at(row, previous_row, index, 0, 0));
        }

        // Whole vectors, then whatever ragged tail is left over.
        let vectored_end = if row.len() > head {
            let (chunked, next) = FilterScoreSimd::score_chunks(row, previous_row, self.bpp, head);
            scores.add(chunked);
            next
        } else {
            head
        };
        for index in vectored_end..row.len() {
            let a = row[index - self.bpp];
            let c = previous_row[index - self.bpp];
            scores.add(self.score_at(row, previous_row, index, a, c));
        }

        scores.best()
    }

    /// One byte position's contribution to all five candidates, with `a`
    /// and `c` supplied by the caller so the zero-neighbour prefix and the
    /// general case share one body.
    fn score_at(
        &self,
        row: &[u8],
        previous_row: &[u8],
        index: usize,
        a: u8,
        c: u8,
    ) -> FilterScores {
        let _ = self;
        let x = row[index];
        let b = previous_row[index];
        FilterScores {
            none: Self::magnitude(x),
            sub: Self::magnitude(x.wrapping_sub(a)),
            up: Self::magnitude(x.wrapping_sub(b)),
            average: Self::magnitude(x.wrapping_sub(u8::midpoint(a, b))),
            paeth: Self::magnitude(x.wrapping_sub(Self::paeth_predictor(a, b, c))),
        }
    }

    /// Selection restricted to the two candidates that differ when the row
    /// above is all zeroes, scored in one pass over the row.
    fn best_without_previous_row(&self, row: &[u8]) -> FilterType {
        let mut none = 0u64;
        let mut sub = 0u64;
        for index in 0..row.len() {
            let x = row[index];
            let a = if index >= self.bpp {
                row[index - self.bpp]
            } else {
                0
            };
            none += Self::magnitude(x);
            sub += Self::magnitude(x.wrapping_sub(a));
        }
        if sub < none {
            FilterType::Sub
        } else {
            FilterType::None
        }
    }

    /// A filtered byte's contribution to the score.
    ///
    /// Filtered bytes are wrapped differences, so `0xFF` means -1, not
    /// 255. Reading it back as `i8` and taking the magnitude is what makes
    /// a residual of -1 as cheap as +1 — small differences in either
    /// direction are exactly what compresses well.
    const fn magnitude(filtered: u8) -> u64 {
        filtered.cast_signed().unsigned_abs() as u64
    }

    /// The Paeth predictor (spec §9.4).
    ///
    /// Duplicated from [`crate::Filter`] rather than shared because it is
    /// four lines of spec formula and the alternative is a cross-module
    /// call in the hottest loop of the encoder; both copies are pinned by
    /// tests asserting they agree.
    const fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
        let p = a as i16 + b as i16 - c as i16;
        let pa = (p - a as i16).abs();
        let pb = (p - b as i16).abs();
        let pc = (p - c as i16).abs();
        if pa <= pb && pa <= pc {
            a
        } else if pb <= pc {
            b
        } else {
            c
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BestFilter, FilterScorer};
    use crate::filter_score_simd::FilterScores;
    use crate::{ColorType, Filter, FilterType, Ihdr, InterlaceMethod};

    /// Reference implementation: the five-pass approach this module
    /// replaces. Kept in the tests as the oracle the fast path must match,
    /// so the optimization is pinned to observable behaviour rather than
    /// to my belief about what it does.
    struct FivePassOracle {
        filter: Filter,
    }

    impl FivePassOracle {
        fn new(ihdr: &Ihdr) -> Self {
            Self {
                filter: Filter::new(ihdr),
            }
        }

        /// Materializes every candidate, scores it, and keeps the first
        /// strict improvement — exactly what `filter_row_adaptive_into`
        /// did before `FilterScorer` existed.
        fn best_filter(&self, row: &[u8], previous_row: &[u8]) -> FilterType {
            let previous_is_zero = previous_row.iter().all(|&b| b == 0);
            let candidates: &[FilterType] = if previous_is_zero {
                &[FilterType::None, FilterType::Sub]
            } else {
                &FilterType::ALL
            };

            let mut best = FilterType::None;
            let mut best_score = u64::MAX;
            for &candidate in candidates {
                let mut scratch = Vec::new();
                self.filter
                    .filter_row_into(candidate, row, previous_row, &mut scratch);
                let score: u64 = scratch[1..]
                    .iter()
                    .map(|&byte| u64::from(byte.cast_signed().unsigned_abs()))
                    .sum();
                if score < best_score {
                    best_score = score;
                    best = candidate;
                }
            }
            best
        }
    }

    /// Deterministic pseudo-random scanline, varied by `step`/`offset` so
    /// a row and the row above it don't coincide by accident.
    fn row(len: usize, step: usize, offset: usize) -> Vec<u8> {
        (0..len)
            .map(|i| u8::try_from((i * step + offset) % 256).expect("masked into u8 range"))
            .collect()
    }

    fn header(width: u32, color_type: ColorType, bit_depth: u8) -> Ihdr {
        Ihdr {
            width,
            height: 1,
            bit_depth,
            color_type,
            interlace: InterlaceMethod::None,
        }
    }

    #[test]
    fn agrees_with_the_five_pass_oracle_across_geometries_and_content() {
        // The property that makes this a safe optimization: for every
        // shape and content we can think of, one-pass scoring picks the
        // same filter the straightforward five-pass version would.
        let headers = [
            header(16, ColorType::Grayscale, 8),
            header(16, ColorType::Truecolor, 8),
            header(16, ColorType::TruecolorAlpha, 8),
            header(16, ColorType::GrayscaleAlpha, 8),
            header(8, ColorType::TruecolorAlpha, 16),
            header(1, ColorType::TruecolorAlpha, 16),
        ];
        for ihdr in headers {
            let stride = ihdr.stride();
            let scorer = FilterScorer::new(ihdr.bytes_per_pixel());
            let oracle = FivePassOracle::new(&ihdr);

            let current = row(stride, 37, 11);
            let candidates = [
                vec![0u8; stride],           // row 0: the restricted path
                current.clone(),             // identical rows: Up wins
                row(stride, 91, 5),          // unrelated content
                row(stride, 37, 11 + 1),     // near-identical: small deltas
                vec![255u8; stride],         // saturated row above
                row(stride, 1, 0),           // gentle ramp
            ];
            for previous in candidates {
                assert_eq!(
                    scorer.best_filter(&current, &previous),
                    oracle.best_filter(&current, &previous),
                    "disagreed for {:?} stride {stride}",
                    ihdr.color_type
                );
            }
        }
    }

    #[test]
    fn identical_rows_select_up() {
        // Every residual becomes zero, which no other filter can beat
        // here -- the clearest case where the heuristic must be right.
        let ihdr = header(32, ColorType::Truecolor, 8);
        let current = row(ihdr.stride(), 7, 13);
        assert_eq!(
            FilterScorer::new(ihdr.bytes_per_pixel()).best_filter(&current, &current),
            FilterType::Up
        );
    }

    #[test]
    fn an_all_zero_previous_row_never_selects_up_average_or_paeth() {
        // With no row above, Up is equivalent to None; selecting it would
        // be a wasted opportunity to pick the genuinely cheaper candidate.
        let ihdr = header(16, ColorType::Grayscale, 8);
        let current = row(ihdr.stride(), 5, 0);
        let chosen = FilterScorer::new(ihdr.bytes_per_pixel())
            .best_filter(&current, &vec![0u8; ihdr.stride()]);
        assert!(matches!(chosen, FilterType::None | FilterType::Sub));
    }

    #[test]
    fn paeth_predictor_matches_the_filters_copy() {
        // The two copies of the predictor must never drift apart; this is
        // what licenses duplicating it for the hot loop's sake.
        for a in [0u8, 1, 5, 100, 200, 255] {
            for b in [0u8, 2, 5, 99, 201, 255] {
                for c in [0u8, 3, 5, 100, 202, 255] {
                    assert_eq!(
                        FilterScorer::paeth_predictor(a, b, c),
                        Filter::paeth_predictor_for_test(a, b, c),
                        "diverged at a={a} b={b} c={c}"
                    );
                }
            }
        }
    }

    #[test]
    fn magnitude_reads_bytes_as_signed_residuals() {
        assert_eq!(FilterScorer::magnitude(0xFF), 1);
        assert_eq!(FilterScorer::magnitude(0x01), 1);
        assert_eq!(FilterScorer::magnitude(0x80), 128);
        assert_eq!(FilterScorer::magnitude(0x00), 0);
    }

    #[test]
    fn ties_resolve_in_spec_order() {
        // All-equal scores must yield None, and a tie between two
        // non-None candidates must go to the earlier one, so encoder
        // output stays byte-for-byte reproducible.
        let all_equal = FilterScores {
            none: 10,
            sub: 10,
            up: 10,
            average: 10,
            paeth: 10,
        };
        assert_eq!(all_equal.best(), FilterType::None);

        let up_and_average_tie = FilterScores {
            none: 99,
            sub: 99,
            up: 10,
            average: 10,
            paeth: 99,
        };
        assert_eq!(up_and_average_tie.best(), FilterType::Up);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn mismatched_row_lengths_panic() {
        let _ = FilterScorer::new(1).best_filter(&[1, 2, 3, 4], &[1, 2, 3]);
    }
}
