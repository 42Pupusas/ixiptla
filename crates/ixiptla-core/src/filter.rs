use crate::{FilterType, Ihdr};

/// Applies PNG's per-scanline byte filtering (spec §9) — the forward
/// direction of [`crate::Defilter`], plus the choice of *which* filter to
/// apply to each row.
///
/// # Why filtering is not simply "defiltering backwards"
///
/// Decoding is forced: the filter-type byte at the head of a row says
/// exactly which reconstruction to run, so a decoder never chooses. An
/// encoder does, and that choice is the whole compression-ratio story —
/// the filters exist to turn pixel data into small, repetitive residuals
/// that DEFLATE can then code cheaply, so picking well matters far more
/// than the arithmetic (which is a byte-wise subtract either way).
///
/// The two directions also differ in data dependence, which is why this
/// can't share the decode-side SIMD. Reconstruction is inherently
/// sequential — `Recon(a)` is the byte the previous iteration just wrote —
/// but filtering reads only the *original* bytes, which are all available
/// up front. Filtering is therefore embarrassingly parallel per byte where
/// defiltering is a serial dependency chain.
#[derive(Debug)]
pub struct Filter {
    /// Bytes per complete pixel — the left-neighbour stride, matching
    /// [`crate::Defilter`]'s `bpp`.
    bpp: usize,
}

impl Filter {
    /// Build a filterer for scanlines matching `ihdr`'s pixel geometry.
    #[must_use]
    pub const fn new(ihdr: &Ihdr) -> Self {
        Self {
            bpp: ihdr.bytes_per_pixel(),
        }
    }

    /// Filter `row` with `filter` into `out`, appending the leading
    /// filter-type byte followed by `row.len()` filtered bytes.
    ///
    /// `row` is the raw (unfiltered) scanline and `previous_row` the raw
    /// scanline above it — all-zero for the first row of an image or Adam7
    /// pass, matching the decode side's convention (spec §9.2 treats the
    /// row above the first as zero).
    ///
    /// # Panics
    ///
    /// If `previous_row` isn't the same length as `row`; both are full
    /// scanlines of the same image geometry, so a mismatch is a caller bug.
    pub fn filter_row_into(
        &self,
        filter: FilterType,
        row: &[u8],
        previous_row: &[u8],
        out: &mut Vec<u8>,
    ) {
        assert_eq!(
            row.len(),
            previous_row.len(),
            "current and previous scanlines must have equal length"
        );

        out.push(filter.to_byte());
        out.reserve(row.len());

        match filter {
            FilterType::None => out.extend_from_slice(row),
            FilterType::Sub => {
                // The first `bpp` bytes have no left neighbour, so `a` is
                // zero there and they pass through unchanged.
                let split = self.bpp.min(row.len());
                out.extend_from_slice(&row[..split]);
                for index in split..row.len() {
                    out.push(row[index].wrapping_sub(row[index - self.bpp]));
                }
            }
            FilterType::Up => {
                for (&x, &b) in row.iter().zip(previous_row) {
                    out.push(x.wrapping_sub(b));
                }
            }
            FilterType::Average => {
                let split = self.bpp.min(row.len());
                for index in 0..split {
                    // No left neighbour yet: `a` is 0, so the predictor is
                    // floor(b / 2).
                    out.push(row[index].wrapping_sub(previous_row[index] / 2));
                }
                for index in split..row.len() {
                    let a = row[index - self.bpp];
                    let b = previous_row[index];
                    out.push(row[index].wrapping_sub(u8::midpoint(a, b)));
                }
            }
            FilterType::Paeth => {
                let split = self.bpp.min(row.len());
                for index in 0..split {
                    // With `a` and `c` both zero the Paeth predictor always
                    // selects `b`.
                    out.push(row[index].wrapping_sub(previous_row[index]));
                }
                for index in split..row.len() {
                    let a = row[index - self.bpp];
                    let b = previous_row[index];
                    let c = previous_row[index - self.bpp];
                    out.push(row[index].wrapping_sub(Self::paeth_predictor(a, b, c)));
                }
            }
        }
    }

    /// Filter `row` with the filter that scores best under the minimum sum
    /// of absolute differences heuristic, appending it to `out` and
    /// returning the filter chosen.
    ///
    /// # The heuristic
    ///
    /// This is libpng's long-standing rule: score each candidate by summing
    /// its filtered bytes interpreted as *signed* residuals, and keep the
    /// smallest total. The reasoning is that DEFLATE codes small,
    /// repetitive values in shorter Huffman codes, so a row whose residuals
    /// cluster near zero compresses better — the sum is a cheap proxy for
    /// "how much entropy did this filter leave behind".
    ///
    /// It is only a proxy: it can't see cross-row redundancy or the actual
    /// Huffman code lengths DEFLATE will assign, so it occasionally picks a
    /// filter that isn't truly optimal. Measuring the real cost would mean
    /// compressing each candidate, which is far too expensive per row. In
    /// practice this heuristic lands within a couple of percent of that
    /// exhaustive search, which is why essentially every encoder uses it.
    ///
    /// # Panics
    ///
    /// If `previous_row` isn't the same length as `row`.
    pub fn filter_row_adaptive_into(
        &self,
        row: &[u8],
        previous_row: &[u8],
        scratch: &mut Vec<u8>,
        out: &mut Vec<u8>,
    ) -> FilterType {
        // Row 0 of an image/pass has an all-zero row above it, so Up
        // degenerates to None, and Average/Paeth degenerate to weaker forms
        // of Sub. Scoring them anyway is pure waste, but correctness does
        // not depend on this shortcut -- it only trims the candidate set.
        let previous_is_zero = previous_row.iter().all(|&b| b == 0);
        let candidates: &[FilterType] = if previous_is_zero {
            &[FilterType::None, FilterType::Sub]
        } else {
            &FilterType::ALL
        };

        let mut best = FilterType::None;
        let mut best_score = u64::MAX;
        for &candidate in candidates {
            scratch.clear();
            self.filter_row_into(candidate, row, previous_row, scratch);
            // Skip the filter-type byte: it is framing, not residual data.
            let score = Self::absolute_difference_score(&scratch[1..]);
            if score < best_score {
                best_score = score;
                best = candidate;
            }
        }

        self.filter_row_into(best, row, previous_row, out);
        best
    }

    /// Sum of the filtered bytes read as signed residuals (libpng's
    /// "minimum sum of absolute differences" metric).
    ///
    /// A filtered byte is a wrapped difference, so `0xFF` means -1, not
    /// 255. Reading it back as `i8` and taking the magnitude is what makes
    /// a residual of -1 score as cheaply as +1, which is the entire point:
    /// small differences in either direction are what compress well.
    fn absolute_difference_score(filtered: &[u8]) -> u64 {
        filtered
            .iter()
            .map(|&byte| u64::from(byte.cast_signed().unsigned_abs()))
            .sum()
    }

    /// The Paeth predictor (spec §9.4), identical to the decode side's.
    ///
    /// Computed in `i16` because `p = a + b - c` can fall outside a `u8` in
    /// both directions before the nearest-of-three comparison picks a
    /// winner; the result is always one of the three inputs, so it fits.
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
    use super::Filter;
    use crate::{ColorType, Defilter, FilterType, Ihdr, InterlaceMethod};

    /// Test-side helper owning the filter/defilter round-trip check and the
    /// synthetic scanlines it runs on, so these stay methods with a clear
    /// owner rather than loose helper functions.
    struct RoundTrip {
        header: Ihdr,
    }

    impl RoundTrip {
        fn new(width: u32, color_type: ColorType, bit_depth: u8) -> Self {
            Self {
                header: Ihdr {
                    width,
                    height: 1,
                    bit_depth,
                    color_type,
                    interlace: InterlaceMethod::None,
                },
            }
        }

        fn header(&self) -> &Ihdr {
            &self.header
        }

        fn stride(&self) -> usize {
            self.header.stride()
        }

        /// A deterministic pseudo-random scanline; `step`/`offset` vary it
        /// so a row and the row above it don't coincide by accident.
        fn row(len: usize, step: usize, offset: usize) -> Vec<u8> {
            (0..len)
                .map(|i| u8::try_from((i * step + offset) % 256).expect("masked into u8 range"))
                .collect()
        }

        /// Filter a row then defilter it, asserting the original comes back.
        fn check(&self, filter_type: FilterType, row: &[u8], previous: &[u8]) {
            let filter = Filter::new(&self.header);
            let mut filtered = Vec::new();
            filter.filter_row_into(filter_type, row, previous, &mut filtered);
            assert_eq!(filtered[0], filter_type.to_byte());
            assert_eq!(filtered.len(), row.len() + 1);

            Defilter::new(&self.header)
                .defilter_row(&mut filtered, previous)
                .unwrap();
            assert_eq!(&filtered[1..], row, "{filter_type:?} did not round-trip");
        }
    }

    #[test]
    fn every_filter_round_trips_through_defilter() {
        // The property that actually matters: whatever we emit, our own
        // decoder must reconstruct exactly. Covers bpp 1 (grayscale) and 4
        // (truecolor+alpha), with a non-zero previous row so Up/Average/
        // Paeth all engage their real predictors.
        let cases = [
            RoundTrip::new(8, ColorType::Grayscale, 8),
            RoundTrip::new(2, ColorType::TruecolorAlpha, 8),
        ];
        for case in cases {
            let stride = case.stride();
            let row = RoundTrip::row(stride, 37, 11);
            let previous = RoundTrip::row(stride, 91, 5);
            for filter_type in FilterType::ALL {
                case.check(filter_type, &row, &previous);
            }
        }
    }

    #[test]
    fn round_trips_with_an_all_zero_previous_row() {
        // Row 0 of every image and every Adam7 pass takes this path.
        let case = RoundTrip::new(4, ColorType::Truecolor, 8);
        let row = RoundTrip::row(12, 17, 3);
        let previous = vec![0u8; 12];
        for filter_type in FilterType::ALL {
            case.check(filter_type, &row, &previous);
        }
    }

    #[test]
    fn round_trips_when_the_row_is_shorter_than_one_pixel() {
        // A 1-pixel-wide 16-bit truecolor+alpha row has stride 8 and bpp 8,
        // so the "no left neighbour" prefix covers the entire row and the
        // main loops never execute.
        let case = RoundTrip::new(1, ColorType::TruecolorAlpha, 16);
        assert_eq!(case.header().bytes_per_pixel(), 8);
        assert_eq!(case.stride(), 8);
        let row = RoundTrip::row(8, 29, 7);
        for filter_type in FilterType::ALL {
            case.check(filter_type, &row, &[0u8; 8]);
            case.check(filter_type, &row, &[200u8; 8]);
        }
    }

    #[test]
    fn sub_filter_emits_left_neighbour_deltas() {
        let case = RoundTrip::new(4, ColorType::Grayscale, 8);
        let mut out = Vec::new();
        Filter::new(case.header()).filter_row_into(
            FilterType::Sub,
            &[10, 13, 20, 18],
            &[0; 4],
            &mut out,
        );
        // First byte passes through; the rest are differences from the
        // left, wrapping: 18 - 20 = -2, which is 254 as a u8.
        assert_eq!(out, vec![FilterType::Sub.to_byte(), 10, 3, 7, 254]);
    }

    #[test]
    fn up_filter_emits_deltas_from_the_row_above() {
        let case = RoundTrip::new(4, ColorType::Grayscale, 8);
        let mut out = Vec::new();
        Filter::new(case.header()).filter_row_into(
            FilterType::Up,
            &[10, 20, 30, 40],
            &[1, 2, 3, 4],
            &mut out,
        );
        assert_eq!(out, vec![FilterType::Up.to_byte(), 9, 18, 27, 36]);
    }

    #[test]
    fn average_filter_matches_the_spec_formula() {
        // Filt(x) = Orig(x) - floor((Orig(a) + Orig(b)) / 2), with a = 0 on
        // the first byte.
        let case = RoundTrip::new(3, ColorType::Grayscale, 8);
        let mut out = Vec::new();
        Filter::new(case.header()).filter_row_into(
            FilterType::Average,
            &[100, 120, 140],
            &[10, 20, 30],
            &mut out,
        );
        assert_eq!(out[1], 100u8.wrapping_sub(10 / 2));
        assert_eq!(out[2], 120u8.wrapping_sub(u8::midpoint(100, 20)));
        assert_eq!(out[3], 140u8.wrapping_sub(u8::midpoint(120, 30)));
    }

    #[test]
    fn paeth_predictor_agrees_with_the_spec_selection_rule() {
        // When a + b - c equals one input exactly, that input wins; ties
        // resolve a, then b, then c.
        assert_eq!(Filter::paeth_predictor(10, 20, 10), 20);
        assert_eq!(Filter::paeth_predictor(20, 10, 10), 20);
        assert_eq!(Filter::paeth_predictor(5, 5, 5), 5);
        // Exercises the branch where c is nearest: p = 200 + 5 - 100 = 105,
        // so pa = 95, pb = 100, pc = 5 -- c wins.
        assert_eq!(Filter::paeth_predictor(200, 5, 100), 100);
    }

    #[test]
    fn adaptive_choice_round_trips_and_beats_or_matches_no_filtering() {
        let case = RoundTrip::new(64, ColorType::Truecolor, 8);
        let stride = case.stride();
        let filter = Filter::new(case.header());

        // A smooth horizontal gradient: strongly favours Sub, and must
        // never score worse than leaving the row unfiltered.
        let row = RoundTrip::row(stride, 3, 0);
        let previous = RoundTrip::row(stride, 3, 0);

        let mut scratch = Vec::new();
        let mut out = Vec::new();
        let chosen = filter.filter_row_adaptive_into(&row, &previous, &mut scratch, &mut out);

        let mut unfiltered = Vec::new();
        filter.filter_row_into(FilterType::None, &row, &previous, &mut unfiltered);
        assert!(
            Filter::absolute_difference_score(&out[1..])
                <= Filter::absolute_difference_score(&unfiltered[1..])
        );

        assert_eq!(out[0], chosen.to_byte());
        Defilter::new(case.header())
            .defilter_row(&mut out, &previous)
            .unwrap();
        assert_eq!(&out[1..], &row[..]);
    }

    #[test]
    fn adaptive_picks_up_for_vertically_repeating_rows() {
        // Identical consecutive rows are exactly what Up is for: every
        // residual becomes zero, which no other filter can match here.
        let case = RoundTrip::new(32, ColorType::Truecolor, 8);
        let row = RoundTrip::row(case.stride(), 7, 13);
        let chosen = Filter::new(case.header()).filter_row_adaptive_into(
            &row,
            &row.clone(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert_eq!(chosen, FilterType::Up);
    }

    #[test]
    fn adaptive_restricts_candidates_on_an_all_zero_previous_row() {
        // With no row above, Up is equivalent to None and must never be
        // selected over it -- the tie would otherwise waste a byte's worth
        // of ambiguity and the shortcut would be unobservable.
        let case = RoundTrip::new(16, ColorType::Grayscale, 8);
        let row = RoundTrip::row(case.stride(), 5, 0);
        let chosen = Filter::new(case.header()).filter_row_adaptive_into(
            &row,
            &vec![0u8; case.stride()],
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert!(matches!(chosen, FilterType::None | FilterType::Sub));
    }

    #[test]
    fn score_reads_filtered_bytes_as_signed_residuals() {
        // 0xFF is -1, so it must score 1, not 255 -- the property that makes
        // the metric symmetric around zero.
        assert_eq!(Filter::absolute_difference_score(&[0xFF]), 1);
        assert_eq!(Filter::absolute_difference_score(&[0x01]), 1);
        assert_eq!(Filter::absolute_difference_score(&[0x80]), 128);
        assert_eq!(Filter::absolute_difference_score(&[0, 0, 0]), 0);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn mismatched_row_lengths_panic() {
        Filter::new(RoundTrip::new(4, ColorType::Grayscale, 8).header()).filter_row_into(
            FilterType::Up,
            &[1, 2, 3, 4],
            &[1, 2, 3],
            &mut Vec::new(),
        );
    }
}
