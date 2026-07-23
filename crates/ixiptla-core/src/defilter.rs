use crate::simd::{WrappingByteAdder, WrappingPrefixSum};
use crate::{FilterType, Ihdr, InvalidFilterType};

/// Reverses PNG's per-scanline byte filtering (spec §9), turning the
/// inflated `IDAT` byte stream (filter-type byte + filtered bytes, repeated
/// per row) back into raw pixel bytes.
///
/// This owns the "previous reconstructed scanline" state that `Up`,
/// `Average`, and `Paeth` all need — keeping it here (rather than having
/// callers pass the previous row in and out by hand) means a caller just
/// calls [`Self::defilter_row`] once per scanline, in order, and the struct
/// tracks continuity itself. Interlacing (`Adam7`) is *not* handled here:
/// each `Adam7` pass is its own independent sequence of scanlines with its
/// own "previous row" state, so the `Adam7` module drives one `Defilter`
/// per pass.
#[derive(Debug)]
pub struct Defilter {
    /// Bytes per complete pixel (spec's "bpp"), used as the `Sub`/`Average`/
    /// `Paeth` predictors' fixed left-neighbor stride.
    bpp: usize,
    /// The previously reconstructed scanline, or all-zero for the first row
    /// (spec §9.2: "Filt(x) is treated as ... equal to zero" for the row
    /// above the first, i.e. `Recon(b) = 0` on row 0).
    previous_row: Vec<u8>,
}

impl Defilter {
    /// Build a defilterer for scanlines matching `ihdr`'s pixel geometry
    /// (bpp = bytes per complete pixel) and stride (row width in bytes).
    #[must_use]
    pub fn new(ihdr: &Ihdr) -> Self {
        Self {
            bpp: ihdr.bytes_per_pixel(),
            previous_row: vec![0u8; ihdr.stride()],
        }
    }

    /// Defilter one scanline in place.
    ///
    /// `row` must be exactly `filter_type_byte + stride bytes` — i.e. the
    /// leading filter-type byte plus the filtered row content, sliced
    /// straight out of the inflated `IDAT` stream. On success, `row[1..]`
    /// holds the reconstructed (unfiltered) pixel bytes, and the struct
    /// remembers this row as `previous_row` for the next call.
    ///
    /// # Errors
    ///
    /// [`InvalidFilterType`] if `row[0]` isn't one of the 5 defined filter
    /// types.
    ///
    /// An empty `row` (no filter-type byte at all -- never produced by a
    /// well-formed `IDAT` stream, whose stride is always >= 0 content
    /// bytes plus exactly one filter-type byte) is treated as `FilterType::
    /// None` over zero content bytes, i.e. a harmless no-op.
    pub fn defilter_row(&mut self, row: &mut [u8]) -> Result<(), InvalidFilterType> {
        let Some((filter_byte, content)) = row.split_first_mut() else {
            return Ok(());
        };
        let filter_type = FilterType::from_byte(*filter_byte)?;

        match filter_type {
            FilterType::None => {}
            FilterType::Sub => self.undo_sub(content),
            FilterType::Up => self.undo_up(content),
            FilterType::Average => self.undo_average(content),
            FilterType::Paeth => self.undo_paeth(content),
        }

        self.previous_row.copy_from_slice(content);
        Ok(())
    }

    /// `Recon(x) = Filt(x) + Recon(a)`, `a` = bpp bytes to the left (0 if
    /// there isn't one).
    //
    // `bpp` interleaved lanes, each independently needing a running
    // ("prefix") sum, are mutually independent but *within* one lane
    // there's a genuine sequential dependency: `lane[i]` depends on
    // `lane[i-1]`. When `bpp == 1` there's only a single lane -- the whole
    // row *is* that one lane -- so `undo_sub` reduces exactly to
    // `WrappingPrefixSum::scan`, which computes it with SSE2's
    // shift-and-add ("Hillis-Steele") trick instead of one dependent
    // scalar add per byte. `bpp == 1` covers grayscale and indexed-color
    // images (including all their sub-byte-depth variants -- `bpp` is
    // always the *byte*-rounded-up stride, never fractional, per spec
    // §9.2), a common real case.
    //
    // For `bpp > 1` (RGB/RGBA/grayscale+alpha), the shift amounts a
    // multi-lane SSE2 scan needs are only expressible as compile-time
    // immediates on stable Rust (`_mm_slli_si128`'s shift count is a
    // `const` generic, and arbitrary `bpp`-dependent const arithmetic
    // needs the unstable `generic_const_exprs` feature), so those stay on
    // the plain scalar loop rather than reaching for `unsafe` machinery
    // this crate can't build on stable across every `bpp` it supports.
    fn undo_sub(&self, content: &mut [u8]) {
        if self.bpp == 1 {
            WrappingPrefixSum::scan(content);
            return;
        }
        for i in self.bpp..content.len() {
            content[i] = content[i].wrapping_add(content[i - self.bpp]);
        }
    }

    /// `Recon(x) = Filt(x) + Recon(b)`, `b` = the byte directly above.
    //
    // Unlike `Sub`/`Average`/`Paeth`, `Up` has no dependency between bytes
    // *within* the row at all -- every output byte depends only on
    // `previous_row`, a value that's already fully known before this call
    // starts. That makes it embarrassingly parallel across the whole row,
    // not just across `bpp` interleaved lanes, so it's handed straight to
    // `WrappingByteAdder`, which does the actual add 16 bytes at a time via
    // SSE2 on x86_64 (falling back to the same scalar loop this used to be,
    // on other architectures).
    fn undo_up(&self, content: &mut [u8]) {
        WrappingByteAdder::add_wrapping(content, &self.previous_row);
    }

    /// `Recon(x) = Filt(x) + floor((Recon(a) + Recon(b)) / 2)`.
    //
    // Split into a branch-free "first pixel" prefix (where `a` is always 0,
    // per spec §9.2 -- there's no left neighbor) and a branch-free "rest of
    // row" suffix, instead of testing `i >= bpp` on every byte. Two reasons:
    //
    // 1. For `a == 0`, `floor((0 + b) / 2) == b / 2` -- no need to even read
    //    a phantom left neighbor.
    // 2. The suffix loop's body has no branch at all now, which both saves
    //    the per-byte compare and lets each of the `bpp` interleaved
    //    dependency chains (`content[i]` depends on `content[i - bpp]`, so
    //    lanes `0..bpp` are mutually independent) pipeline without a branch
    //    misprediction bubble between them.
    //
    // Indexing (rather than `.iter_mut().enumerate()`) is still load-bearing
    // in the suffix loop: each iteration reads back `content[i - bpp]`, an
    // *already reconstructed* earlier element of the same slice we're
    // writing -- not expressible as a plain iterator without a second pass.
    fn undo_average(&self, content: &mut [u8]) {
        let bpp = self.bpp;
        let prefix_len = bpp.min(content.len());
        for (byte, &above) in content[..prefix_len].iter_mut().zip(&self.previous_row) {
            *byte = byte.wrapping_add(above / 2); // a == 0: floor(b/2).
        }
        #[allow(clippy::needless_range_loop)]
        for i in bpp..content.len() {
            let a = content[i - bpp];
            let b = self.previous_row[i];
            // midpoint, but in a wider type: a + b can be up to 510, which
            // doesn't fit in u8, before the floor-divide brings it back down.
            let avg = u16::midpoint(u16::from(a), u16::from(b));
            #[allow(clippy::cast_possible_truncation)]
            let avg = avg as u8; // avg <= 255 always: (255+255)/2 = 255.
            content[i] = content[i].wrapping_add(avg);
        }
    }

    /// `Recon(x) = Filt(x) + PaethPredictor(Recon(a), Recon(b), Recon(c))`.
    //
    // Same prefix/suffix split as `undo_average` above, for the same two
    // reasons (skip the phantom-neighbor branch; let the `bpp` independent
    // dependency chains in the suffix pipeline branch-free). On the first
    // `bpp` bytes, `a == c == 0`, so `PaethPredictor(0, b, 0)` always
    // reduces to `b` (`p = b`, `pa = pc = |b|`, `pb = 0 <= pa`, so `b` wins
    // or ties every comparison against `a`/`c` and wins ties) -- no need to
    // call the general predictor for those bytes at all.
    //
    // Same self-referencing-slice reason as `undo_average` above: `a` and
    // `c` read back already-reconstructed earlier elements of `content`.
    fn undo_paeth(&self, content: &mut [u8]) {
        let bpp = self.bpp;
        let prefix_len = bpp.min(content.len());
        for (byte, &above) in content[..prefix_len].iter_mut().zip(&self.previous_row) {
            *byte = byte.wrapping_add(above);
        }
        #[allow(clippy::needless_range_loop)]
        for i in bpp..content.len() {
            let a = content[i - bpp];
            let b = self.previous_row[i];
            let c = self.previous_row[i - bpp];
            content[i] = content[i].wrapping_add(Self::paeth_predictor(a, b, c));
        }
    }

    /// Spec §9.4's Paeth predictor: picks whichever of the left/above/
    /// upper-left neighbor is closest to `a + b - c` (linear-gradient
    /// prediction), with ties broken in the order left, above, upper-left.
    fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
        let (a, b, c) = (i32::from(a), i32::from(b), i32::from(c));
        let p = a + b - c;
        let pa = (p - a).abs();
        let pb = (p - b).abs();
        let pc = (p - c).abs();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        if pa <= pb && pa <= pc {
            a as u8
        } else if pb <= pc {
            b as u8
        } else {
            c as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Defilter;
    use crate::{ColorType, Ihdr, InterlaceMethod};

    /// A tiny synthetic IHDR: 4-wide, 8-bit grayscale (bpp=1, stride=4).
    fn grayscale_ihdr(width: u32, height: u32) -> Ihdr {
        Ihdr {
            width,
            height,
            bit_depth: 8,
            color_type: ColorType::Grayscale,
            interlace: InterlaceMethod::None,
        }
    }

    #[test]
    fn filter_type_none_is_identity() {
        let ihdr = grayscale_ihdr(4, 2);
        let mut defilter = Defilter::new(&ihdr);
        let mut row = vec![0u8, 10, 20, 30, 40]; // filter byte + 4 content bytes
        defilter.defilter_row(&mut row).unwrap();
        assert_eq!(&row[1..], &[10, 20, 30, 40]);
    }

    #[test]
    fn sub_filter_undoes_left_neighbor_delta() {
        // Original row: [10, 20, 30, 40]. Sub-filtered: [10, 10, 10, 10]
        // (each byte minus its bpp=1 left neighbor, first byte minus 0).
        let ihdr = grayscale_ihdr(4, 1);
        let mut defilter = Defilter::new(&ihdr);
        let mut row = vec![1u8, 10, 10, 10, 10];
        defilter.defilter_row(&mut row).unwrap();
        assert_eq!(&row[1..], &[10, 20, 30, 40]);
    }

    #[test]
    fn up_filter_undoes_delta_from_previous_row() {
        let ihdr = grayscale_ihdr(4, 2);
        let mut defilter = Defilter::new(&ihdr);

        // Row 0, unfiltered, establishes previous_row = [10, 20, 30, 40].
        let mut row0 = vec![0u8, 10, 20, 30, 40];
        defilter.defilter_row(&mut row0).unwrap();

        // Row 1, Up-filtered: each byte is original minus the byte above.
        // Original row 1 = [11, 22, 33, 44] -> filtered = [1, 2, 3, 4].
        let mut row1 = vec![2u8, 1, 2, 3, 4];
        defilter.defilter_row(&mut row1).unwrap();
        assert_eq!(&row1[1..], &[11, 22, 33, 44]);
    }

    #[test]
    fn average_filter_matches_spec_formula_on_first_row() {
        // First row: b (above) is always 0, so avg = floor(a/2).
        // Original: [10, 20, 30, 40].
        // avg-filtered[0] = 10 - floor((0+0)/2) = 10
        // avg-filtered[1] = 20 - floor((10+0)/2) = 15
        // avg-filtered[2] = 30 - floor((20+0)/2) = 20
        // avg-filtered[3] = 40 - floor((30+0)/2) = 25
        let ihdr = grayscale_ihdr(4, 1);
        let mut defilter = Defilter::new(&ihdr);
        let mut row = vec![3u8, 10, 15, 20, 25];
        defilter.defilter_row(&mut row).unwrap();
        assert_eq!(&row[1..], &[10, 20, 30, 40]);
    }

    #[test]
    fn paeth_filter_matches_predictor_on_first_row() {
        // First row: b=c=0 always, so PaethPredictor(a,0,0) reduces to
        // picking a (since p=a, pa=0 is always <= pb=|a-... |). So Paeth on
        // row 0 behaves exactly like Sub.
        let ihdr = grayscale_ihdr(4, 1);
        let mut defilter = Defilter::new(&ihdr);
        let mut row = vec![4u8, 10, 10, 10, 10]; // same deltas as the Sub test
        defilter.defilter_row(&mut row).unwrap();
        assert_eq!(&row[1..], &[10, 20, 30, 40]);
    }

    #[test]
    fn paeth_filter_second_row_matches_hand_worked_example() {
        let ihdr = grayscale_ihdr(3, 2);
        let mut defilter = Defilter::new(&ihdr);

        // Row 0 unfiltered: [5, 10, 15] (bpp=1).
        let mut row0 = vec![0u8, 5, 10, 15];
        defilter.defilter_row(&mut row0).unwrap();

        // Row 1 original: [6, 12, 18].
        // i=0: a=0,b=5,c=0 -> p=5, pa=5,pb=0,pc=5 -> pick b=5 -> filtered = 6-5=1
        // i=1: a=6 (recon),b=10,c=5 -> p=6+10-5=11, pa=5,pb=1,pc=6 -> pick b=10
        //      -> filtered = 12-10=2
        // i=2: a=12,b=15,c=10 -> p=12+15-10=17, pa=5,pb=2,pc=7 -> pick b=15
        //      -> filtered = 18-15=3
        let mut row1 = vec![4u8, 1, 2, 3];
        defilter.defilter_row(&mut row1).unwrap();
        assert_eq!(&row1[1..], &[6, 12, 18]);
    }

    #[test]
    fn rejects_invalid_filter_type_byte() {
        let ihdr = grayscale_ihdr(4, 1);
        let mut defilter = Defilter::new(&ihdr);
        let mut row = vec![5u8, 1, 2, 3, 4];
        assert!(defilter.defilter_row(&mut row).is_err());
    }

    #[test]
    fn multi_byte_pixel_sub_filter_uses_bpp_stride() {
        // 8-bit RGB (bpp=3), width 2 -> stride 6.
        let ihdr = Ihdr {
            width: 2,
            height: 1,
            bit_depth: 8,
            color_type: ColorType::Truecolor,
            interlace: InterlaceMethod::None,
        };
        let mut defilter = Defilter::new(&ihdr);
        // Original pixels: (10,20,30) (15,25,35).
        // Sub-filtered: pixel0 unchanged (no left neighbor), pixel1 minus
        // pixel0 componentwise: (5,5,5).
        let mut row = vec![1u8, 10, 20, 30, 5, 5, 5];
        defilter.defilter_row(&mut row).unwrap();
        assert_eq!(&row[1..], &[10, 20, 30, 15, 25, 35]);
    }
}
