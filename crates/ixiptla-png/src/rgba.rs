use crate::{ColorType, Ihdr, Palette, PaletteError, PixelBits, Transparency};

/// Expands [`crate::Png::pixels`] (raw, defiltered, still in whatever
/// bit-depth/colour-type `IHDR` declared) into a flat RGBA8 buffer — 4
/// bytes per pixel, one byte per channel, in image row-major order.
///
/// This is a distinct concern from decoding (spec §7/§11.2.3's transforms
/// are "how to interpret pixel bytes", not "how to reverse filtering/
/// compression") and from [`Palette`] (which only knows how to look up one
/// index — nothing about bit-packing or the other 4 colour types). Kept as
/// its own struct so it owns exactly one job: for every pixel coordinate,
/// read that pixel's raw sample(s) and produce 4 RGBA8 bytes, regardless
/// of which of PNG's 5 colour types or 5 bit depths produced the input.
#[derive(Debug)]
pub struct RgbaExpander<'a> {
    ihdr: &'a Ihdr,
    palette: Option<&'a Palette>,
    transparency: Option<&'a Transparency>,
}

impl<'a> RgbaExpander<'a> {
    /// `palette` is required (and used) only for
    /// [`ColorType::IndexedColor`] images; ignored otherwise. `transparency`
    /// (spec §11.3.2's `tRNS`) is optional for every colour type it can
    /// apply to — when given, it overrides the alpha this would otherwise
    /// produce (255, or the palette-index lookup) for pixels matching the
    /// designated transparent sample/colour/index.
    #[must_use]
    pub const fn new(
        ihdr: &'a Ihdr,
        palette: Option<&'a Palette>,
        transparency: Option<&'a Transparency>,
    ) -> Self {
        Self {
            ihdr,
            palette,
            transparency,
        }
    }

    /// Expand `pixels` (as returned by [`crate::Png::decode`]: `height`
    /// rows of exactly [`Ihdr::stride`] bytes each, no filter-type bytes,
    /// no interlacing) into `width * height * 4` RGBA8 bytes.
    ///
    /// # Errors
    ///
    /// [`ExpandError::MissingPalette`] if the image is indexed-color and no
    /// palette was given to [`Self::new`]. [`ExpandError::Palette`] if an
    /// index sample points past the palette's end.
    /// [`ExpandError::TruncatedPixels`] if `pixels` is shorter than
    /// `height * stride` (should never happen for a `Png` produced by this
    /// crate's own decoder, but this fn doesn't trust its input blindly).
    pub fn expand(&self, pixels: &[u8]) -> Result<Vec<u8>, ExpandError> {
        if self.ihdr.color_type == ColorType::IndexedColor && self.palette.is_none() {
            return Err(ExpandError::MissingPalette);
        }

        let stride = self.ihdr.stride();
        let width = self.ihdr.width as usize;
        let height = self.ihdr.height as usize;
        if pixels.len() < stride * height {
            return Err(ExpandError::TruncatedPixels);
        }

        // 8-bit TruecolorAlpha is already byte-identical to this fn's own
        // RGBA8 output: 4 channels, no bit-packing to unpack (bit_depth ==
        // 8, so `scale_to_u8` would be the identity on every sample), and
        // no possible tRNS override (spec section 11.3.2 forbids tRNS on
        // color types that already carry their own alpha channel --
        // enforced by `Png::decode` rejecting such a chunk before this
        // ever runs). So for this one very common case (this crate's own
        // benchmarks' dominant fixture, and probably the single most
        // common PNG configuration in the wild) the whole per-pixel decode
        // loop below is provably a no-op transform -- a straight copy of
        // the raw scanlines is the same bytes (stride == width * 4
        // exactly at 8-bit depth, so this is contiguous, no padding to
        // skip).
        if self.ihdr.color_type == ColorType::TruecolorAlpha && self.ihdr.bit_depth == 8 {
            return Ok(pixels[..width * height * 4].to_vec());
        }
        // (See `expand_owned` for a variant of this same fast path that
        // avoids this `to_vec()` copy entirely when the caller already
        // owns `pixels` outright.)

        // 8-bit Grayscale has no true zero-copy path the way 8-bit
        // TruecolorAlpha does above -- the source is 1 byte/pixel and the
        // destination is inescapably 4 (v, v, v, a), so *some* buffer of 4x
        // the size has to be written no matter what. What this fast path
        // avoids instead is everything `expand_pixel`'s fully generic loop
        // pays per pixel that this narrower case doesn't need: the `[u64;
        // 4]` sample-array dance and `scale_to_u8` (both no-ops at
        // bit_depth == 8, but not statically known to the compiler inside
        // the generic loop), the `ColorType` match, the `is_transparent_gray`
        // closure/Option indirection, and -- the actual dominant cost per
        // the `rgba_expand_grayscale` benchmark -- one `Vec::extend_from_
        // slice` call per pixel (4 bytes at a time) instead of writing
        // directly into a single pre-sized `Vec` via direct indexing.
        // `stride == width` here (1 byte/pixel, already byte-aligned), so
        // `pixels[..width*height]` is exactly this image's sample bytes
        // with no per-row padding to skip.
        if self.ihdr.color_type == ColorType::Grayscale && self.ihdr.bit_depth == 8 {
            return Ok(self.expand_grayscale_8bit_fast(&pixels[..width * height]));
        }

        let mut out = Vec::with_capacity(width * height * 4);
        let channels = u32::from(self.ihdr.color_type.channel_count());
        let bit_depth = u32::from(self.ihdr.bit_depth);

        for y in 0..height {
            let row = &pixels[y * stride..][..stride];
            for x in 0..width {
                let rgba = self.expand_pixel(row, x, channels, bit_depth)?;
                out.extend_from_slice(&rgba);
            }
        }
        Ok(out)
    }

    /// Like `expand`, but takes ownership of `pixels` (typically straight
    /// off `crate::Png::decode`'s `pixels` field, which every real caller
    /// -- `ixiptla-conformance`'s reference decoder, this crate's own
    /// benchmarks -- discards immediately after the `expand` call anyway)
    /// instead of borrowing it.
    ///
    /// This exists purely to close a gap `expand`'s own 8-bit
    /// `TruecolorAlpha` fast path can't: when the caller only has a
    /// `&[u8]` borrow, the only legal way to hand back an owned `Vec<u8>`
    /// of those same bytes is a real `to_vec()` copy (a second full-image
    /// memcpy, on top of whichever pass already produced `pixels` --
    /// `defilter_sequential`'s own decoder loop). But when the caller
    /// already owns `pixels` outright, a fresh copy is never the only
    /// option -- `Vec::truncate` on the caller's own buffer is a true
    /// no-op, no bytes moved at all (this fast path's exact-8bpp-RGBA8
    /// case never even has trailing bytes to drop). Every other colour-
    /// type/bit-depth combination still allocates and writes a fresh
    /// output buffer exactly like `expand` does -- there's no owned-input
    /// trick that helps when the transform itself is non-trivial (e.g.
    /// indexed-color's palette lookup or sub-byte bit-unpacking); this is
    /// specifically closing the one case that was a pure identity
    /// transform paying for a copy it didn't need.
    ///
    /// # Errors
    ///
    /// Same as `expand`.
    pub fn expand_owned(&self, mut pixels: Vec<u8>) -> Result<Vec<u8>, ExpandError> {
        if self.ihdr.color_type == ColorType::IndexedColor && self.palette.is_none() {
            return Err(ExpandError::MissingPalette);
        }

        let stride = self.ihdr.stride();
        let width = self.ihdr.width as usize;
        let height = self.ihdr.height as usize;
        if pixels.len() < stride * height {
            return Err(ExpandError::TruncatedPixels);
        }

        // Same fast path as `expand`'s own (see that method's comment for
        // why this transform is provably a no-op at 8-bit TruecolorAlpha)
        // -- but landing on `Vec::truncate` instead of `to_vec()`, since
        // `pixels` is already exactly the right bytes in an owned buffer:
        // no copy needed, just drop whatever (if anything) trails the
        // image's own `width * height * 4` bytes.
        if self.ihdr.color_type == ColorType::TruecolorAlpha && self.ihdr.bit_depth == 8 {
            pixels.truncate(width * height * 4);
            return Ok(pixels);
        }

        // Every other case has no such identity-transform shortcut --
        // delegate to `expand`'s borrowing path exactly as before.
        self.expand(&pixels)
    }

    /// Fast path for `expand` at 8-bit `Grayscale`: writes directly into
    /// one pre-sized output `Vec` via `chunks_exact_mut`, no per-pixel
    /// `Vec::extend_from_slice` call, no `PixelBits`/`u64` sample-array
    /// detour, and `tRNS` (if present) resolved to a plain `u16` compare
    /// against the raw grayscale byte (widened, since `Transparency::
    /// Grayscale` always stores its designated value as `u16` regardless
    /// of the image's own bit depth -- spec §11.3.2).
    fn expand_grayscale_8bit_fast(&self, samples: &[u8]) -> Vec<u8> {
        let transparent_value = self.transparency.and_then(|t| match t {
            Transparency::Grayscale(v) => Some(*v),
            Transparency::Truecolor(..) | Transparency::IndexedColor(_) => None,
        });

        let mut out = vec![0u8; samples.len() * 4];
        for (chunk, &v) in out.as_chunks_mut::<4>().0.iter_mut().zip(samples) {
            let a = if transparent_value == Some(u16::from(v)) {
                0
            } else {
                255
            };
            chunk.copy_from_slice(&[v, v, v, a]);
        }
        out
    }

    /// Read pixel `x`'s `channels` samples (each `bit_depth` bits, packed
    /// contiguously — spec §7.2) out of `row` and turn them into one RGBA8
    /// pixel, per this image's colour type.
    #[allow(clippy::many_single_char_names)] // r/g/b/a mirror the spec's own channel names
    fn expand_pixel(
        &self,
        row: &[u8],
        x: usize,
        channels: u32,
        bit_depth: u32,
    ) -> Result<[u8; 4], ExpandError> {
        // Read every channel's raw sample up front; each colour type below
        // just picks how to combine them into RGBA. 8-bit and 16-bit depths
        // (by far the common case -- every PNG this crate's own benchmarks
        // and most real-world images use) are byte-aligned: every channel
        // starts and ends on a byte boundary, so samples can be read
        // straight out of `row` with plain indexing/`from_be_bytes` instead
        // of `PixelBits::read`'s bit-by-bit loop (which this fn used to call
        // unconditionally -- profiling under `perf` via `ixiptla-bench`'s
        // `decode_loop` showed this was the dominant cost in the whole
        // decode pipeline, ~8-25x slower than the `png`/`image` crates).
        // Only the sub-byte depths (1/2/4, exclusive to Grayscale and
        // IndexedColor per spec §11.2.2) still need actual bit-level
        // packing, so they alone fall through to `PixelBits`.
        let mut samples = [0u64; 4];
        match bit_depth {
            8 => {
                let byte_offset = x * channels as usize;
                for (c, sample) in samples.iter_mut().take(channels as usize).enumerate() {
                    *sample = u64::from(row[byte_offset + c]);
                }
            }
            16 => {
                let byte_offset = x * channels as usize * 2;
                for (c, sample) in samples.iter_mut().take(channels as usize).enumerate() {
                    let hi = row[byte_offset + c * 2];
                    let lo = row[byte_offset + c * 2 + 1];
                    *sample = u64::from(u16::from_be_bytes([hi, lo]));
                }
            }
            _ => {
                let pixel_bits = channels * bit_depth;
                let bit_offset = x * pixel_bits as usize;
                for (c, sample) in samples.iter_mut().take(channels as usize).enumerate() {
                    *sample = PixelBits::read(row, bit_offset + c * bit_depth as usize, bit_depth);
                }
            }
        }

        Ok(match self.ihdr.color_type {
            ColorType::Grayscale => {
                let v = Self::scale_to_u8(samples[0], bit_depth);
                // tRNS (spec §11.3.2) compares the *raw* sample, at the
                // image's own bit depth, not the scaled 8-bit output.
                #[allow(clippy::cast_possible_truncation)]
                let a = if self.is_transparent_gray(samples[0]) {
                    0
                } else {
                    255
                };
                [v, v, v, a]
            }
            ColorType::GrayscaleAlpha => {
                let v = Self::scale_to_u8(samples[0], bit_depth);
                let a = Self::scale_to_u8(samples[1], bit_depth);
                [v, v, v, a]
            }
            ColorType::Truecolor => {
                let r = Self::scale_to_u8(samples[0], bit_depth);
                let g = Self::scale_to_u8(samples[1], bit_depth);
                let b = Self::scale_to_u8(samples[2], bit_depth);
                let a = if self.is_transparent_rgb(samples[0], samples[1], samples[2]) {
                    0
                } else {
                    255
                };
                [r, g, b, a]
            }
            ColorType::TruecolorAlpha => {
                let r = Self::scale_to_u8(samples[0], bit_depth);
                let g = Self::scale_to_u8(samples[1], bit_depth);
                let b = Self::scale_to_u8(samples[2], bit_depth);
                let a = Self::scale_to_u8(samples[3], bit_depth);
                [r, g, b, a]
            }
            ColorType::IndexedColor => {
                // Checked in `expand`: palette is Some whenever colour
                // type is IndexedColor.
                let palette = self.palette.expect("indexed color requires a palette");
                #[allow(clippy::cast_possible_truncation)]
                let index = samples[0] as u8; // index bit depth is always <= 8
                let [r, g, b] = palette.get(index)?;
                let a = self.transparency.map_or(255, |t| t.alpha_for_index(index));
                [r, g, b, a]
            }
        })
    }

    /// Whether raw grayscale `sample` (at the image's own bit depth) is
    /// `tRNS`'s designated transparent value.
    fn is_transparent_gray(&self, sample: u64) -> bool {
        #[allow(clippy::cast_possible_truncation)] // grayscale samples are always <= 16 bits
        let sample = sample as u16;
        self.transparency
            .is_some_and(|t| t.is_transparent_gray(sample))
    }

    /// Whether raw truecolor `(r, g, b)` (at the image's own bit depth) is
    /// `tRNS`'s designated transparent colour.
    fn is_transparent_rgb(&self, r: u64, g: u64, b: u64) -> bool {
        #[allow(clippy::cast_possible_truncation)] // truecolor samples are always <= 16 bits
        let (r, g, b) = (r as u16, g as u16, b as u16);
        self.transparency
            .is_some_and(|t| t.is_transparent_rgb(r, g, b))
    }

    /// Scale a `bit_depth`-wide sample up to the full 8-bit range (spec
    /// §7.3's "linear" sample scaling: `output = round(input *
    /// 255 / max_value)`), so a 1-bit sample of 1 becomes 255 (not 1),
    /// a 4-bit sample of 15 becomes 255, etc. For 8-bit input this is the
    /// identity; for 16-bit input this keeps only the high byte, which is
    /// what every 8-bit-per-channel consumer of 16-bit PNGs does.
    #[allow(clippy::cast_possible_truncation)] // each branch's value is proven <= 255
    const fn scale_to_u8(sample: u64, bit_depth: u32) -> u8 {
        if bit_depth == 8 {
            return sample as u8;
        }
        if bit_depth == 16 {
            return (sample >> 8) as u8;
        }
        let max_value = (1u64 << bit_depth) - 1;
        let scaled = (sample * 255 + max_value / 2) / max_value; // round to nearest
        scaled as u8
    }
}

/// Why [`RgbaExpander::expand`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandError {
    /// The image is [`ColorType::IndexedColor`] but no [`Palette`] was
    /// given to [`RgbaExpander::new`].
    MissingPalette,
    /// An indexed-color pixel's sample pointed past the palette's end.
    Palette(PaletteError),
    /// `pixels` was shorter than `height * stride` bytes.
    TruncatedPixels,
}

impl From<PaletteError> for ExpandError {
    fn from(e: PaletteError) -> Self {
        Self::Palette(e)
    }
}

impl core::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingPalette => {
                write!(f, "indexed-color image requires a PLTE palette to expand")
            }
            Self::Palette(e) => write!(f, "{e}"),
            Self::TruncatedPixels => write!(f, "pixel buffer is shorter than height * stride"),
        }
    }
}

impl std::error::Error for ExpandError {}

#[cfg(test)]
mod tests {
    use super::{ExpandError, RgbaExpander};
    use crate::{ColorType, Ihdr, InterlaceMethod, Palette, Transparency};

    fn ihdr(width: u32, height: u32, bit_depth: u8, color_type: ColorType) -> Ihdr {
        Ihdr {
            width,
            height,
            bit_depth,
            color_type,
            interlace: InterlaceMethod::None,
        }
    }

    #[test]
    fn expands_8bit_grayscale_to_rgba() {
        let ihdr = ihdr(2, 1, 8, ColorType::Grayscale);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [0u8, 128, 255, 0]; // padding irrelevant; stride=2
        let rgba = expander.expand(&pixels[..2]).unwrap();
        assert_eq!(rgba, vec![0, 0, 0, 255, 128, 128, 128, 255]);
    }

    #[test]
    fn expands_1bit_grayscale_scaling_to_full_range() {
        // 1-bit grayscale, width 8: byte 0b1010_0000 -> pixels 1,0,1,0,0,0,0,0
        let ihdr = ihdr(8, 1, 1, ColorType::Grayscale);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [0b1010_0000u8];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]); // pixel 0: 1 -> 255
        assert_eq!(&rgba[4..8], &[0, 0, 0, 255]); // pixel 1: 0 -> 0
        assert_eq!(&rgba[8..12], &[255, 255, 255, 255]); // pixel 2: 1 -> 255
    }

    #[test]
    fn expands_4bit_grayscale_replicates_nibble() {
        // 4-bit grayscale: max sample 15 -> 255, sample 8 -> round(8*255/15)=136
        let ihdr = ihdr(2, 1, 4, ColorType::Grayscale);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [0b1111_1000u8]; // pixel0=15, pixel1=8
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
        assert_eq!(&rgba[4..8], &[136, 136, 136, 255]);
    }

    #[test]
    fn expands_8bit_truecolor() {
        let ihdr = ihdr(1, 1, 8, ColorType::Truecolor);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [10u8, 20, 30];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(rgba, vec![10, 20, 30, 255]);
    }

    #[test]
    fn expands_8bit_truecolor_alpha() {
        let ihdr = ihdr(1, 1, 8, ColorType::TruecolorAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [10u8, 20, 30, 128];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(rgba, vec![10, 20, 30, 128]);
    }

    /// `expand_owned`'s 8-bit `TruecolorAlpha` fast path (`Vec::truncate`
    /// in place, no `to_vec()`) must produce byte-identical output to
    /// `expand`'s own `to_vec()`-based fast path -- this is the whole
    /// point of the method (see its doc comment), so a divergence here
    /// would silently reintroduce the copy this exists to avoid, or worse,
    /// produce wrong pixels.
    #[test]
    fn expand_owned_truecolor_alpha_fast_path_matches_expand() {
        let ihdr = ihdr(2, 2, 8, ColorType::TruecolorAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let via_expand = expander.expand(&pixels).unwrap();
        let via_owned = expander.expand_owned(pixels).unwrap();
        assert_eq!(via_expand, via_owned);
    }

    /// `expand_owned` on any colour type/bit-depth *without* its own
    /// fast path (here: 8-bit `Grayscale`) must delegate to exactly the
    /// same per-pixel logic `expand` uses -- this exercises that fallback
    /// branch specifically, not just the `TruecolorAlpha` fast path above.
    #[test]
    fn expand_owned_falls_back_to_expand_for_non_fast_path_color_types() {
        let ihdr = ihdr(2, 2, 8, ColorType::Grayscale);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = vec![10u8, 20, 30, 40];
        let via_expand = expander.expand(&pixels).unwrap();
        let via_owned = expander.expand_owned(pixels).unwrap();
        assert_eq!(via_expand, via_owned);
    }

    /// `expand_owned` must reject a too-short buffer exactly like `expand`
    /// does, not just take whatever's there and read out of bounds.
    #[test]
    fn expand_owned_rejects_truncated_pixel_buffer() {
        let ihdr = ihdr(2, 2, 8, ColorType::TruecolorAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = vec![0u8; 4]; // needs 16 bytes (2x2x4), only has 4
        let err = expander.expand_owned(pixels).unwrap_err();
        assert!(matches!(err, ExpandError::TruncatedPixels));
    }

    /// `expand_owned` must agree byte-for-byte with `expand` on the same
    /// input for every case `expand` itself is tested against here --
    /// this one specifically exercises the owned fast path's
    /// `Vec::truncate`-instead-of-`to_vec` shortcut at 8-bit
    /// `TruecolorAlpha` (the whole reason `expand_owned` exists), while
    /// the next test below covers the "falls through to `expand`" case
    /// for everything else.
    #[test]
    fn expand_owned_matches_expand_for_8bit_truecolor_alpha_fast_path() {
        let ihdr = ihdr(2, 2, 8, ColorType::TruecolorAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = vec![
            1u8, 2, 3, 4, // row 0, pixel 0
            5, 6, 7, 8, // row 0, pixel 1
            9, 10, 11, 12, // row 1, pixel 0
            13, 14, 15, 16, // row 1, pixel 1
        ];
        let via_borrow = expander.expand(&pixels).unwrap();
        let via_owned = expander.expand_owned(pixels).unwrap();
        assert_eq!(via_owned, via_borrow);
    }

    /// `expand_owned` on a colour type/bit-depth combination that has no
    /// owned-buffer shortcut (falls through to `self.expand(&pixels)`
    /// internally) must still agree with calling `expand` directly.
    #[test]
    fn expand_owned_matches_expand_when_falling_through_to_the_generic_path() {
        let ihdr = ihdr(2, 2, 8, ColorType::Grayscale);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = vec![10u8, 20, 30, 40];
        let via_borrow = expander.expand(&pixels).unwrap();
        let via_owned = expander.expand_owned(pixels).unwrap();
        assert_eq!(via_owned, via_borrow);
    }

    /// `expand_owned` must surface the same errors `expand` would --
    /// taking ownership of `pixels` shouldn't change error behaviour,
    /// just who owns the (unused, on error) input buffer.
    #[test]
    fn expand_owned_propagates_errors_like_expand() {
        let ihdr = ihdr(2, 1, 8, ColorType::TruecolorAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let err = expander.expand_owned(vec![0u8; 4]).unwrap_err();
        assert!(matches!(err, ExpandError::TruncatedPixels));
    }

    /// Exercises the `expand`-level fast path for 8-bit `TruecolorAlpha`
    /// (a straight copy, bypassing `expand_pixel` entirely) across
    /// multiple pixels/rows -- the single-pixel test above wouldn't catch
    /// a stride/row-boundary bug in that fast path, since a 1x1 image's
    /// stride equals its total byte count.
    #[test]
    fn expands_8bit_truecolor_alpha_multi_pixel_matches_per_pixel_path() {
        let ihdr = ihdr(2, 2, 8, ColorType::TruecolorAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        // 2x2 image, 4 bytes/pixel: distinct bytes per pixel/channel so a
        // transposition or off-by-stride bug would show up as a mismatch.
        let pixels = [
            1u8, 2, 3, 4, // row 0, pixel 0
            5, 6, 7, 8, // row 0, pixel 1
            9, 10, 11, 12, // row 1, pixel 0
            13, 14, 15, 16, // row 1, pixel 1
        ];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(rgba, pixels.to_vec());
    }

    /// Exercises the `expand`-level fast path for 8-bit `Grayscale` across
    /// multiple pixels/rows -- a 1x1 image can't distinguish "reads the
    /// right byte" from "reads byte 0 every time", so this uses distinct
    /// values per pixel across 2 rows.
    #[test]
    fn expands_8bit_grayscale_multi_pixel_matches_per_pixel_path() {
        let ihdr = ihdr(2, 2, 8, ColorType::Grayscale);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [10u8, 20, 30, 40]; // row0: 10,20  row1: 30,40
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(
            rgba,
            vec![
                10, 10, 10, 255, // row0 pixel0
                20, 20, 20, 255, // row0 pixel1
                30, 30, 30, 255, // row1 pixel0
                40, 40, 40, 255, // row1 pixel1
            ]
        );
    }

    /// `tRNS` must still be honoured inside the fast path, not just the
    /// generic per-pixel loop -- covers the same scenario as
    /// `trns_makes_matching_grayscale_sample_transparent` but explicitly
    /// through the 8-bit fast-path entry point (both tests hit the same
    /// `expand` call; kept as two tests since either could regress
    /// independently of the other if the fast path's condition ever
    /// changes).
    #[test]
    fn trns_is_honoured_by_the_8bit_grayscale_fast_path() {
        let ihdr = ihdr(2, 1, 8, ColorType::Grayscale);
        let trns = Transparency::parse(ColorType::Grayscale, &[0x00, 0x80]).unwrap(); // sample 128
        let expander = RgbaExpander::new(&ihdr, None, Some(&trns));
        let pixels = [128u8, 200];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(&rgba[0..4], &[128, 128, 128, 0]);
        assert_eq!(&rgba[4..8], &[200, 200, 200, 255]);
    }

    #[test]
    fn expands_8bit_grayscale_alpha() {
        let ihdr = ihdr(1, 1, 8, ColorType::GrayscaleAlpha);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let pixels = [200u8, 50];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(rgba, vec![200, 200, 200, 50]);
    }

    #[test]
    fn expands_16bit_truecolor_keeps_high_byte() {
        let ihdr = ihdr(1, 1, 16, ColorType::Truecolor);
        let expander = RgbaExpander::new(&ihdr, None, None);
        // R = 0xABCD -> high byte 0xAB; G = 0x0102 -> 0x01; B = 0xFFFF -> 0xFF
        let pixels = [0xAB, 0xCD, 0x01, 0x02, 0xFF, 0xFF];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(rgba, vec![0xAB, 0x01, 0xFF, 255]);
    }

    #[test]
    fn expands_indexed_color_via_palette() {
        let ihdr = ihdr(2, 1, 8, ColorType::IndexedColor);
        let palette_data = [10u8, 20, 30, 40, 50, 60]; // 2 entries
        let palette = Palette::parse(&palette_data).unwrap();
        let expander = RgbaExpander::new(&ihdr, Some(&palette), None);
        let pixels = [1u8, 0]; // index 1, then index 0
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(rgba, vec![40, 50, 60, 255, 10, 20, 30, 255]);
    }

    #[test]
    fn indexed_color_without_palette_is_an_error() {
        let ihdr = ihdr(1, 1, 8, ColorType::IndexedColor);
        let expander = RgbaExpander::new(&ihdr, None, None);
        let err = expander.expand(&[0]).unwrap_err();
        assert_eq!(err, ExpandError::MissingPalette);
    }

    #[test]
    fn indexed_color_out_of_range_index_is_an_error() {
        let ihdr = ihdr(1, 1, 8, ColorType::IndexedColor);
        let palette = Palette::parse(&[1, 2, 3]).unwrap(); // 1 entry
        let expander = RgbaExpander::new(&ihdr, Some(&palette), None);
        let err = expander.expand(&[5]).unwrap_err(); // index 5, out of range
        assert!(matches!(err, ExpandError::Palette(_)));
    }

    #[test]
    fn rejects_truncated_pixel_buffer() {
        let ihdr = ihdr(4, 4, 8, ColorType::Grayscale); // needs 16 bytes
        let expander = RgbaExpander::new(&ihdr, None, None);
        let err = expander.expand(&[0u8; 4]).unwrap_err();
        assert_eq!(err, ExpandError::TruncatedPixels);
    }

    #[test]
    fn expands_2bit_indexed_color_with_sub_byte_packing() {
        // 2-bit indexed, width 4: one byte packs all 4 pixels' indices.
        let ihdr = ihdr(4, 1, 2, ColorType::IndexedColor);
        let palette_data = [0u8, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30];
        let palette = Palette::parse(&palette_data).unwrap();
        let expander = RgbaExpander::new(&ihdr, Some(&palette), None);
        let pixels = [0b00_01_10_11u8]; // indices 0,1,2,3
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(
            rgba,
            vec![
                0, 0, 0, 255, 10, 10, 10, 255, 20, 20, 20, 255, 30, 30, 30, 255
            ]
        );
    }

    #[test]
    fn trns_makes_matching_grayscale_sample_transparent() {
        let ihdr = ihdr(2, 1, 8, ColorType::Grayscale);
        let trns = Transparency::parse(ColorType::Grayscale, &[0x00, 0x80]).unwrap(); // sample 128
        let expander = RgbaExpander::new(&ihdr, None, Some(&trns));
        let pixels = [128u8, 200];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(&rgba[0..4], &[128, 128, 128, 0]); // matches -> transparent
        assert_eq!(&rgba[4..8], &[200, 200, 200, 255]); // no match -> opaque
    }

    #[test]
    fn trns_makes_matching_truecolor_pixel_transparent() {
        let ihdr = ihdr(2, 1, 8, ColorType::Truecolor);
        let data = [0x00, 1, 0x00, 2, 0x00, 3]; // transparent color (1,2,3)
        let trns = Transparency::parse(ColorType::Truecolor, &data).unwrap();
        let expander = RgbaExpander::new(&ihdr, None, Some(&trns));
        let pixels = [1u8, 2, 3, 9, 9, 9];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(&rgba[0..4], &[1, 2, 3, 0]);
        assert_eq!(&rgba[4..8], &[9, 9, 9, 255]);
    }

    #[test]
    fn trns_gives_indexed_color_per_entry_alpha() {
        let ihdr = ihdr(3, 1, 8, ColorType::IndexedColor);
        let palette = Palette::parse(&[10, 10, 10, 20, 20, 20, 30, 30, 30]).unwrap();
        let trns = Transparency::parse(ColorType::IndexedColor, &[0, 128]).unwrap(); // index 2: default opaque
        let expander = RgbaExpander::new(&ihdr, Some(&palette), Some(&trns));
        let pixels = [0u8, 1, 2];
        let rgba = expander.expand(&pixels).unwrap();
        assert_eq!(&rgba[0..4], &[10, 10, 10, 0]);
        assert_eq!(&rgba[4..8], &[20, 20, 20, 128]);
        assert_eq!(&rgba[8..12], &[30, 30, 30, 255]); // past tRNS's end: opaque
    }
}
