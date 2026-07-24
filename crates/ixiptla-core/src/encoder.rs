use crate::{
    Adam7, ChunkWriter, ColorType, Filter, FilterType, Ihdr, InterlaceMethod, Palette, Signature,
    Transparency,
};

/// Which filter each scanline gets — the single knob that trades encoding
/// speed against output size.
///
/// The filters exist to turn pixel data into small, repetitive residuals
/// that DEFLATE can code cheaply, so this choice moves compression ratio
/// far more than any arithmetic in the filters themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterStrategy {
    /// Score all five filters per row and keep the best (libpng's minimum
    /// sum of absolute differences). Best ratio, ~5x the per-row filtering
    /// work.
    #[default]
    Adaptive,
    /// Apply one fixed filter to every row. `None` is the fastest possible
    /// and leaves DEFLATE to do all the work; `Up` and `Paeth` are common
    /// fixed choices for photographic data.
    Fixed(FilterType),
}

/// Encodes raw pixel bytes into a complete PNG file.
///
/// # Relationship to [`crate::Png`]
///
/// This is deliberately the inverse of decoding at the *same* level:
/// [`crate::Png::decode`] hands back defiltered, de-interlaced scanlines,
/// and this takes exactly that layout back. Pixel-format concerns (RGBA
/// expansion, palette construction, quantisation) stay outside, mirroring
/// how `Png` is not itself "an RGBA image".
///
/// # Why the pieces are separate structs
///
/// Chunk framing lives in [`ChunkWriter`], scanline filtering in
/// [`Filter`], pass geometry in [`Adam7`], and DEFLATE in `ixchel`. This
/// struct only sequences them and owns the buffers they share, so each part
/// stays independently testable and none of them grows a second
/// responsibility.
#[derive(Debug)]
pub struct Encoder {
    ihdr: Ihdr,
    strategy: FilterStrategy,
    palette: Option<Palette>,
    transparency: Option<Transparency>,
}

impl Encoder {
    /// Start an encoder for an image with `ihdr`'s geometry, filtering
    /// adaptively.
    #[must_use]
    pub const fn new(ihdr: Ihdr) -> Self {
        Self {
            ihdr,
            strategy: FilterStrategy::Adaptive,
            palette: None,
            transparency: None,
        }
    }

    /// Choose how scanline filters are picked.
    #[must_use]
    pub const fn with_filter_strategy(mut self, strategy: FilterStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Attach a `PLTE` chunk. Required for indexed-colour images; allowed
    /// (as a suggested palette) alongside truecolour.
    #[must_use]
    pub fn with_palette(mut self, palette: Palette) -> Self {
        self.palette = Some(palette);
        self
    }

    /// Attach a `tRNS` chunk (simple transparency for colour types with no
    /// alpha channel).
    #[must_use]
    pub fn with_transparency(mut self, transparency: Transparency) -> Self {
        self.transparency = Some(transparency);
        self
    }

    /// Encode `pixels` — `height` scanlines of exactly [`Ihdr::stride`]
    /// bytes, with no filter-type bytes and no interlacing applied — into a
    /// complete PNG file.
    ///
    /// # Errors
    ///
    /// [`EncodeError::WrongPixelBufferSize`] if `pixels` isn't exactly
    /// `height * stride` bytes, or [`EncodeError::MissingPalette`] for an
    /// indexed-colour image with no `PLTE` attached (spec §11.2.3 makes it
    /// mandatory, and our own decoder rejects such a file).
    pub fn encode(&self, pixels: &[u8]) -> Result<Vec<u8>, EncodeError> {
        let expected = self
            .ihdr
            .stride()
            .checked_mul(self.ihdr.height as usize)
            .ok_or(EncodeError::WrongPixelBufferSize {
                expected: usize::MAX,
                actual: pixels.len(),
            })?;
        if pixels.len() != expected {
            return Err(EncodeError::WrongPixelBufferSize {
                expected,
                actual: pixels.len(),
            });
        }
        if self.ihdr.color_type == ColorType::IndexedColor && self.palette.is_none() {
            return Err(EncodeError::MissingPalette);
        }

        let filtered = self.filtered_stream(pixels);
        let compressed = ixchel::zlib::compress(&filtered);

        // Signature + IHDR + optional PLTE/tRNS + IDAT + IEND. Sizing this
        // up front keeps encoding to a single output allocation.
        let mut out = Vec::with_capacity(
            Signature::BYTES.len() + compressed.len() + 4 * ChunkWriter::OVERHEAD + Ihdr::DATA_LEN,
        );
        out.extend_from_slice(&Signature::BYTES);

        let mut writer = ChunkWriter::new(&mut out);
        writer.write(*b"IHDR", &self.ihdr.to_bytes());
        if let Some(palette) = &self.palette {
            writer.write(*b"PLTE", &palette.to_bytes());
        }
        if let Some(transparency) = &self.transparency {
            writer.write(*b"tRNS", &transparency.to_bytes());
        }
        writer.write(*b"IDAT", &compressed);
        writer.write(*b"IEND", &[]);
        Ok(out)
    }

    /// Build the pre-compression byte stream: every scanline prefixed by
    /// its filter-type byte, in the order `IDAT` stores them.
    fn filtered_stream(&self, pixels: &[u8]) -> Vec<u8> {
        match self.ihdr.interlace {
            InterlaceMethod::None => {
                let stride = self.ihdr.stride();
                // Each row grows by exactly one filter-type byte.
                let mut out = Vec::with_capacity(pixels.len() + self.ihdr.height as usize);
                self.filter_scanlines(&self.ihdr, pixels, stride, &mut out);
                out
            }
            InterlaceMethod::Adam7 => {
                let adam7 = Adam7::new(&self.ihdr);
                let mut out = Vec::with_capacity(pixels.len() + self.ihdr.height as usize);
                // Spec §8.2: each pass is filtered as its own independent
                // image -- predictor state does not carry across a pass
                // boundary, so each gets a fresh all-zero previous row.
                for pass in adam7.passes() {
                    if pass.is_empty() {
                        continue;
                    }
                    let pass_pixels = adam7.gather_pass(&pass, pixels);
                    let pass_ihdr = Ihdr {
                        width: pass.width,
                        height: pass.height,
                        ..self.ihdr
                    };
                    let pass_stride = adam7.pass_stride(&pass);
                    self.filter_scanlines(&pass_ihdr, &pass_pixels, pass_stride, &mut out);
                }
                out
            }
        }
    }

    /// Filter consecutive scanlines of `stride` bytes from `pixels` into
    /// `out`, threading each row through as the next row's predictor input.
    fn filter_scanlines(&self, ihdr: &Ihdr, pixels: &[u8], stride: usize, out: &mut Vec<u8>) {
        let filter = Filter::new(ihdr);
        // Row 0's "row above" is all zeroes (spec §9.2), and reusing one
        // buffer avoids allocating per row.
        let mut previous = vec![0u8; stride];

        for row in pixels.chunks_exact(stride) {
            match self.strategy {
                FilterStrategy::Adaptive => {
                    filter.filter_row_adaptive_into(row, &previous, out);
                }
                FilterStrategy::Fixed(filter_type) => {
                    filter.filter_row_into(filter_type, row, &previous, out);
                }
            }
            // The *unfiltered* row is what the next row predicts against.
            previous.copy_from_slice(row);
        }
    }
}

/// Why [`Encoder::encode`] rejected an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// `pixels` wasn't exactly `height * stride` bytes.
    WrongPixelBufferSize { expected: usize, actual: usize },
    /// An indexed-colour image was encoded without a `PLTE` chunk.
    MissingPalette,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongPixelBufferSize { expected, actual } => write!(
                f,
                "pixel buffer is {actual} bytes, expected {expected} (height * stride)"
            ),
            Self::MissingPalette => {
                write!(f, "indexed-color image requires a PLTE chunk")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

#[cfg(test)]
mod tests {
    use super::{EncodeError, Encoder, FilterStrategy};
    use crate::{
        ColorType, FilterType, Ihdr, InterlaceMethod, Palette, Png, Signature, Transparency,
    };

    /// Builds an image of a given geometry and checks it survives an
    /// encode/decode round-trip.
    struct RoundTrip {
        ihdr: Ihdr,
    }

    impl RoundTrip {
        fn new(
            width: u32,
            height: u32,
            color_type: ColorType,
            bit_depth: u8,
            interlace: InterlaceMethod,
        ) -> Self {
            Self {
                ihdr: Ihdr {
                    width,
                    height,
                    bit_depth,
                    color_type,
                    interlace,
                },
            }
        }

        /// Deterministic pseudo-random pixel bytes for this geometry, with
        /// any trailing padding bits cleared.
        ///
        /// At sub-byte depths a scanline's last byte can hold bits beyond
        /// the final pixel (a 9-pixel 1-bit row occupies 2 bytes, of which
        /// 7 bits are padding). Those bits carry no pixel, so nothing is
        /// obliged to preserve them -- Adam7 gathering in particular walks
        /// pixels, not bytes, and legitimately returns them as zero. Real
        /// encoder input has them zeroed, so the fixture does too rather
        /// than asserting on bits that have no defined value.
        fn pixels(&self) -> Vec<u8> {
            let stride = self.ihdr.stride();
            let used_bits = self.ihdr.bits_per_pixel() as usize * self.ihdr.width as usize;
            let mut pixels: Vec<u8> = (0..stride * self.ihdr.height as usize)
                .map(|i| u8::try_from((i * 31 + 7) % 256).expect("masked into u8 range"))
                .collect();

            for row in pixels.chunks_exact_mut(stride) {
                for bit in used_bits..stride * 8 {
                    row[bit / 8] &= !(1u8 << (7 - bit % 8));
                }
            }
            pixels
        }

        /// Encode with `strategy`, decode with our own decoder, and assert
        /// the header and every pixel byte came back intact.
        fn check(&self, strategy: FilterStrategy) -> Vec<u8> {
            let pixels = self.pixels();
            let encoder = Encoder::new(self.ihdr).with_filter_strategy(strategy);
            let file = encoder.encode(&pixels).expect("encode should succeed");

            let decoded = Png::decode(&file, 1 << 24).expect("our own decoder must accept it");
            assert_eq!(decoded.ihdr, self.ihdr, "header mismatch for {strategy:?}");
            assert_eq!(
                decoded.pixels, pixels,
                "pixel mismatch for {strategy:?} at {:?}",
                self.ihdr
            );
            file
        }
    }

    #[test]
    fn round_trips_every_color_type_and_bit_depth() {
        // The core property: anything we emit, our own decoder reproduces
        // byte-for-byte. Widths are deliberately not multiples of 8 so
        // sub-byte depths exercise row padding.
        let cases = [
            (ColorType::Grayscale, 1u8),
            (ColorType::Grayscale, 2),
            (ColorType::Grayscale, 4),
            (ColorType::Grayscale, 8),
            (ColorType::Grayscale, 16),
            (ColorType::Truecolor, 8),
            (ColorType::Truecolor, 16),
            (ColorType::GrayscaleAlpha, 8),
            (ColorType::GrayscaleAlpha, 16),
            (ColorType::TruecolorAlpha, 8),
            (ColorType::TruecolorAlpha, 16),
        ];
        for (color_type, bit_depth) in cases {
            RoundTrip::new(13, 7, color_type, bit_depth, InterlaceMethod::None)
                .check(FilterStrategy::Adaptive);
        }
    }

    #[test]
    fn round_trips_every_fixed_filter() {
        // Each filter must be individually correct, not just whichever the
        // adaptive heuristic happens to pick.
        for filter_type in FilterType::ALL {
            RoundTrip::new(16, 5, ColorType::TruecolorAlpha, 8, InterlaceMethod::None)
                .check(FilterStrategy::Fixed(filter_type));
            RoundTrip::new(16, 5, ColorType::Grayscale, 8, InterlaceMethod::None)
                .check(FilterStrategy::Fixed(filter_type));
        }
    }

    #[test]
    fn round_trips_adam7_interlaced_images() {
        // Interlacing splits the image into 7 independently-filtered
        // sub-images, so this exercises a different path end to end.
        for (color_type, bit_depth) in [
            (ColorType::Grayscale, 8u8),
            (ColorType::Truecolor, 8),
            (ColorType::TruecolorAlpha, 16),
            (ColorType::Grayscale, 1),
        ] {
            RoundTrip::new(9, 7, color_type, bit_depth, InterlaceMethod::Adam7)
                .check(FilterStrategy::Adaptive);
        }
    }

    #[test]
    fn round_trips_images_small_enough_to_have_empty_adam7_passes() {
        // A 1x1 image places its only pixel in pass 1 and leaves the other
        // six empty -- they must contribute no scanlines at all.
        for size in [1u32, 2, 3, 5] {
            RoundTrip::new(size, size, ColorType::Truecolor, 8, InterlaceMethod::Adam7)
                .check(FilterStrategy::Adaptive);
        }
    }

    #[test]
    fn round_trips_single_row_and_single_column_images() {
        RoundTrip::new(1, 64, ColorType::TruecolorAlpha, 8, InterlaceMethod::None)
            .check(FilterStrategy::Adaptive);
        RoundTrip::new(64, 1, ColorType::TruecolorAlpha, 8, InterlaceMethod::None)
            .check(FilterStrategy::Adaptive);
    }

    #[test]
    fn output_starts_with_the_signature_and_ends_with_iend() {
        let file = RoundTrip::new(4, 4, ColorType::Truecolor, 8, InterlaceMethod::None)
            .check(FilterStrategy::Adaptive);
        assert_eq!(&file[..8], &Signature::BYTES);
        // IEND is always the last chunk: 12 bytes of framing, empty payload.
        assert_eq!(&file[file.len() - 8..file.len() - 4], b"IEND");
    }

    #[test]
    fn round_trips_an_indexed_color_image_with_its_palette() {
        let ihdr = Ihdr {
            width: 7,
            height: 3,
            bit_depth: 4,
            color_type: ColorType::IndexedColor,
            interlace: InterlaceMethod::None,
        };
        // 4-bit indices, so every nibble must be a valid entry: 16 entries
        // covers the full range.
        let palette_bytes: Vec<u8> = (0..16u8).flat_map(|i| [i * 16, 255 - i * 16, i]).collect();
        let palette = Palette::parse(&palette_bytes).unwrap();
        let pixels: Vec<u8> = (0..ihdr.stride() * ihdr.height as usize)
            .map(|i| u8::try_from((i * 37 + 3) % 256).unwrap())
            .collect();

        let file = Encoder::new(ihdr)
            .with_palette(palette.clone())
            .encode(&pixels)
            .unwrap();
        let decoded = Png::decode(&file, 1 << 24).unwrap();
        assert_eq!(decoded.pixels, pixels);
        assert_eq!(decoded.palette, Some(palette));
    }

    #[test]
    fn round_trips_a_transparency_chunk() {
        let ihdr = Ihdr {
            width: 8,
            height: 2,
            bit_depth: 8,
            color_type: ColorType::Grayscale,
            interlace: InterlaceMethod::None,
        };
        let pixels = vec![7u8; ihdr.stride() * ihdr.height as usize];
        let file = Encoder::new(ihdr)
            .with_transparency(Transparency::Grayscale(7))
            .encode(&pixels)
            .unwrap();

        let decoded = Png::decode(&file, 1 << 24).unwrap();
        assert_eq!(decoded.transparency, Some(Transparency::Grayscale(7)));
    }

    #[test]
    fn rejects_a_pixel_buffer_of_the_wrong_size() {
        let ihdr = Ihdr {
            width: 4,
            height: 4,
            bit_depth: 8,
            color_type: ColorType::Truecolor,
            interlace: InterlaceMethod::None,
        };
        let error = Encoder::new(ihdr).encode(&[0; 47]).unwrap_err();
        assert_eq!(
            error,
            EncodeError::WrongPixelBufferSize {
                expected: 48,
                actual: 47
            }
        );
    }

    #[test]
    fn rejects_indexed_color_without_a_palette() {
        let ihdr = Ihdr {
            width: 4,
            height: 4,
            bit_depth: 8,
            color_type: ColorType::IndexedColor,
            interlace: InterlaceMethod::None,
        };
        assert_eq!(
            Encoder::new(ihdr).encode(&[0; 16]).unwrap_err(),
            EncodeError::MissingPalette
        );
    }

    #[test]
    fn adaptive_filtering_is_never_larger_than_no_filtering() {
        // The heuristic's entire justification. A vertical gradient is the
        // easy case (Up predicts it perfectly), so if adaptive ever loses
        // here the scoring is wired up wrong.
        let ihdr = Ihdr {
            width: 64,
            height: 64,
            bit_depth: 8,
            color_type: ColorType::Truecolor,
            interlace: InterlaceMethod::None,
        };
        let stride = ihdr.stride();
        let pixels: Vec<u8> = (0..stride * ihdr.height as usize)
            .map(|i| u8::try_from((i / stride) % 256).unwrap())
            .collect();

        let adaptive = Encoder::new(ihdr).encode(&pixels).unwrap().len();
        let unfiltered = Encoder::new(ihdr)
            .with_filter_strategy(FilterStrategy::Fixed(FilterType::None))
            .encode(&pixels)
            .unwrap()
            .len();
        assert!(
            adaptive < unfiltered,
            "adaptive {adaptive} should beat unfiltered {unfiltered} on a vertical gradient"
        );
    }
}
