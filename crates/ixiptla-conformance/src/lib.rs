//! `ixiptla-conformance` — differential ("diff") testing of
//! `ixiptla-core` against the battle-tested [`png`] crate (`image-rs`'s
//! `image-png`, used by GNOME/Chromium and fuzzed on OSS-Fuzz).
//!
//! `ixiptla-core`'s own tests prove its decoder agrees with *itself* — the
//! same hand-rolled understanding of the PNG spec used to build test
//! fixtures is used to check them. That can't catch a shared
//! misunderstanding of the spec. This crate instead builds PNG files with
//! `png`'s encoder and asserts `ixiptla_core::Png::decode` +
//! `RgbaExpander::expand` produces byte-identical RGBA8 output to `png`'s
//! own decoder (with `Transformations::EXPAND | ALPHA | STRIP_16`, so both
//! sides normalize to one comparable representation) for the same file.
//!
//! Kept as its own crate (`dev-dependencies` only, `publish = false`)
//! rather than a `[dev-dependencies]` entry inside `ixiptla-core` itself,
//! so pulling in a whole other PNG implementation for testing purposes
//! never taints `ixiptla-core`'s own zero-dependency build.
//!
//! `png` 0.18's own `Writer::write_image_data` does not implement Adam7
//! interlaced *encoding* (only decoding) — giving it `Info::interlaced =
//! true` either silently mis-writes the file or (newer versions) rejects
//! it outright. So Adam7 fixtures here are hand-assembled: raw chunks
//! written directly (signature/IHDR/PLTE/IDAT/IEND, CRC via `crc32fast`,
//! `IDAT` payload zlib-compressed via `flate2`), decoded solely by both
//! `ixiptla_core::Png::decode` and the `png` crate's *decoder* (which does
//! support Adam7) for comparison — the goal is diffing decode behavior,
//! not exercising `png`'s encoder for this one case.
//!
//! The shared "decode both ways and diff the RGBA8 bytes" logic lives in
//! [`reference_decoder::ReferenceComparison`] rather than only inside
//! `#[cfg(test)]` here, so `tests/pngsuite.rs` — a *separate* test binary
//! that only links this crate's compiled (non-`#[cfg(test)]`) lib code —
//! can reach it too, to diff-test against the real-world `PngSuite`
//! corpus alongside the synthetic fixtures built below.

//! The reverse direction is covered too: [`encoder_conformance`] hands
//! files produced by `ixiptla_core::Encoder` to the `png` crate's decoder,
//! so our *output* is validated by an independent implementation rather
//! than only by our own decoder (which could share a misreading with it).

pub mod encoder_conformance;
pub mod reference_decoder;

pub use encoder_conformance::EncoderConformance;
pub use reference_decoder::ReferenceComparison;

#[cfg(test)]
mod encoder_tests {
    use crate::EncoderConformance;
    use ixiptla_core::{ColorType, FilterStrategy, FilterType, InterlaceMethod};

    #[test]
    fn png_crate_accepts_every_color_type_and_bit_depth_we_encode() {
        // Width 13 is deliberately not a multiple of 8, so sub-byte depths
        // exercise scanline padding.
        for (color_type, bit_depth) in [
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
        ] {
            EncoderConformance::new(13, 7, color_type, bit_depth, InterlaceMethod::None)
                .assert_matches(FilterStrategy::Adaptive);
        }
    }

    #[test]
    fn png_crate_accepts_every_fixed_filter_we_emit() {
        // Each filter must be independently conformant, not just whichever
        // one the adaptive heuristic happens to choose.
        for filter_type in FilterType::ALL {
            for (color_type, bit_depth) in [
                (ColorType::Grayscale, 8u8),
                (ColorType::TruecolorAlpha, 8),
                (ColorType::TruecolorAlpha, 16),
            ] {
                EncoderConformance::new(17, 5, color_type, bit_depth, InterlaceMethod::None)
                    .assert_matches(FilterStrategy::Fixed(filter_type));
            }
        }
    }

    #[test]
    fn png_crate_accepts_our_adam7_interlaced_output() {
        // The `png` crate cannot *write* Adam7, but it does decode it --
        // which is exactly the direction needed to check our pass layout
        // against an independent implementation.
        for (color_type, bit_depth) in [
            (ColorType::Grayscale, 1u8),
            (ColorType::Grayscale, 8),
            (ColorType::Truecolor, 8),
            (ColorType::TruecolorAlpha, 8),
            (ColorType::TruecolorAlpha, 16),
        ] {
            EncoderConformance::new(9, 7, color_type, bit_depth, InterlaceMethod::Adam7)
                .assert_matches(FilterStrategy::Adaptive);
        }
    }

    #[test]
    fn png_crate_accepts_awkward_image_dimensions() {
        // 1-pixel rows/columns, and sizes small enough that most Adam7
        // passes are empty.
        for (width, height) in [(1, 1), (1, 9), (9, 1), (2, 3), (3, 2), (5, 5)] {
            EncoderConformance::new(width, height, ColorType::Truecolor, 8, InterlaceMethod::None)
                .assert_matches(FilterStrategy::Adaptive);
            EncoderConformance::new(width, height, ColorType::Truecolor, 8, InterlaceMethod::Adam7)
                .assert_matches(FilterStrategy::Adaptive);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::reference_decoder::ReferenceComparison;
    use png::{BitDepth, ColorType as PngColorType, Encoder, Info};
    use std::io::Write as _;

    /// A small deterministic (not truly random, just varied-looking)
    /// pixel generator, so every test image has non-trivial content —
    /// distinguishing real decode bugs from "an all-zero buffer matched
    /// trivially" false confidence.
    struct PixelSource {
        state: u32,
    }

    impl PixelSource {
        const fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        /// xorshift32: cheap, deterministic, good enough spread for test
        /// fixtures (not for anything cryptographic).
        fn next_byte(&mut self) -> u8 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 17;
            self.state ^= self.state << 5;
            (self.state & 0xFF) as u8
        }
    }

    /// The subset of a non-interlaced PNG's metadata this suite needs to
    /// set before writing pixel data, bundled into one struct so
    /// `encode_with_png_crate` takes one argument instead of many (clippy's
    /// `too_many_arguments` threshold is 7; a bare parameter list here
    /// would blow past it, and grouping these fields also just reads more
    /// like "here's a PNG's header" than a wall of positional bools).
    struct PngFixtureSpec<'a> {
        width: u32,
        height: u32,
        color_type: PngColorType,
        bit_depth: BitDepth,
        palette: Option<&'a [u8]>,
        trns: Option<&'a [u8]>,
    }

    /// Encode `raw` (already in `spec.color_type`/`spec.bit_depth`'s
    /// packed sample layout) as a complete in-memory non-interlaced PNG
    /// file via the `png` crate's encoder, optionally carrying a
    /// palette/`tRNS`. (Adam7-interlaced fixtures use
    /// [`build_adam7_png`] instead — `png` 0.18's encoder can't write
    /// interlaced files; see module docs.)
    fn encode_with_png_crate(spec: &PngFixtureSpec<'_>, raw: &[u8]) -> Vec<u8> {
        let mut info = Info::with_size(spec.width, spec.height);
        info.color_type = spec.color_type;
        info.bit_depth = spec.bit_depth;
        if let Some(p) = spec.palette {
            info.palette = Some(p.to_vec().into());
        }
        if let Some(t) = spec.trns {
            info.trns = Some(t.to_vec().into());
        }

        let mut out = Vec::new();
        let encoder = Encoder::with_info(&mut out, info).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(raw).unwrap();
        writer.finish().unwrap();
        out
    }

    /// Generate `width * height * samples_per_pixel` raw sample bytes at
    /// 8-bit depth (every color type this suite exercises at 8 bits uses
    /// one byte per sample, packed, no sub-byte cases here — those are
    /// covered separately for grayscale).
    fn random_raw_8bit(width: u32, height: u32, samples_per_pixel: usize, seed: u32) -> Vec<u8> {
        let mut src = PixelSource::new(seed);
        let n = width as usize * height as usize * samples_per_pixel;
        (0..n).map(|_| src.next_byte()).collect()
    }

    /// Adam7's 7 fixed sub-sampling patterns (spec §8.2, table 8.3):
    /// `(x_sampling, x_offset, y_sampling, y_offset)`, one per pass, in
    /// pass order.
    const ADAM7_PASS_CONSTANTS: [(u32, u32, u32, u32); 7] = [
        (8, 0, 8, 0),
        (8, 4, 8, 0),
        (4, 0, 8, 4),
        (4, 2, 4, 0),
        (2, 0, 4, 2),
        (2, 1, 2, 0),
        (1, 0, 2, 1),
    ];

    /// Generate raw (unfiltered) Adam7-interlaced sample bytes for an
    /// image of `width` x `height`, `bytes_per_sample`-wide samples, in
    /// the pass-major order both `ixiptla_core::Png::decode` and the
    /// `png` crate's `Writer::write_image_data` (when `Info::interlaced`
    /// is set) expect: pass 1's rows top-to-bottom, then pass 2's, ...,
    /// pass 7's — each pass being its own tiny sub-sampled image (spec
    /// §8.2), not the full-size raster.
    fn random_raw_adam7(width: u32, height: u32, bytes_per_sample: usize, seed: u32) -> Vec<u8> {
        let mut src = PixelSource::new(seed);
        let mut out = Vec::new();
        for (x_sampling, x_offset, y_sampling, y_offset) in ADAM7_PASS_CONSTANTS {
            let pass_width = width.saturating_sub(x_offset).div_ceil(x_sampling);
            let pass_height = height.saturating_sub(y_offset).div_ceil(y_sampling);
            let n = pass_width as usize * pass_height as usize * bytes_per_sample;
            out.extend((0..n).map(|_| src.next_byte()));
        }
        out
    }

    fn crc32(kind: [u8; 4], data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&kind);
        hasher.update(data);
        hasher.finalize()
    }

    fn write_raw_chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
        out.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
        out.extend_from_slice(&kind);
        out.extend_from_slice(data);
        out.extend_from_slice(&crc32(kind, data).to_be_bytes());
    }

    /// Hand-assemble a complete Adam7-interlaced PNG file: signature,
    /// `IHDR` (`interlace_method = 1`), optional `PLTE`, a single `IDAT`
    /// carrying `random_raw_adam7`-shaped raw bytes with a filter-byte-0
    /// (identity) prefix on every row of every pass, `IEND`. Needed
    /// because `png` 0.18's encoder doesn't support writing Adam7 (see
    /// module docs) — only used to build fixtures, decoding still goes
    /// through both real decoders for the actual comparison.
    fn build_adam7_png(
        width: u32,
        height: u32,
        color_type: PngColorType,
        palette: Option<&[u8]>,
        raw_per_pass_sample_bytes: &[u8],
        bytes_per_sample: usize,
    ) -> Vec<u8> {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(8); // bit depth
        ihdr.push(color_type as u8);
        ihdr.extend_from_slice(&[0, 0, 1]); // compression, filter, interlace=Adam7
        write_raw_chunk(&mut file, *b"IHDR", &ihdr);

        if let Some(p) = palette {
            write_raw_chunk(&mut file, *b"PLTE", p);
        }

        // Insert a filter-byte-0 prefix ahead of every row of every pass,
        // matching how random_raw_adam7 laid out its pass-major samples.
        let mut filtered = Vec::new();
        let mut offset = 0usize;
        for (x_sampling, x_offset, y_sampling, y_offset) in ADAM7_PASS_CONSTANTS {
            let pass_width = width.saturating_sub(x_offset).div_ceil(x_sampling) as usize;
            let pass_height = height.saturating_sub(y_offset).div_ceil(y_sampling) as usize;
            let row_len = pass_width * bytes_per_sample;
            for _ in 0..pass_height {
                filtered.push(0); // filter type None
                filtered.extend_from_slice(&raw_per_pass_sample_bytes[offset..offset + row_len]);
                offset += row_len;
            }
        }

        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&filtered).unwrap();
        let compressed = encoder.finish().unwrap();
        write_raw_chunk(&mut file, *b"IDAT", &compressed);
        write_raw_chunk(&mut file, *b"IEND", &[]);
        file
    }

    #[test]
    fn matches_reference_for_8bit_grayscale() {
        let raw = random_raw_8bit(9, 7, 1, 0x00C0_FFEE);
        let spec = PngFixtureSpec {
            width: 9,
            height: 7,
            color_type: PngColorType::Grayscale,
            bit_depth: BitDepth::Eight,
            palette: None,
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_8bit_grayscale_alpha() {
        let raw = random_raw_8bit(6, 5, 2, 0x00BE_EF01);
        let spec = PngFixtureSpec {
            width: 6,
            height: 5,
            color_type: PngColorType::GrayscaleAlpha,
            bit_depth: BitDepth::Eight,
            palette: None,
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_8bit_truecolor() {
        let raw = random_raw_8bit(11, 4, 3, 0x00AB_CD12);
        let spec = PngFixtureSpec {
            width: 11,
            height: 4,
            color_type: PngColorType::Rgb,
            bit_depth: BitDepth::Eight,
            palette: None,
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_8bit_truecolor_alpha() {
        let raw = random_raw_8bit(8, 8, 4, 0x0013_57AC);
        let spec = PngFixtureSpec {
            width: 8,
            height: 8,
            color_type: PngColorType::Rgba,
            bit_depth: BitDepth::Eight,
            palette: None,
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_indexed_color_with_palette() {
        // 4-entry palette, 8-bit indices into it.
        let palette = [10u8, 20, 30, 200, 100, 50, 0, 255, 0, 255, 0, 255];
        let mut src = PixelSource::new(0x00DE_AD01);
        let raw: Vec<u8> = (0..(5 * 5)).map(|_| src.next_byte() % 4).collect();
        let spec = PngFixtureSpec {
            width: 5,
            height: 5,
            color_type: PngColorType::Indexed,
            bit_depth: BitDepth::Eight,
            palette: Some(&palette),
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_indexed_color_with_trns() {
        let palette = [10u8, 20, 30, 200, 100, 50, 0, 255, 0, 255, 0, 255];
        let trns = [0u8, 255]; // index 0 fully transparent, index 1 opaque
        let mut src = PixelSource::new(0x00FE_ED02);
        let raw: Vec<u8> = (0..(6 * 6)).map(|_| src.next_byte() % 4).collect();
        let spec = PngFixtureSpec {
            width: 6,
            height: 6,
            color_type: PngColorType::Indexed,
            bit_depth: BitDepth::Eight,
            palette: Some(&palette),
            trns: Some(&trns),
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_grayscale_trns() {
        let mut src = PixelSource::new(0x000A_0B0C);
        let raw: Vec<u8> = (0..(5 * 5)).map(|_| src.next_byte()).collect();
        // Pick one value guaranteed to be present so the transparency path
        // actually gets exercised, not just parsed-and-unused.
        let transparent_value = raw[0];
        let trns = [0u8, transparent_value];
        let spec = PngFixtureSpec {
            width: 5,
            height: 5,
            color_type: PngColorType::Grayscale,
            bit_depth: BitDepth::Eight,
            palette: None,
            trns: Some(&trns),
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_truecolor_trns() {
        let mut src = PixelSource::new(0x0011_2233);
        let raw: Vec<u8> = (0..(5 * 5 * 3)).map(|_| src.next_byte()).collect();
        let (r, g, b) = (raw[0], raw[1], raw[2]);
        let trns = [0u8, r, 0, g, 0, b];
        let spec = PngFixtureSpec {
            width: 5,
            height: 5,
            color_type: PngColorType::Rgb,
            bit_depth: BitDepth::Eight,
            palette: None,
            trns: Some(&trns),
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_adam7_interlaced_truecolor() {
        let raw = random_raw_adam7(13, 11, 3, 0x009E_9E9E);
        let file = build_adam7_png(13, 11, PngColorType::Rgb, None, &raw, 3);
        ReferenceComparison::assert_matches(&file);
    }

    #[test]
    fn matches_reference_for_adam7_interlaced_indexed_color() {
        let palette = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        // Indexed color is 1 byte/sample, so reuse random_raw_adam7's
        // pass-shaped layout, then fold values into the 3-entry palette's
        // valid index range.
        let raw: Vec<u8> = random_raw_adam7(13, 11, 1, 0x0042_4242)
            .into_iter()
            .map(|b| b % 3)
            .collect();
        let file = build_adam7_png(13, 11, PngColorType::Indexed, Some(&palette), &raw, 1);
        ReferenceComparison::assert_matches(&file);
    }

    #[test]
    fn matches_reference_for_1bit_grayscale() {
        // 1-bit grayscale, width 10 (spans multiple bytes per row, non-
        // byte-aligned width so padding bits matter): 2 bytes/row.
        let mut src = PixelSource::new(0x1BE0);
        let raw: Vec<u8> = (0..(3 * 2)).map(|_| src.next_byte()).collect();
        let spec = PngFixtureSpec {
            width: 10,
            height: 3,
            color_type: PngColorType::Grayscale,
            bit_depth: BitDepth::One,
            palette: None,
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_4bit_indexed_color() {
        let mut src = PixelSource::new(0x4B17);
        let palette: Vec<u8> = (0..16 * 3).map(|_| src.next_byte()).collect();
        // 4-bit indexed, width 5 -> 3 bytes/row (5 nibbles rounds up to 6, /2 = 3).
        let raw: Vec<u8> = (0..(4 * 3)).map(|_| src.next_byte()).collect();
        let spec = PngFixtureSpec {
            width: 5,
            height: 4,
            color_type: PngColorType::Indexed,
            bit_depth: BitDepth::Four,
            palette: Some(&palette),
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }

    #[test]
    fn matches_reference_for_16bit_truecolor_alpha() {
        let mut src = PixelSource::new(0x0016_0016);
        let raw: Vec<u8> = (0..(4 * 3 * 4 * 2)).map(|_| src.next_byte()).collect();
        let spec = PngFixtureSpec {
            width: 4,
            height: 3,
            color_type: PngColorType::Rgba,
            bit_depth: BitDepth::Sixteen,
            palette: None,
            trns: None,
        };
        ReferenceComparison::assert_matches(&encode_with_png_crate(&spec, &raw));
    }
}
