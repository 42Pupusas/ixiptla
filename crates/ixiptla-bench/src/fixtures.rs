//! Deterministic synthetic PNG file generation for benchmarks.
//!
//! Mirrors `ixiptla-conformance`'s `PixelSource`/`PngFixtureSpec` (same
//! xorshift32 generator, same "encode via the `png` crate" approach) —
//! duplicated rather than shared across crates because conformance's
//! copy lives in a `#[cfg(test)]` module and pulling a whole extra crate
//! dependency edge in just to share ~20 lines of pixel-generation code
//! isn't worth it. Benchmarks want *non-trivial, incompressible-ish*
//! pixel content (a real photo-like image compresses far better — and
//! therefore decodes different code paths at different frequencies —
//! than uniform or all-zero data), so every fixture here is built from
//! the same xorshift32 byte stream.

use png::{BitDepth, ColorType, Encoder};

/// A small deterministic (not truly random, just varied-looking) pixel
/// generator — xorshift32, cheap and reproducible across runs so
/// benchmark numbers are comparable build to build.
struct PixelSource {
    state: u32,
}

impl PixelSource {
    const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    const fn next_byte(&mut self) -> u8 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state & 0xFF) as u8
    }
}

/// A square, non-interlaced, 8-bit fixture at one of the 4 non-indexed
/// colour types PNG defines — [`Self::encode`] hands back complete PNG
/// file bytes, ready to feed to any decoder under benchmark.
///
/// Square + power-of-two-friendly `side` (though any `side` works) keeps
/// the benchmark matrix simple: one number scales both dimensions, and
/// `hotpaths.rs`/`end_to_end.rs` both sweep the same small set of sizes
/// across every colour type without a combinatorial explosion of structs.
#[derive(Debug)]
pub struct SquareFixture {
    pub side: u32,
    pub color_type: ColorType,
}

impl SquareFixture {
    #[must_use]
    pub const fn new(side: u32, color_type: ColorType) -> Self {
        Self { side, color_type }
    }

    /// Samples per pixel for this fixture's colour type (mirrors
    /// `ixiptla_core::ColorType::channel_count`, kept independent since
    /// this crate intentionally doesn't depend on `ixiptla-core`'s
    /// internals to build fixtures — only the `png` crate's own enum).
    const fn channels(&self) -> u32 {
        match self.color_type {
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
            ColorType::Grayscale | ColorType::Indexed => 1,
        }
    }

    /// Encode this fixture as a complete in-memory PNG file, 8-bit depth,
    /// non-interlaced, filled with deterministic pseudo-random pixel
    /// bytes (seeded from `side` and the colour type so every distinct
    /// fixture gets its own byte stream, not an accidental repeat).
    ///
    /// # Panics
    ///
    /// Panics if the `png` crate's encoder itself fails -- never expected
    /// for the well-formed, in-range parameters this struct always
    /// builds.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let channels = self.channels();
        let mut src = PixelSource::new(0x9E37_79B9 ^ (self.side << 8) ^ channels);
        let raw: Vec<u8> = (0..(self.side as usize * self.side as usize * channels as usize))
            .map(|_| src.next_byte())
            .collect();

        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, self.side, self.side);
        encoder.set_color(self.color_type);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&raw).unwrap();
        writer.finish().unwrap();
        out
    }
}

/// The fixed matrix of square sizes every hotpath/end-to-end benchmark
/// sweeps.
///
/// Small enough to keep `cargo bench` runtime reasonable while still
/// covering "tiny" (per-call overhead dominates) through "large enough
/// that the per-pixel inner loop dominates".
pub const BENCH_SIDES: [u32; 4] = [16, 64, 256, 1024];

/// CRC32 of `kind || data`, spec §3.4 -- needed to hand-assemble a raw
/// chunk below (the `png` crate's encoder can't write Adam7-interlaced
/// files at all, same limitation `ixiptla-conformance`'s module docs
/// describe, so Adam7 fixtures here are hand-built rather than going
/// through [`SquareFixture::encode`]).
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

/// Adam7's 7 fixed sub-sampling patterns (spec §8.2, table 8.3):
/// `(x_sampling, x_offset, y_sampling, y_offset)`, one per pass, in pass
/// order -- duplicated from `ixiptla_core::Adam7Pass`'s private constant
/// table on purpose: this crate builds fixture *bytes* independently of
/// `ixiptla_core`'s own Adam7 implementation, so a bug in that
/// implementation can't also corrupt the benchmark input in a way that
/// hides it.
const ADAM7_PASS_CONSTANTS: [(u32, u32, u32, u32); 7] = [
    (8, 0, 8, 0),
    (8, 4, 8, 0),
    (4, 0, 8, 4),
    (4, 2, 4, 0),
    (2, 0, 4, 2),
    (2, 1, 2, 0),
    (1, 0, 2, 1),
];

/// Build a complete `Adam7`-interlaced, 8-bit truecolor (RGB) PNG file of
/// `side x side` pixels.
///
/// Raw chunks assembled by hand (signature/IHDR/IDAT/IEND), pass-major
/// filter-type-0 rows, deterministic pseudo-random pixel content via the
/// same xorshift32 generator [`SquareFixture`] uses.
///
/// # Panics
///
/// Panics if zlib compression itself fails -- never expected for the
/// well-formed input this function always builds.
#[must_use]
pub fn adam7_truecolor_fixture(side: u32) -> Vec<u8> {
    let mut src = PixelSource::new(0xADA7_0000 ^ side);
    let bytes_per_sample = 3usize; // RGB, 8-bit

    let mut filtered = Vec::new();
    for (x_sampling, x_offset, y_sampling, y_offset) in ADAM7_PASS_CONSTANTS {
        let pass_width = side.saturating_sub(x_offset).div_ceil(x_sampling) as usize;
        let pass_height = side.saturating_sub(y_offset).div_ceil(y_sampling) as usize;
        let row_len = pass_width * bytes_per_sample;
        for _ in 0..pass_height {
            filtered.push(0); // filter type None
            filtered.extend((0..row_len).map(|_| src.next_byte()));
        }
    }

    let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&side.to_be_bytes());
    ihdr.extend_from_slice(&side.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // color type: truecolor
    ihdr.extend_from_slice(&[0, 0, 1]); // compression, filter, interlace=Adam7
    write_raw_chunk(&mut file, *b"IHDR", &ihdr);

    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    std::io::Write::write_all(&mut encoder, &filtered).unwrap();
    let compressed = encoder.finish().unwrap();
    write_raw_chunk(&mut file, *b"IDAT", &compressed);
    write_raw_chunk(&mut file, *b"IEND", &[]);
    file
}
