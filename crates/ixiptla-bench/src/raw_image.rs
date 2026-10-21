//! Raw pixel buffers for *encoding* benchmarks.
//!
//! Decode benchmarks start from PNG file bytes; encode benchmarks start
//! from the other end, so they need raw scanlines instead.
//!
//! # Why the content matters more here than for decoding
//!
//! `fixtures.rs` fills its images with xorshift noise, which is the right
//! call for decoding: it defeats compression, so the decoder does maximum
//! work per byte. For *encoding* that choice would be actively misleading.
//! Noise is incompressible, so every encoder emits roughly the input size
//! no matter how good its filtering is — a ratio comparison on noise
//! measures nothing, and a speed comparison on noise exercises DEFLATE's
//! worst case rather than the one real images hit.
//!
//! So this module offers both, and benchmarks are explicit about which
//! they use: [`Content::Photographic`] for anything comparing output size,
//! [`Content::Noise`] to keep the compressor honest about its worst case.

use ixiptla_png::{ColorType, Ihdr, InterlaceMethod};

/// The pixel content a benchmark image is filled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Content {
    /// Smooth gradients with gentle structure — the spatial correlation
    /// real photographs have, and the thing PNG's filters exist to
    /// exploit. Use this whenever output size is being compared.
    Photographic,
    /// Xorshift noise: incompressible, so it measures the compressor's
    /// worst case and makes ratio comparisons meaningless.
    Noise,
}

/// A raw image ready to hand to an encoder: header geometry plus the
/// packed scanline bytes `Encoder::encode` expects.
#[derive(Debug)]
pub struct RawImage {
    pub ihdr: Ihdr,
    pub pixels: Vec<u8>,
}

impl RawImage {
    /// Build a `side` x `side` image of `color_type` at 8 bits per sample.
    #[must_use]
    pub fn square(side: u32, color_type: ColorType, content: Content) -> Self {
        Self::new(side, side, color_type, 8, InterlaceMethod::None, content)
    }

    /// Build an image of arbitrary geometry.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        color_type: ColorType,
        bit_depth: u8,
        interlace: InterlaceMethod,
        content: Content,
    ) -> Self {
        let ihdr = Ihdr {
            width,
            height,
            bit_depth,
            color_type,
            interlace,
        };
        let pixels = match content {
            Content::Photographic => Self::photographic(&ihdr),
            Content::Noise => Self::noise(&ihdr),
        };
        Self { ihdr, pixels }
    }

    /// Samples per pixel, from the colour type.
    const fn channels(ihdr: &Ihdr) -> usize {
        ihdr.color_type.channel_count() as usize
    }

    /// Smooth, spatially-correlated content: each channel is a slowly
    /// varying function of position with a little high-frequency detail on
    /// top, so neighbouring pixels resemble each other both horizontally
    /// and vertically. That is precisely the redundancy PNG's Sub/Up/Paeth
    /// predictors are designed to remove, so filter choice actually shows
    /// up in the output size.
    fn photographic(ihdr: &Ihdr) -> Vec<u8> {
        let channels = Self::channels(ihdr);
        let stride = ihdr.stride();
        let mut pixels = vec![0u8; stride * ihdr.height as usize];

        for y in 0..ihdr.height as usize {
            let row = &mut pixels[y * stride..(y + 1) * stride];
            for x in 0..ihdr.width as usize {
                for channel in 0..channels {
                    // A diagonal ramp per channel, plus a low-amplitude
                    // ripple so the data isn't perfectly predictable (which
                    // would flatter the filters unrealistically).
                    let base = (x * 3 + y * 5 + channel * 40) / 2;
                    let ripple = ((x / 7) ^ (y / 5)) % 11;
                    let value = u8::try_from((base + ripple) % 256).unwrap_or(0);
                    row[x * channels + channel] = value;
                }
            }
        }
        pixels
    }

    /// Xorshift32 noise, matching `fixtures.rs`'s generator so the two
    /// halves of the benchmark suite share one notion of "random bytes".
    fn noise(ihdr: &Ihdr) -> Vec<u8> {
        let mut state = 0x9E37_79B9u32 ^ (ihdr.width << 8) ^ u32::from(ihdr.color_type.to_byte());
        let len = ihdr.stride() * ihdr.height as usize;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                u8::try_from(state & 0xFF).unwrap_or(0)
            })
            .collect()
    }

    /// The `png` crate's equivalent colour type, for handing the same
    /// pixels to the reference encoder.
    ///
    /// # Panics
    ///
    /// Panics for indexed colour, which these benchmarks don't cover
    /// (it would need a palette to be meaningful).
    #[must_use]
    pub fn png_color_type(&self) -> png::ColorType {
        match self.ihdr.color_type {
            ColorType::Grayscale => png::ColorType::Grayscale,
            ColorType::Truecolor => png::ColorType::Rgb,
            ColorType::GrayscaleAlpha => png::ColorType::GrayscaleAlpha,
            ColorType::TruecolorAlpha => png::ColorType::Rgba,
            ColorType::IndexedColor => panic!("indexed colour needs a palette"),
        }
    }
}
