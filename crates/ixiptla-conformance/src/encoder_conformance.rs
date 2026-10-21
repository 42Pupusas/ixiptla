//! Validates `ixiptla_png::Encoder`'s output against an independent
//! decoder.
//!
//! # Why round-tripping through our own decoder isn't enough
//!
//! `Encoder`'s unit tests decode what they encode with `Png::decode`, which
//! proves the two halves of *this* crate agree — but a shared misreading of
//! the spec would pass that test silently. Filter predictors, Adam7 pass
//! layout, and chunk framing are all places where an encoder and decoder
//! can be wrong in exactly compensating ways.
//!
//! Handing our files to the `png` crate closes that gap: it was written
//! from the spec independently, so agreement is real evidence the bytes on
//! disk are conformant rather than merely self-consistent.

use ixiptla_png::{ColorType, Encoder, FilterStrategy, Ihdr, InterlaceMethod};

use crate::ReferenceComparison;

/// Encodes synthetic images with `ixiptla_png` and checks the `png` crate
/// decodes them to the pixels we started from.
#[derive(Debug)]
pub struct EncoderConformance {
    ihdr: Ihdr,
}

impl EncoderConformance {
    /// Describe an image geometry to test.
    #[must_use]
    pub const fn new(
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

    /// Deterministic pixel bytes for this geometry, with any trailing
    /// padding bits in a scanline's last byte cleared.
    ///
    /// Sub-byte depths leave bits past the final pixel undefined; they
    /// carry no image data, so no encoder is obliged to preserve them.
    /// Zeroing them keeps the fixture comparable byte-for-byte.
    #[must_use]
    pub fn pixels(&self) -> Vec<u8> {
        let stride = self.ihdr.stride();
        let used_bits = self.ihdr.bits_per_pixel() as usize * self.ihdr.width as usize;
        // A running u8 counter keeps the arithmetic inside the type, so
        // the wrap is defined behaviour rather than a lossy cast. Any
        // varied, deterministic byte pattern serves the fixture equally
        // well; stepping by an odd number avoids short repeats.
        let mut next = 7u8;
        let mut pixels: Vec<u8> = (0..stride * self.ihdr.height as usize)
            .map(|_| {
                let value = next;
                next = next.wrapping_add(31);
                value
            })
            .collect();

        for row in pixels.chunks_exact_mut(stride) {
            for bit in used_bits..stride * 8 {
                row[bit / 8] &= !(1u8 << (7 - bit % 8));
            }
        }
        pixels
    }

    /// Encode this geometry with `strategy` and confirm the `png` crate
    /// and `ixiptla_png` both decode the result to identical RGBA8.
    ///
    /// # Errors
    ///
    /// Describes the divergence if the encode fails or the two decoders
    /// disagree about the file we produced.
    pub fn compare(&self, strategy: FilterStrategy) -> Result<(), String> {
        let pixels = self.pixels();
        let file = Encoder::new(self.ihdr)
            .with_filter_strategy(strategy)
            .encode(&pixels)
            .map_err(|e| format!("encode {:?}: {e}", self.ihdr))?;

        ReferenceComparison::compare(&file)
            .map_err(|e| format!("{:?} with {strategy:?}: {e}", self.ihdr))
    }

    /// Assert the `png` crate agrees with us about a file we encoded.
    ///
    /// # Panics
    ///
    /// Panics with the divergence description if it doesn't — this is a
    /// test assertion helper.
    pub fn assert_matches(&self, strategy: FilterStrategy) {
        if let Err(msg) = self.compare(strategy) {
            panic!("{msg}");
        }
    }
}
