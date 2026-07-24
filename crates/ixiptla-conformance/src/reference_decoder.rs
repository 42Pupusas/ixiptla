//! The comparison logic shared between the synthetic fixture tests in
//! `src/lib.rs` and the real-world `PngSuite` corpus walked by
//! `tests/pngsuite.rs`.
//!
//! Pulled into its own struct (rather than duplicated free functions in
//! both test targets, or left `#[cfg(test)]`-only where the integration
//! test binary couldn't reach it) so there is exactly one implementation
//! of "decode with both crates and diff the RGBA8 bytes" to keep correct.

use ixiptla_core::{Png, RgbaExpander};
use png::{ColorType as PngColorType, Transformations};
use std::io::Cursor;

/// Decodes a PNG file with both `ixiptla_core` and the `png` crate,
/// normalizes both to RGBA8, and reports whether they agree.
#[derive(Debug)]
pub struct ReferenceComparison;

impl ReferenceComparison {
    /// Decode `file` with `ixiptla_core`, expanded to RGBA8.
    fn decode_with_ixiptla(file: &[u8]) -> Result<Vec<u8>, String> {
        let decoded =
            Png::decode(file, 1 << 24).map_err(|e| format!("ixiptla_core decode: {e}"))?;
        let expander = RgbaExpander::new(
            &decoded.ihdr,
            decoded.palette.as_ref(),
            decoded.transparency.as_ref(),
        );
        expander
            .expand(&decoded.pixels)
            .map_err(|e| format!("ixiptla_core expand: {e}"))
    }

    /// Decode `file` with the `png` crate, normalized to RGBA8.
    ///
    /// `EXPAND | ALPHA | STRIP_16` gets every color type to "8 bits/
    /// sample, alpha present", but `png` 0.18 has no built-in "replicate
    /// gray into R/G/B" transform (that flag exists in some other PNG
    /// libraries under the name `GRAY_TO_RGB`, not this crate) — so
    /// grayscale/grayscale+alpha output is still 1 or 2 samples/pixel
    /// here; [`Self::normalize_to_rgba8`] below does that replication by
    /// hand so both sides land on the same 4-bytes/pixel layout
    /// `ixiptla_core::RgbaExpander` produces.
    fn decode_with_png_crate(file: &[u8]) -> Result<Vec<u8>, String> {
        let mut decoder = png::Decoder::new(Cursor::new(file));
        decoder.set_transformations(
            Transformations::EXPAND | Transformations::ALPHA | Transformations::STRIP_16,
        );
        let mut reader = decoder
            .read_info()
            .map_err(|e| format!("png crate read_info: {e}"))?;
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| format!("png crate next_frame: {e}"))?;
        buf.truncate(info.buffer_size());
        Ok(Self::normalize_to_rgba8(&buf, reader.output_color_type().0))
    }

    /// Replicate grayscale (with `EXPAND | ALPHA` already applied, so 1
    /// sample + 1 alpha byte/pixel) into an RGBA quad; pass RGBA-shaped
    /// data straight through unchanged.
    fn normalize_to_rgba8(buf: &[u8], color_type: PngColorType) -> Vec<u8> {
        match color_type {
            PngColorType::Grayscale => buf.iter().flat_map(|&v| [v, v, v, 255]).collect(),
            PngColorType::GrayscaleAlpha => buf
                .chunks_exact(2)
                .flat_map(|px| [px[0], px[0], px[0], px[1]])
                .collect(),
            PngColorType::Rgba => buf.to_vec(),
            other => panic!("unexpected post-EXPAND|ALPHA color type: {other:?}"),
        }
    }

    /// Decode `file` with both decoders and compare. `Ok(())` means they
    /// agree pixel-for-pixel (or both reject the file as invalid); `Err`
    /// describes any divergence, including one decoder accepting a file
    /// the other rejects.
    pub fn compare(file: &[u8]) -> Result<(), String> {
        match (
            Self::decode_with_png_crate(file),
            Self::decode_with_ixiptla(file),
        ) {
            (Ok(reference), Ok(ours)) if ours == reference => Ok(()),
            (Ok(_), Ok(_)) => Err("RGBA8 bytes diverge between decoders".to_string()),
            (Err(e), Ok(_)) => Err(format!(
                "png crate rejected the file but ixiptla_core accepted it: {e}"
            )),
            (Ok(_), Err(e)) => Err(format!(
                "ixiptla_core rejected the file but the png crate accepted it: {e}"
            )),
            (Err(png_err), Err(ixiptla_err)) => {
                // Both reject it -- that's agreement too, just log why in
                // case the *reasons* diverge in a way worth investigating.
                let _ = (png_err, ixiptla_err);
                Ok(())
            }
        }
    }

    /// Assert both decoders agree, pixel-for-pixel, on this exact file
    /// (panicking test-assertion wrapper around [`Self::compare`]).
    ///
    /// # Panics
    ///
    /// Panics (with the divergence description) if the two decoders
    /// disagree — that's this function's entire purpose as a test
    /// assertion helper.
    pub fn assert_matches(file: &[u8]) {
        if let Err(msg) = Self::compare(file) {
            panic!("{msg}");
        }
    }
}
