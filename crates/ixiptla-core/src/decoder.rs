use crate::{
    Adam7, ChunkError, ChunkReader, ColorType, Defilter, Ihdr, IhdrError, InterlaceMethod,
    InvalidFilterType, Palette, PaletteError, Signature, SignatureError,
};

/// A decoded PNG: its header metadata plus the raw, defiltered pixel bytes
/// (scanlines concatenated, each exactly [`Ihdr::stride`] bytes, no filter-
/// type byte, no interlacing left to undo).
///
/// This is intentionally *not* yet "an RGBA image" — turning e.g. indexed-
/// color + a palette into RGBA, or expanding sub-byte bit depths into one
/// byte per sample, is a distinct concern (pixel format expansion) that
/// belongs in its own module once needed. `Png` hands you exactly what the
/// `IDAT` stream decompresses and defilters to, laid out per [`Self::ihdr`].
#[derive(Debug, Clone)]
pub struct Png {
    pub ihdr: Ihdr,
    pub pixels: Vec<u8>,
    /// The `PLTE` chunk, parsed if present. Always `Some` for a
    /// successfully-decoded [`crate::ColorType::IndexedColor`] image (spec
    /// §11.2.3: `PLTE` is mandatory for indexed color) — [`Self::decode`]
    /// rejects an indexed-color image missing one rather than handing back
    /// a `Png` an [`crate::RgbaExpander`] couldn't use. May also be present
    /// (and is simply carried along, unused by decoding itself) alongside
    /// truecolor/truecolor-alpha images, where `PLTE` is an optional
    /// suggested-palette hint rather than something pixel bytes reference.
    pub palette: Option<Palette>,
}

impl Png {
    /// Decode a complete PNG file's bytes: signature, chunk stream, `IHDR`,
    /// concatenated `IDAT` payloads inflated via `ixchel::zlib`, and each
    /// resulting scanline defiltered.
    ///
    /// `max_decompressed` bounds the inflated `IDAT` size (a
    /// decompression-bomb guard — forwarded to `ixchel::zlib::decompress`).
    ///
    /// # Errors
    ///
    /// See [`DecodeError`]'s variants.
    pub fn decode(input: &[u8], max_decompressed: usize) -> Result<Self, DecodeError> {
        let chunk_stream = Signature::strip(input)?;
        let mut reader = ChunkReader::new(chunk_stream);

        let first = reader.next_chunk()?.ok_or(DecodeError::MissingIhdr)?;
        if first.kind_str() != Some("IHDR") {
            return Err(DecodeError::MissingIhdr);
        }
        let ihdr = Ihdr::parse(first.data())?;

        // Spec §5.6: "the IDAT chunks... are treated as a single stream" --
        // concatenate their data in file order before handing it to zlib.
        // PLTE (spec §4.3, §11.2.3) must precede the first IDAT when present;
        // we don't enforce that ordering strictly here (a spec-violating but
        // otherwise-decodable file), just take whichever PLTE data we see.
        let mut idat = Vec::new();
        let mut palette = None;
        while let Some(chunk) = reader.next_chunk()? {
            match chunk.kind_str() {
                Some("IDAT") => idat.extend_from_slice(chunk.data()),
                Some("PLTE") => palette = Some(Palette::parse(chunk.data())?),
                Some("IEND") => break,
                _ => {} // ancillary or unhandled critical chunk: skip for now
            }
        }
        if idat.is_empty() {
            return Err(DecodeError::NoImageData);
        }
        if ihdr.color_type == ColorType::IndexedColor && palette.is_none() {
            return Err(DecodeError::MissingPalette);
        }

        let inflated = ixchel::zlib::decompress(&idat, max_decompressed)?;

        let pixels = match ihdr.interlace {
            InterlaceMethod::None => Self::defilter_sequential(&ihdr, inflated)?,
            InterlaceMethod::Adam7 => Self::defilter_adam7(&ihdr, &inflated)?,
        };

        Ok(Self { ihdr, pixels, palette })
    }

    /// Defilter a non-interlaced `IDAT` stream: one flat sequence of
    /// `1 + stride`-byte scanlines, top to bottom.
    fn defilter_sequential(ihdr: &Ihdr, mut inflated: Vec<u8>) -> Result<Vec<u8>, DecodeError> {
        let row_len = 1 + ihdr.stride();
        if inflated.len() % row_len != 0 {
            return Err(DecodeError::TruncatedImageData);
        }
        let row_count = inflated.len() / row_len;
        if row_count != ihdr.height as usize {
            return Err(DecodeError::RowCountMismatch {
                expected: ihdr.height as usize,
                actual: row_count,
            });
        }

        let mut defilter = Defilter::new(ihdr);
        let mut pixels = Vec::with_capacity(ihdr.stride() * row_count);
        for row in inflated.chunks_mut(row_len) {
            defilter.defilter_row(row)?;
            pixels.extend_from_slice(&row[1..]);
        }
        Ok(pixels)
    }

    /// Defilter and reassemble an `Adam7`-interlaced `IDAT` stream.
    ///
    /// The 7 passes are concatenated back to back in the same inflated
    /// stream, each pass being its own tiny sequential image (spec §8.2):
    /// its own row count/stride derived from [`Adam7Pass`]'s geometry, and
    /// crucially its own independent filter-predictor state (a pass's
    /// first row has no "previous row" from the pass before it — passes
    /// are unrelated sequences, not one continuous scanline stream).
    fn defilter_adam7(ihdr: &Ihdr, inflated: &[u8]) -> Result<Vec<u8>, DecodeError> {
        let adam7 = Adam7::new(ihdr);
        let passes = adam7.passes();
        let mut full_image = vec![0u8; ihdr.stride() * ihdr.height as usize];
        let mut offset = 0usize;

        for pass in &passes {
            if pass.is_empty() {
                continue;
            }
            // A pass is itself a valid tiny image at its own bit depth/
            // color type, so its per-row stride follows the same
            // bits-per-pixel math as the full image, just at the pass's
            // narrower width.
            let pass_stride = (ihdr.bits_per_pixel() as usize * pass.width as usize).div_ceil(8);
            let row_len = 1 + pass_stride;
            let pass_bytes = row_len * pass.height as usize;

            let pass_end = offset
                .checked_add(pass_bytes)
                .filter(|&end| end <= inflated.len())
                .ok_or(DecodeError::TruncatedImageData)?;
            let mut pass_data = inflated[offset..pass_end].to_vec();
            offset = pass_end;

            let mut defilter = Defilter::new(&Ihdr {
                width: pass.width,
                height: pass.height,
                bit_depth: ihdr.bit_depth,
                color_type: ihdr.color_type,
                interlace: InterlaceMethod::None,
            });
            let mut pass_pixels = Vec::with_capacity(pass_stride * pass.height as usize);
            for row in pass_data.chunks_mut(row_len) {
                defilter.defilter_row(row)?;
                pass_pixels.extend_from_slice(&row[1..]);
            }

            adam7.scatter_pass(pass, &pass_pixels, &mut full_image);
        }

        if offset != inflated.len() {
            return Err(DecodeError::TruncatedImageData);
        }

        Ok(full_image)
    }
}

/// Why [`Png::decode`] failed.
#[derive(Debug)]
pub enum DecodeError {
    Signature(SignatureError),
    Chunk(ChunkError),
    Ihdr(IhdrError),
    /// The chunk stream's first chunk wasn't `IHDR` (spec §5.6 requires it
    /// to be first), or there was no `IHDR` chunk at all.
    MissingIhdr,
    /// No `IDAT` chunks were present before `IEND`/end of stream.
    NoImageData,
    /// The image is [`ColorType::IndexedColor`] but no `PLTE` chunk was
    /// present (spec §11.2.3: mandatory for indexed color).
    MissingPalette,
    Palette(PaletteError),
    /// The inflated `IDAT` stream's length isn't a whole number of
    /// `1 + stride`-byte scanlines.
    TruncatedImageData,
    /// The inflated data decoded to a different number of scanlines than
    /// `IHDR.height` promised.
    RowCountMismatch { expected: usize, actual: usize },
    Zlib(ixchel::ZlibError),
    Filter(InvalidFilterType),
}

impl From<SignatureError> for DecodeError {
    fn from(e: SignatureError) -> Self {
        Self::Signature(e)
    }
}

impl From<ChunkError> for DecodeError {
    fn from(e: ChunkError) -> Self {
        Self::Chunk(e)
    }
}

impl From<IhdrError> for DecodeError {
    fn from(e: IhdrError) -> Self {
        Self::Ihdr(e)
    }
}

impl From<ixchel::ZlibError> for DecodeError {
    fn from(e: ixchel::ZlibError) -> Self {
        Self::Zlib(e)
    }
}

impl From<InvalidFilterType> for DecodeError {
    fn from(e: InvalidFilterType) -> Self {
        Self::Filter(e)
    }
}

impl From<PaletteError> for DecodeError {
    fn from(e: PaletteError) -> Self {
        Self::Palette(e)
    }
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Signature(e) => write!(f, "{e}"),
            Self::Chunk(e) => write!(f, "{e}"),
            Self::Ihdr(e) => write!(f, "{e}"),
            Self::MissingIhdr => write!(f, "PNG chunk stream does not start with IHDR"),
            Self::NoImageData => write!(f, "no IDAT chunks found"),
            Self::MissingPalette => {
                write!(f, "indexed-color image has no PLTE chunk")
            }
            Self::Palette(e) => write!(f, "{e}"),
            Self::TruncatedImageData => {
                write!(f, "inflated IDAT data is not a whole number of scanlines")
            }
            Self::RowCountMismatch { expected, actual } => write!(
                f,
                "expected {expected} scanlines per IHDR height, got {actual}"
            ),
            Self::Zlib(e) => write!(f, "{e}"),
            Self::Filter(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    use super::{DecodeError, Png};
    use crate::Crc32;

    /// Build a minimal well-formed PNG file in memory: signature + IHDR +
    /// one IDAT (zlib-wrapped, uncompressed-filter-byte-0 scanlines) +
    /// IEND, so the whole pipeline can be exercised without needing a real
    /// PNG fixture file on disk.
    fn encode_chunk(kind: [u8; 4], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
        out.extend_from_slice(&kind);
        out.extend_from_slice(data);
        let crc = Crc32::new().update(&kind).update(data).finalize();
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    fn build_png(width: u32, height: u32, color_type: u8, raw_scanlines: &[u8]) -> Vec<u8> {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&width.to_be_bytes());
        ihdr_data.extend_from_slice(&height.to_be_bytes());
        ihdr_data.push(8); // bit depth
        ihdr_data.push(color_type);
        ihdr_data.push(0); // compression
        ihdr_data.push(0); // filter
        ihdr_data.push(0); // interlace: none
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));

        let zlib_data = ixchel::zlib::compress(raw_scanlines);
        file.extend(encode_chunk(*b"IDAT", &zlib_data));

        file.extend(encode_chunk(*b"IEND", b""));
        file
    }

    #[test]
    fn decodes_minimal_unfiltered_grayscale_png() {
        // 2x2, 8-bit grayscale (color type 0): each row is filter-byte(0)
        // + 2 content bytes.
        let raw = [0u8, 10, 20, 0, 30, 40];
        let file = build_png(2, 2, 0, &raw);

        let png = Png::decode(&file, 1 << 20).unwrap();
        assert_eq!(png.ihdr.width, 2);
        assert_eq!(png.ihdr.height, 2);
        assert_eq!(png.pixels, vec![10, 20, 30, 40]);
    }

    #[test]
    fn decodes_multi_row_truecolor_with_sub_filter() {
        // 2x1, 8-bit RGB (color type 2): one row, Sub-filtered.
        // Original pixels: (1,2,3) (4,5,6). Sub-filtered: (1,2,3) (3,3,3).
        let raw = [1u8, 1, 2, 3, 3, 3, 3];
        let file = build_png(2, 1, 2, &raw);

        let png = Png::decode(&file, 1 << 20).unwrap();
        assert_eq!(png.pixels, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rejects_missing_signature() {
        let err = Png::decode(b"not a png", 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::Signature(_)));
    }

    #[test]
    fn rejects_stream_not_starting_with_ihdr() {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        file.extend(encode_chunk(*b"IEND", b""));
        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::MissingIhdr));
    }

    #[test]
    fn rejects_missing_idat() {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&[8, 0, 0, 0, 0]);
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        file.extend(encode_chunk(*b"IEND", b""));

        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::NoImageData));
    }

    #[test]
    fn rejects_row_count_mismatch() {
        // IHDR says height 3 but only 2 rows are actually present.
        let raw = [0u8, 10, 20, 0, 30, 40];
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&2u32.to_be_bytes());
        ihdr_data.extend_from_slice(&3u32.to_be_bytes()); // claims height 3
        ihdr_data.extend_from_slice(&[8, 0, 0, 0, 0]);
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        let zlib_data = ixchel::zlib::compress(&raw); // only 2 actual rows
        file.extend(encode_chunk(*b"IDAT", &zlib_data));
        file.extend(encode_chunk(*b"IEND", b""));

        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::RowCountMismatch { expected: 3, actual: 2 }));
    }

    #[test]
    fn idat_split_across_multiple_chunks_is_concatenated() {
        let raw = [0u8, 10, 20, 0, 30, 40];
        let zlib_data = ixchel::zlib::compress(&raw);
        let mid = zlib_data.len() / 2;

        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&2u32.to_be_bytes());
        ihdr_data.extend_from_slice(&2u32.to_be_bytes());
        ihdr_data.extend_from_slice(&[8, 0, 0, 0, 0]);
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        file.extend(encode_chunk(*b"IDAT", &zlib_data[..mid]));
        file.extend(encode_chunk(*b"IDAT", &zlib_data[mid..]));
        file.extend(encode_chunk(*b"IEND", b""));

        let png = Png::decode(&file, 1 << 20).unwrap();
        assert_eq!(png.pixels, vec![10, 20, 30, 40]);
    }

    /// Build an `Adam7`-interlaced PNG file: `raw_scanlines` is provided
    /// per-pass (already in the right order/shape for each of the 7
    /// passes), each pass's rows filter-byte-0 (identity) prefixed.
    fn build_adam7_png(
        width: u32,
        height: u32,
        color_type: u8,
        bit_depth: u8,
        pass_raw_scanlines: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&width.to_be_bytes());
        ihdr_data.extend_from_slice(&height.to_be_bytes());
        ihdr_data.push(bit_depth);
        ihdr_data.push(color_type);
        ihdr_data.push(0);
        ihdr_data.push(0);
        ihdr_data.push(1); // Adam7
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));

        let mut idat_source = Vec::new();
        for pass in pass_raw_scanlines {
            idat_source.extend_from_slice(pass);
        }
        let zlib_data = ixchel::zlib::compress(&idat_source);
        file.extend(encode_chunk(*b"IDAT", &zlib_data));
        file.extend(encode_chunk(*b"IEND", b""));
        file
    }

    #[test]
    fn decodes_adam7_interlaced_2x2_grayscale() {
        // 2x2, 8-bit grayscale: for a 2x2 image, only passes 1 (pixel
        // (0,0)), 6 (pixel (1,0)), and 7 (row y=1, both columns) are
        // non-empty -- passes 2..5 all contribute zero pixels (spec
        // §8.2, small-image case; verified by
        // `tiny_image_has_empty_late_passes`-style geometry: pass 6's
        // x_start=1 still falls inside a width-2 image).
        // Target pixel grid: (0,0)=42 (1,0)=7 (0,1)=99 (1,1)=100.
        let pass1 = vec![0u8, 42]; // 1x1: filter byte + 1 content byte
        let pass6 = vec![0u8, 7]; // 1x1: filter byte + 1 content byte
        let pass7 = vec![0u8, 99, 100]; // 2x1: filter byte + 2 content bytes
        let file = build_adam7_png(2, 2, 0, 8, &[
            pass1,
            vec![], // pass 2: empty
            vec![], // pass 3: empty
            vec![], // pass 4: empty
            vec![], // pass 5: empty
            pass6,
            pass7,
        ]);

        let png = Png::decode(&file, 1 << 20).unwrap();
        assert_eq!(png.pixels, vec![42, 7, 99, 100]);
    }

    #[test]
    fn rejects_truncated_adam7_stream() {
        let file = build_adam7_png(2, 2, 0, 8, &[vec![0u8, 42]]); // missing pass 7's data
        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::TruncatedImageData));
    }

    #[test]
    fn parses_plte_chunk_for_indexed_color_image() {
        // 2x1, 8-bit indexed color (color type 3): PLTE with 2 entries.
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&2u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&[8, 3, 0, 0, 0]);
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        file.extend(encode_chunk(*b"PLTE", &[10, 20, 30, 40, 50, 60]));
        let raw = [0u8, 1, 0]; // filter byte + indices 1, 0
        let zlib_data = ixchel::zlib::compress(&raw);
        file.extend(encode_chunk(*b"IDAT", &zlib_data));
        file.extend(encode_chunk(*b"IEND", b""));

        let png = Png::decode(&file, 1 << 20).unwrap();
        let palette = png.palette.unwrap();
        assert_eq!(palette.get(0).unwrap(), [10, 20, 30]);
        assert_eq!(palette.get(1).unwrap(), [40, 50, 60]);
        assert_eq!(png.pixels, vec![1, 0]);
    }

    #[test]
    fn rejects_indexed_color_image_missing_plte() {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&[8, 3, 0, 0, 0]);
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        let raw = [0u8, 0];
        let zlib_data = ixchel::zlib::compress(&raw);
        file.extend(encode_chunk(*b"IDAT", &zlib_data));
        file.extend(encode_chunk(*b"IEND", b""));

        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::MissingPalette));
    }

    #[test]
    fn non_indexed_image_decodes_fine_without_plte() {
        let raw = [0u8, 10, 20, 0, 30, 40];
        let file = build_png(2, 2, 0, &raw);
        let png = Png::decode(&file, 1 << 20).unwrap();
        assert!(png.palette.is_none());
    }
}
