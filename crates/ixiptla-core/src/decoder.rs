use crate::{
    ChunkError, ChunkReader, Defilter, Ihdr, IhdrError, InterlaceMethod, InvalidFilterType,
    Signature, SignatureError,
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

        // Interlacing (Adam7) needs its own per-pass reassembly step this
        // crate doesn't implement yet -- reject explicitly rather than
        // silently defiltering interlaced data as if it were sequential.
        if ihdr.interlace != InterlaceMethod::None {
            return Err(DecodeError::UnsupportedInterlace);
        }

        // Spec §5.6: "the IDAT chunks... are treated as a single stream" --
        // concatenate their data in file order before handing it to zlib.
        let mut idat = Vec::new();
        while let Some(chunk) = reader.next_chunk()? {
            match chunk.kind_str() {
                Some("IDAT") => idat.extend_from_slice(chunk.data()),
                Some("IEND") => break,
                _ => {} // ancillary or unhandled critical chunk: skip for now
            }
        }
        if idat.is_empty() {
            return Err(DecodeError::NoImageData);
        }

        let inflated = ixchel::zlib::decompress(&idat, max_decompressed)?;

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

        let mut defilter = Defilter::new(&ihdr);
        let mut pixels = Vec::with_capacity(ihdr.stride() * row_count);
        let mut inflated = inflated;
        for row in inflated.chunks_mut(row_len) {
            defilter.defilter_row(row)?;
            pixels.extend_from_slice(&row[1..]);
        }

        Ok(Self { ihdr, pixels })
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
    /// This crate doesn't yet reassemble `Adam7`-interlaced scanlines.
    UnsupportedInterlace,
    /// No `IDAT` chunks were present before `IEND`/end of stream.
    NoImageData,
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

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Signature(e) => write!(f, "{e}"),
            Self::Chunk(e) => write!(f, "{e}"),
            Self::Ihdr(e) => write!(f, "{e}"),
            Self::MissingIhdr => write!(f, "PNG chunk stream does not start with IHDR"),
            Self::UnsupportedInterlace => write!(f, "Adam7 interlacing is not yet supported"),
            Self::NoImageData => write!(f, "no IDAT chunks found"),
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
    fn encode_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = Crc32::new().update(kind).update(data).finalize();
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
        file.extend(encode_chunk(b"IHDR", &ihdr_data));

        let zlib_data = ixchel::zlib::compress(raw_scanlines);
        file.extend(encode_chunk(b"IDAT", &zlib_data));

        file.extend(encode_chunk(b"IEND", b""));
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
        file.extend(encode_chunk(b"IEND", b""));
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
        file.extend(encode_chunk(b"IHDR", &ihdr_data));
        file.extend(encode_chunk(b"IEND", b""));

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
        file.extend(encode_chunk(b"IHDR", &ihdr_data));
        let zlib_data = ixchel::zlib::compress(&raw); // only 2 actual rows
        file.extend(encode_chunk(b"IDAT", &zlib_data));
        file.extend(encode_chunk(b"IEND", b""));

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
        file.extend(encode_chunk(b"IHDR", &ihdr_data));
        file.extend(encode_chunk(b"IDAT", &zlib_data[..mid]));
        file.extend(encode_chunk(b"IDAT", &zlib_data[mid..]));
        file.extend(encode_chunk(b"IEND", b""));

        let png = Png::decode(&file, 1 << 20).unwrap();
        assert_eq!(png.pixels, vec![10, 20, 30, 40]);
    }

    #[test]
    fn rejects_adam7_interlace() {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.push(8);
        ihdr_data.push(0);
        ihdr_data.push(0);
        ihdr_data.push(0);
        ihdr_data.push(1); // Adam7
        file.extend(encode_chunk(b"IHDR", &ihdr_data));
        file.extend(encode_chunk(b"IDAT", b"anything"));
        file.extend(encode_chunk(b"IEND", b""));

        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::UnsupportedInterlace));
    }
}
