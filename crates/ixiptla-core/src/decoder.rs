use crate::{
    Adam7, ChunkError, ChunkReader, ColorType, Defilter, Ihdr, IhdrError, InterlaceMethod,
    InvalidFilterType, Palette, PaletteError, Signature, SignatureError, Transparency,
    TransparencyError,
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
    /// The `tRNS` chunk, parsed if present (spec §11.3.2: simple
    /// transparency for color types without their own alpha channel).
    /// Always `None` for [`crate::ColorType::GrayscaleAlpha`]/
    /// [`crate::ColorType::TruecolorAlpha`] images — [`Self::decode`]
    /// rejects a `tRNS` chunk there rather than silently ignoring
    /// conflicting transparency data.
    pub transparency: Option<Transparency>,
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
        // `input.len()` upper-bounds how many bytes of `IDAT` payload
        // could possibly exist in this file (they're a subset of it, plus
        // per-chunk overhead) -- reserving against that avoids `idat`
        // paying for `Vec`'s repeated grow-and-copy doubling as chunks
        // accumulate, for the cost of one conservative up-front
        // allocation instead.
        let mut idat = Vec::with_capacity(input.len());
        let mut palette = None;
        let mut transparency = None;
        while let Some(chunk) = reader.next_chunk()? {
            match chunk.kind_str() {
                Some("IDAT") => idat.extend_from_slice(chunk.data()),
                Some("PLTE") => palette = Some(Palette::parse(chunk.data())?),
                Some("tRNS") => {
                    transparency = Some(Transparency::parse(ihdr.color_type, chunk.data())?);
                }
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

        // The exact inflated size is knowable from `IHDR` alone, before
        // looking at a single compressed byte -- one `1 + stride`-byte
        // scanline per row for a non-interlaced image, or the sum of each
        // `Adam7` pass's own (smaller) row count/stride for an interlaced
        // one (spec §8.2: passes are concatenated back to back in the same
        // inflated stream). Passing this through avoids `Inflater` falling
        // back to its `input.len() * 3` guess, which for PNG's already-
        // compressed IDAT bytes is frequently a large over- or under-shoot
        // depending on how compressible the pixel data was -- an exact
        // hint turns however many doubling reallocations a wrong guess
        // costs into exactly one.
        let inflated_size_hint = Self::expected_inflated_size(&ihdr);
        let inflated =
            ixchel::zlib::decompress_with_hint(&idat, max_decompressed, Some(inflated_size_hint))?;

        let pixels = match ihdr.interlace {
            InterlaceMethod::None => Self::defilter_sequential(&ihdr, inflated)?,
            InterlaceMethod::Adam7 => Self::defilter_adam7(&ihdr, &inflated)?,
        };

        Ok(Self {
            ihdr,
            pixels,
            palette,
            transparency,
        })
    }

    /// The exact byte count `IDAT` inflates to for this `IHDR`, computed
    /// without looking at the compressed bytes at all -- see the call
    /// site's comment in [`Self::decode`] for why this is worth passing to
    /// `ixchel::zlib::decompress_with_hint` instead of letting it guess.
    ///
    /// Non-interlaced: `height` scanlines, each `1 + stride` bytes (the
    /// leading filter-type byte plus the row's content).
    ///
    /// `Adam7`: the same per-row math, but per pass -- each pass is its
    /// own tiny sequential image at [`Adam7Pass`]'s narrower width/height
    /// (spec section 8.2), and passes are concatenated back to back in the
    /// inflated stream, so the total is simply their sum. A pass an image
    /// is too small to contribute any pixels to (`Adam7Pass::is_empty`)
    /// contributes zero bytes, matching `defilter_adam7`'s own `continue`
    /// for that case.
    fn expected_inflated_size(ihdr: &Ihdr) -> usize {
        match ihdr.interlace {
            InterlaceMethod::None => ihdr.height as usize * (1 + ihdr.stride()),
            InterlaceMethod::Adam7 => Adam7::new(ihdr)
                .passes()
                .iter()
                .filter(|pass| !pass.is_empty())
                .map(|pass| {
                    let pass_stride =
                        (ihdr.bits_per_pixel() as usize * pass.width as usize).div_ceil(8);
                    pass.height as usize * (1 + pass_stride)
                })
                .sum(),
        }
    }

    /// Defilter a non-interlaced `IDAT` stream: one flat sequence of
    /// `1 + stride`-byte scanlines, top to bottom.
    //
    // Unfilters in place and then compacts the filter-type bytes out,
    // rather than defiltering into `inflated` and `extend_from_slice`-ing
    // each row's content into a second, freshly-allocated `pixels` buffer.
    // The upstream `png` crate's own optimization notes (see their 2026
    // "gets even faster" writeup) call this out as one of their largest
    // wins for exactly the same reason it helps here: touching each byte
    // once for defiltering and once again for the copy makes two passes
    // over an image-sized buffer instead of one, and doubles the amount
    // of data that has to round-trip through cache -- for a large image
    // neither the filtered row nor the growing output vector fits in L1/L2
    // at the same time, so the second pass is mostly-cold reads/writes,
    // not something the CPU can hide.
    //
    // The compaction below is a `copy_within` shifting each row's content
    // left by (however many filter-type bytes have been dropped so far),
    // which -- because `write_offset <= read row start` always -- is
    // exactly the same self-overlapping-but-safe backward-compatible
    // shrink `Vec::dedup`-style algorithms use; `chunks_mut` still drives
    // the defiltering itself unchanged; only what happens to the result
    // differs.
    //
    // `Defilter` itself no longer keeps its own `previous_row` copy (see
    // its doc comment) -- that "previous reconstructed scanline" is
    // instead read straight out of `inflated`'s already-compacted prefix
    // (`inflated[write_offset - stride..write_offset]`, exactly where the
    // compaction below leaves it) via `split_at_mut`, a genuine zero-copy
    // borrow rather than a second `memcpy` of every row's content: because
    // `write_offset` always trails `read_offset` (compaction only ever
    // removes bytes), the previous row's already-written bytes and the
    // current row's not-yet-defiltered bytes never overlap, so the borrow
    // checker can be shown that directly instead of this needing `unsafe`.
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

        let defilter = Defilter::new(ihdr);
        let stride = ihdr.stride();
        // Spec §9.2's "row above the first is all zero" -- unlike the
        // empty slice an earlier version of this loop passed for row 0,
        // `Defilter::defilter_row`'s `Up`/`Average`/`Paeth` paths (via
        // `WrappingByteAdder`/`WrappingAverageScan`/`WrappingPaethScan`)
        // all assert `previous_row.len() == content.len()`, so row 0
        // needs a real all-zero `stride`-byte buffer, not a 0-length one.
        let zero_row = vec![0u8; stride];
        let mut write_offset = 0usize;
        for read_offset in (0..inflated.len()).step_by(row_len) {
            // `write_offset <= read_offset` always (compaction only ever
            // removes bytes), so `front` (everything before `read_offset`,
            // including the already-compacted previous row) and `back`
            // (this row, not yet compacted) never alias -- `split_at_mut`
            // hands back two genuinely disjoint mutable/immutable borrows
            // of the same `Vec`, no `unsafe` needed.
            let (front, back) = inflated.split_at_mut(read_offset);
            let previous_row: &[u8] = if write_offset == 0 {
                &zero_row
            } else {
                &front[write_offset - stride..write_offset]
            };
            let row = &mut back[..row_len];
            defilter.defilter_row(row, previous_row)?;
            // `row[1..]` (the defiltered content, skipping this row's
            // filter-type byte) needs to end up at
            // `inflated[write_offset..][..stride]`. `write_offset` only
            // ever trails `read_offset + 1` (one filter-type byte has been
            // dropped per row already processed, and this row's own byte
            // makes it two once we account for the byte at `read_offset`
            // itself), so the source and destination ranges never overlap
            // in a way `copy_within` can't handle -- same shape as
            // `Vec::dedup`'s in-place compaction.
            inflated.copy_within(read_offset + 1..read_offset + row_len, write_offset);
            write_offset += stride;
        }
        inflated.truncate(row_count * stride);
        // `inflated`'s capacity was sized for the *filtered* stream
        // (`row_count * row_len`, one byte per row larger than the
        // defiltered content it now holds); `shrink_to_fit` releases that
        // slack back to the allocator instead of carrying it in every
        // `Png` this function returns for the rest of that `Png`'s
        // lifetime -- worth an extra realloc+copy here since it's O(one
        // row's worth of bytes) of overhead, not O(image size).
        inflated.shrink_to_fit();
        Ok(inflated)
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

            let defilter = Defilter::new(&Ihdr {
                width: pass.width,
                height: pass.height,
                bit_depth: ihdr.bit_depth,
                color_type: ihdr.color_type,
                interlace: InterlaceMethod::None,
            });
            // Spec §9.2's "row above the first is all zero", sized to
            // this pass's own stride (each pass is its own tiny image --
            // see this fn's doc comment) -- needed as a real all-zero
            // buffer, not an empty one, since `Defilter::defilter_row`'s
            // `Up`/`Average`/`Paeth` paths assert `previous_row.len() ==
            // content.len()`.
            let zero_row = vec![0u8; pass_stride];
            let mut pass_pixels = Vec::with_capacity(pass_stride * pass.height as usize);
            for row in pass_data.chunks_mut(row_len) {
                // Zero-copy "previous row" borrow, same shape as
                // `defilter_sequential` above -- and here it's even more
                // direct, since `row` (borrowed from `pass_data`) and
                // `pass_pixels` are two entirely separate `Vec`s, not two
                // views of the same one, so there's no aliasing to reason
                // about at all: the last `pass_stride` bytes already
                // written to `pass_pixels` *are* the previous row, or
                // `zero_row` on the pass's first row (spec §9.2's row-0
                // rule again).
                let previous_row: &[u8] = if pass_pixels.is_empty() {
                    &zero_row
                } else {
                    &pass_pixels[pass_pixels.len() - pass_stride..]
                };
                defilter.defilter_row(row, previous_row)?;
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
    Transparency(TransparencyError),
    /// The inflated `IDAT` stream's length isn't a whole number of
    /// `1 + stride`-byte scanlines.
    TruncatedImageData,
    /// The inflated data decoded to a different number of scanlines than
    /// `IHDR.height` promised.
    RowCountMismatch {
        expected: usize,
        actual: usize,
    },
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

impl From<TransparencyError> for DecodeError {
    fn from(e: TransparencyError) -> Self {
        Self::Transparency(e)
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
            Self::Transparency(e) => write!(f, "{e}"),
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
        assert!(matches!(
            err,
            DecodeError::RowCountMismatch {
                expected: 3,
                actual: 2
            }
        ));
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
        let file = build_adam7_png(
            2,
            2,
            0,
            8,
            &[
                pass1,
                vec![], // pass 2: empty
                vec![], // pass 3: empty
                vec![], // pass 4: empty
                vec![], // pass 5: empty
                pass6,
                pass7,
            ],
        );

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

    #[test]
    fn parses_trns_chunk_for_grayscale_image() {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&[8, 0, 0, 0, 0]);
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        file.extend(encode_chunk(*b"tRNS", &[0x00, 0x2A])); // transparent sample 42
        let raw = [0u8, 42];
        let zlib_data = ixchel::zlib::compress(&raw);
        file.extend(encode_chunk(*b"IDAT", &zlib_data));
        file.extend(encode_chunk(*b"IEND", b""));

        let png = Png::decode(&file, 1 << 20).unwrap();
        assert!(png.transparency.is_some());
    }

    #[test]
    fn rejects_trns_on_alpha_color_type() {
        let mut file = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&[8, 4, 0, 0, 0]); // grayscale+alpha
        file.extend(encode_chunk(*b"IHDR", &ihdr_data));
        file.extend(encode_chunk(*b"tRNS", &[0x00, 0x2A]));
        let raw = [0u8, 42, 255];
        let zlib_data = ixchel::zlib::compress(&raw);
        file.extend(encode_chunk(*b"IDAT", &zlib_data));
        file.extend(encode_chunk(*b"IEND", b""));

        let err = Png::decode(&file, 1 << 20).unwrap_err();
        assert!(matches!(err, DecodeError::Transparency(_)));
    }
}
