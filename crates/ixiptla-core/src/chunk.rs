use crate::Crc32;

/// One raw PNG chunk (spec §5.3): a 4-byte type tag, its data payload, and
/// the CRC-32 that covers type+data (already verified by the time you have
/// one of these — see [`ChunkReader::next_chunk`]).
///
/// This is deliberately the *raw* chunk, not any parsed/interpreted form
/// (IHDR fields, palette entries, ...) — those parses live in their own
/// modules that consume a `Chunk`'s `data`, keeping "split the byte stream
/// into chunks" and "interpret a chunk's meaning" as separate concerns.
#[derive(Debug, Clone, Copy)]
pub struct Chunk<'a> {
    kind: [u8; 4],
    data: &'a [u8],
}

impl<'a> Chunk<'a> {
    /// The 4-byte type tag, e.g. `b"IHDR"`, `b"IDAT"`, `b"IEND"`.
    #[must_use]
    pub const fn kind(&self) -> [u8; 4] {
        self.kind
    }

    /// The type tag as `&str`, when it's the ASCII PNG chunk names always
    /// are; `None` for a (spec-violating) non-ASCII tag.
    #[must_use]
    pub fn kind_str(&self) -> Option<&str> {
        core::str::from_utf8(&self.kind).ok()
    }

    /// The chunk's payload, excluding the length/type/CRC framing.
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Ancillary chunks (lowercase first letter of the type) are safe to
    /// skip if unrecognized (spec §5.4); critical chunks (uppercase) are
    /// not.
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        self.kind[0].is_ascii_uppercase()
    }
}

/// Iterates the chunk stream following the 8-byte PNG signature, verifying
/// each chunk's CRC-32 as it goes (spec §5.3: "a CRC error... is a
/// fatal condition").
///
/// # Why not a plain function returning `Vec<Chunk>`
///
/// PNG's chunk stream is meant to be processed as a stream, and `IDAT` data
/// in particular is a poor fit for eager collection into a temporary
/// `Vec<Chunk>` before scanning it — an image can legally split its
/// compressed data across many `IDAT` chunks. Consumers that just want to
/// concatenate `IDAT` payloads can already do that by iterating; nothing
/// forces materializing every chunk up front.
#[derive(Debug)]
pub struct ChunkReader<'a> {
    remaining: &'a [u8],
}

impl<'a> ChunkReader<'a> {
    /// Wrap the chunk stream that follows the 8-byte PNG signature (i.e.
    /// whatever [`crate::Signature::strip`] returned).
    #[must_use]
    pub const fn new(chunk_stream: &'a [u8]) -> Self {
        Self {
            remaining: chunk_stream,
        }
    }

    /// Parse and CRC-verify the next chunk, advancing past it.
    ///
    /// Returns `Ok(None)` once the stream is exhausted (a well-formed PNG
    /// always ends this at exactly the byte after `IEND`'s CRC, but this
    /// reader doesn't require that — callers that care can check for it).
    ///
    /// # Panics
    ///
    /// Never in practice: the internal `try_into()`s convert slices this
    /// function has just produced via `split_at(4)`, which are always
    /// exactly 4 bytes long.
    pub fn next_chunk(&mut self) -> Result<Option<Chunk<'a>>, ChunkError> {
        if self.remaining.is_empty() {
            return Ok(None);
        }
        // Length(4) + Type(4) + CRC(4) = 12 bytes of framing at minimum,
        // plus `length` bytes of data in between.
        if self.remaining.len() < 12 {
            return Err(ChunkError::Truncated);
        }

        let (len_bytes, rest) = self.remaining.split_at(4);
        let len_array: [u8; 4] = len_bytes.try_into().expect("split_at(4) yields 4 bytes");
        let len = u32::from_be_bytes(len_array) as usize;

        let (kind_bytes, rest) = rest.split_at(4);
        let kind: [u8; 4] = kind_bytes.try_into().expect("split_at(4) yields 4 bytes");

        if rest.len() < len + 4 {
            return Err(ChunkError::Truncated);
        }
        let (data, rest) = rest.split_at(len);
        let (crc_bytes, rest) = rest.split_at(4);
        let crc_array: [u8; 4] = crc_bytes.try_into().expect("split_at(4) yields 4 bytes");
        let stored_crc = u32::from_be_bytes(crc_array);

        let computed_crc = Crc32::new().update(kind_bytes).update(data).finalize();
        if computed_crc != stored_crc {
            return Err(ChunkError::CrcMismatch {
                kind,
                expected: stored_crc,
                computed: computed_crc,
            });
        }

        self.remaining = rest;
        Ok(Some(Chunk { kind, data }))
    }
}

impl<'a> Iterator for ChunkReader<'a> {
    type Item = Result<Chunk<'a>, ChunkError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => {
                // A parse/CRC error is unrecoverable for this stream (we
                // don't know where the next chunk boundary is), so make
                // sure a subsequent `next()` call yields `None` instead of
                // re-parsing from the same bad position.
                self.remaining = &[];
                Some(Err(e))
            }
        }
    }
}

/// Why [`ChunkReader::next_chunk`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkError {
    /// Fewer bytes remained than the chunk's declared length requires.
    Truncated,
    /// The stored CRC-32 doesn't match the one computed over type+data.
    CrcMismatch {
        kind: [u8; 4],
        expected: u32,
        computed: u32,
    },
}

impl core::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "chunk stream truncated before a complete chunk"),
            Self::CrcMismatch {
                kind,
                expected,
                computed,
            } => {
                let name = core::str::from_utf8(kind).unwrap_or("????");
                write!(
                    f,
                    "CRC mismatch in {name} chunk: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
        }
    }
}

impl std::error::Error for ChunkError {}

#[cfg(test)]
mod tests {
    use super::{ChunkError, ChunkReader};
    use crate::Crc32;

    fn encode_chunk(kind: [u8; 4], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
        out.extend_from_slice(&kind);
        out.extend_from_slice(data);
        let crc = Crc32::new().update(&kind).update(data).finalize();
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    #[test]
    fn reads_single_well_formed_chunk() {
        let stream = encode_chunk(*b"IHDR", b"hello");
        let mut reader = ChunkReader::new(&stream);
        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(&chunk.kind(), b"IHDR");
        assert_eq!(chunk.data(), b"hello");
        assert!(reader.next_chunk().unwrap().is_none());
    }

    #[test]
    fn reads_multiple_chunks_in_sequence() {
        let mut stream = encode_chunk(*b"IHDR", b"abc");
        stream.extend(encode_chunk(*b"IDAT", b"xyz123"));
        stream.extend(encode_chunk(*b"IEND", b""));

        let reader = ChunkReader::new(&stream);
        let chunks: Result<Vec<_>, _> = reader.collect();
        let chunks = chunks.unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(&chunks[0].kind(), b"IHDR");
        assert_eq!(&chunks[1].kind(), b"IDAT");
        assert_eq!(chunks[1].data(), b"xyz123");
        assert_eq!(&chunks[2].kind(), b"IEND");
        assert_eq!(chunks[2].data(), b"");
    }

    #[test]
    fn detects_crc_mismatch() {
        let mut stream = encode_chunk(*b"IHDR", b"hello");
        // Corrupt one data byte without fixing the CRC.
        let data_start = 8;
        stream[data_start] ^= 0xFF;

        let mut reader = ChunkReader::new(&stream);
        let err = reader.next_chunk().unwrap_err();
        assert!(matches!(err, ChunkError::CrcMismatch { kind, .. } if &kind == b"IHDR"));
    }

    #[test]
    fn detects_truncated_stream() {
        let stream = encode_chunk(*b"IHDR", b"hello world");
        let truncated = &stream[..stream.len() - 3];
        let mut reader = ChunkReader::new(truncated);
        assert_eq!(reader.next_chunk().unwrap_err(), ChunkError::Truncated);
    }

    #[test]
    fn iterator_stops_after_error_instead_of_looping() {
        let stream = encode_chunk(*b"IHDR", b"hello");
        let mut broken = stream;
        broken[8] ^= 0xFF; // corrupt data, CRC now wrong
        let reader = ChunkReader::new(&broken);
        let results: Vec<_> = reader.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }

    #[test]
    fn is_critical_distinguishes_case() {
        let stream = encode_chunk(*b"tEXt", b"comment");
        let mut reader = ChunkReader::new(&stream);
        let chunk = reader.next_chunk().unwrap().unwrap();
        assert!(!chunk.is_critical());

        let stream = encode_chunk(*b"IHDR", b"data");
        let mut reader = ChunkReader::new(&stream);
        let chunk = reader.next_chunk().unwrap().unwrap();
        assert!(chunk.is_critical());
    }

    #[test]
    fn empty_stream_yields_none() {
        let mut reader = ChunkReader::new(&[]);
        assert!(reader.next_chunk().unwrap().is_none());
    }
}
