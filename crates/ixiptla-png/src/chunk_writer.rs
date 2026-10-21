use crate::Crc32;

/// Serializes PNG chunks (spec §5.3) into a caller-owned byte buffer: the
/// 4-byte big-endian length, the 4-byte type tag, the payload, and the
/// CRC-32 over type+payload.
///
/// # Why a struct and not a free `write_chunk` helper
///
/// The framing rules here are shared by every chunk an encoder emits, and
/// they carry two invariants worth owning in one place: the length field
/// covers *only* the payload (not the type tag), while the CRC covers the
/// type tag *and* the payload. Those two nearly-identical-but-different
/// spans are exactly the kind of detail that rots when it's copy-pasted per
/// chunk kind, so all of it lives behind [`Self::write`].
///
/// # Why it borrows the output buffer instead of returning `Vec`s
///
/// A PNG is one contiguous byte stream, and the largest chunk (`IDAT`) is
/// the whole compressed image. Returning a fresh `Vec` per chunk would mean
/// allocating the image twice and copying it into the final buffer; writing
/// straight into the caller's buffer keeps encoding to a single output
/// allocation, mirroring the decode side's "don't copy what the caller
/// already has" stance.
#[derive(Debug)]
pub struct ChunkWriter<'a> {
    out: &'a mut Vec<u8>,
}

impl<'a> ChunkWriter<'a> {
    /// Chunk framing overhead in bytes: length(4) + type(4) + CRC(4).
    ///
    /// Exposed so callers sizing an output buffer up front can budget for
    /// framing without re-deriving the constant.
    pub const OVERHEAD: usize = 12;

    /// Wrap the buffer that chunks will be appended to.
    pub const fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out }
    }

    /// Append one complete chunk with `kind` as its type tag.
    ///
    /// # Panics
    ///
    /// If `data` exceeds `u32::MAX` bytes, which the spec's 4-byte length
    /// field cannot express. This is unreachable for images that fit in
    /// memory on a 64-bit host and would indicate a caller bug rather than
    /// malformed input, so it is an assertion rather than a `Result` that
    /// every call site would have to thread through.
    pub fn write(&mut self, kind: [u8; 4], data: &[u8]) {
        let length = u32::try_from(data.len())
            .expect("chunk payload exceeds the 4-byte length field PNG allows");

        self.out.extend_from_slice(&length.to_be_bytes());
        self.out.extend_from_slice(&kind);
        self.out.extend_from_slice(data);

        // Spec §5.3: the CRC is computed over the type tag and the payload,
        // but *not* the length field -- so it can't simply run over the
        // bytes just appended.
        let mut crc = Crc32::new();
        crc.update(&kind);
        crc.update(data);
        self.out.extend_from_slice(&crc.finalize().to_be_bytes());
    }

    /// Append a chunk whose payload is built directly into the output
    /// buffer by `fill`, avoiding a staging allocation.
    ///
    /// `IDAT` payloads are the entire compressed image, so materializing
    /// one in a temporary `Vec` just to copy it into the output would
    /// double both the allocation and the memory traffic for the largest
    /// chunk in the file. Instead the length field is written as a
    /// placeholder, `fill` appends the payload in place, and the length is
    /// patched afterwards once its true size is known.
    ///
    /// # Panics
    ///
    /// If `fill` appends more than `u32::MAX` bytes; see [`Self::write`].
    pub fn write_with<F>(&mut self, kind: [u8; 4], fill: F)
    where
        F: FnOnce(&mut Vec<u8>),
    {
        // Reserve the length field; its real value isn't known until `fill`
        // has run, so patch it back in below.
        let length_at = self.out.len();
        self.out.extend_from_slice(&[0; 4]);
        self.out.extend_from_slice(&kind);

        let payload_at = self.out.len();
        fill(self.out);
        let payload_len = self.out.len() - payload_at;
        let length = u32::try_from(payload_len)
            .expect("chunk payload exceeds the 4-byte length field PNG allows");
        self.out[length_at..length_at + 4].copy_from_slice(&length.to_be_bytes());

        let mut crc = Crc32::new();
        crc.update(&kind);
        crc.update(&self.out[payload_at..payload_at + payload_len]);
        let checksum = crc.finalize();
        self.out.extend_from_slice(&checksum.to_be_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::ChunkWriter;
    use crate::{ChunkReader, Crc32};

    #[test]
    fn writes_framing_the_chunk_reader_accepts() {
        let mut out = Vec::new();
        ChunkWriter::new(&mut out).write(*b"IHDR", &[1, 2, 3]);

        // Length covers only the payload, not the type tag.
        assert_eq!(&out[..4], &3u32.to_be_bytes());
        assert_eq!(&out[4..8], b"IHDR");
        assert_eq!(&out[8..11], &[1, 2, 3]);
        assert_eq!(out.len(), 3 + ChunkWriter::OVERHEAD);

        let mut reader = ChunkReader::new(&out);
        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk.kind(), *b"IHDR");
        assert_eq!(chunk.data(), &[1, 2, 3]);
        assert!(reader.next_chunk().unwrap().is_none());
    }

    #[test]
    fn crc_covers_type_and_data_but_not_length() {
        let mut out = Vec::new();
        ChunkWriter::new(&mut out).write(*b"IEND", &[]);

        let written = u32::from_be_bytes(out[8..12].try_into().unwrap());
        let mut expected = Crc32::new();
        expected.update(b"IEND");
        assert_eq!(written, expected.finalize());
    }

    #[test]
    fn empty_payload_is_just_framing() {
        let mut out = Vec::new();
        ChunkWriter::new(&mut out).write(*b"IEND", &[]);
        assert_eq!(out.len(), ChunkWriter::OVERHEAD);

        let mut reader = ChunkReader::new(&out);
        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk.data(), b"");
    }

    #[test]
    fn write_with_matches_write_byte_for_byte() {
        // The in-place path must be indistinguishable from the staged one,
        // including the patched-back length field and the CRC.
        let payload: Vec<u8> = (0..=255u8).collect();

        let mut staged = Vec::new();
        ChunkWriter::new(&mut staged).write(*b"IDAT", &payload);

        let mut in_place = Vec::new();
        ChunkWriter::new(&mut in_place).write_with(*b"IDAT", |out| {
            out.extend_from_slice(&payload);
        });

        assert_eq!(staged, in_place);
    }

    #[test]
    fn write_with_handles_an_empty_payload() {
        let mut staged = Vec::new();
        ChunkWriter::new(&mut staged).write(*b"IDAT", &[]);

        let mut in_place = Vec::new();
        ChunkWriter::new(&mut in_place).write_with(*b"IDAT", |_| {});

        assert_eq!(staged, in_place);
    }

    #[test]
    fn appends_successive_chunks_without_disturbing_earlier_ones() {
        let mut out = Vec::new();
        let mut writer = ChunkWriter::new(&mut out);
        writer.write(*b"IHDR", &[9]);
        writer.write_with(*b"IDAT", |o| o.extend_from_slice(b"payload"));
        writer.write(*b"IEND", &[]);

        let mut reader = ChunkReader::new(&out);
        let kinds: Vec<[u8; 4]> = core::iter::from_fn(|| reader.next_chunk().unwrap())
            .map(|c| c.kind())
            .collect();
        assert_eq!(kinds, vec![*b"IHDR", *b"IDAT", *b"IEND"]);
    }
}
