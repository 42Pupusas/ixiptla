/// Incremental CRC-32 (ISO 3309 / ITU-T V.42, polynomial `0xEDB8_8320`)
/// as specified by PNG's Annex D — every chunk's trailing 4 bytes are this
/// CRC over the chunk's type + data.
///
/// PNG's own spec ships a reference table-based implementation; this is
/// that same algorithm, kept as its own struct rather than a free function
/// so a chunk can be fed to it incrementally (type bytes, then data) without
/// concatenating a temporary buffer first.
#[derive(Debug, Clone)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    /// Precomputed table of CRC remainders for all 256 possible byte
    /// values, built once at compile time (PNG Annex D's `make_crc_table`).
    const TABLE: [u32; 256] = Self::build_table();

    #[allow(clippy::cast_possible_truncation)]
    const fn build_table() -> [u32; 256] {
        let mut table = [0u32; 256];
        let mut n = 0;
        while n < 256 {
            // n never exceeds 255 (loop bound), so this cast is exact.
            let mut c = n as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[n] = c;
            n += 1;
        }
        table
    }

    /// Start a new running CRC (PNG's `crc = 0xffffffff` initial state).
    #[must_use]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Fold more bytes into the running CRC.
    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        let mut c = self.state;
        for &byte in bytes {
            c = Self::TABLE[((c ^ u32::from(byte)) & 0xFF) as usize] ^ (c >> 8);
        }
        self.state = c;
        self
    }

    /// Finalize: PNG XORs the running state with `0xffffffff` to produce
    /// the stored CRC value.
    #[must_use]
    pub const fn finalize(&self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }

    /// Convenience one-shot CRC over a single byte slice.
    #[must_use]
    pub fn of(bytes: &[u8]) -> u32 {
        Self::new().update(bytes).finalize()
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Crc32;

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(Crc32::of(b""), 0);
    }

    #[test]
    fn known_vector_check() {
        // "123456789" is the standard CRC-32/ISO-HDLC (== PNG's variant)
        // check vector; the expected value is the widely published one for
        // this polynomial/init/refin/refout/xorout combination.
        assert_eq!(Crc32::of(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let one_shot = Crc32::of(b"IHDRhello world");
        let mut incremental = Crc32::new();
        incremental.update(b"IHDR").update(b"hello world");
        assert_eq!(incremental.finalize(), one_shot);
    }

    #[test]
    fn ihdr_type_bytes_known_crc() {
        // A minimal real PNG IHDR chunk's CRC, cross-checked against the
        // canonical 1x1 white PNG (a widely reproduced test fixture).
        let mut data = Vec::new();
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&1u32.to_be_bytes()); // width
        data.extend_from_slice(&1u32.to_be_bytes()); // height
        data.push(8); // bit depth
        data.push(2); // color type: truecolor
        data.push(0); // compression
        data.push(0); // filter
        data.push(0); // interlace
        assert_eq!(Crc32::of(&data), 2_423_739_358);
    }
}
