/// Incremental CRC-32 (ISO 3309 / ITU-T V.42, polynomial `0xEDB8_8320`)
/// as specified by PNG's Annex D — every chunk's trailing 4 bytes are this
/// CRC over the chunk's type + data.
///
/// PNG's own spec ships a reference table-based implementation processing
/// one byte per table lookup; this instead uses the "slicing-by-8"
/// technique (Intel's well-known CRC speedup: 8 precomputed tables, one
/// per byte lane, so 8 input bytes are folded per loop iteration instead
/// of 1) since a single `IDAT` chunk's CRC can cover a multi-megabyte
/// compressed payload — the dominant per-chunk cost for any real image.
/// Kept as its own struct rather than a free function so a chunk can be
/// fed to it incrementally (type bytes, then data) without concatenating a
/// temporary buffer first.
#[derive(Debug, Clone)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    /// Precomputed tables of CRC remainders for all 256 possible byte
    /// values at each of 8 byte lanes, built once at compile time.
    /// `TABLES[0]` is PNG Annex D's ordinary `make_crc_table`; `TABLES[n]`
    /// for `n > 0` is `TABLES[n-1]` run through the same shift-XOR step
    /// again -- equivalently, `TABLES[n][b]` is the CRC contribution of
    /// byte `b` sitting `n` positions back in the stream, which is what
    /// lets 8 bytes be combined via 8 table lookups + XORs instead of 8
    /// sequential single-byte folds.
    const TABLES: [[u32; 256]; 8] = Self::build_tables();

    #[allow(clippy::cast_possible_truncation)]
    const fn build_tables() -> [[u32; 256]; 8] {
        let mut tables = [[0u32; 256]; 8];

        // Lane 0 is the ordinary reference table.
        let mut n = 0;
        while n < 256 {
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
            tables[0][n] = c;
            n += 1;
        }

        // Each subsequent lane is derived from the previous one: folding
        // a zero byte through the CRC step is exactly `table[c & 0xFF] ^
        // (c >> 8)` with the low byte taken as an index -- iterating that
        // gives the "this byte is one position further back" tables the
        // slicing-by-8 combine step needs.
        let mut lane = 1;
        while lane < 8 {
            let mut n = 0;
            while n < 256 {
                let prev = tables[lane - 1][n];
                tables[lane][n] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
                n += 1;
            }
            lane += 1;
        }

        tables
    }

    /// Start a new running CRC (PNG's `crc = 0xffffffff` initial state).
    #[must_use]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Fold more bytes into the running CRC.
    ///
    /// On `x86_64`, tries the wider VPCLMULQDQ+AVX2 folding path first
    /// (see [`crate::crc32_vpclmulqdq::Vpclmulqdq`]), then the plain
    /// PCLMULQDQ path (see [`crate::crc32_pclmulqdq::Pclmulqdq`]) -- each
    /// declines (returning control here) for short inputs or CPUs
    /// lacking the needed instructions, in which case (and on every
    /// other architecture) this falls back to the scalar slicing-by-8
    /// path below.
    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        self.state = Self::update_state(self.state, bytes);
        self
    }

    fn update_state(state: u32, bytes: &[u8]) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            // Try the wider VPCLMULQDQ+AVX2 fold first (halves the
            // fold-by-64 loop's multiply-instruction count over the
            // plain PCLMULQDQ path -- see `crc32_vpclmulqdq`'s module
            // doc) -- it declines on CPUs lacking VPCLMULQDQ/AVX2, in
            // which case the plain PCLMULQDQ path (declining in turn for
            // CPUs lacking even that, or short inputs) is tried next.
            if let Some(folded) = crate::crc32_vpclmulqdq::Vpclmulqdq::try_update(state, bytes) {
                return folded;
            }
            if let Some(folded) = crate::crc32_pclmulqdq::Pclmulqdq::try_update(state, bytes) {
                return folded;
            }
        }
        Self::update_state_scalar(state, bytes)
    }

    /// The plain slicing-by-8 core, operating directly on a CRC register
    /// value rather than `&mut self` -- pulled out of [`Self::update`] so
    /// [`crate::crc32_pclmulqdq::Pclmulqdq`] can hand off its own
    /// sub-16-byte tail (after PCLMULQDQ folding + Barrett reduction has
    /// already produced a register-domain CRC value identical in kind to
    /// this function's `state` parameter) to the *same* verified scalar
    /// implementation, instead of a second hand-ported copy of it.
    pub(crate) fn update_state_scalar(state: u32, bytes: &[u8]) -> u32 {
        let mut c = state;
        let (blocks, remainder) = bytes.as_chunks::<8>();
        for chunk in blocks {
            // Fold the running CRC's 4 bytes in as the first 4 input
            // bytes of this block (standard slicing-by-8 combine), then
            // look up each of the 8 block bytes in its own lane table and
            // XOR them all together. Table indices run from the
            // most-significant lane (byte 7, farthest back) down to the
            // least (byte 0, most recent), matching `TABLES[n]`'s "n
            // positions back" derivation above.
            let c0 = (c & 0xFF) as u8 ^ chunk[0];
            let c1 = ((c >> 8) & 0xFF) as u8 ^ chunk[1];
            let c2 = ((c >> 16) & 0xFF) as u8 ^ chunk[2];
            let c3 = ((c >> 24) & 0xFF) as u8 ^ chunk[3];
            c = Self::TABLES[7][c0 as usize]
                ^ Self::TABLES[6][c1 as usize]
                ^ Self::TABLES[5][c2 as usize]
                ^ Self::TABLES[4][c3 as usize]
                ^ Self::TABLES[3][chunk[4] as usize]
                ^ Self::TABLES[2][chunk[5] as usize]
                ^ Self::TABLES[1][chunk[6] as usize]
                ^ Self::TABLES[0][chunk[7] as usize];
        }
        for &byte in remainder {
            c = Self::TABLES[0][((c ^ u32::from(byte)) & 0xFF) as usize] ^ (c >> 8);
        }
        c
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

    /// Exercises the slicing-by-8 fast path plus its `< 8`-byte remainder
    /// tail across a range of lengths straddling multiple 8-byte block
    /// boundaries, cross-checked against the plain byte-at-a-time
    /// algorithm (lane-0-only, the pre-slicing-by-8 implementation) on
    /// the same deterministic pseudo-random input -- catches an off-by-
    /// one in the chunking/remainder split that a single fixed-size test
    /// vector wouldn't.
    #[test]
    fn slicing_by_8_matches_byte_at_a_time_across_lengths() {
        fn byte_at_a_time(bytes: &[u8]) -> u32 {
            let table = Crc32::build_tables()[0];
            let mut c = 0xFFFF_FFFFu32;
            for &byte in bytes {
                c = table[((c ^ u32::from(byte)) & 0xFF) as usize] ^ (c >> 8);
            }
            c ^ 0xFFFF_FFFF
        }

        let mut state = 0x1234_5678u32;
        let mut next_byte = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFF) as u8
        };
        let data: Vec<u8> = (0..2000).map(|_| next_byte()).collect();

        for len in [0, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1000, 1999, 2000] {
            let slice = &data[..len];
            assert_eq!(
                Crc32::of(slice),
                byte_at_a_time(slice),
                "mismatch at len {len}"
            );
        }
    }
}
