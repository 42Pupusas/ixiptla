/// Reads/writes an arbitrary-width (up to 64-bit), MSB-first-packed bit
/// field at a given bit offset within a byte slice.
///
/// PNG rows are only byte-aligned at the *start* of each scanline (spec
/// §7.2: "Pixels are packed into bytes... with the leftmost pixel in the
/// high-order bits") — sub-byte bit depths (1/2/4) pack multiple pixels per
/// byte, and one "pixel" here means `bits_per_pixel` contiguous bits
/// starting wherever the previous pixel left off, not necessarily on a byte
/// boundary. `Adam7` reassembly needs to move one whole pixel's bits
/// (spanning however many channels/bit-depth combine to) from a pass's
/// compact row into the right bit position of the full image's row without
/// caring what the bits *mean* — this is that single mechanical operation,
/// factored out so `Adam7` itself only deals with pixel *coordinates*.
#[derive(Debug, Clone, Copy)]
pub struct PixelBits;

impl PixelBits {
    /// Read `width_bits` (<= 64) bits starting at `bit_offset` (0 = the
    /// high-order bit of `row[0]`), MSB-first, returned right-aligned in a
    /// `u64`.
    #[must_use]
    pub fn read(row: &[u8], bit_offset: usize, width_bits: u32) -> u64 {
        let mut value: u64 = 0;
        for i in 0..width_bits as usize {
            let bit_pos = bit_offset + i;
            let byte = row[bit_pos / 8];
            let bit = (byte >> (7 - bit_pos % 8)) & 1;
            value = (value << 1) | u64::from(bit);
        }
        value
    }

    /// Write the low `width_bits` (<= 64) bits of `value` into `row`
    /// starting at `bit_offset`, MSB-first — the exact inverse of
    /// [`Self::read`].
    pub fn write(row: &mut [u8], bit_offset: usize, width_bits: u32, value: u64) {
        for i in 0..width_bits as usize {
            let bit_pos = bit_offset + i;
            // Walk `value`'s bits from most- to least-significant, matching
            // the MSB-first order `read` extracted them in.
            let bit = (value >> (width_bits as usize - 1 - i)) & 1;
            let byte = &mut row[bit_pos / 8];
            let mask = 1u8 << (7 - bit_pos % 8);
            if bit == 1 {
                *byte |= mask;
            } else {
                *byte &= !mask;
            }
        }
    }

    /// Copy one pixel's `bits_per_pixel`-wide bit field from `src` at pixel
    /// index `src_x` to `dst` at pixel index `dst_x` — the composed
    /// read-then-write [`Adam7`](crate::Adam7Pass) reassembly actually
    /// performs per pixel.
    pub fn copy_pixel(src: &[u8], src_x: usize, dst: &mut [u8], dst_x: usize, bits_per_pixel: u32) {
        let value = Self::read(src, src_x * bits_per_pixel as usize, bits_per_pixel);
        Self::write(dst, dst_x * bits_per_pixel as usize, bits_per_pixel, value);
    }
}

#[cfg(test)]
mod tests {
    use super::PixelBits;

    #[test]
    fn reads_byte_aligned_8bit_field() {
        let row = [0xAB, 0xCD];
        assert_eq!(PixelBits::read(&row, 0, 8), 0xAB);
        assert_eq!(PixelBits::read(&row, 8, 8), 0xCD);
    }

    #[test]
    fn reads_sub_byte_fields_msb_first() {
        // 0b1011_0010: four 2-bit fields: 10, 11, 00, 10.
        let row = [0b1011_0010];
        assert_eq!(PixelBits::read(&row, 0, 2), 0b10);
        assert_eq!(PixelBits::read(&row, 2, 2), 0b11);
        assert_eq!(PixelBits::read(&row, 4, 2), 0b00);
        assert_eq!(PixelBits::read(&row, 6, 2), 0b10);
    }

    #[test]
    fn reads_field_spanning_a_byte_boundary() {
        // bits 4..12 span byte 0's low nibble and byte 1's high nibble.
        let row = [0b0000_1010, 0b1100_0000];
        assert_eq!(PixelBits::read(&row, 4, 8), 0b1010_1100);
    }

    #[test]
    fn write_then_read_round_trips_for_various_widths() {
        for width_bits in [1, 2, 4, 8, 16, 24, 32, 64] {
            let byte_len = (width_bits as usize * 3).div_ceil(8) + 1;
            let mut row = vec![0u8; byte_len];
            let value = if width_bits == 64 {
                0xDEAD_BEEF_1234_5678
            } else {
                (1u64 << width_bits) - 1 // all-ones pattern for this width
            };
            PixelBits::write(&mut row, 5, width_bits, value);
            assert_eq!(PixelBits::read(&row, 5, width_bits), value);
        }
    }

    #[test]
    fn write_does_not_disturb_neighboring_bits() {
        let mut row = [0xFFu8];
        // Zero out the middle 4 bits, leaving the outer 2+2 bits set.
        PixelBits::write(&mut row, 2, 4, 0b0000);
        assert_eq!(row[0], 0b1100_0011);
    }

    #[test]
    fn copy_pixel_moves_a_field_between_independent_buffers() {
        let src = [0b1010_0000u8]; // pixel 0 (4 bits) = 0b1010
        let mut dst = vec![0u8; 1];
        PixelBits::copy_pixel(&src, 0, &mut dst, 1, 4);
        assert_eq!(PixelBits::read(&dst, 4, 4), 0b1010);
    }
}
