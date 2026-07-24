use crate::{ColorType, InvalidColorType};

/// PNG's interlace method (spec §11.2.2's last `IHDR` field, table 11.1's
/// "Interlace method" — only two values are defined).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlaceMethod {
    /// Scanlines stored top-to-bottom, no interlacing.
    None,
    /// Adam7 seven-pass interlacing (spec §8.2).
    Adam7,
}

impl InterlaceMethod {
    const fn from_byte(byte: u8) -> Result<Self, IhdrError> {
        match byte {
            0 => Ok(Self::None),
            1 => Ok(Self::Adam7),
            other => Err(IhdrError::InvalidInterlaceMethod(other)),
        }
    }
}

/// The parsed `IHDR` chunk (spec §11.2.2) — the only chunk every PNG must
/// have, and always the first one.
///
/// Everything downstream (how many bytes a scanline is, how to interpret
/// each `IDAT`-decompressed byte, whether there's an `Adam7` pass structure
/// to undo) is driven by these fields, so this is parsed once into a typed
/// struct rather than re-reading the raw 13 bytes at each later step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ihdr {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: ColorType,
    pub interlace: InterlaceMethod,
}

impl Ihdr {
    /// `IHDR`'s data is always exactly this many bytes (spec §11.2.2):
    /// width(4) + height(4) + bit depth(1) + color type(1) +
    /// compression method(1) + filter method(1) + interlace method(1).
    pub const DATA_LEN: usize = 13;

    /// Parse and validate an `IHDR` chunk's data payload (the bytes between
    /// the chunk's type tag and CRC, i.e. [`crate::Chunk::data`] for the
    /// `IHDR` chunk).
    pub fn parse(data: &[u8]) -> Result<Self, IhdrError> {
        if data.len() != Self::DATA_LEN {
            return Err(IhdrError::WrongLength(data.len()));
        }

        let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        if width == 0 || height == 0 {
            return Err(IhdrError::ZeroDimension);
        }

        let bit_depth = data[8];
        let color_type = ColorType::from_byte(data[9])?;
        if !color_type.allowed_bit_depths().contains(&bit_depth) {
            return Err(IhdrError::InvalidBitDepth {
                bit_depth,
                color_type,
            });
        }

        // Spec §11.2.2: compression method 0 (deflate/inflate, RFC 1950
        // framing) and filter method 0 (the five adaptive per-byte filters
        // in §9) are the only values ever defined; both are simply
        // reserved for future extension, not a field decoders act on.
        let compression_method = data[10];
        if compression_method != 0 {
            return Err(IhdrError::InvalidCompressionMethod(compression_method));
        }
        let filter_method = data[11];
        if filter_method != 0 {
            return Err(IhdrError::InvalidFilterMethod(filter_method));
        }

        let interlace = InterlaceMethod::from_byte(data[12])?;

        Ok(Self {
            width,
            height,
            bit_depth,
            color_type,
            interlace,
        })
    }

    /// Bits needed to store one pixel's samples, before any byte-alignment
    /// padding a scanline's row start gets (spec §9.1: "bpp is defined as
    /// the number of bytes per complete pixel, rounding up to one").
    #[must_use]
    pub const fn bits_per_pixel(&self) -> u32 {
        self.color_type.channel_count() as u32 * self.bit_depth as u32
    }

    /// Bytes per *complete pixel*, rounded up — the "bpp" spec §9.2's
    /// filters use as their fixed left/up-pixel stride (never fractional,
    /// even for sub-byte bit depths: at 1-bit grayscale, bpp is still 1).
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> usize {
        self.bits_per_pixel().div_ceil(8) as usize
    }

    /// Bytes in one undecoded (post-defilter) scanline, not counting the
    /// leading filter-type byte each row carries in the `IDAT` stream
    /// (spec §9.1).
    #[must_use]
    pub const fn stride(&self) -> usize {
        (self.bits_per_pixel() as usize * self.width as usize).div_ceil(8)
    }

    /// Serialize back to the 13-byte `IHDR` payload — the inverse of
    /// [`Self::parse`].
    ///
    /// The compression and filter method bytes are always 0: those are the
    /// only values the spec has ever defined, and `parse` rejects anything
    /// else, so writing them as constants keeps the two directions
    /// symmetric.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::DATA_LEN] {
        let mut out = [0u8; Self::DATA_LEN];
        out[..4].copy_from_slice(&self.width.to_be_bytes());
        out[4..8].copy_from_slice(&self.height.to_be_bytes());
        out[8] = self.bit_depth;
        out[9] = self.color_type.to_byte();
        out[10] = 0;
        out[11] = 0;
        out[12] = match self.interlace {
            InterlaceMethod::None => 0,
            InterlaceMethod::Adam7 => 1,
        };
        out
    }
}

/// Why [`Ihdr::parse`] rejected a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IhdrError {
    /// `IHDR`'s data wasn't exactly [`Ihdr::DATA_LEN`] bytes.
    WrongLength(usize),
    /// Width or height was zero (spec §11.2.2 requires positive dimensions).
    ZeroDimension,
    /// The color type byte wasn't one of the five PNG defines.
    InvalidColorType(InvalidColorType),
    /// This bit depth isn't legal for this color type (spec table 11.1).
    InvalidBitDepth {
        bit_depth: u8,
        color_type: ColorType,
    },
    /// Compression method byte wasn't 0 (the only value the spec defines).
    InvalidCompressionMethod(u8),
    /// Filter method byte wasn't 0 (the only value the spec defines).
    InvalidFilterMethod(u8),
    /// Interlace method byte wasn't 0 (`None`) or 1 (`Adam7`).
    InvalidInterlaceMethod(u8),
}

impl From<InvalidColorType> for IhdrError {
    fn from(e: InvalidColorType) -> Self {
        Self::InvalidColorType(e)
    }
}

impl core::fmt::Display for IhdrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongLength(len) => {
                write!(f, "IHDR data is {len} bytes, expected {}", Ihdr::DATA_LEN)
            }
            Self::ZeroDimension => write!(f, "IHDR width or height is zero"),
            Self::InvalidColorType(e) => write!(f, "{e}"),
            Self::InvalidBitDepth {
                bit_depth,
                color_type,
            } => write!(
                f,
                "bit depth {bit_depth} is not valid for color type {color_type:?}"
            ),
            Self::InvalidCompressionMethod(m) => write!(f, "invalid compression method: {m}"),
            Self::InvalidFilterMethod(m) => write!(f, "invalid filter method: {m}"),
            Self::InvalidInterlaceMethod(m) => write!(f, "invalid interlace method: {m}"),
        }
    }
}

impl std::error::Error for IhdrError {}

#[cfg(test)]
mod tests {
    use super::{ColorType, Ihdr, IhdrError, InterlaceMethod};

    fn encode_ihdr(
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        compression: u8,
        filter: u8,
        interlace: u8,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&width.to_be_bytes());
        data.extend_from_slice(&height.to_be_bytes());
        data.push(bit_depth);
        data.push(color_type);
        data.push(compression);
        data.push(filter);
        data.push(interlace);
        data
    }

    #[test]
    fn parses_well_formed_truecolor_ihdr() {
        let data = encode_ihdr(800, 600, 8, 2, 0, 0, 0);
        let ihdr = Ihdr::parse(&data).unwrap();
        assert_eq!(ihdr.width, 800);
        assert_eq!(ihdr.height, 600);
        assert_eq!(ihdr.bit_depth, 8);
        assert_eq!(ihdr.color_type, ColorType::Truecolor);
        assert_eq!(ihdr.interlace, InterlaceMethod::None);
    }

    #[test]
    fn parses_adam7_interlace() {
        let data = encode_ihdr(10, 10, 8, 6, 0, 0, 1);
        let ihdr = Ihdr::parse(&data).unwrap();
        assert_eq!(ihdr.interlace, InterlaceMethod::Adam7);
    }

    #[test]
    fn to_bytes_round_trips_through_parse() {
        for interlace in [InterlaceMethod::None, InterlaceMethod::Adam7] {
            for (color_type, bit_depth) in [
                (ColorType::Grayscale, 1u8),
                (ColorType::Grayscale, 16),
                (ColorType::Truecolor, 8),
                (ColorType::IndexedColor, 4),
                (ColorType::GrayscaleAlpha, 8),
                (ColorType::TruecolorAlpha, 16),
            ] {
                let ihdr = Ihdr {
                    width: 640,
                    height: 480,
                    bit_depth,
                    color_type,
                    interlace,
                };
                assert_eq!(Ihdr::parse(&ihdr.to_bytes()), Ok(ihdr));
            }
        }
    }

    #[test]
    fn rejects_wrong_length() {
        let err = Ihdr::parse(&[0u8; 10]).unwrap_err();
        assert_eq!(err, IhdrError::WrongLength(10));
    }

    #[test]
    fn rejects_zero_width_or_height() {
        let data = encode_ihdr(0, 100, 8, 2, 0, 0, 0);
        assert_eq!(Ihdr::parse(&data).unwrap_err(), IhdrError::ZeroDimension);

        let data = encode_ihdr(100, 0, 8, 2, 0, 0, 0);
        assert_eq!(Ihdr::parse(&data).unwrap_err(), IhdrError::ZeroDimension);
    }

    #[test]
    fn rejects_invalid_color_type() {
        let data = encode_ihdr(10, 10, 8, 5, 0, 0, 0);
        assert!(matches!(
            Ihdr::parse(&data).unwrap_err(),
            IhdrError::InvalidColorType(_)
        ));
    }

    #[test]
    fn rejects_bit_depth_invalid_for_color_type() {
        // 16-bit depth is not legal for indexed color (max 8).
        let data = encode_ihdr(10, 10, 16, 3, 0, 0, 0);
        assert!(matches!(
            Ihdr::parse(&data).unwrap_err(),
            IhdrError::InvalidBitDepth { .. }
        ));
    }

    #[test]
    fn rejects_nonzero_compression_or_filter_method() {
        let data = encode_ihdr(10, 10, 8, 2, 1, 0, 0);
        assert!(matches!(
            Ihdr::parse(&data).unwrap_err(),
            IhdrError::InvalidCompressionMethod(1)
        ));

        let data = encode_ihdr(10, 10, 8, 2, 0, 1, 0);
        assert!(matches!(
            Ihdr::parse(&data).unwrap_err(),
            IhdrError::InvalidFilterMethod(1)
        ));
    }

    #[test]
    fn rejects_invalid_interlace_method() {
        let data = encode_ihdr(10, 10, 8, 2, 0, 0, 2);
        assert!(matches!(
            Ihdr::parse(&data).unwrap_err(),
            IhdrError::InvalidInterlaceMethod(2)
        ));
    }

    #[test]
    fn stride_and_bpp_for_8bit_truecolor() {
        // 8-bit RGB: 3 bytes/pixel, width 10 -> 30-byte stride.
        let data = encode_ihdr(10, 5, 8, 2, 0, 0, 0);
        let ihdr = Ihdr::parse(&data).unwrap();
        assert_eq!(ihdr.bytes_per_pixel(), 3);
        assert_eq!(ihdr.stride(), 30);
    }

    #[test]
    fn stride_rounds_up_for_sub_byte_depths() {
        // 1-bit grayscale, width 10 -> 10 bits -> 2 bytes (ceil(10/8)).
        let data = encode_ihdr(10, 5, 1, 0, 0, 0, 0);
        let ihdr = Ihdr::parse(&data).unwrap();
        assert_eq!(ihdr.bits_per_pixel(), 1);
        // bpp (bytes per pixel, spec's filter stride) still rounds up to 1.
        assert_eq!(ihdr.bytes_per_pixel(), 1);
        assert_eq!(ihdr.stride(), 2);
    }

    #[test]
    fn stride_for_16bit_truecolor_alpha() {
        // 16-bit RGBA: 4 channels * 16 bits = 64 bits = 8 bytes/pixel.
        let data = encode_ihdr(4, 4, 16, 6, 0, 0, 0);
        let ihdr = Ihdr::parse(&data).unwrap();
        assert_eq!(ihdr.bytes_per_pixel(), 8);
        assert_eq!(ihdr.stride(), 32);
    }
}
