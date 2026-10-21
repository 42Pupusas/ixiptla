/// PNG's `IHDR` colour type byte (spec §11.2.2, table 11.1).
///
/// Determines which channels a pixel carries, and — combined with bit
/// depth — how many bits wide one pixel is before filtering/compression
/// ever enters the picture.
///
/// Kept as its own enum (rather than a bare `u8` threaded everywhere) so
/// `channel_count()` has one obvious owner, and so an invalid colour type
/// byte is rejected exactly once, at parse time, instead of silently
/// propagating as a magic number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    /// Each pixel is a single greyscale sample.
    Grayscale,
    /// Each pixel is red, green, blue samples.
    Truecolor,
    /// Each pixel is a single byte/index into the `PLTE` palette chunk.
    IndexedColor,
    /// Each pixel is a single greyscale sample plus alpha.
    GrayscaleAlpha,
    /// Each pixel is red, green, blue, alpha samples.
    TruecolorAlpha,
}

impl ColorType {
    /// Decode the raw `IHDR` colour type byte, per spec table 11.1. Not
    /// every byte value is valid (1, 5, 7 are reserved/unused).
    pub const fn from_byte(byte: u8) -> Result<Self, InvalidColorType> {
        match byte {
            0 => Ok(Self::Grayscale),
            2 => Ok(Self::Truecolor),
            3 => Ok(Self::IndexedColor),
            4 => Ok(Self::GrayscaleAlpha),
            6 => Ok(Self::TruecolorAlpha),
            _ => Err(InvalidColorType(byte)),
        }
    }

    /// The `IHDR` colour type byte for this variant — the exact inverse of
    /// [`Self::from_byte`], kept beside it so both directions of table 11.1
    /// are defined together.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::Grayscale => 0,
            Self::Truecolor => 2,
            Self::IndexedColor => 3,
            Self::GrayscaleAlpha => 4,
            Self::TruecolorAlpha => 6,
        }
    }

    /// Samples per pixel — the multiplier between bit depth and the actual
    /// bits-per-pixel a scanline's filter/defilter math needs (spec
    /// §11.2.2's "Channels" column).
    #[must_use]
    pub const fn channel_count(self) -> u8 {
        match self {
            Self::Grayscale | Self::IndexedColor => 1,
            Self::GrayscaleAlpha => 2,
            Self::Truecolor => 3,
            Self::TruecolorAlpha => 4,
        }
    }

    /// Which bit depths are legal for this colour type (spec table 11.1's
    /// "Allowed bit depths" column) — indexed-color and greyscale allow the
    /// sub-byte depths 1/2/4 that truecolor/alpha forms don't.
    #[must_use]
    pub const fn allowed_bit_depths(self) -> &'static [u8] {
        match self {
            Self::Grayscale => &[1, 2, 4, 8, 16],
            Self::IndexedColor => &[1, 2, 4, 8],
            Self::Truecolor | Self::GrayscaleAlpha | Self::TruecolorAlpha => &[8, 16],
        }
    }
}

/// The `IHDR` colour type byte was not one of the five values PNG defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidColorType(pub u8);

impl core::fmt::Display for InvalidColorType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid PNG color type byte: {}", self.0)
    }
}

impl std::error::Error for InvalidColorType {}

#[cfg(test)]
mod tests {
    use super::ColorType;

    #[test]
    fn decodes_all_valid_color_types() {
        assert_eq!(ColorType::from_byte(0), Ok(ColorType::Grayscale));
        assert_eq!(ColorType::from_byte(2), Ok(ColorType::Truecolor));
        assert_eq!(ColorType::from_byte(3), Ok(ColorType::IndexedColor));
        assert_eq!(ColorType::from_byte(4), Ok(ColorType::GrayscaleAlpha));
        assert_eq!(ColorType::from_byte(6), Ok(ColorType::TruecolorAlpha));
    }

    #[test]
    fn to_byte_round_trips_through_from_byte() {
        for color_type in [
            ColorType::Grayscale,
            ColorType::Truecolor,
            ColorType::IndexedColor,
            ColorType::GrayscaleAlpha,
            ColorType::TruecolorAlpha,
        ] {
            assert_eq!(ColorType::from_byte(color_type.to_byte()), Ok(color_type));
        }
    }

    #[test]
    fn rejects_reserved_values() {
        for reserved in [1, 5, 7, 8, 255] {
            assert!(ColorType::from_byte(reserved).is_err());
        }
    }

    #[test]
    fn channel_counts_match_spec_table() {
        assert_eq!(ColorType::Grayscale.channel_count(), 1);
        assert_eq!(ColorType::Truecolor.channel_count(), 3);
        assert_eq!(ColorType::IndexedColor.channel_count(), 1);
        assert_eq!(ColorType::GrayscaleAlpha.channel_count(), 2);
        assert_eq!(ColorType::TruecolorAlpha.channel_count(), 4);
    }

    #[test]
    fn indexed_color_disallows_16_bit_depth() {
        assert!(!ColorType::IndexedColor.allowed_bit_depths().contains(&16));
        assert!(ColorType::Grayscale.allowed_bit_depths().contains(&16));
    }

    #[test]
    fn truecolor_disallows_sub_byte_depths() {
        assert!(!ColorType::Truecolor.allowed_bit_depths().contains(&4));
        assert!(ColorType::Truecolor.allowed_bit_depths().contains(&8));
    }
}
