use crate::ColorType;

/// A parsed `tRNS` chunk (spec §11.3.2): simple transparency for color
/// types that don't otherwise carry an alpha channel.
///
/// `tRNS`'s payload shape depends on the image's [`ColorType`] (unlike
/// `PLTE`, which is always RGB triples), so this is an enum rather than a
/// flat byte buffer — each variant is exactly the data spec §11.3.2 defines
/// for that color type, and [`Self::parse`] is the one place that maps
/// `(color_type, raw bytes)` to a validated value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transparency {
    /// Color type 0: the single gray sample value that should be treated
    /// as fully transparent, at the image's own bit depth (spec: stored
    /// as 2 bytes regardless of bit depth, high bits unused for depths
    /// under 16).
    Grayscale(u16),
    /// Color type 2: the single (R, G, B) sample triple that should be
    /// treated as fully transparent, same 2-bytes-per-sample storage as
    /// above.
    Truecolor(u16, u16, u16),
    /// Color type 3: per-palette-entry alpha values, in palette order.
    /// May be shorter than the palette (spec: entries beyond the end are
    /// assumed fully opaque, 255) but never longer.
    IndexedColor(Vec<u8>),
}

impl Transparency {
    /// Parse a `tRNS` chunk's raw data payload for an image of the given
    /// `color_type`.
    ///
    /// # Errors
    ///
    /// [`TransparencyError::NotApplicable`] if `color_type` already has an
    /// alpha channel (spec §11.3.2: `tRNS` is invalid for
    /// [`ColorType::GrayscaleAlpha`] and [`ColorType::TruecolorAlpha`]).
    /// [`TransparencyError::InvalidLength`] if the payload doesn't match
    /// the fixed size §11.3.2 defines for grayscale/truecolor, or exceeds
    /// 256 bytes for indexed color.
    pub fn parse(color_type: ColorType, data: &[u8]) -> Result<Self, TransparencyError> {
        match color_type {
            ColorType::Grayscale => {
                let bytes: [u8; 2] = data
                    .try_into()
                    .map_err(|_| TransparencyError::InvalidLength(data.len()))?;
                Ok(Self::Grayscale(u16::from_be_bytes(bytes)))
            }
            ColorType::Truecolor => {
                if data.len() != 6 {
                    return Err(TransparencyError::InvalidLength(data.len()));
                }
                let sample = |i: usize| u16::from_be_bytes([data[i], data[i + 1]]);
                Ok(Self::Truecolor(sample(0), sample(2), sample(4)))
            }
            ColorType::IndexedColor => {
                if data.len() > 256 {
                    return Err(TransparencyError::InvalidLength(data.len()));
                }
                Ok(Self::IndexedColor(data.to_vec()))
            }
            ColorType::GrayscaleAlpha | ColorType::TruecolorAlpha => {
                Err(TransparencyError::NotApplicable(color_type))
            }
        }
    }

    /// Serialize back to a `tRNS` chunk payload — the inverse of
    /// [`Self::parse`].
    ///
    /// Grayscale and truecolour samples are always stored as 2 bytes each
    /// regardless of the image's bit depth (spec §11.3.2), which is why
    /// this doesn't need to know the bit depth to write them.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Grayscale(gray) => gray.to_be_bytes().to_vec(),
            Self::Truecolor(r, g, b) => {
                let mut out = Vec::with_capacity(6);
                out.extend_from_slice(&r.to_be_bytes());
                out.extend_from_slice(&g.to_be_bytes());
                out.extend_from_slice(&b.to_be_bytes());
                out
            }
            Self::IndexedColor(alphas) => alphas.clone(),
        }
    }

    /// The alpha value for palette index `index`, per spec §11.3.2:
    /// entries past the end of an [`Self::IndexedColor`] list (or any
    /// index queried against a non-indexed variant) default to fully
    /// opaque, 255.
    #[must_use]
    pub fn alpha_for_index(&self, index: u8) -> u8 {
        match self {
            Self::IndexedColor(alphas) => alphas.get(usize::from(index)).copied().unwrap_or(255),
            Self::Grayscale(_) | Self::Truecolor(..) => 255,
        }
    }

    /// True if a grayscale pixel's raw sample value is the designated
    /// transparent value. False (never transparent via `tRNS`) for any
    /// other variant.
    #[must_use]
    pub const fn is_transparent_gray(&self, sample: u16) -> bool {
        matches!(self, Self::Grayscale(v) if *v == sample)
    }

    /// True if a truecolor pixel's raw (R, G, B) sample triple is the
    /// designated transparent color. False for any other variant.
    #[must_use]
    pub fn is_transparent_rgb(&self, r: u16, g: u16, b: u16) -> bool {
        matches!(self, Self::Truecolor(tr, tg, tb) if (*tr, *tg, *tb) == (r, g, b))
    }
}

/// Why [`Transparency::parse`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransparencyError {
    /// `tRNS` is meaningless for a color type that already has an alpha
    /// channel.
    NotApplicable(ColorType),
    /// The payload wasn't the fixed size §11.3.2 requires for
    /// grayscale/truecolor, or exceeded 256 bytes for indexed color.
    InvalidLength(usize),
}

impl core::fmt::Display for TransparencyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotApplicable(color_type) => {
                write!(f, "tRNS is not applicable to color type {color_type:?}")
            }
            Self::InvalidLength(len) => write!(f, "tRNS data length {len} is invalid"),
        }
    }
}

impl std::error::Error for TransparencyError {}

#[cfg(test)]
mod tests {
    use super::{Transparency, TransparencyError};
    use crate::ColorType;

    #[test]
    fn parses_grayscale_transparency() {
        let t = Transparency::parse(ColorType::Grayscale, &[0x00, 0x2A]).unwrap();
        assert_eq!(t, Transparency::Grayscale(42));
        assert!(t.is_transparent_gray(42));
        assert!(!t.is_transparent_gray(43));
    }

    #[test]
    fn parses_truecolor_transparency() {
        let data = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03];
        let t = Transparency::parse(ColorType::Truecolor, &data).unwrap();
        assert_eq!(t, Transparency::Truecolor(1, 2, 3));
        assert!(t.is_transparent_rgb(1, 2, 3));
        assert!(!t.is_transparent_rgb(1, 2, 4));
    }

    #[test]
    fn parses_indexed_transparency_with_default_opaque_tail() {
        let t = Transparency::parse(ColorType::IndexedColor, &[0, 128, 255]).unwrap();
        assert_eq!(t.alpha_for_index(0), 0);
        assert_eq!(t.alpha_for_index(1), 128);
        assert_eq!(t.alpha_for_index(2), 255);
        assert_eq!(t.alpha_for_index(3), 255); // past the end: opaque
    }

    #[test]
    fn to_bytes_round_trips_through_parse() {
        for (color_type, transparency) in [
            (ColorType::Grayscale, Transparency::Grayscale(0x1234)),
            (
                ColorType::Truecolor,
                Transparency::Truecolor(0x0102, 0x0304, 0x0506),
            ),
            (
                ColorType::IndexedColor,
                Transparency::IndexedColor(vec![0, 128, 255]),
            ),
        ] {
            let bytes = transparency.to_bytes();
            assert_eq!(Transparency::parse(color_type, &bytes), Ok(transparency));
        }
    }

    #[test]
    fn rejects_alpha_color_types() {
        let err = Transparency::parse(ColorType::GrayscaleAlpha, &[0, 0]).unwrap_err();
        assert!(matches!(
            err,
            TransparencyError::NotApplicable(ColorType::GrayscaleAlpha)
        ));
        let err = Transparency::parse(ColorType::TruecolorAlpha, &[0, 0]).unwrap_err();
        assert!(matches!(
            err,
            TransparencyError::NotApplicable(ColorType::TruecolorAlpha)
        ));
    }

    #[test]
    fn rejects_wrong_length_for_grayscale() {
        let err = Transparency::parse(ColorType::Grayscale, &[0]).unwrap_err();
        assert_eq!(err, TransparencyError::InvalidLength(1));
    }

    #[test]
    fn rejects_wrong_length_for_truecolor() {
        let err = Transparency::parse(ColorType::Truecolor, &[0, 0, 0, 0]).unwrap_err();
        assert_eq!(err, TransparencyError::InvalidLength(4));
    }

    #[test]
    fn rejects_too_many_indexed_entries() {
        let data = vec![255u8; 257];
        let err = Transparency::parse(ColorType::IndexedColor, &data).unwrap_err();
        assert_eq!(err, TransparencyError::InvalidLength(257));
    }
}
