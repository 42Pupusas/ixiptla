/// A parsed `PLTE` chunk (spec §11.2.3): up to 256 RGB entries, indexed by
/// an [`crate::ColorType::IndexedColor`] pixel's sample value.
///
/// Kept as its own tiny struct (rather than a bare `Vec<[u8;3]>` threaded
/// around) so "a `PLTE` chunk's bytes must be a multiple of 3, and at most
/// 256 entries" is validated exactly once at parse time, and so palette
/// lookups have one obvious owner with one bounds-checked accessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    entries: Vec<[u8; 3]>,
}

impl Palette {
    /// Parse a `PLTE` chunk's raw data payload: a flat sequence of R,G,B
    /// byte triples (spec §11.2.3).
    ///
    /// # Errors
    ///
    /// [`PaletteError::InvalidLength`] if the data isn't a whole number of
    /// 3-byte entries, or [`PaletteError::TooManyEntries`] if there are
    /// more than the 256 an 8-bit index could ever address.
    pub fn parse(data: &[u8]) -> Result<Self, PaletteError> {
        if data.len() % 3 != 0 {
            return Err(PaletteError::InvalidLength(data.len()));
        }
        let entries: Vec<[u8; 3]> = data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        if entries.len() > 256 {
            return Err(PaletteError::TooManyEntries(entries.len()));
        }
        Ok(Self { entries })
    }

    /// How many entries this palette has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if this palette has no entries at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the RGB triple for palette index `index`.
    ///
    /// # Errors
    ///
    /// [`PaletteError::IndexOutOfRange`] if `index` isn't a valid entry —
    /// an indexed-color pixel sample pointing past the end of a
    /// shorter-than-expected `PLTE` chunk (spec-violating input, not
    /// something a well-formed encoder produces).
    pub fn get(&self, index: u8) -> Result<[u8; 3], PaletteError> {
        self.entries
            .get(usize::from(index))
            .copied()
            .ok_or(PaletteError::IndexOutOfRange {
                index,
                len: self.entries.len(),
            })
    }
}

/// Why [`Palette::parse`] or [`Palette::get`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteError {
    /// `PLTE` data wasn't a whole number of 3-byte RGB entries.
    InvalidLength(usize),
    /// More than 256 entries were present (spec §11.2.3: never legal, no
    /// index could ever reach past 255).
    TooManyEntries(usize),
    /// A pixel's palette index pointed past the parsed palette's length.
    IndexOutOfRange { index: u8, len: usize },
}

impl core::fmt::Display for PaletteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLength(len) => {
                write!(f, "PLTE data length {len} is not a multiple of 3")
            }
            Self::TooManyEntries(n) => write!(f, "PLTE has {n} entries, more than the max 256"),
            Self::IndexOutOfRange { index, len } => {
                write!(
                    f,
                    "palette index {index} is out of range for a {len}-entry palette"
                )
            }
        }
    }
}

impl std::error::Error for PaletteError {}

#[cfg(test)]
mod tests {
    use super::{Palette, PaletteError};

    #[test]
    fn parses_well_formed_palette() {
        let data = [255u8, 0, 0, 0, 255, 0, 0, 0, 255]; // red, green, blue
        let palette = Palette::parse(&data).unwrap();
        assert_eq!(palette.len(), 3);
        assert_eq!(palette.get(0).unwrap(), [255, 0, 0]);
        assert_eq!(palette.get(1).unwrap(), [0, 255, 0]);
        assert_eq!(palette.get(2).unwrap(), [0, 0, 255]);
    }

    #[test]
    fn empty_palette_is_valid_but_empty() {
        let palette = Palette::parse(&[]).unwrap();
        assert!(palette.is_empty());
    }

    #[test]
    fn rejects_length_not_a_multiple_of_three() {
        let err = Palette::parse(&[1, 2, 3, 4]).unwrap_err();
        assert_eq!(err, PaletteError::InvalidLength(4));
    }

    #[test]
    fn rejects_more_than_256_entries() {
        let data = vec![0u8; 257 * 3];
        let err = Palette::parse(&data).unwrap_err();
        assert_eq!(err, PaletteError::TooManyEntries(257));
    }

    #[test]
    fn rejects_out_of_range_index() {
        let data = [1u8, 2, 3];
        let palette = Palette::parse(&data).unwrap();
        let err = palette.get(1).unwrap_err();
        assert_eq!(err, PaletteError::IndexOutOfRange { index: 1, len: 1 });
    }
}
