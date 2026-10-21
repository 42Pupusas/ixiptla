/// The filter-type byte prefixing every scanline in the defiltered `IDAT`
/// stream (spec §9.2, table 9.1) — one of 5 defined reconstruction
/// functions, chosen per-row by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// `Recon(x) = Filt(x)` — no filtering.
    None,
    /// `Recon(x) = Filt(x) + Recon(a)` where `a` is the pixel to the left.
    Sub,
    /// `Recon(x) = Filt(x) + Recon(b)` where `b` is the pixel above.
    Up,
    /// `Recon(x) = Filt(x) + floor((Recon(a) + Recon(b)) / 2)`.
    Average,
    /// `Recon(x) = Filt(x) + PaethPredictor(Recon(a), Recon(b), Recon(c))`
    /// where `c` is the pixel diagonally above-left.
    Paeth,
}

impl FilterType {
    pub const fn from_byte(byte: u8) -> Result<Self, InvalidFilterType> {
        match byte {
            0 => Ok(Self::None),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Up),
            3 => Ok(Self::Average),
            4 => Ok(Self::Paeth),
            other => Err(InvalidFilterType(other)),
        }
    }

    /// The filter-type byte an encoder writes at the head of a scanline —
    /// the exact inverse of [`Self::from_byte`].
    ///
    /// Kept next to `from_byte` (rather than derived via `as u8` at call
    /// sites) so the two directions of the spec's table 9.1 mapping are
    /// defined together and can't silently drift apart; the round-trip is
    /// asserted in this module's tests.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Sub => 1,
            Self::Up => 2,
            Self::Average => 3,
            Self::Paeth => 4,
        }
    }

    /// All five filter types, in spec order — the candidate set an encoder
    /// scores per row when choosing a filter.
    pub const ALL: [Self; 5] = [Self::None, Self::Sub, Self::Up, Self::Average, Self::Paeth];
}

/// A scanline's leading filter-type byte was outside 0..=4 (spec §9.2's
/// five defined values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidFilterType(pub u8);

impl core::fmt::Display for InvalidFilterType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid PNG scanline filter type byte: {}", self.0)
    }
}

impl std::error::Error for InvalidFilterType {}

#[cfg(test)]
mod tests {
    use super::FilterType;

    #[test]
    fn to_byte_round_trips_through_from_byte() {
        // Guards the two halves of table 9.1 against drifting apart: every
        // variant must survive a full encode/decode cycle, and every byte
        // the decoder accepts must map back to the same byte.
        for filter in FilterType::ALL {
            assert_eq!(FilterType::from_byte(filter.to_byte()), Ok(filter));
        }
        for byte in 0..=4u8 {
            assert_eq!(FilterType::from_byte(byte).unwrap().to_byte(), byte);
        }
    }

    #[test]
    fn all_lists_every_variant_in_spec_order() {
        assert_eq!(FilterType::ALL.len(), 5);
        for (index, filter) in FilterType::ALL.iter().enumerate() {
            assert_eq!(u8::try_from(index).unwrap(), filter.to_byte());
        }
    }

    #[test]
    fn decodes_all_five_filter_types() {
        assert_eq!(FilterType::from_byte(0), Ok(FilterType::None));
        assert_eq!(FilterType::from_byte(1), Ok(FilterType::Sub));
        assert_eq!(FilterType::from_byte(2), Ok(FilterType::Up));
        assert_eq!(FilterType::from_byte(3), Ok(FilterType::Average));
        assert_eq!(FilterType::from_byte(4), Ok(FilterType::Paeth));
    }

    #[test]
    fn rejects_out_of_range_byte() {
        assert!(FilterType::from_byte(5).is_err());
        assert!(FilterType::from_byte(255).is_err());
    }
}
