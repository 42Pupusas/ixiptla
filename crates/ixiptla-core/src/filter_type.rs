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
