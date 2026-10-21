/// A channel sample type an [`crate::Image`] can hold.
///
/// `to_u8` is the one normalisation every codec and the conformance harness
/// agree on: take the high byte of a 16-bit sample (what PNG's `STRIP_16`
/// and most viewers do) and clamp-and-round a float in `0.0..=1.0`.
pub trait Sample: Copy + Default + PartialEq + core::fmt::Debug {
    const MAX: Self;
    fn to_u8(self) -> u8;
}

impl Sample for u8 {
    const MAX: Self = Self::MAX;

    fn to_u8(self) -> u8 {
        self
    }
}

impl Sample for u16 {
    const MAX: Self = Self::MAX;

    fn to_u8(self) -> u8 {
        self.to_be_bytes()[0]
    }
}

impl Sample for f32 {
    const MAX: Self = 1.0;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn to_u8(self) -> u8 {
        (self.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::Sample;

    #[test]
    fn sixteen_bit_samples_keep_their_high_byte() {
        assert_eq!(0xABCDu16.to_u8(), 0xAB);
        assert_eq!(u16::MAX.to_u8(), 255);
    }

    #[test]
    fn float_samples_clamp_and_round() {
        assert_eq!((-0.5f32).to_u8(), 0);
        assert_eq!(0.5f32.to_u8(), 128);
        assert_eq!(2.0f32.to_u8(), 255);
    }
}
