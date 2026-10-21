/// Which channels a pixel carries, in storage order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelLayout {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    /// Adobe-style four-channel JPEG output.
    Cmyk,
}

impl PixelLayout {
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba | Self::Cmyk => 4,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::PixelLayout;

    #[test]
    fn channel_counts_and_alpha_agree_with_the_layout_names() {
        let expected = [
            (PixelLayout::Gray, 1, false),
            (PixelLayout::GrayAlpha, 2, true),
            (PixelLayout::Rgb, 3, false),
            (PixelLayout::Rgba, 4, true),
            (PixelLayout::Cmyk, 4, false),
        ];
        for (layout, channels, alpha) in expected {
            assert_eq!(layout.channels(), channels, "{layout:?}");
            assert_eq!(layout.has_alpha(), alpha, "{layout:?}");
        }
    }
}
