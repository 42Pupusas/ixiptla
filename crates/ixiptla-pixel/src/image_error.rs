/// Why an [`crate::Image`] could not be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageError {
    /// `width * height * channels` overflows `usize`.
    TooLarge,
    /// The sample buffer is not exactly `width * height * channels` long.
    WrongSampleCount { expected: usize, actual: usize },
}

impl core::fmt::Display for ImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooLarge => write!(f, "image dimensions overflow the address space"),
            Self::WrongSampleCount { expected, actual } => {
                write!(f, "image has {actual} samples, expected {expected}")
            }
        }
    }
}

impl std::error::Error for ImageError {}
