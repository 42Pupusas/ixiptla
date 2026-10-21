use crate::{ImageError, PixelLayout, Sample};

/// A decoded image: interleaved samples, row-major, no padding between rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image<S> {
    width: u32,
    height: u32,
    layout: PixelLayout,
    samples: Vec<S>,
}

impl<S: Sample> Image<S> {
    /// Wrap `samples`, which must hold exactly `width * height` pixels of
    /// `layout`.
    pub fn new(
        width: u32,
        height: u32,
        layout: PixelLayout,
        samples: Vec<S>,
    ) -> Result<Self, ImageError> {
        let expected = Self::sample_count(width, height, layout)?;
        if samples.len() != expected {
            return Err(ImageError::WrongSampleCount {
                expected,
                actual: samples.len(),
            });
        }
        Ok(Self {
            width,
            height,
            layout,
            samples,
        })
    }

    /// An image of the given shape with every sample at its default (zero).
    pub fn blank(width: u32, height: u32, layout: PixelLayout) -> Result<Self, ImageError> {
        let count = Self::sample_count(width, height, layout)?;
        Self::new(width, height, layout, vec![S::default(); count])
    }

    fn sample_count(width: u32, height: u32, layout: PixelLayout) -> Result<usize, ImageError> {
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(layout.channels()))
            .ok_or(ImageError::TooLarge)
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn layout(&self) -> PixelLayout {
        self.layout
    }

    #[must_use]
    pub fn samples(&self) -> &[S] {
        &self.samples
    }

    pub fn samples_mut(&mut self) -> &mut [S] {
        &mut self.samples
    }

    #[must_use]
    pub fn into_samples(self) -> Vec<S> {
        self.samples
    }

    /// Samples per row.
    #[must_use]
    pub const fn row_len(&self) -> usize {
        self.width as usize * self.layout.channels()
    }

    /// One row's samples.
    ///
    /// # Panics
    ///
    /// If `y >= height`.
    #[must_use]
    pub fn row(&self, y: u32) -> &[S] {
        assert!(y < self.height, "row {y} out of {}", self.height);
        let len = self.row_len();
        &self.samples[y as usize * len..][..len]
    }

    /// One pixel's channels.
    ///
    /// # Panics
    ///
    /// If `(x, y)` is outside the image.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> &[S] {
        assert!(x < self.width, "column {x} out of {}", self.width);
        let channels = self.layout.channels();
        &self.row(y)[x as usize * channels..][..channels]
    }

    /// The same pixels as interleaved RGBA8, the common form conformance
    /// tests compare. Gray replicates into R, G and B; missing alpha is
    /// opaque. CMYK is converted naively (`255 - C - K`, saturating), which
    /// is only meant for diffing two decoders, not for display.
    #[must_use]
    pub fn to_rgba8(&self) -> Image<u8> {
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize * 4);
        for pixel in self.samples.chunks_exact(self.layout.channels()) {
            let rgba = match self.layout {
                PixelLayout::Gray => {
                    let v = pixel[0].to_u8();
                    [v, v, v, 255]
                }
                PixelLayout::GrayAlpha => {
                    let v = pixel[0].to_u8();
                    [v, v, v, pixel[1].to_u8()]
                }
                PixelLayout::Rgb => [pixel[0].to_u8(), pixel[1].to_u8(), pixel[2].to_u8(), 255],
                PixelLayout::Rgba => [
                    pixel[0].to_u8(),
                    pixel[1].to_u8(),
                    pixel[2].to_u8(),
                    pixel[3].to_u8(),
                ],
                PixelLayout::Cmyk => {
                    let k = pixel[3].to_u8();
                    let channel = |c: S| 255u8.saturating_sub(c.to_u8()).saturating_sub(k);
                    [channel(pixel[0]), channel(pixel[1]), channel(pixel[2]), 255]
                }
            };
            out.extend_from_slice(&rgba);
        }
        Image {
            width: self.width,
            height: self.height,
            layout: PixelLayout::Rgba,
            samples: out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Image;
    use crate::{ImageError, PixelLayout};

    #[test]
    fn construction_checks_the_sample_count() {
        assert!(Image::<u8>::new(2, 3, PixelLayout::Rgb, vec![0; 18]).is_ok());
        assert_eq!(
            Image::<u8>::new(2, 3, PixelLayout::Rgb, vec![0; 17]).unwrap_err(),
            ImageError::WrongSampleCount {
                expected: 18,
                actual: 17
            }
        );
    }

    #[test]
    fn overflowing_dimensions_are_rejected_before_allocating() {
        assert_eq!(
            Image::<u8>::blank(u32::MAX, u32::MAX, PixelLayout::Rgba).unwrap_err(),
            ImageError::TooLarge
        );
    }

    #[test]
    fn rows_and_pixels_address_the_interleaved_buffer() {
        let samples: Vec<u8> = (0..24).collect();
        let image = Image::new(4, 3, PixelLayout::GrayAlpha, samples).unwrap();
        assert_eq!(image.row(1), &[8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(image.pixel(2, 2), &[20, 21]);
    }

    #[test]
    fn rgba8_normalisation_covers_every_layout() {
        let gray = Image::new(1, 1, PixelLayout::Gray, vec![0x1234u16]).unwrap();
        assert_eq!(gray.to_rgba8().samples(), &[0x12, 0x12, 0x12, 255]);

        let gray_alpha = Image::new(1, 1, PixelLayout::GrayAlpha, vec![7u8, 9]).unwrap();
        assert_eq!(gray_alpha.to_rgba8().samples(), &[7, 7, 7, 9]);

        let rgb = Image::new(1, 1, PixelLayout::Rgb, vec![1.0f32, 0.0, 0.5]).unwrap();
        assert_eq!(rgb.to_rgba8().samples(), &[255, 0, 128, 255]);

        let cmyk = Image::new(1, 1, PixelLayout::Cmyk, vec![0u8, 255, 100, 50]).unwrap();
        assert_eq!(cmyk.to_rgba8().samples(), &[205, 0, 105, 255]);
    }
}
