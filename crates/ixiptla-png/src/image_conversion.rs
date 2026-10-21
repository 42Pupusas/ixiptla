use crate::{ExpandError, Png, RgbaExpander};
use ixiptla_pixel::{Image, PixelLayout};

impl Png {
    /// This image as RGBA8 in the format-neutral [`Image`], the form the
    /// JPEG and JPEG XL codecs also produce.
    ///
    /// # Panics
    ///
    /// Never for a `Png` from [`Png::decode`]; the expander's output length
    /// is `width * height * 4` by construction.
    pub fn into_rgba8_image(self) -> Result<Image<u8>, ExpandError> {
        let rgba = RgbaExpander::new(
            &self.ihdr,
            self.palette.as_ref(),
            self.transparency.as_ref(),
        )
        .expand_owned(self.pixels)?;
        Ok(
            Image::new(self.ihdr.width, self.ihdr.height, PixelLayout::Rgba, rgba)
                .expect("RgbaExpander yields exactly width * height * 4 bytes"),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{ColorType, Encoder, Ihdr, InterlaceMethod, Png};
    use ixiptla_pixel::PixelLayout;

    #[test]
    fn a_decoded_png_becomes_an_rgba8_image() {
        let ihdr = Ihdr {
            width: 3,
            height: 2,
            bit_depth: 8,
            color_type: ColorType::Grayscale,
            interlace: InterlaceMethod::None,
        };
        let file = Encoder::new(ihdr)
            .encode(&[0, 50, 100, 150, 200, 250])
            .unwrap();
        let image = Png::decode(&file, 1 << 20)
            .unwrap()
            .into_rgba8_image()
            .unwrap();
        assert_eq!((image.width(), image.height()), (3, 2));
        assert_eq!(image.layout(), PixelLayout::Rgba);
        assert_eq!(image.pixel(1, 1), &[200, 200, 200, 255]);
    }
}
