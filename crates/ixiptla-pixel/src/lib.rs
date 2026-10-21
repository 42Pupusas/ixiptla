//! `ixiptla-pixel` — the decoded-image representation every `ixiptla` codec
//! produces and consumes, so callers and the conformance harness handle PNG,
//! JPEG and JPEG XL output the same way.

mod image;
mod image_error;
mod pixel_layout;
mod sample;

pub use image::Image;
pub use image_error::ImageError;
pub use pixel_layout::PixelLayout;
pub use sample::Sample;
