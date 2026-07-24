use crate::{Ihdr, PixelBits};

/// One of `Adam7`'s 7 interlace passes (spec §8.2, figure 8.3): each pass
/// picks out a regular subgrid of the full image's pixels, starting at
/// `(x_start, y_start)` and stepping by `(x_step, y_step)`.
///
/// Passes are 1-indexed in the spec's own text (`"pass 1"`..`"pass 7"`),
/// kept here as a 0-indexed array position instead since nothing but
/// display needs the human numbering.
#[derive(Debug, Clone, Copy)]
struct PassGeometry {
    x_start: u32,
    y_start: u32,
    x_step: u32,
    y_step: u32,
}

/// The 7 fixed `Adam7` pass geometries, spec §8.2 table 8.2.
const PASS_GEOMETRIES: [PassGeometry; 7] = [
    PassGeometry {
        x_start: 0,
        y_start: 0,
        x_step: 8,
        y_step: 8,
    },
    PassGeometry {
        x_start: 4,
        y_start: 0,
        x_step: 8,
        y_step: 8,
    },
    PassGeometry {
        x_start: 0,
        y_start: 4,
        x_step: 4,
        y_step: 8,
    },
    PassGeometry {
        x_start: 2,
        y_start: 0,
        x_step: 4,
        y_step: 4,
    },
    PassGeometry {
        x_start: 0,
        y_start: 2,
        x_step: 2,
        y_step: 4,
    },
    PassGeometry {
        x_start: 1,
        y_start: 0,
        x_step: 2,
        y_step: 2,
    },
    PassGeometry {
        x_start: 0,
        y_start: 1,
        x_step: 1,
        y_step: 2,
    },
];

/// One `Adam7` pass's pixel geometry against a specific full image size.
///
/// Tracks how many pixel columns/rows this pass contributes, and how to
/// map a pass-local pixel coordinate back to the full image's coordinate
/// space. Kept distinct from [`PassGeometry`] (which is just the 7 constant
/// step/start rules) because *this* also depends on the image's actual
/// width/height — a small image can have passes that contribute zero
/// pixels entirely (spec §8.2: "if the image is... small, some passes
/// will be empty"), which every consumer (row counting, `IHDR` synthesis,
/// reassembly) needs to agree on identically.
#[derive(Debug, Clone, Copy)]
pub struct Adam7Pass {
    geometry: PassGeometry,
    /// This pass's width/height in *its own* pixels, i.e. the size of the
    /// sub-image this pass decodes as if it were a standalone non-interlaced
    /// image.
    pub width: u32,
    pub height: u32,
}

impl Adam7Pass {
    /// The 7 passes' geometries against a `width` x `height` full image, in
    /// pass order (spec §8.2's numbering, pass 1 first).
    #[must_use]
    pub fn all(width: u32, height: u32) -> [Self; 7] {
        let mut passes = [Self {
            geometry: PASS_GEOMETRIES[0],
            width: 0,
            height: 0,
        }; 7];
        for (i, geometry) in PASS_GEOMETRIES.into_iter().enumerate() {
            passes[i] = Self::for_geometry(geometry, width, height);
        }
        passes
    }

    const fn for_geometry(geometry: PassGeometry, full_width: u32, full_height: u32) -> Self {
        let width = Self::pixel_count(full_width, geometry.x_start, geometry.x_step);
        let height = Self::pixel_count(full_height, geometry.y_start, geometry.y_step);
        Self {
            geometry,
            width,
            height,
        }
    }

    /// How many `{start, start+step, start+2*step, ...}` values fall below
    /// `extent` — the count of pixel columns (or rows) this pass covers
    /// along one axis.
    const fn pixel_count(extent: u32, start: u32, step: u32) -> u32 {
        if start >= extent {
            0
        } else {
            (extent - start).div_ceil(step)
        }
    }

    /// True if this pass contributes no pixels at all for this image size
    /// (spec §8.2 — happens for small images on the later, finer passes).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Map this pass's local pixel column `pass_x` to the full image's
    /// pixel column.
    #[must_use]
    pub const fn full_x(&self, pass_x: u32) -> u32 {
        self.geometry.x_start + pass_x * self.geometry.x_step
    }

    /// Map this pass's local pixel row `pass_y` to the full image's pixel
    /// row.
    #[must_use]
    pub const fn full_y(&self, pass_y: u32) -> u32 {
        self.geometry.y_start + pass_y * self.geometry.y_step
    }
}

/// Reassembles the 7 `Adam7`-interlaced passes' independently-defiltered
/// pixel data into one full image's worth of packed scanlines.
///
/// Each pass, once its own `IDAT`-slice has been inflated and defiltered
/// (by its own [`crate::Defilter`] instance — passes do *not* share
/// filter-predictor state, spec §8.2), is itself a tiny valid non-
/// interlaced image at `Adam7Pass::{width,height}`. This struct's only job
/// is scattering each pass's pixels into their true position in the full
/// image, which is exactly the bit-copy [`PixelBits::copy_pixel`]
/// performs, driven by [`Adam7Pass::full_x`]/[`Adam7Pass::full_y`].
#[derive(Debug)]
pub struct Adam7 {
    width: u32,
    height: u32,
    bits_per_pixel: u32,
    stride: usize,
}

impl Adam7 {
    #[must_use]
    pub const fn new(ihdr: &Ihdr) -> Self {
        Self {
            width: ihdr.width,
            height: ihdr.height,
            bits_per_pixel: ihdr.bits_per_pixel(),
            stride: ihdr.stride(),
        }
    }

    /// The 7 pass geometries for this image's dimensions, in pass order.
    #[must_use]
    pub fn passes(&self) -> [Adam7Pass; 7] {
        Adam7Pass::all(self.width, self.height)
    }

    /// Scatter one pass's already-defiltered pixel bytes (`pass.height`
    /// scanlines of `pass`'s own stride, content only — no filter-type
    /// bytes) into `full_image`, a `height`-scanline buffer of this
    /// image's real `stride`-byte rows.
    ///
    /// `full_image` must already be sized to `height * stride` bytes
    /// (typically zero-initialized before the first pass is scattered in —
    /// every pixel is covered by exactly one pass across all 7, so nothing
    /// is ever double-written, but a pass that starts past the image edge
    /// contributes nothing and leaves those bytes as whatever the caller
    /// initialized them to).
    pub fn scatter_pass(&self, pass: &Adam7Pass, pass_pixels: &[u8], full_image: &mut [u8]) {
        if pass.is_empty() {
            return;
        }
        let pass_stride = (self.bits_per_pixel as usize * pass.width as usize).div_ceil(8);

        for pass_y in 0..pass.height {
            let src_row = &pass_pixels[pass_y as usize * pass_stride..][..pass_stride];
            let full_y = pass.full_y(pass_y) as usize;
            let dst_row = &mut full_image[full_y * self.stride..][..self.stride];

            for pass_x in 0..pass.width {
                let full_x = pass.full_x(pass_x);
                PixelBits::copy_pixel(
                    src_row,
                    pass_x as usize,
                    dst_row,
                    full_x as usize,
                    self.bits_per_pixel,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Adam7, Adam7Pass};
    use crate::{ColorType, Ihdr, InterlaceMethod};

    #[test]
    fn pass_pixel_counts_match_spec_worked_example_8x8() {
        // Spec §8.2's canonical 8x8 example: each pass's pixel count.
        let passes = Adam7Pass::all(8, 8);
        let expected = [
            (1, 1), // pass 1: just (0,0)
            (1, 1), // pass 2: just (4,0)
            (2, 1), // pass 3: (0,4),(4,4)
            (2, 2), // pass 4: x=2,6 y=0,4
            (4, 2), // pass 5: x=0,2,4,6 y=2,6
            (4, 4), // pass 6: x=1,3,5,7 y=0,2,4,6
            (8, 4), // pass 7: all x, y=1,3,5,7
        ];
        for (pass, &(w, h)) in passes.iter().zip(expected.iter()) {
            assert_eq!((pass.width, pass.height), (w, h));
        }
    }

    #[test]
    fn all_pixels_covered_exactly_once_across_passes() {
        // For every (x, y) in a non-trivial image, exactly one pass should
        // map some (pass_x, pass_y) to it -- verify by simulating the
        // inverse: for each pass, mark every (full_x, full_y) it touches,
        // then confirm the union is the whole grid with no overlaps.
        let (width, height) = (13, 9); // deliberately not a multiple of 8
        let passes = Adam7Pass::all(width, height);
        let mut covered = vec![false; (width * height) as usize];

        for pass in &passes {
            for pass_y in 0..pass.height {
                for pass_x in 0..pass.width {
                    let fx = pass.full_x(pass_x);
                    let fy = pass.full_y(pass_y);
                    let idx = (fy * width + fx) as usize;
                    assert!(!covered[idx], "pixel ({fx},{fy}) covered twice");
                    covered[idx] = true;
                }
            }
        }
        assert!(covered.into_iter().all(|c| c), "some pixel never covered");
    }

    #[test]
    fn tiny_image_has_empty_late_passes() {
        // A 1x1 image: only pass 1 (which always includes (0,0)) is
        // non-empty; every other pass starts at an offset >= 1.
        let passes = Adam7Pass::all(1, 1);
        assert!(!passes[0].is_empty());
        for pass in &passes[1..] {
            assert!(pass.is_empty());
        }
    }

    #[test]
    fn scatter_pass_places_8bit_grayscale_pixels_correctly() {
        let ihdr = Ihdr {
            width: 2,
            height: 2,
            bit_depth: 8,
            color_type: ColorType::Grayscale,
            interlace: InterlaceMethod::Adam7,
        };
        let adam7 = Adam7::new(&ihdr);
        let passes = adam7.passes();

        // Pass 1 (x_start=0,y_start=0,step=8) covers only pixel (0,0) in a
        // 2x2 image.
        let pass1 = &passes[0];
        assert_eq!((pass1.width, pass1.height), (1, 1));
        let mut full_image = vec![0u8; 4]; // 2x2, 1 byte/pixel
        adam7.scatter_pass(pass1, &[42], &mut full_image);
        assert_eq!(full_image, vec![42, 0, 0, 0]);

        // Pass 7 (x_start=0,y_start=1,step 1x2) covers row y=1, both
        // columns, in a 2x2 image: pass pixels [7, 8] -> full row 1.
        let pass7 = &passes[6];
        assert_eq!((pass7.width, pass7.height), (2, 1));
        adam7.scatter_pass(pass7, &[7, 8], &mut full_image);
        assert_eq!(full_image, vec![42, 0, 7, 8]);
    }

    #[test]
    fn scatter_pass_handles_sub_byte_bit_depth() {
        // 1-bit grayscale, 8x1 image: pass 1 touches only pixel (0,0),
        // pass 6 touches (1,0),(3,0),(5,0),(7,0), pass 4 touches (2,0),(6,0).
        let ihdr = Ihdr {
            width: 8,
            height: 1,
            bit_depth: 1,
            color_type: ColorType::Grayscale,
            interlace: InterlaceMethod::Adam7,
        };
        let adam7 = Adam7::new(&ihdr);
        let passes = adam7.passes();
        let mut full_image = vec![0u8; 1]; // 8 pixels * 1 bit = 1 byte

        // Pass 1: pixel (0,0) = 1.
        adam7.scatter_pass(&passes[0], &[0b1000_0000], &mut full_image);
        assert_eq!(full_image[0], 0b1000_0000);

        // Pass 6 (x_start=1,step=2): pixels 1,3,5,7 <- pass-local pixels
        // 0,1,2,3, set to 1,0,1,0.
        adam7.scatter_pass(&passes[5], &[0b1010_0000], &mut full_image);
        // full pixel positions (MSB-first): 0:1(from pass1) 1:1 2:0 3:0
        // 4:0 5:1 6:0 7:0
        assert_eq!(full_image[0], 0b1100_0100);
    }
}
