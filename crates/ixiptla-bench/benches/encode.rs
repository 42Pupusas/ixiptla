//! End-to-end whole-file *encode* benchmarks: `ixiptla_core::Encoder`
//! versus the `png` crate's encoder, given identical raw pixel buffers.
//! Run with `cargo bench -p ixiptla-bench --bench encode`.
//!
//! # Speed is only half the result
//!
//! An encoder that is twice as fast but emits 30% larger files has not
//! won anything — for PNG, output size is usually the point. So the
//! `size_report` benchmark below prints bytes-out for every encoder and
//! configuration alongside these timings, and both numbers should be read
//! together.
//!
//! Content is [`Content::Photographic`] throughout: smooth, spatially
//! correlated data of the kind PNG's filters exist to exploit. Timing the
//! encoders on xorshift noise instead would measure DEFLATE's worst case
//! and make every filter strategy look identical on size.

use divan::Bencher;
use ixiptla_bench::raw_image::{Content, RawImage};
use ixiptla_core::{ColorType, Encoder, FilterStrategy, FilterType};
use png::{BitDepth, Encoder as PngEncoder, Filter as PngFilter};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// Sizes swept by the encode suite. Smaller than the decode suite's top
/// end because encoding is the slower direction and 1024 already makes
/// the per-row costs dominate.
const ENCODE_SIDES: [u32; 3] = [64, 256, 1024];

/// Encode with the `png` crate, matching our output shape as closely as
/// its API allows.
fn encode_with_png_crate(image: &RawImage, filter: PngFilter) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = PngEncoder::new(&mut out, image.ihdr.width, image.ihdr.height);
    encoder.set_color(image.png_color_type());
    encoder.set_depth(BitDepth::Eight);
    encoder.set_filter(filter);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&image.pixels).unwrap();
    writer.finish().unwrap();
    out
}

#[divan::bench(args = ENCODE_SIDES)]
fn ixiptla_truecolor_alpha_adaptive(bencher: Bencher, side: u32) {
    let image = RawImage::square(side, ColorType::TruecolorAlpha, Content::Photographic);
    let encoder = Encoder::new(image.ihdr);
    bencher.bench_local(|| encoder.encode(divan::black_box(&image.pixels)).unwrap());
}

#[divan::bench(args = ENCODE_SIDES)]
fn ixiptla_truecolor_alpha_fixed_up(bencher: Bencher, side: u32) {
    // The cheapest useful strategy: one filter, no per-row scoring.
    let image = RawImage::square(side, ColorType::TruecolorAlpha, Content::Photographic);
    let encoder =
        Encoder::new(image.ihdr).with_filter_strategy(FilterStrategy::Fixed(FilterType::Up));
    bencher.bench_local(|| encoder.encode(divan::black_box(&image.pixels)).unwrap());
}

#[divan::bench(args = ENCODE_SIDES)]
fn ixiptla_truecolor_alpha_fixed_none(bencher: Bencher, side: u32) {
    // No filtering at all: isolates DEFLATE's share of the total.
    let image = RawImage::square(side, ColorType::TruecolorAlpha, Content::Photographic);
    let encoder =
        Encoder::new(image.ihdr).with_filter_strategy(FilterStrategy::Fixed(FilterType::None));
    bencher.bench_local(|| encoder.encode(divan::black_box(&image.pixels)).unwrap());
}

#[divan::bench(args = ENCODE_SIDES)]
fn png_crate_truecolor_alpha_adaptive(bencher: Bencher, side: u32) {
    let image = RawImage::square(side, ColorType::TruecolorAlpha, Content::Photographic);
    bencher.bench_local(|| encode_with_png_crate(divan::black_box(&image), PngFilter::Adaptive));
}

#[divan::bench(args = ENCODE_SIDES)]
fn png_crate_truecolor_alpha_fixed_up(bencher: Bencher, side: u32) {
    let image = RawImage::square(side, ColorType::TruecolorAlpha, Content::Photographic);
    bencher.bench_local(|| encode_with_png_crate(divan::black_box(&image), PngFilter::Up));
}

#[divan::bench(args = ENCODE_SIDES)]
fn ixiptla_grayscale_adaptive(bencher: Bencher, side: u32) {
    // bpp == 1, where the decode side found its worst per-byte costs.
    let image = RawImage::square(side, ColorType::Grayscale, Content::Photographic);
    let encoder = Encoder::new(image.ihdr);
    bencher.bench_local(|| encoder.encode(divan::black_box(&image.pixels)).unwrap());
}

#[divan::bench(args = ENCODE_SIDES)]
fn png_crate_grayscale_adaptive(bencher: Bencher, side: u32) {
    let image = RawImage::square(side, ColorType::Grayscale, Content::Photographic);
    bencher.bench_local(|| encode_with_png_crate(divan::black_box(&image), PngFilter::Adaptive));
}

#[divan::bench(args = ENCODE_SIDES)]
fn ixiptla_truecolor_adaptive(bencher: Bencher, side: u32) {
    let image = RawImage::square(side, ColorType::Truecolor, Content::Photographic);
    let encoder = Encoder::new(image.ihdr);
    bencher.bench_local(|| encoder.encode(divan::black_box(&image.pixels)).unwrap());
}

#[divan::bench(args = ENCODE_SIDES)]
fn png_crate_truecolor_adaptive(bencher: Bencher, side: u32) {
    let image = RawImage::square(side, ColorType::Truecolor, Content::Photographic);
    bencher.bench_local(|| encode_with_png_crate(divan::black_box(&image), PngFilter::Adaptive));
}

/// Prints output sizes for every encoder/strategy pair.
///
/// Not a timing benchmark — it runs once and reports bytes, because a
/// speed win that costs compression ratio is not a win. divan has no
/// "report a number" mode, so this piggybacks on a trivial benchmark and
/// prints to stderr on its first run.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn size_report() {
    use std::sync::Once;
    static ONCE: Once = Once::new();

    ONCE.call_once(|| {
        eprintln!("\n=== output size (bytes, photographic content) ===");
        for color_type in [
            ColorType::Grayscale,
            ColorType::Truecolor,
            ColorType::TruecolorAlpha,
        ] {
            for side in [256u32, 1024] {
                let image = RawImage::square(side, color_type, Content::Photographic);
                let raw = image.pixels.len();

                let adaptive = Encoder::new(image.ihdr).encode(&image.pixels).unwrap().len();
                let fixed_up = Encoder::new(image.ihdr)
                    .with_filter_strategy(FilterStrategy::Fixed(FilterType::Up))
                    .encode(&image.pixels)
                    .unwrap()
                    .len();
                let unfiltered = Encoder::new(image.ihdr)
                    .with_filter_strategy(FilterStrategy::Fixed(FilterType::None))
                    .encode(&image.pixels)
                    .unwrap()
                    .len();
                let png_adaptive = encode_with_png_crate(&image, PngFilter::Adaptive).len();
                let png_up = encode_with_png_crate(&image, PngFilter::Up).len();

                eprintln!(
                    "{color_type:?} {side}x{side}: raw {raw} | ixiptla adaptive {adaptive} \
                     fixed-up {fixed_up} none {unfiltered} | png adaptive {png_adaptive} \
                     up {png_up}"
                );
            }
        }
        eprintln!();
    });
}
