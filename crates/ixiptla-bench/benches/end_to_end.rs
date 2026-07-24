//! End-to-end whole-file decode benchmarks: `ixiptla_core::Png::decode` +
//! `RgbaExpander::expand` versus the `png` crate's own decoder and
//! `image`'s `PngDecoder` (which itself wraps the `png` crate, but goes
//! through `image`'s own buffer-management/`ImageDecoder` trait
//! machinery — a different call path worth timing separately from raw
//! `png`). Run with `cargo bench -p ixiptla-bench --bench end_to_end`.
//!
//! All three decoders are handed the exact same file bytes (built once
//! via [`SquareFixture`], outside the timed closure) and are asked to
//! normalize to one full RGBA8/equivalent buffer, so this measures
//! comparable work per decoder rather than "decode API surface
//! convenience".
//!
//! The global allocator below is wrapped in `divan::AllocProfiler`, so
//! every result also reports each decoder's alloc/dealloc/grow counts
//! and byte totals alongside timing -- useful for telling apart "this
//! decoder is slower because it does more CPU work per byte" from "this
//! decoder is slower because it allocates/copies more".

use divan::Bencher;
use image::ImageDecoder as _;
use ixiptla_bench::fixtures::{BENCH_SIDES, SquareFixture};
use ixiptla_core::{Png, RgbaExpander};
use png::{ColorType as PngColorType, Transformations};
use std::io::Cursor;

// Wraps the system allocator so divan reports per-benchmark allocation
// counts/bytes alongside timing for each of the three decoders -- a
// decoder being merely "slower" versus "allocates far more" call for
// different fixes, and this makes the difference visible without a
// separate profiling run.
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// `ixiptla_core::Png::decode` + `RgbaExpander::expand` on a truecolor-
/// alpha file -- the same normalized-to-RGBA8 output shape the other two
/// decoders below produce, so all three are doing equivalent work.
#[divan::bench(args = BENCH_SIDES)]
fn ixiptla_truecolor_alpha(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Rgba).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let decoded = Png::decode(file, 1 << 24).unwrap();
        let expander = RgbaExpander::new(
            &decoded.ihdr,
            decoded.palette.as_ref(),
            decoded.transparency.as_ref(),
        );
        // `expand_owned`, not `expand(&decoded.pixels)`: `decoded.pixels`
        // isn't used again after this call, so this benchmark should time
        // the same call shape a real caller would use -- letting the
        // 8-bit TruecolorAlpha fast path skip its otherwise-redundant
        // full-image copy (see `RgbaExpander::expand_owned`'s doc
        // comment) rather than measuring a copy no real caller need pay
        // for.
        expander.expand_owned(decoded.pixels).unwrap()
    });
}

/// The `png` crate's own decoder, `EXPAND | ALPHA | STRIP_16`-normalized
/// (same transform set `ixiptla-conformance`'s `reference_decoder` uses)
/// -- for RGBA8 input this is already 4 bytes/pixel with no further
/// reshaping needed, so it's a fair "decode to RGBA8" comparison against
/// `ixiptla_truecolor_alpha` above.
#[divan::bench(args = BENCH_SIDES)]
fn png_crate_truecolor_alpha(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Rgba).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let mut decoder = png::Decoder::new(Cursor::new(file));
        decoder.set_transformations(
            Transformations::EXPAND | Transformations::ALPHA | Transformations::STRIP_16,
        );
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        buf
    });
}

/// `image`'s `PngDecoder` + `ImageDecoder::read_image` -- `image`'s own
/// buffer-management path on top of the same underlying `png` crate,
/// timed separately since real-world callers usually reach for `image`
/// rather than `png` directly.
#[divan::bench(args = BENCH_SIDES)]
fn image_crate_truecolor_alpha(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Rgba).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let decoder = image::codecs::png::PngDecoder::new(Cursor::new(file)).unwrap();
        let mut buf = vec![0u8; usize::try_from(decoder.total_bytes()).unwrap()];
        decoder.read_image(&mut buf).unwrap();
        buf
    });
}

/// Same three-way comparison, but on an 8-bit grayscale file -- exercises
/// each decoder's "expand a narrower colour type up to RGBA" path, not
/// just the already-RGBA case above.
#[divan::bench(args = BENCH_SIDES)]
fn ixiptla_grayscale(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Grayscale).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let decoded = Png::decode(file, 1 << 24).unwrap();
        let expander = RgbaExpander::new(
            &decoded.ihdr,
            decoded.palette.as_ref(),
            decoded.transparency.as_ref(),
        );
        expander.expand(&decoded.pixels).unwrap()
    });
}

#[divan::bench(args = BENCH_SIDES)]
fn png_crate_grayscale(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Grayscale).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let mut decoder = png::Decoder::new(Cursor::new(file));
        decoder.set_transformations(
            Transformations::EXPAND | Transformations::ALPHA | Transformations::STRIP_16,
        );
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        buf
    });
}

/// **Not** a fair comparison to `ixiptla_grayscale`/`png_crate_grayscale`
/// above: `image`'s `PngDecoder::color_type()` (and hence `total_bytes`/
/// `read_image`) reports the PNG's *native* colour type -- `L8` (1 byte/
/// pixel) for a `Grayscale` source -- rather than forcing an RGBA8
/// expansion the way `ixiptla_grayscale` and `png_crate_grayscale`'s
/// explicit `Transformations::EXPAND | ALPHA` both do. Left in (renamed
/// to say so) purely as a "how fast is `image`'s own preferred/native
/// decode path" data point; `image_crate_grayscale_to_rgba8` below is
/// the actual apples-to-apples one, doing the same `L8 -> RGBA8` fan-out
/// this crate's `RgbaExpander` does, via `DynamicImage::into_rgba8`.
#[divan::bench(args = BENCH_SIDES)]
fn image_crate_grayscale_native_l8(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Grayscale).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let decoder = image::codecs::png::PngDecoder::new(Cursor::new(file)).unwrap();
        let mut buf = vec![0u8; usize::try_from(decoder.total_bytes()).unwrap()];
        decoder.read_image(&mut buf).unwrap();
        buf
    });
}

/// The actual apples-to-apples grayscale comparison: `image`'s own
/// `L8 -> RGBA8` expansion (`DynamicImage::into_rgba8`), so this does the
/// same amount of output work (`width * height * 4` bytes) as
/// `ixiptla_grayscale` and `png_crate_grayscale` above --
/// `image_crate_grayscale_native_l8` above instead stops at `image`'s
/// native `L8`, 4x less output, which made it look implausibly fast
/// next to the other two decoders.
#[divan::bench(args = BENCH_SIDES)]
fn image_crate_grayscale_to_rgba8(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Grayscale).encode();
    bencher.bench_local(|| {
        let file = divan::black_box(&file);
        let img = image::load_from_memory(file).unwrap();
        img.into_rgba8().into_raw()
    });
}
