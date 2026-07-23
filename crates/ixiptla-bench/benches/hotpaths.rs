//! Internal-hotpath benchmarks for `ixiptla-core`'s own pipeline stages,
//! via `divan`. Run with `cargo bench -p ixiptla-bench --bench hotpaths`.
//!
//! Unlike `end_to_end.rs` (which times whole-file decode against other
//! crates), this file isolates each stage `ixiptla_core::Png::decode`
//! chains together — defiltering, `Adam7` reassembly, and RGBA
//! expansion — so a regression in one stage doesn't hide inside an
//! aggregate "decode got slower" number. Each stage is fed pre-decoded
//! input matching exactly what that stage would see mid-pipeline (e.g.
//! `Defilter` benchmarks operate on already-inflated `IDAT` bytes, not a
//! whole PNG file), isolating that stage's own cost from zlib inflate,
//! chunk parsing, etc.
//!
//! The global allocator below is wrapped in `divan::AllocProfiler`, so
//! every result also reports alloc/dealloc/grow counts and byte totals
//! alongside timing -- lets a slowdown be diagnosed as "more CPU work"
//! vs. "more allocator churn" without a separate profiling pass.

use divan::Bencher;
use ixiptla_bench::fixtures::{SquareFixture, BENCH_SIDES};
use ixiptla_core::{ColorType, Defilter, Ihdr, InterlaceMethod, Png, RgbaExpander};
use png::ColorType as PngColorType;

// Wraps the system allocator so divan reports per-benchmark allocation
// counts/bytes alongside timing -- lets us see e.g. that
// `rgba_expand_*` allocates exactly one output Vec per call (as
// expected) versus catching an accidental extra allocation regression
// in the hotpath without needing a separate profiling run.
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// One in-memory "already inflated, still filtered" scanline stream plus
/// the `Ihdr` describing it — [`Defilter`] benchmarks' fixed input,
/// generated once per benchmark case (not on every sample iteration).
struct FilteredStream {
    ihdr: Ihdr,
    /// `height` rows of `1 + stride` bytes each: filter-type byte +
    /// filtered content, exactly what `Png::defilter_sequential` chunks
    /// through in the real pipeline.
    inflated: Vec<u8>,
}

impl FilteredStream {
    /// Build a non-interlaced, 8-bit fixture at `side x side` for
    /// `color_type`/`png_color_type` (the same colour type in both
    /// crates' vocabularies), filtered via the `png` crate's adaptive
    /// filter selection so this exercises a realistic mix of the 5
    /// filter types across rows, not just one uniformly.
    fn build(side: u32, color_type: ColorType, png_color_type: PngColorType) -> Self {
        let file = SquareFixture::new(side, png_color_type).encode();
        let ihdr = Ihdr {
            width: side,
            height: side,
            bit_depth: 8,
            color_type,
            interlace: InterlaceMethod::None,
        };
        // Decode via ixiptla_core to get `pixels` (already unfiltered) --
        // simpler than re-implementing chunk parsing/inflate here just to
        // recover the filtered bytes. This deliberately re-wraps them as
        // filter-type-None rows rather than re-deriving Sub/Up/Average/
        // Paeth-filtered ones: `Defilter::defilter_row`'s per-byte
        // reconstruction cost is identical regardless of which of the 5
        // filter types produced the input (same loop shape, same amount
        // of arithmetic per byte), so a filter-type-None row is
        // representative of all 5 for benchmarking purposes.
        let decoded = Png::decode(&file, 1 << 24).expect("fixture must decode");
        let stride = ihdr.stride();
        let mut inflated = Vec::with_capacity((1 + stride) * side as usize);
        for row in decoded.pixels.chunks_exact(stride) {
            inflated.push(0); // filter type None
            inflated.extend_from_slice(row);
        }
        Self { ihdr, inflated }
    }
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build(side, ColorType::Truecolor, PngColorType::Rgb);
    let row_len = 1 + stream.ihdr.stride();
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            let mut defilter = Defilter::new(&stream.ihdr);
            for row in inflated.chunks_mut(row_len) {
                defilter.defilter_row(divan::black_box(row)).unwrap();
            }
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor_alpha(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build(side, ColorType::TruecolorAlpha, PngColorType::Rgba);
    let row_len = 1 + stream.ihdr.stride();
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            let mut defilter = Defilter::new(&stream.ihdr);
            for row in inflated.chunks_mut(row_len) {
                defilter.defilter_row(divan::black_box(row)).unwrap();
            }
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_grayscale(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build(side, ColorType::Grayscale, PngColorType::Grayscale);
    let row_len = 1 + stream.ihdr.stride();
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            let mut defilter = Defilter::new(&stream.ihdr);
            for row in inflated.chunks_mut(row_len) {
                defilter.defilter_row(divan::black_box(row)).unwrap();
            }
            inflated
        });
}

/// [`RgbaExpander::expand`] in isolation: pre-decoded (defiltered,
/// non-interlaced) pixel bytes in, RGBA8 out -- isolates the
/// bit-unpacking/scaling/palette-lookup cost from everything upstream of
/// it in `Png::decode`.
#[divan::bench(args = BENCH_SIDES)]
fn rgba_expand_truecolor_alpha(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Rgba).encode();
    let decoded = Png::decode(&file, 1 << 24).unwrap();
    let expander = RgbaExpander::new(&decoded.ihdr, None, None);
    bencher.bench_local(|| expander.expand(divan::black_box(&decoded.pixels)).unwrap());
}

#[divan::bench(args = BENCH_SIDES)]
fn rgba_expand_grayscale(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Grayscale).encode();
    let decoded = Png::decode(&file, 1 << 24).unwrap();
    let expander = RgbaExpander::new(&decoded.ihdr, None, None);
    bencher.bench_local(|| expander.expand(divan::black_box(&decoded.pixels)).unwrap());
}

/// Full `Png::decode` (chunk parsing + zlib inflate + defilter), fed a
/// non-interlaced truecolor-alpha file -- the aggregate baseline the
/// per-stage benchmarks above break down, and also what `end_to_end.rs`
/// compares against other crates' equivalent whole-file decode.
#[divan::bench(args = BENCH_SIDES)]
fn decode_truecolor_alpha(bencher: Bencher, side: u32) {
    let file = SquareFixture::new(side, PngColorType::Rgba).encode();
    bencher.bench_local(|| Png::decode(divan::black_box(&file), 1 << 24).unwrap());
}

/// Full `Png::decode` on an `Adam7`-interlaced file -- isolates the
/// `Adam7` pass-scatter/reassembly overhead (§8.2) that non-interlaced
/// files never pay, since `png`'s own encoder can't write interlaced
/// files (see `ixiptla-conformance`'s module docs), this crate's own
/// hand-assembled builder is used instead of `SquareFixture`.
#[divan::bench(args = BENCH_SIDES)]
fn decode_adam7_truecolor(bencher: Bencher, side: u32) {
    let file = ixiptla_bench::fixtures::adam7_truecolor_fixture(side);
    bencher.bench_local(|| Png::decode(divan::black_box(&file), 1 << 24).unwrap());
}
