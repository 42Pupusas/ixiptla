//! Internal-hotpath benchmarks for `ixiptla-png`'s own pipeline stages,
//! via `divan`. Run with `cargo bench -p ixiptla-bench --bench hotpaths`.
//!
//! Unlike `end_to_end.rs` (which times whole-file decode against other
//! crates), this file isolates each stage `ixiptla_png::Png::decode`
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
use ixiptla_bench::fixtures::{BENCH_SIDES, SquareFixture};
use ixiptla_png::{ColorType, Crc32, Defilter, Ihdr, InterlaceMethod, Png, RgbaExpander};
use png::ColorType as PngColorType;
use png::Filter as PngFilter;

/// Concatenate a PNG file's `IDAT` chunk payloads in file order (spec
/// §5.6), skipping everything else -- a minimal, dependency-free chunk
/// walk (just enough to find `IDAT`s, no CRC/PLTE/tRNS handling) used only
/// to recover `FilteredStream::build`'s genuinely-filtered scanline bytes
/// from a `SquareFixture`-encoded file without going through
/// `Png::decode` (which would defilter them before this benchmark gets a
/// chance to time that step).
fn extract_idat(file: &[u8]) -> Vec<u8> {
    let mut idat = Vec::new();
    let mut pos = 8; // skip the 8-byte PNG signature
    while pos + 8 <= file.len() {
        let len = u32::from_be_bytes(file[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &file[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start + len;
        if kind == b"IDAT" {
            idat.extend_from_slice(&file[data_start..data_end]);
        }
        pos = data_end + 4; // skip the trailing 4-byte CRC
    }
    idat
}

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
    /// Defilter every row of `inflated` in place, mirroring
    /// `ixiptla_png::Png::defilter_sequential`'s own zero-copy "previous
    /// row" borrow (split at the read cursor, all-zero buffer for row 0)
    /// -- `Defilter::defilter_row` no longer tracks `previous_row`
    /// internally, so every caller (this benchmark included) is
    /// responsible for it now. Kept as a benchmark-local helper rather
    /// than exposing `Png`'s own private loop, since this crate
    /// intentionally isolates `Defilter` from the rest of `Png::decode`'s
    /// pipeline (chunk parsing, zlib inflate) for these benchmarks.
    fn defilter_all_rows(ihdr: &Ihdr, inflated: &mut [u8]) {
        let defilter = Defilter::new(ihdr);
        let stride = ihdr.stride();
        let row_len = 1 + stride;
        let zero_row = vec![0u8; stride];
        let mut write_offset = 0usize;
        for read_offset in (0..inflated.len()).step_by(row_len) {
            let (front, back) = inflated.split_at_mut(read_offset);
            let previous_row: &[u8] = if write_offset == 0 {
                &zero_row
            } else {
                &front[write_offset - stride..write_offset]
            };
            let row = &mut back[..row_len];
            defilter
                .defilter_row(divan::black_box(row), previous_row)
                .unwrap();
            write_offset += stride;
        }
    }

    /// Build a non-interlaced, 8-bit fixture at `side x side` for
    /// `color_type`/`png_color_type` (the same colour type in both
    /// crates' vocabularies), filtered via the `png` crate's adaptive
    /// filter selection so this exercises a realistic mix of the 5
    /// filter types across rows, not just one uniformly.
    fn build(side: u32, color_type: ColorType, png_color_type: PngColorType) -> Self {
        Self::build_with_filter(side, color_type, png_color_type, None)
    }

    /// Like [`Self::build`], but pins every row to a single `FilterType`
    /// (via `SquareFixture::encode_with_filter`) instead of `png`'s
    /// default adaptive per-row mix -- used to isolate one filter type's
    /// reconstruction cost at a time, e.g. to check whether `undo_paeth`/
    /// `undo_average` (the two filter types with no SIMD fast path) are
    /// actually the dominant cost within a realistic adaptive-filtered
    /// row mix, rather than guessing from the aggregate `defilter_row`
    /// number alone.
    fn build_with_filter(
        side: u32,
        color_type: ColorType,
        png_color_type: PngColorType,
        filter: Option<PngFilter>,
    ) -> Self {
        let file = SquareFixture::new(side, png_color_type).encode_with_filter(filter);
        let ihdr = Ihdr {
            width: side,
            height: side,
            bit_depth: 8,
            color_type,
            interlace: InterlaceMethod::None,
        };
        // Recover the *actually filtered* scanline bytes -- concatenate
        // this file's IDAT chunks and zlib-inflate them, stopping short of
        // calling `Defilter::defilter_row` at all. This deliberately does
        // NOT go through `Png::decode` and re-wrap the result as
        // filter-type-None rows the way an earlier version of this
        // benchmark did: `png`'s own encoder defaults to `Filter::
        // Adaptive` (picks whichever of Sub/Up/Average/Paeth/None
        // minimizes a per-row heuristic cost, per row), so a real file's
        // rows are a genuine mix of all 5 filter types, each with real
        // per-byte reconstruction arithmetic to do -- `FilterType::None`
        // is a literal no-op (`match { None => {} ... }`), so benchmarking
        // only-None rows was measuring a best case that essentially never
        // occurs in practice and made `defilter_row` look far cheaper
        // than it actually is in `Png::decode`'s real pipeline (confirmed
        // via `perf` on `decode_loop`: `defilter_row` is ~47% of total
        // decode cycles there, nowhere near what the old benchmark here
        // implied).
        let idat = extract_idat(&file);
        let inflated = ixchel::ZlibDecoder::new(1 << 24)
            .decompress(&idat)
            .expect("fixture zlib decodes");
        Self { ihdr, inflated }
    }
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build(side, ColorType::Truecolor, PngColorType::Rgb);
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor_alpha(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build(side, ColorType::TruecolorAlpha, PngColorType::Rgba);
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

/// Isolates one `FilterType`'s reconstruction cost at a time on
/// truecolor-alpha (`bpp == 4`) rows, pinned via `png`'s encoder rather
/// than left to its adaptive per-row choice -- lets `Sub`/`Up` (SIMD
/// fast-pathed in `simd.rs`) be compared directly against `Average`/
/// `Paeth` (still pure scalar loops in `defilter.rs`) to find which of
/// the two is the actual dominant cost within `defilter_row`'s realistic
/// adaptive-filtered aggregate number above.
#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor_alpha_sub_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::TruecolorAlpha,
        PngColorType::Rgba,
        Some(PngFilter::Sub),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor_alpha_up_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::TruecolorAlpha,
        PngColorType::Rgba,
        Some(PngFilter::Up),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor_alpha_average_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::TruecolorAlpha,
        PngColorType::Rgba,
        Some(PngFilter::Avg),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_truecolor_alpha_paeth_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::TruecolorAlpha,
        PngColorType::Rgba,
        Some(PngFilter::Paeth),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_grayscale(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build(side, ColorType::Grayscale, PngColorType::Grayscale);
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

/// The same per-filter isolation as the `defilter_truecolor_alpha_*`
/// benches above, but at `bpp == 1` (grayscale).
///
/// These exist because `defilter_grayscale` at 1024px measured *slower*
/// than `defilter_truecolor_alpha` (~2.1ms vs ~1.15ms) despite having
/// exactly a quarter as many bytes to reconstruct -- roughly 7x worse
/// per byte. That inversion has to come from a missing fast path rather
/// than from the work itself, and `simd.rs` only instantiates its
/// `Average`/`Paeth` SSE2 loops for `BPP` in `{3, 4}`, so `bpp == 1`
/// falls back to `scan_suffix_scalar`. Splitting the aggregate per
/// filter type is what turns that from a plausible story into a
/// measured one -- if the scalar fallback is the cause, `average_only`
/// and `paeth_only` should dominate while `sub_only`/`up_only` (which
/// do have `bpp == 1` paths, via `WrappingPrefixSum`/`WrappingByteAdder`)
/// stay cheap.
#[divan::bench(args = BENCH_SIDES)]
fn defilter_grayscale_sub_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::Grayscale,
        PngColorType::Grayscale,
        Some(PngFilter::Sub),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_grayscale_up_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::Grayscale,
        PngColorType::Grayscale,
        Some(PngFilter::Up),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_grayscale_average_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::Grayscale,
        PngColorType::Grayscale,
        Some(PngFilter::Avg),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
            inflated
        });
}

#[divan::bench(args = BENCH_SIDES)]
fn defilter_grayscale_paeth_only(bencher: Bencher, side: u32) {
    let stream = FilteredStream::build_with_filter(
        side,
        ColorType::Grayscale,
        PngColorType::Grayscale,
        Some(PngFilter::Paeth),
    );
    bencher
        .with_inputs(|| stream.inflated.clone())
        .bench_local_values(|mut inflated| {
            FilteredStream::defilter_all_rows(&stream.ihdr, &mut inflated);
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

/// [`Crc32`] in isolation over a buffer sized like a real large `IDAT`
/// payload -- isolates the PCLMULQDQ fold's own per-byte cost from
/// everything else `Png::decode` does, mirroring how the `defilter_*`
/// benchmarks above isolate `Defilter` from inflate/chunk-parsing. Sizes
/// mirror `BENCH_SIDES` scaled to roughly a truecolor-alpha image's byte
/// count (`side * side * 4`), so this stays comparable to the other
/// per-stage numbers at the same nominal image size.
#[divan::bench(args = BENCH_SIDES)]
fn crc32_of_idat_sized_buffer(bencher: Bencher, side: u32) {
    let len = side as usize * side as usize * 4;
    let mut src = 0x1234_5678u32;
    let mut next_byte = move || {
        src ^= src << 13;
        src ^= src >> 17;
        src ^= src << 5;
        (src & 0xFF) as u8
    };
    let data: Vec<u8> = (0..len).map(|_| next_byte()).collect();
    bencher.bench_local(|| Crc32::of(divan::black_box(&data)));
}
