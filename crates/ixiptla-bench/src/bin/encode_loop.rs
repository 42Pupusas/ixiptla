//! A plain, non-`divan` binary that encodes a synthetic raw image in a
//! tight loop, meant to be run under `perf record`/`perf stat` rather
//! than through `cargo bench`.
//!
//! This is the encode-side twin of `decode_loop`, and exists for the same
//! reason: `divan` wraps every iteration in timing bookkeeping and
//! statistics gathering, so a flat profile taken through `cargo bench`
//! ends up dominated by the harness instead of the code under test. Worse
//! for our purposes, `perf` invoked around `cargo` profiles *cargo* —
//! build-system symbols, `rustup` shims and `malloc` — and never reaches
//! the benchmark binary at all. One uninterrupted loop of real encode
//! work sidesteps both problems.
//!
//! Usage: `encode_loop [side] [iterations] [strategy]`, all optional.
//! `strategy` is `adaptive` (the default, and the one worth profiling —
//! it evaluates every candidate filter per row) or `up` (a single fixed
//! filter, useful as a baseline showing what the *rest* of the pipeline
//! costs once filter selection is removed).

use ixiptla_bench::raw_image::{Content, RawImage};
use ixiptla_png::{ColorType, Encoder, FilterStrategy, FilterType};
use std::hint::black_box;
use std::time::Instant;

/// Which filtering strategy the profiled encode should use.
///
/// A named enum rather than passing `FilterStrategy` around directly, so
/// this binary owns its own CLI vocabulary and printed label instead of
/// leaking the core crate's spelling onto the command line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// Score all five filters per row and keep the best — the expensive
    /// path, and the reason this binary exists.
    Adaptive,
    /// A single fixed `Up` filter: the baseline that shows how much of
    /// the run time is filter *selection* versus everything else.
    FixedUp,
}

impl Strategy {
    /// Parses the CLI spelling, defaulting to `Adaptive` for anything
    /// unrecognized (including absent).
    fn from_arg(arg: Option<&str>) -> Self {
        match arg {
            Some("up" | "fixed-up") => Self::FixedUp,
            _ => Self::Adaptive,
        }
    }

    /// The core crate's strategy this corresponds to.
    const fn filter_strategy(self) -> FilterStrategy {
        match self {
            Self::Adaptive => FilterStrategy::Adaptive,
            Self::FixedUp => FilterStrategy::Fixed(FilterType::Up),
        }
    }

    /// Human-readable label for the summary line.
    const fn label(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::FixedUp => "fixed-up",
        }
    }
}

/// Command-line configuration for one profiling run.
struct RunConfig {
    side: u32,
    iterations: u32,
    strategy: Strategy,
}

impl RunConfig {
    /// Parses `side`, `iterations` and `strategy` positionally from
    /// `argv[1..]`, defaulting to a run lasting a few seconds — long
    /// enough for a meaningful sample count, short enough to stay
    /// interactive.
    fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        let side = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let iterations = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
        let strategy = Strategy::from_arg(args.next().as_deref());
        Self {
            side,
            iterations,
            strategy,
        }
    }
}

/// One profiling workload: repeatedly encodes the same raw truecolor-alpha
/// image, so `perf` sees nothing but real encode work for the whole run.
///
/// The content is [`Content::Photographic`] deliberately. Noise would make
/// the profile a study of DEFLATE's incompressible worst case, where no
/// match is ever emitted and filter choice is irrelevant — the opposite of
/// the workload we want to speed up.
struct EncodeLoop {
    image: RawImage,
    strategy: FilterStrategy,
    iterations: u32,
}

impl EncodeLoop {
    /// Builds the requested fixture once and pairs it with the iteration
    /// count and strategy to run it under.
    fn new(config: &RunConfig) -> Self {
        let image = RawImage::square(
            config.side,
            ColorType::TruecolorAlpha,
            Content::Photographic,
        );
        Self {
            image,
            strategy: config.strategy.filter_strategy(),
            iterations: config.iterations,
        }
    }

    /// Runs the encoder `self.iterations` times, wrapping input and output
    /// in `black_box` so the optimizer can't prove the result unused and
    /// elide the work `perf` is meant to observe.
    fn run(&self) -> std::time::Duration {
        let encoder = Encoder::new(self.image.ihdr).with_filter_strategy(self.strategy);
        let start = Instant::now();
        for _ in 0..self.iterations {
            let png = encoder
                .encode(black_box(&self.image.pixels))
                .expect("fixture encodes");
            black_box(png);
        }
        start.elapsed()
    }
}

fn main() {
    let config = RunConfig::from_args();
    let side = config.side;
    let iterations = config.iterations;
    let label = config.strategy.label();
    let workload = EncodeLoop::new(&config);
    let elapsed = workload.run();
    println!(
        "encode_loop: {iterations} iterations of {side}x{side} rgba {label} in {elapsed:?} ({:?}/iter)",
        elapsed / iterations.max(1),
    );
}
