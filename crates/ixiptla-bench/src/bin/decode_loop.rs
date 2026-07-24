//! A plain, non-`divan` binary that decodes a synthetic PNG in a tight
//! loop, meant to be run under `perf record`/`perf stat` (via the
//! `perf` tool's `run` target, or `perf record -- cargo run --release
//! -p ixiptla-bench --bin decode_loop` directly) rather than through
//! `cargo bench`.
//!
//! Benchmarks harnessed by `divan` are hard to get clean `perf` samples
//! from: each iteration is short, wrapped in per-sample timing
//! bookkeeping, and interleaved with divan's own statistics-gathering
//! code, so a flat profile ends up dominated by the harness rather than
//! the code under test. This binary instead does one big, uninterrupted
//! loop of real decode work with no harness in the hot path at all --
//! `perf record`'s sampling profiler then attributes (essentially) every
//! sample to `ixiptla_core` itself, giving a much cleaner call graph /
//! flat profile to read.
//!
//! Usage: `decode_loop [side] [iterations]` (both optional, defaulting
//! to a large image and a moderate iteration count -- large enough that
//! per-call overhead is negligible next to the per-pixel inner loops
//! this is meant to profile).

use ixiptla_bench::fixtures::SquareFixture;
use ixiptla_core::{Png, RgbaExpander};
use png::ColorType as PngColorType;
use std::hint::black_box;
use std::time::Instant;

/// Command-line configuration for one profiling run: how big a fixture
/// to decode and how many times to decode it.
struct RunConfig {
    side: u32,
    iterations: u32,
}

impl RunConfig {
    /// Parses `side` and `iterations` positionally from `argv[1..]`,
    /// falling back to defaults chosen to give `perf` a run lasting a
    /// few seconds on a typical dev machine -- long enough for a
    /// meaningful number of samples, short enough to stay interactive.
    fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        let side = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let iterations = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
        Self { side, iterations }
    }
}

/// One profiling workload: repeatedly runs `ixiptla_core`'s full
/// decode-to-RGBA8 pipeline (`Png::decode` + `RgbaExpander::expand`)
/// over the same in-memory file bytes, so `perf` sees nothing but real
/// pipeline work for the whole run.
struct DecodeLoop {
    file: Vec<u8>,
    iterations: u32,
}

impl DecodeLoop {
    /// Builds a truecolor-alpha fixture once (the color type whose
    /// `RgbaExpander::expand` path is the current optimization target,
    /// per the `ixiptla-bench` allocation/timing findings) and pairs it
    /// with the iteration count to run it for.
    fn new(config: &RunConfig) -> Self {
        let file = SquareFixture::new(config.side, PngColorType::Rgba).encode();
        Self {
            file,
            iterations: config.iterations,
        }
    }

    /// Runs the full decode pipeline `self.iterations` times, wrapping
    /// each output in `black_box` so the optimizer can't prove the
    /// result is unused and elide the work `perf` is meant to observe.
    ///
    /// Returns total wall-clock time, printed by `main` as a sanity
    /// check alongside whatever `perf` reports -- this binary's own
    /// timing isn't a substitute for `ixiptla-bench`'s `divan`
    /// benchmarks, just a way to confirm the loop actually ran the
    /// expected amount of work.
    fn run(&self) -> std::time::Duration {
        let start = Instant::now();
        for _ in 0..self.iterations {
            let decoded = Png::decode(black_box(&self.file), 1 << 24).expect("fixture decodes");
            let expander = RgbaExpander::new(&decoded.ihdr, None, None);
            // `expand_owned`: matches the call shape a real caller
            // (nothing else needs `decoded.pixels` after this) would use
            // -- see `RgbaExpander::expand_owned`'s doc comment.
            let rgba = expander
                .expand_owned(decoded.pixels)
                .expect("expand succeeds");
            black_box(rgba);
        }
        start.elapsed()
    }
}

fn main() {
    let config = RunConfig::from_args();
    let side = config.side;
    let iterations = config.iterations;
    let workload = DecodeLoop::new(&config);
    let elapsed = workload.run();
    println!(
        "decode_loop: {iterations} iterations of {side}x{side} truecolor-alpha in {elapsed:?} ({:?}/iter)",
        elapsed / iterations.max(1),
    );
}
