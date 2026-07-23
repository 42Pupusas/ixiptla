//! `ixiptla-bench` — `divan` benchmarks for `ixiptla-core`: internal
//! hotpath coverage (`benches/hotpaths.rs`) plus end-to-end wall-clock
//! comparisons against the `png` and `image` crates (`benches/
//! end_to_end.rs`) decoding the exact same file bytes.
//!
//! Kept as its own crate (mirroring `ixiptla-conformance`'s reasoning)
//! rather than `[dev-dependencies]`/`[[bench]]` entries inside
//! `ixiptla-core` itself, so pulling in `png`, `image`, and `divan` for
//! benchmarking purposes never taints `ixiptla-core`'s zero-dependency
//! build.
//!
//! [`fixtures`] is the one deterministic-synthetic-PNG-file builder shared
//! by both bench binaries, so "what exact bytes does the `truecolor_512`
//! fixture decode to" only has one definition to keep consistent between
//! the hotpath and end-to-end suites.

pub mod fixtures;
