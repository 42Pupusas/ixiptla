//! Diff-tests `ixiptla_core` against the `png` crate over the real-world
//! `PngSuite` corpus (Willem van Schaik's "the official set of PNG test
//! images", vendored under `tests/pngsuite/` from `image-rs/image-png`'s
//! own copy — see `tests/pngsuite/PngSuite.README` /
//! `PngSuite.LICENSE`, freeware).
//!
//! This is the "real-world hardening" pass beyond `src/lib.rs`'s
//! synthetic, `png`-crate-encoded fixtures: `PngSuite` exercises every
//! color type/bit-depth/interlacing/ancillary-chunk combination the
//! format defines, all through one battle-tested corpus other decoders
//! (libpng, image-png itself, ...) are also validated against.
//!
//! `PngSuite`'s naming convention (`PngSuite.README`) marks intentionally
//! *invalid* files with a leading `x` (corrupted signature, bad bit
//! depth, missing IDAT, ...) — [`PngSuiteCorpus::run`] expects those to
//! be rejected by both decoders, not decoded and compared.

use ixiptla_conformance::reference_decoder::ReferenceComparison;
use std::fs;
use std::path::{Path, PathBuf};

/// The vendored `PngSuite` corpus directory and the pass/fail bookkeeping
/// for walking it.
struct PngSuiteCorpus {
    dir: PathBuf,
}

impl PngSuiteCorpus {
    fn new() -> Self {
        Self {
            dir: Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pngsuite"),
        }
    }

    /// `PngSuite`'s file-naming convention (`PngSuite.README`): a filename
    /// starting with `x` denotes a deliberately-invalid file (bad
    /// signature, bad bit depth, missing `IDAT`, ...) that a correct
    /// decoder must *reject*, not decode.
    fn expects_rejection(path: &Path) -> bool {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('x'))
    }

    /// Walk every `*.png` fixture, diff-testing valid ones against the
    /// `png` crate and asserting deliberately-invalid ones are rejected
    /// by `ixiptla_core` too. Panics with every failing filename at once
    /// (rather than stopping at the first) so a single run reports the
    /// full extent of any regression.
    fn run(&self) {
        let mut entries: Vec<PathBuf> = fs::read_dir(&self.dir)
            .unwrap_or_else(|e| panic!("reading {}: {e}", self.dir.display()))
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
            .collect();
        entries.sort();
        assert!(
            !entries.is_empty(),
            "no .png fixtures found under {} -- corpus missing?",
            self.dir.display()
        );

        let mut failures = Vec::new();
        for path in &entries {
            let file = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            let name = path.file_name().unwrap().to_string_lossy().into_owned();

            if Self::expects_rejection(path) {
                if ixiptla_core::Png::decode(&file, 1 << 24).is_ok() {
                    failures.push(format!(
                        "{name}: expected ixiptla_core to reject this (PngSuite 'x' = deliberately invalid), but it decoded successfully"
                    ));
                }
                continue;
            }

            if let Err(msg) = ReferenceComparison::compare(&file) {
                failures.push(format!("{name}: {msg}"));
            }
        }

        assert!(
            failures.is_empty(),
            "{} of {} PngSuite fixtures failed:\n{}",
            failures.len(),
            entries.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn matches_reference_across_pngsuite_corpus() {
    PngSuiteCorpus::new().run();
}
