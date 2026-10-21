//! VPCLMULQDQ+AVX2-accelerated CRC-32 folding (`x86_64` only) -- a wider
//! front end for [`crate::crc32_pclmulqdq::Pclmulqdq`]'s fold-by-64 loop.
//!
//! # Why this is safe to add without deriving new fold constants
//!
//! [`crate::crc32_pclmulqdq::Pclmulqdq::fold`]'s main loop keeps 4
//! independent 128-bit accumulators (`x3, x2, x1, x0`) and, every 64-byte
//! block, folds each one forward by one block via its own `PCLMULQDQ`
//! pair (`reduce128`, using fold-distance constants K1/K2). Those 4
//! `reduce128` calls are *already* fully independent of each other within
//! one iteration -- nothing about the algorithm requires them to run as 4
//! separate 128-bit instructions instead of 2 wider 256-bit ones.
//!
//! `VPCLMULQDQ` (AVX2's carry-less-multiply extension, `_mm256_
//! clmulepi64_epi128`) computes the *same* per-lane carry-less multiply
//! [`_mm_clmulepi64_si128`] does, just twice at once -- once per 128-bit
//! lane of a 256-bit register, with no cross-lane interaction. So packing
//! `(x3, x2)` into one `__m256i` and `(x1, x0)` into another, with K1/K2
//! broadcast into both lanes of a 256-bit key register, and running one
//! [`Self::reduce256`] each is *provably* the same computation as running
//! [`crate::crc32_pclmulqdq::Pclmulqdq`]'s 4 `reduce128` calls --
//! algebraically identical, not a new derivation, so nothing about the
//! well-tested K1..K5/Barrett-reduction math this reuses (via
//! [`crate::crc32_pclmulqdq::Pclmulqdq::converge`]) needs re-verifying
//! from scratch. Cross-checked against the plain PCLMULQDQ path anyway
//! (see this module's tests) since a silent CRC mismatch would corrupt
//! every accelerated decode without any visible error.
//!
//! This halves the pure-multiply instruction count of the fold-by-64
//! loop's hot path (8 `PCLMULQDQ`s/iteration down to 4 `VPCLMULQDQ`s) --
//! the loads, XORs, and everything before/after the loop (tail
//! convergence, Barrett reduction, sub-16-byte remainder) are unchanged,
//! reusing [`crate::crc32_pclmulqdq::Pclmulqdq::converge`] directly.

use crate::crc32_pclmulqdq::{K1, K2, Pclmulqdq};
use std::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_xor_si128, _mm256_clmulepi64_epi128, _mm256_inserti128_si256,
    _mm256_loadu_si256, _mm256_set_epi64x, _mm256_xor_si256,
};

/// Same threshold rationale as [`crate::crc32_pclmulqdq::Pclmulqdq::
/// MIN_ACCELERATED_LEN`] -- below this, PCLMULQDQ's own setup/reduction
/// overhead already isn't worth it, so the wider AVX2 front end (needing
/// at least the same one 64-byte block to have anything to widen) isn't
/// either.
const MIN_ACCELERATED_LEN: usize = 128;

/// VPCLMULQDQ+AVX2-accelerated CRC-32 folding. Stateless, per this
/// crate's "no free functions" convention.
#[derive(Debug)]
pub struct Vpclmulqdq;

impl Vpclmulqdq {
    /// Attempts to fold `bytes` into `state` (register-domain, see
    /// [`crate::crc32_pclmulqdq`]'s module doc for the domain convention)
    /// via VPCLMULQDQ+AVX2.
    ///
    /// Returns `None` -- declining, not erroring -- when the running CPU
    /// lacks VPCLMULQDQ/AVX2, or `bytes` is too short; callers fall back
    /// to [`Pclmulqdq::try_update`] (itself falling back to
    /// [`crate::Crc32`]'s scalar path) in either case.
    #[must_use]
    pub fn try_update(state: u32, bytes: &[u8]) -> Option<u32> {
        if bytes.len() < MIN_ACCELERATED_LEN {
            return None;
        }
        // `_mm256_clmulepi64_epi128` needs `vpclmulqdq` *and* `avx2` (the
        // 256-bit register width/load-store surface); `sse2`/`sse4.1`
        // are needed transitively by `Pclmulqdq::converge`, which this
        // hands off to for the tail.
        if !std::arch::is_x86_feature_detected!("vpclmulqdq")
            || !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("pclmulqdq")
            || !std::arch::is_x86_feature_detected!("sse2")
            || !std::arch::is_x86_feature_detected!("sse4.1")
        {
            return None;
        }
        // SAFETY: the feature checks above guarantee every intrinsic
        // `Self::fold` (and the `Pclmulqdq::converge` it calls) uses is
        // supported by the running CPU.
        Some(unsafe { Self::fold(state, bytes) })
    }

    /// The AVX2 kernel: same 4-accumulator fold-by-64 loop
    /// [`crate::crc32_pclmulqdq::Pclmulqdq::fold`] runs, with the 4
    /// per-iteration `reduce128` calls fused pairwise into 2
    /// [`Self::reduce256`] calls (see this module's doc comment for why
    /// that's exactly the same computation, not a new algorithm).
    ///
    /// # Safety
    ///
    /// Caller must have already confirmed `vpclmulqdq`, `avx2`,
    /// `pclmulqdq`, `sse2`, and `sse4.1` are available (see
    /// [`Self::try_update`]).
    #[target_feature(
        enable = "vpclmulqdq",
        enable = "avx2",
        enable = "pclmulqdq",
        enable = "sse2",
        enable = "sse4.1"
    )]
    unsafe fn fold(state: u32, mut data: &[u8]) -> u32 {
        unsafe {
            // Step 1: same initial 4x16-byte load as the plain PCLMULQDQ
            // path, packed two lanes per __m256i: `hi` holds (x3, x2),
            // `lo` holds (x1, x0) -- lane 0 (low 128 bits) is the first
            // (x3/x1), lane 1 (high 128 bits) the second (x2/x0), an
            // arbitrary but fixed convention kept consistent with
            // `Self::load_pair`'s load order and `Self::split`'s
            // extraction order below.
            let mut hi = Self::load_pair(&mut data); // (x3, x2)
            let mut lo = Self::load_pair(&mut data); // (x1, x0)

            // Fold the incoming running CRC into the stream's first
            // 128-bit block (x3, lane 0 of `hi`) -- identical to
            // `Pclmulqdq::fold`'s own state-folding step, just reaching
            // into the packed register's low lane instead of a bare
            // `__m128i`.
            #[allow(clippy::cast_possible_wrap)]
            let state_vec = _mm_cvtsi32_si128(state as i32);
            let hi_lo_lane = std::arch::x86_64::_mm256_castsi256_si128(hi);
            let hi_lo_lane = _mm_xor_si128(hi_lo_lane, state_vec);
            hi = _mm256_inserti128_si256(hi, hi_lo_lane, 0);

            let k1k2 = _mm256_set_epi64x(K2, K1, K2, K1);
            while data.len() >= 64 {
                let b_hi = Self::load_pair(&mut data);
                let b_lo = Self::load_pair(&mut data);
                hi = Self::reduce256(hi, b_hi, k1k2);
                lo = Self::reduce256(lo, b_lo, k1k2);
            }

            let (x3, x2) = Self::split(hi);
            let (x1, x0) = Self::split(lo);

            // Hand off to the already-verified convergence/Barrett-
            // reduction tail -- identical accumulator shape to what
            // `Pclmulqdq::fold`'s own loop produces, so no new math
            // needed here at all.
            Pclmulqdq::converge(x3, x2, x1, x0, data)
        }
    }

    /// One fold-by-64 reduction step, 2 lanes at once: each 128-bit lane
    /// of `b` `XORed` with the corresponding lane of `a` folded forward by
    /// the distance `keys` encodes -- the exact same formula as
    /// [`crate::crc32_pclmulqdq::Pclmulqdq`]'s private `reduce128`,
    /// applied independently per-lane by `VPCLMULQDQ` itself (see this
    /// module's doc comment for why that's provably equivalent to running
    /// `reduce128` twice).
    // SAFETY (this fn as a whole): every intrinsic used requires only
    // `vpclmulqdq`/`avx2`, both guaranteed by the `#[target_feature]`
    // attribute above (which callers must uphold per this fn's own
    // `unsafe` contract, checked by `Self::try_update`).
    #[target_feature(enable = "vpclmulqdq", enable = "avx2")]
    unsafe fn reduce256(a: __m256i, b: __m256i, keys: __m256i) -> __m256i {
        let t1 = _mm256_clmulepi64_epi128(a, keys, 0x00);
        let t2 = _mm256_clmulepi64_epi128(a, keys, 0x11);
        _mm256_xor_si256(_mm256_xor_si256(b, t1), t2)
    }

    /// Loads the next 32 bytes off the front of `*data` as a single
    /// `__m256i` (two logical 16-byte accumulator lanes at once),
    /// advancing `*data` past them.
    ///
    /// # Safety
    ///
    /// Caller must ensure `data.len() >= 32`.
    #[target_feature(enable = "avx2")]
    unsafe fn load_pair(data: &mut &[u8]) -> __m256i {
        debug_assert!(data.len() >= 32);
        // SAFETY: caller-guaranteed `len >= 32`; `_mm256_loadu_si256` is
        // the unaligned load form, so `data`'s alignment never matters.
        let v = unsafe { _mm256_loadu_si256(data.as_ptr().cast()) };
        *data = &data[32..];
        v
    }

    /// Splits a packed `__m256i` back into its two 128-bit lanes, in the
    /// same `(first, second)` order [`Self::load_pair`] loaded them.
    #[target_feature(enable = "avx2")]
    unsafe fn split(packed: __m256i) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
        let first = std::arch::x86_64::_mm256_castsi256_si128(packed);
        let second = std::arch::x86_64::_mm256_extracti128_si256(packed, 1);
        (first, second)
    }
}

#[cfg(test)]
mod tests {
    use super::Vpclmulqdq;
    use crate::Crc32;

    /// Same deterministic pseudo-random byte generator
    /// [`crate::crc32_pclmulqdq::tests`] uses.
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_byte(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 & 0xFF) as u8
        }
    }

    fn gen_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut generator = Xorshift32(seed);
        (0..len).map(|_| generator.next_byte()).collect()
    }

    /// [`Vpclmulqdq::try_update`] must agree with the scalar slicing-by-8
    /// path bit-for-bit -- the same length/seed/initial-state sweep
    /// [`crate::crc32_pclmulqdq::tests`] runs against the plain PCLMULQDQ
    /// path, straddling every structural boundary this module's wider
    /// loop introduces (32-byte pair loads, 64-byte block loop, and the
    /// same `< 64`-byte tail handoff to `Pclmulqdq::converge`).
    #[test]
    fn matches_scalar_reference_across_lengths_and_seeds() {
        if !std::arch::is_x86_feature_detected!("vpclmulqdq")
            || !std::arch::is_x86_feature_detected!("avx2")
        {
            eprintln!("skipping: VPCLMULQDQ/AVX2 not available on this CPU");
            return;
        }

        for len in [
            0, 1, 15, 16, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 1000, 4096, 4111,
        ] {
            for seed in [0x1234_5678u32, 0x9E37_79B9, 0, 0xFFFF_FFFF] {
                let bytes = gen_bytes(seed, len);
                for initial_state in [0xFFFF_FFFFu32, 0, 0x1234_5678] {
                    let scalar = Crc32::update_state_scalar(initial_state, &bytes);
                    let accelerated = Vpclmulqdq::try_update(initial_state, &bytes);
                    if len >= 128 {
                        assert_eq!(
                            accelerated,
                            Some(scalar),
                            "mismatch at len={len} seed={seed:#x} state={initial_state:#x}"
                        );
                    } else {
                        assert_eq!(
                            accelerated, None,
                            "expected decline below acceleration threshold at len={len}"
                        );
                    }
                }
            }
        }
    }

    /// Cross-checks [`Vpclmulqdq::try_update`] against
    /// [`crate::crc32_pclmulqdq::Pclmulqdq::try_update`] directly (not
    /// just the scalar reference) -- both accelerated paths must produce
    /// identical output for the same input, since [`crate::Crc32::
    /// update_state`] picks between them (and the scalar fallback)
    /// silently based on what the running CPU supports.
    #[test]
    fn matches_plain_pclmulqdq_path() {
        if !std::arch::is_x86_feature_detected!("vpclmulqdq")
            || !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("pclmulqdq")
        {
            eprintln!("skipping: required features not available on this CPU");
            return;
        }
        for len in [128, 129, 256, 1000, 8192, 65536] {
            let bytes = gen_bytes(0xC0FF_EE42, len);
            for initial_state in [0xFFFF_FFFFu32, 0, 0xABCD_1234] {
                let via_avx2 = Vpclmulqdq::try_update(initial_state, &bytes);
                let via_sse = crate::crc32_pclmulqdq::Pclmulqdq::try_update(initial_state, &bytes);
                assert_eq!(via_avx2, via_sse, "mismatch at len={len}");
            }
        }
    }

    /// End-to-end sanity check via a large buffer, cross-checked against
    /// this crate's from-scratch-verified scalar path -- same shape as
    /// `Pclmulqdq`'s own end-to-end test.
    #[test]
    fn full_crc32_matches_scalar_only_computation() {
        if !std::arch::is_x86_feature_detected!("vpclmulqdq")
            || !std::arch::is_x86_feature_detected!("avx2")
        {
            eprintln!("skipping: VPCLMULQDQ/AVX2 not available on this CPU");
            return;
        }
        let mut data = Vec::new();
        for _ in 0..2000 {
            data.extend_from_slice(b"123456789");
        }
        let via_scalar_only = {
            let mut c = 0xFFFF_FFFFu32;
            c = Crc32::update_state_scalar(c, &data);
            c ^ 0xFFFF_FFFF
        };
        let accelerated = Vpclmulqdq::try_update(0xFFFF_FFFF, &data)
            .expect("buffer well above acceleration threshold");
        assert_eq!(accelerated ^ 0xFFFF_FFFF, via_scalar_only);
    }
}
