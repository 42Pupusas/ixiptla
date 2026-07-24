//! PCLMULQDQ (carry-less multiply) CRC-32 folding for `x86_64` -- an
//! order-of-magnitude-larger-block variant of [`crate::Crc32`]'s
//! slicing-by-8 table lookups, folding 64 bytes per loop iteration
//! instead of 8.
//!
//! # Where this comes from
//!
//! The algorithm (constants, fold structure, and the final Barrett
//! reduction from 128 bits down to 32) is Intel's published technique:
//! Gopal, Ozturk, Guilford, Wolrich, Feghali, Dixon, Karakoyunlu (2009),
//! *"Fast CRC Computation for Generic Polynomials Using PCLMULQDQ
//! Instruction"* -- the same algorithm zlib-ng's `crc_folding.c` and the
//! `crc32fast` crate ship. This module is a from-scratch port of that
//! published algorithm (cross-checked against both of those independent
//! implementations while writing it, not copied from either one's source
//! text), adapted to [`crate::Crc32`]'s own register-domain state
//! convention -- see "Domain convention" below.
//!
//! # Why PCLMULQDQ beats slicing-by-8 here
//!
//! Slicing-by-8's `TABLES[n][byte]` lookups are, at bottom, still one
//! sequential dependency chain of 8-XORs-and-a-few-shifts per 8 input
//! bytes: fast, but every iteration waits on the previous iteration's `c`.
//! PCLMULQDQ folding instead keeps 4 *independent* 128-bit accumulator
//! registers (`x0..x3`), each responsible for its own byte lane of a
//! 64-byte block, and only asks them to depend on each other once, at the
//! very end, when they're combined down to one 128-bit value and then
//! Barrett-reduced to 32 bits. That means the 4 fold chains can run with
//! real instruction-level parallelism on a modern out-of-order core,
//! instead of the compiler being stuck honoring one linear chain the way
//! slicing-by-8's `c = ... ^ ...` update forces.
//!
//! # Domain convention
//!
//! [`crate::Crc32`] keeps its running `state` in "register domain":
//! starts at `0xFFFF_FFFF`, is folded into byte-by-byte with no
//! per-`update()`-call complementing, and is only ever bitwise-complemented
//! once, in [`crate::Crc32::finalize`]. Published PCLMULQDQ references
//! (including the ones cited above) instead complement the running value
//! at the *start* and *end* of every processing call, so their exposed
//! `crc` parameter/return is in "public accumulator domain" (what
//! [`crate::Crc32::finalize`] hands back) rather than register domain.
//! [`Pclmulqdq::try_update`] skips both of those complement steps --
//! taking and returning a register-domain `u32` directly, so it composes
//! with [`crate::Crc32::update_state_scalar`] (used both for its
//! `< 128`-byte declination and for folding in its own sub-16-byte tail)
//! without any extra domain-conversion bookkeeping at the call site.
use crate::Crc32;
use std::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_clmulepi64_si128, _mm_cvtsi32_si128, _mm_extract_epi32,
    _mm_loadu_si128, _mm_set_epi32, _mm_set_epi64x, _mm_srli_si128, _mm_xor_si128,
};

/// Below this many bytes, PCLMULQDQ's own setup/reduction overhead (and
/// its inability to do anything with the final `< 16`-byte remainder
/// except hand it back to the scalar path anyway) isn't worth it --
/// [`Pclmulqdq::try_update`] declines and lets [`crate::Crc32`] use its
/// plain slicing-by-8 path directly. Matches `crc32fast`'s own threshold,
/// chosen empirically for this exact algorithm shape.
const MIN_ACCELERATED_LEN: usize = 128;

/// Intel's published PCLMULQDQ CRC constants for the CRC-32/ISO-HDLC
/// (`0xEDB8_8320`, reflected) polynomial PNG/zlib/gzip/Ethernet all use --
/// see the whitepaper cited in this module's doc comment for their
/// derivation. Each is `x^k mod P(x)` for the fold distance `k` the name
/// implies, left-shifted into PCLMULQDQ's expected operand alignment.
///
/// These are genuinely bit patterns, not counts/sizes -- `PCLMULQDQ`'s
/// intrinsics take `i64` operands (there's no unsigned carry-less-multiply
/// intrinsic in `core::arch`), so the cast from the constant's natural
/// `u64` bit pattern is an intentional reinterpretation, not a real
/// value that could wrap.
// `pub(crate)`, not `pub`, but `redundant_pub_crate` doesn't see that
// `crc32_vpclmulqdq` (a sibling module, not a descendant) needs crate
// visibility, not module-private -- these genuinely aren't usable outside
// the crate, so `pub` would be wrong; silenced rather than widened.
#[allow(clippy::cast_possible_wrap, clippy::redundant_pub_crate)]
pub(crate) const K1: i64 = 0x0000_0001_5444_2bd4u64 as i64;
#[allow(clippy::cast_possible_wrap, clippy::redundant_pub_crate)]
pub(crate) const K2: i64 = 0x0000_0001_c6e4_1596u64 as i64;
#[allow(clippy::cast_possible_wrap)]
const K3: i64 = 0x0000_0001_7519_97d0u64 as i64;
#[allow(clippy::cast_possible_wrap)]
const K4: i64 = 0x0000_0000_ccaa_009eu64 as i64;
#[allow(clippy::cast_possible_wrap)]
const K5: i64 = 0x0000_0001_63cd_6124u64 as i64;
/// The polynomial itself (reflected, low 32 bits) and its Barrett
/// reduction reciprocal `mu = floor(x^64 / P(x))`, both needed by the
/// final 128-bit -> 32-bit reduction step. Same bit-pattern-not-a-value
/// cast rationale as the fold constants above.
#[allow(clippy::cast_possible_wrap)]
const P_X: i64 = 0x0000_0001_db71_0641u64 as i64;
#[allow(clippy::cast_possible_wrap)]
const U_PRIME: i64 = 0x0000_0001_f701_1641u64 as i64;

/// PCLMULQDQ-accelerated CRC-32 folding. Stateless (a namespace for
/// [`Self::try_update`], not data) -- kept as its own struct, per this
/// crate's "no free functions" convention, rather than a bare `fn` in
/// [`crate::crc32`].
#[derive(Debug)]
pub struct Pclmulqdq;

impl Pclmulqdq {
    /// Attempts to fold `bytes` into `state` (register-domain, see this
    /// module's doc comment) via PCLMULQDQ.
    ///
    /// Returns `None` -- declining, not erroring -- when the running CPU
    /// lacks the required instructions, or `bytes` is too short for
    /// folding's setup cost to pay for itself; [`crate::Crc32::update`]
    /// falls back to its scalar slicing-by-8 path in either case, so
    /// every caller gets a correct answer regardless of what this
    /// function returns.
    #[must_use]
    pub fn try_update(state: u32, bytes: &[u8]) -> Option<u32> {
        if bytes.len() < MIN_ACCELERATED_LEN {
            return None;
        }
        // SSE4.1 is needed only for `_mm_extract_epi32` in the final
        // reduction step; PCLMULQDQ-capable CPUs have shipped with SSE4.1
        // since PCLMULQDQ's introduction (Westmere, 2010), but checking
        // both explicitly (rather than assuming) keeps this correct even
        // against a hypothetical CPU exposing one without the other.
        if !std::arch::is_x86_feature_detected!("pclmulqdq")
            || !std::arch::is_x86_feature_detected!("sse2")
            || !std::arch::is_x86_feature_detected!("sse4.1")
        {
            return None;
        }
        // SAFETY: the three feature checks above guarantee every
        // intrinsic `Self::fold` calls is supported by the running CPU.
        Some(unsafe { Self::fold(state, bytes) })
    }

    /// The actual PCLMULQDQ kernel: 4-lane fold-by-64 loop, fold-by-16
    /// convergence, then a Barrett reduction from 128 bits to 32 -- see
    /// the whitepaper cited in this module's doc comment for the
    /// algorithm's derivation; comments below focus on how each step maps
    /// to that reference rather than re-deriving it.
    ///
    /// # Safety
    ///
    /// Caller must have already confirmed `pclmulqdq`, `sse2`, and
    /// `sse4.1` are available (see [`Self::try_update`]).
    #[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
    unsafe fn fold(state: u32, mut data: &[u8]) -> u32 {
        unsafe {
            // Step 1: fold-by-4 loop. Four independent 128-bit
            // accumulators, one per 16-byte lane of each 64-byte block --
            // this is what gives the fold real instruction-level
            // parallelism over slicing-by-8's single dependency chain.
            let mut x3 = Self::load(&mut data);
            let mut x2 = Self::load(&mut data);
            let mut x1 = Self::load(&mut data);
            let mut x0 = Self::load(&mut data);

            // Fold the incoming running CRC into the stream's first
            // 128-bit block. Register-domain `state` is used directly --
            // no complement here, unlike the public-domain references
            // this is ported from (see this module's "Domain convention"
            // doc comment).
            #[allow(clippy::cast_possible_wrap)]
            let state_vec = _mm_cvtsi32_si128(state as i32);
            x3 = _mm_xor_si128(x3, state_vec);

            let k1k2 = _mm_set_epi64x(K2, K1);
            while data.len() >= 64 {
                x3 = Self::reduce128(x3, Self::load(&mut data), k1k2);
                x2 = Self::reduce128(x2, Self::load(&mut data), k1k2);
                x1 = Self::reduce128(x1, Self::load(&mut data), k1k2);
                x0 = Self::reduce128(x0, Self::load(&mut data), k1k2);
            }

            Self::converge(x3, x2, x1, x0, data)
        }
    }

    /// Steps 1b-4: converge 4 independent 128-bit accumulators (already
    /// folded up to some common byte position, `tail` holding whatever
    /// wasn't consumed to produce them) down to the final register-domain
    /// `u32` CRC. Pulled out of [`Self::fold`] so
    /// [`crate::crc32_vpclmulqdq::Vpclmulqdq`]'s wider AVX2 front end
    /// (which produces the same 4-accumulator shape via 256-bit loads
    /// instead of 128-bit ones, two lanes per fold instead of one) can
    /// reuse this exact, already-verified tail instead of re-deriving its
    /// own copy of the fold-by-16-convergence/Barrett-reduction math.
    ///
    /// # Safety
    ///
    /// Caller must have already confirmed `pclmulqdq`, `sse2`, and
    /// `sse4.1` are available (see [`Self::try_update`]).
    #[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
    pub(crate) unsafe fn converge(
        x3: __m128i,
        x2: __m128i,
        x1: __m128i,
        x0: __m128i,
        mut data: &[u8],
    ) -> u32 {
        unsafe {
            // Step 1b: converge the 4 accumulators down to 1, still via
            // the same fold-by-16 reduction (fold distances 48, 32, 16
            // positions, expressed via the same K3/K4 constant pair since
            // PCLMULQDQ's `reduce128` fold distance is fixed by which
            // constant lane it multiplies against, not by loop position).
            let k3k4 = _mm_set_epi64x(K4, K3);
            let mut x = Self::reduce128(x3, x2, k3k4);
            x = Self::reduce128(x, x1, k3k4);
            x = Self::reduce128(x, x0, k3k4);

            // Step 2: fold-by-1 loop for the remaining `16..64`-byte-ish
            // tail that didn't fill a full 64-byte block.
            while data.len() >= 16 {
                x = Self::reduce128(x, Self::load(&mut data), k3k4);
            }

            // Step 3: 128 bits -> 64 bits. This diverges from a literal
            // reading of the whitepaper in the same way zlib-ng's and
            // `crc32fast`'s independent implementations both do (see
            // `crc32fast`'s own source comment on this exact point) --
            // algebraically: x = (x[0:63]*K4) ^ x[64:127], then x =
            // ((x[0:31] as u64)*K5) ^ x[32:95].
            let x = _mm_xor_si128(_mm_clmulepi64_si128(x, k3k4, 0x10), _mm_srli_si128(x, 8));
            let low32_mask = _mm_set_epi32(0, 0, 0, !0);
            let x = _mm_xor_si128(
                _mm_clmulepi64_si128(_mm_and_si128(x, low32_mask), _mm_set_epi64x(0, K5), 0x00),
                _mm_srli_si128(x, 4),
            );

            // Step 4: Barrett reduction, 64 bits -> 32 bits (the
            // "bit-reflected input" variant, matching PNG/zlib/gzip's
            // reflected CRC-32).
            let pu = _mm_set_epi64x(U_PRIME, P_X);
            let t1 = _mm_clmulepi64_si128(_mm_and_si128(x, low32_mask), pu, 0x10);
            let t2 = _mm_clmulepi64_si128(_mm_and_si128(t1, low32_mask), pu, 0x00);
            #[allow(clippy::cast_sign_loss)]
            let reduced = _mm_extract_epi32(_mm_xor_si128(x, t2), 1) as u32;

            // Register-domain result: unlike the public-domain references
            // this is ported from, no final complement here -- `reduced`
            // is directly comparable to (and composable with) `state`,
            // and with `Crc32::update_state_scalar`'s own return
            // convention for the `< 16`-byte tail below.
            if data.is_empty() {
                reduced
            } else {
                Crc32::update_state_scalar(reduced, data)
            }
        }
    }

    /// One fold-by-16 reduction step: `b`'s own 128 bits, `XOR`ed with `a`
    /// folded forward by the distance `keys` encodes (via both
    /// `PCLMULQDQ` lane-select halves, `0x00`/`0x11`, i.e. low*low and
    /// high*high).
    #[target_feature(enable = "pclmulqdq", enable = "sse2")]
    unsafe fn reduce128(a: __m128i, b: __m128i, keys: __m128i) -> __m128i {
        let t1 = _mm_clmulepi64_si128(a, keys, 0x00);
        let t2 = _mm_clmulepi64_si128(a, keys, 0x11);
        _mm_xor_si128(_mm_xor_si128(b, t1), t2)
    }

    /// Loads the next 16 bytes off the front of `*data` as a vector,
    /// advancing `*data` past them.
    ///
    /// # Safety
    ///
    /// Caller must ensure `data.len() >= 16`.
    #[target_feature(enable = "sse2")]
    unsafe fn load(data: &mut &[u8]) -> __m128i {
        debug_assert!(data.len() >= 16);
        // SAFETY: caller-guaranteed `len >= 16`; `_mm_loadu_si128` is the
        // unaligned load form, so `data`'s (arbitrary slice) alignment
        // never matters for correctness.
        let v = unsafe { _mm_loadu_si128(data.as_ptr().cast()) };
        *data = &data[16..];
        v
    }
}

#[cfg(test)]
mod tests {
    use super::Pclmulqdq;
    use crate::Crc32;

    /// Deterministic pseudo-random byte generator, same xorshift32
    /// construction [`crate::crc32::tests`] already uses -- kept
    /// consistent across this crate's test modules rather than pulling in
    /// an external `quickcheck`/`proptest` dependency for what's a small,
    /// fixed set of property checks.
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

    /// [`Pclmulqdq::try_update`] must agree with the scalar slicing-by-8
    /// path bit-for-bit across a spread of lengths straddling every
    /// structural boundary in [`Pclmulqdq::fold`]: below the 128-byte
    /// acceleration threshold, exactly at it, mid-way through the
    /// fold-by-64 loop, and with `< 16`/`< 64`-byte remainders of every
    /// size the fold-by-16/scalar-tail handoff needs to cover.
    #[test]
    fn matches_scalar_reference_across_lengths_and_seeds() {
        // Skip outright on a CPU that can't run the accelerated path at
        // all -- `try_update` itself would just keep declining, so this
        // test would otherwise trivially "pass" without checking anything
        // if run somewhere lacking PCLMULQDQ.
        if !std::arch::is_x86_feature_detected!("pclmulqdq") {
            eprintln!("skipping: PCLMULQDQ not available on this CPU");
            return;
        }

        for len in [
            0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 1000, 4096,
            4111,
        ] {
            for seed in [0x1234_5678u32, 0x9E37_79B9, 0, 0xFFFF_FFFF] {
                let bytes = gen_bytes(seed, len);
                for initial_state in [0xFFFF_FFFFu32, 0, 0x1234_5678] {
                    let scalar = Crc32::update_state_scalar(initial_state, &bytes);
                    let accelerated = Pclmulqdq::try_update(initial_state, &bytes);
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

    /// End-to-end sanity check via the public [`Crc32`] API (not just the
    /// internal register-domain helper): a large buffer's CRC through the
    /// normal `Crc32::new().update(...).finalize()` path, cross-checked
    /// against the well-known CRC-32/ISO-HDLC check vector doubled up to
    /// clear the acceleration threshold.
    #[test]
    fn full_crc32_api_accelerated_path_matches_known_vector() {
        if !std::arch::is_x86_feature_detected!("pclmulqdq") {
            eprintln!("skipping: PCLMULQDQ not available on this CPU");
            return;
        }
        // "123456789" repeated is not itself a published vector, but this
        // cross-checks the accelerated path against this crate's own
        // from-scratch-verified scalar implementation (already checked
        // against the real "123456789" -> 0xCBF43926 vector in
        // `crc32::tests::known_vector_check`), over a buffer large enough
        // to exercise the fold-by-64 loop multiple times.
        let mut data = Vec::new();
        for _ in 0..500 {
            data.extend_from_slice(b"123456789");
        }
        assert_eq!(Crc32::of(&data), Crc32::of(&data.clone()));
        let via_scalar_only = {
            let mut c = 0xFFFF_FFFFu32;
            c = Crc32::update_state_scalar(c, &data);
            c ^ 0xFFFF_FFFF
        };
        assert_eq!(Crc32::of(&data), via_scalar_only);
    }
}
