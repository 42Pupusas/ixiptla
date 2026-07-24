//! SIMD-accelerated wide-lane byte addition for defiltering ops that have
//! no true per-byte data dependency across the row.
//!
//! `Up` (`Recon(x) = Filt(x) + Recon(above)`) is the motivating case: every
//! output byte only depends on the byte directly above it in the previous
//! row, never on another byte in the *same* row, so all bytes in a row can
//! be reconstructed in parallel -- unlike `Sub`/`Average`/`Paeth`, which
//! have a genuine sequential dependency at `bpp` stride within the row
//! itself. That makes `Up` the natural first (and easiest, and fully
//! correct with zero cross-lane bookkeeping) SIMD target.
//!
//! Only `x86_64` gets a real SIMD implementation here, per this crate's
//! current target. SSE2 is part of the `x86_64` baseline ABI (every `x86_64`
//! CPU has it -- it's not an optional extension the way AVX2/AVX-512 are),
//! so no runtime feature detection is needed to use it unconditionally.
//! Non-x86_64 targets fall back to the plain scalar loop so the crate still
//! builds and behaves correctly everywhere; the two implementations live
//! behind a single `#[cfg(target_arch = "x86_64")]` split *inside this one
//! method*, rather than sprinkling architecture `#[cfg]`s through
//! `Defilter`'s business logic -- callers just see one `WrappingByteAdder::
//! add_wrapping` regardless of target.

/// Wrapping (mod-256) elementwise byte addition, `dst[i] += src[i]`.
#[derive(Debug)]
pub struct WrappingByteAdder;

impl WrappingByteAdder {
    /// Adds `src[i]` into `dst[i]` in place for every `i`, wrapping on
    /// overflow -- exactly PNG's `u8::wrapping_add` semantics.
    ///
    /// # Panics
    ///
    /// If `dst.len() != src.len()`.
    pub fn add_wrapping(dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len(), "add_wrapping: length mismatch");
        #[cfg(target_arch = "x86_64")]
        {
            Self::add_wrapping_sse2(dst, src);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::add_wrapping_scalar(dst, src);
        }
    }

    /// SSE2 byte-lane add (`paddb`), 16 bytes per instruction.
    //
    // `_mm_add_epi8` operates on 8-bit lanes with silent wraparound on
    // overflow (there's no signed/unsigned distinction at the instruction
    // level for plain addition, and no saturation -- that's `paddsb`/
    // `paddusb`, different instructions), which is precisely `wrapping_add`
    // -- no masking or correction pass needed after the SIMD op.
    #[cfg(target_arch = "x86_64")]
    fn add_wrapping_sse2(dst: &mut [u8], src: &[u8]) {
        use std::arch::x86_64::{_mm_add_epi8, _mm_loadu_si128, _mm_storeu_si128};

        const LANES: usize = 16; // one __m128i = 16 bytes.
        let len = dst.len();
        let chunks = len / LANES;

        // SAFETY: SSE2 is guaranteed present on every x86_64 target (it's
        // part of the baseline ABI, unlike AVX2/AVX-512), so this isn't
        // gated behind runtime `is_x86_feature_detected!` the way an
        // optional extension would need. `_mm_loadu_si128`/
        // `_mm_storeu_si128` are the *unaligned* load/store forms
        // specifically so `dst`/`src`'s alignment (arbitrary -- they're
        // slices into a `Vec<u8>` sliced at row boundaries, not aligned
        // allocations) never matters for correctness, only for speed.
        // Each `offset` stays within `0..chunks*LANES <= len`, so every
        // pointer + 16-byte read/write is in bounds for both slices (equal
        // length, asserted above).
        unsafe {
            for i in 0..chunks {
                let offset = i * LANES;
                let d = _mm_loadu_si128(dst.as_ptr().add(offset).cast());
                let s = _mm_loadu_si128(src.as_ptr().add(offset).cast());
                let sum = _mm_add_epi8(d, s);
                _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast(), sum);
            }
        }

        Self::add_wrapping_scalar(&mut dst[chunks * LANES..], &src[chunks * LANES..]);
    }

    /// Plain byte-at-a-time fallback: used for the tail remainder past the
    /// last full SIMD chunk on `x86_64`, and as the entire implementation on
    /// any other architecture.
    fn add_wrapping_scalar(dst: &mut [u8], src: &[u8]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = d.wrapping_add(s);
        }
    }
}

/// SIMD-accelerated wrapping prefix sum, the operation `Sub`/`Average`/
/// `Paeth` all reduce to once you deinterleave a row into its `bpp`
/// mutually-independent lanes.
///
/// `Sub`'s reconstruction is `Recon(x) = Filt(x) + Recon(x - bpp)`: within
/// *one* of the `bpp` interleaved lanes (all bytes at a fixed `i % bpp`),
/// that's exactly `lane[i] = lane[i] + lane[i-1] + lane[i-2] + ... +
/// lane[0]` mod 256 -- an inclusive running ("prefix") sum. The lanes
/// themselves have no dependency on each other, only elements *within* a
/// lane depend on earlier elements of that same lane; that's the
/// structural reason `Sub` can't be vectorized directly the way `Up` was
/// (adjacent bytes in the row belong to different lanes and don't share a
/// dependency, but a byte and the one `bpp` positions before it, in the
/// *same* lane, is a true chain), but *within* one deinterleaved lane, a
/// prefix sum is a textbook parallel ("SWAR"/Hillis-Steele scan)
/// primitive.
#[derive(Debug)]
pub struct WrappingPrefixSum;

impl WrappingPrefixSum {
    /// Computes an in-place inclusive wrapping (mod-256) prefix sum:
    /// afterwards `data[i] == data_before[0] + data_before[1] + ... +
    /// data_before[i]` (mod 256), for every `i`.
    pub fn scan(data: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            Self::scan_sse2(data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::scan_scalar(data);
        }
    }

    /// SSE2 Hillis-Steele scan: within each 16-byte chunk, four
    /// `shift-then-add` rounds (shifting by 1, 2, 4, then 8 bytes) compute
    /// the chunk's own inclusive prefix sum in `log2(16) = 4` dependent
    /// SIMD instructions instead of 16 dependent scalar `add`s. Chunks
    /// then need their running total carried in from all chunks before
    /// them -- that carry is still an inherently serial per-chunk step,
    /// but there are only `len / 16` chunks, not `len` individual bytes,
    /// so the serial part of the work shrinks by ~16x.
    //
    // `_mm_slli_si128(x, n)` is a *byte* shift of the whole 128-bit lane
    // (not a per-element shift -- there's no such per-byte-element shift
    // instruction), shifting bytes toward the high end and filling the
    // low `n` bytes with zero. Adding a byte-shifted copy of `x` into
    // itself is exactly one round of a prefix-sum reduction tree: after
    // shifting by 1 and adding, `x[i]` holds `orig[i] + orig[i-1]`; doing
    // that again shifted by 2 combines pairs of pairs into runs of 4; by
    // 4 into runs of 8; by 8 into the full 16-byte prefix sum. `paddb`
    // (`_mm_add_epi8`) wraps mod 256 on overflow with no extra masking,
    // which is exactly PNG's `wrapping_add` semantics.
    #[cfg(target_arch = "x86_64")]
    fn scan_sse2(data: &mut [u8]) {
        use std::arch::x86_64::{
            _mm_add_epi8, _mm_loadu_si128, _mm_set1_epi8, _mm_setzero_si128, _mm_slli_si128,
            _mm_storeu_si128,
        };

        const LANES: usize = 16;
        let len = data.len();
        let chunks = len / LANES;

        // Running total carried out of the previous chunk, broadcast to
        // all 16 lanes so it can just be added into the next chunk's own
        // (already-computed) internal prefix sum. Starts at 0 (there's no
        // carry into the first chunk).
        //
        // SAFETY: `offset` ranges over `0..chunks*LANES <= len`, so every
        // 16-byte unaligned load/store stays in bounds; `_mm_loadu_si128`/
        // `_mm_storeu_si128` don't require any particular alignment.
        unsafe {
            let mut carry = _mm_setzero_si128();
            for i in 0..chunks {
                let offset = i * LANES;
                let mut x = _mm_loadu_si128(data.as_ptr().add(offset).cast());
                x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
                x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
                x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
                x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
                x = _mm_add_epi8(x, carry);
                _mm_storeu_si128(data.as_mut_ptr().add(offset).cast(), x);

                // The new carry for the *next* chunk is this chunk's last
                // (highest-index) lane, i.e. its own total -- broadcast
                // across all 16 lanes via a byte array round-trip (no
                // dedicated "broadcast last lane" SSE2 instruction; that's
                // `pshufb`, SSSE3+, one step above this baseline).
                let mut last = [0u8; 16];
                _mm_storeu_si128(last.as_mut_ptr().cast(), x);
                #[allow(clippy::cast_possible_wrap)] // bit pattern only: paddb doesn't care about signedness.
                let carry_byte = last[15] as i8;
                carry = _mm_set1_epi8(carry_byte);
            }
        }

        let tail_start = chunks * LANES;
        // The scalar tail continues the running sum from wherever the
        // vectorized part left off: that's just the last byte the SIMD
        // loop wrote (its own inclusive prefix sum already folds in every
        // earlier chunk's carry), or 0 if there were no full chunks at
        // all.
        let carry_in = tail_start.checked_sub(1).map_or(0, |i| data[i]);
        Self::scan_scalar_from(&mut data[tail_start..], carry_in);
    }

    /// Plain scalar running sum, starting from `carry_in` instead of 0 --
    /// used both as the entire implementation on non-`x86_64` targets and
    /// as the tail continuation after the SSE2 chunks on `x86_64`.
    fn scan_scalar_from(data: &mut [u8], carry_in: u8) {
        let mut running = carry_in;
        for byte in data {
            running = running.wrapping_add(*byte);
            *byte = running;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn scan_scalar(data: &mut [u8]) {
        Self::scan_scalar_from(data, 0);
    }
}

/// SIMD-accelerated wrapping prefix sum over `bpp`-interleaved lanes --
/// the multi-channel generalization of [`WrappingPrefixSum`] (which only
/// handles the single-lane `bpp == 1` case).
///
/// `Sub`'s reconstruction `Recon(x) = Filt(x) + Recon(x - bpp)` is, for
/// `bpp > 1`, `bpp` *independent* running sums interleaved byte-for-byte:
/// lane `k` (all bytes at position `i` with `i % bpp == k`) needs its own
/// prefix sum, but lane `k`'s sum has no dependency on any other lane.
/// Grayscale/indexed (`bpp == 1`) is the special case where there's only
/// one lane and the whole row is it; every other color type/bit-depth
/// combination PNG defines has `bpp` in `{2, 3, 4, 6, 8}` (2/4-channel
/// 8/16-bit and 3-channel 8/16-bit truecolor), which is what this handles.
#[derive(Debug)]
pub struct WrappingLaneScan;

impl WrappingLaneScan {
    /// Computes an in-place inclusive wrapping (mod-256) prefix sum
    /// independently within each of `bpp` interleaved lanes: afterwards,
    /// for every `i`, `data[i]` holds the wrapping sum of `data_before[j]`
    /// over every `j <= i` with `j % bpp == i % bpp` (i.e. every earlier
    /// position sharing `i`'s lane).
    ///
    /// `bpp` must be in `2..=8` for the SSE2 fast path to trigger; any
    /// other value (including the `bpp == 1` case handled by
    /// [`WrappingPrefixSum`] instead) falls back to the scalar per-lane
    /// loop, which is correct for every `bpp` -- the SIMD path is a pure
    /// optimization of the same math, not a different algorithm.
    pub fn scan(data: &mut [u8], bpp: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            match bpp {
                2 => return Self::scan_sse2::<2, 3>(data, [2, 4, 8]),
                3 => return Self::scan_sse2::<3, 3>(data, [3, 6, 12]),
                4 => return Self::scan_sse2::<4, 2>(data, [4, 8, 0]),
                6 => return Self::scan_sse2::<6, 2>(data, [6, 12, 0]),
                8 => return Self::scan_sse2::<8, 1>(data, [8, 0, 0]),
                _ => {}
            }
        }
        Self::scan_scalar(data, bpp);
    }

    /// Plain per-lane scalar running sum: used as the fallback for `bpp`
    /// values the SSE2 path doesn't special-case, and as the entire
    /// implementation on non-`x86_64` targets.
    fn scan_scalar(data: &mut [u8], bpp: usize) {
        for lane in 0..bpp {
            let mut running = 0u8;
            let mut i = lane;
            while i < data.len() {
                running = running.wrapping_add(data[i]);
                data[i] = running;
                i += bpp;
            }
        }
    }

    /// SSE2 per-lane Hillis-Steele scan, generalizing [`WrappingPrefixSum::
    /// scan_sse2`]'s single-lane version to `BPP` interleaved lanes.
    ///
    /// `SHIFTS` is the sequence of byte-shift amounts (as multiples of
    /// `BPP`, doubling each round: `BPP, 2*BPP, 4*BPP, ...`) needed to
    /// fully combine every element within a 16-byte register into its
    /// lane's own prefix sum -- unused trailing rounds beyond `PASSES` are
    /// padded with `0` and skipped. Shifting the *whole* 128-bit register
    /// by a multiple of `BPP` bytes is what keeps lane alignment intact
    /// across the shift (`result[i] = a[i - k*BPP]`, and `(i - k*BPP) %
    /// BPP == i % BPP`), unlike shifting by an arbitrary amount, which
    /// would mix bytes from different color channels together.
    ///
    /// Processes `chunk_bytes = (16 / BPP) * BPP` bytes per iteration --
    /// the largest whole number of *complete pixels* that fits in one
    /// 16-byte register -- rather than a full 16 bytes every time, so
    /// that consecutive chunks stay phase-aligned at lane 0 even when
    /// `BPP` doesn't evenly divide 16 (`BPP == 3` or `6`); the register
    /// still loads (and, transiently, computes over) up to 16 bytes for
    /// simplicity, but only the pixel-aligned prefix is ever written back,
    /// so the unwritten remainder is re-read intact (still original,
    /// unfiltered-only) by the next iteration or the scalar tail.
    #[cfg(target_arch = "x86_64")]
    fn scan_sse2<const BPP: usize, const PASSES: usize>(data: &mut [u8], shifts: [i32; 3]) {
        use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128};

        let pixels_per_chunk = 16 / BPP;
        let chunk_bytes = pixels_per_chunk * BPP;
        let len = data.len();

        let mut carry = [0u8; BPP]; // this lane's running total carried out of the previous chunk.
        let mut offset = 0usize;
        // Each iteration needs a full 16-byte lookahead to load safely,
        // even though it only advances (and writes back) `chunk_bytes` --
        // stop the SIMD loop once that lookahead would run past the end,
        // and let the scalar tail (which needs no such lookahead) finish
        // the rest, continuing from this loop's own last carry.
        while offset + 16 <= len {
            // SAFETY: `offset + 16 <= len` was just checked, so the
            // 16-byte unaligned load is in bounds; only the first
            // `chunk_bytes <= 16` bytes of `scanned` are ever written back
            // into `data`, and `offset + chunk_bytes <= offset + 16 <=
            // len`, so that store is in bounds too.
            let mut x = unsafe { _mm_loadu_si128(data.as_ptr().add(offset).cast()) };
            // SAFETY: shift amounts used here are compile-time constants
            // (elements of `shifts`, threaded through as plain `i32`
            // function arguments) in `0..=15` for every `BPP` this is
            // instantiated with, which `_mm_slli_si128`'s immediate-shift
            // intrinsic requires -- checked once per call below rather
            // than relying on `shifts`' construction alone.
            for &shift in shifts.iter().take(PASSES) {
                x = Self::shift_and_add(x, shift);
            }
            let mut scanned = [0u8; 16];
            // SAFETY: `scanned` is a local, fully-owned 16-byte buffer.
            unsafe { _mm_storeu_si128(scanned.as_mut_ptr().cast(), x) };

            // Fold in the previous chunk's per-lane carry (broadcast to
            // every pixel's worth of that lane within this chunk) before
            // writing back -- can't be part of the SIMD reduction above
            // since it's a *per-lane*, not per-register-wide, addend.
            for (i, byte) in scanned.iter_mut().enumerate().take(chunk_bytes) {
                *byte = byte.wrapping_add(carry[i % BPP]);
            }

            data[offset..offset + chunk_bytes].copy_from_slice(&scanned[..chunk_bytes]);

            // The new carry is this chunk's own last pixel, one byte per
            // lane, read back from what was just written (already
            // includes the fold-in above).
            carry.copy_from_slice(&data[offset + chunk_bytes - BPP..offset + chunk_bytes]);

            offset += chunk_bytes;
        }

        // Scalar tail: same per-lane running-sum shape as `scan_scalar`,
        // but seeded from this loop's carry instead of starting at 0, and
        // walking `data[offset..]` instead of the whole slice.
        let tail = &mut data[offset..];
        for (lane, &carry_in) in carry.iter().enumerate().take(BPP) {
            let mut running = carry_in;
            let mut i = lane;
            while i < tail.len() {
                running = running.wrapping_add(tail[i]);
                tail[i] = running;
                i += BPP;
            }
        }
    }

    /// Dispatches to the one `_mm_slli_si128` immediate this call needs --
    /// `shift` is always one of a small fixed set of compile-time-known
    /// values (an element of `scan_sse2`'s own `shifts` array literals),
    /// but arrives here as a plain runtime `i32` (it's read out of an
    /// array via a loop, not spelled as a literal at the call site), and
    /// `_mm_slli_si128`'s shift count must be a `const` -- hence the
    /// match turning "one of a few runtime values" back into "one of a
    /// few compile-time immediates" the intrinsic can actually take.
    #[cfg(target_arch = "x86_64")]
    fn shift_and_add(x: std::arch::x86_64::__m128i, shift: i32) -> std::arch::x86_64::__m128i {
        use std::arch::x86_64::{_mm_add_epi8, _mm_slli_si128};
        let shifted = match shift {
            1 => unsafe { _mm_slli_si128::<1>(x) },
            2 => unsafe { _mm_slli_si128::<2>(x) },
            3 => unsafe { _mm_slli_si128::<3>(x) },
            4 => unsafe { _mm_slli_si128::<4>(x) },
            6 => unsafe { _mm_slli_si128::<6>(x) },
            8 => unsafe { _mm_slli_si128::<8>(x) },
            12 => unsafe { _mm_slli_si128::<12>(x) },
            _ => return x, // shift == 0: padding entry for unused passes, a no-op.
        };
        unsafe { _mm_add_epi8(x, shifted) }
    }
}

/// `Average` filter reconstruction: `Recon(x) = Filt(x) + floor((Recon(a) +
/// Recon(b)) / 2)`, where `a` is the `bpp`-strided left neighbor (already
/// reconstructed, same row) and `b` is the byte directly above (the
/// previous row, entirely already known).
///
/// # Why this stays scalar
///
/// An earlier version of this struct vectorized the per-pixel `avg(a, b)`
/// computation with SSE2's `pavgb`, loading each pixel's `bpp` (<=8) bytes
/// into a 128-bit register via a zero-padded on-stack buffer, computing,
/// and storing back out through another stack buffer. Benchmarked against
/// `ixiptla-bench`'s isolated `defilter_truecolor_alpha_average_only` case
/// at 1024px, that version was *consistently slower* than a plain scalar
/// loop (~13ms vs ~3.7ms median) -- the load/store round-trip for a
/// register that's using at most 8 of its 16 lanes, once per *single
/// pixel* (this recurrence is inherently one-pixel-at-a-time -- `a`
/// depends on the immediately preceding pixel), costs far more than the
/// handful of scalar ALU ops it replaces. `Sub`/`Up`'s SIMD wins work
/// because they either batch many pixels per register (`Sub`, via
/// `WrappingLaneScan`) or have no per-row sequential dependency at all
/// (`Up`). `Average` has neither: it's genuinely one pixel per step, and
/// at `bpp <= 8` a pixel is too narrow to amortize a 128-bit load/store.
/// So this is deliberately the plain scalar loop, kept as its own struct
/// (rather than inlined into `Defilter`) purely for symmetry with the
/// other `Wrapping*` scan types and to keep `defilter.rs` itself free of
/// the loop body.
#[derive(Debug)]
pub struct WrappingAverageScan;

impl WrappingAverageScan {
    /// Reconstructs `content` in place given `previous_row` (`b`, same
    /// length) and `bpp` (the fixed left-neighbor stride). The first `bpp`
    /// bytes (`a == 0`) must already be handled by the caller -- this only
    /// walks the general `i >= bpp` suffix, exactly like `Defilter::
    /// undo_average`'s own split.
    ///
    /// # Panics
    ///
    /// If `previous_row.len()` != `content.len()`.
    pub fn scan_suffix(content: &mut [u8], previous_row: &[u8], bpp: usize) {
        assert_eq!(
            content.len(),
            previous_row.len(),
            "scan_suffix: length mismatch"
        );
        #[allow(clippy::needless_range_loop)]
        for i in bpp..content.len() {
            let a = content[i - bpp];
            let b = previous_row[i];
            // midpoint, but in a wider type: a + b can be up to 510, which
            // doesn't fit in u8, before the floor-divide brings it back
            // down.
            let avg = u16::midpoint(u16::from(a), u16::from(b));
            #[allow(clippy::cast_possible_truncation)]
            let avg = avg as u8; // avg <= 255 always: (255+255)/2 = 255.
            content[i] = content[i].wrapping_add(avg);
        }
    }
}

/// `Paeth` filter reconstruction: `Recon(x) = Filt(x) + PaethPredictor(
/// Recon(a), Recon(b), Recon(c))`.
///
/// # Why this stays scalar
///
/// Same finding as [`WrappingAverageScan`], more pronounced: an earlier
/// SSE2 version (zero-extending each pixel's `bpp` bytes into `i16`
/// lanes, computing the 3-way `abs`/compare/select predictor there, and
/// packing back to `u8`, all through on-stack load/store buffers per
/// single pixel) benchmarked *slower* than the plain scalar loop on
/// `defilter_truecolor_alpha_paeth_only` at 1024px (~16.5ms vs ~14.2ms
/// median) -- `Paeth`'s recurrence is one pixel per step just like
/// `Average`, so the same narrow-register load/store tax applies, and
/// `Paeth`'s predictor needs *more* instructions per register op (3-way
/// abs+compare+blend against baseline SSE2's missing pieces: no `pabsw`,
/// no `pblendvb`) than `Average`'s single `pavgb`, making the tradeoff
/// even worse. Kept as its own struct for the same symmetry/SRP reasons
/// as `WrappingAverageScan`.
#[derive(Debug)]
pub struct WrappingPaethScan;

impl WrappingPaethScan {
    /// Reconstructs `content` in place given `previous_row` (`b`/`c`
    /// source, same length) and `bpp`. The first `bpp` bytes (`a == c ==
    /// 0`, where `PaethPredictor` always reduces to `b`) must already be
    /// handled by the caller, exactly like `Defilter::undo_paeth`'s own
    /// split.
    ///
    /// # Panics
    ///
    /// If `previous_row.len()` != `content.len()`.
    pub fn scan_suffix(content: &mut [u8], previous_row: &[u8], bpp: usize) {
        assert_eq!(
            content.len(),
            previous_row.len(),
            "scan_suffix: length mismatch"
        );
        #[allow(clippy::needless_range_loop)]
        for i in bpp..content.len() {
            let a = content[i - bpp];
            let b = previous_row[i];
            let c = previous_row[i - bpp];
            content[i] = content[i].wrapping_add(Self::paeth_predictor(a, b, c));
        }
    }

    /// Spec §9.4's Paeth predictor: picks whichever of the left/above/
    /// upper-left neighbor is closest to `a + b - c` (linear-gradient
    /// prediction), with ties broken in the order left, above, upper-left.
    fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
        let (a, b, c) = (i32::from(a), i32::from(b), i32::from(c));
        let p = a + b - c;
        let pa = (p - a).abs();
        let pb = (p - b).abs();
        let pc = (p - c).abs();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        if pa <= pb && pa <= pc {
            a as u8
        } else if pb <= pc {
            b as u8
        } else {
            c as u8
        }
    }
}


#[cfg(test)]
mod tests {
    use super::{WrappingAverageScan, WrappingByteAdder, WrappingLaneScan, WrappingPaethScan, WrappingPrefixSum};

    #[test]
    fn matches_scalar_wrapping_add_elementwise() {
        #[allow(clippy::cast_possible_truncation)]
        let mut dst: Vec<u8> = (0..250u16).map(|n| (n % 256) as u8).collect();
        #[allow(clippy::cast_possible_truncation)]
        let src: Vec<u8> = (0..250u16).map(|n| ((n * 7 + 3) % 256) as u8).collect();
        let expected: Vec<u8> = dst
            .iter()
            .zip(&src)
            .map(|(&d, &s)| d.wrapping_add(s))
            .collect();
        WrappingByteAdder::add_wrapping(&mut dst, &src);
        assert_eq!(dst, expected);
    }

    #[test]
    fn handles_lengths_not_a_multiple_of_simd_width() {
        for len in 0..40 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let mut dst: Vec<u8> = (0..len as u16).map(|n| (n * 3) as u8).collect();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let src: Vec<u8> = (0..len as u16).map(|n| (n * 11) as u8).collect();
            let expected: Vec<u8> = dst
                .iter()
                .zip(&src)
                .map(|(&d, &s)| d.wrapping_add(s))
                .collect();
            WrappingByteAdder::add_wrapping(&mut dst, &src);
            assert_eq!(dst, expected, "len={len}");
        }
    }

    #[test]
    fn wraps_on_overflow() {
        let mut dst = vec![250u8, 255, 200];
        let src = vec![10u8, 1, 100];
        WrappingByteAdder::add_wrapping(&mut dst, &src);
        assert_eq!(dst, vec![4u8, 0, 44]); // 260%256=4, 256%256=0, 300%256=44
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn panics_on_length_mismatch() {
        let mut dst = vec![1u8, 2, 3];
        let src = vec![1u8, 2];
        WrappingByteAdder::add_wrapping(&mut dst, &src);
    }

    /// Reference scalar prefix sum, mirroring `Sub`'s own reconstruction
    /// formula at `bpp == 1` -- used to check `WrappingPrefixSum::scan`
    /// against, independent of its own internal SIMD/scalar split.
    fn scalar_prefix_sum(data: &[u8]) -> Vec<u8> {
        let mut running = 0u8;
        data.iter()
            .map(|&b| {
                running = running.wrapping_add(b);
                running
            })
            .collect()
    }

    #[test]
    fn matches_scalar_prefix_sum_for_various_lengths() {
        // Cover both sides of every SSE2 chunk boundary (0, 15, 16, 17,
        // ..., a couple of full chunks, and past several chunks) so both
        // the vectorized loop and its scalar tail continuation get
        // exercised.
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 100, 257] {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let data: Vec<u8> = (0..len as u32).map(|n| ((n * 37 + 5) % 256) as u8).collect();
            let expected = scalar_prefix_sum(&data);
            let mut actual = data.clone();
            WrappingPrefixSum::scan(&mut actual);
            assert_eq!(actual, expected, "len={len}");
        }
    }

    #[test]
    fn prefix_sum_wraps_on_overflow() {
        let mut data = vec![200u8, 100, 50, 10];
        // running: 200, 300%256=44, 44+50=94, 94+10=104
        WrappingPrefixSum::scan(&mut data);
        assert_eq!(data, vec![200u8, 44, 94, 104]);
    }

    #[test]
    fn prefix_sum_of_empty_slice_is_a_no_op() {
        let mut data: Vec<u8> = vec![];
        WrappingPrefixSum::scan(&mut data);
        assert_eq!(data, Vec::<u8>::new());
    }

    /// Reference scalar per-lane prefix sum, mirroring `Sub`'s own
    /// reconstruction formula at general `bpp` -- used to check
    /// `WrappingLaneScan::scan` independent of its own SIMD/scalar split.
    fn scalar_lane_scan(data: &[u8], bpp: usize) -> Vec<u8> {
        let mut out = data.to_vec();
        for lane in 0..bpp {
            let mut running = 0u8;
            let mut i = lane;
            while i < out.len() {
                running = running.wrapping_add(out[i]);
                out[i] = running;
                i += bpp;
            }
        }
        out
    }

    #[test]
    fn lane_scan_matches_scalar_reference_for_every_supported_bpp() {
        // Every bpp PNG's spec actually produces (1 handled separately by
        // WrappingPrefixSum; 2/3/4/6/8 here), across a spread of lengths
        // that cross the SSE2 chunk boundary at various phases relative to
        // bpp (a chunk is (16/bpp)*bpp bytes, not always 16), including
        // lengths not a whole number of pixels (truncated final pixel --
        // never produced by a real IDAT stream, but the scan must still
        // not go out of bounds or panic).
        for bpp in [2usize, 3, 4, 6, 8] {
            for len in [0usize, 1, bpp, bpp * 2, 15, 16, 17, 31, 32, 33, 100, 251] {
                #[allow(clippy::cast_possible_truncation)]
                let data: Vec<u8> = (0..len as u32).map(|n| ((n * 41 + 13) % 256) as u8).collect();
                let expected = scalar_lane_scan(&data, bpp);
                let mut actual = data.clone();
                WrappingLaneScan::scan(&mut actual, bpp);
                assert_eq!(actual, expected, "bpp={bpp} len={len}");
            }
        }
    }

    #[test]
    fn lane_scan_wraps_on_overflow_per_lane() {
        // bpp=2, two interleaved lanes: [200, 5, 100, 250, 50, 10]
        // lane 0 (idx 0,2,4): 200, 300%256=44, 44+50=94
        // lane 1 (idx 1,3,5): 5, 5+250=255, 255+10=265%256=9
        let mut data = vec![200u8, 5, 100, 250, 50, 10];
        WrappingLaneScan::scan(&mut data, 2);
        assert_eq!(data, vec![200u8, 5, 44, 255, 94, 9]);
    }

    #[test]
    fn lane_scan_falls_back_correctly_for_unsupported_bpp() {
        // bpp values PNG never actually produces (e.g. 5, 7) still need to
        // work correctly via the scalar fallback -- WrappingLaneScan::scan
        // doesn't assume bpp is one of the SSE2-special-cased values.
        for bpp in [5usize, 7, 9] {
            #[allow(clippy::cast_possible_truncation)]
            let data: Vec<u8> = (0..40u32).map(|n| (n * 3) as u8).collect();
            let expected = scalar_lane_scan(&data, bpp);
            let mut actual = data.clone();
            WrappingLaneScan::scan(&mut actual, bpp);
            assert_eq!(actual, expected, "bpp={bpp}");
        }
    }

    /// Reference scalar Average-filter suffix reconstruction, mirroring
    /// `Defilter::undo_average`'s own `i >= bpp` loop -- used to check
    /// `WrappingAverageScan::scan_suffix` independent of its own SIMD/
    /// scalar split.
    fn scalar_average_suffix(content: &[u8], previous_row: &[u8], bpp: usize) -> Vec<u8> {
        let mut out = content.to_vec();
        #[allow(clippy::needless_range_loop)]
        for i in bpp..out.len() {
            let a = out[i - bpp];
            let b = previous_row[i];
            let avg = u16::midpoint(u16::from(a), u16::from(b));
            #[allow(clippy::cast_possible_truncation)]
            let avg = avg as u8;
            out[i] = out[i].wrapping_add(avg);
        }
        out
    }

    #[test]
    fn average_scan_matches_scalar_reference_for_every_supported_bpp() {
        // bpp in {1, 2, 3, 4, 6, 8} covers every channel/bit-depth
        // combination PNG actually defines (bpp==1 falls back to scalar
        // internally -- no SSE2 special case, since there's only one lane
        // -- but must still produce the right answer); lengths span both
        // sides of the SSE2 pixel-chunk loop's stopping point.
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            for len in [0usize, bpp, bpp * 2, bpp * 3, 15, 16, 17, 31, 32, 100, 251] {
                if len < bpp {
                    continue;
                }
                #[allow(clippy::cast_possible_truncation)]
                let content: Vec<u8> = (0..len as u32).map(|n| ((n * 29 + 7) % 256) as u8).collect();
                #[allow(clippy::cast_possible_truncation)]
                let previous_row: Vec<u8> = (0..len as u32).map(|n| ((n * 53 + 11) % 256) as u8).collect();
                let expected = scalar_average_suffix(&content, &previous_row, bpp);
                let mut actual = content.clone();
                WrappingAverageScan::scan_suffix(&mut actual, &previous_row, bpp);
                assert_eq!(actual, expected, "bpp={bpp} len={len}");
            }
        }
    }

    #[test]
    fn average_scan_floors_not_rounds() {
        // bpp=1: a=255 (content[0], acting as the lane-0 "already
        // reconstructed" predecessor), b=0 (previous_row[1]) -> avg =
        // floor(255/2) = 127, not 128 -- catches an off-by-one toward
        // pavgb's built-in ceil rounding if the correction is ever wrong.
        let mut content = vec![255u8, 0];
        let previous_row = vec![0u8, 0];
        WrappingAverageScan::scan_suffix(&mut content, &previous_row, 1);
        assert_eq!(content, vec![255u8, 127]);
    }

    /// Reference scalar Paeth-filter suffix reconstruction, mirroring
    /// `Defilter::undo_paeth`'s own `i >= bpp` loop -- used to check
    /// `WrappingPaethScan::scan_suffix` independent of its own SIMD/
    /// scalar split.
    fn scalar_paeth_suffix(content: &[u8], previous_row: &[u8], bpp: usize) -> Vec<u8> {
        fn predictor(a: u8, b: u8, c: u8) -> u8 {
            let (a, b, c) = (i32::from(a), i32::from(b), i32::from(c));
            let p = a + b - c;
            let pa = (p - a).abs();
            let pb = (p - b).abs();
            let pc = (p - c).abs();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            if pa <= pb && pa <= pc {
                a as u8
            } else if pb <= pc {
                b as u8
            } else {
                c as u8
            }
        }
        let mut out = content.to_vec();
        #[allow(clippy::needless_range_loop)]
        for i in bpp..out.len() {
            let a = out[i - bpp];
            let b = previous_row[i];
            let c = previous_row[i - bpp];
            out[i] = out[i].wrapping_add(predictor(a, b, c));
        }
        out
    }

    #[test]
    fn paeth_scan_matches_scalar_reference_for_every_supported_bpp() {
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            for len in [0usize, bpp, bpp * 2, bpp * 3, 15, 16, 17, 31, 32, 100, 251] {
                if len < bpp {
                    continue;
                }
                #[allow(clippy::cast_possible_truncation)]
                let content: Vec<u8> = (0..len as u32).map(|n| ((n * 71 + 3) % 256) as u8).collect();
                #[allow(clippy::cast_possible_truncation)]
                let previous_row: Vec<u8> = (0..len as u32).map(|n| ((n * 97 + 17) % 256) as u8).collect();
                let expected = scalar_paeth_suffix(&content, &previous_row, bpp);
                let mut actual = content.clone();
                WrappingPaethScan::scan_suffix(&mut actual, &previous_row, bpp);
                assert_eq!(actual, expected, "bpp={bpp} len={len}");
            }
        }
    }

    #[test]
    fn paeth_scan_falls_back_correctly_for_unsupported_bpp() {
        // bpp > 8 never occurs in a real PNG (max channels x max byte
        // width = 4 x 2 = 8), but the scalar fallback must still be
        // correct if ever exercised.
        for bpp in [9usize, 12] {
            #[allow(clippy::cast_possible_truncation)]
            let content: Vec<u8> = (0..40u32).map(|n| (n * 5) as u8).collect();
            #[allow(clippy::cast_possible_truncation)]
            let previous_row: Vec<u8> = (0..40u32).map(|n| (n * 13) as u8).collect();
            let expected = scalar_paeth_suffix(&content, &previous_row, bpp);
            let mut actual = content.clone();
            WrappingPaethScan::scan_suffix(&mut actual, &previous_row, bpp);
            assert_eq!(actual, expected, "bpp={bpp}");
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn average_scan_panics_on_length_mismatch() {
        let mut content = vec![1u8, 2, 3];
        let previous_row = vec![1u8, 2];
        WrappingAverageScan::scan_suffix(&mut content, &previous_row, 1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn paeth_scan_panics_on_length_mismatch() {
        let mut content = vec![1u8, 2, 3];
        let previous_row = vec![1u8, 2];
        WrappingPaethScan::scan_suffix(&mut content, &previous_row, 1);
    }
}
