//! `ixiptla-core` — a zero-dependency PNG (ISO/IEC 15948) decoder core.
//!
//! Named (with `ixchel`) to keep the weaving theme: if `ixchel` unweaves
//! DEFLATE's compressed bitstream, `ixiptla` is the figure that stands in
//! for the real image once the pixels are rewoven from it (Nahuatl
//! *ixiptla*, "representation/likeness" — the ritual stand-in for a thing).
//!
//! This crate owns chunk parsing, scanline defiltering, and orchestrating
//! `ixchel::zlib` over the concatenated `IDAT` stream. It does not depend on
//! any external PNG/compression crate.

mod chunk;
mod crc32;
mod signature;

pub use chunk::{Chunk, ChunkError, ChunkReader};
pub use crc32::Crc32;
pub use signature::{Signature, SignatureError};
