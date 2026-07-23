/// The 8-byte magic PNG files must start with (spec §5.2): high bit set (to
/// catch 7-bit transmission), "PNG", CRLF, Ctrl-Z, LF (catches text-mode
/// CR/LF translation and truncation of the file).
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Recognizes/strips the fixed 8-byte PNG signature at the front of a file.
///
/// Kept as its own zero-sized-ish struct (rather than a free function) so
/// it has a natural home for the "why" documentation above and so the
/// decoder's top level can name what it's doing (`Signature::strip(bytes)`)
/// rather than an anonymous slice comparison.
#[derive(Debug)]
pub struct Signature;

impl Signature {
    /// Verify `input` starts with the PNG signature and return the
    /// remaining bytes (the chunk stream) after it.
    pub fn strip(input: &[u8]) -> Result<&[u8], SignatureError> {
        if input.len() < PNG_SIGNATURE.len() {
            return Err(SignatureError::TooShort);
        }
        let (head, rest) = input.split_at(PNG_SIGNATURE.len());
        if head == PNG_SIGNATURE {
            Ok(rest)
        } else {
            Err(SignatureError::Mismatch)
        }
    }
}

/// Why [`Signature::strip`] rejected an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureError {
    /// Fewer than 8 bytes were provided, so there isn't even a full
    /// signature to compare.
    TooShort,
    /// The first 8 bytes are present but don't match the PNG magic.
    Mismatch,
}

impl core::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort => write!(f, "input shorter than the 8-byte PNG signature"),
            Self::Mismatch => write!(f, "input does not start with the PNG signature"),
        }
    }
}

impl std::error::Error for SignatureError {}

#[cfg(test)]
mod tests {
    use super::{Signature, SignatureError};

    #[test]
    fn strips_valid_signature() {
        let mut input = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        input.extend_from_slice(b"rest of the file");
        let rest = Signature::strip(&input).unwrap();
        assert_eq!(rest, b"rest of the file");
    }

    #[test]
    fn rejects_too_short() {
        let input = [0x89, b'P', b'N'];
        assert_eq!(Signature::strip(&input), Err(SignatureError::TooShort));
    }

    #[test]
    fn rejects_wrong_magic() {
        let input = [0u8; 16];
        assert_eq!(Signature::strip(&input), Err(SignatureError::Mismatch));
    }

    #[test]
    fn rejects_text_mode_mangled_signature() {
        // A common corruption: CRLF -> LF translation by a text-mode
        // transfer, dropping the 0x0D byte.
        let input = [0x89, b'P', b'N', b'G', 0x0A, 0x1A, 0x0A, 0x00];
        assert_eq!(Signature::strip(&input), Err(SignatureError::Mismatch));
    }
}
