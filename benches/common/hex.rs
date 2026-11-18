//! Dependency-free lowercase hexadecimal formatting for non-production support.

/// Formats bytes as lowercase hexadecimal, preserving byte order and leading
/// zero digits.
pub(crate) fn encode(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut text, "{byte:02x}").expect("writing to String");
    }
    text
}
