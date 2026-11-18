//! Shared narrowing rejection classification.
//!
//! The policy-neutral `f64` to `f32` operation remains in the foundation
//! numerical `narrowing` module. This module owns only the shared rejection
//! classification and its error-specific behavior.

use std::fmt;

/// Classification of a rejected narrowing attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NarrowingError {
    /// Source was non-finite.
    SourceNonFinite,
    /// Source was finite but not representable as a finite `f32`.
    Overflow,
}

impl fmt::Display for NarrowingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceNonFinite => formatter.write_str("non-finite source value"),
            Self::Overflow => formatter.write_str("source value overflows f32"),
        }
    }
}

impl std::error::Error for NarrowingError {}
