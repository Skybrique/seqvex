//! Strict weight-slice serving rejection classification.
//!
//! The conversion operation remains in `models::linear::strict_serving`; this
//! module owns the shared family rejection classification and its behavior.

use std::fmt;

/// First rejection encountered while converting a strict weight slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrictWeightConversionError {
    /// A weight source was non-finite.
    SourceNonFinite { index: usize },
    /// A finite weight source overflowed `f32`.
    Overflow { index: usize },
    /// A nonzero weight source rounded to zero.
    Underflow { index: usize },
}

impl fmt::Display for StrictWeightConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceNonFinite { index } => {
                write!(formatter, "non-finite weight source at index {index}")
            }
            Self::Overflow { index } => write!(formatter, "weight overflow at index {index}"),
            Self::Underflow { index } => write!(formatter, "weight underflow at index {index}"),
        }
    }
}

impl std::error::Error for StrictWeightConversionError {}
