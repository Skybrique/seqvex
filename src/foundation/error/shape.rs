//! Shared shape errors.
//!
//! `DimensionMismatch` is defined here once for the shared foundation error
//! layer. Numerical containers and model families import or re-export this same
//! type; the leaf vector module keeps a compatibility re-export.

/// Error returned when operands or parameters have incompatible dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch {
    /// The dimension required by the operation.
    pub expected: usize,
    /// The dimension actually supplied.
    pub actual: usize,
}

impl std::fmt::Display for DimensionMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "dimension mismatch: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for DimensionMismatch {}
