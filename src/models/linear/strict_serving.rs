//! Shared strict `f64` to `f32` serving conversion for RLS and SGD.
//!
//! The source weights are borrowed and never modified. Weights are visited in
//! ascending index order and conversion stops at the first rejection. Underflow
//! is rejected: a nonzero source that rounds to zero is not an acceptable strict
//! serving parameter. On success the returned vector owns the converted values;
//! on failure any partially built vector is dropped and no serving snapshot is
//! returned.
//!
//! This is the linear-family strict serving policy. The policy-neutral scalar
//! narrowing operation lives in the foundation numerical `narrowing` module; the
//! shared rejection classification is owned by `models::linear::error`. Public
//! RLS/SGD errors, dimension checks and source-state ownership remain with their
//! algorithms.

use crate::foundation::numerical::narrowing::{Narrowing, NarrowingError, narrow_f64};
use crate::models::linear::error::StrictWeightConversionError;

/// Converts every weight in ascending index order, stopping at the first
/// rejection.
///
/// The source slice is borrowed and never modified. On success the returned
/// vector owns the converted values; on failure any partially built vector is
/// dropped and no snapshot is returned.
pub(crate) fn convert_strict_weights(
    sources: &[f64],
) -> Result<Vec<f32>, StrictWeightConversionError> {
    let mut weights = Vec::with_capacity(sources.len());
    for (index, &source) in sources.iter().enumerate() {
        match narrow_f64(source) {
            Err(NarrowingError::SourceNonFinite) => {
                return Err(StrictWeightConversionError::SourceNonFinite { index });
            }
            Err(NarrowingError::Overflow) => {
                return Err(StrictWeightConversionError::Overflow { index });
            }
            Ok(Narrowing::Underflow(_)) => {
                return Err(StrictWeightConversionError::Underflow { index });
            }
            Ok(Narrowing::Value(value)) => weights.push(value),
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn strict_conversion_preserves_bits_and_stops_at_first_failure() {
        let converted = convert_strict_weights(&[1.0, -0.0, 1.5]).unwrap();
        assert_eq!(converted[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(converted[1].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(converted[2].to_bits(), 1.5_f32.to_bits());

        assert_eq!(
            convert_strict_weights(&[1.0, f64::NAN, f64::MAX]),
            Err(StrictWeightConversionError::SourceNonFinite { index: 1 })
        );
        assert_eq!(
            convert_strict_weights(&[1.0, f64::MAX]),
            Err(StrictWeightConversionError::Overflow { index: 1 })
        );
        assert_eq!(
            convert_strict_weights(&[1.0, f64::from(f32::MIN_POSITIVE) * 1.0e-10]),
            Err(StrictWeightConversionError::Underflow { index: 1 })
        );
        assert!(
            StrictWeightConversionError::Underflow { index: 0 }
                .source()
                .is_none()
        );
    }
}
