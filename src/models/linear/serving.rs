//! Policy-neutral `f64` to `f32` narrowing for linear serving conversion.
//!
//! The helper classifies only the numerical outcome. Each algorithm applies its
//! own public error variant and underflow policy.

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Narrowing {
    /// Finite in-range source, with ordinary `as f32` rounding.
    Value(f32),
    /// Finite nonzero source that narrowed to signed zero.
    Underflow(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NarrowingError {
    SourceNonFinite,
    Overflow,
}

pub(crate) fn narrow_f64(value: f64) -> Result<Narrowing, NarrowingError> {
    if !value.is_finite() {
        return Err(NarrowingError::SourceNonFinite);
    }
    let converted = value as f32;
    if !converted.is_finite() {
        return Err(NarrowingError::Overflow);
    }
    if value != 0.0 && converted == 0.0 {
        Ok(Narrowing::Underflow(converted))
    } else {
        Ok(Narrowing::Value(converted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_non_finite_overflow_signed_zero_subnormal_and_underflow() {
        assert_eq!(narrow_f64(f64::NAN), Err(NarrowingError::SourceNonFinite));
        assert_eq!(
            narrow_f64(f64::INFINITY),
            Err(NarrowingError::SourceNonFinite)
        );
        assert_eq!(
            narrow_f64(f64::NEG_INFINITY),
            Err(NarrowingError::SourceNonFinite)
        );
        assert_eq!(narrow_f64(f64::MAX), Err(NarrowingError::Overflow));
        assert_eq!(narrow_f64(0.0), Ok(Narrowing::Value(0.0)));
        assert_eq!(
            narrow_f64(-0.0).map(|n| match n {
                Narrowing::Value(v) => v.to_bits(),
                Narrowing::Underflow(v) => v.to_bits(),
            }),
            Ok((-0.0_f32).to_bits())
        );
        assert_eq!(narrow_f64(1.5), Ok(Narrowing::Value(1.5)));
        assert!(matches!(
            narrow_f64(f64::from(f32::MIN_POSITIVE) * 0.5),
            Ok(Narrowing::Value(value)) if value.is_subnormal() && value != 0.0
        ));
        assert!(matches!(
            narrow_f64(f64::from(f32::MIN_POSITIVE) * 1.0e-10),
            Ok(Narrowing::Underflow(value)) if value == 0.0
        ));
    }
}
