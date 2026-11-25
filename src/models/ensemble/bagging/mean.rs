//! Reviewed serving-precision mean contract for the regression bagger.

use std::fmt;

/// Aggregation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanError {
    /// The reduction overflowed or produced a non-finite result.
    MeanOverflow,
    /// The residual bound could not certify the result or its sign.
    MeanCancellation,
}

impl fmt::Display for MeanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeanOverflow => formatter.write_str("bag mean overflowed"),
            Self::MeanCancellation => {
                formatter.write_str("bag mean cancellation could not be certified")
            }
        }
    }
}

impl std::error::Error for MeanError {}

macro_rules! mean_impl {
    (
        $module:ident,
        $ty:ty,
        $bits:ty,
        $eta:expr,
        $min_normal:expr,
        $frac_bits:expr,
        $sign_shift:expr,
        $tau:expr,
        $u:expr
    ) => {
        mod $module {
            use super::MeanError;

            const ETA: $ty = $eta;
            const MIN_NORMAL: $ty = $min_normal;
            const FRAC_BITS: u32 = $frac_bits;
            const SIGN_SHIFT: u32 = $sign_shift;
            const SIGN_MASK: $bits = (1 as $bits) << SIGN_SHIFT;
            const FRAC_MASK: $bits = ((1 as $bits) << FRAC_BITS) - 1;
            const TAU: $ty = $tau;
            const UNIT_ROUNDOFF: $ty = $u;
            const TWO_ETA: $ty = ETA + ETA;

            fn next_up(x: $ty) -> $ty {
                if x.is_nan() || x == <$ty>::INFINITY {
                    return x;
                }
                if x == 0.0 {
                    return <$ty>::from_bits(1);
                }
                let bits = x.to_bits();
                if x > 0.0 {
                    <$ty>::from_bits(bits + 1)
                } else {
                    <$ty>::from_bits(bits - 1)
                }
            }

            fn next_down(x: $ty) -> $ty {
                if x.is_nan() || x == <$ty>::NEG_INFINITY {
                    return x;
                }
                if x == 0.0 {
                    return -<$ty>::from_bits(1);
                }
                let bits = x.to_bits();
                if x > 0.0 {
                    <$ty>::from_bits(bits - 1)
                } else {
                    <$ty>::from_bits(bits + 1)
                }
            }

            fn two_sum(a: $ty, b: $ty) -> ($ty, $ty) {
                let s = a + b;
                let bb = s - a;
                let e = (a - (s - bb)) + (b - bb);
                (s, e)
            }

            fn add_up(a: $ty, b: $ty) -> $ty {
                if a == 0.0 {
                    b
                } else if b == 0.0 {
                    a
                } else {
                    let (s, e) = two_sum(a, b);
                    if e > 0.0 { next_up(s) } else { s }
                }
            }

            fn add_down(a: $ty, b: $ty) -> $ty {
                if a == 0.0 {
                    b
                } else if b == 0.0 {
                    a
                } else {
                    let (s, e) = two_sum(a, b);
                    if e < 0.0 { next_down(s) } else { s }
                }
            }

            fn sub_down(a: $ty, b: $ty) -> $ty {
                if b == 0.0 {
                    a
                } else if a == 0.0 {
                    -b
                } else {
                    let (s, e) = two_sum(a, -b);
                    if e < 0.0 { next_down(s) } else { s }
                }
            }

            fn sub_up(a: $ty, b: $ty) -> $ty {
                if b == 0.0 {
                    a
                } else if a == 0.0 {
                    -b
                } else {
                    let (s, e) = two_sum(a, -b);
                    if e > 0.0 { next_up(s) } else { s }
                }
            }

            fn div_up_pos(a: $ty, m: usize) -> $ty {
                if a == 0.0 {
                    return 0.0;
                }
                let q = a / (m as $ty);
                if q == 0.0 {
                    return ETA;
                }
                let r = (-q).mul_add(m as $ty, a);
                if r > 0.0 { next_up(q) } else { q }
            }

            fn all_negative_zero(predictions: &[$ty]) -> bool {
                predictions.iter().all(|value| value.to_bits() == SIGN_MASK)
            }

            fn zero_for(predictions: &[$ty]) -> $ty {
                if all_negative_zero(predictions) {
                    -0.0
                } else {
                    0.0
                }
            }

            fn pairwise_exact_zero(predictions: &[$ty]) -> bool {
                let mut used = vec![false; predictions.len()];
                for i in 0..predictions.len() {
                    if used[i] || predictions[i] == 0.0 {
                        continue;
                    }
                    let target = predictions[i].to_bits() ^ SIGN_MASK;
                    let mut found = false;
                    for j in 0..predictions.len() {
                        if i != j && !used[j] && predictions[j].to_bits() == target {
                            used[i] = true;
                            used[j] = true;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return false;
                    }
                }
                true
            }

            fn all_nonzero_subnormal(predictions: &[$ty]) -> bool {
                predictions
                    .iter()
                    .all(|value| value == &0.0 || value.abs() < MIN_NORMAL)
            }

            fn signed_subnormal(predictions: &[$ty]) -> Result<$ty, MeanError> {
                let m = predictions.len() as i128;
                let mut sum: i128 = 0;
                for value in predictions {
                    let bits = value.to_bits();
                    let magnitude = (bits & FRAC_MASK) as i128;
                    if bits & SIGN_MASK != 0 {
                        sum -= magnitude;
                    } else {
                        sum += magnitude;
                    }
                }
                if sum == 0 {
                    return Ok(zero_for(predictions));
                }
                let negative = sum < 0;
                let absolute = sum.unsigned_abs();
                let q = absolute / m as u128;
                let r = absolute % m as u128;
                let mut rounded = q;
                let twice = r * 2;
                if twice > m as u128 || (twice == m as u128 && q % 2 == 1) {
                    rounded += 1;
                }
                let bits = if rounded == 0 {
                    0
                } else if rounded < (1u128 << FRAC_BITS) {
                    rounded as $bits
                } else {
                    let value = (rounded as $ty) * ETA;
                    return Ok(if negative { -value } else { value });
                };
                let bits = if negative { bits | SIGN_MASK } else { bits };
                Ok(<$ty>::from_bits(bits))
            }

            pub(super) fn mean(predictions: &[$ty]) -> Result<$ty, MeanError> {
                if predictions.is_empty() {
                    return Err(MeanError::MeanCancellation);
                }
                if predictions.iter().any(|value| !value.is_finite()) {
                    return Err(MeanError::MeanOverflow);
                }
                let max_abs = predictions
                    .iter()
                    .fold(0.0 as $ty, |acc, value| acc.max(value.abs()));
                if max_abs == 0.0 {
                    return Ok(zero_for(predictions));
                }
                if all_nonzero_subnormal(predictions) {
                    return signed_subnormal(predictions);
                }
                if pairwise_exact_zero(predictions) {
                    return Ok(zero_for(predictions));
                }

                let mut hi = 0.0 as $ty;
                let mut lo = 0.0 as $ty;
                let mut positive_lo = 0.0 as $ty;
                let mut positive_hi = 0.0 as $ty;
                let mut negative_lo = 0.0 as $ty;
                let mut negative_hi = 0.0 as $ty;

                for value in predictions {
                    let (s, e) = two_sum(hi, *value);
                    let (t, d) = two_sum(lo, e);
                    let (next_hi, next_lo) = two_sum(s, t);
                    hi = next_hi;
                    lo = next_lo;
                    if d > 0.0 {
                        positive_lo = add_down(positive_lo, d);
                        positive_hi = add_up(positive_hi, d);
                    } else if d < 0.0 {
                        negative_lo = add_down(negative_lo, -d);
                        negative_hi = add_up(negative_hi, -d);
                    }
                }

                if !hi.is_finite() || !lo.is_finite() {
                    return Err(MeanError::MeanOverflow);
                }

                let m = predictions.len();
                let mean_est = hi / (m as $ty);
                if !mean_est.is_finite() {
                    return Err(MeanError::MeanOverflow);
                }

                let d_sum_upper = add_up(positive_hi, negative_hi);
                let d_mean_upper = div_up_pos(d_sum_upper, m);
                let lo_upper = div_up_pos(lo.abs(), m);
                let r_div = if mean_est.abs() >= MIN_NORMAL {
                    let term = UNIT_ROUNDOFF * mean_est.abs();
                    if term == 0.0 { ETA } else { next_up(term) }
                } else {
                    ETA
                };
                let total = add_up(add_up(d_mean_upper, lo_upper), r_div);
                if !total.is_finite() {
                    return Err(MeanError::MeanCancellation);
                }

                if mean_est.abs() >= MIN_NORMAL {
                    let threshold = next_down((TAU / 2.0) * mean_est.abs());
                    if total <= threshold {
                        Ok(mean_est)
                    } else {
                        Err(MeanError::MeanCancellation)
                    }
                } else if total <= TWO_ETA {
                    if mean_est != 0.0 {
                        return Ok(mean_est);
                    }
                    let low = add_down(add_down(sub_down(positive_lo, negative_hi), hi), lo);
                    let high = add_up(add_up(sub_up(positive_hi, negative_lo), hi), lo);
                    if low > 0.0 {
                        Ok(0.0)
                    } else if high < 0.0 {
                        Ok(-0.0)
                    } else if low == 0.0 && high == 0.0 {
                        Ok(zero_for(predictions))
                    } else {
                        Err(MeanError::MeanCancellation)
                    }
                } else {
                    Err(MeanError::MeanCancellation)
                }
            }
        }
    };
}

mean_impl!(
    f64_impl,
    f64,
    u64,
    f64::from_bits(1),
    f64::from_bits(1u64 << 52),
    52,
    63,
    f64::from_bits((1023u64 - 40) << 52),
    f64::from_bits((1023u64 - 53) << 52)
);

mean_impl!(
    f32_impl,
    f32,
    u32,
    f32::from_bits(1),
    f32::from_bits(1u32 << 23),
    23,
    31,
    f32::from_bits((127u32 - 20) << 23),
    f32::from_bits((127u32 - 24) << 23)
);

pub(crate) fn mean_f64(predictions: &[f64]) -> Result<f64, MeanError> {
    f64_impl::mean(predictions)
}

pub(crate) fn mean_f32(predictions: &[f32]) -> Result<f32, MeanError> {
    f32_impl::mean(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ETA64: f64 = f64::from_bits(1);
    const ETA32: f32 = f32::from_bits(1);
    const M64: f64 = f64::from_bits((1u64 << 52) - 1);
    const M32: f32 = f32::from_bits((1u32 << 23) - 1);

    #[test]
    fn ordinary_means() {
        assert_eq!(mean_f64(&[2.0, 6.0]).unwrap(), 4.0);
        assert_eq!(mean_f32(&[2.0, 6.0]).unwrap(), 4.0);
        assert_eq!(mean_f64(&[1.0, ETA64]).unwrap(), 0.5);
        assert_eq!(mean_f32(&[1.0, ETA32]).unwrap(), 0.5);
        assert_eq!(mean_f64(&[ETA64, ETA64, ETA64, 2.0]).unwrap(), 0.5);
        assert_eq!(mean_f32(&[ETA32, ETA32, ETA32, 2.0]).unwrap(), 0.5);
    }

    #[test]
    fn min_normal_and_subnormal_boundaries() {
        let n64 = f64::from_bits(1u64 << 52);
        assert_eq!(mean_f64(&[n64, -(n64 - 2.0 * ETA64)]).unwrap(), ETA64);
        let n32 = f32::from_bits(1u32 << 23);
        assert_eq!(mean_f32(&[n32, -(n32 - 2.0 * ETA32)]).unwrap(), ETA32);
        assert_eq!(
            mean_f64(&[M64, M64, M64 - 2.0 * ETA64]).unwrap(),
            M64 - ETA64
        );
        assert_eq!(
            mean_f32(&[M32, M32, M32 - 2.0 * ETA32]).unwrap(),
            M32 - ETA32
        );
    }

    #[test]
    fn signed_zero_cases() {
        assert_eq!(mean_f64(&[ETA64, 0.0]).unwrap().to_bits(), 0);
        assert_eq!(
            mean_f64(&[-ETA64, 0.0]).unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(mean_f64(&[ETA64, -ETA64]).unwrap().to_bits(), 0);
        assert_eq!(
            mean_f64(&[-0.0, -0.0]).unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(mean_f64(&[0.0, -0.0]).unwrap().to_bits(), 0);
    }

    #[test]
    fn exact_zero_and_cancellation() {
        assert_eq!(mean_f64(&[1.0, -1.0]).unwrap().to_bits(), 0);
        let max = f64::MAX;
        assert_eq!(mean_f64(&[max, max, -max, -max]).unwrap().to_bits(), 0);
        assert_eq!(
            mean_f64(&[2f64.powi(60), 1.0, 2f64.powi(-60), -2f64.powi(60), -1.0]),
            Err(MeanError::MeanCancellation)
        );
        assert_eq!(mean_f64(&[max, max, max]), Err(MeanError::MeanOverflow));
    }

    #[test]
    fn mixed_sign_zero_fixtures() {
        assert_eq!(
            mean_f64(&[
                2f64.powi(60),
                1.0,
                2.0 * ETA64,
                -ETA64,
                -2f64.powi(60),
                -1.0
            ])
            .unwrap()
            .to_bits(),
            0
        );
        assert_eq!(
            mean_f64(&[
                2f64.powi(60),
                1.0,
                -2.0 * ETA64,
                ETA64,
                -2f64.powi(60),
                -1.0
            ])
            .unwrap()
            .to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            mean_f64(&[2f64.powi(60), 1.0, -ETA64, -2f64.powi(60), -1.0])
                .unwrap()
                .to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn computed_negative_zero_uses_signed_interval() {
        let positive = [
            2f64.powi(60),
            1.0,
            2.0 * ETA64,
            -2f64.powi(60),
            -1.0,
            -ETA64,
        ];
        assert_eq!(mean_f64(&positive).unwrap().to_bits(), 0);
        let exact_zero = [
            2f64.powi(60),
            1.0,
            2.0 * ETA64,
            -2f64.powi(60),
            -1.0,
            -ETA64,
            -ETA64,
        ];
        assert_eq!(mean_f64(&exact_zero).unwrap().to_bits(), 0);

        let positive32 = [
            2f32.powi(60),
            1.0,
            2.0 * ETA32,
            -2f32.powi(60),
            -1.0,
            -ETA32,
        ];
        assert_eq!(mean_f32(&positive32).unwrap().to_bits(), 0);
        let exact_zero32 = [
            2f32.powi(60),
            1.0,
            2.0 * ETA32,
            -2f32.powi(60),
            -1.0,
            -ETA32,
            -ETA32,
        ];
        assert_eq!(mean_f32(&exact_zero32).unwrap().to_bits(), 0);
    }

    #[test]
    fn near_zero_success() {
        assert!(mean_f64(&[1.0, 1e-20, -1.0]).unwrap().is_finite());
        assert_eq!(
            mean_f64(&[1.0, -1.0 + 2f64.powi(-52)]).unwrap(),
            2f64.powi(-53)
        );
        assert_eq!(
            mean_f32(&[1.0, -1.0 + 2f32.powi(-16)]).unwrap(),
            2f32.powi(-17)
        );
    }
}
