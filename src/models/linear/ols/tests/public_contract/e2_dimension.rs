use super::support::{assert_matches_f64_oracle, model};
use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::RegressionError;

// --- E.2 dimension correctness ----------------------------------------------

#[test]
fn predict_rejects_short_and_long_inputs() {
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0, 2.0, 3.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn predict_rejects_zero_length_input() {
    let model = model(&[1.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[])),
        Err(RegressionError::DimensionMismatch {
            expected: 1,
            actual: 0,
        })
    );
}

#[test]
fn validation_order_dimension_before_finiteness() {
    // Wrong dimension and non-finite simultaneously: the documented order
    // reports the dimension mismatch first.
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[f32::NAN])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn non_finite_input_rejected_all_kinds() {
    let model = model(&[1.0, 2.0], 0.0);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            model.predict(&Vector::from_slice(&[1.0, value])),
            Err(RegressionError::NonFiniteInput),
            "value = {value}"
        );
    }
}

#[test]
fn matching_dimension_succeeds() {
    for dimension in [1_usize, 2, 4, 8] {
        let weights: Vec<f32> = (0..dimension).map(|index| index as f32 * 0.25).collect();
        let features: Vec<f32> = (0..dimension).map(|index| index as f32 * -0.5).collect();
        let model = model(&weights, 0.125);
        let prediction = model.predict(&Vector::from_slice(&features)).unwrap();
        assert!(prediction.is_finite());
        assert_matches_f64_oracle(&model, &features);
    }
}
