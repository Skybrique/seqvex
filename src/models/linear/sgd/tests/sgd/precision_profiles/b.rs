//! Checked B serving conversion, replacement and prediction evidence.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd64, SgdError, SgdServingConversionError, SgdState64};

use super::support::{bits64, reference_update64, sample64};
use crate::bits;

#[test]
fn conversion_exact_bits_ties_and_special_values() {
    let model = Sgd64::new(4, 0.1).expect("valid configuration");
    let state = SgdState64 {
        w: Vector64::from_slice(&[1.0, -2.0, 1.0 + 2f64.powi(-24), 1.0 + 3.0 * 2f64.powi(-24)]),
    };
    let snapshot = model.to_serving_f32(&state).expect("serving snapshot");
    let expected = [1.0f32, -2.0, 1.0, 1.0 + 2f32.powi(-22)];
    for (actual, expected) in snapshot.weights().as_slice().iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    let model = Sgd64::new(4, 0.1).expect("valid configuration");
    let state = SgdState64 {
        w: Vector64::from_slice(&[f32::MAX as f64, f32::from_bits(1) as f64, -0.0, 0.0]),
    };
    let snapshot = model.to_serving_f32(&state).expect("special values");
    assert_eq!(
        snapshot.weights().as_slice()[0].to_bits(),
        f32::MAX.to_bits()
    );
    assert_eq!(snapshot.weights().as_slice()[1].to_bits(), 1);
    assert_eq!(
        snapshot.weights().as_slice()[2].to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(snapshot.weights().as_slice()[3].to_bits(), 0.0f32.to_bits());

    let model = Sgd64::new(1, 0.1).expect("valid configuration");
    let underflow = SgdState64 {
        w: Vector64::from_slice(&[2f64.powi(-200)]),
    };
    assert_eq!(
        model.to_serving_f32(&underflow).unwrap_err(),
        SgdServingConversionError::Underflow { index: 0 }
    );
}

#[test]
fn conversion_dimension_overflow_nonfinite_and_preservation() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let source = SgdState64 {
        w: Vector64::from_slice(&[f64::NAN]),
    };
    let source_bits = bits64(&source.w);
    assert_eq!(
        model.to_serving_f32(&source).unwrap_err(),
        SgdServingConversionError::StateDimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(bits64(&source.w), source_bits);

    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let source = SgdState64 {
        w: Vector64::from_slice(&[f64::NAN, 1.0]),
    };
    assert_eq!(
        model.to_serving_f32(&source).unwrap_err(),
        SgdServingConversionError::SourceNonFinite { index: 0 }
    );
    let source = SgdState64 {
        w: Vector64::from_slice(&[1.0, f64::INFINITY]),
    };
    assert_eq!(
        model.to_serving_f32(&source).unwrap_err(),
        SgdServingConversionError::SourceNonFinite { index: 1 }
    );

    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let source = SgdState64 {
        w: Vector64::from_slice(&[f32::MAX as f64, f64::MAX]),
    };
    let source_bits = bits64(&source.w);
    assert_eq!(
        model.to_serving_f32(&source).unwrap_err(),
        SgdServingConversionError::Overflow { index: 1 }
    );
    assert_eq!(bits64(&source.w), source_bits);

    let model = Sgd64::new(1, 0.1).expect("valid configuration");
    let active = model
        .to_serving_f32(&SgdState64 {
            w: Vector64::from_slice(&[1.0]),
        })
        .expect("active snapshot");
    let active_bits = bits(active.weights());
    let failed = model.to_serving_f32(&SgdState64 {
        w: Vector64::from_slice(&[f64::MAX]),
    });
    assert_eq!(
        failed.unwrap_err(),
        SgdServingConversionError::Overflow { index: 0 }
    );
    assert_eq!(bits(active.weights()), active_bits);
}

#[test]
fn continued_learning_and_serving_replacement() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let mut s0 = model
        .update(&model.initial_state(), &sample64(&[1.0, -0.5], 0.5))
        .expect("learned state");
    let b0 = model.to_serving_f32(&s0).expect("snapshot B0");
    let b0_bits = bits(b0.weights());
    let b0_prediction = b0
        .predict(&Vector::from_slice(&[0.5, -0.5]))
        .expect("B0 prediction")
        .to_bits();

    let expected_s1 = reference_update64(0.1, s0.w.as_slice(), &[0.25, 1.0], -0.25);
    s0 = model
        .update(&s0, &sample64(&[0.25, 1.0], -0.25))
        .expect("continued learning");
    assert_eq!(bits(b0.weights()), b0_bits, "B0 must remain immutable");
    let s1 = s0;
    assert_eq!(
        bits64(&s1.w),
        expected_s1.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let b1 = model.to_serving_f32(&s1).expect("snapshot B1");
    assert_ne!(bits(b1.weights()), b0_bits);
    assert_eq!(
        b0.predict(&Vector::from_slice(&[0.5, -0.5]))
            .expect("B0 prediction")
            .to_bits(),
        b0_prediction
    );

    let other_model = Sgd64::new(3, 0.1).expect("valid configuration");
    let s1_bits = bits64(&s1.w);
    assert_eq!(
        other_model.to_serving_f32(&s1).unwrap_err(),
        SgdServingConversionError::StateDimensionMismatch {
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(bits64(&s1.w), s1_bits);
    assert_eq!(bits(b0.weights()), b0_bits);

    let candidate_model = Sgd64::new(1, 0.1).expect("valid configuration");
    let candidate = candidate_model
        .to_serving_f32(&SgdState64 {
            w: Vector64::from_slice(&[f32::MAX as f64]),
        })
        .expect("candidate snapshot");
    assert_eq!(
        candidate.predict(&Vector::from_slice(&[2.0])).unwrap_err(),
        SgdError::NonFiniteOutput
    );
    assert_eq!(
        bits(b0.weights()),
        b0_bits,
        "failed replacement preserves active"
    );
}

#[test]
fn b_prediction_validation_and_overflow() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let snapshot = model
        .to_serving_f32(&SgdState64 {
            w: Vector64::from_slice(&[1.0, -2.0]),
        })
        .expect("snapshot");
    assert_eq!(
        snapshot.predict(&Vector::from_slice(&[1.0])).unwrap_err(),
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        snapshot
            .predict(&Vector::from_slice(&[f32::NAN, 0.0]))
            .unwrap_err(),
        SgdError::NonFiniteInput
    );
    let bits_before = bits(snapshot.weights());
    let _ = snapshot.predict(&Vector::from_slice(&[0.5, -0.5]));
    assert_eq!(bits(snapshot.weights()), bits_before);

    let model = Sgd64::new(1, 0.1).expect("valid configuration");
    let overflow = model
        .to_serving_f32(&SgdState64 {
            w: Vector64::from_slice(&[f32::MAX as f64]),
        })
        .expect("overflow-capable snapshot");
    assert_eq!(
        overflow.predict(&Vector::from_slice(&[2.0])).unwrap_err(),
        SgdError::NonFiniteOutput
    );
}
