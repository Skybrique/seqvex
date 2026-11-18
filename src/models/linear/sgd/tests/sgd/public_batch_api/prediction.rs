//! Grouped-prediction evidence.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdPredictionError, SgdPredictionError64, SgdServing32,
};

use super::support::{
    assert_prediction_at, assert_prediction_at64, bits, bits64, features, features64, state32,
    state64,
};

#[test]
fn prediction_failure_positions_and_no_partial_vector() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let state = state32(&[0.5, -0.5]);
    let state_bits = bits(&state.w);
    let valid = [
        features(&[0.5, 0.5]),
        features(&[-0.5, 0.5]),
        features(&[0.5, -0.5]),
    ];
    for failure_index in [0, 1, 2] {
        let mut rows = valid.to_vec();
        rows.insert(failure_index, features(&[f32::NAN, 0.0]));
        match model.predict_batch(&state, &rows, rows.len()) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, failure_index);
                assert_eq!(error, SgdError::NonFiniteInput);
            }
            other => panic!("expected At, got {other:?}"),
        }
        assert_eq!(bits(&state.w), state_bits);
    }

    let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
    let state64 = state64(&[0.5, -0.5]);
    let state64_bits = bits64(&state64.w);
    let rows64 = [features64(&[0.5, 0.5]), features64(&[f64::NAN, 0.0])];
    match model64.predict_batch(&state64, &rows64, rows64.len()) {
        Err(SgdPredictionError64::At { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(error, SgdError64::NonFiniteInput);
        }
        other => panic!("expected f64 At, got {other:?}"),
    }
    assert_eq!(bits64(&state64.w), state64_bits);
}

#[test]
fn b_snapshot_prediction_failure_preserves_snapshot() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let snapshot: SgdServing32 = model
        .to_serving_f32(&state64(&[1.0, -2.0]))
        .expect("snapshot");
    let snapshot_bits = bits(snapshot.weights());
    let rows = [features(&[0.5, 0.5]), features(&[f32::NAN, 0.0])];
    match snapshot.predict_batch(&rows, rows.len()) {
        Err(SgdPredictionError::At { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(error, SgdError::NonFiniteInput);
        }
        other => panic!("expected B At, got {other:?}"),
    }
    assert_eq!(bits(snapshot.weights()), snapshot_bits);
}

#[test]
fn prediction_failure_positions_cover_a_b_and_c_at_three_rows() {
    let model_c = Sgd::new(2, 0.1).expect("valid C configuration");
    let state_c = state32(&[0.5, -0.5]);
    let c_bits = bits(&state_c.w);
    let valid_c = [
        features(&[0.5, 0.5]),
        features(&[-0.5, 0.5]),
        features(&[0.25, -0.75]),
    ];
    for failure_index in [0, 1, valid_c.len() - 1] {
        let mut rows = valid_c.to_vec();
        rows[failure_index] = features(&[f32::NAN, 0.0]);
        match model_c.predict_batch(&state_c, &rows, rows.len()) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, failure_index);
                assert_eq!(error, SgdError::NonFiniteInput);
            }
            other => panic!("expected C At, got {other:?}"),
        }
        assert_eq!(bits(&state_c.w), c_bits);
    }

    let model_a = Sgd64::new(2, 0.1).expect("valid A configuration");
    let state_a = state64(&[0.5, -0.5]);
    let a_bits = bits64(&state_a.w);
    let valid_a = [
        features64(&[0.5, 0.5]),
        features64(&[-0.5, 0.5]),
        features64(&[0.25, -0.75]),
    ];
    for failure_index in [0, 1, valid_a.len() - 1] {
        let mut rows = valid_a.to_vec();
        rows[failure_index] = features64(&[f64::NAN, 0.0]);
        match model_a.predict_batch(&state_a, &rows, rows.len()) {
            Err(SgdPredictionError64::At { index, error }) => {
                assert_eq!(index, failure_index);
                assert_eq!(error, SgdError64::NonFiniteInput);
            }
            other => panic!("expected A At, got {other:?}"),
        }
        assert_eq!(bits64(&state_a.w), a_bits);
    }

    let snapshot = model_a.to_serving_f32(&state_a).expect("B snapshot");
    let b_bits = bits(snapshot.weights());
    let valid_b = [
        features(&[0.5, 0.5]),
        features(&[-0.5, 0.5]),
        features(&[0.25, -0.75]),
    ];
    for failure_index in [0, 1, valid_b.len() - 1] {
        let mut rows = valid_b.to_vec();
        rows[failure_index] = features(&[f32::NAN, 0.0]);
        match snapshot.predict_batch(&rows, rows.len()) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, failure_index);
                assert_eq!(error, SgdError::NonFiniteInput);
            }
            other => panic!("expected B At, got {other:?}"),
        }
        assert_eq!(bits(snapshot.weights()), b_bits);
    }
}

#[test]
fn b_zero_limit_and_oversize_reject_before_validation() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let snapshot = model
        .to_serving_f32(&state64(&[1.0, -2.0]))
        .expect("snapshot");
    let snapshot_bits = bits(snapshot.weights());
    let rows = [
        features(&[f32::NAN, 0.0]),
        features(&[0.5, 0.5]),
        features(&[-0.5, 0.5]),
    ];
    assert!(matches!(
        snapshot.predict_batch(&rows, 0),
        Err(SgdPredictionError::TooLarge { max: 0, actual: 3 })
    ));
    assert!(matches!(
        snapshot.predict_batch(&rows, rows.len() - 1),
        Err(SgdPredictionError::TooLarge { max: 2, actual: 3 })
    ));
    assert!(matches!(
        snapshot.predict_batch(&rows[..1], 0),
        Err(SgdPredictionError::TooLarge { max: 0, actual: 1 })
    ));
    assert_eq!(bits(snapshot.weights()), snapshot_bits);
}

#[test]
fn empty_prediction_with_zero_limit_succeeds_for_a_and_b() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let state = state64(&[0.5, -0.5]);
    let empty_a: [Observation<Vector64>; 0] = [];
    assert!(
        model
            .predict_batch(&state, &empty_a, 0)
            .expect("empty A prediction")
            .is_empty()
    );

    let snapshot = model.to_serving_f32(&state).expect("snapshot");
    let empty_b: [Observation<Vector>; 0] = [];
    assert!(
        snapshot
            .predict_batch(&empty_b, 0)
            .expect("empty B prediction")
            .is_empty()
    );
}

#[test]
fn prediction_validation_precedence_32() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let valid = features(&[0.5, 0.5]);
    assert_prediction_at(
        &model,
        &initial,
        &[valid.clone(), features(&[1.0])],
        1,
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 1,
        },
    );
    assert_prediction_at(
        &model,
        &state32(&[0.0, 0.0, 0.0]),
        &[valid.clone(), features(&[1.0, 1.0])],
        0,
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 3,
        },
    );
    assert_prediction_at(
        &model,
        &initial,
        &[valid.clone(), features(&[f32::NAN, 0.0])],
        1,
        SgdError::NonFiniteInput,
    );
    assert_prediction_at(
        &model,
        &state32(&[f32::MAX, 0.0]),
        &[features(&[0.0, 0.0]), features(&[f32::MAX, 0.0])],
        1,
        SgdError::NonFiniteOutput,
    );
}

#[test]
fn prediction_validation_precedence_64() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let valid = features64(&[0.5, 0.5]);
    assert_prediction_at64(
        &model,
        &initial,
        &[valid.clone(), features64(&[1.0])],
        1,
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 1,
        },
    );
    assert_prediction_at64(
        &model,
        &state64(&[0.0, 0.0, 0.0]),
        &[valid.clone(), features64(&[1.0, 1.0])],
        0,
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 3,
        },
    );
    assert_prediction_at64(
        &model,
        &initial,
        &[valid.clone(), features64(&[f64::NAN, 0.0])],
        1,
        SgdError64::NonFiniteInput,
    );
    assert_prediction_at64(
        &model,
        &state64(&[f64::MAX, 0.0]),
        &[features64(&[0.0, 0.0]), features64(&[f64::MAX, 0.0])],
        1,
        SgdError64::NonFiniteOutput,
    );
}

#[test]
fn prediction_validation_precedence_b() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let snapshot = model
        .to_serving_f32(&state64(&[1.0, -2.0]))
        .expect("snapshot");
    let valid = features(&[0.5, 0.5]);
    match snapshot.predict_batch(&[valid.clone(), features(&[1.0])], 2) {
        Err(SgdPredictionError::At { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(
                error,
                SgdError::DimensionMismatch {
                    expected: 2,
                    actual: 1
                }
            );
        }
        other => panic!("expected B dimension At, got {other:?}"),
    }
    match snapshot.predict_batch(&[valid, features(&[f32::NAN, 0.0])], 2) {
        Err(SgdPredictionError::At { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(error, SgdError::NonFiniteInput);
        }
        other => panic!("expected B input At, got {other:?}"),
    }
    let overflow_snapshot = model
        .to_serving_f32(&state64(&[f32::MAX as f64, 0.0]))
        .expect("overflow-capable snapshot");
    match overflow_snapshot.predict_batch(&[features(&[0.0, 0.0]), features(&[f32::MAX, 0.0])], 2) {
        Err(SgdPredictionError::At { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(error, SgdError::NonFiniteOutput);
        }
        other => panic!("expected B output At, got {other:?}"),
    }
}
