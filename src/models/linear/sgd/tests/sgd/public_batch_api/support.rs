//! Shared helpers for the public grouped SGD API evidence.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdPredictionError, SgdPredictionError64, SgdSample,
    SgdSample64, SgdState, SgdState64, SgdUpdateBatchError, SgdUpdateBatchError64,
};

pub(super) fn sample(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

pub(super) fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

pub(super) fn features(values: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(values))
}

pub(super) fn features64(values: &[f64]) -> Observation<Vector64> {
    Observation::new(Vector64::from_slice(values))
}

pub(super) fn state32(values: &[f32]) -> SgdState {
    SgdState {
        w: Vector::from_slice(values),
    }
}

pub(super) fn state64(values: &[f64]) -> SgdState64 {
    SgdState64 {
        w: Vector64::from_slice(values),
    }
}

pub(super) fn bits(vector: &Vector) -> Vec<u32> {
    vector
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

pub(super) fn bits64(vector: &Vector64) -> Vec<u64> {
    vector
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

pub(super) fn bits32_vec(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub(super) fn bits64_vec(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub(super) fn assert_learning_transition_at(
    model: &Sgd,
    initial: &SgdState,
    rows: &[Observation<SgdSample>],
    expected_index: usize,
    expected: SgdError,
) {
    match model.update_batch(initial.clone(), rows, rows.len()) {
        Err(SgdUpdateBatchError::Transition {
            failed_index,
            committed,
            error,
            ..
        }) => {
            assert_eq!(failed_index, expected_index);
            assert_eq!(committed, expected_index);
            assert_eq!(error, expected);
        }
        other => panic!("expected learning Transition, got {other:?}"),
    }
}

pub(super) fn assert_learning_transition_at64(
    model: &Sgd64,
    initial: &SgdState64,
    rows: &[Observation<SgdSample64>],
    expected_index: usize,
    expected: SgdError64,
) {
    match model.update_batch(initial.clone(), rows, rows.len()) {
        Err(SgdUpdateBatchError64::Transition {
            failed_index,
            committed,
            error,
            ..
        }) => {
            assert_eq!(failed_index, expected_index);
            assert_eq!(committed, expected_index);
            assert_eq!(error, expected);
        }
        other => panic!("expected f64 learning Transition, got {other:?}"),
    }
}

pub(super) fn assert_prediction_at(
    model: &Sgd,
    state: &SgdState,
    rows: &[Observation<Vector>],
    expected_index: usize,
    expected: SgdError,
) {
    match model.predict_batch(state, rows, rows.len()) {
        Err(SgdPredictionError::At { index, error }) => {
            assert_eq!(index, expected_index);
            assert_eq!(error, expected);
        }
        other => panic!("expected prediction At, got {other:?}"),
    }
}

pub(super) fn assert_prediction_at64(
    model: &Sgd64,
    state: &SgdState64,
    rows: &[Observation<Vector64>],
    expected_index: usize,
    expected: SgdError64,
) {
    match model.predict_batch(state, rows, rows.len()) {
        Err(SgdPredictionError64::At { index, error }) => {
            assert_eq!(index, expected_index);
            assert_eq!(error, expected);
        }
        other => panic!("expected f64 prediction At, got {other:?}"),
    }
}
