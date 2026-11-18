//! Ordered grouped-learning evidence.

use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdSample64, SgdUpdateBatchError, SgdUpdateBatchError64,
};

use super::support::{
    assert_learning_transition_at, assert_learning_transition_at64, bits, bits64, sample, sample64,
    state32, state64,
};

#[test]
fn chunked_and_continuous_folds_match() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let rows = [
        sample(&[0.5, 0.5], 1.0),
        sample(&[-0.5, 0.5], 0.0),
        sample(&[0.5, -0.5], -1.0),
        sample(&[-0.5, -0.5], 0.0),
        sample(&[0.25, 0.75], 0.5),
        sample(&[-0.25, -0.75], -0.5),
    ];
    let continuous = rows.iter().fold(initial.clone(), |state, row| {
        model.update(&state, row).expect("update")
    });
    let single = model
        .update_batch(initial.clone(), &rows, rows.len())
        .expect("single group");
    assert_eq!(bits(&single.w), bits(&continuous.w));
    let mut chunked = initial.clone();
    for chunk in [&rows[..1], &rows[1..3], &rows[3..]] {
        chunked = model
            .update_batch(chunked, chunk, chunk.len())
            .expect("chunk");
    }
    assert_eq!(bits(&chunked.w), bits(&continuous.w));

    let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial64 = model64.initial_state();
    let rows64: Vec<Observation<SgdSample64>> = (0..6)
        .map(|index| {
            let value = index as f64 * 0.125;
            sample64(&[value, -value], value * 0.5)
        })
        .collect();
    let continuous64 = rows64.iter().fold(initial64.clone(), |state, row| {
        model64.update(&state, row).expect("update")
    });
    let mut chunked64 = initial64.clone();
    for chunk in [&rows64[..2], &rows64[2..5], &rows64[5..]] {
        chunked64 = model64
            .update_batch(chunked64, chunk, chunk.len())
            .expect("f64 chunk");
    }
    assert_eq!(bits64(&chunked64.w), bits64(&continuous64.w));
}

#[test]
fn failure_positions_prefix_slice_retry_and_correction() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let valid = [
        sample(&[0.5, 0.5], 1.0),
        sample(&[-0.5, 0.5], 0.0),
        sample(&[0.5, -0.5], -1.0),
        sample(&[-0.5, -0.5], 0.0),
    ];
    for failure_index in [0, 2, 4] {
        let mut rows = valid.to_vec();
        rows.insert(failure_index, sample(&[1.0], 1.0));
        let err = model
            .update_batch(initial.clone(), &rows, rows.len())
            .unwrap_err();
        let (failed_index, committed, prefix) = match &err {
            SgdUpdateBatchError::Transition {
                failed_index,
                committed,
                state,
                error,
            } => {
                assert_eq!(*failed_index, failure_index);
                assert_eq!(*committed, failure_index);
                assert_eq!(
                    *error,
                    SgdError::DimensionMismatch {
                        expected: 2,
                        actual: 1
                    }
                );
                (*failed_index, *committed, state.clone())
            }
            other => panic!("expected Transition, got {other:?}"),
        };
        assert_eq!(failed_index, committed);
        let expected_prefix = rows[..failure_index]
            .iter()
            .fold(initial.clone(), |state, row| {
                model.update(&state, row).expect("update")
            });
        assert_eq!(bits(&prefix.w), bits(&expected_prefix.w));

        let retry = model
            .update_batch(
                prefix.clone(),
                &rows[failure_index..],
                rows.len() - failure_index,
            )
            .unwrap_err();
        match retry {
            SgdUpdateBatchError::Transition {
                failed_index: retry_index,
                committed: retry_committed,
                state,
                error,
            } => {
                assert_eq!(retry_index, 0);
                assert_eq!(retry_committed, 0);
                assert_eq!(bits(&state.w), bits(&prefix.w));
                assert_eq!(
                    error,
                    SgdError::DimensionMismatch {
                        expected: 2,
                        actual: 1
                    }
                );
            }
            other => panic!("expected unchanged retry Transition, got {other:?}"),
        }

        let mut corrected = rows.clone();
        corrected[failure_index] = sample(&[0.25, -0.25], 0.5);
        let expected = model
            .update_batch(initial.clone(), &corrected, corrected.len())
            .expect("corrected full sequence");
        let retried = model
            .update_batch(
                prefix,
                &corrected[failure_index..],
                corrected.len() - failure_index,
            )
            .expect("corrected retry");
        assert_eq!(bits(&retried.w), bits(&expected.w));
    }
}

#[test]
fn f64_failure_positions_prefix_slice_retry_and_correction() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let valid = [
        sample64(&[0.5, 0.5], 1.0),
        sample64(&[-0.5, 0.5], 0.0),
        sample64(&[0.5, -0.5], -1.0),
        sample64(&[-0.5, -0.5], 0.0),
    ];
    for failure_index in [0, 2, 4] {
        let mut rows = valid.to_vec();
        rows.insert(failure_index, sample64(&[1.0], 1.0));
        let err = model
            .update_batch(initial.clone(), &rows, rows.len())
            .unwrap_err();
        let (prefix, failed_index) = match &err {
            SgdUpdateBatchError64::Transition {
                failed_index,
                committed,
                state,
                error,
            } => {
                assert_eq!(*failed_index, failure_index);
                assert_eq!(*committed, failure_index);
                assert_eq!(
                    *error,
                    SgdError64::DimensionMismatch {
                        expected: 2,
                        actual: 1
                    }
                );
                (state.clone(), *failed_index)
            }
            other => panic!("expected f64 Transition, got {other:?}"),
        };
        let expected_prefix = rows[..failed_index]
            .iter()
            .fold(initial.clone(), |state, row| {
                model.update(&state, row).expect("update")
            });
        assert_eq!(bits64(&prefix.w), bits64(&expected_prefix.w));

        let retry = model
            .update_batch(
                prefix.clone(),
                &rows[failed_index..],
                rows.len() - failed_index,
            )
            .unwrap_err();
        match retry {
            SgdUpdateBatchError64::Transition {
                failed_index: retry_index,
                committed: retry_committed,
                state,
                error,
            } => {
                assert_eq!(retry_index, 0);
                assert_eq!(retry_committed, 0);
                assert_eq!(bits64(&state.w), bits64(&prefix.w));
                assert_eq!(
                    error,
                    SgdError64::DimensionMismatch {
                        expected: 2,
                        actual: 1
                    }
                );
            }
            other => panic!("expected f64 unchanged retry Transition, got {other:?}"),
        }

        let mut corrected = rows.clone();
        corrected[failed_index] = sample64(&[0.25, -0.25], 0.5);
        let expected = model
            .update_batch(initial.clone(), &corrected, corrected.len())
            .expect("corrected f64 full sequence");
        let retried = model
            .update_batch(
                prefix,
                &corrected[failed_index..],
                corrected.len() - failed_index,
            )
            .expect("corrected f64 retry");
        assert_eq!(bits64(&retried.w), bits64(&expected.w));
    }
}

#[test]
fn learning_validation_precedence_32() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let valid = sample(&[0.5, 0.5], 1.0);
    assert_learning_transition_at(
        &model,
        &initial,
        &[valid.clone(), sample(&[1.0], 1.0)],
        1,
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 1,
        },
    );
    assert_learning_transition_at(
        &model,
        &state32(&[0.0, 0.0, 0.0]),
        &[valid.clone(), sample(&[1.0, 1.0], 1.0)],
        0,
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 3,
        },
    );
    assert_learning_transition_at(
        &model,
        &initial,
        &[valid.clone(), sample(&[f32::NAN, 1.0], 1.0)],
        1,
        SgdError::NonFiniteInput,
    );
    assert_learning_transition_at(
        &model,
        &initial,
        &[valid, sample(&[1.0, 1.0], f32::NAN)],
        1,
        SgdError::NonFiniteInput,
    );
    let overflow_model = Sgd::new(1, 1.0e38).expect("valid configuration");
    assert_learning_transition_at(
        &overflow_model,
        &state32(&[1.0e38]),
        &[sample(&[0.0], 0.0), sample(&[1.0e38], 0.0)],
        1,
        SgdError::NonFiniteCandidate,
    );
}

#[test]
fn learning_validation_precedence_64() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let valid = sample64(&[0.5, 0.5], 1.0);
    assert_learning_transition_at64(
        &model,
        &initial,
        &[valid.clone(), sample64(&[1.0], 1.0)],
        1,
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 1,
        },
    );
    assert_learning_transition_at64(
        &model,
        &state64(&[0.0, 0.0, 0.0]),
        &[valid.clone(), sample64(&[1.0, 1.0], 1.0)],
        0,
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 3,
        },
    );
    assert_learning_transition_at64(
        &model,
        &initial,
        &[valid.clone(), sample64(&[f64::NAN, 1.0], 1.0)],
        1,
        SgdError64::NonFiniteInput,
    );
    assert_learning_transition_at64(
        &model,
        &initial,
        &[valid, sample64(&[1.0, 1.0], f64::NAN)],
        1,
        SgdError64::NonFiniteInput,
    );
    let overflow_model = Sgd64::new(1, 1.0e308).expect("valid configuration");
    assert_learning_transition_at64(
        &overflow_model,
        &state64(&[1.0e308]),
        &[sample64(&[0.0], 0.0), sample64(&[1.0e308], 0.0)],
        1,
        SgdError64::NonFiniteCandidate,
    );
}
