use crate::support::{
    BatchFailure, PredictionFailure, PredictionRecord, collect32, collect64, replay_sgd32_observed,
    replay_sgd64_observed,
};

use crate::{assert_f64_bits, rls_sample32, rls_sample64, sample32, sample64};

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, Rls64, RlsBatchError, RlsError};
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdUpdateBatchError, SgdUpdateBatchError64,
};

#[test]
fn bounded_sgd_and_rls_admission_and_oversize_recovery() {
    let rows32 = vec![
        sample32(&[1.0, 0.0], 1.0),
        sample32(&[0.0, 1.0], -1.0),
        sample32(&[1.0, 1.0], 0.5),
    ];
    let model32 = Sgd::new(2, 0.125).expect("valid SGD configuration");
    let learned32 = model32
        .update(&model32.initial_state(), &rows32[0])
        .expect("nonzero learned C state");
    let supplied32 = learned32.clone();
    match model32.update_batch(learned32, &rows32, 2) {
        Err(SgdUpdateBatchError::TooLarge { max, actual, state }) => {
            assert_eq!((max, actual), (2, 3));
            assert_eq!(state, supplied32);
            let continued = model32
                .update_batch(state.clone(), &rows32[..2], 2)
                .unwrap();
            let expected = rows32[..2]
                .iter()
                .try_fold(supplied32, |state, row| model32.update(&state, row))
                .unwrap();
            assert_eq!(continued, expected);
        }
        other => panic!("expected oversized C rejection, got {other:?}"),
    }

    let rows64 = vec![
        sample64(&[1.0, 0.0], 1.0),
        sample64(&[0.0, 1.0], -1.0),
        sample64(&[1.0, 1.0], 0.5),
    ];
    let model64 = Sgd64::new(2, 0.125).expect("valid f64 SGD configuration");
    let learned64 = model64
        .update(&model64.initial_state(), &rows64[0])
        .expect("nonzero learned A state");
    let supplied64 = learned64.clone();
    match model64.update_batch(learned64, &rows64, 2) {
        Err(SgdUpdateBatchError64::TooLarge { state, .. }) => {
            assert_eq!(state, supplied64);
            let continued = model64
                .update_batch(state.clone(), &rows64[..2], 2)
                .unwrap();
            let expected = rows64[..2]
                .iter()
                .try_fold(supplied64, |state, row| model64.update(&state, row))
                .unwrap();
            assert_eq!(continued, expected);
        }
        other => panic!("expected oversized A rejection, got {other:?}"),
    }

    // Oversize wins over invalid observations, and the state is returned.
    let invalid = vec![
        sample32(&[1.0, 0.0], 1.0),
        sample32(&[0.0, f32::NAN], 1.0),
        sample32(&[1.0, 1.0], 1.0),
    ];
    match model32.update_batch(model32.initial_state(), &invalid, 1) {
        Err(SgdUpdateBatchError::TooLarge { state, .. }) => {
            assert_eq!(state, model32.initial_state());
        }
        other => panic!("expected bound precedence, got {other:?}"),
    }

    let rls_rows = vec![
        rls_sample32(&[1.0, 0.0], 1.0),
        rls_sample32(&[0.0, 1.0], -1.0),
        rls_sample32(&[1.0, 1.0], 0.5),
    ];
    let rls = Rls::new(2, 0.9, 1.0).expect("valid RLS configuration");
    let rls_learned = rls
        .update(&rls.initial_state(), &rls_rows[0])
        .expect("nonzero learned RLS state");
    match rls.process_batch_reference(rls_learned.clone(), &rls_rows, 2) {
        Err(RlsBatchError::BatchTooLarge { max, actual, state }) => {
            assert_eq!((max, actual), (2, 3));
            assert_eq!(state, rls_learned);
        }
        other => panic!("expected RLS oversized rejection, got {other:?}"),
    }

    let rls64_rows = vec![
        rls_sample64(&[1.0, 0.0], 1.0),
        rls_sample64(&[0.0, 1.0], -1.0),
    ];
    let rls64 = Rls64::new(2, 0.9, 1.0).expect("valid f64 RLS configuration");
    let rls64_state = rls64.initial_state();
    assert!(
        rls64
            .process_batch_reference(rls64_state.clone(), &rls64_rows, 2)
            .is_ok()
    );
    assert!(
        rls64
            .process_batch_reference(rls64_state.clone(), &rls64_rows, 1)
            .is_err()
    );
    assert!(
        rls.process_batch_reference(rls.initial_state(), &[], 0)
            .is_ok()
    );
    assert!(
        model32
            .update_batch(model32.initial_state(), &[], 0)
            .is_ok()
    );
}

#[test]
fn strict_fail_stop_prefix_and_continuation_are_exact() {
    let rows = vec![
        sample32(&[1.0, 0.0], 1.0),
        sample32(&[0.0, 1.0], -1.0),
        sample32(&[1.0, 1.0], f32::NAN),
        sample32(&[0.5, -0.5], 0.25),
        sample32(&[0.25, 0.75], -0.5),
    ];
    let model = Sgd::new(2, 0.125).expect("valid SGD configuration");
    let mut expected_prefix = model.initial_state();
    for row in &rows[..2] {
        expected_prefix = model.update(&expected_prefix, row).unwrap();
    }
    match model.update_batch(model.initial_state(), &rows, rows.len()) {
        Err(SgdUpdateBatchError::Transition {
            failed_index,
            committed,
            state,
            error,
        }) => {
            assert_eq!(failed_index, 2);
            assert_eq!(committed, 2);
            assert_eq!(state, expected_prefix);
            assert_eq!(error, SgdError::NonFiniteInput);
        }
        other => panic!("expected prefix failure, got {other:?}"),
    }

    // A corrected retry starts at the retained prefix and commits exactly once.
    let mut corrected = rows.clone();
    corrected[2] = sample32(&[1.0, 1.0], 0.0);
    let prefix = model.update(&model.initial_state(), &rows[0]).unwrap();
    let prefix = model.update(&prefix, &rows[1]).unwrap();
    let retried = model
        .update_batch(prefix.clone(), &corrected[2..], 3)
        .unwrap();
    let uninterrupted = corrected[..]
        .iter()
        .try_fold(model.initial_state(), |state, row| {
            model.update(&state, row)
        })
        .unwrap();
    assert_eq!(retried, uninterrupted);

    // The caller loop observes the failed item and does not attempt its suffix.
    let mut attempted = 0;
    let mut state = model.initial_state();
    let mut failed = None;
    for (index, row) in rows.iter().enumerate() {
        attempted += 1;
        match model.update(&state, row) {
            Ok(next) => state = next,
            Err(error) => {
                failed = Some((index, error));
                break;
            }
        }
    }
    assert_eq!(attempted, 3);
    assert_eq!(failed, Some((2, SgdError::NonFiniteInput)));
    assert_eq!(state, expected_prefix);

    // RLS exposes the same prefix boundary through its public bounded API.
    let rls_rows = vec![
        rls_sample32(&[1.0, 0.0], 1.0),
        rls_sample32(&[0.0, 1.0], -1.0),
        rls_sample32(&[1.0, 1.0], f32::NAN),
        rls_sample32(&[0.5, -0.5], 0.25),
    ];
    let rls = Rls::new(2, 0.9, 1.0).expect("valid RLS configuration");
    let mut rls_prefix = rls.initial_state();
    for row in &rls_rows[..2] {
        rls_prefix = rls.update(&rls_prefix, row).unwrap();
    }
    match rls.process_batch_reference(rls.initial_state(), &rls_rows, 4) {
        Err(RlsBatchError::Transition {
            failed_index,
            state,
            error,
        }) => {
            assert_eq!(failed_index, 2);
            assert_eq!(state, rls_prefix);
            assert_eq!(error, RlsError::NonFiniteInput);
        }
        other => panic!("expected RLS prefix failure, got {other:?}"),
    }
}

#[test]
fn sgd_a_c_bounds_failures_and_corrected_retries_cover_each_position() {
    let rows32 = [
        sample32(&[1.0, 0.0], 1.0),
        sample32(&[0.0, 1.0], -1.0),
        sample32(&[1.0, 1.0], 0.5),
    ];
    let model32 = Sgd::new(2, 0.125).unwrap();
    let initial32 = model32.initial_state();
    let singleton32 = model32
        .update_batch(initial32.clone(), &rows32[..1], 1)
        .unwrap();
    assert_eq!(singleton32, model32.update(&initial32, &rows32[0]).unwrap());
    assert_eq!(
        model32.update_batch(initial32.clone(), &[], 0).unwrap(),
        initial32
    );
    let exact32 = model32
        .update_batch(initial32.clone(), &rows32, rows32.len())
        .unwrap();
    let reference32 = rows32
        .iter()
        .try_fold(initial32.clone(), |state, row| model32.update(&state, row))
        .unwrap();
    assert_eq!(exact32, reference32);
    let learned32 = model32.update(&initial32, &rows32[0]).unwrap();
    match model32.update_batch(learned32.clone(), &rows32[..1], 0) {
        Err(SgdUpdateBatchError::TooLarge {
            max: 0,
            actual: 1,
            state,
        }) => assert_eq!(state, learned32),
        other => panic!("zero limit must reject nonempty SGD C input: {other:?}"),
    }

    for failed_at in 0..rows32.len() {
        let mut bad = rows32.clone();
        bad[failed_at] = sample32(&[1.0, 1.0], f32::NAN);
        let mut attempted = Vec::new();
        let failure =
            replay_sgd32_observed(&model32, initial32.clone(), &bad, bad.len(), |index| {
                attempted.push(index)
            })
            .unwrap_err();
        let expected_prefix = rows32[..failed_at]
            .iter()
            .try_fold(initial32.clone(), |state, row| model32.update(&state, row))
            .unwrap();
        match failure {
            BatchFailure::Transition {
                failed_index,
                committed,
                state,
                error,
            } => {
                assert_eq!(failed_index, failed_at);
                assert_eq!(committed, failed_at);
                assert_eq!(state, expected_prefix);
                assert_eq!(error, SgdError::NonFiniteInput);
            }
            other => panic!("expected transition failure: {other:?}"),
        }
        assert_eq!(attempted, (0..=failed_at).collect::<Vec<_>>());

        let mut corrected = bad;
        corrected[failed_at] = rows32[failed_at].clone();
        let resumed = model32
            .update_batch(
                expected_prefix,
                &corrected[failed_at..],
                corrected.len() - failed_at,
            )
            .unwrap();
        let uninterrupted = corrected
            .iter()
            .try_fold(initial32.clone(), |state, row| model32.update(&state, row))
            .unwrap();
        assert_eq!(resumed, uninterrupted);
    }

    let rows64 = [
        sample64(&[1.0, 0.0], 1.0),
        sample64(&[0.0, 1.0], -1.0),
        sample64(&[1.0, 1.0], 0.5),
    ];
    let model64 = Sgd64::new(2, 0.125).unwrap();
    let initial64 = model64.initial_state();
    let singleton64 = model64
        .update_batch(initial64.clone(), &rows64[..1], 1)
        .unwrap();
    assert_eq!(singleton64, model64.update(&initial64, &rows64[0]).unwrap());
    assert_eq!(
        model64.update_batch(initial64.clone(), &[], 0).unwrap(),
        initial64
    );
    let exact64 = model64
        .update_batch(initial64.clone(), &rows64, rows64.len())
        .unwrap();
    let reference64 = rows64
        .iter()
        .try_fold(initial64.clone(), |state, row| model64.update(&state, row))
        .unwrap();
    assert_eq!(exact64, reference64);
    let learned64 = model64.update(&initial64, &rows64[0]).unwrap();
    match model64.update_batch(learned64.clone(), &rows64[..1], 0) {
        Err(SgdUpdateBatchError64::TooLarge {
            max: 0,
            actual: 1,
            state,
        }) => assert_eq!(state, learned64),
        other => panic!("zero limit must reject nonempty SGD A input: {other:?}"),
    }

    for failed_at in 0..rows64.len() {
        let mut bad = rows64.clone();
        bad[failed_at] = sample64(&[1.0, 1.0], f64::NAN);
        let mut attempted = Vec::new();
        let failure =
            replay_sgd64_observed(&model64, initial64.clone(), &bad, bad.len(), |index| {
                attempted.push(index)
            })
            .unwrap_err();
        let expected_prefix = rows64[..failed_at]
            .iter()
            .try_fold(initial64.clone(), |state, row| model64.update(&state, row))
            .unwrap();
        match failure {
            BatchFailure::Transition {
                failed_index,
                committed,
                state,
                error,
            } => {
                assert_eq!(failed_index, failed_at);
                assert_eq!(committed, failed_at);
                assert_eq!(state, expected_prefix);
                assert_eq!(
                    error,
                    seqvex::models::linear::sgd::SgdError64::NonFiniteInput
                );
            }
            other => panic!("expected transition failure: {other:?}"),
        }
        assert_eq!(attempted, (0..=failed_at).collect::<Vec<_>>());

        let mut corrected = bad;
        corrected[failed_at] = rows64[failed_at].clone();
        let resumed = model64
            .update_batch(
                expected_prefix,
                &corrected[failed_at..],
                corrected.len() - failed_at,
            )
            .unwrap();
        let uninterrupted = corrected
            .iter()
            .try_fold(initial64.clone(), |state, row| model64.update(&state, row))
            .unwrap();
        assert_eq!(resumed, uninterrupted);
    }

    let mut chunked = model64.initial_state();
    let mut reference = model64.initial_state();
    for chunk in rows64.chunks(2) {
        chunked = model64.update_batch(chunked, chunk, chunk.len()).unwrap();
        reference = chunk
            .iter()
            .try_fold(reference, |state, row| model64.update(&state, row))
            .unwrap();
        let feature = chunk.last().unwrap().value().features.clone();
        assert_f64_bits(
            model64.predict(&chunked, &feature).unwrap(),
            model64.predict(&reference, &feature).unwrap(),
        );
    }
}

#[test]
fn uneven_chunks_preserve_learning_states_and_boundary_predictions() {
    let rows = [
        sample32(&[0.25, -0.5], 0.5),
        sample32(&[0.5, 0.25], -0.25),
        sample32(&[-0.5, 0.75], 1.0),
        sample32(&[0.75, -0.25], -0.5),
        sample32(&[0.0, 0.5], 0.0),
    ];
    let model = Sgd::new(2, 0.05).expect("valid SGD configuration");
    let continuous = rows
        .iter()
        .try_fold(model.initial_state(), |state, row| {
            model.update(&state, row)
        })
        .unwrap();
    let mut chunked = model.initial_state();
    let mut reference = model.initial_state();
    let mut boundaries = Vec::new();
    let mut expected_boundaries = Vec::new();
    for chunk in rows.chunks(2) {
        chunked = model.update_batch(chunked, chunk, chunk.len()).unwrap();
        for row in chunk {
            reference = model.update(&reference, row).unwrap();
        }
        boundaries.push(model.predict(&chunked, &chunk[0].value().features).unwrap());
        expected_boundaries.push(
            model
                .predict(&reference, &chunk[0].value().features)
                .unwrap(),
        );
    }
    chunked = model.update_batch(chunked, &[], 0).unwrap();
    assert_eq!(chunked, continuous);
    assert_eq!(boundaries, expected_boundaries);
    assert_eq!(boundaries.len(), 3);

    let rls_rows = vec![
        rls_sample32(&[0.25, -0.5], 0.5),
        rls_sample32(&[0.5, 0.25], -0.25),
        rls_sample32(&[-0.5, 0.75], 1.0),
        rls_sample32(&[0.75, -0.25], -0.5),
    ];
    let rls = Rls::new(2, 0.9, 1.0).expect("valid RLS configuration");
    let rls_continuous = rls
        .process_batch_reference(rls.initial_state(), &rls_rows, rls_rows.len())
        .unwrap();
    let mut rls_chunked = rls.initial_state();
    for chunk in rls_rows.chunks(3) {
        rls_chunked = rls
            .process_batch_reference(rls_chunked, chunk, chunk.len())
            .unwrap();
    }
    rls_chunked = rls.process_batch_reference(rls_chunked, &[], 0).unwrap();
    assert_eq!(rls_chunked, rls_continuous);
}

#[test]
fn calculation_collectors_are_all_or_error_and_distinguish_bounds() {
    let rows = vec![
        Observation::new(Vector::from_slice(&[1.0])),
        Observation::new(Vector::from_slice(&[2.0])),
        Observation::new(Vector::from_slice(&[3.0])),
    ];
    let empty: Vec<Observation<Vector>> = Vec::new();
    assert_eq!(collect32(0, &empty, |_| Ok::<f32, ()>(0.0)), Ok(vec![]));
    assert_eq!(
        collect32(2, &rows, |_| Ok::<f32, ()>(0.0)),
        Err(PredictionFailure::TooLarge { max: 2, actual: 3 })
    );
    for failed_at in 0..rows.len() {
        let mut calls = Vec::new();
        let failed = collect32(3, &rows, |row| {
            calls.push(row.as_slice()[0] as usize - 1);
            if row.as_slice()[0] as usize - 1 == failed_at {
                Err(7u8)
            } else {
                Ok(row.as_slice()[0])
            }
        });
        assert_eq!(
            failed,
            Err(PredictionFailure::Failed {
                index: failed_at,
                error: 7
            })
        );
        assert_eq!(calls, (0..=failed_at).collect::<Vec<_>>());
    }

    let rows64 = vec![
        Observation::new(Vector64::from_slice(&[1.0])),
        Observation::new(Vector64::from_slice(&[2.0])),
    ];
    assert_eq!(
        collect64(2, &rows64, |row| Ok::<f64, ()>(row.as_slice()[0])),
        Ok(vec![1.0, 2.0])
    );
    assert!(collect64(1, &rows64, |_| Ok::<f64, ()>(0.0)).is_err());
    let rows64_failure = vec![
        Observation::new(Vector64::from_slice(&[1.0])),
        Observation::new(Vector64::from_slice(&[2.0])),
        Observation::new(Vector64::from_slice(&[3.0])),
    ];
    for failed_at in 0..rows64_failure.len() {
        let mut calls = Vec::new();
        let failed = collect64(3, &rows64_failure, |row| {
            let index = row.as_slice()[0] as usize - 1;
            calls.push(index);
            if index == failed_at {
                Err(9u8)
            } else {
                Ok(row.as_slice()[0])
            }
        });
        assert_eq!(
            failed,
            Err(PredictionFailure::Failed {
                index: failed_at,
                error: 9
            })
        );
        assert_eq!(calls, (0..=failed_at).collect::<Vec<_>>());
    }

    let model = Sgd::new(1, 0.1).unwrap();
    let state = model.initial_state();
    let streaming_rows = [
        Observation::new(Vector::from_slice(&[1.0])),
        Observation::new(Vector::from_slice(&[f32::NAN])),
        Observation::new(Vector::from_slice(&[3.0])),
    ];
    let mut recorded = Vec::new();
    let mut attempted = Vec::new();
    for (index, row) in streaming_rows.iter().enumerate() {
        attempted.push(index);
        match model.predict(&state, row.value()) {
            Ok(output) => recorded.push(PredictionRecord {
                observation_index: index,
                model_identity: 7,
                output,
            }),
            Err(_) => break,
        }
    }
    assert_eq!(attempted, vec![0, 1]);
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].observation_index, 0);
    assert_eq!(recorded[0].model_identity, 7);
}
