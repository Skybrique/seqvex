#[path = "support/linear_execution.rs"]
mod support;

use support::{
    AcceptanceDecision, ActiveOls, BatchFailure, FeatureSnapshot, FreshnessCheck, FreshnessPolicy,
    FreshnessPolicyError, InstallError, ObservingSink, OlsCandidate, OlsProgress, OrderedSource,
    PendingBudget, PredictionFailure, PredictionRecord, Progress, PublicationAction,
    PublicationOutcome, RetainedBatch, ServingRecord, TemporalFailure, WorkflowMode,
    apply_pending_prefix, check_freshness, collect32, collect64, publish_retained,
    recover_missing_prediction, replay_sgd32, replay_sgd32_observed, replay_sgd64,
    replay_sgd64_observed, try_install_ols,
};

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Matrix, Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, FitConfig32, FitConfig64, LinearRegression,
    LinearRegression64, UnderflowPolicy, fit_f32, fit_f64,
};
use seqvex::models::linear::rls::{Rls, Rls64, RlsBatchError, RlsError, RlsSample, RlsSample64};
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdError, SgdSample, SgdSample64, SgdState64};

fn sample32(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn rls_sample32(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn rls_sample64(features: &[f64], target: f64) -> Observation<RlsSample64> {
    Observation::new(RlsSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn assert_f32_bits(left: f32, right: f32) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left:?} != {right:?}");
}

fn assert_f64_bits(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left:?} != {right:?}");
}

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
    match replay_sgd32(&model32, learned32, &rows32, 2) {
        Err(BatchFailure::TooLarge { max, actual, state }) => {
            assert_eq!((max, actual), (2, 3));
            assert_eq!(state, supplied32);
            let continued = replay_sgd32(&model32, state.clone(), &rows32[..2], 2).unwrap();
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
    match replay_sgd64(&model64, learned64, &rows64, 2) {
        Err(BatchFailure::TooLarge { state, .. }) => {
            assert_eq!(state, supplied64);
            let continued = replay_sgd64(&model64, state.clone(), &rows64[..2], 2).unwrap();
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
    match replay_sgd32(&model32, model32.initial_state(), &invalid, 1) {
        Err(BatchFailure::TooLarge { state, .. }) => {
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
    assert!(replay_sgd32(&model32, model32.initial_state(), &[], 0).is_ok());
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
    match replay_sgd32(&model, model.initial_state(), &rows, rows.len()) {
        Err(BatchFailure::Transition {
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
    let retried = replay_sgd32(&model, prefix.clone(), &corrected[2..], 3).unwrap();
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
    let singleton32 = replay_sgd32(&model32, initial32.clone(), &rows32[..1], 1).unwrap();
    assert_eq!(singleton32, model32.update(&initial32, &rows32[0]).unwrap());
    assert_eq!(
        replay_sgd32(&model32, initial32.clone(), &[], 0).unwrap(),
        initial32
    );
    let exact32 = replay_sgd32(&model32, initial32.clone(), &rows32, rows32.len()).unwrap();
    let reference32 = rows32
        .iter()
        .try_fold(initial32.clone(), |state, row| model32.update(&state, row))
        .unwrap();
    assert_eq!(exact32, reference32);
    let learned32 = model32.update(&initial32, &rows32[0]).unwrap();
    match replay_sgd32(&model32, learned32.clone(), &rows32[..1], 0) {
        Err(BatchFailure::TooLarge {
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
        let resumed = replay_sgd32(
            &model32,
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
    let singleton64 = replay_sgd64(&model64, initial64.clone(), &rows64[..1], 1).unwrap();
    assert_eq!(singleton64, model64.update(&initial64, &rows64[0]).unwrap());
    assert_eq!(
        replay_sgd64(&model64, initial64.clone(), &[], 0).unwrap(),
        initial64
    );
    let exact64 = replay_sgd64(&model64, initial64.clone(), &rows64, rows64.len()).unwrap();
    let reference64 = rows64
        .iter()
        .try_fold(initial64.clone(), |state, row| model64.update(&state, row))
        .unwrap();
    assert_eq!(exact64, reference64);
    let learned64 = model64.update(&initial64, &rows64[0]).unwrap();
    match replay_sgd64(&model64, learned64.clone(), &rows64[..1], 0) {
        Err(BatchFailure::TooLarge {
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
        let resumed = replay_sgd64(
            &model64,
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
        chunked = replay_sgd64(&model64, chunked, chunk, chunk.len()).unwrap();
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
        chunked = replay_sgd32(&model, chunked, chunk, chunk.len()).unwrap();
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
    chunked = replay_sgd32(&model, chunked, &[], 0).unwrap();
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

#[test]
fn fixed_prediction_matches_single_models_for_a_b_and_c() {
    let sgd = Sgd::new(2, 0.1).expect("valid SGD configuration");
    let sgd_state = sgd
        .update(&sgd.initial_state(), &sample32(&[1.0, -0.5], 0.5))
        .unwrap();
    let rows32 = vec![
        Observation::new(Vector::from_slice(&[0.5, 0.25])),
        Observation::new(Vector::from_slice(&[-0.25, 0.75])),
    ];
    let direct32: Vec<f32> = rows32
        .iter()
        .map(|row| sgd.predict(&sgd_state, row.value()).unwrap())
        .collect();
    let grouped32 = collect32(2, &rows32, |row| sgd.predict(&sgd_state, row)).unwrap();
    for (left, right) in direct32.iter().zip(&grouped32) {
        assert_f32_bits(*left, *right);
    }

    let sgd64 = Sgd64::new(2, 0.1).expect("valid f64 SGD configuration");
    let sgd64_state = sgd64
        .update(&sgd64.initial_state(), &sample64(&[1.0, -0.5], 0.5))
        .unwrap();
    let serving = sgd64.to_serving_f32(&sgd64_state).unwrap();
    let b_direct: Vec<f32> = rows32
        .iter()
        .map(|row| serving.predict(row.value()).unwrap())
        .collect();
    let b_grouped = collect32(2, &rows32, |row| serving.predict(row)).unwrap();
    assert_eq!(b_direct, b_grouped);

    let rows64 = vec![
        Observation::new(Vector64::from_slice(&[0.5, 0.25])),
        Observation::new(Vector64::from_slice(&[-0.25, 0.75])),
    ];
    let direct_a: Vec<f64> = rows64
        .iter()
        .map(|row| sgd64.predict(&sgd64_state, row.value()).unwrap())
        .collect();
    let grouped_a = collect64(2, &rows64, |row| sgd64.predict(&sgd64_state, row)).unwrap();
    for (left, right) in direct_a.iter().zip(&grouped_a) {
        assert_f64_bits(*left, *right);
    }

    let rls = Rls64::new(2, 1.0, 1.0).expect("valid f64 RLS configuration");
    let rls_state = rls
        .update(&rls.initial_state(), &rls_sample64(&[1.0, -0.5], 0.5))
        .unwrap();
    let rls_rows = vec![
        Observation::new(Vector64::from_slice(&[0.5, 0.25])),
        Observation::new(Vector64::from_slice(&[-0.25, 0.75])),
    ];
    let rls_direct: Vec<f64> = rls_rows
        .iter()
        .map(|row| rls.predict_checked(&rls_state, row.value()).unwrap())
        .collect();
    let rls_grouped = collect64(2, &rls_rows, |row| rls.predict_checked(&rls_state, row)).unwrap();
    assert_eq!(rls_direct, rls_grouped);

    let rls32 = Rls::new(2, 0.9, 1.0).expect("valid C-profile RLS configuration");
    let rls32_state = rls32
        .update(&rls32.initial_state(), &rls_sample32(&[1.0, -0.5], 0.5))
        .unwrap();
    let rls32_rows = vec![
        Observation::new(Vector::from_slice(&[0.5, 0.25])),
        Observation::new(Vector::from_slice(&[-0.25, 0.75])),
    ];
    let rls32_single: Vec<f32> = rls32_rows
        .iter()
        .map(|row| rls32.predict_checked(&rls32_state, row.value()).unwrap())
        .collect();
    let rls32_grouped = collect32(2, &rls32_rows, |row| {
        rls32.predict_checked(&rls32_state, row)
    })
    .unwrap();
    assert_eq!(rls32_single, rls32_grouped);

    let rls_serving = rls.to_serving_f32(&rls_state).unwrap();
    let b_rows = vec![
        Observation::new(Vector::from_slice(&[0.5, 0.25])),
        Observation::new(Vector::from_slice(&[-0.25, 0.75])),
    ];
    let b_grouped = collect32(2, &b_rows, |row| rls_serving.predict_checked(row)).unwrap();
    let b_single: Vec<f32> = b_rows
        .iter()
        .map(|row| rls_serving.predict_checked(row.value()).unwrap())
        .collect();
    assert_eq!(b_grouped, b_single);
}

#[test]
fn historical_and_live_equal_time_control_preserves_phase_order() {
    let model = Rls64::new(1, 1.0, 1.0).expect("valid RLS configuration");
    let s0 = model.initial_state();
    let b0 = model.to_serving_f32(&s0).unwrap();
    let observation1 = rls_sample64(&[1.0], 1.0);
    let observation2 = rls_sample64(&[2.0], 0.0);

    let prediction1 = b0
        .predict_checked(&Vector::from_slice(&[1.0]))
        .expect("earlier prediction");
    let record1 = PredictionRecord {
        observation_index: 0,
        model_identity: 0,
        output: prediction1,
    };
    let snapshot1 = FeatureSnapshot {
        observation_index: 0,
        model_identity: 0,
        preprocessing_identity: 0,
        features: Vector64::from_slice(&[1.0]),
        preprocessing: (),
        selected_state: s0.clone(),
    };
    assert_eq!(record1.output.to_bits(), prediction1.to_bits());
    assert_eq!(snapshot1.features.as_slice(), &[1.0]);
    assert_eq!(snapshot1.model_identity, 0);

    // P(10): observation 2 prediction uses S0 before L(10).
    let prediction2_before = b0
        .predict_checked(&Vector::from_slice(&[2.0]))
        .expect("pre-label prediction");
    assert_eq!(prediction2_before.to_bits(), 0.0f32.to_bits());

    // L(10): observation 1's label commits; then B1 is installed.
    let s1 = model.update(&s0, &observation1).unwrap();
    let b1 = model.to_serving_f32(&s1).unwrap();
    let prediction_after = b1
        .predict_checked(&Vector::from_slice(&[1.0]))
        .expect("post-label prediction");
    assert_ne!(prediction_after.to_bits(), prediction1.to_bits());

    let mut progress = Progress::default();
    let mut serving = (
        b0.clone(),
        ServingRecord {
            identity: 0,
            served_through: 0,
        },
    );
    assert_eq!(
        check_freshness(None, &serving.1, progress.learning),
        Err(FreshnessPolicyError::MissingPolicy)
    );
    assert_eq!(
        check_freshness(
            Some(FreshnessPolicy::LatestEligible),
            &serving.1,
            progress.learning
        ),
        Ok(FreshnessCheck::Ready)
    );
    assert_eq!(
        serving
            .0
            .predict_checked(&Vector::from_slice(&[2.0]))
            .unwrap()
            .to_bits(),
        prediction2_before.to_bits()
    );

    let updated = model.update(&s0, &observation1).unwrap();
    let mut committed = s0.clone();
    progress
        .commit_update(&mut committed, Ok::<_, ()>(updated))
        .unwrap();
    assert_eq!(committed, s1);
    assert_eq!(progress.learning, 1);
    assert_eq!(
        check_freshness(
            Some(FreshnessPolicy::LatestEligible),
            &serving.1,
            progress.learning
        ),
        Ok(FreshnessCheck::Blocked {
            learned: 1,
            served: 0
        })
    );
    progress
        .install_serving(&mut serving, Ok::<_, ()>(b1.clone()), 1)
        .unwrap();
    assert_eq!(serving.1.served_through, progress.learning);
    assert_eq!(
        check_freshness(
            Some(FreshnessPolicy::LatestEligible),
            &serving.1,
            progress.learning
        ),
        Ok(FreshnessCheck::Ready)
    );
    assert!(matches!(
        WorkflowMode::HistoricalReplay,
        WorkflowMode::HistoricalReplay
    ));
    assert!(matches!(
        WorkflowMode::LiveAdaptive,
        WorkflowMode::LiveAdaptive
    ));
    assert!(matches!(
        FreshnessPolicy::LatestEligible,
        FreshnessPolicy::LatestEligible
    ));
    let frozen = FreshnessPolicy::FrozenSelected {
        serving_identity: 0,
        served_through: 0,
    };
    match frozen {
        FreshnessPolicy::FrozenSelected {
            serving_identity, ..
        } => assert_eq!(serving_identity, 0),
        FreshnessPolicy::LatestEligible => panic!("expected frozen policy"),
    }

    // HistoricalReplay may apply labels without per-row predictions.
    let historical_state = model
        .update(&model.update(&s0, &observation1).unwrap(), &observation2)
        .unwrap();
    assert_eq!(historical_state, model.update(&s1, &observation2).unwrap());
}

#[test]
fn delayed_labels_wait_for_the_original_ordered_prefix_across_a_c_profiles() {
    fn exercise<S, T, E, F>(initial: S, rows: [Observation<T>; 3], mut update: F)
    where
        S: Clone + PartialEq + std::fmt::Debug,
        T: Clone + std::fmt::Debug,
        F: FnMut(&S, &Observation<T>) -> Result<S, E>,
        E: std::fmt::Debug,
    {
        let mut pending = PendingBudget::new_at(2, 0);
        pending.admit_later(1, rows[1].clone()).unwrap();
        pending.admit_later(2, rows[2].clone()).unwrap();
        let mut state = initial.clone();
        let mut progress = Progress::default();
        assert!(
            apply_pending_prefix(&mut pending, &mut state, &mut progress, |state, row| {
                update(state, row)
            })
            .unwrap()
            .is_empty()
        );
        assert_eq!(state, initial);
        assert_eq!(progress.learning, 0);
        pending.admit_head(0, rows[0].clone()).unwrap();
        let intermediate =
            apply_pending_prefix(&mut pending, &mut state, &mut progress, |state, row| {
                update(state, row)
            })
            .unwrap();
        let expected0 = update(&initial, &rows[0]).unwrap();
        let expected1 = update(&expected0, &rows[1]).unwrap();
        let expected2 = update(&expected1, &rows[2]).unwrap();
        assert_eq!(intermediate, vec![expected0, expected1, expected2.clone()]);
        assert_eq!(state, expected2);
        assert_eq!(progress.learning, 3);
        assert_eq!(pending.pending_len(), 0);
    }

    let sgd = Sgd::new(1, 0.1).unwrap();
    exercise(
        sgd.initial_state(),
        [
            sample32(&[1.0], 1.0),
            sample32(&[2.0], -1.0),
            sample32(&[3.0], 0.5),
        ],
        |state, row| sgd.update(state, row),
    );
    let sgd64 = Sgd64::new(1, 0.1).unwrap();
    exercise(
        sgd64.initial_state(),
        [
            sample64(&[1.0], 1.0),
            sample64(&[2.0], -1.0),
            sample64(&[3.0], 0.5),
        ],
        |state, row| sgd64.update(state, row),
    );
    let rls = Rls::new(1, 0.9, 1.0).unwrap();
    exercise(
        rls.initial_state(),
        [
            rls_sample32(&[1.0], 1.0),
            rls_sample32(&[2.0], -1.0),
            rls_sample32(&[3.0], 0.5),
        ],
        |state, row| rls.update(state, row),
    );
    let rls64 = Rls64::new(1, 0.9, 1.0).unwrap();
    exercise(
        rls64.initial_state(),
        [
            rls_sample64(&[1.0], 1.0),
            rls_sample64(&[2.0], -1.0),
            rls_sample64(&[3.0], 0.5),
        ],
        |state, row| rls64.update(state, row),
    );
}

#[test]
fn missing_prediction_uses_only_retained_context_and_source_stops_at_gap() {
    let model = Rls64::new(1, 1.0, 1.0).unwrap();
    let initial = model.initial_state();
    let row = rls_sample64(&[1.0], 1.0);
    let context = FeatureSnapshot {
        observation_index: 0,
        model_identity: 17,
        preprocessing_identity: 4,
        features: Vector64::from_slice(&[1.0]),
        preprocessing: (),
        selected_state: initial.clone(),
    };
    let mut pending = PendingBudget::new_at(1, 0);
    pending.admit_head(0, row.clone()).unwrap();
    let mut learner = initial.clone();
    let mut predictions = 0;
    let missing = recover_missing_prediction(
        (0, row.value().target),
        context.model_identity,
        context.preprocessing_identity,
        None,
        None::<&FeatureSnapshot<Vector64, (), seqvex::models::linear::rls::RlsState64>>,
        |_, _, _| {
            predictions += 1;
            Ok::<f64, ()>(0.0)
        },
    );
    assert_eq!(
        missing,
        Err(support::PredictionResolutionFailure::MissingPrediction { item: (0, 1.0) })
    );
    assert_eq!(predictions, 0);
    assert_eq!(learner, initial);
    assert_eq!(pending.pending_len(), 1);

    let mut wrong_context = context.clone();
    wrong_context.model_identity = 18;
    assert!(matches!(
        recover_missing_prediction(
            (0, row.value().target),
            context.model_identity,
            context.preprocessing_identity,
            None,
            Some(&wrong_context),
            |_, _, _| {
                predictions += 1;
                Ok::<f64, ()>(0.0)
            }
        ),
        Err(support::PredictionResolutionFailure::MissingPrediction { .. })
    ));
    assert_eq!(predictions, 0);
    assert_eq!(pending.pending_len(), 1);

    let mut wrong_preprocessing = context.clone();
    wrong_preprocessing.preprocessing_identity = 5;
    assert!(matches!(
        recover_missing_prediction(
            (0, row.value().target),
            context.model_identity,
            context.preprocessing_identity,
            None,
            Some(&wrong_preprocessing),
            |_, _, _| {
                predictions += 1;
                Ok::<f64, ()>(0.0)
            }
        ),
        Err(support::PredictionResolutionFailure::MissingPrediction { .. })
    ));
    assert_eq!(predictions, 0);
    assert_eq!(pending.pending_len(), 1);

    let recovered = recover_missing_prediction(
        (0, row.value().target),
        context.model_identity,
        context.preprocessing_identity,
        None,
        Some(&context),
        |selected_state, features, preprocessing| {
            assert_eq!(preprocessing, &());
            predictions += 1;
            model.predict_checked(selected_state, features)
        },
    )
    .unwrap();
    assert_eq!(recovered.observation_index, 0);
    assert_eq!(recovered.model_identity, 17);
    assert_eq!(recovered.output.to_bits(), 0.0f64.to_bits());
    assert_eq!(predictions, 1);
    let newer = model.update(&initial, &rls_sample64(&[1.0], 1.0)).unwrap();
    assert_ne!(
        recovered.output.to_bits(),
        model
            .predict_checked(&newer, &Vector64::from_slice(&[1.0]))
            .unwrap()
            .to_bits()
    );
    let mut progress = Progress::default();
    apply_pending_prefix(&mut pending, &mut learner, &mut progress, |state, row| {
        model.update(state, row)
    })
    .unwrap();
    assert_eq!(progress.learning, 1);
    assert_eq!(pending.pending_len(), 0);

    let mut source = OrderedSource::new(vec![(6, "six"), (7, "seven")]);
    assert_eq!(
        source.next_expected(5),
        Err(TemporalFailure::SourceCannotDeliverGap)
    );
    assert_eq!(source.consumed(), 0);
    assert_eq!(source.remaining(), 2);
    assert_eq!(source.next_expected(6), Ok(Some((6, "six"))));
    assert_eq!(source.consumed(), 1);
    assert_eq!(source.remaining(), 1);
}

#[test]
fn future_perturbation_and_sequence_isolation_preserve_earlier_records() {
    let model = Rls64::new(1, 1.0, 1.0).expect("valid RLS configuration");
    let earlier = [rls_sample64(&[1.0], 1.0), rls_sample64(&[2.0], 2.0)];
    let future = rls_sample64(&[3.0], 3.0);
    let perturbed = rls_sample64(&[9.0], -9.0);

    let trace = |future: &Observation<RlsSample64>| {
        let mut state = model.initial_state();
        let mut progress = Progress::default();
        let serving0 = model.to_serving_f32(&state).unwrap();
        let mut active = (
            serving0,
            ServingRecord {
                identity: 0,
                served_through: 0,
            },
        );
        let p1 = active
            .0
            .predict_checked(&Vector::from_slice(&[1.0]))
            .unwrap();
        let record1 = PredictionRecord {
            observation_index: 0,
            model_identity: active.1.identity,
            output: p1,
        };
        let update0 = model.update(&state, &earlier[0]).unwrap();
        progress
            .commit_update(&mut state, Ok::<_, ()>(update0))
            .unwrap();
        let serving1 = model.to_serving_f32(&state).unwrap();
        progress
            .install_serving(&mut active, Ok::<_, ()>(serving1), 1)
            .unwrap();
        let p2 = active
            .0
            .predict_checked(&Vector::from_slice(&[2.0]))
            .unwrap();
        let record2 = PredictionRecord {
            observation_index: 1,
            model_identity: active.1.identity,
            output: p2,
        };
        let update1 = model.update(&state, &earlier[1]).unwrap();
        progress
            .commit_update(&mut state, Ok::<_, ()>(update1))
            .unwrap();
        let serving2 = model.to_serving_f32(&state).unwrap();
        progress
            .install_serving(&mut active, Ok::<_, ()>(serving2), 2)
            .unwrap();
        let p3 = active
            .0
            .predict_checked(&Vector::from_slice(&[
                future.value().features.as_slice()[0] as f32,
            ]))
            .unwrap();
        let future_update = model.update(&state, future).unwrap();
        progress
            .commit_update(&mut state, Ok::<_, ()>(future_update))
            .unwrap();
        ([record1, record2], p3, state, progress)
    };
    let (earlier_before, future_before, state_before, progress_before) = trace(&future);
    let (earlier_after, future_after, state_after, progress_after) = trace(&perturbed);
    assert_eq!(earlier_before, earlier_after);
    assert_ne!(future_before.to_bits(), future_after.to_bits());
    assert_ne!(state_before, state_after);
    assert_eq!(progress_before.learning, 3);
    assert_eq!(progress_after.learning, 3);

    let mut state_a = model.initial_state();
    let state_b = model.initial_state();
    let mut budget_a = PendingBudget::new_at(1, 1);
    let mut budget_b = PendingBudget::new_at(1, 2);
    budget_a.admit_head(1, "a").unwrap();
    budget_b.admit_head(2, "b").unwrap();
    let mut progress_a = Progress::default();
    let mut progress_b = Progress::default();
    let mut serving_a = (
        model.to_serving_f32(&state_a).unwrap(),
        ServingRecord {
            identity: 0,
            served_through: 0,
        },
    );
    let serving_b = (
        model.to_serving_f32(&state_b).unwrap(),
        ServingRecord {
            identity: 1,
            served_through: 0,
        },
    );
    let b_prediction_before = serving_b
        .0
        .predict_checked(&Vector::from_slice(&[1.0]))
        .unwrap();
    let retained_b = RetainedBatch {
        observation_indices: vec![12],
        values: vec![b_prediction_before],
        model_identity: serving_b.1.identity,
        calculated_count: 1,
    };
    let update = model.update(&state_a, &earlier[0]).unwrap();
    progress_a
        .commit_update(&mut state_a, Ok::<_, ()>(update))
        .unwrap();
    let serving_a_candidate = model.to_serving_f32(&state_a).unwrap();
    progress_a
        .install_serving(&mut serving_a, Ok::<_, ()>(serving_a_candidate), 2)
        .unwrap();
    let (selected_state_a, selected_a) = progress_a.select(&state_a, 2);
    let (selected_state_b, selected_b) = progress_b.select(&state_b, 3);
    assert_eq!(selected_state_a, state_a);
    assert_eq!(selected_state_b, state_b);
    assert_ne!(state_a, state_b);
    assert_eq!(progress_a.learning, 1);
    assert_eq!(state_b, model.initial_state());
    assert_eq!(progress_b.learning, 0);
    assert_eq!(
        serving_b
            .0
            .predict_checked(&Vector::from_slice(&[1.0]))
            .unwrap()
            .to_bits(),
        b_prediction_before.to_bits()
    );
    let retained_a = RetainedBatch {
        observation_indices: vec![11],
        values: vec![
            serving_a
                .0
                .predict_checked(&Vector::from_slice(&[1.0]))
                .unwrap(),
        ],
        model_identity: serving_a.1.identity,
        calculated_count: 1,
    };
    assert_eq!(retained_b.observation_indices, vec![12]);
    assert_eq!(
        retained_b.values[0].to_bits(),
        b_prediction_before.to_bits()
    );
    assert_eq!(retained_b.model_identity, serving_b.1.identity);
    assert_eq!(retained_a.model_identity, serving_a.1.identity);
    assert_eq!(budget_a.inspect_head(), Some((1, &"a")));
    assert_eq!(budget_b.inspect_head(), Some((2, &"b")));
    assert_ne!(selected_a, selected_b);
    assert_ne!(serving_a.1, serving_b.1);
    assert_eq!(progress_a.served, 1);
    assert_eq!(progress_b.served, 0);
    assert_eq!(progress_a.published, 0);
    assert_eq!(progress_b.published, 0);
}

#[test]
fn b_conversion_and_candidate_failure_never_repeat_learning() {
    let model64 = Sgd64::new(1, 0.1).expect("valid f64 SGD configuration");
    let huge = model64.initial_state();
    let learned = model64
        .update(&huge, &sample64(&[1.0], 1.0e40))
        .expect("successful A update");
    assert!(model64.to_serving_f32(&learned).is_err());
    assert!(model64.to_serving_f32(&learned).is_err());
    let after_retry = learned.clone();
    assert_eq!(learned, after_retry);

    let normal = SgdState64 {
        w: Vector64::from_slice(&[1.0]),
    };
    let active = model64.to_serving_f32(&normal).unwrap();
    let active_bits = active.weights().as_slice()[0].to_bits();
    let candidate = OlsCandidate {
        predictor: model64.to_serving_f32(&normal).unwrap(),
        preprocessing: "candidate",
        candidate_identity: 2,
        preprocessing_identity: 10,
    };
    let mut active_ols = ActiveOls {
        predictor: active,
        preprocessing: "active",
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    assert_eq!(
        try_install_ols(
            &mut active_ols,
            candidate,
            AcceptanceDecision::<&str>::Rejected,
            10
        ),
        Err(InstallError::Rejected)
    );
    assert_eq!(
        active_ols.predictor.weights().as_slice()[0].to_bits(),
        active_bits
    );
    assert_eq!(active_ols.installed_identity, 1);

    let rls64 = Rls64::new(1, 1.0, 1.0).expect("valid f64 RLS configuration");
    let huge_state = rls64.initial_state();
    let rls_learned = rls64
        .update(&huge_state, &rls_sample64(&[1.0], 1.0e40))
        .expect("successful RLS update");
    assert!(rls64.to_serving_f32(&rls_learned).is_err());
    let rls_after_retry = rls_learned.clone();
    assert_eq!(rls_learned, rls_after_retry);
}

#[test]
fn progress_identities_and_b_freshness_remain_separate() {
    let mut progress = Progress::default();
    assert_eq!(progress.learning, 0);
    assert_eq!(progress.selected, 0);
    assert_eq!(progress.served, 0);
    assert_eq!(progress.calculated, 0);
    assert_eq!(progress.published, 0);
    let model = Sgd64::new(1, 0.1).unwrap();
    let mut learner = model.initial_state();
    let failed = model.update(&learner, &sample64(&[1.0], f64::NAN));
    assert!(progress.commit_update(&mut learner, failed).is_err());
    assert_eq!(progress.learning, 0);
    assert_eq!(learner, model.initial_state());
    let update = model.update(&learner, &sample64(&[1.0], 1.0)).unwrap();
    progress
        .commit_update(
            &mut learner,
            Ok::<_, seqvex::models::linear::sgd::SgdError64>(update),
        )
        .unwrap();
    assert_eq!(progress.learning, 1);
    let (selected_state, selected) = progress.select(&learner, 7);
    assert_eq!(selected_state, learner);
    assert_eq!(selected.selected_through, 1);

    let old_serving = model.to_serving_f32(&model.initial_state()).unwrap();
    let mut active = (
        old_serving,
        ServingRecord {
            identity: 8,
            served_through: 0,
        },
    );
    let mut failed_learner = model.initial_state();
    let mut failed_progress = Progress::default();
    let failed_update = model
        .update(&failed_learner, &sample64(&[1.0], 1.0e40))
        .unwrap();
    failed_progress
        .commit_update(&mut failed_learner, Ok::<_, ()>(failed_update))
        .unwrap();
    assert_eq!(failed_progress.learning, 1);
    assert_eq!(failed_progress.served, 0);
    let overflow_learned = failed_learner;
    let conversion = model.to_serving_f32(&overflow_learned);
    assert!(conversion.is_err());
    let active_before = active.clone();
    assert!(
        failed_progress
            .install_serving(&mut active, conversion, 9)
            .is_err()
    );
    assert_eq!(active, active_before);
    assert_eq!(failed_progress.learning, 1);
    assert_eq!(failed_progress.served, 0);
    assert_eq!(
        check_freshness(
            Some(FreshnessPolicy::LatestEligible),
            &active.1,
            failed_progress.learning
        ),
        Ok(FreshnessCheck::Blocked {
            learned: 1,
            served: 0
        })
    );

    let rejected_candidate = model
        .to_serving_f32(&learner)
        .map_err(|_| "conversion")
        .and_then(|candidate| {
            if candidate
                .predict(&Vector::from_slice(&[1.0]))
                .unwrap()
                .is_finite()
            {
                Err("candidate rejected")
            } else {
                Ok(candidate)
            }
        });
    let active_before_rejection = active.clone();
    assert!(
        progress
            .install_serving(&mut active, rejected_candidate, 9)
            .is_err()
    );
    assert_eq!(active, active_before_rejection);
    assert_eq!(progress.served, 0);

    let converted = model.to_serving_f32(&learner).unwrap();
    progress
        .install_serving(&mut active, Ok::<_, ()>(converted), 9)
        .unwrap();
    assert_eq!(progress.learning, 1);
    assert_eq!(progress.served, 1);
    assert_eq!(active.1.identity, 9);
    assert_eq!(active.1.served_through, 1);

    let rows = [Observation::new(Vector::from_slice(&[1.0]))];
    let selected_serving = model.to_serving_f32(&selected_state).unwrap();
    let outputs = collect32(1, &rows, |row| selected_serving.predict(row)).unwrap();
    let batch = progress
        .retain_calculation(Ok::<_, ()>(RetainedBatch {
            observation_indices: vec![0],
            values: outputs,
            model_identity: selected.identity,
            calculated_count: 0,
        }))
        .unwrap();
    assert_eq!(batch.model_identity, selected.identity);
    assert_eq!(progress.calculated, 1);
    let mut retained = Some(batch);
    let mut sink = ObservingSink::<f32, ()>::new(vec![PublicationOutcome::FullyAcknowledged {
        acknowledged_count: 1,
    }]);
    assert_eq!(
        publish_retained(&mut retained, &mut progress, &mut sink),
        PublicationAction::Complete
    );
    assert_eq!(progress.published, 1);
    assert!(retained.is_none());

    assert_ne!(selected.identity, active.1.identity);
    assert_eq!(selected.selected_through, active.1.served_through);
}

#[test]
fn publication_outcomes_preserve_acknowledged_prefixes() {
    let mut progress = Progress::default();
    let batch = progress
        .retain_calculation(Ok::<_, ()>(RetainedBatch {
            observation_indices: vec![0, 1, 2, 3, 4],
            values: vec![10, 20, 30, 40, 50],
            model_identity: 7,
            calculated_count: 0,
        }))
        .unwrap();
    assert_eq!(batch.calculated_count, 5);
    let mut retained = Some(batch.clone());
    let mut sink = ObservingSink::<i32, ()>::new(vec![
        PublicationOutcome::NoWrite,
        PublicationOutcome::AcknowledgedPrefixUnwrittenSuffix {
            acknowledged_count: 2,
        },
        PublicationOutcome::FullyAcknowledged {
            acknowledged_count: 3,
        },
    ]);
    assert_eq!(
        publish_retained(&mut retained, &mut progress, &mut sink),
        PublicationAction::Retry
    );
    assert_eq!(retained, Some(batch.clone()));
    assert_eq!(progress.published, 0);
    assert_eq!(
        publish_retained(&mut retained, &mut progress, &mut sink),
        PublicationAction::Retry
    );
    assert_eq!(progress.published, 2);
    assert_eq!(
        retained.as_ref().unwrap().observation_indices,
        vec![2, 3, 4]
    );
    assert_eq!(retained.as_ref().unwrap().values, vec![30, 40, 50]);
    assert_eq!(retained.as_ref().unwrap().model_identity, 7);
    assert_eq!(
        publish_retained(&mut retained, &mut progress, &mut sink),
        PublicationAction::Complete
    );
    assert!(retained.is_none());
    assert_eq!(progress.published, 5);
    assert_eq!(sink.calls(), 3);
    assert_eq!(sink.submitted()[0], batch);
    assert_eq!(sink.submitted()[1], batch);
    assert_eq!(sink.submitted()[2].observation_indices, vec![2, 3, 4]);
    assert_eq!(sink.submitted()[2].values, vec![30, 40, 50]);
    assert_eq!(sink.submitted()[2].model_identity, 7);

    let mut uncertain_progress = Progress::default();
    let mut uncertain_batch = Some(RetainedBatch {
        observation_indices: vec![0, 1, 2, 3, 4],
        values: vec![10, 20, 30, 40, 50],
        model_identity: 7,
        calculated_count: 5,
    });
    let mut uncertain_sink = ObservingSink::<i32, ()>::new(vec![
        PublicationOutcome::AcknowledgedPrefixUncertainSuffix {
            acknowledged_count: 2,
        },
        PublicationOutcome::FullyAcknowledged {
            acknowledged_count: 3,
        },
    ]);
    assert_eq!(
        publish_retained(
            &mut uncertain_batch,
            &mut uncertain_progress,
            &mut uncertain_sink
        ),
        PublicationAction::Reconcile
    );
    assert_eq!(uncertain_progress.published, 2);
    assert_eq!(
        uncertain_batch.as_ref().unwrap().observation_indices,
        vec![2, 3, 4]
    );
    assert_eq!(uncertain_sink.calls(), 1);

    let mut stopped_progress = Progress::default();
    let mut stopped_batch = Some(RetainedBatch {
        observation_indices: vec![0, 1],
        values: vec![10, 20],
        model_identity: 8,
        calculated_count: 2,
    });
    let mut stopped_sink =
        ObservingSink::<i32, &str>::new(vec![PublicationOutcome::Unclassified {
            error: "unknown",
        }]);
    assert_eq!(
        publish_retained(&mut stopped_batch, &mut stopped_progress, &mut stopped_sink),
        PublicationAction::Stop("unknown")
    );
    assert_eq!(stopped_progress.published, 0);
    assert_eq!(
        stopped_batch.as_ref().unwrap().observation_indices,
        vec![0, 1]
    );
    assert_eq!(stopped_sink.calls(), 1);

    // Publication retries reuse calculation-time values and identity after learning advances.
    let model = Sgd::new(1, 0.1).unwrap();
    let rows = [
        Observation::new(Vector::from_slice(&[1.0])),
        Observation::new(Vector::from_slice(&[2.0])),
    ];
    let mut learner = model.initial_state();
    let mut tracked = Progress::default();
    let first_update = model.update(&learner, &sample32(&[1.0], 1.0)).unwrap();
    tracked
        .commit_update(&mut learner, Ok::<_, ()>(first_update))
        .unwrap();
    let mut calculations = 0;
    let values = collect32(2, &rows, |row| {
        calculations += 1;
        model.predict(&learner, row)
    })
    .unwrap();
    let retained_batch = tracked
        .retain_calculation(Ok::<_, ()>(RetainedBatch {
            observation_indices: vec![8, 9],
            values: values.clone(),
            model_identity: 22,
            calculated_count: 0,
        }))
        .unwrap();
    let outputs_before_retry = retained_batch.values.clone();
    let mut retained_batch = Some(retained_batch);
    let mut retry_sink = ObservingSink::<f32, ()>::new(vec![
        PublicationOutcome::NoWrite,
        PublicationOutcome::FullyAcknowledged {
            acknowledged_count: 2,
        },
    ]);
    assert_eq!(
        publish_retained(&mut retained_batch, &mut tracked, &mut retry_sink),
        PublicationAction::Retry
    );
    let second_update = model.update(&learner, &sample32(&[1.0], -1.0)).unwrap();
    tracked
        .commit_update(&mut learner, Ok::<_, ()>(second_update))
        .unwrap();
    assert_eq!(
        publish_retained(&mut retained_batch, &mut tracked, &mut retry_sink),
        PublicationAction::Complete
    );
    assert_eq!(tracked.learning, 2);
    assert_eq!(tracked.calculated, 2);
    assert_eq!(tracked.published, 2);
    assert_eq!(calculations, 2);
    assert_eq!(retry_sink.submitted()[0].values, outputs_before_retry);
    assert_eq!(retry_sink.submitted()[1].values, outputs_before_retry);
    assert!(
        retry_sink
            .submitted()
            .iter()
            .all(|submission| submission.model_identity == 22)
    );
}

#[test]
fn pending_budget_returns_refused_payloads_and_commits_only_on_success() {
    let mut failed_budget = PendingBudget::new_at(1, 0);
    let good0 = sample32(&[1.0], 1.0);
    let good1 = sample32(&[2.0], -1.0);
    let bad0 = sample32(&[1.0], f32::NAN);
    failed_budget.admit_later(1, good1.clone()).unwrap();
    failed_budget.admit_head(0, bad0).unwrap();
    let model = Sgd::new(1, 0.1).unwrap();
    let mut state = model.initial_state();
    let initial = state.clone();
    let mut progress = Progress::default();
    let failure = apply_pending_prefix(
        &mut failed_budget,
        &mut state,
        &mut progress,
        |state, row| model.update(state, row),
    )
    .unwrap_err();
    assert_eq!(failure.observation_index, 0);
    assert_eq!(failure.error, SgdError::NonFiniteInput);
    assert_eq!(state, initial);
    assert_eq!(progress.learning, 0);
    assert_eq!(failed_budget.pending_len(), 2);
    assert_eq!(
        failed_budget.inspect_head().map(|(index, _)| index),
        Some(0)
    );
    failed_budget.replace_head(0, good0.clone()).unwrap();
    let committed_states = apply_pending_prefix(
        &mut failed_budget,
        &mut state,
        &mut progress,
        |state, row| model.update(state, row),
    )
    .unwrap();
    let expected0 = model.update(&initial, &good0).unwrap();
    let expected1 = model.update(&expected0, &good1).unwrap();
    assert_eq!(committed_states, vec![expected0, expected1.clone()]);
    assert_eq!(state, expected1);
    assert_eq!(progress.learning, 2);
    assert_eq!(failed_budget.pending_len(), 0);

    let mut budget = PendingBudget::new_at(2, 4);
    budget.admit_later(5, "later-5").unwrap();
    budget.admit_later(6, "later-6").unwrap();
    let refused = budget.admit_later(7, "later-7").expect_err("ordinary full");
    assert_eq!(refused.failure, TemporalFailure::CapacityRefused);
    assert_eq!(refused.item, (7, "later-7"));
    budget.admit_head(4, "head-4").unwrap();
    assert_eq!(budget.pending_len(), 3);
    assert_eq!(budget.inspect_head(), Some((4, &"head-4")));
    let overflow = budget.admit_later(8, "overflow").unwrap_err();
    assert_eq!(overflow.item, (8, "overflow"));
    assert_eq!(budget.pending_len(), 3);

    // Commitment promotes each contiguous buffered label in original order.
    assert_eq!(budget.commit_head(), Some((4, "head-4")));
    assert_eq!(budget.pending_len(), 2);
    assert_eq!(budget.inspect_head(), Some((5, &"later-5")));
    assert_eq!(budget.commit_head(), Some((5, "later-5")));
    assert_eq!(budget.inspect_head(), Some((6, &"later-6")));
    assert_eq!(budget.commit_head(), Some((6, "later-6")));
    assert_eq!(budget.commit_head(), None);
    assert_eq!(budget.pending_len(), 0);

    let mut blocked_gap = PendingBudget::new_at(2, 4);
    blocked_gap.admit_later(6, "later-6").unwrap();
    blocked_gap.admit_head(4, "head-4").unwrap();
    assert_eq!(blocked_gap.commit_head(), Some((4, "head-4")));
    assert_eq!(blocked_gap.inspect_head(), None);
    assert_eq!(blocked_gap.pending_len(), 1);
    blocked_gap.admit_head(5, "gap-5").unwrap();
    assert_eq!(blocked_gap.commit_head(), Some((5, "gap-5")));
    assert_eq!(blocked_gap.inspect_head(), Some((6, &"later-6")));
    assert_eq!(blocked_gap.commit_head(), Some((6, "later-6")));

    let mut zero = PendingBudget::new_at(0, 0);
    assert_eq!(
        zero.admit_later(1, "x")
            .expect_err("no ordinary capacity")
            .item,
        (1, "x")
    );
    zero.admit_head(0, "head").unwrap();
    assert_eq!(zero.commit_head(), Some((0, "head")));
    assert_eq!(zero.pending_len(), 0);

    let mut discarded = PendingBudget::new_at(1, 1);
    discarded.admit_head(1, "discard").unwrap();
    assert_eq!(discarded.discard_head(), Some((1, "discard")));
    assert_eq!(discarded.pending_len(), 0);
}

#[test]
fn ols_refit_acceptance_pairing_and_discriminating_probe() {
    let features = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0])];
    let targets = [1.0f32, 4.0];
    let indices = [0, 1];
    let candidate = fit_f32(&features, &targets, &indices, &FitConfig32::default()).unwrap();
    let old = LinearRegression::new(Vector::from_slice(&[2.0]), 1.0).unwrap();
    let new = candidate.predictor();
    let probe = Vector::from_slice(&[1.0]);
    assert_eq!(
        old.predict_checked(&probe).unwrap().to_bits(),
        3.0f32.to_bits()
    );
    assert!((new.predict_checked(&probe).unwrap() - 1.0).abs() < 1e-5);

    let mut active = ActiveOls {
        predictor: old,
        preprocessing: "active",
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    let candidate_pair = OlsCandidate {
        predictor: new,
        preprocessing: "candidate",
        candidate_identity: 2,
        preprocessing_identity: 20,
    };
    assert_eq!(
        try_install_ols(
            &mut active,
            candidate_pair,
            AcceptanceDecision::<&str>::Accepted,
            10
        ),
        Err(InstallError::Pairing {
            expected: 10,
            actual: 20
        })
    );
    assert_eq!(active.installed_identity, 1);
    assert_eq!(active.preprocessing, "active");
    assert_eq!(active.preprocessing_identity, 10);
    assert_eq!(
        active.predictor.predict_checked(&probe).unwrap().to_bits(),
        3.0f32.to_bits()
    );

    let candidate_pair = OlsCandidate {
        predictor: candidate.predictor(),
        preprocessing: "candidate",
        candidate_identity: 2,
        preprocessing_identity: 10,
    };
    assert_eq!(
        try_install_ols(
            &mut active,
            candidate_pair,
            AcceptanceDecision::<&str>::Accepted,
            10
        ),
        Ok(2)
    );
    assert_eq!(active.installed_identity, 2);
    assert_eq!(active.preprocessing, "candidate");
    assert!((active.predictor.predict_checked(&probe).unwrap() - 1.0).abs() < 1e-5);
    let before_rejection = active.clone();

    let rejected = OlsCandidate {
        predictor: LinearRegression::new(Vector::from_slice(&[2.0]), 1.0).unwrap(),
        preprocessing: "rejected",
        candidate_identity: 3,
        preprocessing_identity: 10,
    };
    assert_eq!(
        try_install_ols(
            &mut active,
            rejected,
            AcceptanceDecision::<&str>::Rejected,
            10
        ),
        Err(InstallError::Rejected)
    );
    assert_eq!(active, before_rejection);
    let error_candidate = OlsCandidate {
        predictor: LinearRegression::new(Vector::from_slice(&[2.0]), 1.0).unwrap(),
        preprocessing: "error",
        candidate_identity: 4,
        preprocessing_identity: 10,
    };
    let before_error = active.clone();
    assert_eq!(
        try_install_ols(
            &mut active,
            error_candidate,
            AcceptanceDecision::<&str>::Error("reject"),
            10
        ),
        Err(InstallError::Decision("reject"))
    );
    assert_eq!(active, before_error);

    let features64 = [Vector64::from_slice(&[1.0]), Vector64::from_slice(&[2.0])];
    let targets64 = [1.0f64, 4.0];
    let candidate64 = fit_f64(&features64, &targets64, &indices, &FitConfig64::default()).unwrap();
    let reg64 = candidate64.predictor();
    assert!(
        (reg64
            .predict_checked(&Vector64::from_slice(&[1.0]))
            .unwrap()
            - 1.0)
            .abs()
            < 1e-12
    );
    let serving = candidate64.to_serving_f32(UnderflowPolicy::Reject).unwrap();
    assert!(
        (serving
            .predict_checked(&Vector::from_slice(&[1.0]))
            .unwrap()
            - 1.0)
            .abs()
            < 1e-5
    );
    let active_before_failures = active.clone();
    assert!(fit_f64(&[], &[], &[], &FitConfig64::default()).is_err());
    assert_eq!(active, active_before_failures);
    let overflow_features = [Vector64::from_slice(&[1.0]), Vector64::from_slice(&[2.0])];
    let overflow_targets = [1.0e40f64, 2.0e40f64];
    let overflow_candidate = fit_f64(
        &overflow_features,
        &overflow_targets,
        &indices,
        &FitConfig64::default(),
    )
    .expect("finite f64 candidate fits");
    assert!(
        overflow_candidate
            .to_serving_f32(UnderflowPolicy::Reject)
            .is_err()
    );
    assert_eq!(active, active_before_failures);
    let dimension_candidate =
        fit_f64(&features64, &targets64, &indices, &FitConfig64::default()).unwrap();
    assert!(
        dimension_candidate
            .predictor()
            .predict_checked(&Vector64::from_slice(&[1.0, 2.0]))
            .is_err()
    );
    assert_eq!(active, active_before_failures);

    let acceptance_features = [Vector64::from_slice(&[1.0]), Vector64::from_slice(&[2.0])];
    let acceptance_targets = [2.0f64, 4.0];
    let acceptance_candidate = fit_f64(
        &acceptance_features,
        &acceptance_targets,
        &indices,
        &FitConfig64::default(),
    )
    .unwrap()
    .predictor();
    let evaluated_prediction = acceptance_candidate
        .predict_checked(&Vector64::from_slice(&[3.0]))
        .unwrap();
    let evaluated_residual = (evaluated_prediction - 7.0).abs();
    assert!(evaluated_residual > 1e-12);
    let decision = if evaluated_residual <= 1e-12 {
        AcceptanceDecision::<&str>::Accepted
    } else {
        AcceptanceDecision::<&str>::Rejected
    };
    let mut active64 = ActiveOls {
        predictor: LinearRegression64::new(Vector64::from_slice(&[3.0]), -2.0).unwrap(),
        preprocessing: "active",
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    let rejected64 = OlsCandidate {
        predictor: acceptance_candidate,
        preprocessing: "candidate",
        candidate_identity: 2,
        preprocessing_identity: 10,
    };
    assert_eq!(
        try_install_ols(&mut active64, rejected64, decision, 10),
        Err(InstallError::Rejected)
    );
    assert_eq!(active64.preprocessing, "active");
    assert_eq!(active64.installed_identity, 1);
    assert_eq!(active64.preprocessing_identity, 10);
    assert_eq!(
        active64
            .predictor
            .predict_checked(&Vector64::from_slice(&[1.0]))
            .unwrap(),
        1.0
    );

    let active64 = ActiveOls {
        predictor: LinearRegression64::new(Vector64::from_slice(&[2.0]), 1.0).unwrap(),
        preprocessing: "active",
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    assert_eq!(
        active64
            .predictor
            .predict_checked(&Vector64::from_slice(&[1.0]))
            .unwrap(),
        3.0
    );
}

#[test]
fn ols_a_b_c_group_and_stream_records_use_installed_identities() {
    let features32 = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0])];
    let targets32 = [1.0f32, 4.0];
    let indices = [0, 1];
    let rows32 = [
        Observation::new(Vector::from_slice(&[1.0])),
        Observation::new(Vector::from_slice(&[2.0])),
        Observation::new(Vector::from_slice(&[3.0])),
    ];
    let candidate_c = fit_f32(&features32, &targets32, &indices, &FitConfig32::default()).unwrap();
    let old = LinearRegression::new(Vector::from_slice(&[2.0]), 1.0).unwrap();
    let mut ols_progress = OlsProgress {
        candidate_identity: None,
        installed_identity: 1,
        calculated: 0,
        published: 0,
    };
    let mut active_c = ActiveOls {
        predictor: old,
        preprocessing: (),
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    let candidate_pair = ols_progress
        .record_candidate(Ok::<_, ()>(OlsCandidate {
            predictor: candidate_c.predictor(),
            preprocessing: (),
            candidate_identity: 2,
            preprocessing_identity: 10,
        }))
        .unwrap();
    assert_eq!(ols_progress.candidate_identity, Some(2));
    let installed_c = ols_progress
        .record_install(try_install_ols(
            &mut active_c,
            candidate_pair,
            AcceptanceDecision::<&str>::Accepted,
            10,
        ))
        .unwrap();
    assert_eq!(ols_progress.installed_identity, 2);
    assert_eq!(ols_progress.candidate_identity, None);
    let group_c = active_c.predictor.predict_batch_checked(&rows32).unwrap();
    let singles_c: Vec<f32> = rows32
        .iter()
        .map(|row| active_c.predictor.predict_checked(row.value()).unwrap())
        .collect();
    let checked_c = CheckedLinearRegression::new(&active_c.predictor);
    let mut stream_c = StreamingExecutor::new(&checked_c, 0.0f32);
    let streamed_c: Vec<f32> = rows32
        .iter()
        .map(|row| *stream_c.process_one(row).unwrap())
        .collect();
    assert_eq!(installed_c, active_c.installed_identity);
    assert_eq!(group_c, singles_c);
    assert_eq!(group_c, streamed_c);
    let records_c: Vec<_> = group_c
        .iter()
        .enumerate()
        .map(|(index, output)| PredictionRecord {
            observation_index: index,
            model_identity: active_c.installed_identity,
            output: *output,
        })
        .collect();
    assert!(
        records_c
            .iter()
            .all(|record| record.model_identity == installed_c)
    );

    let features64 = [Vector64::from_slice(&[1.0]), Vector64::from_slice(&[2.0])];
    let targets64 = [1.0f64, 4.0];
    let rows64 = [
        Observation::new(Vector64::from_slice(&[1.0])),
        Observation::new(Vector64::from_slice(&[2.0])),
        Observation::new(Vector64::from_slice(&[3.0])),
    ];
    let candidate_a = fit_f64(&features64, &targets64, &indices, &FitConfig64::default()).unwrap();
    let mut active_a = ActiveOls {
        predictor: LinearRegression64::new(Vector64::from_slice(&[2.0]), 1.0).unwrap(),
        preprocessing: (),
        installed_identity: 10,
        preprocessing_identity: 10,
    };
    let installed_a = try_install_ols(
        &mut active_a,
        OlsCandidate {
            predictor: candidate_a.predictor(),
            preprocessing: (),
            candidate_identity: 11,
            preprocessing_identity: 10,
        },
        AcceptanceDecision::<&str>::Accepted,
        10,
    )
    .unwrap();
    let predictor_a = &active_a.predictor;
    let group_a = predictor_a.predict_batch_checked(&rows64).unwrap();
    let singles_a: Vec<f64> = rows64
        .iter()
        .map(|row| predictor_a.predict_checked(row.value()).unwrap())
        .collect();
    let checked_a = CheckedLinearRegression64::new(predictor_a);
    let mut stream_a = StreamingExecutor::new(&checked_a, 0.0f64);
    let streamed_a: Vec<f64> = rows64
        .iter()
        .map(|row| *stream_a.process_one(row).unwrap())
        .collect();
    assert_eq!(group_a, singles_a);
    assert_eq!(group_a, streamed_a);
    let identity_a = installed_a;
    let records_a: Vec<_> = group_a
        .iter()
        .enumerate()
        .map(|(index, output)| PredictionRecord {
            observation_index: index,
            model_identity: identity_a,
            output: *output,
        })
        .collect();
    assert!(
        records_a
            .iter()
            .all(|record| record.model_identity == identity_a)
    );

    let serving_b = candidate_a.to_serving_f32(UnderflowPolicy::Reject).unwrap();
    let mut active_b = ActiveOls {
        predictor: LinearRegression::new(Vector::from_slice(&[2.0]), 1.0).unwrap(),
        preprocessing: (),
        installed_identity: 11,
        preprocessing_identity: 10,
    };
    let installed_b = try_install_ols(
        &mut active_b,
        OlsCandidate {
            predictor: serving_b,
            preprocessing: (),
            candidate_identity: 12,
            preprocessing_identity: 10,
        },
        AcceptanceDecision::<&str>::Accepted,
        10,
    )
    .unwrap();
    let group_b = active_b.predictor.predict_batch_checked(&rows32).unwrap();
    let singles_b: Vec<f32> = rows32
        .iter()
        .map(|row| active_b.predictor.predict_checked(row.value()).unwrap())
        .collect();
    let checked_b = CheckedLinearRegression::new(&active_b.predictor);
    let mut stream_b = StreamingExecutor::new(&checked_b, 0.0f32);
    let streamed_b: Vec<f32> = rows32
        .iter()
        .map(|row| *stream_b.process_one(row).unwrap())
        .collect();
    assert_eq!(group_b, singles_b);
    assert_eq!(group_b, streamed_b);
    let identity_b = installed_b;
    let records_b: Vec<_> = group_b
        .iter()
        .enumerate()
        .map(|(index, output)| PredictionRecord {
            observation_index: index,
            model_identity: identity_b,
            output: *output,
        })
        .collect();
    assert!(
        records_b
            .iter()
            .all(|record| record.model_identity == identity_b)
    );
}

#[test]
fn typed_failures_remain_explicit() {
    let sgd = Sgd::new(1, 0.1).unwrap();
    assert_eq!(
        sgd.predict(&sgd.initial_state(), &Vector::from_slice(&[1.0, 2.0])),
        Err(SgdError::DimensionMismatch {
            expected: 1,
            actual: 2
        })
    );
    assert_eq!(
        sgd.predict(&sgd.initial_state(), &Vector::from_slice(&[f32::NAN])),
        Err(SgdError::NonFiniteInput)
    );
    let rls = Rls::new(1, 1.0, 1.0).unwrap();
    assert_eq!(
        rls.predict_checked(&rls.initial_state(), &Vector::from_slice(&[f32::NAN])),
        Err(RlsError::NonFiniteInput)
    );
    let ols = LinearRegression::new(Vector::from_slice(&[1.0]), 0.0).unwrap();
    assert!(
        ols.predict_checked(&Vector::from_slice(&[f32::NAN]))
            .is_err()
    );
    let matrix = Matrix::from_fn(1, 1, |_, _| 1.0);
    assert_eq!(matrix.rows(), 1);
}

#[test]
fn progress_and_publication_counts_do_not_use_observation_indices() {
    let progress = Progress {
        learning: 3,
        calculated: 3,
        published: 2,
        ..Progress::default()
    };
    assert_eq!(progress.published + 1, 3);
    assert_ne!(progress.published, 3);
    let _ = OlsProgress {
        candidate_identity: Some(1),
        installed_identity: 1,
        calculated: 1,
        published: 0,
    };
}
