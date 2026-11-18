//! RLS ordered/bounded fold tests: streaming, streaming publication, micro-batch,
//! bounded reference batching, ordered chunk continuity, shared-model independence,
//! and precision-profile continuation.

use super::support::*;
use super::*;
use std::error::Error as StdError;

#[test]
fn process_stream_matches_repeated_update_bitwise() {
    let model = Rls::new(2, 0.5, 4.0).unwrap();
    let observations = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[0.0, 1.0], -2.0),
        sample(&[0.5, 0.5], 3.0),
    ];

    let mut direct = model.initial_state();
    for observation in &observations {
        direct = model.update(&direct, observation).unwrap();
    }

    let streamed = process_stream(&model, model.initial_state(), observations).unwrap();
    assert_state_bitwise_eq(&direct, &streamed);
}

#[test]
fn process_stream_continues_after_a_failure() {
    let model = Rls::new(2, 1.0, 4.0).unwrap();
    let observations = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[f32::NAN, 0.0], 1.0),
        sample(&[0.0, 1.0], 2.0),
    ];
    let mut failures = 0;

    let summary = process_stream_continuing(
        &model,
        model.initial_state(),
        observations,
        |position, error| {
            assert_eq!(position, 1);
            assert_eq!(*error, RlsError::NonFiniteInput);
            failures += 1;
        },
    )
    .unwrap();

    assert_eq!(failures, 1);
    let final_state = summary.state;
    let mut expected = model.initial_state();
    expected = model.update(&expected, &sample(&[1.0, 0.0], 1.0)).unwrap();
    expected = model.update(&expected, &sample(&[0.0, 1.0], 2.0)).unwrap();
    assert_state_bitwise_eq(&final_state, &expected);
}

#[test]
fn reset_starts_a_new_sequence() {
    let streaming_model = Rls::new(2, 0.9, 2.0).unwrap();
    let reference = streaming_model.clone();
    let initial = streaming_model.initial_state();
    let mut executor = StreamingExecutor::new(&streaming_model, initial.clone());

    executor.process_one(&sample(&[1.0, 2.0], 1.0)).unwrap();
    executor.reset(initial.clone());
    assert_state_bitwise_eq(executor.state(), &initial);

    executor.process_one(&sample(&[0.5, 0.5], 2.0)).unwrap();
    let expected = reference
        .update(&reference.initial_state(), &sample(&[0.5, 0.5], 2.0))
        .unwrap();
    assert_state_bitwise_eq(executor.state(), &expected);
}

#[test]
fn bounded_fold_matches_repeated_process_one_bitwise() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let batch = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[0.0, 1.0], -2.0),
        sample(&[0.5, 0.5], 3.0),
    ];

    let folded = process_batch(&model, model.initial_state(), batch.clone()).unwrap();

    let mut manual = model.initial_state();
    for observation in &batch {
        manual = process_one(&model, &manual, observation).unwrap();
    }
    assert_state_bitwise_eq(&folded, &manual);
}

#[test]
fn bounded_fold_stops_on_first_failure_and_keeps_last_valid_state() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let first = sample(&[1.0, 0.0], 1.0);
    let bad = sample(&[f32::NAN, 0.0], 1.0);
    let third = sample(&[0.0, 1.0], 2.0);

    let failure = process_batch(
        &model,
        model.initial_state(),
        vec![first.clone(), bad, third],
    )
    .unwrap_err();

    assert_eq!(failure.cause, FailureCause::Model(RlsError::NonFiniteInput));
    let expected = model.update(&model.initial_state(), &first).unwrap();
    assert_state_bitwise_eq(&failure.state, &expected);
}

// --- bounded reference batch (Rls::process_batch_reference) -------------------
//
// The RLS-local bounded reference batching contract: an ordered, bound-first,
// state-only fold over direct `Rls::update` calls. It is not the foundation
// `process_batch` (which stays unbounded); equivalence with the foundation fold
// is asserted bitwise rather than by delegating to it.

#[test]
fn bounded_reference_matches_reference_paths_bitwise() {
    for lambda in [1.0_f32, 0.9] {
        let model = Rls::new(2, lambda, 5.0).unwrap();
        let batch = vec![
            sample(&[1.0, 0.0], 1.0),
            sample(&[0.0, 1.0], -2.0),
            sample(&[0.5, 0.5], 3.0),
            sample(&[-0.25, 0.75], 0.5),
        ];

        let mut repeated = model.initial_state();
        for observation in &batch {
            repeated = model.update(&repeated, observation).unwrap();
        }

        let mut executor = StreamingExecutor::new(&model, model.initial_state());
        for observation in &batch {
            executor.process_one(observation).unwrap();
        }
        assert_state_bitwise_eq(executor.state(), &repeated);

        let folded = process_batch(&model, model.initial_state(), batch.iter().cloned()).unwrap();
        assert_state_bitwise_eq(&folded, &repeated);

        let batched = model
            .process_batch_reference(model.initial_state(), &batch, batch.len())
            .unwrap();
        assert_state_bitwise_eq(&batched, &repeated);
    }
}

#[test]
fn bounded_reference_matches_independent_oracle() {
    let model = Rls::new(3, 0.95, 1.0e3).unwrap();
    let xs: Vec<Vec<f32>> = (0..9)
        .map(|k| {
            vec![
                (k as f32 * 0.31).sin(),
                (k as f32 * 0.17).cos(),
                (k as f32 * 0.05).sin(),
            ]
        })
        .collect();
    let ys: Vec<f32> = (0..9).map(|k| (k as f32 * 0.23).cos()).collect();
    let batch: Vec<Observation<RlsSample>> =
        xs.iter().zip(&ys).map(|(x, &y)| sample(x, y)).collect();

    let batched = model
        .process_batch_reference(model.initial_state(), &batch, batch.len())
        .unwrap();

    // Independent scalar recurrence, bitwise.
    let mut reference = ref_initial(3, 1.0e3);
    for (x, &y) in xs.iter().zip(&ys) {
        reference = ref_step(&reference, x, y, 0.95);
    }
    assert_state_bitwise_eq(&batched, &to_state(&reference));

    // Independent literal-objective f64 oracle, tolerance-based.
    assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
}

#[test]
fn bounded_reference_permutation_semantics() {
    let d = 2;
    let n = 30;
    let delta = 1.0e3_f32;
    let (xs, ys) = sinusoidal_data(d, n);
    let permutation: Vec<usize> = (0..n).map(|index| (index * 7 + 13) % n).collect();
    let permuted_xs: Vec<Vec<f32>> = permutation.iter().map(|&index| xs[index].clone()).collect();
    let permuted_ys: Vec<f32> = permutation.iter().map(|&index| ys[index]).collect();

    let batch: Vec<Observation<RlsSample>> =
        xs.iter().zip(&ys).map(|(x, &y)| sample(x, y)).collect();
    let permuted_batch: Vec<Observation<RlsSample>> = permuted_xs
        .iter()
        .zip(&permuted_ys)
        .map(|(x, &y)| sample(x, y))
        .collect();

    // lambda = 1: the multiset fixes the minimizer, so permutations agree within
    // tolerance (not bitwise: the reduction order differs).
    let model_one = Rls::new(d, 1.0, delta).unwrap();
    let forward = model_one
        .process_batch_reference(model_one.initial_state(), &batch, batch.len())
        .unwrap();
    let permuted = model_one
        .process_batch_reference(
            model_one.initial_state(),
            &permuted_batch,
            permuted_batch.len(),
        )
        .unwrap();
    let weights = to_f64_vec(forward.w.as_slice());
    let difference = max_abs_difference(&weights, &to_f64_vec(permuted.w.as_slice()));
    assert!(
        difference <= 1e-4 * (1.0 + infinity_norm(&weights)),
        "lambda=1 batch permutation changed the fit by {difference}"
    );

    // lambda < 1: recency weighting makes order semantically significant.
    let model_below = Rls::new(d, 0.5, delta).unwrap();
    let ordered = model_below
        .process_batch_reference(model_below.initial_state(), &batch, batch.len())
        .unwrap();
    let reordered = model_below
        .process_batch_reference(
            model_below.initial_state(),
            &permuted_batch,
            permuted_batch.len(),
        )
        .unwrap();
    let difference = max_abs_difference(
        &to_f64_vec(ordered.w.as_slice()),
        &to_f64_vec(reordered.w.as_slice()),
    );
    assert!(
        difference > 1e-3,
        "lambda<1 batch permutation should change the weighted solution, difference {difference}"
    );
}

#[test]
fn bounded_reference_chunk_continuity_including_empty() {
    let model = Rls::new(2, 0.97, 5.0).unwrap();
    let observations: Vec<Observation<RlsSample>> = (0..7)
        .map(|k| {
            sample(
                &[(k as f32 * 0.4).sin(), (k as f32 * 0.25).cos()],
                (k as f32 * 0.3).cos(),
            )
        })
        .collect();

    let continuous = model
        .process_batch_reference(model.initial_state(), &observations, observations.len())
        .unwrap();

    let mut chunked = model.initial_state();
    for chunk in observations.chunks(2) {
        chunked = model
            .process_batch_reference(chunked, chunk, chunk.len())
            .unwrap();
        // An empty chunk is a no-op that returns the supplied state.
        chunked = model.process_batch_reference(chunked, &[], 0).unwrap();
    }
    assert_state_bitwise_eq(&chunked, &continuous);
}

#[test]
fn bounded_reference_prefixes_are_causal() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let observations: Vec<Observation<RlsSample>> = (0..8)
        .map(|k| {
            sample(
                &[(k as f32 * 0.2).sin(), (k as f32 * 0.6).cos()],
                k as f32 * 0.1,
            )
        })
        .collect();

    for prefix in [1_usize, 3, 5, 8] {
        let batched = model
            .process_batch_reference(model.initial_state(), &observations[..prefix], prefix)
            .unwrap();

        let mut repeated = model.initial_state();
        for observation in &observations[..prefix] {
            repeated = model.update(&repeated, observation).unwrap();
        }
        assert_state_bitwise_eq(&batched, &repeated);
    }
}

#[test]
fn bounded_reference_bound_rejection_and_empty_input() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let batch = vec![
        sample(&[0.1, 0.2], 1.0),
        sample(&[0.3, -0.4], 2.0),
        sample(&[0.5, 0.6], 3.0),
    ];

    // The exact bound succeeds.
    assert!(
        model
            .process_batch_reference(model.initial_state(), &batch, batch.len())
            .is_ok()
    );

    // Oversized: rejected before any transition; the moved-in state is
    // recovered bitwise without a defensive caller clone.
    let supplied = state(&[0.25, -0.5], &[&[1.0, 0.0], &[0.0, 1.0]]);
    match model.process_batch_reference(supplied.clone(), &batch, 2) {
        Err(RlsBatchError::BatchTooLarge {
            max,
            actual,
            state: returned,
        }) => {
            assert_eq!(max, 2);
            assert_eq!(actual, 3);
            assert_state_bitwise_eq(&returned, &supplied);
        }
        other => panic!("expected BatchTooLarge, got {other:?}"),
    }

    // max_batch == 0 accepts only the empty batch.
    match model.process_batch_reference(model.initial_state(), &batch, 0) {
        Err(RlsBatchError::BatchTooLarge { max, actual, .. }) => {
            assert_eq!(max, 0);
            assert_eq!(actual, 3);
        }
        other => panic!("expected BatchTooLarge for max_batch == 0, got {other:?}"),
    }

    let supplied = state(&[1.0, -1.0], &[&[2.0, 0.0], &[0.0, 3.0]]);
    let empty = model
        .process_batch_reference(supplied.clone(), &[], 0)
        .unwrap();
    assert_state_bitwise_eq(&empty, &supplied);
}

#[test]
fn bounded_reference_returns_supplied_state_unvalidated() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    // Dimension-mismatched and NaN-containing: not a valid RLS state anywhere.
    let corrupt = state(&[f32::NAN, 0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
    let batch = vec![sample(&[0.1, 0.2], 1.0), sample(&[0.3, 0.4], 2.0)];

    // Oversized: the bound is checked before any state validation, so the
    // corrupt state is returned bitwise unchanged (NaN bits included).
    match model.process_batch_reference(corrupt.clone(), &batch, 1) {
        Err(RlsBatchError::BatchTooLarge {
            state: returned, ..
        }) => assert_state_bitwise_eq(&returned, &corrupt),
        other => panic!("expected BatchTooLarge, got {other:?}"),
    }

    // Empty: returned bitwise unchanged without validation.
    let empty = model
        .process_batch_reference(corrupt.clone(), &[], 0)
        .unwrap();
    assert_state_bitwise_eq(&empty, &corrupt);
}

#[test]
fn bounded_reference_failure_state_matches_committed_prefix() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let good = |k: usize| {
        sample(
            &[(k as f32 * 0.3).sin(), (k as f32 * 0.2).cos()],
            k as f32 * 0.5,
        )
    };
    let bad = sample(&[f32::NAN, 0.0], 1.0);

    // Start from a committed state that is not the initial state, so a
    // whole-batch rollback to the supplied state would be observable.
    let base = model.update(&model.initial_state(), &good(9)).unwrap();

    for k in [0_usize, 2, 4] {
        let mut observations: Vec<Observation<RlsSample>> = (0..5).map(good).collect();
        observations[k] = bad.clone();

        match model.process_batch_reference(base.clone(), &observations, observations.len()) {
            Err(RlsBatchError::Transition {
                failed_index,
                state: returned,
                error,
            }) => {
                assert_eq!(failed_index, k);
                assert_eq!(error, RlsError::NonFiniteInput);

                // Both w and P bitwise-equal the fold over batch[..k]; the
                // supplied batch-entry state applies only when k == 0.
                let mut expected = base.clone();
                for observation in &observations[..k] {
                    expected = model.update(&expected, observation).unwrap();
                }
                assert_state_bitwise_eq(&returned, &expected);

                // The caller can continue from the returned state.
                let mut continued = returned.clone();
                let mut reference = expected;
                for observation in &observations[k + 1..] {
                    continued = model.update(&continued, observation).unwrap();
                    reference = model.update(&reference, observation).unwrap();
                }
                assert_state_bitwise_eq(&continued, &reference);
            }
            other => panic!("expected Transition at k={k}, got {other:?}"),
        }
    }
}

#[test]
fn bounded_reference_moved_state_recovery_from_both_errors() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let good = |k: usize| {
        sample(
            &[(k as f32 * 0.25).sin(), (k as f32 * 0.15).cos()],
            k as f32 * 0.2,
        )
    };
    let batch: Vec<Observation<RlsSample>> = (0..4).map(good).collect();

    // BatchTooLarge: recover the supplied state without a clone, then continue.
    let supplied = model.update(&model.initial_state(), &good(11)).unwrap();
    let recovered = match model.process_batch_reference(supplied.clone(), &batch, 1) {
        Err(RlsBatchError::BatchTooLarge { state, .. }) => state,
        other => panic!("expected BatchTooLarge, got {other:?}"),
    };
    assert_state_bitwise_eq(&recovered, &supplied);
    let continued = model
        .process_batch_reference(recovered, &batch[..1], 1)
        .unwrap();
    let expected = model.update(&supplied, &batch[0]).unwrap();
    assert_state_bitwise_eq(&continued, &expected);

    // Transition: recover the last committed prefix state, then continue.
    let mut failing = batch.clone();
    failing[1] = sample(&[f32::NAN, 0.0], 0.0);
    let recovered = match model.process_batch_reference(supplied.clone(), &failing, failing.len()) {
        Err(RlsBatchError::Transition {
            failed_index,
            state,
            ..
        }) => {
            assert_eq!(failed_index, 1);
            state
        }
        other => panic!("expected Transition, got {other:?}"),
    };
    let expected_prefix = model.update(&supplied, &batch[0]).unwrap();
    assert_state_bitwise_eq(&recovered, &expected_prefix);

    let continued = model
        .process_batch_reference(recovered, &batch[2..], batch.len() - 2)
        .unwrap();
    let mut reference = expected_prefix;
    for observation in &batch[2..] {
        reference = model.update(&reference, observation).unwrap();
    }
    assert_state_bitwise_eq(&continued, &reference);
}

#[test]
fn bounded_reference_transition_error_mapping() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();

    // Non-finite target, wrapped as a transition failure at its index.
    let non_finite = vec![sample(&[1.0, 0.0], 1.0), sample(&[0.0, 1.0], f32::INFINITY)];
    match model.process_batch_reference(model.initial_state(), &non_finite, non_finite.len()) {
        Err(RlsBatchError::Transition {
            failed_index,
            error,
            ..
        }) => {
            assert_eq!(failed_index, 1);
            assert_eq!(error, RlsError::NonFiniteInput);
        }
        other => panic!("expected Transition for a non-finite target, got {other:?}"),
    }

    // Wrong feature dimension, wrapped as a transition failure.
    let mismatched = vec![sample(&[1.0, 0.0], 1.0), sample(&[1.0], 2.0)];
    match model.process_batch_reference(model.initial_state(), &mismatched, mismatched.len()) {
        Err(RlsBatchError::Transition {
            failed_index,
            error,
            ..
        }) => {
            assert_eq!(failed_index, 1);
            assert_eq!(
                error,
                RlsError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
        }
        other => panic!("expected Transition for a dimension mismatch, got {other:?}"),
    }
}

#[test]
fn bounded_reference_returns_supplied_state_after_first_update_rejects_it() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let supplied = state(&[12345.0], &[&[67890.0]]);
    let observation = [sample(&[0.1, 0.2], 1.0)];

    match model.process_batch_reference(supplied.clone(), &observation, 1) {
        Err(RlsBatchError::Transition {
            failed_index,
            state: returned,
            error,
        }) => {
            assert_eq!(failed_index, 0);
            assert_eq!(
                error,
                RlsError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
            assert_state_bitwise_eq(&returned, &supplied);
            let message = RlsBatchError::Transition {
                failed_index,
                state: returned,
                error,
            }
            .to_string();
            assert!(message.contains("observation 0"));
            assert!(message.contains("no observation was committed"));
            assert!(message.contains("returned unchanged"));
            assert!(!message.contains("12345"));
            assert!(!message.contains("67890"));
        }
        other => panic!("expected Transition at observation 0, got {other:?}"),
    }
}

#[test]
fn one_shared_model_drives_independent_stream_states() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let sequence_a = vec![sample(&[1.0, 0.0], 1.0), sample(&[0.0, 1.0], 2.0)];
    let sequence_b = vec![sample(&[0.0, 1.0], 3.0), sample(&[1.0, 0.5], -1.0)];

    let mut state_a = model.initial_state();
    for observation in &sequence_a {
        state_a = process_one(&model, &state_a, observation).unwrap();
    }
    let state_b = process_stream(&model, model.initial_state(), sequence_b.clone()).unwrap();

    assert_ne!(state_a, state_b);

    let mut reference_a = model.initial_state();
    for observation in &sequence_a {
        reference_a = model.update(&reference_a, observation).unwrap();
    }
    assert_state_bitwise_eq(&state_a, &reference_a);

    let mut reference_b = model.initial_state();
    for observation in &sequence_b {
        reference_b = model.update(&reference_b, observation).unwrap();
    }
    assert_state_bitwise_eq(&state_b, &reference_b);
}

#[test]
fn bounded_fold_recovers_exact_prefixes_bounds_and_continuation() {
    let model = model64(1, 1.0, 1.0);
    let first = model
        .update(&model.initial_state(), &sample64(&[1.0], 1.0))
        .unwrap();
    let second = model.update(&first, &sample64(&[1.0], 2.0)).unwrap();
    let valid = [
        sample64(&[1.0], 1.0),
        sample64(&[1.0], 2.0),
        sample64(&[1.0], 3.0),
    ];
    let prefixes = [model.initial_state(), first.clone(), second];

    for (failed_index, expected) in prefixes.into_iter().enumerate() {
        let mut batch = valid.clone();
        batch[failed_index] = sample64(&[1.0], f64::NAN);
        let error = model
            .process_batch_reference(model.initial_state(), &batch, 3)
            .unwrap_err();
        match error {
            RlsBatchError64::Transition {
                failed_index: actual_index,
                state,
                error,
            } => {
                assert_eq!(actual_index, failed_index);
                assert_state64_bitwise_eq(&state, &expected);
                assert_eq!(error, RlsError64::NonFiniteInput);
                assert_eq!(
                    StdError::source(&error).map(|source| source.to_string()),
                    None
                );
            }
            other => panic!("unexpected batch failure: {other:?}"),
        }
    }

    let mut invalid = valid.clone();
    invalid[1] = sample64(&[1.0], f64::NAN);
    let error = model
        .process_batch_reference(model.initial_state(), &invalid, 3)
        .unwrap_err();
    let display = error.to_string();
    assert!(display.contains("1 observations were committed"));
    assert!(display.contains("no whole-batch rollback"));
    assert_eq!(
        StdError::source(&error).and_then(|source| source.downcast_ref::<RlsError64>()),
        Some(&RlsError64::NonFiniteInput)
    );

    let too_large_error = model
        .process_batch_reference(model.initial_state(), &valid, 2)
        .unwrap_err();
    let message = too_large_error.to_string();
    assert!(message.contains("exceeds maximum 2"));
    assert!(too_large_error.source().is_none());
    let returned = match too_large_error {
        RlsBatchError64::BatchTooLarge { max, actual, state } => {
            assert_eq!((max, actual), (2, 3));
            state
        }
        other => panic!("unexpected bound failure: {other:?}"),
    };
    let empty: [Observation<RlsSample64>; 0] = [];
    assert_state64_bitwise_eq(
        &model
            .process_batch_reference(returned.clone(), &empty, 0)
            .unwrap(),
        &returned,
    );
    let foreign_model = model64(2, 1.0, 1.0);
    let foreign = foreign_model.initial_state();
    assert_state64_bitwise_eq(
        &model.process_batch_reference(foreign, &empty, 0).unwrap(),
        &foreign_model.initial_state(),
    );
    let nonempty_zero_bound = model
        .process_batch_reference(returned.clone(), &valid, 0)
        .unwrap_err();
    assert!(matches!(
        nonempty_zero_bound,
        RlsBatchError64::BatchTooLarge { .. }
    ));

    let moved = match model.process_batch_reference(returned, &valid, 2) {
        Err(RlsBatchError64::BatchTooLarge { state, .. }) => state,
        _ => panic!("expected bound rejection"),
    };
    let moved_next = model.update(&moved, &sample64(&[1.0], 1.0)).unwrap();
    let direct_next = model
        .update(&model.initial_state(), &sample64(&[1.0], 1.0))
        .unwrap();
    assert_state64_bitwise_eq(&moved_next, &direct_next);

    let (model, learned) = learned_state64();
    let sequence = [
        (vec![1.0, 0.0], 2.0),
        (vec![1.0, 0.0], 3.0),
        (vec![0.0, 1.0], -1.0),
        (vec![0.5, 0.5], 0.25),
        (vec![-1.0, 2.0], 0.75),
    ];
    let observations: Vec<_> = sequence.iter().map(|(x, y)| sample64(x, *y)).collect();
    let mut direct = learned.clone();
    for (x, y) in &sequence {
        direct = model.update(&direct, &sample64(x, *y)).unwrap();
    }
    let mut chunked = learned.clone();
    for (start, end, bound) in [(0, 2, 2), (2, 2, 0), (2, 3, 1), (3, 5, 2)] {
        chunked = model
            .process_batch_reference(chunked, &observations[start..end], bound)
            .unwrap();
    }
    assert_state64_bitwise_eq(&chunked, &direct);
}

#[test]
fn streaming_continuation_is_ordered_and_independent_streams_do_not_interfere() {
    let model = model64(1, 1.0, 1.0);
    let observations = [
        sample64(&[1.0], 1.0),
        sample64(&[1.0], f64::NAN),
        sample64(&[1.0], 2.0),
    ];
    let mut failures = 0_usize;
    let summary = process_stream_continuing(
        &model,
        model.initial_state(),
        observations.clone(),
        |position, _error| {
            assert_eq!(position, 1);
            failures += 1;
        },
    )
    .unwrap();
    assert_eq!(failures, 1);
    let streamed = summary.state;
    let first = model
        .update(&model.initial_state(), &sample64(&[1.0], 1.0))
        .unwrap();
    let expected = model.update(&first, &sample64(&[1.0], 2.0)).unwrap();
    assert_state64_bitwise_eq(&streamed, &expected);
    let bounded = model
        .process_batch_reference(model.initial_state(), &observations, 3)
        .unwrap_err();
    match bounded {
        RlsBatchError64::Transition {
            failed_index,
            state,
            ..
        } => {
            assert_eq!(failed_index, 1);
            assert_state64_bitwise_eq(&state, &first);
        }
        other => panic!("unexpected bounded failure: {other:?}"),
    }

    let (model, _) = learned_state64();
    let mut left = model.initial_state();
    let mut right = model.initial_state();
    let left_rows = [([0.5, -0.5], 1.0), ([1.0, 2.0], -1.0)];
    let right_rows = [([-1.0, 0.25], 3.0), ([0.0, -2.0], 0.5)];
    for ((x, y), (x2, y2)) in left_rows.iter().zip(&right_rows) {
        left = model.update(&left, &sample64(x, *y)).unwrap();
        right = model.update(&right, &sample64(x2, *y2)).unwrap();
    }
    let mut left_reference = model.initial_state();
    for (x, y) in &left_rows {
        left_reference = model.update(&left_reference, &sample64(x, *y)).unwrap();
    }
    assert_state64_bitwise_eq(&left, &left_reference);
    let mut right_reference = model.initial_state();
    for (x, y) in &right_rows {
        right_reference = model.update(&right_reference, &sample64(x, *y)).unwrap();
    }
    assert_state64_bitwise_eq(&right, &right_reference);
}

#[test]
fn streaming_publication_keeps_prior_success_after_later_checked_failure() {
    let model = model64(1, 1.0, 1.0);
    let source = state64_with_weights(&model, &[f64::from(f32::MAX)]);
    let source_before = source.clone();
    let active = model.to_serving_f32(&source).unwrap();
    let active_before = snapshot_bits(&active);
    let rows = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0])];
    let mut published = Vec::new();
    let mut failure = None;
    for row in &rows {
        match active.predict_checked(row) {
            Ok(prediction) => published.push(prediction),
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    assert_eq!(failure, Some(RlsError::NonFiniteOutput));
    assert_eq!(published, [f32::MAX]);
    assert_eq!(snapshot_bits(&active), active_before);
    assert_state64_bitwise_eq(&source, &source_before);
}

#[test]
fn chunking_preserves_the_ordered_trajectory() {
    // Property under test: state continuation across chunk boundaries. The
    // committed state returned by one chunk is the initial state of the next.
    // Three unequal chunk sizes (2, 3, 2) over ordered x1..x7 must reproduce the
    // repeated-process_one reference bitwise (same operations, same order).
    let model = Rls::new(2, 0.97, 5.0).unwrap();
    let observations: Vec<Observation<RlsSample>> = (0..7)
        .map(|k| {
            sample(
                &[(k as f32 * 0.4).sin(), (k as f32 * 0.25).cos()],
                (k as f32 * 0.3).cos(),
            )
        })
        .collect();

    let mut reference = model.initial_state();
    for observation in &observations {
        reference = process_one(&model, &reference, observation).unwrap();
    }

    let s1 = process_batch(&model, model.initial_state(), observations[0..2].to_vec()).unwrap();
    let s2 = process_batch(&model, s1, observations[2..5].to_vec()).unwrap();
    let s3 = process_batch(&model, s2, observations[5..7].to_vec()).unwrap();

    assert_state_bitwise_eq(&reference, &s3);
}
