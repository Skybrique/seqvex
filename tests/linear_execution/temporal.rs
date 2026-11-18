use crate::support;
use crate::support::{
    FeatureSnapshot, FreshnessCheck, FreshnessPolicy, FreshnessPolicyError, OrderedSource,
    PendingBudget, PredictionRecord, Progress, RetainedBatch, ServingRecord, TemporalFailure,
    WorkflowMode, apply_pending_prefix, check_freshness, collect32, collect64,
    recover_missing_prediction,
};

use crate::{assert_f32_bits, assert_f64_bits, rls_sample32, rls_sample64, sample32, sample64};

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample64};
use seqvex::models::linear::sgd::{Sgd, Sgd64};

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
    let grouped32 = sgd
        .predict_batch(&sgd_state, &rows32, rows32.len())
        .unwrap();
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
    let b_grouped = serving.predict_batch(&rows32, rows32.len()).unwrap();
    for (left, right) in b_direct.iter().zip(&b_grouped) {
        assert_f32_bits(*left, *right);
    }

    let rows64 = vec![
        Observation::new(Vector64::from_slice(&[0.5, 0.25])),
        Observation::new(Vector64::from_slice(&[-0.25, 0.75])),
    ];
    let direct_a: Vec<f64> = rows64
        .iter()
        .map(|row| sgd64.predict(&sgd64_state, row.value()).unwrap())
        .collect();
    let grouped_a = sgd64
        .predict_batch(&sgd64_state, &rows64, rows64.len())
        .unwrap();
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
