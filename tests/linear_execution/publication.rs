use crate::support::{
    AcceptanceDecision, ActiveOls, InstallError, ObservingSink, OlsCandidate, PendingBudget,
    Progress, PublicationAction, PublicationOutcome, RetainedBatch, TemporalFailure,
    apply_pending_prefix, collect32, publish_retained, try_install_ols,
};

use crate::{rls_sample64, sample32, sample64};

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::Rls64;
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdError, SgdState64};

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
