use crate::support;
use crate::support::{
    FeatureSnapshot, FreshnessCheck, FreshnessPolicy, FreshnessPolicyError, ObservingSink,
    OrderedSource, PendingBudget, PredictionRecord, Progress, PublicationAction,
    PublicationOutcome, RetainedBatch, ServingRecord, TemporalFailure, WorkflowMode,
    apply_pending_prefix, check_freshness, collect32, publish_retained, recover_missing_prediction,
};

use crate::{rls_sample64, sgd_sample};

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls64, RlsSample64};
use seqvex::models::linear::sgd::Sgd;

pub(super) fn run_temporal_and_publication() -> Result<(), String> {
    let model =
        Rls64::new(1, 1.0, 1.0).map_err(|error| format!("RLS64 config failed: {error:?}"))?;
    let observation1 = rls_sample64(&[1.0], 1.0);
    let observation2 = rls_sample64(&[2.0], 0.0);
    let historical_mode = WorkflowMode::HistoricalReplay;
    let live_mode = WorkflowMode::LiveAdaptive;
    let historical_state = match historical_mode {
        WorkflowMode::HistoricalReplay => [observation1.clone(), observation2.clone()]
            .iter()
            .try_fold(model.initial_state(), |state, row| {
                model.update(&state, row)
            })
            .map_err(|error| format!("HistoricalReplay failed: {error:?}"))?,
        WorkflowMode::LiveAdaptive => unreachable!("separate live trace below"),
    };
    if !matches!(live_mode, WorkflowMode::LiveAdaptive) {
        return Err("LiveAdaptive mode was not selected".to_string());
    }
    let mut learner = model.initial_state();
    let mut progress = Progress::default();
    let b0 = model
        .to_serving_f32(&learner)
        .map_err(|error| format!("B0 conversion failed: {error:?}"))?;
    let mut active = (
        b0,
        ServingRecord {
            identity: 0,
            served_through: 0,
        },
    );
    if check_freshness(None, &active.1, progress.learning)
        != Err(FreshnessPolicyError::MissingPolicy)
    {
        return Err("missing B policy was not rejected".to_string());
    }
    if check_freshness(
        Some(FreshnessPolicy::LatestEligible),
        &active.1,
        progress.learning,
    ) != Ok(FreshnessCheck::Ready)
    {
        return Err("P(10) was blocked by same-time L(10) labels".to_string());
    }
    let p10 = active
        .0
        .predict_checked(&Vector::from_slice(
            observation2
                .value()
                .features
                .as_slice()
                .iter()
                .map(|v| *v as f32)
                .collect::<Vec<_>>()
                .as_slice(),
        ))
        .map_err(|error| format!("P(10) prediction failed: {error:?}"))?;
    let record2 = PredictionRecord {
        observation_index: 1,
        model_identity: active.1.identity,
        output: p10,
    };
    let update = model
        .update(&learner, &observation1)
        .map_err(|error| format!("L(10) update failed: {error:?}"))?;
    progress
        .commit_update(&mut learner, Ok::<_, ()>(update))
        .map_err(|_| "L(10) progress commit failed")?;
    if check_freshness(
        Some(FreshnessPolicy::LatestEligible),
        &active.1,
        progress.learning,
    ) != Ok(FreshnessCheck::Blocked {
        learned: 1,
        served: 0,
    }) {
        return Err("later LatestEligible prediction did not block stale B0".to_string());
    }
    let b1 = model
        .to_serving_f32(&learner)
        .map_err(|error| format!("B1 conversion failed: {error:?}"))?;
    progress
        .install_serving(&mut active, Ok::<_, ()>(b1), 1)
        .map_err(|_| "B1 installation failed")?;
    if check_freshness(
        Some(FreshnessPolicy::LatestEligible),
        &active.1,
        progress.learning,
    ) != Ok(FreshnessCheck::Ready)
    {
        return Err("installed B1 did not satisfy LatestEligible".to_string());
    }
    if check_freshness(
        Some(FreshnessPolicy::FrozenSelected {
            serving_identity: active.1.identity,
            served_through: active.1.served_through,
        }),
        &active.1,
        progress.learning,
    ) != Ok(FreshnessCheck::Ready)
    {
        return Err("FrozenSelected did not retain the pinned snapshot".to_string());
    }
    let p11 = active
        .0
        .predict_checked(&Vector::from_slice(&[1.0]))
        .map_err(|error| format!("later prediction failed: {error:?}"))?;
    if p11.to_bits() == 0.0f32.to_bits() || record2.model_identity != 0 {
        return Err("equal-time snapshot identities were not preserved".to_string());
    }

    let mut delayed = PendingBudget::new_at(2, 0);
    let ordered_rows = [
        rls_sample64(&[1.0], 1.0),
        rls_sample64(&[2.0], -1.0),
        rls_sample64(&[3.0], 0.5),
    ];
    delayed
        .admit_later(1, ordered_rows[1].clone())
        .map_err(|_| "label 1 buffer full")?;
    delayed
        .admit_later(2, ordered_rows[2].clone())
        .map_err(|_| "label 2 buffer full")?;
    let mut delayed_state = model.initial_state();
    let mut delayed_progress = Progress::default();
    if !apply_pending_prefix(
        &mut delayed,
        &mut delayed_state,
        &mut delayed_progress,
        |state, row| model.update(state, row),
    )
    .map_err(|error| format!("unexpected delayed update failure: {error:?}"))?
    .is_empty()
    {
        return Err("later labels were applied across a missing head".to_string());
    }
    delayed
        .admit_head(0, ordered_rows[0].clone())
        .map_err(|_| "head admission failed")?;
    let states = apply_pending_prefix(
        &mut delayed,
        &mut delayed_state,
        &mut delayed_progress,
        |state, row| model.update(state, row),
    )
    .map_err(|error| format!("ordered delayed release failed: {error:?}"))?;
    let expected = ordered_rows
        .iter()
        .try_fold(model.initial_state(), |state, row| {
            model.update(&state, row)
        })
        .map_err(|error| format!("ordered reference failed: {error:?}"))?;
    if states.len() != 3 || delayed_state != expected || delayed_progress.learning != 3 {
        return Err("delayed labels did not commit in original order".to_string());
    }
    let historical_reference = model
        .update(
            &model.update(&model.initial_state(), &observation1).unwrap(),
            &observation2,
        )
        .map_err(|error| format!("HistoricalReplay reference failed: {error:?}"))?;
    if historical_state != historical_reference {
        return Err("HistoricalReplay did not match its ordered reference".to_string());
    }

    // The full ordinary buffer still admits the required gap head within C+1.
    let mut gap = PendingBudget::new_at(2, 4);
    let gap_rows = [
        rls_sample64(&[4.0], 0.25),
        rls_sample64(&[5.0], 0.5),
        rls_sample64(&[6.0], -0.5),
    ];
    gap.admit_later(5, gap_rows[1].clone())
        .map_err(|_| "pending 5 failed")?;
    gap.admit_later(6, gap_rows[2].clone())
        .map_err(|_| "pending 6 failed")?;
    gap.admit_head(4, gap_rows[0].clone())
        .map_err(|_| "reserved head failed")?;
    let mut gap_state = model.initial_state();
    let mut gap_progress = Progress::default();
    let gap_states =
        apply_pending_prefix(&mut gap, &mut gap_state, &mut gap_progress, |state, row| {
            model.update(state, row)
        })
        .map_err(|error| format!("gap recovery failed: {error:?}"))?;
    if gap_states.len() != 3 || gap_progress.learning != 3 || gap.pending_len() != 0 {
        return Err("C+1 pending gap did not advance in order".to_string());
    }

    let mut missing_budget = PendingBudget::new_at(1, 4);
    missing_budget
        .admit_head(4, gap_rows[0].clone())
        .map_err(|_| "missing head failed")?;
    let missing = recover_missing_prediction(
        (4, gap_rows[0].value().target),
        9,
        3,
        None,
        None::<&FeatureSnapshot<Vector64, (), seqvex::models::linear::rls::RlsState64>>,
        |_, _, _| Ok::<f64, ()>(0.0),
    );
    if !matches!(
        missing,
        Err(support::PredictionResolutionFailure::MissingPrediction { item: (4, _) })
    ) || missing_budget.pending_len() != 1
    {
        return Err("missing prediction did not preserve unapplied work".to_string());
    }
    let mut source = OrderedSource::new(vec![(6, "six"), (7, "seven")]);
    if source.next_expected(5) != Err(TemporalFailure::SourceCannotDeliverGap)
        || source.consumed() != 0
        || source.remaining() != 2
    {
        return Err("source backpressure consumed later work".to_string());
    }

    let future_trace = |future: &Observation<RlsSample64>| {
        let mut state = model.initial_state();
        let b0 = model.to_serving_f32(&state).unwrap();
        let earlier1 = b0.predict_checked(&Vector::from_slice(&[1.0])).unwrap();
        state = model.update(&state, &ordered_rows[0]).unwrap();
        let b1 = model.to_serving_f32(&state).unwrap();
        let earlier2 = b1.predict_checked(&Vector::from_slice(&[2.0])).unwrap();
        state = model.update(&state, &ordered_rows[1]).unwrap();
        state = model.update(&state, future).unwrap();
        ([earlier1, earlier2], state)
    };
    let future_a = rls_sample64(&[4.0], 1.0);
    let future_b = rls_sample64(&[9.0], -3.0);
    let (earlier_a, state_a) = future_trace(&future_a);
    let (earlier_b, state_b) = future_trace(&future_b);
    if earlier_a != earlier_b || state_a == state_b {
        return Err(
            "future perturbation changed earlier outputs or not later learning".to_string(),
        );
    }
    let isolated_b = model.initial_state();
    let isolated_prediction = model
        .to_serving_f32(&isolated_b)
        .unwrap()
        .predict_checked(&Vector::from_slice(&[1.0]))
        .unwrap();
    let mut isolated_a = model.initial_state();
    let mut isolated_progress_a = Progress::default();
    let isolated_progress_b = Progress::default();
    let mut pending_a = PendingBudget::new_at(1, 0);
    let pending_b = PendingBudget::<Observation<RlsSample64>>::new_at(1, 0);
    pending_a
        .admit_head(0, "sequence A")
        .map_err(|_| "sequence A pending failed")?;
    let retained_b = RetainedBatch {
        observation_indices: vec![0],
        values: vec![isolated_prediction],
        model_identity: 51,
        calculated_count: 1,
    };
    let isolated_update = model.update(&isolated_a, &future_a).unwrap();
    isolated_progress_a
        .commit_update(&mut isolated_a, Ok::<_, ()>(isolated_update))
        .map_err(|_| "sequence A update tracking failed")?;
    let _ = pending_a.inspect_head();
    if isolated_a == isolated_b
        || model
            .to_serving_f32(&isolated_b)
            .unwrap()
            .predict_checked(&Vector::from_slice(&[1.0]))
            .unwrap()
            .to_bits()
            != isolated_prediction.to_bits()
        || isolated_progress_a.learning != 1
        || isolated_progress_b.learning != 0
        || pending_b.pending_len() != 0
        || retained_b.values[0].to_bits() != isolated_prediction.to_bits()
        || retained_b.model_identity != 51
    {
        return Err("sequence isolation changed the other sequence".to_string());
    }

    let publication_model =
        Sgd::new(1, 0.1).map_err(|error| format!("publication SGD config: {error:?}"))?;
    let mut publication_state = publication_model.initial_state();
    let mut publication_progress = Progress::default();
    let learned = publication_model
        .update(&publication_state, &sgd_sample(&[1.0], 1.0))
        .map_err(|error| format!("publication learning failed: {error:?}"))?;
    publication_progress
        .commit_update(&mut publication_state, Ok::<_, ()>(learned))
        .map_err(|_| "publication learning progress failed")?;
    let (selected, selected_record) = publication_progress.select(&publication_state, 77);
    let publication_rows: Vec<_> = (0..5)
        .map(|index| Observation::new(Vector::from_slice(&[index as f32 + 1.0])))
        .collect();
    let calculated = collect32(5, &publication_rows, |features| {
        publication_model.predict(&selected, features)
    })
    .map_err(|error| format!("publication calculation failed: {error:?}"))?;
    let batch = publication_progress
        .retain_calculation(Ok::<_, ()>(RetainedBatch {
            observation_indices: vec![0, 1, 2, 3, 4],
            values: calculated,
            model_identity: selected_record.identity,
            calculated_count: 0,
        }))
        .map_err(|_| "retain calculated outputs failed")?;
    let retained_values = batch.values.clone();
    let mut retained = Some(batch);
    let mut sink = ObservingSink::<f32, &str>::new(vec![
        PublicationOutcome::NoWrite,
        PublicationOutcome::AcknowledgedPrefixUnwrittenSuffix {
            acknowledged_count: 2,
        },
        PublicationOutcome::FullyAcknowledged {
            acknowledged_count: 3,
        },
    ]);
    if publish_retained(&mut retained, &mut publication_progress, &mut sink)
        != PublicationAction::Retry
    {
        return Err("NoWrite did not retain publication outputs".to_string());
    }
    let newer = publication_model
        .update(&publication_state, &sgd_sample(&[1.0], -1.0))
        .map_err(|error| format!("newer learning failed: {error:?}"))?;
    publication_progress
        .commit_update(&mut publication_state, Ok::<_, ()>(newer))
        .map_err(|_| "newer learning progress failed")?;
    for _ in 0..2 {
        match publish_retained(&mut retained, &mut publication_progress, &mut sink) {
            PublicationAction::Retry => continue,
            PublicationAction::Complete => break,
            other => return Err(format!("safe publication retry stopped: {other:?}")),
        }
    }
    if retained.is_some()
        || publication_progress.published != 5
        || sink.calls() != 3
        || sink.submitted()[0].values != retained_values
        || sink.submitted()[1].values != retained_values
        || sink.submitted()[2].observation_indices != vec![2, 3, 4]
        || sink.submitted()[2].model_identity != selected_record.identity
    {
        return Err(
            "publication did not reuse retained outputs or exact unwritten suffix".to_string(),
        );
    }
    let suffix = RetainedBatch {
        observation_indices: vec![0, 1, 2],
        values: vec![1.0f32, 2.0, 3.0],
        model_identity: 88,
        calculated_count: 3,
    };
    let mut uncertain = Some(suffix.clone());
    let mut uncertain_progress = Progress::default();
    let mut uncertain_sink = ObservingSink::<f32, &str>::new(vec![
        PublicationOutcome::AcknowledgedPrefixUncertainSuffix {
            acknowledged_count: 1,
        },
    ]);
    if publish_retained(&mut uncertain, &mut uncertain_progress, &mut uncertain_sink)
        != PublicationAction::Reconcile
        || uncertain_progress.published != 1
        || uncertain.as_ref().unwrap().observation_indices != vec![1, 2]
        || uncertain_sink.calls() != 1
    {
        return Err("uncertain suffix was blindly retried".to_string());
    }
    let mut unclassified = Some(suffix);
    let mut unclassified_progress = Progress::default();
    let mut unclassified_sink =
        ObservingSink::<f32, &str>::new(vec![PublicationOutcome::Unclassified {
            error: "unclassified",
        }]);
    if publish_retained(
        &mut unclassified,
        &mut unclassified_progress,
        &mut unclassified_sink,
    ) != PublicationAction::Stop("unclassified")
        || unclassified_progress.published != 0
        || unclassified_sink.calls() != 1
    {
        return Err("unclassified outcome was assumed to be no-write".to_string());
    }
    println!(
        "Temporal: P(10) {:.7} on B0, later {:.7} on B1; delayed labels {}, gap pending {}; publication acknowledged {} of {}",
        record2.output,
        p11,
        delayed_progress.learning,
        gap.pending_len(),
        publication_progress.published,
        publication_progress.calculated
    );
    Ok(())
}
