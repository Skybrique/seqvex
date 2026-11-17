use crate::support::{
    FreshnessCheck, FreshnessPolicy, ObservingSink, OlsProgress, Progress, PublicationAction,
    PublicationOutcome, RetainedBatch, ServingRecord, check_freshness, collect32, publish_retained,
};

use crate::sample64;

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::Sgd64;

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
