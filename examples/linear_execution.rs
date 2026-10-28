//! Deterministic end-to-end caller workflows for OLS, RLS and SGD.
//!
//! This is a caller protocol example, not a quality or performance benchmark.

#[path = "../tests/support/linear_execution.rs"]
mod support;

use support::{
    AcceptanceDecision, ActiveOls, BatchFailure, FeatureSnapshot, FreshnessCheck, FreshnessPolicy,
    FreshnessPolicyError, InstallError, ObservingSink, OlsCandidate, OlsProgress, OrderedSource,
    PendingBudget, PredictionRecord, Progress, PublicationAction, PublicationOutcome,
    RetainedBatch, ServingRecord, TemporalFailure, WorkflowMode, apply_pending_prefix,
    check_freshness, collect32, collect64, publish_retained, recover_missing_prediction,
    replay_sgd32, replay_sgd64, try_install_ols,
};

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::ols::{
    FitConfig32, FitConfig64, LinearRegression, UnderflowPolicy, fit_f32, fit_f64,
};
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64};
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdSample, SgdSample64};

fn sgd_sample(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn sgd_sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn rls_sample(features: &[f32], target: f32) -> Observation<RlsSample> {
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

fn run_ols() -> Result<(), String> {
    let features = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0])];
    let targets = [1.0f32, 4.0];
    let indices = [0, 1];
    let candidate_c = fit_f32(&features, &targets, &indices, &FitConfig32::default())
        .map_err(|error| format!("OLS fit failed: {error:?}"))?;
    let features64 = [Vector64::from_slice(&[1.0]), Vector64::from_slice(&[2.0])];
    let targets64 = [1.0f64, 4.0];
    let candidate_a = fit_f64(&features64, &targets64, &indices, &FitConfig64::default())
        .map_err(|error| format!("OLS A fit failed: {error:?}"))?;
    let candidate_b = candidate_a
        .to_serving_f32(UnderflowPolicy::Reject)
        .map_err(|error| format!("OLS B conversion failed: {error:?}"))?;
    let tiny_targets = [1.0e-50f64, 2.0e-50];
    let tiny_candidate = fit_f64(
        &features64,
        &tiny_targets,
        &indices,
        &FitConfig64::default(),
    )
    .map_err(|error| format!("tiny OLS A fit failed: {error:?}"))?;
    if tiny_candidate
        .to_serving_f32(UnderflowPolicy::Reject)
        .is_ok()
    {
        return Err("OLS B underflow policy did not reject the conversion".to_string());
    }
    let underflow_retry = tiny_candidate
        .to_serving_f32(UnderflowPolicy::AllowLoss)
        .map_err(|error| format!("OLS conversion-only retry failed: {error:?}"))?;
    if underflow_retry
        .predict_checked(&Vector::from_slice(&[1.0]))
        .is_err()
    {
        return Err("OLS conversion-only retry produced an invalid predictor".to_string());
    }
    let c_probe = candidate_c
        .predictor()
        .predict_checked(&Vector::from_slice(&[1.0]))
        .map_err(|error| format!("OLS C probe failed: {error:?}"))?;
    if (c_probe - 1.0).abs() >= 1e-5 {
        return Err("native-f32 OLS candidate missed its declared probe tolerance".to_string());
    }
    let old = LinearRegression::new(Vector::from_slice(&[2.0]), 1.0)
        .map_err(|error| format!("old OLS predictor failed: {error:?}"))?;
    let mut active = ActiveOls {
        predictor: old,
        preprocessing: "active-transform",
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    let acceptance_features = [Vector64::from_slice(&[1.0]), Vector64::from_slice(&[2.0])];
    let acceptance_targets = [2.0f64, 4.0];
    let acceptance_candidate = fit_f64(
        &acceptance_features,
        &acceptance_targets,
        &indices,
        &FitConfig64::default(),
    )
    .map_err(|error| format!("rejected OLS fit failed: {error:?}"))?
    .predictor();
    let residual = (acceptance_candidate
        .predict_checked(&Vector64::from_slice(&[3.0]))
        .map_err(|error| format!("rejected OLS evaluation failed: {error:?}"))?
        - 7.0)
        .abs();
    if residual <= 1e-12 {
        return Err("finite fitted OLS candidate should be rejected".to_string());
    }
    let mut active_a = ActiveOls {
        predictor: seqvex::models::linear::ols::LinearRegression64::new(
            Vector64::from_slice(&[2.0]),
            1.0,
        )
        .map_err(|error| format!("active OLS A predictor failed: {error:?}"))?,
        preprocessing: "active-transform",
        installed_identity: 1,
        preprocessing_identity: 10,
    };
    let rejected = try_install_ols(
        &mut active_a,
        OlsCandidate {
            predictor: acceptance_candidate,
            preprocessing: "rejected-transform",
            candidate_identity: 3,
            preprocessing_identity: 10,
        },
        AcceptanceDecision::<&str>::Rejected,
        10,
    );
    if rejected != Err(InstallError::Rejected)
        || active_a.installed_identity != 1
        || active_a.preprocessing != "active-transform"
        || active_a.preprocessing_identity != 10
        || active_a
            .predictor
            .predict_checked(&Vector64::from_slice(&[1.0]))
            .unwrap()
            != 3.0
    {
        return Err("rejected OLS candidate changed the active pair".to_string());
    }
    let mut progress = OlsProgress {
        candidate_identity: None,
        installed_identity: active.installed_identity,
        calculated: 0,
        published: 0,
    };
    let accepted_residual = (candidate_b
        .predict_checked(&Vector::from_slice(&[3.0]))
        .map_err(|error| format!("accepted OLS evaluation failed: {error:?}"))?
        - 7.0)
        .abs() as f64;
    if accepted_residual > 1e-12 {
        return Err("accepted OLS candidate did not meet the declared 1e-12 predicate".to_string());
    }
    let candidate_pair = progress
        .record_candidate(Ok::<_, &str>(OlsCandidate {
            predictor: candidate_b.clone(),
            preprocessing: "candidate-transform",
            candidate_identity: 2,
            preprocessing_identity: 10,
        }))
        .map_err(|error| format!("OLS candidate tracking failed: {error}"))?;
    let installed = progress
        .record_install(try_install_ols(
            &mut active,
            candidate_pair,
            AcceptanceDecision::<&str>::Accepted,
            10,
        ))
        .map_err(|error| format!("OLS install failed: {error:?}"))?;
    let rows = [
        Observation::new(Vector::from_slice(&[1.0])),
        Observation::new(Vector::from_slice(&[2.0])),
        Observation::new(Vector::from_slice(&[3.0])),
    ];
    let grouped = active
        .predictor
        .predict_batch_checked(&rows)
        .map_err(|error| format!("OLS group prediction failed: {error:?}"))?;
    let checked = seqvex::models::linear::ols::CheckedLinearRegression::new(&active.predictor);
    let mut stream = StreamingExecutor::new(&checked, 0.0f32);
    let streamed: Vec<f32> = rows
        .iter()
        .map(|row| *stream.process_one(row).expect("checked stream prediction"))
        .collect();
    if grouped != streamed {
        return Err("OLS installed grouped and streaming outputs differ".to_string());
    }
    let probe = Vector::from_slice(&[1.0]);
    let installed_prediction = active
        .predictor
        .predict_checked(&probe)
        .map_err(|error| format!("OLS probe failed: {error:?}"))?;
    let converted_prediction = candidate_b
        .predict_checked(&probe)
        .map_err(|error| format!("OLS B probe failed: {error:?}"))?;
    if installed != 2 || (installed_prediction - 1.0).abs() >= 1e-5 {
        return Err("OLS installed probe was not distinguishable".to_string());
    }
    if (converted_prediction - 1.0).abs() >= 1e-5 {
        return Err("OLS B serving probe was not distinguishable".to_string());
    }
    let _ = OlsProgress {
        candidate_identity: Some(installed),
        installed_identity: active.installed_identity,
        calculated: 1,
        published: 0,
    };
    println!(
        "OLS A/B/C: underflow rejected/recovered by conversion only; acceptance rejection preserved identity 1; installed B identity {} preprocessing {} probe {:.7}; C probe {:.7}; A probe {:.7}; group/stream {}",
        active.installed_identity,
        active.preprocessing,
        installed_prediction,
        c_probe,
        converted_prediction,
        grouped.len()
    );
    Ok(())
}

fn run_sgd_and_rls() -> Result<(), String> {
    let rows32 = vec![
        sgd_sample(&[1.0, 0.0], 1.0),
        sgd_sample(&[0.0, 1.0], -1.0),
        sgd_sample(&[1.0, 1.0], 0.5),
    ];
    let sgd = Sgd::new(2, 0.125).map_err(|error| format!("SGD config failed: {error:?}"))?;
    let sgd_initial = sgd.initial_state();
    let sgd_single = sgd
        .update(&sgd_initial, &rows32[0])
        .map_err(|error| format!("SGD single update failed: {error:?}"))?;
    if replay_sgd32(&sgd, sgd_initial.clone(), &[], 0).is_err()
        || replay_sgd32(&sgd, sgd_initial.clone(), &rows32[..1], 1).is_err()
    {
        return Err("SGD C empty/singleton bounded path failed".to_string());
    }
    let sgd_state = rows32
        .iter()
        .try_fold(sgd_initial.clone(), |state, row| sgd.update(&state, row))
        .map_err(|error| format!("SGD learning failed: {error:?}"))?;
    let bounded = replay_sgd32(&sgd, sgd_initial.clone(), &rows32, 3)
        .map_err(|error| format!("SGD bounded replay failed: {error:?}"))?;
    if bounded != sgd_state {
        return Err("SGD bounded replay diverged".to_string());
    }
    match replay_sgd32(&sgd, sgd_initial.clone(), &rows32, 2) {
        Err(BatchFailure::TooLarge { state, .. }) => {
            if state != sgd_initial {
                return Err("SGD oversized rejection lost its state".to_string());
            }
        }
        other => return Err(format!("SGD oversize result was {other:?}")),
    }
    let mut bad32 = rows32.clone();
    bad32[1] = sgd_sample(&[0.0, 1.0], f32::NAN);
    match replay_sgd32(&sgd, sgd.initial_state(), &bad32, bad32.len()) {
        Err(BatchFailure::Transition {
            failed_index: 1,
            state,
            error: seqvex::models::linear::sgd::SgdError::NonFiniteInput,
            ..
        }) if state == sgd_single => {}
        other => return Err(format!("SGD C middle failure was {other:?}")),
    }
    let prediction = sgd
        .predict(&sgd_state, &Vector::from_slice(&[0.5, -0.5]))
        .map_err(|error| format!("SGD prediction failed: {error:?}"))?;
    let prediction_rows = vec![
        Observation::new(Vector::from_slice(&[0.5, -0.5])),
        Observation::new(Vector::from_slice(&[0.25, 0.25])),
    ];
    let collected = collect32(2, &prediction_rows, |row| sgd.predict(&sgd_state, row))
        .map_err(|error| format!("SGD collection failed: {error:?}"))?;
    if collected.first().copied() != Some(prediction) {
        return Err("SGD grouped prediction diverged".to_string());
    }

    let sgd64 = Sgd64::new(2, 0.125).map_err(|error| format!("SGD64 config failed: {error:?}"))?;
    let rows64 = [
        sgd_sample64(&[1.0, 0.0], 1.0),
        sgd_sample64(&[0.0, 1.0], -1.0),
        sgd_sample64(&[1.0, 1.0], 0.5),
    ];
    let sgd64_initial = sgd64.initial_state();
    let sgd64_single = sgd64
        .update(&sgd64_initial, &rows64[0])
        .map_err(|error| format!("SGD64 single update failed: {error:?}"))?;
    if replay_sgd64(&sgd64, sgd64_initial.clone(), &[], 0).is_err()
        || replay_sgd64(&sgd64, sgd64_initial.clone(), &rows64[..1], 1).is_err()
    {
        return Err("SGD A empty/singleton bounded path failed".to_string());
    }
    let mut tracked = Progress::default();
    let mut sgd64_state = sgd64_initial.clone();
    for row in &rows64 {
        let update = sgd64
            .update(&sgd64_state, row)
            .map_err(|error| format!("SGD64 learning failed: {error:?}"))?;
        tracked
            .commit_update(&mut sgd64_state, Ok::<_, ()>(update))
            .map_err(|_| "SGD64 progress commit failed")?;
    }
    let bounded64 = replay_sgd64(&sgd64, sgd64_initial.clone(), &rows64, rows64.len())
        .map_err(|error| format!("SGD64 bounded replay failed: {error:?}"))?;
    if bounded64 != sgd64_state {
        return Err("SGD64 bounded replay diverged".to_string());
    }
    if !matches!(
        replay_sgd64(&sgd64, sgd64_initial.clone(), &rows64, rows64.len() - 1),
        Err(BatchFailure::TooLarge { .. })
    ) {
        return Err("SGD64 oversize group was accepted".to_string());
    }
    let a_rows = [
        Observation::new(Vector64::from_slice(&[0.5, -0.5])),
        Observation::new(Vector64::from_slice(&[0.25, 0.25])),
    ];
    let a_single: Vec<f64> = a_rows
        .iter()
        .map(|row| sgd64.predict(&sgd64_state, row.value()).unwrap())
        .collect();
    let a_grouped = collect64(2, &a_rows, |row| sgd64.predict(&sgd64_state, row))
        .map_err(|error| format!("SGD A group prediction failed: {error:?}"))?;
    if a_single != a_grouped {
        return Err("SGD A grouped prediction diverged".to_string());
    }
    let serving = sgd64
        .to_serving_f32(&sgd64_state)
        .map_err(|error| format!("SGD B conversion failed: {error:?}"))?;
    let b_prediction = serving
        .predict(&Vector::from_slice(&[0.5, -0.5]))
        .map_err(|error| format!("SGD B prediction failed: {error:?}"))?;
    let b_group = collect32(2, &prediction_rows, |features| serving.predict(features))
        .map_err(|error| format!("SGD B grouped prediction failed: {error:?}"))?;
    let b_single: Vec<f32> = prediction_rows
        .iter()
        .map(|row| serving.predict(row.value()).unwrap())
        .collect();
    if b_group != b_single {
        return Err("SGD B grouped prediction diverged".to_string());
    }
    let mut bad64 = rows64.clone();
    bad64[1] = sgd_sample64(&[0.0, 1.0], f64::NAN);
    match replay_sgd64(&sgd64, sgd64.initial_state(), &bad64, bad64.len()) {
        Err(BatchFailure::Transition {
            failed_index: 1,
            state,
            error: seqvex::models::linear::sgd::SgdError64::NonFiniteInput,
            ..
        }) if state == sgd64_single => {}
        other => return Err(format!("SGD A middle failure was {other:?}")),
    }
    let huge_state = sgd64.initial_state();
    let huge_learned = sgd64
        .update(&huge_state, &sgd_sample64(&[1.0, 0.0], 1.0e40))
        .map_err(|error| format!("SGD huge-state update failed: {error:?}"))?;
    if sgd64.to_serving_f32(&huge_learned).is_ok() || sgd64.to_serving_f32(&huge_learned).is_ok() {
        return Err("SGD B conversion overflow was not retained on retry".to_string());
    }
    let mut conversion_progress = Progress::default();
    let mut committed_huge = huge_state;
    let actual_update = sgd64
        .update(&committed_huge, &sgd_sample64(&[1.0, 0.0], 1.0e40))
        .map_err(|error| format!("tracked huge SGD update failed: {error:?}"))?;
    conversion_progress
        .commit_update(&mut committed_huge, Ok::<_, ()>(actual_update))
        .map_err(|_| "tracked huge update commit failed")?;
    let mut conversion_active = (
        sgd64
            .to_serving_f32(&sgd64_initial)
            .map_err(|error| format!("old B snapshot conversion failed: {error:?}"))?,
        ServingRecord {
            identity: 41,
            served_through: 0,
        },
    );
    let old_active = conversion_active.clone();
    let failed_conversion = sgd64.to_serving_f32(&committed_huge);
    if failed_conversion.is_ok()
        || conversion_progress
            .install_serving(&mut conversion_active, failed_conversion, 42)
            .is_ok()
        || conversion_active != old_active
        || conversion_progress.learning != 1
        || conversion_progress.served != 0
    {
        return Err("B conversion failure changed committed or serving progress".to_string());
    }
    let mut active_b = (
        sgd64
            .to_serving_f32(&sgd64_initial)
            .map_err(|error| format!("initial SGD B conversion failed: {error:?}"))?,
        ServingRecord {
            identity: 0,
            served_through: 0,
        },
    );
    tracked
        .install_serving(&mut active_b, Ok::<_, ()>(serving), 1)
        .map_err(|_| "initial SGD B installation failed")?;
    let rejected_conversion = sgd64
        .to_serving_f32(&sgd64_state)
        .map_err(|_| "conversion failed")
        .and_then(|snapshot| {
            if snapshot.predict(&Vector::from_slice(&[0.5, -0.5])).is_ok() {
                Err("caller rejected candidate")
            } else {
                Ok(snapshot)
            }
        });
    if tracked
        .install_serving(&mut active_b, rejected_conversion, 2)
        .is_ok()
        || active_b.1.identity != 1
        || tracked.served != 3
    {
        return Err("rejected B candidate replaced the active snapshot".to_string());
    }
    let converted_retry = sgd64
        .to_serving_f32(&sgd64_state)
        .map_err(|error| format!("B conversion-only retry failed: {error:?}"))?;
    tracked
        .install_serving(&mut active_b, Ok::<_, ()>(converted_retry), 2)
        .map_err(|_| "B conversion-only install failed")?;
    if tracked.learning != 3 || tracked.served != 3 || active_b.1.identity != 2 {
        return Err("B conversion retry changed learning progress".to_string());
    }

    let rls_rows32 = vec![
        rls_sample(&[1.0, 0.0], 1.0),
        rls_sample(&[0.0, 1.0], -1.0),
        rls_sample(&[1.0, 1.0], 0.5),
    ];
    let rls = Rls::new(2, 0.9, 1.0).map_err(|error| format!("RLS config failed: {error:?}"))?;
    let rls_initial = rls.initial_state();
    let rls_single = rls
        .update(&rls_initial, &rls_rows32[0])
        .map_err(|error| format!("RLS C single update failed: {error:?}"))?;
    if rls
        .process_batch_reference(rls_initial.clone(), &[], 0)
        .is_err()
        || rls
            .process_batch_reference(rls_initial.clone(), &rls_rows32[..1], 1)
            .is_err()
    {
        return Err("RLS C empty/singleton bounded path failed".to_string());
    }
    let rls_state = rls
        .process_batch_reference(rls_initial.clone(), &rls_rows32, rls_rows32.len())
        .map_err(|error| format!("RLS bounded learning failed: {error:?}"))?;
    if rls
        .process_batch_reference(rls_initial.clone(), &rls_rows32, 2)
        .is_ok()
    {
        return Err("RLS C oversize group was accepted".to_string());
    }
    let mut bad_rls32 = rls_rows32.clone();
    bad_rls32[1] = rls_sample(&[0.0, 1.0], f32::NAN);
    match rls.process_batch_reference(rls_initial, &bad_rls32, 3) {
        Err(seqvex::models::linear::rls::RlsBatchError::Transition {
            failed_index: 1,
            state,
            ..
        }) if state == rls_single => {}
        other => return Err(format!("RLS C middle failure was {other:?}")),
    }
    let rls_prediction = rls
        .predict_checked(&rls_state, &Vector::from_slice(&[0.5, -0.5]))
        .map_err(|error| format!("RLS prediction failed: {error:?}"))?;
    let rls_c_rows = [
        Observation::new(Vector::from_slice(&[0.5, -0.5])),
        Observation::new(Vector::from_slice(&[0.25, 0.25])),
    ];
    let rls_c_single: Vec<f32> = rls_c_rows
        .iter()
        .map(|row| rls.predict_checked(&rls_state, row.value()).unwrap())
        .collect();
    if collect32(2, &rls_c_rows, |row| rls.predict_checked(&rls_state, row))
        .map_err(|error| format!("RLS C grouped prediction failed: {error:?}"))?
        != rls_c_single
    {
        return Err("RLS C grouped prediction diverged".to_string());
    }
    let rls64 =
        Rls64::new(2, 0.9, 1.0).map_err(|error| format!("RLS64 config failed: {error:?}"))?;
    let rls_rows64 = [
        rls_sample64(&[1.0, 0.0], 1.0),
        rls_sample64(&[0.0, 1.0], -1.0),
        rls_sample64(&[1.0, 1.0], 0.5),
    ];
    let rls64_initial = rls64.initial_state();
    let rls64_single = rls64
        .update(&rls64_initial, &rls_rows64[0])
        .map_err(|error| format!("RLS A single update failed: {error:?}"))?;
    if rls64
        .process_batch_reference(rls64_initial.clone(), &[], 0)
        .is_err()
        || rls64
            .process_batch_reference(rls64_initial.clone(), &rls_rows64[..1], 1)
            .is_err()
    {
        return Err("RLS A empty/singleton bounded path failed".to_string());
    }
    let rls64_state = rls64
        .process_batch_reference(rls64_initial.clone(), &rls_rows64, 3)
        .map_err(|error| format!("RLS A bounded learning failed: {error:?}"))?;
    if rls64
        .process_batch_reference(rls64_initial.clone(), &rls_rows64, 2)
        .is_ok()
    {
        return Err("RLS A oversize group was accepted".to_string());
    }
    let mut bad_rls64 = rls_rows64.clone();
    bad_rls64[1] = rls_sample64(&[0.0, 1.0], f64::NAN);
    match rls64.process_batch_reference(rls64_initial, &bad_rls64, 3) {
        Err(seqvex::models::linear::rls::RlsBatchError64::Transition {
            failed_index: 1,
            state,
            ..
        }) if state == rls64_single => {}
        other => return Err(format!("RLS A middle failure was {other:?}")),
    }
    let rls_a_rows = [
        Observation::new(Vector64::from_slice(&[0.5, -0.5])),
        Observation::new(Vector64::from_slice(&[0.25, 0.25])),
    ];
    let rls_a_single: Vec<f64> = rls_a_rows
        .iter()
        .map(|row| rls64.predict_checked(&rls64_state, row.value()).unwrap())
        .collect();
    if collect64(2, &rls_a_rows, |row| {
        rls64.predict_checked(&rls64_state, row)
    })
    .map_err(|error| format!("RLS A grouped prediction failed: {error:?}"))?
        != rls_a_single
    {
        return Err("RLS A grouped prediction diverged".to_string());
    }
    let rls_serving = rls64
        .to_serving_f32(&rls64_state)
        .map_err(|error| format!("RLS B conversion failed: {error:?}"))?;
    let rls_b_rows = [
        Observation::new(Vector::from_slice(&[0.5, -0.5])),
        Observation::new(Vector::from_slice(&[0.25, 0.25])),
    ];
    let rls_b_single: Vec<f32> = rls_b_rows
        .iter()
        .map(|row| rls_serving.predict_checked(row.value()).unwrap())
        .collect();
    if collect32(2, &rls_b_rows, |row| rls_serving.predict_checked(row))
        .map_err(|error| format!("RLS B grouped prediction failed: {error:?}"))?
        != rls_b_single
    {
        return Err("RLS B grouped prediction diverged".to_string());
    }
    let single_c_prediction = sgd
        .predict(&sgd_single, &Vector::from_slice(&[0.5, -0.5]))
        .map_err(|error| format!("SGD single prediction failed: {error:?}"))?;
    println!(
        "SGD/RLS A/B/C: C prediction {:.7}; SGD B {:.7}; RLS C {:.7}; RLS B ready; single C {:.7}",
        prediction, b_prediction, rls_prediction, single_c_prediction
    );
    Ok(())
}

fn run_temporal_and_publication() -> Result<(), String> {
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

fn run() -> Result<(), String> {
    run_ols()?;
    run_sgd_and_rls()?;
    run_temporal_and_publication()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("linear_execution failed: {error}");
        std::process::exit(1);
    }
    println!("linear_execution completed");
}

#[cfg(test)]
mod tests {
    #[test]
    fn deterministic_example_runs() {
        super::run().expect("example workflow succeeds");
    }
}
