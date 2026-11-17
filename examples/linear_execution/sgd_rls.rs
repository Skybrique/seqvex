use crate::support::{Progress, ServingRecord, collect32, collect64};

use crate::{rls_sample, rls_sample64, sgd_sample, sgd_sample64};

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, Rls64};
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdUpdateBatchError, SgdUpdateBatchError64};

pub(super) fn run_sgd_and_rls() -> Result<(), String> {
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
    if sgd.update_batch(sgd_initial.clone(), &[], 0).is_err()
        || sgd
            .update_batch(sgd_initial.clone(), &rows32[..1], 1)
            .is_err()
    {
        return Err("SGD C empty/singleton bounded path failed".to_string());
    }
    let sgd_state = rows32
        .iter()
        .try_fold(sgd_initial.clone(), |state, row| sgd.update(&state, row))
        .map_err(|error| format!("SGD learning failed: {error:?}"))?;
    let bounded = sgd
        .update_batch(sgd_initial.clone(), &rows32, 3)
        .map_err(|error| format!("SGD bounded update failed: {error:?}"))?;
    if bounded != sgd_state {
        return Err("SGD bounded replay diverged".to_string());
    }
    match sgd.update_batch(sgd_initial.clone(), &rows32, 2) {
        Err(SgdUpdateBatchError::TooLarge { state, .. }) => {
            if state != sgd_initial {
                return Err("SGD oversized rejection lost its state".to_string());
            }
        }
        other => return Err(format!("SGD oversize result was {other:?}")),
    }
    let mut bad32 = rows32.clone();
    bad32[1] = sgd_sample(&[0.0, 1.0], f32::NAN);
    match sgd.update_batch(sgd.initial_state(), &bad32, bad32.len()) {
        Err(SgdUpdateBatchError::Transition {
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
    let collected = sgd
        .predict_batch(&sgd_state, &prediction_rows, prediction_rows.len())
        .map_err(|error| format!("SGD grouped prediction failed: {error:?}"))?;
    if collected.first().map(|value| value.to_bits()) != Some(prediction.to_bits()) {
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
    if sgd64.update_batch(sgd64_initial.clone(), &[], 0).is_err()
        || sgd64
            .update_batch(sgd64_initial.clone(), &rows64[..1], 1)
            .is_err()
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
    let bounded64 = sgd64
        .update_batch(sgd64_initial.clone(), &rows64, rows64.len())
        .map_err(|error| format!("SGD64 bounded update failed: {error:?}"))?;
    if bounded64 != sgd64_state {
        return Err("SGD64 bounded replay diverged".to_string());
    }
    if !matches!(
        sgd64.update_batch(sgd64_initial.clone(), &rows64, rows64.len() - 1),
        Err(SgdUpdateBatchError64::TooLarge { .. })
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
    let a_grouped = sgd64
        .predict_batch(&sgd64_state, &a_rows, a_rows.len())
        .map_err(|error| format!("SGD A grouped prediction failed: {error:?}"))?;
    if a_single
        .iter()
        .zip(&a_grouped)
        .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err("SGD A grouped prediction diverged".to_string());
    }
    let serving = sgd64
        .to_serving_f32(&sgd64_state)
        .map_err(|error| format!("SGD B conversion failed: {error:?}"))?;
    let b_prediction = serving
        .predict(&Vector::from_slice(&[0.5, -0.5]))
        .map_err(|error| format!("SGD B prediction failed: {error:?}"))?;
    let b_group = serving
        .predict_batch(&prediction_rows, prediction_rows.len())
        .map_err(|error| format!("SGD B grouped prediction failed: {error:?}"))?;
    let b_single: Vec<f32> = prediction_rows
        .iter()
        .map(|row| serving.predict(row.value()).unwrap())
        .collect();
    if b_group
        .iter()
        .zip(&b_single)
        .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err("SGD B grouped prediction diverged".to_string());
    }
    let mut bad64 = rows64.clone();
    bad64[1] = sgd_sample64(&[0.0, 1.0], f64::NAN);
    match sgd64.update_batch(sgd64.initial_state(), &bad64, bad64.len()) {
        Err(SgdUpdateBatchError64::Transition {
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
