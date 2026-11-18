use crate::support::{
    AcceptanceDecision, ActiveOls, InstallError, OlsCandidate, OlsProgress, try_install_ols,
};

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    FitConfig32, FitConfig64, LinearRegression, UnderflowPolicy, fit_f32, fit_f64,
};

pub(super) fn run_ols() -> Result<(), String> {
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
