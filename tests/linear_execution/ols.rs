use crate::support::{
    AcceptanceDecision, ActiveOls, InstallError, OlsCandidate, OlsProgress, PredictionRecord,
    try_install_ols,
};

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Matrix, Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, FitConfig32, FitConfig64, LinearRegression,
    LinearRegression64, UnderflowPolicy, fit_f32, fit_f64,
};
use seqvex::models::linear::rls::{Rls, RlsError};
use seqvex::models::linear::sgd::{Sgd, SgdError};

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
