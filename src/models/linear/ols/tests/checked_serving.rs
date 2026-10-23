//! Checked-serving behavior: opt-in prediction that rejects non-finite outputs,
//! plus the borrowing `CheckedLinearRegression` streaming wrapper.
//!
//! The bare `predict` / `predict_batch` paths are asserted to keep their
//! existing behavior on the same cases.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_batch;
use seqvex::models::linear::ols::{
    AcceptanceCheckError, CheckedLinearRegression, FitConfig, LinearRegression, RegressionError,
    StoredDiagnostic, fit,
};

fn model(weights: &[f32], bias: f32) -> LinearRegression {
    LinearRegression::new(Vector::from_slice(weights), bias).unwrap()
}

fn observation(features: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(features))
}

fn assert_bits_eq(left: f32, right: f32) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left} vs {right}");
}

#[test]
fn finite_single_prediction_matches_bare_bits() {
    let model = model(&[1.0, 2.0], 0.5);
    let features = Vector::from_slice(&[3.0, 4.0]);
    let bare = model.predict(&features).unwrap();
    let checked = model.predict_checked(&features).unwrap();
    assert_bits_eq(bare, checked);
    assert_eq!(checked, 11.5);
}

#[test]
fn finite_group_prediction_matches_bare_bits() {
    let model = model(&[2.0], 1.0);
    let batch = [
        observation(&[1.0]),
        observation(&[-3.0]),
        observation(&[0.0]),
    ];
    let bare = model.predict_batch(&batch).unwrap();
    let checked = model.predict_batch_checked(&batch).unwrap();
    assert_eq!(bare.len(), checked.len());
    for (left, right) in bare.iter().zip(&checked) {
        assert_bits_eq(*left, *right);
    }
}

#[test]
fn input_errors_are_retained() {
    let model = model(&[2.0], 1.0);
    assert_eq!(
        model.predict_checked(&Vector::from_slice(&[1.0, 2.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        model.predict_checked(&Vector::from_slice(&[f32::NAN])),
        Err(RegressionError::NonFiniteInput)
    );
    let batch = [observation(&[1.0]), observation(&[f32::INFINITY])];
    assert_eq!(
        model.predict_batch_checked(&batch),
        Err(RegressionError::NonFiniteInput)
    );
}

#[test]
fn product_overflow_is_rejected_when_checked() {
    let model = model(&[1.0e20], 0.0);
    let features = Vector::from_slice(&[1.0e20]);
    // Bare path is unchanged: it returns the non-finite value.
    assert!(model.predict(&features).unwrap().is_infinite());
    assert_eq!(
        model.predict_checked(&features),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );
}

#[test]
fn bias_addition_overflow_is_rejected_when_checked() {
    // Finite product (3.2e38) plus a finite bias overflows f32::MAX.
    let model = model(&[1.6e38], 2.0e38);
    let features = Vector::from_slice(&[2.0]);
    assert!(model.predict(&features).unwrap().is_infinite());
    assert_eq!(
        model.predict_checked(&features),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );
}

#[test]
fn nan_from_arithmetic_is_rejected_when_checked() {
    // +inf and -inf products summed produce NaN.
    let model = model(&[1.0e20, 1.0e20], 0.0);
    let features = Vector::from_slice(&[1.0e20, -1.0e20]);
    assert!(model.predict(&features).unwrap().is_nan());
    assert_eq!(
        model.predict_checked(&features),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );
}

#[test]
fn group_reports_first_failing_position() {
    let model = model(&[2.0], 1.0);
    let batch = [
        observation(&[1.0]),
        observation(&[2.0e38]),
        observation(&[5.0]),
    ];
    let error = model.predict_batch_checked(&batch).unwrap_err();
    assert_eq!(
        error,
        RegressionError::NonFiniteOutput {
            observation: Some(1)
        }
    );
    assert!(error.to_string().contains("observation 1"));

    let first_fails = [observation(&[2.0e38]), observation(&[5.0])];
    let error = model.predict_batch_checked(&first_fails).unwrap_err();
    assert_eq!(
        error,
        RegressionError::NonFiniteOutput {
            observation: Some(0)
        }
    );
    assert!(error.to_string().contains("observation 0"));
}

#[test]
fn empty_group_succeeds_with_empty_output() {
    let model = model(&[2.0], 1.0);
    assert_eq!(model.predict_batch_checked(&[]), Ok(Vec::new()));
}

#[test]
fn earlier_output_failure_is_not_masked_by_later_invalid_input() {
    let model = model(&[2.0], 1.0);
    let batch = [
        observation(&[1.0]),      // finite
        observation(&[2.0e38]),   // overflow -> NonFiniteOutput at index 1
        observation(&[1.0, 2.0]), // would be DimensionMismatch at index 2
    ];
    assert_eq!(
        model.predict_batch_checked(&batch),
        Err(RegressionError::NonFiniteOutput {
            observation: Some(1)
        })
    );
}

#[test]
fn checked_streaming_matches_bare_streaming_bits() {
    let model = model(&[2.0], 1.0);
    let observations = [
        observation(&[1.0]),
        observation(&[4.0]),
        observation(&[-2.0]),
    ];

    let mut bare = StreamingExecutor::new(&model, 0.0f32);
    let bare_values: Vec<f32> = observations
        .iter()
        .map(|obs| *bare.process_one(obs).unwrap())
        .collect();

    let checked = CheckedLinearRegression::new(&model);
    let mut checked_executor = StreamingExecutor::new(&checked, 0.0f32);
    let checked_values: Vec<f32> = observations
        .iter()
        .map(|obs| *checked_executor.process_one(obs).unwrap())
        .collect();

    assert_eq!(bare_values.len(), checked_values.len());
    for (left, right) in bare_values.iter().zip(&checked_values) {
        assert_bits_eq(*left, *right);
    }
}

#[test]
fn failing_observation_uses_existing_executor_error_path_and_preserves_state() {
    let model = model(&[2.0], 1.0);
    let checked = CheckedLinearRegression::new(&model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);

    assert_bits_eq(*executor.process_one(&observation(&[3.0])).unwrap(), 7.0);
    let error = executor.process_one(&observation(&[2.0e38])).unwrap_err();
    assert_eq!(
        error,
        RegressionError::NonFiniteOutput { observation: None }
    );
    // The last committed valid state is preserved.
    assert_bits_eq(*executor.state(), 7.0);
}

#[test]
fn process_stream_reports_failure_and_continues() {
    let model = model(&[2.0], 1.0);
    let checked = CheckedLinearRegression::new(&model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);

    let mut failures: Vec<String> = Vec::new();
    let observations = vec![
        observation(&[1.0]),    // 3.0
        observation(&[2.0e38]), // failure
        observation(&[5.0]),    // 11.0
    ];
    let final_state = *executor.process_stream(observations, |error| {
        failures.push(format!("{error:?}"));
    });

    assert_eq!(failures.len(), 1);
    assert!(failures[0].contains("NonFiniteOutput"));
    assert_bits_eq(final_state, 11.0);
}

#[test]
fn grouped_process_batch_stops_at_first_failure() {
    let model = model(&[2.0], 1.0);
    let checked = CheckedLinearRegression::new(&model);
    let observations = vec![
        observation(&[3.0]),    // 7.0
        observation(&[2.0e38]), // failure
        observation(&[5.0]),    // not reached
    ];
    let failure = process_batch(&checked, 0.0f32, observations).unwrap_err();
    assert_eq!(
        failure.error,
        RegressionError::NonFiniteOutput { observation: None }
    );
    // The last committed valid state before the failure is returned.
    assert_bits_eq(failure.state, 7.0);
}

#[test]
fn acceptance_rejection_preserves_candidate_and_active_predictor() {
    let candidate_features = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0])];
    let candidate_targets = [f32::MAX, f32::MAX];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let candidate = fit(&candidate_features, &candidate_targets, &[0, 1], &config)
        .expect("legacy fit succeeds");
    let weights_before = candidate.weights().clone();
    let bias_before = candidate.intercept();
    let diagnostics_before = candidate.diagnostics().clone();

    let active = model(&[2.0], 1.0);
    let active_row = Vector::from_slice(&[3.0]);
    let active_before = active.predict_checked(&active_row).unwrap();
    let error = candidate
        .check_acceptance(&candidate_features, &candidate_targets, &[0])
        .unwrap_err();

    assert_eq!(
        error,
        AcceptanceCheckError::NonFiniteStoredDiagnostic {
            diagnostic: StoredDiagnostic::RssF32Predictor,
        }
    );
    assert_eq!(candidate.weights(), &weights_before);
    assert_eq!(candidate.intercept(), bias_before);
    assert_eq!(candidate.diagnostics(), &diagnostics_before);
    assert_bits_eq(active.predict_checked(&active_row).unwrap(), active_before);
}

#[test]
fn admission_and_candidate_verification_precede_explicit_install() {
    let features = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0])];
    let targets = [1.0f32, 4.0];
    let indices = [0, 1];
    let candidate = fit(&features, &targets, &indices, &FitConfig::default()).expect("fit");

    let old_active = model(&[2.0], 1.0);
    let active_row = Vector::from_slice(&[3.0]);
    let old_active_prediction = old_active.predict_checked(&active_row).unwrap();
    let mut active = Some(old_active);

    candidate
        .check_acceptance(&features, &targets, &indices)
        .expect("candidate admitted before conversion");
    let candidate_predictor = candidate.into_predictor();
    {
        let checked = CheckedLinearRegression::new(&candidate_predictor);
        let mut executor = StreamingExecutor::new(&checked, 0.0f32);
        for (index, row) in features.iter().enumerate() {
            let direct = candidate_predictor.predict_checked(row).unwrap();
            let streamed = *executor
                .process_one(&observation(row.as_slice()))
                .expect("candidate executor verification");
            assert_bits_eq(direct, streamed);
            assert_bits_eq(direct, targets[index]);
        }
    }

    assert_bits_eq(
        active
            .as_ref()
            .unwrap()
            .predict_checked(&active_row)
            .unwrap(),
        old_active_prediction,
    );
    active = Some(candidate_predictor);
    let installed = active.as_ref().expect("explicitly installed candidate");
    assert_bits_eq(installed.predict_checked(&active_row).unwrap(), 7.0);
}
