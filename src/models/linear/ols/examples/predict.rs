//! Fitted OLS prediction workflow.
//!
//! Demonstrates the implemented OLS path end to end: a named, self-contained
//! labelled dataset → existing `fit` → validated learned parameters and
//! diagnostics → `FitResult::into_predictor` → checked single, streaming and
//! grouped predictions from that same learned model.
//!
//! Dataset `synthetic-exam-score-v1` has two features (`study_hours`,
//! `absences`) and one target (`score`). Training rows lie exactly on
//! `score = 3.0 * study_hours - 2.0 * absences + 40.0`; evaluation targets are
//! that formula plus declared residuals `[+0.5, -1.0, 0.0, +0.5]`. The row
//! order is not time, so this is not temporal validation.
//!
//! The example also preserves the documented bare-prediction `f32` overflow
//! envelope and the checked-error boundary. It does not add a solver,
//! precision policy, artifact, shared metric, or shared error abstraction.
//!
//! Run with:
//!
//! ```text
//! cargo run --example ols_predict
//! ```
//!
//! Expected results:
//!
//! ```text
//! dataset = synthetic-exam-score-v1
//! training rows = 6, evaluation rows = 4
//! learned weights = [3.0, -2.0]
//! learned intercept = 40.0
//! evaluation predictions = [41.0, 42.0, 50.0, 34.0]
//! MAE = 0.5
//! R² = 0.9876288659793815
//! R² = undefined (constant targets)
//! shifted intercept = 50.0
//! shifted predictions = [51.0, 52.0, 60.0, 44.0]
//! ```
//!
//! Any unexpected fit, learned parameter, prediction, metric, error or
//! compatibility result returns `Err`; the process exits with
//! `ExitCode::FAILURE`.

use std::process::ExitCode;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, FitConfig, FitDiagnostics, LinearRegression, RegressionError, fit,
};

const DATASET_NAME: &str = "synthetic-exam-score-v1";
const FEATURE_NAMES: [&str; 2] = ["study_hours", "absences"];
const TRAIN_INDICES: [usize; 6] = [0, 1, 2, 3, 4, 5];
const EVAL_INDICES: [usize; 4] = [6, 7, 8, 9];
const TRUE_WEIGHTS: [f32; 2] = [3.0, -2.0];
const TRUE_BIAS: f32 = 40.0;
const TOLERANCE: f32 = 1e-5;
const METRIC_TOLERANCE: f64 = 1e-5;
const RSS_TOLERANCE: f64 = 1e-6;
const TARGET_SHIFT: f32 = 10.0;

const EXPECTED_EVAL_PREDICTIONS: [f32; 4] = [41.0, 42.0, 50.0, 34.0];
const EXPECTED_MAE: f64 = 0.5;
const EXPECTED_R2: f64 = 0.987_628_865_979_381_4;
const EXPECTED_SHIFTED_INTERCEPT: f32 = 50.0;
const EXPECTED_SHIFTED_PREDICTIONS: [f32; 4] = [51.0, 52.0, 60.0, 44.0];

const OVERFLOW_COMPAT_WEIGHTS: [f32; 1] = [1.0e20];
const OVERFLOW_COMPAT_BIAS: f32 = 0.0;
const OVERFLOW_COMPAT_FEATURES: [[f32; 1]; 2] = [[1.0], [1.0e20]];

#[derive(Clone, Copy)]
struct LabelledRow {
    features: [f32; 2],
    target: f32,
}

const ROWS: [LabelledRow; 10] = [
    LabelledRow {
        features: [0.0, 0.0],
        target: 40.0,
    },
    LabelledRow {
        features: [1.0, 0.0],
        target: 43.0,
    },
    LabelledRow {
        features: [0.0, 1.0],
        target: 38.0,
    },
    LabelledRow {
        features: [2.0, 1.0],
        target: 44.0,
    },
    LabelledRow {
        features: [1.0, 2.0],
        target: 39.0,
    },
    LabelledRow {
        features: [3.0, 1.0],
        target: 47.0,
    },
    LabelledRow {
        features: [1.0, 1.0],
        target: 41.5,
    },
    LabelledRow {
        features: [2.0, 2.0],
        target: 41.0,
    },
    LabelledRow {
        features: [4.0, 1.0],
        target: 50.0,
    },
    LabelledRow {
        features: [0.0, 3.0],
        target: 34.5,
    },
];

fn observation(features: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(features))
}

fn check_close(label: &str, actual: f32, expected: f32, tolerance: f32) -> Result<(), String> {
    let difference = (actual - expected).abs();
    if difference <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{label}: expected {expected}, got {actual} (difference {difference}, tolerance {tolerance})"
        ))
    }
}

fn check_close_f64(label: &str, actual: f64, expected: f64, tolerance: f64) -> Result<(), String> {
    let difference = (actual - expected).abs();
    if difference <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{label}: expected {expected}, got {actual} (difference {difference}, tolerance {tolerance})"
        ))
    }
}

fn check_vec_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: expected {} values, got {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        check_close(&format!("{label}[{index}]"), actual, expected, tolerance)?;
    }
    Ok(())
}

fn check_vec_bits(label: &str, actual: &[f32], expected: &[f32]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: expected {} values, got {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(format!(
                "{label}[{index}]: expected bits {:08x}, got {:08x}",
                expected.to_bits(),
                actual.to_bits()
            ));
        }
    }
    Ok(())
}

fn mean_absolute_error(actual: &[f32], predicted: &[f32]) -> Result<f64, String> {
    if actual.is_empty() || actual.len() != predicted.len() {
        return Err("MAE requires non-empty slices of equal length".into());
    }
    if !actual
        .iter()
        .chain(predicted.iter())
        .all(|value| value.is_finite())
    {
        return Err("MAE received a non-finite value".into());
    }
    let total = actual
        .iter()
        .zip(predicted)
        .map(|(&actual, &predicted)| (f64::from(actual) - f64::from(predicted)).abs())
        .sum::<f64>();
    let value = total / actual.len() as f64;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("MAE produced a non-finite result".into())
    }
}

fn r_squared(actual: &[f32], predicted: &[f32]) -> Result<Option<f64>, String> {
    if actual.is_empty() || actual.len() != predicted.len() {
        return Err("R² requires non-empty slices of equal length".into());
    }
    if !actual
        .iter()
        .chain(predicted.iter())
        .all(|value| value.is_finite())
    {
        return Err("R² received a non-finite value".into());
    }
    let mean = actual.iter().map(|&value| f64::from(value)).sum::<f64>() / actual.len() as f64;
    let total = actual
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>();
    if !total.is_finite() {
        return Err("R² produced a non-finite total deviation".into());
    }
    if total == 0.0 {
        return Ok(None);
    }
    let residual = actual
        .iter()
        .zip(predicted)
        .map(|(&actual, &predicted)| (f64::from(actual) - f64::from(predicted)).powi(2))
        .sum::<f64>();
    let value = 1.0 - residual / total;
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err("R² produced a non-finite result".into())
    }
}

fn validate_diagnostics(diagnostics: &FitDiagnostics) -> Result<(), String> {
    if diagnostics.rows != TRAIN_INDICES.len() {
        return Err(format!(
            "fit diagnostics rows: expected {}, got {}",
            TRAIN_INDICES.len(),
            diagnostics.rows
        ));
    }
    if diagnostics.features != FEATURE_NAMES.len() {
        return Err(format!(
            "fit diagnostics features: expected {}, got {}",
            FEATURE_NAMES.len(),
            diagnostics.features
        ));
    }
    if !diagnostics.intercept {
        return Err("fit diagnostics: expected an intercept".into());
    }
    if diagnostics.rank != diagnostics.required_rank {
        return Err(format!(
            "fit diagnostics rank: expected {}, got {}",
            diagnostics.required_rank, diagnostics.rank
        ));
    }
    if !diagnostics.rank_threshold.is_finite() || diagnostics.rank_threshold < 0.0 {
        return Err(format!(
            "fit diagnostics rank threshold is invalid: {}",
            diagnostics.rank_threshold
        ));
    }
    if !diagnostics.rss_f64_reference.is_finite() || diagnostics.rss_f64_reference < 0.0 {
        return Err(format!(
            "fit diagnostics f64 RSS is invalid: {}",
            diagnostics.rss_f64_reference
        ));
    }
    if !diagnostics.rss_f32_predictor.is_finite() || diagnostics.rss_f32_predictor < 0.0 {
        return Err(format!(
            "fit diagnostics f32 RSS is invalid: {}",
            diagnostics.rss_f32_predictor
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let features: Vec<Vector> = ROWS
        .iter()
        .map(|row| Vector::from_slice(&row.features))
        .collect();
    let targets: Vec<f32> = ROWS.iter().map(|row| row.target).collect();

    let result = fit(&features, &targets, &TRAIN_INDICES, &FitConfig::default())
        .map_err(|error| format!("fit failed: {error:?}"))?;
    let diagnostics = result.diagnostics().clone();
    validate_diagnostics(&diagnostics)?;

    let learned_weights = result.weights().clone();
    let learned_intercept = result.intercept();
    if !learned_intercept.is_finite()
        || !learned_weights
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    {
        return Err("fit produced a non-finite learned parameter".into());
    }
    for (index, (&actual, &expected)) in learned_weights
        .as_slice()
        .iter()
        .zip(TRUE_WEIGHTS.iter())
        .enumerate()
    {
        check_close(
            &format!("learned weight[{index}]"),
            actual,
            expected,
            TOLERANCE,
        )?;
    }
    check_close("learned intercept", learned_intercept, TRUE_BIAS, TOLERANCE)?;
    check_close_f64(
        "training rss_f32_predictor",
        diagnostics.rss_f32_predictor,
        0.0,
        RSS_TOLERANCE,
    )?;

    let model = result.into_predictor();
    let eval_features: Vec<Vector> = EVAL_INDICES
        .iter()
        .map(|&index| features[index].clone())
        .collect();
    let eval_targets: Vec<f32> = EVAL_INDICES.iter().map(|&index| targets[index]).collect();
    let eval_observations: Vec<Observation<Vector>> = eval_features
        .iter()
        .map(|feature_vector| observation(feature_vector.as_slice()))
        .collect();

    let direct: Vec<f32> = eval_features
        .iter()
        .map(|feature_vector| {
            model
                .predict_checked(feature_vector)
                .map_err(|error| format!("checked direct prediction failed: {error:?}"))
        })
        .collect::<Result<Vec<f32>, String>>()?;

    let checked = CheckedLinearRegression::new(&model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let mut streamed = Vec::with_capacity(EVAL_INDICES.len());
    for (position, feature_vector) in eval_features.iter().enumerate() {
        let value = *executor
            .process_one(&observation(feature_vector.as_slice()))
            .map_err(|error| {
                format!("checked streaming failed at evaluation position {position}: {error:?}")
            })?;
        streamed.push(value);
    }

    let grouped = model
        .predict_batch_checked(&eval_observations)
        .map_err(|error| format!("checked grouped prediction failed: {error:?}"))?;

    check_vec_bits("direct vs streaming", &direct, &streamed)?;
    check_vec_bits("direct vs grouped", &direct, &grouped)?;
    check_vec_close(
        "evaluation predictions",
        &direct,
        &EXPECTED_EVAL_PREDICTIONS,
        TOLERANCE,
    )?;

    let bare: Vec<f32> = eval_features
        .iter()
        .map(|feature_vector| {
            model
                .predict(feature_vector)
                .map_err(|error| format!("bare prediction failed: {error:?}"))
        })
        .collect::<Result<Vec<f32>, String>>()?;
    let bare_grouped = model
        .predict_batch(&eval_observations)
        .map_err(|error| format!("bare grouped prediction failed: {error:?}"))?;
    check_vec_bits("bare vs checked", &bare, &direct)?;
    check_vec_bits("bare grouped vs checked grouped", &bare_grouped, &grouped)?;

    let empty = model
        .predict_batch_checked(&[])
        .map_err(|error| format!("empty grouped prediction failed: {error:?}"))?;
    if !empty.is_empty() {
        return Err(format!(
            "empty grouped prediction returned {} values",
            empty.len()
        ));
    }

    let mae = mean_absolute_error(&eval_targets, &direct)?;
    check_close_f64("MAE", mae, EXPECTED_MAE, METRIC_TOLERANCE)?;
    let r2 = match r_squared(&eval_targets, &direct)? {
        Some(value) => value,
        None => return Err("expected defined R² for nonconstant evaluation targets".into()),
    };
    check_close_f64("R²", r2, EXPECTED_R2, METRIC_TOLERANCE)?;

    let constant_targets = [5.0_f32, 5.0];
    let constant_predictions = [5.0_f32, 6.0];
    if r_squared(&constant_targets, &constant_predictions)?.is_some() {
        return Err("expected undefined R² for constant targets".into());
    }
    if mean_absolute_error(&[1.0_f32], &[f32::NAN]).is_ok() {
        return Err("non-finite metric input was unexpectedly accepted".into());
    }

    let mut shifted_targets = targets.clone();
    for &index in &TRAIN_INDICES {
        shifted_targets[index] += TARGET_SHIFT;
    }
    let shifted_result = fit(
        &features,
        &shifted_targets,
        &TRAIN_INDICES,
        &FitConfig::default(),
    )
    .map_err(|error| format!("shifted fit failed: {error:?}"))?;
    let shifted_intercept = shifted_result.intercept();
    check_close(
        "shifted intercept",
        shifted_intercept,
        EXPECTED_SHIFTED_INTERCEPT,
        TOLERANCE,
    )?;
    let shifted_model = shifted_result.into_predictor();
    let shifted_predictions: Vec<f32> = eval_features
        .iter()
        .map(|feature_vector| {
            shifted_model
                .predict_checked(feature_vector)
                .map_err(|error| format!("shifted checked prediction failed: {error:?}"))
        })
        .collect::<Result<Vec<f32>, String>>()?;
    check_vec_close(
        "shifted predictions",
        &shifted_predictions,
        &EXPECTED_SHIFTED_PREDICTIONS,
        TOLERANCE,
    )?;
    for (index, (&base, &shifted)) in direct.iter().zip(shifted_predictions.iter()).enumerate() {
        if base.to_bits() == shifted.to_bits() {
            return Err(format!(
                "training-target shift did not change prediction {index}"
            ));
        }
        check_close(
            &format!("shifted prediction {index} delta"),
            shifted - base,
            TARGET_SHIFT,
            TOLERANCE,
        )?;
    }

    match LinearRegression::new(Vector::zeros(0), 0.0) {
        Err(RegressionError::ZeroDimension) => {}
        other => return Err(format!("expected ZeroDimension, got {other:?}")),
    }
    match LinearRegression::new(Vector::from_slice(&[1.0, f32::NAN]), 0.0) {
        Err(RegressionError::NonFiniteParameter) => {}
        other => return Err(format!("expected NonFiniteParameter, got {other:?}")),
    }
    match model.predict(&Vector::from_slice(&[1.0])) {
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }) => {}
        other => {
            return Err(format!(
                "expected DimensionMismatch {{ expected: 2, actual: 1 }}, got {other:?}"
            ));
        }
    }
    match model.predict(&Vector::from_slice(&[1.0, f32::NAN])) {
        Err(RegressionError::NonFiniteInput) => {}
        other => return Err(format!("expected NonFiniteInput, got {other:?}")),
    }

    let overflow_model = LinearRegression::new(
        Vector::from_slice(&OVERFLOW_COMPAT_WEIGHTS),
        OVERFLOW_COMPAT_BIAS,
    )
    .map_err(|error| format!("overflow model construction failed: {error:?}"))?;
    let overflow_bare = overflow_model
        .predict(&Vector::from_slice(&OVERFLOW_COMPAT_FEATURES[1]))
        .map_err(|error| format!("overflow bare prediction failed: {error:?}"))?;
    if overflow_bare.to_bits() != f32::INFINITY.to_bits() {
        return Err(format!(
            "expected positive infinite bare overflow prediction, got {overflow_bare}"
        ));
    }
    match overflow_model.predict_checked(&Vector::from_slice(&OVERFLOW_COMPAT_FEATURES[1])) {
        Err(RegressionError::NonFiniteOutput { observation: None }) => {}
        other => {
            return Err(format!(
                "expected single NonFiniteOutput {{ observation: None }}, got {other:?}"
            ));
        }
    }
    let overflow_observations: Vec<Observation<Vector>> = OVERFLOW_COMPAT_FEATURES
        .iter()
        .map(|features| observation(&features[..]))
        .collect();
    match overflow_model.predict_batch_checked(&overflow_observations) {
        Err(RegressionError::NonFiniteOutput {
            observation: Some(1),
        }) => {}
        other => {
            return Err(format!(
                "expected grouped NonFiniteOutput {{ observation: Some(1) }}, got {other:?}"
            ));
        }
    }
    let overflow_first = overflow_model
        .predict(&Vector::from_slice(&OVERFLOW_COMPAT_FEATURES[0]))
        .map_err(|error| format!("overflow first prediction failed: {error:?}"))?;
    if overflow_first.to_bits() != 1.0e20_f32.to_bits() {
        return Err(format!(
            "expected finite first overflow prediction 1.0e20, got {overflow_first}"
        ));
    }

    println!("dataset = {DATASET_NAME}");
    println!("features = {FEATURE_NAMES:?}");
    println!(
        "training rows = {}, evaluation rows = {}",
        TRAIN_INDICES.len(),
        EVAL_INDICES.len()
    );
    println!("learned weights = {:?}", learned_weights.as_slice());
    println!("learned intercept = {learned_intercept}");
    println!(
        "diagnostics: rows={} features={} intercept={} rank={}/{} rss_f32={:.6}",
        diagnostics.rows,
        diagnostics.features,
        diagnostics.intercept,
        diagnostics.rank,
        diagnostics.required_rank,
        diagnostics.rss_f32_predictor
    );
    println!("evaluation predictions = {direct:?}");
    println!("MAE = {mae}");
    println!("R² = {r2}");
    println!("R² = undefined (constant targets)");
    println!("shifted intercept = {shifted_intercept}");
    println!("shifted predictions = {shifted_predictions:?}");
    println!("bare == checked (finite) = true");
    println!("overflow bare = {overflow_bare}; checked = NonFiniteOutput {{ observation: None }}");
    println!("grouped overflow = NonFiniteOutput {{ observation: Some(1) }}");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ols_predict: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ols_predict_workflow_succeeds() {
        super::run().expect("ols_predict workflow");
    }
}
