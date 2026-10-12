//! Fitter tests for the first working OLS workflow.
//!
//! Independent references are closed-form / algebraically known values; the
//! fitter's own faer QR path is never reused as the oracle.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{ConversionCause, FitConfig, FitError, UnderflowPolicy, fit};

fn features(rows: &[&[f32]]) -> Vec<Vector> {
    rows.iter().map(|row| Vector::from_slice(row)).collect()
}

fn approx(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual} (tol {tol})"
    );
}

#[test]
fn exact_line_with_intercept() {
    // y = 2x + 1 through (0,1), (1,3), (2,5).
    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 3.0, 5.0];
    let result = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    approx(result.weights().as_slice()[0], 2.0, 1e-6);
    approx(result.intercept(), 1.0, 1e-6);
    assert!(result.diagnostics().rss_f32_predictor < 1e-10);
    let model = result.into_predictor();
    approx(
        model.predict(&Vector::from_slice(&[3.0])).unwrap(),
        7.0,
        1e-4,
    );
}

#[test]
fn noisy_line_matches_closed_form_reference() {
    // x = [0,1,2], y = [1,2,2]; closed form: b = 7/6, w = 1/2.
    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 2.0, 2.0];
    let result = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    approx(result.weights().as_slice()[0], 0.5, 1e-6);
    approx(result.intercept(), 7.0 / 6.0, 1e-6);
}

#[test]
fn multiple_features_exact_recovery() {
    // Construct y = A w* with intercept; recover w* in original order.
    let x = features(&[&[1.0, 2.0], &[2.0, 3.0], &[3.0, 5.0], &[4.0, 7.0]]);
    let w_star = [0.5f64, 2.0, -1.0];
    let y: Vec<f32> = [
        [1.0f64, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 5.0],
        [1.0, 4.0, 7.0],
    ]
    .iter()
    .map(|row| row.iter().zip(&w_star).map(|(a, b)| a * b).sum::<f64>() as f32)
    .collect();
    let result = fit(&x, &y, &[0, 1, 2, 3], &FitConfig::default()).expect("fit");
    approx(result.intercept(), w_star[0] as f32, 1e-5);
    approx(result.weights().as_slice()[0], w_star[1] as f32, 1e-5);
    approx(result.weights().as_slice()[1], w_star[2] as f32, 1e-5);
}

#[test]
fn fit_without_intercept() {
    let x = features(&[&[1.0], &[2.0], &[3.0]]);
    let y = [3.0f32, 6.0, 9.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1, 2], &config).expect("fit");
    approx(result.weights().as_slice()[0], 3.0, 1e-6);
    approx(result.intercept(), 0.0, 0.0);
}

#[test]
fn coefficients_are_in_original_column_order_after_pivoting() {
    // Feature columns of very different scale force column pivoting; the
    // recovered coefficients must remain in the original feature order.
    let x = features(&[
        &[1.0, 1.0e-4],
        &[2.0, 3.0e-4],
        &[3.0, 2.0e-4],
        &[4.0, 5.0e-4],
    ]);
    let w_star = [0.5f64, 2.0, 3.0e4]; // intercept, feature1, feature2
    let y: Vec<f32> = [
        [1.0f64, 1.0, 1.0e-4],
        [1.0, 2.0, 3.0e-4],
        [1.0, 3.0, 2.0e-4],
        [1.0, 4.0, 5.0e-4],
    ]
    .iter()
    .map(|row| row.iter().zip(&w_star).map(|(a, b)| a * b).sum::<f64>() as f32)
    .collect();
    let result = fit(&x, &y, &[0, 1, 2, 3], &FitConfig::default()).expect("fit");
    approx(result.intercept(), w_star[0] as f32, 1e-4);
    approx(result.weights().as_slice()[0], w_star[1] as f32, 1e-3);
    let got = result.weights().as_slice()[1];
    assert!(
        (got - w_star[2] as f32).abs() / (w_star[2] as f32) < 1e-4,
        "feature2 coefficient {got} vs {}",
        w_star[2]
    );
}

#[test]
fn training_only_access_ignores_unselected_rows() {
    // The middle row has a non-finite target but is not selected.
    let x = features(&[&[0.0], &[2.0], &[5.0]]);
    let y = [0.0f32, f32::NAN, 5.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 2], &config).expect("held-out target not inspected");
    approx(result.weights().as_slice()[0], 1.0, 1e-6);
    assert_eq!(result.diagnostics().rows, 2);
}

#[test]
fn rejects_row_target_mismatch() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32];
    match fit(&x, &y, &[0], &FitConfig::default()) {
        Err(FitError::RowTargetMismatch { features, targets }) => {
            assert_eq!((features, targets), (2, 1));
        }
        other => panic!("expected RowTargetMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_empty_training_selection() {
    let x = features(&[&[1.0]]);
    let y = [1.0f32];
    assert_eq!(
        fit(&x, &y, &[], &FitConfig::default()),
        Err(FitError::TooFewRows)
    );
}

#[test]
fn rejects_out_of_range_index() {
    let x = features(&[&[1.0]]);
    let y = [1.0f32];
    assert_eq!(
        fit(&x, &y, &[0, 5], &FitConfig::default()),
        Err(FitError::IndexOutOfRange { index: 5, rows: 1 })
    );
}

#[test]
fn rejects_zero_features() {
    let x = vec![Vector::zeros(0)];
    let y = [1.0f32];
    assert_eq!(
        fit(&x, &y, &[0], &FitConfig::default()),
        Err(FitError::ZeroFeatures)
    );
}

#[test]
fn rejects_ragged_features() {
    let x = features(&[&[1.0, 2.0], &[3.0]]);
    let y = [1.0f32, 2.0];
    match fit(&x, &y, &[0, 1], &FitConfig::default()) {
        Err(FitError::RaggedFeatures {
            row,
            expected,
            actual,
        }) => {
            assert_eq!((row, expected, actual), (1, 2, 1));
        }
        other => panic!("expected RaggedFeatures, got {other:?}"),
    }
}

#[test]
fn rejects_duplicate_index() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 2.0];
    assert_eq!(
        fit(&x, &y, &[0, 0], &FitConfig::default()),
        Err(FitError::DuplicateIndex { index: 0 })
    );
}

#[test]
fn rejects_non_finite_training_feature_and_target() {
    let x = features(&[&[f32::INFINITY], &[2.0]]);
    let y = [1.0f32, 2.0];
    match fit(&x, &y, &[0, 1], &FitConfig::default()) {
        Err(FitError::NonFiniteTrainingValue { row, column }) => {
            assert_eq!((row, column), (0, 0));
        }
        other => panic!("expected NonFiniteTrainingValue, got {other:?}"),
    }

    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, f32::NAN];
    match fit(&x, &y, &[0, 1], &FitConfig::default()) {
        Err(FitError::NonFiniteTrainingValue { row, column }) => {
            assert_eq!((row, column), (1, 1));
        }
        other => panic!("expected NonFiniteTrainingValue, got {other:?}"),
    }
}

#[test]
fn rejects_duplicate_columns_rank_deficient() {
    // Second column is exactly twice the first: rank 1 < required 2.
    let x = features(&[&[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0]]);
    let y = [1.0f32, 2.0, 3.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1, 2], &config) {
        Err(FitError::RankDeficient { rank, required }) => {
            assert_eq!((rank, required), (1, 2));
        }
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn rejects_constant_column_with_intercept() {
    // A constant feature duplicates the intercept column: rank 1 < required 2.
    let x = features(&[&[1.0], &[1.0], &[1.0]]);
    let y = [1.0f32, 2.0, 3.0];
    match fit(&x, &y, &[0, 1, 2], &FitConfig::default()) {
        Err(FitError::RankDeficient { rank, required }) => {
            assert_eq!((rank, required), (1, 2));
        }
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn rejects_over_budget_workspace() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 2.0];
    let config = FitConfig {
        max_workspace_bytes: 1,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1], &config) {
        Err(FitError::Capacity { requested, limit }) => {
            assert!(requested > limit);
            assert_eq!(limit, 1);
        }
        other => panic!("expected Capacity, got {other:?}"),
    }
}

#[test]
fn conversion_underflow_rejects_by_default_and_allows_opt_in() {
    // x = 2^100, y = 2^-100, no intercept -> exact coefficient 2^-200, whose
    // f32 conversion is zero. Default policy rejects; opt-in deploys zero.
    let x = features(&[&[2.0f32.powi(0) * (2.0f64.powi(100) as f32)]]);
    let y = [(2.0f64.powi(-100)) as f32];
    let rejected = fit(
        &x,
        &y,
        &[0],
        &FitConfig {
            intercept: false,
            ..FitConfig::default()
        },
    );
    assert_eq!(
        rejected,
        Err(FitError::ConversionRejected(
            ConversionCause::CoefficientUnderflow
        ))
    );

    let allowed = fit(
        &x,
        &y,
        &[0],
        &FitConfig {
            intercept: false,
            on_underflow: UnderflowPolicy::AllowLoss,
            ..FitConfig::default()
        },
    )
    .expect("opt-in underflow");
    assert_eq!(allowed.weights().as_slice(), &[0.0f32]);
}

#[test]
fn diagnostics_distinguish_f64_reference_and_f32_predictor() {
    let x = features(&[&[1.0], &[2.0], &[3.0]]);
    let y = [2.0f32, 4.0, 6.0];
    let result = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    let diag = result.diagnostics();
    assert_eq!(diag.rows, 3);
    assert_eq!(diag.features, 1);
    assert_eq!(diag.rank, diag.required_rank);
    assert!(diag.rank_threshold >= 0.0);
    assert!(diag.requested_workspace_bytes > 0);
    assert!(diag.rss_f64_reference.is_finite());
    assert!(diag.rss_f32_predictor.is_finite());
}

#[test]
fn predictor_streaming_behavior_is_preserved() {
    let x = features(&[&[1.0], &[2.0], &[3.0]]);
    let y = [2.0f32, 4.0, 6.0];
    let model = fit(&x, &y, &[0, 1, 2], &FitConfig::default())
        .expect("fit")
        .into_predictor();
    let mut executor = StreamingExecutor::new(&model, 0.0f32);
    let streamed: Vec<f32> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|value| {
            *executor
                .process_one(&Observation::new(Vector::from_slice(&[*value])))
                .unwrap()
        })
        .collect();
    let grouped = model
        .predict_batch(&[
            Observation::new(Vector::from_slice(&[1.0])),
            Observation::new(Vector::from_slice(&[2.0])),
            Observation::new(Vector::from_slice(&[3.0])),
        ])
        .unwrap();
    assert_eq!(streamed, grouped);
}
