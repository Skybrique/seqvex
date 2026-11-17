use super::support::{approx, features};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{FitConfig, fit};

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

#[test]
fn nonzero_residual_diagnostics_match_independent_expected() {
    // No intercept, x=[0,1], y=[0.5,0.5]: closed form w=0.5, residuals
    // [0.5, 0], so RSS = 0.25 exactly (both f64 and f32).
    let x = features(&[&[0.0], &[1.0]]);
    let y = [0.5f32, 0.5];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1], &config).expect("fit");
    approx(result.weights().as_slice()[0], 0.5, 1e-7);
    let diagnostics = result.diagnostics();
    assert!((diagnostics.rss_f64_reference - 0.25).abs() < 1e-12);
    assert!((diagnostics.rss_f32_predictor - 0.25).abs() < 1e-6);
}
