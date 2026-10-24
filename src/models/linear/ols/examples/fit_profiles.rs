//! Typed in-memory quality smoke test for OLS A/B/C.
//!
//! Training and held-out rows are explicit. Each profile is fitted on training
//! rows only, then predicts the held-out rows with its learned parameters.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::models::linear::ols::{FitConfig32, FitConfig64, UnderflowPolicy, fit_f32, fit_f64};

const PROFILE_QUALITY_FIXTURE: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
const PROFILE_QUALITY_TARGETS: [f32; 8] = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0];
const TRAIN: [usize; 5] = [0, 1, 2, 3, 4];
const HELD_OUT: [usize; 3] = [5, 6, 7];

fn quality_metrics(predictions: &[f64]) -> (f64, f64) {
    let targets: Vec<f64> = HELD_OUT
        .iter()
        .map(|&index| f64::from(PROFILE_QUALITY_TARGETS[index]))
        .collect();
    assert_eq!(predictions.len(), targets.len());
    assert!(!targets.is_empty());
    assert!(predictions.iter().all(|prediction| prediction.is_finite()));

    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let mut absolute_error = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut total_variance = 0.0f64;
    for (&prediction, &target) in predictions.iter().zip(&targets) {
        let residual = prediction - target;
        absolute_error += residual.abs();
        squared_error += residual * residual;
        let centered = target - mean;
        total_variance += centered * centered;
    }
    assert!(total_variance > 0.0, "held-out targets must be nonconstant");
    let mae = absolute_error / targets.len() as f64;
    let r2 = 1.0 - squared_error / total_variance;
    assert!(mae.is_finite() && mae >= 0.0);
    assert!(r2.is_finite());
    (mae, r2)
}

fn report(profile: &str, predictions: &[f64]) {
    let (mae, r2) = quality_metrics(predictions);
    println!("{profile}: MAE={mae:.6} R2={r2:.6}");
}

fn main() {
    let a_features: Vec<Vector64> = PROFILE_QUALITY_FIXTURE
        .iter()
        .map(|&value| Vector64::from_slice(&[f64::from(value)]))
        .collect();
    let a_targets: Vec<f64> = PROFILE_QUALITY_TARGETS
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    let a = fit_f64(&a_features, &a_targets, &TRAIN, &FitConfig64::default()).expect("A fit");
    let a_predictor = a.predictor();
    let a_predictions: Vec<f64> = HELD_OUT
        .iter()
        .map(|&index| {
            a_predictor
                .predict_checked(&a_features[index])
                .expect("A held-out prediction")
        })
        .collect();
    report("A", &a_predictions);

    let b_predictor = a
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B serving conversion");
    let b_predictions: Vec<f64> = HELD_OUT
        .iter()
        .map(|&index| {
            let row = Vector::from_slice(&[PROFILE_QUALITY_FIXTURE[index]]);
            f64::from(
                b_predictor
                    .predict_checked(&row)
                    .expect("B held-out prediction"),
            )
        })
        .collect();
    report("B", &b_predictions);

    let c_features: Vec<Vector> = PROFILE_QUALITY_FIXTURE
        .iter()
        .map(|&value| Vector::from_slice(&[value]))
        .collect();
    let c = fit_f32(
        &c_features,
        &PROFILE_QUALITY_TARGETS,
        &TRAIN,
        &FitConfig32::default(),
    )
    .expect("C fit");
    let c_predictor = c.into_predictor();
    let c_predictions: Vec<f64> = HELD_OUT
        .iter()
        .map(|&index| {
            f64::from(
                c_predictor
                    .predict_checked(&c_features[index])
                    .expect("C held-out prediction"),
            )
        })
        .collect();
    report("C", &c_predictions);
}
