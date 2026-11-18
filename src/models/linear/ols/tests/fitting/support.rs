use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::FitResult;

pub(super) fn features(rows: &[&[f32]]) -> Vec<Vector> {
    rows.iter().map(|row| Vector::from_slice(row)).collect()
}

pub(super) fn approx(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual} (tol {tol})"
    );
}

pub(super) fn independent_acceptance_rss(
    candidate: &FitResult,
    features: &[Vector],
    targets: &[f32],
    indices: &[usize],
) -> f64 {
    let mut rss = 0.0f64;
    for &index in indices {
        let prediction = candidate
            .weights()
            .as_slice()
            .iter()
            .zip(features[index].as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + candidate.intercept();
        let residual = f64::from(prediction) - f64::from(targets[index]);
        rss += residual * residual;
    }
    rss
}
