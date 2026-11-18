use super::support::{assert_bits_eq, model, scalar_predict_f32, scalar_predict_f64, tolerance};
use seqvex::foundation::numerical::Vector;

// --- E.11 dimension scaling -------------------------------------------------

#[test]
fn dimension_scaling_oracle_remains_valid() {
    for dimension in [1_usize, 2, 4, 8, 32, 128, 256] {
        let weights: Vec<f32> = (0..dimension)
            .map(|index| (index as f32 * 0.37).sin())
            .collect();
        let features: Vec<f32> = (0..dimension)
            .map(|index| (index as f32 * 0.31 + 0.5).cos())
            .collect();
        let model = model(&weights, 0.125);
        let produced = model.predict(&Vector::from_slice(&features)).unwrap();
        assert!(produced.is_finite(), "non-finite at D={dimension}");
        assert_bits_eq(produced, scalar_predict_f32(&weights, 0.125, &features));
        let oracle = scalar_predict_f64(&weights, 0.125, &features);
        let error = (f64::from(produced) - oracle).abs();
        let tolerance = tolerance(&weights, 0.125, &features);
        println!("dimension D={dimension}: abs_err={error:e} tol={tolerance:e}");
        assert!(
            error <= tolerance,
            "D={dimension}: |error| {error} > tolerance {tolerance}"
        );
    }
}

#[test]
#[ignore = "release-only diagnostic: D=256 at 1e10 scale (finite-envelope characterization)"]
fn dimension_scaling_extreme_ignored() {
    let dimension = 256;
    let weights: Vec<f32> = (0..dimension)
        .map(|index| if index % 2 == 0 { 1.0e10 } else { -1.0e10 })
        .collect();
    let features: Vec<f32> = (0..dimension)
        .map(|index| (index as f32 * 0.37).sin() * 1.0e10)
        .collect();
    let model = model(&weights, 0.0);
    let produced = model.predict(&Vector::from_slice(&features)).unwrap();
    // Products are ≈ ±1e20 and partial sums stay well below f32::MAX.
    assert!(produced.is_finite(), "unexpected non-finite: {produced}");
}
