use super::support::{assert_bits_eq, model, scalar_predict_f32, scalar_predict_f64, tolerance};
use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::RegressionError;

// --- E.9 / E.10 numerical behavior and f32/f64 diagnostic -------------------

fn assert_numerical_case(weights: &[f32], bias: f32, features: &[f32]) {
    let model = model(weights, bias);
    let produced = model.predict(&Vector::from_slice(features)).unwrap();
    assert!(produced.is_finite(), "unexpected non-finite output");
    assert_bits_eq(produced, scalar_predict_f32(weights, bias, features));
    let oracle = scalar_predict_f64(weights, bias, features);
    let error = (f64::from(produced) - oracle).abs();
    let tolerance = tolerance(weights, bias, features);
    assert!(
        error <= tolerance,
        "|error| {error} > tolerance {tolerance} (f32 {produced}, f64 {oracle})"
    );
}

#[test]
fn numerical_small_magnitudes() {
    for scale in [1.0e-6_f32, 1.0e-4, 1.0e-2] {
        let weights = [scale, -2.0 * scale, 0.5 * scale];
        let features = [scale, scale, -scale];
        let bias = 0.25 * scale;
        assert_numerical_case(&weights, bias, &features);
    }
}

#[test]
fn numerical_normal_magnitudes() {
    for scale in [0.1_f32, 1.0, 10.0, 100.0] {
        let weights = [scale, -0.5 * scale, 0.25 * scale];
        let features = [scale, scale, scale];
        let bias = 0.125 * scale;
        assert_numerical_case(&weights, bias, &features);
    }
}

#[test]
fn numerical_large_finite_magnitudes() {
    for scale in [1.0e3_f32, 1.0e4, 1.0e5, 1.0e10] {
        let weights = [scale, -0.5 * scale, 0.25 * scale];
        let features = [scale, scale, scale];
        let bias = 0.125 * scale;
        assert_numerical_case(&weights, bias, &features);
    }
}

#[test]
fn numerical_cancellation_is_expected_rounding() {
    // At |1e8| the f32 ulp is 8, so the residual 1.0 is added to a 1e8 term
    // first and absorbed, and the subsequent exact cancellation leaves 0.0.
    // Reordering the same terms gives 1.0. This is expected f32 rounding, not a
    // defect.
    let model = model(&[1.0, 1.0, 1.0], 0.0);
    let absorbing = [1.0e8_f32, 1.0, -1.0e8];
    assert_bits_eq(model.predict(&Vector::from_slice(&absorbing)).unwrap(), 0.0);

    let reordered = [1.0e8_f32, -1.0e8, 1.0];
    assert_bits_eq(model.predict(&Vector::from_slice(&reordered)).unwrap(), 1.0);
}

#[test]
fn numerical_dynamic_range_reports_boundary() {
    let weights = [0.5_f32, -1.25, 2.0];
    let features = [1.0e-3_f32, 1.0, 1.0e3];
    assert_numerical_case(&weights, -0.375, &features);
}

#[test]
fn finite_input_non_finite_output_is_rejected() {
    // Ordinary prediction validates the output. Finite parameters and inputs
    // can overflow f32; the public result is an explicit error, not Ok(inf).
    let product_overflow = model(&[1.0e20], 0.0);
    assert_eq!(
        product_overflow.predict(&Vector::from_slice(&[1.0e20])),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );

    let summation_overflow = model(&[2.0e38, 2.0e38], 0.0);
    assert_eq!(
        summation_overflow.predict(&Vector::from_slice(&[1.0, 1.0])),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );

    let nan_from_infinities = model(&[1.0e20, -1.0e20], 0.0);
    assert_eq!(
        nan_from_infinities.predict(&Vector::from_slice(&[1.0e20, 1.0e20])),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );

    // Boundary bracket: 1e38 is finite, 1e40 is not.
    let below = model(&[1.0e19], 0.0);
    assert!(
        below
            .predict(&Vector::from_slice(&[1.0e19]))
            .unwrap()
            .is_finite()
    );
}

#[test]
fn f32_vs_f64_error_envelope() {
    // Diagnostic table for the audit report. The only assertion is the
    // pre-registered tolerance; the printed values are the measured envelope.
    let cases: [(&str, f32, [f32; 3], [f32; 3]); 6] = [
        ("small-1e-6", 1.0e-6, [1.0, -2.0, 0.5], [1.0, 1.0, -1.0]),
        ("small-1e-2", 1.0e-2, [1.0, -2.0, 0.5], [1.0, 1.0, -1.0]),
        ("normal-1", 1.0, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
        ("normal-100", 100.0, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
        ("large-1e5", 1.0e5, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
        ("large-1e10", 1.0e10, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
    ];
    for (label, scale, weight_factors, feature_factors) in cases {
        let weights: Vec<f32> = weight_factors
            .iter()
            .map(|&factor| factor * scale)
            .collect();
        let features: Vec<f32> = feature_factors
            .iter()
            .map(|&factor| factor * scale)
            .collect();
        let bias = 0.125 * scale;
        let model = model(&weights, bias);
        let produced = model.predict(&Vector::from_slice(&features)).unwrap();
        let oracle = scalar_predict_f64(&weights, bias, &features);
        let absolute = (f64::from(produced) - oracle).abs();
        let relative = absolute / oracle.abs().max(1.0e-30);
        let tolerance = tolerance(&weights, bias, &features);
        println!(
            "f32/f64 {label}: scale={scale:e} D={} abs_err={absolute:e} rel_err={relative:e} tol={tolerance:e}",
            weights.len()
        );
        assert!(produced.is_finite(), "{label}: non-finite output");
        assert!(absolute <= tolerance, "{label}: {absolute} > {tolerance}");
    }
}
