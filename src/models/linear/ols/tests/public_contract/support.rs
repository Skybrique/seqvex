use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::LinearRegression;

pub(super) fn model(weights: &[f32], bias: f32) -> LinearRegression {
    LinearRegression::new(Vector::from_slice(weights), bias).unwrap()
}

pub(super) fn observation(features: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(features))
}

pub(super) fn assert_bits_eq(left: f32, right: f32) {
    assert_eq!(
        left.to_bits(),
        right.to_bits(),
        "expected bitwise equality: {left} vs {right}"
    );
}
/// Independent scalar oracle in `f32`, accumulated in weight-index order from
/// the baseline negative-zero reduction identity.
///
/// Deliberately a plain index loop, not `Vector::dot`: it is a different code
/// path and a different formulation, so agreement catches wiring/omission
/// errors and pins the documented reduction order.
// The explicit scalar index loop is the independent formulation, not an
// iterator rewrite of the production path.
#[allow(clippy::needless_range_loop)]
pub(super) fn scalar_predict_f32(weights: &[f32], bias: f32, features: &[f32]) -> f32 {
    let mut accumulator = -0.0_f32;
    for index in 0..weights.len() {
        accumulator += weights[index] * features[index];
    }
    accumulator + bias
}

/// Independent `f64` oracle used as a numerical diagnostic, not as the
/// definition of correctness.
#[allow(clippy::needless_range_loop)]
pub(super) fn scalar_predict_f64(weights: &[f32], bias: f32, features: &[f32]) -> f64 {
    let mut accumulator = 0.0_f64;
    for index in 0..weights.len() {
        accumulator += f64::from(weights[index]) * f64::from(features[index]);
    }
    accumulator + f64::from(bias)
}

/// `Σ |w_i| · |x_i|`, the scale of the `f64` forward-error bound.
pub(super) fn sum_abs_products(weights: &[f32], features: &[f32]) -> f64 {
    weights
        .iter()
        .zip(features)
        .map(|(&weight, &feature)| f64::from(weight.abs()) * f64::from(feature.abs()))
        .sum()
}

/// Pre-registered absolute tolerance: `8 · (D+1) · f32::EPSILON · scale`.
///
/// `f32::EPSILON = 2u` is 2× the standard unit roundoff; the factor 8 covers the
/// `f64` oracle, the differing reduction precision, and the bias add. Effective
/// envelope ≈ `16·(D+1)·u·scale`.
pub(super) fn tolerance(weights: &[f32], bias: f32, features: &[f32]) -> f64 {
    let scale = sum_abs_products(weights, features) + f64::from(bias.abs());
    if scale == 0.0 {
        return 0.0;
    }
    8.0 * (weights.len() as f64 + 1.0) * f64::from(f32::EPSILON) * scale
}

pub(super) fn assert_matches_f64_oracle(model: &LinearRegression, features: &[f32]) {
    let weights = model.weights().as_slice();
    let produced = f64::from(model.predict(&Vector::from_slice(features)).unwrap());
    let oracle = scalar_predict_f64(weights, model.bias(), features);
    let error = (produced - oracle).abs();
    let tolerance = tolerance(weights, model.bias(), features);
    assert!(
        error <= tolerance,
        "f32 {produced} vs f64 oracle {oracle}: |error| {error} > tolerance {tolerance}"
    );
}

pub(super) fn snapshot_params(model: &LinearRegression) -> (Vec<f32>, f32) {
    (model.weights().as_slice().to_vec(), model.bias())
}

pub(super) fn assert_params_unchanged(model: &LinearRegression, snapshot: &(Vec<f32>, f32)) {
    assert_eq!(
        model.weights().as_slice(),
        snapshot.0.as_slice(),
        "weights changed"
    );
    assert_bits_eq(model.bias(), snapshot.1);
}

#[path = "../../../../../../benches/common/fixture_rng.rs"]
pub(crate) mod fixture_rng;

pub(super) use fixture_rng::Lcg64 as Lcg;

/// Deterministic bijection on `0..len` (`gcd(7, len) = 1` for `len = 64`).
pub(super) fn permutation(len: usize) -> Vec<usize> {
    (0..len).map(|index| (index * 7 + 13) % len).collect()
}

pub(super) fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0_usize; order.len()];
    for (position, &source) in order.iter().enumerate() {
        inverse[source] = position;
    }
    inverse
}

pub(super) fn deterministic_features(index: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|coordinate| ((index + 1) as f32 * 0.37 + (coordinate + 1) as f32 * 0.11).sin())
        .collect()
}

pub(super) fn assert_series_matches_oracle(model: &LinearRegression, series: &[Vec<f32>]) {
    for features in series {
        let produced = model.predict(&Vector::from_slice(features)).unwrap();
        let scalar = scalar_predict_f32(model.weights().as_slice(), model.bias(), features);
        assert_bits_eq(produced, scalar);
        assert_matches_f64_oracle(model, features);
    }
}
