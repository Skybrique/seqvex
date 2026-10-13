//! Contract tests for the minimal numerical foundation (`f64` statistics).
//!
//! The panic contract and edge semantics are documented in
//! `src/foundation/numerical/statistics.rs` and in
//! `docs/KILO_CONNECTOME_SPRINT_REVISED.md` §9.

use seqvex::foundation::numerical;

fn approx(left: f64, right: f64) {
    assert!(
        (left - right).abs() <= 1e-12,
        "expected {right}, got {left}"
    );
}

#[test]
fn sum_adds_all_values() {
    assert_eq!(numerical::sum(&[1.0, 2.0, 3.0]), 6.0);
    assert_eq!(numerical::sum(&[]), 0.0);
}

#[test]
fn mean_is_arithmetic_average() {
    assert_eq!(numerical::mean(&[1.0, 2.0, 3.0]), 2.0);
}

#[test]
fn variance_is_population_variance() {
    assert_eq!(numerical::variance(&[1.0, 1.0, 1.0]), 0.0);
    // Singleton variance is zero.
    assert_eq!(numerical::variance(&[42.0]), 0.0);
    // Population variance of [1, 2, 3] is 2/3, not the sample variance 1.
    approx(numerical::variance(&[1.0, 2.0, 3.0]), 2.0 / 3.0);
}

#[test]
fn dot_multiplies_corresponding_elements() {
    assert_eq!(numerical::dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
}

#[test]
fn norm_is_euclidean_length() {
    assert_eq!(numerical::norm(&[3.0, 4.0]), 5.0);
}

#[test]
fn online_mean_tracks_running_average() {
    let after_first = numerical::online_mean(0.0, 0, 2.0);
    assert_eq!(after_first, 2.0);
    let after_second = numerical::online_mean(after_first, 1, 4.0);
    assert_eq!(after_second, 3.0);
}

#[test]
fn online_variance_matches_batch_population_variance() {
    let mut mean = 0.0;
    let mut variance = 0.0;
    for (count, value) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
        variance = numerical::online_variance(variance, mean, count as u64, value);
        mean = numerical::online_mean(mean, count as u64, value);
    }
    approx(variance, numerical::variance(&[1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn online_variance_stays_non_negative() {
    let value = numerical::online_variance(0.0, 2.0, 1, 4.0);
    assert!(value >= 0.0);
}

#[test]
#[should_panic(expected = "empty slice undefined")]
fn mean_panics_on_empty_input() {
    numerical::mean(&[]);
}

#[test]
#[should_panic(expected = "empty slice undefined")]
fn variance_panics_on_empty_input() {
    numerical::variance(&[]);
}

#[test]
#[should_panic(expected = "empty slice undefined")]
fn norm_panics_on_empty_input() {
    numerical::norm(&[]);
}

#[test]
#[should_panic(expected = "equal length")]
fn dot_panics_on_length_mismatch() {
    numerical::dot(&[1.0, 2.0], &[1.0]);
}

#[test]
#[should_panic(expected = "must be non-negative")]
fn variance_panics_on_nan_input() {
    numerical::variance(&[f64::NAN]);
}

#[test]
#[should_panic(expected = "must be non-negative")]
fn variance_panics_on_positive_infinity_input() {
    numerical::variance(&[f64::INFINITY]);
}

#[test]
#[should_panic(expected = "must be non-negative")]
fn variance_panics_on_negative_infinity_input() {
    numerical::variance(&[f64::NEG_INFINITY]);
}

#[test]
#[should_panic(expected = "must be non-negative")]
fn norm_panics_on_nan_input() {
    numerical::norm(&[f64::NAN]);
}

#[test]
fn norm_returns_positive_infinity_for_infinite_magnitudes() {
    assert_eq!(numerical::norm(&[f64::INFINITY]), f64::INFINITY);
    assert_eq!(numerical::norm(&[f64::NEG_INFINITY]), f64::INFINITY);
    assert_eq!(numerical::norm(&[1.0, f64::INFINITY]), f64::INFINITY);
}

#[test]
fn sum_and_mean_propagate_non_finite_values() {
    assert_eq!(numerical::sum(&[f64::INFINITY, 1.0]), f64::INFINITY);
    assert_eq!(numerical::sum(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
    assert!(numerical::sum(&[f64::NAN]).is_nan());
    assert!(numerical::mean(&[f64::NAN, 1.0]).is_nan());
    assert_eq!(numerical::mean(&[f64::INFINITY]), f64::INFINITY);
    assert_eq!(numerical::mean(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
}

#[test]
fn dot_propagates_non_finite_values() {
    assert_eq!(numerical::dot(&[f64::INFINITY], &[1.0]), f64::INFINITY);
    assert!(numerical::dot(&[f64::NAN], &[1.0]).is_nan());
}

#[test]
#[should_panic(expected = "count is the count before the new value")]
fn online_mean_panics_on_inconsistent_count() {
    numerical::online_mean(1.0, 0, 5.0);
}

#[test]
#[should_panic(expected = "count is the count before the new value")]
fn online_variance_panics_on_inconsistent_count() {
    numerical::online_variance(0.5, 0.0, 0, 5.0);
}

// --- f32 substrate: Vector ---------------------------------------------------

#[test]
fn vector_supports_construction_and_indexed_access() {
    let zeros = numerical::Vector::zeros(3);
    assert_eq!(zeros.as_slice(), &[0.0, 0.0, 0.0]);
    assert_eq!(zeros.len(), 3);
    assert!(!zeros.is_empty());

    let built = numerical::Vector::from_fn(3, |index| index as f32);
    assert_eq!(built.as_slice(), &[0.0, 1.0, 2.0]);
    assert_eq!(built.get(2), Some(2.0));
    assert_eq!(built.get(3), None);

    let copied = numerical::Vector::from_slice(&[1.0, 2.0]);
    assert_eq!(copied.as_slice(), &[1.0, 2.0]);
    assert!(numerical::Vector::zeros(0).is_empty());
}

#[test]
fn vector_elementwise_operations_are_elementwise() {
    let left = numerical::Vector::from_slice(&[1.0, 2.0, 3.0]);
    let right = numerical::Vector::from_slice(&[4.0, 5.0, 6.0]);

    assert_eq!(left.add(&right).unwrap().as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!(
        left.multiply(&right).unwrap().as_slice(),
        &[4.0, 10.0, 18.0]
    );
    assert_eq!(left.scale(2.0).as_slice(), &[2.0, 4.0, 6.0]);
    assert_eq!(left.complement().as_slice(), &[0.0, -1.0, -2.0]);
}

#[test]
fn vector_dot_multiplies_corresponding_elements() {
    let left = numerical::Vector::from_slice(&[1.0, 2.0, 3.0]);
    let right = numerical::Vector::from_slice(&[4.0, 5.0, 6.0]);
    assert_eq!(left.dot(&right), Ok(32.0));
    assert!(left.dot(&numerical::Vector::from_slice(&[1.0])).is_err());
}

#[test]
fn vector_elementwise_operations_reject_length_mismatch() {
    let left = numerical::Vector::from_slice(&[1.0, 2.0]);
    let right = numerical::Vector::from_slice(&[1.0]);
    assert_eq!(
        left.add(&right),
        Err(numerical::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert!(left.multiply(&right).is_err());
}

// --- f32 substrate: Matrix and mat-vec --------------------------------------

#[test]
fn matrix_is_row_major_and_reports_shape() {
    let matrix = numerical::Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(matrix.get(1, 2), Some(6.0));
    assert_eq!(matrix.get(2, 0), None);
}

#[test]
fn matrix_rejects_ragged_rows() {
    assert_eq!(
        numerical::Matrix::from_rows(&[&[1.0, 2.0], &[3.0]]),
        Err(numerical::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn matvec_computes_y_equals_w_times_x() {
    let weights = numerical::Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    let input = numerical::Vector::from_slice(&[1.0, 0.0]);
    assert_eq!(weights.mul_vector(&input).unwrap().as_slice(), &[1.0, 3.0]);
}

#[test]
fn matvec_rejects_input_dimension_mismatch() {
    let weights = numerical::Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    let input = numerical::Vector::from_slice(&[1.0]);
    assert_eq!(
        weights.mul_vector(&input),
        Err(numerical::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

// --- f32 substrate: activations ---------------------------------------------

#[test]
fn activations_match_known_values() {
    assert_eq!(numerical::sigmoid(0.0), 0.5);
    assert_eq!(numerical::tanh(0.0), 0.0);
}

#[test]
fn activations_obey_symmetry() {
    for value in [-3.0_f32, -1.0, 0.25, 2.0] {
        assert!((numerical::sigmoid(-value) - (1.0 - numerical::sigmoid(value))).abs() <= 1e-6);
        assert!((numerical::tanh(-value) + numerical::tanh(value)).abs() <= 1e-6);
    }
}

#[test]
fn activations_saturate_at_large_magnitudes() {
    assert_eq!(numerical::sigmoid(100.0), 1.0);
    assert_eq!(numerical::sigmoid(-100.0), 0.0);
    assert_eq!(numerical::tanh(100.0), 1.0);
    assert_eq!(numerical::tanh(-100.0), -1.0);
    assert!(numerical::sigmoid(f32::NAN).is_nan());
    assert!(numerical::tanh(f32::NAN).is_nan());
}

#[test]
fn vector_mapping_agrees_with_scalar_activations() {
    let input = numerical::Vector::from_slice(&[-1.0, 0.0, 1.0]);
    let mapped = input.map(numerical::sigmoid);
    let scalar = numerical::Vector::from_slice(&[
        numerical::sigmoid(-1.0),
        numerical::sigmoid(0.0),
        numerical::sigmoid(1.0),
    ]);
    assert_eq!(mapped.as_slice(), scalar.as_slice());
}
