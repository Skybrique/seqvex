//! Minimal statistical primitives over `f64` slices.
//!
//! These are the foundation's established numerical contracts
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §9). They operate on borrowed
//! slices and allocate nothing.
//!
//! # Semantics
//!
//! - `sum(&[]) == 0.0`.
//! - `mean`, `variance`, and `norm` panic on an empty slice, where the result is
//!   undefined.
//! - `dot` panics when the operands differ in length.
//! - `online_mean`/`online_variance` panic when the `count` is inconsistent
//!   with the running statistics (`count == 0` with a non-zero mean/variance).
//! - `variance` is the **population** variance (divide by `N`), computed in two
//!   passes for numerical stability.
//! - IEEE-754 `NaN`/`±Inf` propagate through `sum`, `mean`, and `dot`.
//! - `variance` asserts a non-negative result; any non-finite input makes the
//!   two-pass result `NaN`, so `variance` panics on `NaN` and on `±Inf`.
//! - `norm` asserts a non-negative result. A `NaN` input yields `NaN` and
//!   panics; a `±Inf` input yields `+Inf`, which satisfies the assertion, so
//!   `norm` returns `+Inf` without panicking.
//!
//! The panic contract is intentional for these `f64` primitives: it is the
//! existing hand-written behavior and `§9` forbids silently changing semantics.
//! Fallible contracts are introduced separately in the `f32` substrate.

/// Sum of all values; the empty slice sums to `0.0`.
pub fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Arithmetic mean of `values`; undefined for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "empty slice undefined");
    sum(values) / values.len() as f64
}

/// Population variance of `values`; must be non-negative.
pub fn variance(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "empty slice undefined");
    let mean = mean(values);
    let square_diff_sum: f64 = values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum();
    let result = square_diff_sum / values.len() as f64;
    assert!(result >= 0.0, "must be non-negative");
    result
}

/// Dot product of two equal-length slices.
pub fn dot(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "Vectors must be of equal length");
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

/// Euclidean (L2) norm of a vector; must be non-negative.
pub fn norm(vector: &[f64]) -> f64 {
    assert!(!vector.is_empty(), "empty slice undefined");
    let sum_of_squares: f64 = vector.iter().map(|x| x * x).sum();
    let result = sum_of_squares.sqrt();
    assert!(result >= 0.0, "must be non-negative");
    result
}

/// Online mean update from a running mean, a prior count, and the next value.
///
/// `count` is the number of values observed *before* `value`.
pub fn online_mean(previous_mean: f64, count: u64, value: f64) -> f64 {
    assert!(
        !(count == 0 && previous_mean != 0.0),
        "count is the count before the new value"
    );
    previous_mean + (value - previous_mean) / (count + 1) as f64
}

/// Online population variance update from running statistics and the next value.
///
/// `count` is the number of values observed *before* `value`.
pub fn online_variance(previous_variance: f64, previous_mean: f64, count: u64, value: f64) -> f64 {
    assert!(
        !(count == 0 && (previous_mean != 0.0 || previous_variance != 0.0)),
        "count is the count before the new value"
    );

    let next_count = count + 1;
    let next_count_f = next_count as f64;

    // Reconstruct the prior sum of squared deviations from the population
    // variance, then apply the Welford update.
    let previous_m2 = count as f64 * previous_variance;
    let delta = value - previous_mean;
    let next_mean = previous_mean + (delta / next_count_f);
    let next_m2 = previous_m2 + delta * (value - next_mean);

    next_m2 / next_count_f
}
