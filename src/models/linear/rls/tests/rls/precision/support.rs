//! Precision-only helpers for the RLS precision/objective/order evidence:
//! objective assertions, prefix/pair runners, permutation/matrix helpers, and
//! order/objective diagnostics. Shared scalar/WLS machinery stays in the root
//! `support` module.

use super::super::support::*;
use super::super::*;

pub(super) fn covariance_residual64(a: &[Vec<f64>], p_rec: &Matrix64) -> f64 {
    let d = p_rec.rows();
    let mut max_residual = 0.0_f64;
    for (row, a_row) in a.iter().enumerate() {
        for column in 0..d {
            let product: f64 = a_row
                .iter()
                .enumerate()
                .map(|(k, &coefficient)| coefficient * p_rec.get(k, column).unwrap())
                .sum();
            let identity = if row == column { 1.0 } else { 0.0 };
            max_residual = max_residual.max((product - identity).abs());
        }
    }
    max_residual
}

pub(super) fn assert_a_objective(
    lambda: f64,
    delta: f64,
    state: &RlsState64,
    xs: &[Vec<f64>],
    ys: &[f64],
    weight_tolerance: f64,
    covariance_tolerance: f64,
) {
    let (a, b) = objective_normal_equations(lambda, delta, xs, ys);
    let w_ref = solve(&a, &b);
    let mut max_weight_error = 0.0_f64;
    for (index, (&actual, &expected)) in state.weights().as_slice().iter().zip(&w_ref).enumerate() {
        let weight_error = (actual - expected).abs() / expected.abs().max(1.0);
        max_weight_error = max_weight_error.max(weight_error);
        assert!(
            weight_error <= weight_tolerance,
            "A w[{index}]: recurrence {actual} vs objective oracle {expected} \
             (scaled error {weight_error}, tolerance {weight_tolerance})"
        );
    }
    assert!(max_weight_error <= weight_tolerance);
    assert!(
        max_abs_normal_residual(&a, &b, &w_ref) <= 1e-12,
        "the native-f64 oracle solution does not satisfy its own normal equations"
    );
    let residual = covariance_residual64(&a, state.covariance());
    assert!(
        residual <= covariance_tolerance,
        "A covariance residual {residual} exceeds {covariance_tolerance}"
    );
}

pub(super) fn assert_c_objective(
    lambda: f64,
    delta: f64,
    state: &RlsState,
    xs: &[Vec<f32>],
    ys: &[f32],
) {
    let (a, b, w_ref) = oracle_solution(lambda, delta, xs, ys);
    assert_weights_match_oracle(state, &w_ref, 5e-3);
    assert!(
        max_abs_normal_residual(&a, &b, &w_ref) <= 1e-9,
        "the f32-data objective oracle does not satisfy its own normal equations"
    );
    let residual = covariance_residual(&a, &state.p);
    assert!(
        residual <= 1e-2,
        "C covariance residual {residual} exceeds 1e-2"
    );
}

pub(super) fn sinusoidal_data64(d: usize, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for k in 0..n {
        let x: Vec<f64> = (0..d)
            .map(|j| (((k + 1) * (j + 1)) as f64 * 0.37_f64).sin())
            .collect();
        let y = (k as f64 * 0.29_f64).cos() + 0.25_f64 * x[0];
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

pub(super) fn run_a_prefixes(model: &Rls64, xs: &[Vec<f64>], ys: &[f64]) -> RlsState64 {
    let mut state = model.initial_state();
    let mut reference = ref_initial64(model.dimension(), model.delta());
    assert_state64_bitwise_eq(&state, &to_state64(&reference));
    for (index, (x, &y)) in xs.iter().zip(ys).enumerate() {
        let expected_prediction = ref_predict64(&reference, x);
        let actual_prediction = model
            .predict(&state, &Vector64::from_slice(x))
            .unwrap_or_else(|error| panic!("A prediction failed at row {index}: {error:?}"));
        assert_f64_bits_eq(actual_prediction, expected_prediction);
        state = model
            .update(&state, &sample64(x, y))
            .unwrap_or_else(|error| panic!("A update failed at row {index}: {error:?}"));
        reference = ref_step64(&reference, x, y, model.lambda());
        assert_state64_bitwise_eq(&state, &to_state64(&reference));
        assert_a_objective(
            model.lambda(),
            model.delta(),
            &state,
            &xs[..=index],
            &ys[..=index],
            1e-12,
            1e-10,
        );
    }
    state
}

pub(super) fn run_c_prefixes(model: &Rls, xs: &[Vec<f32>], ys: &[f32]) -> RlsState {
    let mut state = model.initial_state();
    let mut reference = ref_initial(model.dimension(), model.delta());
    for (index, (x, &y)) in xs.iter().zip(ys).enumerate() {
        let expected_prediction = ref_predict(&reference, x);
        let actual_prediction = model
            .predict(&state, &Vector::from_slice(x))
            .unwrap_or_else(|error| panic!("C prediction failed at row {index}: {error:?}"));
        assert_f32_bits_eq(actual_prediction, expected_prediction);
        state = model
            .update(&state, &sample(x, y))
            .unwrap_or_else(|error| panic!("C update failed at row {index}: {error:?}"));
        reference = ref_step(&reference, x, y, model.lambda());
        assert_state_bitwise_eq(&state, &to_state(&reference));
        assert_c_objective(
            f64::from(model.lambda()),
            f64::from(model.delta()),
            &state,
            &xs[..=index],
            &ys[..=index],
        );
    }
    state
}

pub(super) fn run_c_pairs(model: &Rls, pairs: &[(Vec<f32>, f32)]) -> RlsState {
    let mut state = model.initial_state();
    let mut reference = ref_initial(model.dimension(), model.delta());
    for (index, (x, y)) in pairs.iter().enumerate() {
        assert_f32_bits_eq(
            model.predict(&state, &Vector::from_slice(x)).unwrap(),
            ref_predict(&reference, x),
        );
        state = model
            .update(&state, &sample(x, *y))
            .unwrap_or_else(|error| panic!("C pair update failed at {index}: {error:?}"));
        reference = ref_step(&reference, x, *y, model.lambda());
        assert_state_bitwise_eq(&state, &to_state(&reference));
    }
    state
}

pub(super) fn permutation_pairs64(dimension: usize) -> Vec<(Vec<f64>, f64)> {
    let (xs, ys) = sinusoidal_data64(dimension, 30);
    xs.into_iter().zip(ys).collect()
}

pub(super) fn permutation_pairs32() -> Vec<(Vec<f32>, f32)> {
    let (xs, ys) = sinusoidal_data(2, 30);
    xs.into_iter().zip(ys).collect()
}

pub(super) fn permuted<T: Clone>(pairs: &[(Vec<T>, T)]) -> Vec<(Vec<T>, T)> {
    let n = pairs.len();
    (0..n).map(|k| pairs[(7 * k + 13) % n].clone()).collect()
}

pub(super) fn matrix_rows64(matrix: &Matrix64) -> Vec<Vec<f64>> {
    (0..matrix.rows())
        .map(|row| {
            (0..matrix.cols())
                .map(|column| matrix.get(row, column).unwrap())
                .collect()
        })
        .collect()
}

pub(super) fn matrix_rows32(matrix: &Matrix) -> Vec<Vec<f64>> {
    (0..matrix.rows())
        .map(|row| {
            (0..matrix.cols())
                .map(|column| f64::from(matrix.get(row, column).unwrap()))
                .collect()
        })
        .collect()
}

pub(super) fn matrix_infinity_norm64(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

pub(super) fn matrix_difference_infinity_norm64(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(left, right)| (left - right).abs())
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max)
}

pub(super) fn a_order_diagnostics(
    forward: &RlsState64,
    permuted: &RlsState64,
) -> (f64, f64, f64, f64) {
    let d_w = max_abs_difference(forward.weights().as_slice(), permuted.weights().as_slice());
    let s_w = 1.0 + infinity_norm(forward.weights().as_slice());
    let forward_p = matrix_rows64(forward.covariance());
    let permuted_p = matrix_rows64(permuted.covariance());
    let d_p = matrix_difference_infinity_norm64(&forward_p, &permuted_p);
    let s_p = 1.0 + matrix_infinity_norm64(&forward_p);
    (d_w, s_w, d_p, s_p)
}

pub(super) fn c_order_diagnostics(forward: &RlsState, permuted: &RlsState) -> (f64, f64, f64, f64) {
    let forward_w: Vec<f64> = forward.w.as_slice().iter().map(|&v| f64::from(v)).collect();
    let permuted_w: Vec<f64> = permuted
        .w
        .as_slice()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let d_w = max_abs_difference(&forward_w, &permuted_w);
    let s_w = 1.0 + infinity_norm(&forward_w);
    let forward_p = matrix_rows32(&forward.p);
    let permuted_p = matrix_rows32(&permuted.p);
    let d_p = matrix_difference_infinity_norm64(&forward_p, &permuted_p);
    let s_p = 1.0 + matrix_infinity_norm64(&forward_p);
    (d_w, s_w, d_p, s_p)
}

pub(super) fn a_objective_for_pairs(
    model: &Rls64,
    state: &RlsState64,
    pairs: &[(Vec<f64>, f64)],
    weight_tolerance: f64,
    covariance_tolerance: f64,
) {
    let xs: Vec<Vec<f64>> = pairs.iter().map(|(x, _)| x.clone()).collect();
    let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    assert_a_objective(
        model.lambda(),
        model.delta(),
        state,
        &xs,
        &ys,
        weight_tolerance,
        covariance_tolerance,
    );
}

pub(super) fn c_objective_for_pairs(model: &Rls, state: &RlsState, pairs: &[(Vec<f32>, f32)]) {
    let xs: Vec<Vec<f32>> = pairs.iter().map(|(x, _)| x.clone()).collect();
    let ys: Vec<f32> = pairs.iter().map(|(_, y)| *y).collect();
    assert_c_objective(
        f64::from(model.lambda()),
        f64::from(model.delta()),
        state,
        &xs,
        &ys,
    );
}
