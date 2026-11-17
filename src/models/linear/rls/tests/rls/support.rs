//! RLS test support: shared samples, state/bit assertions, scalar/WLS diagnostic
//! machinery, the independent scalar recurrence and literal-objective oracle,
//! and shared f32/f64 reference fixtures. Precision-only objective/order
//! helpers live in `precision::support`.

use super::*;

pub(super) fn sample(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

pub(super) fn state(w: &[f32], p: &[&[f32]]) -> RlsState {
    RlsState {
        w: Vector::from_slice(w),
        p: Matrix::from_rows(p).unwrap(),
    }
}

pub(super) fn assert_f32_bits_eq(actual: f32, expected: f32) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "expected bitwise equality: {actual} vs {expected}"
    );
}

pub(super) fn assert_state_bitwise_eq(actual: &RlsState, expected: &RlsState) {
    assert_eq!(actual.w.len(), expected.w.len(), "w length mismatch");
    for (actual, expected) in actual.w.as_slice().iter().zip(expected.w.as_slice()) {
        assert_f32_bits_eq(*actual, *expected);
    }
    assert_eq!(
        actual.p.as_slice().len(),
        expected.p.as_slice().len(),
        "P length mismatch"
    );
    for (actual, expected) in actual.p.as_slice().iter().zip(expected.p.as_slice()) {
        assert_f32_bits_eq(*actual, *expected);
    }
}

pub(super) struct RefState {
    w: Vec<f32>,
    p: Vec<Vec<f32>>,
}

pub(super) fn ref_initial(dimension: usize, delta: f32) -> RefState {
    RefState {
        w: vec![0.0; dimension],
        p: (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|col| if row == col { delta } else { 0.0 })
                    .collect()
            })
            .collect(),
    }
}

pub(super) fn ref_step(state: &RefState, x: &[f32], y: f32, lambda: f32) -> RefState {
    let v: Vec<f32> = state
        .p
        .iter()
        .map(|row| row.iter().zip(x).map(|(p, x)| p * x).sum())
        .collect();
    let denominator = lambda + x.iter().zip(&v).map(|(x, v)| x * v).sum::<f32>();
    let prediction = state.w.iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
    let error = y - prediction;
    let next_w: Vec<f32> = state
        .w
        .iter()
        .zip(&v)
        .map(|(w, v)| w + (v / denominator) * error)
        .collect();
    let next_p: Vec<Vec<f32>> = state
        .p
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(col, p)| (p - (v[row] * v[col]) / denominator) / lambda)
                .collect()
        })
        .collect();
    RefState {
        w: next_w,
        p: next_p,
    }
}

pub(super) fn to_state(reference: &RefState) -> RlsState {
    let rows: Vec<&[f32]> = reference.p.iter().map(Vec::as_slice).collect();
    RlsState {
        w: Vector::from_slice(&reference.w),
        p: Matrix::from_rows(&rows).unwrap(),
    }
}

#[path = "../../../../../../benches/common/fixture_rng.rs"]
mod fixture_rng;

pub(super) use fixture_rng::Lcg64 as Lcg;

pub(super) fn to_f64_vec(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

// --- independent literal-objective WLS oracle (f64) --------------------------
//
// `ref_step` above is the production recurrence written out in scalars: it pins
// the wiring bitwise but shares the derivation, so it cannot catch a wrong
// gain / weighting / prior convention. The helpers below instead evaluate the
// *final weighted objective literally* for the last observation's position:
//
//   J_N(w) = Σ_j λ^{N-1-j} (y_j − wᵀx_j)² + (λ^N/δ)‖w‖²
//   A_N = Σ_j λ^{N-1-j} x_j x_jᵀ + (λ^N/δ) I
//   b_N = Σ_j λ^{N-1-j} y_j x_j
//
// The recurrence `A = λA + x xᵀ` / `b = λb + y x` deliberately does not appear
// here: if it did, this oracle would reproduce the implementation's derivation
// and stop being independent. It runs in `f64` and solves with Gaussian
// elimination, a different precision and algorithm from the `f32` recursion.

/// Evaluates `A_N` and `b_N` by summing each observation literally with its
/// absolute weight `λ^(N-1-j)`; the prior `λ^N/δ` is added separately.
pub(super) fn objective_normal_equations(
    lambda: f64,
    delta: f64,
    xs: &[Vec<f64>],
    ys: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = xs.len();
    let d = xs[0].len();
    let mut a = vec![vec![0.0_f64; d]; d];
    let mut b = vec![0.0_f64; d];
    for (j, (x, &y)) in xs.iter().zip(ys).enumerate() {
        // Absolute position from the end: the last observation has weight λ^0.
        let weight = lambda.powi((n - 1 - j) as i32);
        for row in 0..d {
            b[row] += weight * y * x[row];
            for col in 0..d {
                a[row][col] += weight * x[row] * x[col];
            }
        }
    }
    // The prior is a separate ridge term, not accumulated with the data.
    let prior = lambda.powi(n as i32) / delta;
    for (k, row) in a.iter_mut().enumerate() {
        row[k] += prior;
    }
    (a, b)
}

/// Gaussian elimination with partial pivoting. Panics on a singular system;
/// callers choose non-singular cases.
// Index arithmetic is inherent to elimination: the update reads and writes the
// same rows, which a single iterator cannot express without aliasing `m`.
#[allow(clippy::needless_range_loop)]
pub(super) fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        assert!(
            m[col][col].abs() > 0.0,
            "singular system in the D1 oracle; choose a non-singular case"
        );
        for row in (col + 1)..n {
            let factor = m[row][col] / m[col][col];
            for k in col..n {
                m[row][k] -= factor * m[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut w = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= m[row][k] * w[k];
        }
        w[row] = sum / m[row][row];
    }
    w
}

/// `max|A w − b|` scaled by `1 + max|b|`; validates `w` without a solve.
pub(super) fn max_abs_normal_residual(a: &[Vec<f64>], b: &[f64], w: &[f64]) -> f64 {
    let d = w.len();
    let mut max_residual = 0.0_f64;
    for row in 0..d {
        let dot: f64 = (0..d).map(|col| a[row][col] * w[col]).sum();
        max_residual = max_residual.max((dot - b[row]).abs());
    }
    let scale = 1.0 + b.iter().fold(0.0_f64, |m, value| m.max(value.abs()));
    max_residual / scale
}

/// `max|A · P_rec − I|`, validating that the recurrence's `P` inverts the
/// literal-objective `A`. Uses the `f32` `P` cast to `f64`.
pub(super) fn covariance_residual(a: &[Vec<f64>], p_rec: &Matrix) -> f64 {
    let d = p_rec.rows();
    let mut max_residual = 0.0_f64;
    for (row, a_row) in a.iter().enumerate() {
        for col in 0..d {
            let product: f64 = a_row
                .iter()
                .enumerate()
                .map(|(k, &coefficient)| coefficient * f64::from(p_rec.get(k, col).unwrap()))
                .sum();
            let identity = if row == col { 1.0 } else { 0.0 };
            max_residual = max_residual.max((product - identity).abs());
        }
    }
    max_residual
}

/// `D = 1` closed form: returns the scalar `(A_t, b_t)` so callers can form
/// `w_t = b_t/A_t` and `P_t = 1/A_t`.
pub(super) fn d1_closed_form(lambda: f64, delta: f64, xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len();
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    for j in 0..n {
        let weight = lambda.powi((n - 1 - j) as i32);
        a += weight * xs[j] * xs[j];
        b += weight * xs[j] * ys[j];
    }
    a += lambda.powi(n as i32) / delta;
    (a, b)
}

pub(super) fn oracle_solution(
    lambda: f64,
    delta: f64,
    xs: &[Vec<f32>],
    ys: &[f32],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let xs64: Vec<Vec<f64>> = xs.iter().map(|x| to_f64_vec(x)).collect();
    let ys64 = to_f64_vec(ys);
    let (a, b) = objective_normal_equations(lambda, delta, &xs64, &ys64);
    let w = solve(&a, &b);
    (a, b, w)
}

pub(super) fn run_sequence(model: &Rls, xs: &[Vec<f32>], ys: &[f32]) -> RlsState {
    let mut state = model.initial_state();
    for (x, &y) in xs.iter().zip(ys) {
        state = model.update(&state, &sample(x, y)).unwrap();
    }
    state
}

pub(super) fn assert_weights_match_oracle(state: &RlsState, oracle: &[f64], tolerance: f64) {
    let scale = 1.0 + oracle.iter().fold(0.0_f64, |m, value| m.max(value.abs()));
    for (index, (&actual, &expected)) in state.w.as_slice().iter().zip(oracle).enumerate() {
        let actual = f64::from(actual);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "w[{index}]: recurrence {actual} vs objective oracle {expected} \
             (tolerance {tolerance} relative to scale {scale})"
        );
    }
}

/// Asserts the production recurrence matches a literal final-objective solve.
/// `weight_tolerance` is relative to `1 + ‖w_oracle‖∞`; `covariance_tolerance`
/// bounds `‖A_oracle·P − I‖∞`. Neither is bitwise: `f32` recursion vs `f64`
/// direct solve plus conditioning.
pub(super) fn assert_matches_literal_objective(
    model: &Rls,
    xs: &[Vec<f32>],
    ys: &[f32],
    weight_tolerance: f64,
    covariance_tolerance: f64,
) {
    let (a, b, w_oracle) =
        oracle_solution(f64::from(model.lambda()), f64::from(model.delta()), xs, ys);
    let state = run_sequence(model, xs, ys);
    assert_weights_match_oracle(&state, &w_oracle, weight_tolerance);
    assert!(
        max_abs_normal_residual(&a, &b, &w_oracle) <= 1e-9,
        "the f64 oracle solution does not satisfy its own normal equations"
    );
    let residual = covariance_residual(&a, &state.p);
    assert!(
        residual <= covariance_tolerance,
        "covariance residual {residual} exceeds {covariance_tolerance} \
         (‖A·P − I‖∞; A from the literal objective)"
    );
}

pub(super) fn sinusoidal_data(d: usize, n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for k in 0..n {
        let x: Vec<f32> = (0..d)
            .map(|j| (((k + 1) * (j + 1)) as f32 * 0.37).sin())
            .collect();
        let y = (k as f32 * 0.29).cos() + 0.25 * x[0];
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

pub(super) fn phase_shifted_data(d: usize, n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for k in 0..n {
        let x: Vec<f32> = (0..d)
            .map(|j| {
                let frequency = 0.31 + 0.23 * j as f32;
                let phase = 0.7 * j as f32;
                (k as f32 * frequency + phase).sin()
            })
            .collect();
        let y = (k as f32 * 0.17).sin();
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

pub(super) fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, value| m.max(value.abs()))
}

pub(super) fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
}

/// Induced infinity norm: the maximum absolute row sum.
pub(super) fn matrix_infinity_norm(matrix: &Matrix) -> f64 {
    let d = matrix.rows();
    let mut max_row_sum = 0.0_f64;
    for row in 0..d {
        let sum: f64 = (0..d)
            .map(|col| f64::from(matrix.get(row, col).unwrap()).abs())
            .sum();
        max_row_sum = max_row_sum.max(sum);
    }
    max_row_sum
}

/// Diagonal entry `matrix[index][index]`.
pub(super) fn diagonal(matrix: &Matrix, index: usize) -> f64 {
    f64::from(matrix.get(index, index).unwrap())
}

/// `max|P[i,j] − P[j,i]|`; expected to be exactly `0.0` for an RLS `P`.
pub(super) fn symmetry_residual(matrix: &Matrix) -> f64 {
    let mut max = 0.0_f64;
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            let difference =
                f64::from(matrix.get(row, col).unwrap()) - f64::from(matrix.get(col, row).unwrap());
            max = max.max(difference.abs());
        }
    }
    max
}

/// Runs the ordered sequence and returns the last committed state plus the first
/// failure `(0-based observation index, error)`, if any.
pub(super) fn run_until_first_failure(
    model: &Rls,
    xs: &[Vec<f32>],
    ys: &[f32],
) -> (RlsState, Option<(usize, RlsError)>) {
    let mut state = model.initial_state();
    for (index, (x, &y)) in xs.iter().zip(ys).enumerate() {
        match model.update(&state, &sample(x, y)) {
            Ok(next) => state = next,
            Err(error) => return (state, Some((index, error))),
        }
    }
    (state, None)
}

/// Test-side `d = λ + xᵀPx` (diagnostic; the production denominator is not
/// exposed).
pub(super) fn denominator(model: &Rls, state: &RlsState, x: &[f32]) -> f64 {
    let dimension = state.p.rows();
    let mut quadratic = 0.0_f64;
    for (row, &x_row) in x.iter().enumerate().take(dimension) {
        let mut px = 0.0_f64;
        for (col, &x_col) in x.iter().enumerate().take(dimension) {
            px += f64::from(state.p.get(row, col).unwrap()) * f64::from(x_col);
        }
        quadratic += f64::from(x_row) * px;
    }
    f64::from(model.lambda()) + quadratic
}

/// `A_ref^{-1}` as rows, by solving `A p_k = e_k` for each unit column.
pub(super) fn reference_covariance(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let dimension = a.len();
    let columns: Vec<Vec<f64>> = (0..dimension)
        .map(|col| {
            let mut basis = vec![0.0_f64; dimension];
            basis[col] = 1.0;
            solve(a, &basis)
        })
        .collect();
    (0..dimension)
        .map(|row| (0..dimension).map(|col| columns[col][row]).collect())
        .collect()
}

pub(super) fn default_true_weights(dimension: usize) -> Vec<f32> {
    match dimension {
        1 => vec![0.7],
        2 => vec![0.7, -1.3],
        4 => vec![0.7, -1.3, 0.4, 2.1],
        other => vec![1.0; other],
    }
}

pub(super) fn linear_targets(xs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(weights)
                .map(|(value, weight)| value * weight)
                .sum()
        })
        .collect()
}

/// Period-4 orthonormal pair with exact empirical correlation `ρ`:
/// `Σu² = Σw² = 1`, `Σuw = 0`, so `Σx xᵀ = [[1,ρ],[ρ,1]]`.
pub(super) fn correlated_data(n: usize, rho: f64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let scale = (2.0_f64 / n as f64).sqrt();
    let complement = (1.0 - rho * rho).sqrt();
    let mut xs = Vec::with_capacity(n);
    for t in 0..n {
        let angle = std::f64::consts::PI * t as f64 / 2.0;
        let u = scale * angle.cos();
        let w = scale * angle.sin();
        xs.push(vec![u as f32, (rho * u + complement * w) as f32]);
    }
    let ys = linear_targets(&xs, &default_true_weights(2));
    (xs, ys)
}

/// Phase-shifted design with one axis permanently zeroed.
pub(super) fn unexcited_axis_data(d: usize, n: usize, axis: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let (mut xs, _) = phase_shifted_data(d, n);
    for x in &mut xs {
        x[axis] = 0.0;
    }
    let mut weights = default_true_weights(d);
    weights[axis] = 0.0;
    let ys = linear_targets(&xs, &weights);
    (xs, ys)
}

/// Persistently exciting design whose target switches from `w_a` to `w_b` at
/// `split`.
pub(super) fn regime_data(
    d: usize,
    n: usize,
    split: usize,
    w_a: &[f32],
    w_b: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let (xs, _) = phase_shifted_data(d, n);
    let ys = xs
        .iter()
        .enumerate()
        .map(|(index, x)| {
            let weights = if index < split { w_a } else { w_b };
            x.iter()
                .zip(weights)
                .map(|(value, weight)| value * weight)
                .sum()
        })
        .collect();
    (xs, ys)
}

/// `‖w − target‖∞` against an `f64` reference such as the pooled oracle.
pub(super) fn weight_distance_f64(state: &RlsState, target: &[f64]) -> f64 {
    state
        .w
        .as_slice()
        .iter()
        .zip(target)
        .fold(0.0_f64, |max, (&actual, &expected)| {
            max.max((f64::from(actual) - expected).abs())
        })
}

pub(super) fn scale_features(xs: &[Vec<f32>], factors: &[f32]) -> Vec<Vec<f32>> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(factors)
                .map(|(value, factor)| value * factor)
                .collect()
        })
        .collect()
}

pub(super) fn oracle_predictions(weights: &[f64], xs: &[Vec<f32>]) -> Vec<f64> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(weights)
                .map(|(&value, &weight)| f64::from(value) * weight)
                .sum()
        })
        .collect()
}

pub(super) fn prediction_error(state: &RlsState, xs: &[Vec<f32>], weights: &[f64]) -> f64 {
    let references = oracle_predictions(weights, xs);
    xs.iter()
        .zip(&references)
        .fold(0.0_f64, |max, (x, &reference)| {
            let actual: f64 = state
                .w
                .as_slice()
                .iter()
                .zip(x)
                .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                .sum();
            max.max((actual - reference).abs() / (1.0 + reference.abs()))
        })
}

/// Exact condition number of a symmetric `2×2` matrix via closed-form
/// eigenvalues (not a generic eigensolver).
pub(super) fn kappa_2x2(a: &[Vec<f64>]) -> f64 {
    let trace = a[0][0] + a[1][1];
    let determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let half = trace / 2.0;
    let root = (half * half - determinant).max(0.0).sqrt();
    (half + root) / (half - root)
}

pub(super) fn max_abs_matrix(matrix: &Matrix) -> f64 {
    matrix
        .as_slice()
        .iter()
        .fold(0.0_f64, |max, &value| max.max(f64::from(value).abs()))
}

pub(super) fn sample64(features: &[f64], target: f64) -> Observation<RlsSample64> {
    Observation::new(RlsSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

pub(super) fn model64(dimension: usize, lambda: f64, delta: f64) -> Rls64 {
    Rls64::new(dimension, lambda, delta).unwrap()
}

pub(super) struct RefState64 {
    pub(super) w: Vec<f64>,
    pub(super) p: Vec<Vec<f64>>,
}

pub(super) fn ref_initial64(dimension: usize, delta: f64) -> RefState64 {
    RefState64 {
        w: vec![0.0; dimension],
        p: (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|column| if row == column { delta } else { 0.0 })
                    .collect()
            })
            .collect(),
    }
}

pub(super) fn ref_predict64(state: &RefState64, x: &[f64]) -> f64 {
    state.w.iter().zip(x).map(|(w, x)| w * x).sum()
}

pub(super) fn ref_predict(state: &RefState, x: &[f32]) -> f32 {
    state.w.iter().zip(x).map(|(w, x)| w * x).sum()
}

pub(super) fn ref_step64(state: &RefState64, x: &[f64], y: f64, lambda: f64) -> RefState64 {
    let v: Vec<f64> = state
        .p
        .iter()
        .map(|row| row.iter().zip(x).map(|(p, x)| p * x).sum())
        .collect();
    let denominator = lambda + x.iter().zip(&v).map(|(x, v)| x * v).sum::<f64>();
    let prediction = ref_predict64(state, x);
    let error = y - prediction;
    let next_w: Vec<f64> = state
        .w
        .iter()
        .zip(&v)
        .map(|(w, v)| w + (v / denominator) * error)
        .collect();
    let next_p: Vec<Vec<f64>> = state
        .p
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(column, p)| (p - (v[row] * v[column]) / denominator) / lambda)
                .collect()
        })
        .collect();
    RefState64 {
        w: next_w,
        p: next_p,
    }
}

pub(super) fn to_state64(reference: &RefState64) -> RlsState64 {
    let rows: Vec<&[f64]> = reference.p.iter().map(Vec::as_slice).collect();
    RlsState64::admit(
        Vector64::from_slice(&reference.w),
        Matrix64::from_rows(&rows).unwrap(),
    )
    .unwrap()
}

pub(super) fn assert_f64_bits_eq(actual: f64, expected: f64) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "expected bitwise equality: {actual} vs {expected}"
    );
}

pub(super) fn assert_state64_bitwise_eq(actual: &RlsState64, expected: &RlsState64) {
    assert_eq!(
        actual.weights().len(),
        expected.weights().len(),
        "w length mismatch"
    );
    for (actual, expected) in actual
        .weights()
        .as_slice()
        .iter()
        .zip(expected.weights().as_slice())
    {
        assert_f64_bits_eq(*actual, *expected);
    }
    assert_eq!(
        actual.covariance().as_slice().len(),
        expected.covariance().as_slice().len(),
        "P length mismatch"
    );
    for (actual, expected) in actual
        .covariance()
        .as_slice()
        .iter()
        .zip(expected.covariance().as_slice())
    {
        assert_f64_bits_eq(*actual, *expected);
    }
}

pub(super) fn run_a_pairs(model: &Rls64, pairs: &[(Vec<f64>, f64)]) -> RlsState64 {
    let mut state = model.initial_state();
    let mut reference = ref_initial64(model.dimension(), model.delta());
    for (index, (x, y)) in pairs.iter().enumerate() {
        assert_f64_bits_eq(
            model.predict(&state, &Vector64::from_slice(x)).unwrap(),
            ref_predict64(&reference, x),
        );
        state = model
            .update(&state, &sample64(x, *y))
            .unwrap_or_else(|error| panic!("A pair update failed at {index}: {error:?}"));
        reference = ref_step64(&reference, x, *y, model.lambda());
        assert_state64_bitwise_eq(&state, &to_state64(&reference));
    }
    state
}

pub(super) fn learned_state64() -> (Rls64, RlsState64) {
    let model = model64(2, 0.9, 1000.0);
    let pairs = [
        (vec![1.0, -0.5], 2.0),
        (vec![0.25, 1.0], -1.0),
        (vec![-0.75, 0.5], 0.5),
    ];
    let state = run_a_pairs(&model, &pairs);
    (model, state)
}

pub(super) fn diagonal64(values: &[f64]) -> Matrix64 {
    Matrix64::from_fn(values.len(), values.len(), |row, column| {
        if row == column { values[row] } else { 0.0 }
    })
}

pub(super) fn state64_with_weights(model: &Rls64, weights: &[f64]) -> RlsState64 {
    let diagonal = vec![1.0; model.dimension()];
    RlsState64::admit(Vector64::from_slice(weights), diagonal64(&diagonal)).unwrap()
}

pub(super) fn snapshot_bits(snapshot: &RlsServing32) -> Vec<u32> {
    snapshot
        .weights()
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}
