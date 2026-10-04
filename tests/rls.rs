//! RLS behavior tests.
//!
//! Two references are used, for two different purposes:
//!
//! - `ref_step` is a scalar `Vec<f32>` implementation of the production
//!   recurrence, accumulating in the same order as the library. Every state
//!   comparison against it is bitwise, because both paths use the same
//!   operations in the same order.
//! - the literal-objective oracle (`objective_normal_equations` plus an `f64`
//!   direct solve) evaluates the final weighted least-squares objective without
//!   using the information-matrix recurrence. Comparisons against it are
//!   tolerance-based, because the production `f32` recursion and the `f64`
//!   direct solve differ in precision and reduction order.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Matrix, Vector};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{StateModel, process_batch, process_one, process_stream};
use seqvex::models::online::rls::{Rls, RlsBatchError, RlsError, RlsSample, RlsState};

fn model(dimension: usize) -> Rls {
    Rls::new(dimension, 0.99, 1.0e4).unwrap()
}

fn sample(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn state(w: &[f32], p: &[&[f32]]) -> RlsState {
    RlsState {
        w: Vector::from_slice(w),
        p: Matrix::from_rows(p).unwrap(),
    }
}

fn assert_f32_bits_eq(actual: f32, expected: f32) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "expected bitwise equality: {actual} vs {expected}"
    );
}

fn assert_state_bitwise_eq(actual: &RlsState, expected: &RlsState) {
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

// --- independent scalar reference -------------------------------------------

struct RefState {
    w: Vec<f32>,
    p: Vec<Vec<f32>>,
}

fn ref_initial(dimension: usize, delta: f32) -> RefState {
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

fn ref_step(state: &RefState, x: &[f32], y: f32, lambda: f32) -> RefState {
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

fn to_state(reference: &RefState) -> RlsState {
    let rows: Vec<&[f32]> = reference.p.iter().map(Vec::as_slice).collect();
    RlsState {
        w: Vector::from_slice(&reference.w),
        p: Matrix::from_rows(&rows).unwrap(),
    }
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

/// Fixed-seed linear congruential generator (mirrors the style of
/// `GruParameters::deterministic`); deterministic pseudorandom test inputs only.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
    }
}

fn to_f64_vec(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

/// Evaluates `A_N` and `b_N` by summing each observation literally with its
/// absolute weight `λ^(N-1-j)`; the prior `λ^N/δ` is added separately.
fn objective_normal_equations(
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
fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
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
fn max_abs_normal_residual(a: &[Vec<f64>], b: &[f64], w: &[f64]) -> f64 {
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
fn covariance_residual(a: &[Vec<f64>], p_rec: &Matrix) -> f64 {
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
fn d1_closed_form(lambda: f64, delta: f64, xs: &[f64], ys: &[f64]) -> (f64, f64) {
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

fn oracle_solution(
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

fn run_sequence(model: &Rls, xs: &[Vec<f32>], ys: &[f32]) -> RlsState {
    let mut state = model.initial_state();
    for (x, &y) in xs.iter().zip(ys) {
        state = model.update(&state, &sample(x, y)).unwrap();
    }
    state
}

fn assert_weights_match_oracle(state: &RlsState, oracle: &[f64], tolerance: f64) {
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
fn assert_matches_literal_objective(
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

fn sinusoidal_data(d: usize, n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
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

fn phase_shifted_data(d: usize, n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
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

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, value| m.max(value.abs()))
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
}

/// Induced infinity norm: the maximum absolute row sum.
fn matrix_infinity_norm(matrix: &Matrix) -> f64 {
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

fn matrix_difference_infinity_norm(left: &Matrix, right: &Matrix) -> f64 {
    let d = left.rows();
    let mut max_row_sum = 0.0_f64;
    for row in 0..d {
        let sum: f64 = (0..d)
            .map(|col| {
                (f64::from(left.get(row, col).unwrap()) - f64::from(right.get(row, col).unwrap()))
                    .abs()
            })
            .sum();
        max_row_sum = max_row_sum.max(sum);
    }
    max_row_sum
}

// --- mathematical correctness -------------------------------------------------

#[test]
fn hand_computed_two_feature_updates_are_exact() {
    let model = Rls::new(2, 1.0, 1.0).unwrap();
    let mut state = model.initial_state();

    state = model.update(&state, &sample(&[1.0, 0.0], 1.0)).unwrap();
    assert_eq!(state.w.as_slice(), &[0.5, 0.0]);
    assert_eq!(state.p.as_slice(), &[0.5, 0.0, 0.0, 1.0]);

    state = model.update(&state, &sample(&[0.0, 1.0], 3.0)).unwrap();
    assert_eq!(state.w.as_slice(), &[0.5, 1.5]);
    assert_eq!(state.p.as_slice(), &[0.5, 0.0, 0.0, 0.5]);

    assert_eq!(
        model.predict(&state, &Vector::from_slice(&[1.0, 1.0])),
        Ok(2.0)
    );
}

#[test]
fn sequence_matches_independent_scalar_reference_bitwise() {
    let model = Rls::new(3, 0.99, 1.0e4).unwrap();
    let mut state = model.initial_state();
    let mut reference = ref_initial(3, 1.0e4);

    for step in 0..200 {
        let features = [
            (step as f32 * 0.37).sin(),
            (step as f32 * 0.11).cos(),
            (step as f32 * 0.53).sin(),
        ];
        let target = (step as f32 * 0.29).cos();
        let observation = sample(&features, target);

        state = model.update(&state, &observation).unwrap();
        reference = ref_step(&reference, &features, target, 0.99);
        assert_state_bitwise_eq(&state, &to_state(&reference));
    }
}

#[test]
fn predict_uses_committed_weights() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    let mut state = model.initial_state();
    state = model.update(&state, &sample(&[2.0], 4.0)).unwrap();
    let prediction = model.predict(&state, &Vector::from_slice(&[3.0])).unwrap();
    assert_f32_bits_eq(prediction, state.w.as_slice()[0] * 3.0);
}

// --- literal-objective WLS oracle (D1) ----------------------------------------

#[test]
fn wls_oracle_d1_closed_form() {
    // D = 1: A_t and b_t are scalars, so the normal equations reduce to the
    // closed form w_t = b_t/A_t and P_t = 1/A_t with
    //   A_t = Σ λ^{t-i} x_i² + λ^t/δ,   b_t = Σ λ^{t-i} x_i y_i.
    // The oracle is f64 and the recurrence f32, so the comparison is relative
    // (5e-3 on w, 1e-2 on |A·P − 1|), not bitwise.
    let cases: [(f32, Vec<f32>, Vec<f32>); 3] = [
        (0.9, vec![0.0; 12], (0..12).map(|i| i as f32).collect()),
        (
            0.9,
            vec![1.5; 30],
            (0..30).map(|i| (i as f32 * 0.3).sin()).collect(),
        ),
        (
            0.99,
            (0..40).map(|i| (i as f32 * 0.7).cos()).collect(),
            (0..40).map(|i| (i as f32 * 0.2).sin()).collect(),
        ),
    ];

    for (lambda, xs, ys) in cases {
        let delta = 1.0e3_f32;
        let model = Rls::new(1, lambda, delta).unwrap();
        let mut state = model.initial_state();
        for t in 1..=xs.len() {
            state = model
                .update(&state, &sample(&xs[t - 1..t], ys[t - 1]))
                .unwrap();

            let xs_t: Vec<f64> = xs[..t].iter().map(|&value| f64::from(value)).collect();
            let ys_t: Vec<f64> = ys[..t].iter().map(|&value| f64::from(value)).collect();
            let (a, b) = d1_closed_form(f64::from(lambda), f64::from(delta), &xs_t, &ys_t);
            let w_oracle = b / a;

            let w_rec = f64::from(state.w.as_slice()[0]);
            let p_rec = f64::from(state.p.get(0, 0).unwrap());
            assert!(
                (w_rec - w_oracle).abs() <= 5e-3 * (1.0 + w_oracle.abs()),
                "lambda={lambda} t={t}: w {w_rec} vs closed form {w_oracle}"
            );
            assert!(
                (p_rec * a - 1.0).abs() <= 1e-2,
                "lambda={lambda} t={t}: |A·P − 1| = {}",
                (p_rec * a - 1.0).abs()
            );
        }
    }
}

#[test]
fn wls_oracle_matches_lambda_one() {
    // λ = 1: uniform weights and a fixed ridge (1/δ)‖w‖². The 5e-3 relative w
    // tolerance and 1e-2 covariance bound cover f32 recursion vs f64 literal
    // objective plus conditioning; the residual is insensitive to the solver.
    let delta = 1.0e3_f32;
    for &d in &[2_usize, 4] {
        for &n in &[20_usize, 100] {
            let (xs, ys) = sinusoidal_data(d, n);
            let model = Rls::new(d, 1.0, delta).unwrap();
            assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
        }
    }
}

#[test]
fn wls_oracle_matches_lambda_less_than_one() {
    // λ < 1: weights λ^{N-i} decay and the prior λ^N/δ is a decaying ridge.
    // Tolerances are as above; this exercises the weighting/prior convention.
    let delta = 1.0e3_f32;
    for &lambda in &[0.9_f32, 0.99] {
        for &d in &[2_usize, 4] {
            for &n in &[20_usize, 60] {
                let (xs, ys) = sinusoidal_data(d, n);
                let model = Rls::new(d, lambda, delta).unwrap();
                assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
            }
        }
    }
}

#[test]
fn wls_oracle_full_rank_excitation() {
    // Phase-shifted sinusoids spread across all four coordinates, so A has no
    // near-zero eigenvalue and P stays well-conditioned.
    let (xs, ys) = phase_shifted_data(4, 50);
    let model = Rls::new(4, 0.99, 1.0e3).unwrap();
    let (a, _, _) = oracle_solution(0.99, 1.0e3, &xs, &ys);
    let state = run_sequence(&model, &xs, &ys);
    let residual = covariance_residual(&a, &state.p);
    assert!(
        residual <= 1e-2,
        "full-rank excitation covariance residual {residual}"
    );
}

#[test]
fn wls_oracle_partial_excitation() {
    // The last coordinate is never excited, so A becomes λ^N/δ there. Keeping
    // λ = 0.999 and N = 30 keeps A invertible and lets w match the direct solve;
    // the unexcited weight is exactly zero (v and b are zero on that axis).
    let d = 4;
    let n = 30;
    let (mut xs, ys) = sinusoidal_data(d, n);
    for x in &mut xs {
        x[d - 1] = 0.0;
    }
    let model = Rls::new(d, 0.999, 1.0e3).unwrap();
    let (_, _, w_oracle) = oracle_solution(0.999, 1.0e3, &xs, &ys);
    let state = run_sequence(&model, &xs, &ys);
    assert_weights_match_oracle(&state, &w_oracle, 5e-3);
    assert_eq!(state.w.as_slice()[d - 1], 0.0);
}

#[test]
fn wls_oracle_repeated_observation() {
    // The same (x, y) repeated N times excites one direction only. δ = 1 makes
    // the prior significant, keeping A well-conditioned so w matches the solve.
    let x = vec![1.0_f32, 2.0];
    let y = 3.0_f32;
    let n = 40;
    let xs = vec![x; n];
    let ys = vec![y; n];
    let model = Rls::new(2, 0.9, 1.0).unwrap();
    assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
}

#[test]
fn wls_oracle_varying_observations() {
    // Deterministic pseudorandom observations (fixed LCG) exercise a general,
    // well-excited A at D = 3, N = 80.
    let mut lcg = Lcg::new(0x5eed_5eed_5eed_5eed);
    let d = 3;
    let n = 80;
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| lcg.next_f32()).collect())
        .collect();
    let ys: Vec<f32> = (0..n).map(|_| lcg.next_f32()).collect();
    let model = Rls::new(d, 0.99, 1.0e3).unwrap();
    assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
}

// --- dimension validation -----------------------------------------------------

#[test]
fn zero_dimension_is_rejected() {
    assert_eq!(Rls::new(0, 0.99, 1.0), Err(RlsError::ZeroDimension));
}

#[test]
fn dimension_mismatch_is_reported_for_features_weights_and_covariance() {
    let model = Rls::new(2, 0.99, 1.0).unwrap();
    let good = model.initial_state();

    assert_eq!(
        model.update(&good, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_w = state(&[0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
    assert_eq!(
        model
            .update(&short_w, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_rows = state(&[0.0, 0.0], &[&[1.0, 0.0]]);
    assert_eq!(
        model
            .update(&short_rows, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_cols = state(&[0.0, 0.0], &[&[1.0], &[0.0]]);
    assert_eq!(
        model
            .update(&short_cols, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    assert_eq!(
        model
            .predict(&good, &Vector::from_slice(&[1.0]))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

// --- parameter validation -----------------------------------------------------

#[test]
fn invalid_forgetting_factor_is_rejected() {
    for lambda in [0.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Rls::new(1, lambda, 1.0),
            Err(RlsError::InvalidForgettingFactor),
            "lambda = {lambda}"
        );
    }
    assert!(Rls::new(1, 1.0, 1.0).is_ok());
}

#[test]
fn invalid_initial_covariance_is_rejected() {
    for delta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Rls::new(1, 0.5, delta),
            Err(RlsError::InvalidInitialCovariance),
            "delta = {delta}"
        );
    }
}

// --- finite and denominator validation ---------------------------------------

#[test]
fn non_finite_observation_is_rejected() {
    let model = Rls::new(2, 1.0, 1.0).unwrap();
    let state = model.initial_state();
    assert_eq!(
        model
            .update(&state, &sample(&[f32::NAN, 0.0], 0.0))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
    assert_eq!(
        model
            .update(&state, &sample(&[1.0, 2.0], f32::INFINITY))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
    assert_eq!(
        model
            .predict(&state, &Vector::from_slice(&[f32::INFINITY, 0.0]))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
}

#[test]
fn non_positive_denominator_is_rejected() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    // P = [-1] makes d = λ + xᵀPx = 1 - 1 = 0.
    let corrupted = state(&[0.0], &[&[-1.0]]);
    assert_eq!(
        model.update(&corrupted, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::InvalidDenominator
    );

    // A non-finite P makes d non-finite.
    let nan_p = state(&[0.0], &[&[f32::NAN]]);
    assert_eq!(
        model.update(&nan_p, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::InvalidDenominator
    );
}

#[test]
fn non_finite_candidate_is_rejected() {
    // A tiny δ keeps d finite while w = f32::MAX overflows the prediction error.
    let model = Rls::new(1, 1.0, 1.0e-30).unwrap();
    let explosive = state(&[f32::MAX], &[&[1.0e-30]]);
    assert_eq!(
        model
            .update(&explosive, &sample(&[1.0], -f32::MAX))
            .unwrap_err(),
        RlsError::NonFiniteCandidate
    );
}

// --- atomicity ----------------------------------------------------------------

#[test]
fn failed_update_leaves_committed_state_bitwise_unchanged() {
    let model = Rls::new(2, 0.99, 10.0).unwrap();
    let mut committed = model
        .update(&model.initial_state(), &sample(&[1.0, 2.0], 1.0))
        .unwrap();
    let snapshot = committed.clone();

    assert!(model.update(&committed, &sample(&[1.0], 0.0)).is_err());
    assert_state_bitwise_eq(&committed, &snapshot);

    assert!(
        model
            .update(&committed, &sample(&[f32::NAN, 0.0], 0.0))
            .is_err()
    );
    assert_state_bitwise_eq(&committed, &snapshot);

    // The next valid observation continues from the untouched committed state.
    committed = model
        .update(&committed, &sample(&[0.5, -0.5], 1.0))
        .unwrap();
    let expected = model.update(&snapshot, &sample(&[0.5, -0.5], 1.0)).unwrap();
    assert_state_bitwise_eq(&committed, &expected);
}

#[test]
fn executor_failure_preserves_state_and_the_stream_continues() {
    let streaming_model = Rls::new(2, 0.99, 10.0).unwrap();
    let reference = streaming_model.clone();
    let initial = streaming_model.initial_state();
    let mut executor = StreamingExecutor::new(&streaming_model, initial);

    executor.process_one(&sample(&[1.0, 2.0], 1.0)).unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor.process_one(&sample(&[1.0], 0.0)).unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    assert_state_bitwise_eq(executor.state(), &committed);

    executor.process_one(&sample(&[0.5, -0.5], 1.0)).unwrap();
    let expected = reference
        .update(&committed, &sample(&[0.5, -0.5], 1.0))
        .unwrap();
    assert_state_bitwise_eq(executor.state(), &expected);
}

// --- sequential equivalence, reset, process_stream ----------------------------

#[test]
fn process_stream_matches_repeated_update_bitwise() {
    let model = Rls::new(2, 0.5, 4.0).unwrap();
    let observations = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[0.0, 1.0], -2.0),
        sample(&[0.5, 0.5], 3.0),
    ];

    let mut direct = model.initial_state();
    for observation in &observations {
        direct = model.update(&direct, observation).unwrap();
    }

    let streamed = process_stream(&model, model.initial_state(), observations, |_| {
        panic!("unexpected failure")
    });
    assert_state_bitwise_eq(&direct, &streamed);
}

#[test]
fn process_stream_continues_after_a_failure() {
    let model = Rls::new(2, 1.0, 4.0).unwrap();
    let observations = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[f32::NAN, 0.0], 1.0),
        sample(&[0.0, 1.0], 2.0),
    ];
    let mut failures = 0;

    let final_state = process_stream(&model, model.initial_state(), observations, |error| {
        assert_eq!(*error, RlsError::NonFiniteInput);
        failures += 1;
    });

    assert_eq!(failures, 1);
    let mut expected = model.initial_state();
    expected = model.update(&expected, &sample(&[1.0, 0.0], 1.0)).unwrap();
    expected = model.update(&expected, &sample(&[0.0, 1.0], 2.0)).unwrap();
    assert_state_bitwise_eq(&final_state, &expected);
}

#[test]
fn reset_starts_a_new_sequence() {
    let streaming_model = Rls::new(2, 0.9, 2.0).unwrap();
    let reference = streaming_model.clone();
    let initial = streaming_model.initial_state();
    let mut executor = StreamingExecutor::new(&streaming_model, initial.clone());

    executor.process_one(&sample(&[1.0, 2.0], 1.0)).unwrap();
    executor.reset(initial.clone());
    assert_state_bitwise_eq(executor.state(), &initial);

    executor.process_one(&sample(&[0.5, 0.5], 2.0)).unwrap();
    let expected = reference
        .update(&reference.initial_state(), &sample(&[0.5, 0.5], 2.0))
        .unwrap();
    assert_state_bitwise_eq(executor.state(), &expected);
}

// --- bounded micro-batch (foundation ordered fold) ----------------------------

#[test]
fn bounded_fold_matches_repeated_process_one_bitwise() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let batch = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[0.0, 1.0], -2.0),
        sample(&[0.5, 0.5], 3.0),
    ];

    let folded = process_batch(&model, model.initial_state(), batch.clone()).unwrap();

    let mut manual = model.initial_state();
    for observation in &batch {
        manual = process_one(&model, &manual, observation).unwrap();
    }
    assert_state_bitwise_eq(&folded, &manual);
}

#[test]
fn bounded_fold_stops_on_first_failure_and_keeps_last_valid_state() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let first = sample(&[1.0, 0.0], 1.0);
    let bad = sample(&[f32::NAN, 0.0], 1.0);
    let third = sample(&[0.0, 1.0], 2.0);

    let failure = process_batch(
        &model,
        model.initial_state(),
        vec![first.clone(), bad, third],
    )
    .unwrap_err();

    assert_eq!(failure.error, RlsError::NonFiniteInput);
    let expected = model.update(&model.initial_state(), &first).unwrap();
    assert_state_bitwise_eq(&failure.state, &expected);
}

// --- bounded reference batch (Rls::process_batch_reference) -------------------
//
// The RLS-local bounded reference batching contract: an ordered, bound-first,
// state-only fold over direct `Rls::update` calls. It is not the foundation
// `process_batch` (which stays unbounded); equivalence with the foundation fold
// is asserted bitwise rather than by delegating to it.

#[test]
fn bounded_reference_matches_reference_paths_bitwise() {
    for lambda in [1.0_f32, 0.9] {
        let model = Rls::new(2, lambda, 5.0).unwrap();
        let batch = vec![
            sample(&[1.0, 0.0], 1.0),
            sample(&[0.0, 1.0], -2.0),
            sample(&[0.5, 0.5], 3.0),
            sample(&[-0.25, 0.75], 0.5),
        ];

        let mut repeated = model.initial_state();
        for observation in &batch {
            repeated = model.update(&repeated, observation).unwrap();
        }

        let mut executor = StreamingExecutor::new(&model, model.initial_state());
        for observation in &batch {
            executor.process_one(observation).unwrap();
        }
        assert_state_bitwise_eq(executor.state(), &repeated);

        let folded = process_batch(&model, model.initial_state(), batch.iter().cloned()).unwrap();
        assert_state_bitwise_eq(&folded, &repeated);

        let batched = model
            .process_batch_reference(model.initial_state(), &batch, batch.len())
            .unwrap();
        assert_state_bitwise_eq(&batched, &repeated);
    }
}

#[test]
fn bounded_reference_matches_independent_oracle() {
    let model = Rls::new(3, 0.95, 1.0e3).unwrap();
    let xs: Vec<Vec<f32>> = (0..9)
        .map(|k| {
            vec![
                (k as f32 * 0.31).sin(),
                (k as f32 * 0.17).cos(),
                (k as f32 * 0.05).sin(),
            ]
        })
        .collect();
    let ys: Vec<f32> = (0..9).map(|k| (k as f32 * 0.23).cos()).collect();
    let batch: Vec<Observation<RlsSample>> =
        xs.iter().zip(&ys).map(|(x, &y)| sample(x, y)).collect();

    let batched = model
        .process_batch_reference(model.initial_state(), &batch, batch.len())
        .unwrap();

    // Independent scalar recurrence, bitwise.
    let mut reference = ref_initial(3, 1.0e3);
    for (x, &y) in xs.iter().zip(&ys) {
        reference = ref_step(&reference, x, y, 0.95);
    }
    assert_state_bitwise_eq(&batched, &to_state(&reference));

    // Independent literal-objective f64 oracle, tolerance-based.
    assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
}

#[test]
fn bounded_reference_permutation_semantics() {
    let d = 2;
    let n = 30;
    let delta = 1.0e3_f32;
    let (xs, ys) = sinusoidal_data(d, n);
    let permutation: Vec<usize> = (0..n).map(|index| (index * 7 + 13) % n).collect();
    let permuted_xs: Vec<Vec<f32>> = permutation.iter().map(|&index| xs[index].clone()).collect();
    let permuted_ys: Vec<f32> = permutation.iter().map(|&index| ys[index]).collect();

    let batch: Vec<Observation<RlsSample>> =
        xs.iter().zip(&ys).map(|(x, &y)| sample(x, y)).collect();
    let permuted_batch: Vec<Observation<RlsSample>> = permuted_xs
        .iter()
        .zip(&permuted_ys)
        .map(|(x, &y)| sample(x, y))
        .collect();

    // lambda = 1: the multiset fixes the minimizer, so permutations agree within
    // tolerance (not bitwise: the reduction order differs).
    let model_one = Rls::new(d, 1.0, delta).unwrap();
    let forward = model_one
        .process_batch_reference(model_one.initial_state(), &batch, batch.len())
        .unwrap();
    let permuted = model_one
        .process_batch_reference(
            model_one.initial_state(),
            &permuted_batch,
            permuted_batch.len(),
        )
        .unwrap();
    let weights = to_f64_vec(forward.w.as_slice());
    let difference = max_abs_difference(&weights, &to_f64_vec(permuted.w.as_slice()));
    assert!(
        difference <= 1e-4 * (1.0 + infinity_norm(&weights)),
        "lambda=1 batch permutation changed the fit by {difference}"
    );

    // lambda < 1: recency weighting makes order semantically significant.
    let model_below = Rls::new(d, 0.5, delta).unwrap();
    let ordered = model_below
        .process_batch_reference(model_below.initial_state(), &batch, batch.len())
        .unwrap();
    let reordered = model_below
        .process_batch_reference(
            model_below.initial_state(),
            &permuted_batch,
            permuted_batch.len(),
        )
        .unwrap();
    let difference = max_abs_difference(
        &to_f64_vec(ordered.w.as_slice()),
        &to_f64_vec(reordered.w.as_slice()),
    );
    assert!(
        difference > 1e-3,
        "lambda<1 batch permutation should change the weighted solution, difference {difference}"
    );
}

#[test]
fn bounded_reference_chunk_continuity_including_empty() {
    let model = Rls::new(2, 0.97, 5.0).unwrap();
    let observations: Vec<Observation<RlsSample>> = (0..7)
        .map(|k| {
            sample(
                &[(k as f32 * 0.4).sin(), (k as f32 * 0.25).cos()],
                (k as f32 * 0.3).cos(),
            )
        })
        .collect();

    let continuous = model
        .process_batch_reference(model.initial_state(), &observations, observations.len())
        .unwrap();

    let mut chunked = model.initial_state();
    for chunk in observations.chunks(2) {
        chunked = model
            .process_batch_reference(chunked, chunk, chunk.len())
            .unwrap();
        // An empty chunk is a no-op that returns the supplied state.
        chunked = model.process_batch_reference(chunked, &[], 0).unwrap();
    }
    assert_state_bitwise_eq(&chunked, &continuous);
}

#[test]
fn bounded_reference_prefixes_are_causal() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let observations: Vec<Observation<RlsSample>> = (0..8)
        .map(|k| {
            sample(
                &[(k as f32 * 0.2).sin(), (k as f32 * 0.6).cos()],
                k as f32 * 0.1,
            )
        })
        .collect();

    for prefix in [1_usize, 3, 5, 8] {
        let batched = model
            .process_batch_reference(model.initial_state(), &observations[..prefix], prefix)
            .unwrap();

        let mut repeated = model.initial_state();
        for observation in &observations[..prefix] {
            repeated = model.update(&repeated, observation).unwrap();
        }
        assert_state_bitwise_eq(&batched, &repeated);
    }
}

#[test]
fn bounded_reference_bound_rejection_and_empty_input() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let batch = vec![
        sample(&[0.1, 0.2], 1.0),
        sample(&[0.3, -0.4], 2.0),
        sample(&[0.5, 0.6], 3.0),
    ];

    // The exact bound succeeds.
    assert!(
        model
            .process_batch_reference(model.initial_state(), &batch, batch.len())
            .is_ok()
    );

    // Oversized: rejected before any transition; the moved-in state is
    // recovered bitwise without a defensive caller clone.
    let supplied = state(&[0.25, -0.5], &[&[1.0, 0.0], &[0.0, 1.0]]);
    match model.process_batch_reference(supplied.clone(), &batch, 2) {
        Err(RlsBatchError::BatchTooLarge {
            max,
            actual,
            state: returned,
        }) => {
            assert_eq!(max, 2);
            assert_eq!(actual, 3);
            assert_state_bitwise_eq(&returned, &supplied);
        }
        other => panic!("expected BatchTooLarge, got {other:?}"),
    }

    // max_batch == 0 accepts only the empty batch.
    match model.process_batch_reference(model.initial_state(), &batch, 0) {
        Err(RlsBatchError::BatchTooLarge { max, actual, .. }) => {
            assert_eq!(max, 0);
            assert_eq!(actual, 3);
        }
        other => panic!("expected BatchTooLarge for max_batch == 0, got {other:?}"),
    }

    let supplied = state(&[1.0, -1.0], &[&[2.0, 0.0], &[0.0, 3.0]]);
    let empty = model
        .process_batch_reference(supplied.clone(), &[], 0)
        .unwrap();
    assert_state_bitwise_eq(&empty, &supplied);
}

#[test]
fn bounded_reference_returns_supplied_state_unvalidated() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    // Dimension-mismatched and NaN-containing: not a valid RLS state anywhere.
    let corrupt = state(&[f32::NAN, 0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
    let batch = vec![sample(&[0.1, 0.2], 1.0), sample(&[0.3, 0.4], 2.0)];

    // Oversized: the bound is checked before any state validation, so the
    // corrupt state is returned bitwise unchanged (NaN bits included).
    match model.process_batch_reference(corrupt.clone(), &batch, 1) {
        Err(RlsBatchError::BatchTooLarge {
            state: returned, ..
        }) => assert_state_bitwise_eq(&returned, &corrupt),
        other => panic!("expected BatchTooLarge, got {other:?}"),
    }

    // Empty: returned bitwise unchanged without validation.
    let empty = model
        .process_batch_reference(corrupt.clone(), &[], 0)
        .unwrap();
    assert_state_bitwise_eq(&empty, &corrupt);
}

#[test]
fn bounded_reference_failure_state_matches_committed_prefix() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let good = |k: usize| {
        sample(
            &[(k as f32 * 0.3).sin(), (k as f32 * 0.2).cos()],
            k as f32 * 0.5,
        )
    };
    let bad = sample(&[f32::NAN, 0.0], 1.0);

    // Start from a committed state that is not the initial state, so a
    // whole-batch rollback to the supplied state would be observable.
    let base = model.update(&model.initial_state(), &good(9)).unwrap();

    for k in [0_usize, 2, 4] {
        let mut observations: Vec<Observation<RlsSample>> = (0..5).map(good).collect();
        observations[k] = bad.clone();

        match model.process_batch_reference(base.clone(), &observations, observations.len()) {
            Err(RlsBatchError::Transition {
                failed_index,
                state: returned,
                error,
            }) => {
                assert_eq!(failed_index, k);
                assert_eq!(error, RlsError::NonFiniteInput);

                // Both w and P bitwise-equal the fold over batch[..k]; the
                // supplied batch-entry state applies only when k == 0.
                let mut expected = base.clone();
                for observation in &observations[..k] {
                    expected = model.update(&expected, observation).unwrap();
                }
                assert_state_bitwise_eq(&returned, &expected);

                // The caller can continue from the returned state.
                let mut continued = returned.clone();
                let mut reference = expected;
                for observation in &observations[k + 1..] {
                    continued = model.update(&continued, observation).unwrap();
                    reference = model.update(&reference, observation).unwrap();
                }
                assert_state_bitwise_eq(&continued, &reference);
            }
            other => panic!("expected Transition at k={k}, got {other:?}"),
        }
    }
}

#[test]
fn bounded_reference_moved_state_recovery_from_both_errors() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let good = |k: usize| {
        sample(
            &[(k as f32 * 0.25).sin(), (k as f32 * 0.15).cos()],
            k as f32 * 0.2,
        )
    };
    let batch: Vec<Observation<RlsSample>> = (0..4).map(good).collect();

    // BatchTooLarge: recover the supplied state without a clone, then continue.
    let supplied = model.update(&model.initial_state(), &good(11)).unwrap();
    let recovered = match model.process_batch_reference(supplied.clone(), &batch, 1) {
        Err(RlsBatchError::BatchTooLarge { state, .. }) => state,
        other => panic!("expected BatchTooLarge, got {other:?}"),
    };
    assert_state_bitwise_eq(&recovered, &supplied);
    let continued = model
        .process_batch_reference(recovered, &batch[..1], 1)
        .unwrap();
    let expected = model.update(&supplied, &batch[0]).unwrap();
    assert_state_bitwise_eq(&continued, &expected);

    // Transition: recover the last committed prefix state, then continue.
    let mut failing = batch.clone();
    failing[1] = sample(&[f32::NAN, 0.0], 0.0);
    let recovered = match model.process_batch_reference(supplied.clone(), &failing, failing.len()) {
        Err(RlsBatchError::Transition {
            failed_index,
            state,
            ..
        }) => {
            assert_eq!(failed_index, 1);
            state
        }
        other => panic!("expected Transition, got {other:?}"),
    };
    let expected_prefix = model.update(&supplied, &batch[0]).unwrap();
    assert_state_bitwise_eq(&recovered, &expected_prefix);

    let continued = model
        .process_batch_reference(recovered, &batch[2..], batch.len() - 2)
        .unwrap();
    let mut reference = expected_prefix;
    for observation in &batch[2..] {
        reference = model.update(&reference, observation).unwrap();
    }
    assert_state_bitwise_eq(&continued, &reference);
}

#[test]
fn bounded_reference_transition_error_mapping() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();

    // Non-finite target, wrapped as a transition failure at its index.
    let non_finite = vec![sample(&[1.0, 0.0], 1.0), sample(&[0.0, 1.0], f32::INFINITY)];
    match model.process_batch_reference(model.initial_state(), &non_finite, non_finite.len()) {
        Err(RlsBatchError::Transition {
            failed_index,
            error,
            ..
        }) => {
            assert_eq!(failed_index, 1);
            assert_eq!(error, RlsError::NonFiniteInput);
        }
        other => panic!("expected Transition for a non-finite target, got {other:?}"),
    }

    // Wrong feature dimension, wrapped as a transition failure.
    let mismatched = vec![sample(&[1.0, 0.0], 1.0), sample(&[1.0], 2.0)];
    match model.process_batch_reference(model.initial_state(), &mismatched, mismatched.len()) {
        Err(RlsBatchError::Transition {
            failed_index,
            error,
            ..
        }) => {
            assert_eq!(failed_index, 1);
            assert_eq!(
                error,
                RlsError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
        }
        other => panic!("expected Transition for a dimension mismatch, got {other:?}"),
    }
}

// --- shared model, independent states (architecture evidence) -----------------

#[test]
fn one_shared_model_drives_independent_stream_states() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let sequence_a = vec![sample(&[1.0, 0.0], 1.0), sample(&[0.0, 1.0], 2.0)];
    let sequence_b = vec![sample(&[0.0, 1.0], 3.0), sample(&[1.0, 0.5], -1.0)];

    let mut state_a = model.initial_state();
    for observation in &sequence_a {
        state_a = process_one(&model, &state_a, observation).unwrap();
    }
    let state_b = process_stream(&model, model.initial_state(), sequence_b.clone(), |_| {
        panic!("unexpected failure")
    });

    assert_ne!(state_a, state_b);

    let mut reference_a = model.initial_state();
    for observation in &sequence_a {
        reference_a = model.update(&reference_a, observation).unwrap();
    }
    assert_state_bitwise_eq(&state_a, &reference_a);

    let mut reference_b = model.initial_state();
    for observation in &sequence_b {
        reference_b = model.update(&reference_b, observation).unwrap();
    }
    assert_state_bitwise_eq(&state_b, &reference_b);
}

// --- construction and sequence context ----------------------------------------

#[test]
fn initial_state_is_zero_weights_and_scaled_identity() {
    let state = model(3).initial_state();
    assert_eq!(state.w.as_slice(), &[0.0, 0.0, 0.0]);
    assert_eq!(
        state.p.as_slice(),
        &[1.0e4, 0.0, 0.0, 0.0, 1.0e4, 0.0, 0.0, 0.0, 1.0e4]
    );
}

#[test]
fn sequence_context_does_not_change_the_computation() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    let plain = model
        .update(&model.initial_state(), &sample(&[2.0], 4.0))
        .unwrap();
    let ordered = model
        .update(
            &model.initial_state(),
            &sample(&[2.0], 4.0)
                .with_sequence(seqvex::foundation::observation::SequenceNumber::new(7)),
        )
        .unwrap();
    assert_state_bitwise_eq(&plain, &ordered);
}

// --- long-run numerical stability ---------------------------------------------

#[test]
fn long_run_stays_finite_symmetric_and_matches_reference() {
    let model = Rls::new(4, 0.999, 1.0e3).unwrap();
    let dimension = model.dimension();
    let mut state = model.initial_state();
    let mut reference = ref_initial(dimension, 1.0e3);
    let mut max_prediction = 0.0_f32;

    for step in 0..10_000 {
        let features = [
            (step as f32 * 0.17).sin(),
            (step as f32 * 0.07).cos(),
            (step as f32 * 0.31).sin(),
            (step as f32 * 0.23).cos(),
        ];
        let target = (step as f32 * 0.13).sin();
        let observation = sample(&features, target);

        state = model.update(&state, &observation).unwrap();
        reference = ref_step(&reference, &features, target, 0.999);

        assert!(
            state.w.as_slice().iter().all(|value| value.is_finite()),
            "non-finite w at step {step}"
        );
        assert!(
            state.p.as_slice().iter().all(|value| value.is_finite()),
            "non-finite P at step {step}"
        );
        let prediction = model
            .predict(&state, &observation.value().features)
            .unwrap();
        max_prediction = max_prediction.max(prediction.abs());
    }

    assert_state_bitwise_eq(&state, &to_state(&reference));

    // Symmetry is preserved exactly: v_i * v_j == v_j * v_i in IEEE f32.
    for row in 0..dimension {
        for col in 0..dimension {
            assert_f32_bits_eq(
                state.p.get(row, col).unwrap(),
                state.p.get(col, row).unwrap(),
            );
        }
    }
    println!("RLS long run: max |prediction| = {max_prediction}");
    assert!(max_prediction.is_finite());
}

// --- ordering / permutation semantics (D4) ------------------------------------

#[test]
fn permutation_is_invariant_for_lambda_one() {
    // λ = 1 weights every observation equally, so the minimizer depends on the
    // multiset, not the order. The results agree within tolerance but not
    // bitwise: the f32 recurrence reduces in a different order for each
    // permutation. Tolerances: 1e-4 relative on w, 1e-3 relative on the induced
    // infinity norm of P, for this bounded, well-conditioned case.
    let d = 2;
    let n = 30;
    let delta = 1.0e3_f32;
    let (xs, ys) = sinusoidal_data(d, n);
    let permutation: Vec<usize> = (0..n).map(|index| (index * 7 + 13) % n).collect();
    let permuted_xs: Vec<Vec<f32>> = permutation.iter().map(|&index| xs[index].clone()).collect();
    let permuted_ys: Vec<f32> = permutation.iter().map(|&index| ys[index]).collect();

    let model = Rls::new(d, 1.0, delta).unwrap();
    let state_a = run_sequence(&model, &xs, &ys);
    let state_b = run_sequence(&model, &permuted_xs, &permuted_ys);

    let weights_a = to_f64_vec(state_a.w.as_slice());
    let weights_b = to_f64_vec(state_b.w.as_slice());
    let weight_difference = max_abs_difference(&weights_a, &weights_b);
    assert!(
        weight_difference <= 1e-4 * (1.0 + infinity_norm(&weights_a)),
        "w differs by {weight_difference} under a λ = 1 permutation"
    );

    let covariance_difference = matrix_difference_infinity_norm(&state_a.p, &state_b.p);
    assert!(
        covariance_difference <= 1e-3 * (1.0 + matrix_infinity_norm(&state_a.p)),
        "P differs by {covariance_difference} (infinity norm) under permutation"
    );

    // Stronger, solver-insensitive check: A from the literal λ = 1 objective
    // must invert each order's P.
    let (a, _, _) = oracle_solution(1.0, f64::from(delta), &xs, &ys);
    for (order, state) in [("A", &state_a), ("B", &state_b)] {
        let residual = covariance_residual(&a, &state.p);
        assert!(
            residual <= 1e-2,
            "order {order}: covariance residual {residual}"
        );
    }
}

#[test]
fn permutation_changes_the_weighted_solution_for_lambda_less_than_one() {
    // For λ < 1 the weight λ^{t-i} makes recency matter, so order is
    // mathematically significant and a permutation changes the minimizer. This
    // is a semantic requirement, not a defect: each order is then checked
    // against its own literal-objective oracle.
    let d = 2;
    let n = 30;
    let lambda = 0.5_f32;
    let delta = 1.0e3_f32;
    let (xs, ys) = sinusoidal_data(d, n);
    let permutation: Vec<usize> = (0..n).map(|index| (index * 7 + 13) % n).collect();
    let permuted_xs: Vec<Vec<f32>> = permutation.iter().map(|&index| xs[index].clone()).collect();
    let permuted_ys: Vec<f32> = permutation.iter().map(|&index| ys[index]).collect();

    let model = Rls::new(d, lambda, delta).unwrap();
    let state_a = run_sequence(&model, &xs, &ys);
    let state_b = run_sequence(&model, &permuted_xs, &permuted_ys);

    let weight_difference = max_abs_difference(
        &to_f64_vec(state_a.w.as_slice()),
        &to_f64_vec(state_b.w.as_slice()),
    );
    assert!(
        weight_difference > 1e-3,
        "λ < 1 permutation should change the weighted solution, difference {weight_difference}"
    );

    assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
    assert_matches_literal_objective(&model, &permuted_xs, &permuted_ys, 5e-3, 1e-2);
}

#[test]
fn chunking_preserves_the_ordered_trajectory() {
    // Property under test: state continuation across chunk boundaries. The
    // committed state returned by one chunk is the initial state of the next.
    // Three unequal chunk sizes (2, 3, 2) over ordered x1..x7 must reproduce the
    // repeated-process_one reference bitwise (same operations, same order).
    let model = Rls::new(2, 0.97, 5.0).unwrap();
    let observations: Vec<Observation<RlsSample>> = (0..7)
        .map(|k| {
            sample(
                &[(k as f32 * 0.4).sin(), (k as f32 * 0.25).cos()],
                (k as f32 * 0.3).cos(),
            )
        })
        .collect();

    let mut reference = model.initial_state();
    for observation in &observations {
        reference = process_one(&model, &reference, observation).unwrap();
    }

    let s1 = process_batch(&model, model.initial_state(), observations[0..2].to_vec()).unwrap();
    let s2 = process_batch(&model, s1, observations[2..5].to_vec()).unwrap();
    let s3 = process_batch(&model, s2, observations[5..7].to_vec()).unwrap();

    assert_state_bitwise_eq(&reference, &s3);
}

// --- RLS excitation: test-side diagnostics (R1) -------------------------------
//
// All R1 diagnostics are local to this file and computed in `f64`, from the
// committed `f32` state and the literal-objective `A_ref` produced by the D1
// helpers above. No production primitive, monitor, or eigensolver is added.

/// Sum of the diagonal of a square matrix.
fn trace(matrix: &Matrix) -> f64 {
    (0..matrix.rows())
        .map(|index| f64::from(matrix.get(index, index).unwrap()))
        .sum()
}

/// Diagonal entry `matrix[index][index]`.
fn diagonal(matrix: &Matrix, index: usize) -> f64 {
    f64::from(matrix.get(index, index).unwrap())
}

/// `max|P[i,j] − P[j,i]|`; expected to be exactly `0.0` for an RLS `P`.
fn symmetry_residual(matrix: &Matrix) -> f64 {
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
fn run_until_first_failure(
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
fn denominator(model: &Rls, state: &RlsState, x: &[f32]) -> f64 {
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
fn reference_covariance(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Scale-aware `‖P − P_ref‖∞ / (1 + ‖P_ref‖∞)`.
fn reference_covariance_error(p: &Matrix, p_ref: &[Vec<f64>]) -> f64 {
    let max_ref = p_ref
        .iter()
        .flatten()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    let scale = 1.0 + max_ref;
    let mut max_difference = 0.0_f64;
    for (row, reference_row) in p_ref.iter().enumerate() {
        for (col, &reference) in reference_row.iter().enumerate() {
            let actual = f64::from(p.get(row, col).unwrap());
            max_difference = max_difference.max((actual - reference).abs());
        }
    }
    max_difference / scale
}

/// `k = ⌊ln(f32::MAX/δ)/ln(1/λ)⌋ + 1`: the smallest update number whose
/// candidate `δ/λ^k` overflows `f32`. The failing update's 0-based observation
/// index is `k − 1`, so `k − 1` updates commit.
fn overflow_update_number(lambda: f32, delta: f32) -> usize {
    let ratio = (f64::from(f32::MAX) / f64::from(delta)).ln() / (1.0_f64 / f64::from(lambda)).ln();
    ratio.floor() as usize + 1
}

fn default_true_weights(dimension: usize) -> Vec<f32> {
    match dimension {
        1 => vec![0.7],
        2 => vec![0.7, -1.3],
        4 => vec![0.7, -1.3, 0.4, 2.1],
        other => vec![1.0; other],
    }
}

fn linear_targets(xs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
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
fn correlated_data(n: usize, rho: f64) -> (Vec<Vec<f32>>, Vec<f32>) {
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
fn unexcited_axis_data(d: usize, n: usize, axis: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let (mut xs, _) = phase_shifted_data(d, n);
    for x in &mut xs {
        x[axis] = 0.0;
    }
    let mut weights = default_true_weights(d);
    weights[axis] = 0.0;
    let ys = linear_targets(&xs, &weights);
    (xs, ys)
}

/// Strong first axis and weak (`ε`-scaled) second axis.
fn weak_axis_data(n: usize, epsilon: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|t| vec![(t as f32 * 0.37).sin(), epsilon * (t as f32 * 1.1).sin()])
        .collect();
    let ys = linear_targets(&xs, &default_true_weights(2));
    (xs, ys)
}

/// Unit vector rotating with the given period.
fn rotating_data(n: usize, period: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|t| {
            let angle = 2.0 * std::f64::consts::PI * t as f64 / period as f64;
            vec![angle.cos() as f32, angle.sin() as f32]
        })
        .collect();
    let ys = linear_targets(&xs, &default_true_weights(2));
    (xs, ys)
}

// --- RLS excitation: E1 full-rank PE ------------------------------------------

#[test]
fn excitation_e1_full_rank_pe_matches_objective() {
    // H1: every direction is excited, so the f32 state stays finite and
    // symmetric and the literal-objective residual stays small; no guard
    // activates (the helper unwraps every update).
    let delta = 1.0e3_f32;
    for &d in &[2_usize, 4] {
        for &lambda in &[1.0_f32, 0.9, 0.99] {
            let (xs, ys) = phase_shifted_data(d, 200);
            let model = Rls::new(d, lambda, delta).unwrap();
            let state = run_sequence(&model, &xs, &ys);
            for value in state.w.as_slice().iter().chain(state.p.as_slice()) {
                assert!(
                    value.is_finite(),
                    "E1 d={d} lambda={lambda}: non-finite state"
                );
            }
            assert_eq!(
                symmetry_residual(&state.p),
                0.0,
                "E1 symmetry d={d} lambda={lambda}"
            );
            assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
        }
    }
}

// --- RLS excitation: E2 partial excitation ------------------------------------

#[test]
fn excitation_e2_partial_lambda_one_fixed_ridge() {
    // H2: with one axis never excited, lambda = 1 keeps that direction at the
    // fixed ridge delta exactly, and its weight stays exactly zero.
    let d = 4;
    let axis = d - 1;
    for &delta in &[1.0_f32, 1.0e3] {
        let (xs, ys) = unexcited_axis_data(d, 100, axis);
        let model = Rls::new(d, 1.0, delta).unwrap();
        let mut state = model.initial_state();
        for (x, &y) in xs.iter().zip(&ys) {
            state = model.update(&state, &sample(x, y)).unwrap();
            assert_f32_bits_eq(state.p.get(axis, axis).unwrap(), delta);
            assert_f32_bits_eq(state.w.as_slice()[axis], 0.0);
            assert!(denominator(&model, &state, x) > 0.0);
        }
        assert_eq!(symmetry_residual(&state.p), 0.0);
        assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
    }
}

#[test]
fn excitation_e2_partial_lambda_less_than_one_inflates() {
    // H3 (short form): with lambda < 1 the unexcited axis inflates by 1/lambda
    // per step and its weight stays exactly zero.
    let d = 4;
    let axis = d - 1;
    let delta = 1.0e3_f32;
    let lambda = 0.9_f32;
    let (xs, ys) = unexcited_axis_data(d, 100, axis);
    let model = Rls::new(d, lambda, delta).unwrap();
    let mut state = model.initial_state();
    let mut previous = delta;
    for (step, (x, &y)) in xs.iter().zip(&ys).enumerate() {
        state = model.update(&state, &sample(x, y)).unwrap();
        let current = state.p.get(axis, axis).unwrap();
        assert!(
            current > previous,
            "E2 step {step}: P_ee not increasing ({previous} -> {current})"
        );
        let ratio = f64::from(current) / f64::from(previous);
        let expected = 1.0 / f64::from(lambda);
        assert!(
            (ratio - expected).abs() <= 1e-4 * expected,
            "E2 step {step}: ratio {ratio} vs {expected}"
        );
        assert_f32_bits_eq(state.w.as_slice()[axis], 0.0);
        previous = current;
    }
    assert_matches_literal_objective(&model, &xs, &ys, 5e-3, 1e-2);
}

// --- RLS excitation: E3 permanently unexcited + overflow ----------------------

#[test]
fn excitation_e3_unexcited_axis_trajectory_and_overflow() {
    // H3/H7/H8: P_ee(t) = delta / lambda^t until the candidate overflows; the
    // failure is NonFiniteCandidate; the committed count is k-1 within one.
    let lambda = 0.9_f32;
    for &delta in &[1.0_f32, 1.0e3, 1.0e6] {
        let k = overflow_update_number(lambda, delta);
        let n = k + 50;
        let (xs, ys) = unexcited_axis_data(2, n, 1);
        let model = Rls::new(2, lambda, delta).unwrap();

        let prefix = (k * 3 / 4).max(1);
        let mut state = model.initial_state();
        for (step, (x, &y)) in xs.iter().take(prefix).zip(&ys).enumerate() {
            state = model.update(&state, &sample(x, y)).unwrap();
            let t = step + 1;
            let expected = f64::from(delta) / f64::from(lambda).powi(t as i32);
            let actual = diagonal(&state.p, 1);
            assert!(
                (actual - expected).abs() <= 1e-3 * expected,
                "E3 delta={delta} t={t}: P_ee {actual} vs {expected}"
            );
            assert_f32_bits_eq(state.w.as_slice()[1], 0.0);
        }

        let (last, failure) = run_until_first_failure(&model, &xs, &ys);
        let (index, error) = failure.expect("E3: expected an overflow guard trip");
        assert_eq!(error, RlsError::NonFiniteCandidate, "E3 delta={delta}");
        // The failing candidate is rejected from the committed state, so the
        // last valid prefix is not partially committed.
        let rejected = model
            .update(&last, &sample(&xs[index], ys[index]))
            .unwrap_err();
        assert_eq!(rejected, error, "E3 delta={delta}: re-attempt error");
        assert!(
            index.abs_diff(k.saturating_sub(1)) <= 1,
            "E3 delta={delta}: committed {index}, predicted {}",
            k.saturating_sub(1)
        );
        let power_before = f64::from(lambda).powi(index as i32);
        let power_after = f64::from(lambda).powi((index + 1) as i32);
        assert!(f64::from(delta) / power_before <= f64::from(f32::MAX));
        assert!(f64::from(delta) / power_after > f64::from(f32::MAX));
        let expected_last = f64::from(delta) / power_before;
        assert!((diagonal(&last.p, 1) - expected_last).abs() <= 1e-2 * expected_last);
        assert_f32_bits_eq(last.w.as_slice()[1], 0.0);
    }
}

#[test]
#[ignore = "release-only diagnostic: ~8k-update unexcited overflow at lambda=0.99"]
fn excitation_e3_unexcited_overflow_lambda_099_ignored() {
    let lambda = 0.99_f32;
    let delta = 1.0e3_f32;
    let k = overflow_update_number(lambda, delta);
    let n = k + 50;
    let (xs, ys) = unexcited_axis_data(2, n, 1);
    let model = Rls::new(2, lambda, delta).unwrap();
    let (last, failure) = run_until_first_failure(&model, &xs, &ys);
    let (index, error) = failure.expect("E3 lambda=0.99: expected overflow guard");
    assert_eq!(error, RlsError::NonFiniteCandidate);
    assert!(index.abs_diff(k.saturating_sub(1)) <= 1);
    assert_f32_bits_eq(last.w.as_slice()[1], 0.0);
}

// --- RLS excitation: E4 correlated features -----------------------------------

#[test]
fn excitation_e4_correlated_features_bounded_ill_conditioning() {
    // H5: the data Gram is [[1,rho],[rho,1]], so for lambda = 1 the induced
    // infinity norm of P is exactly 1/(1 - rho + 1/delta); P stays bounded and
    // respects the f64 reference. The residual is a correctness check, not a
    // rho-discriminator (D = 2 is well-conditioned).
    let delta = 1.0e3_f32;
    let n = 200;
    let mut previous_norm = -1.0_f64;
    for &rho in &[0.0_f64, 0.5, 0.9, 0.99] {
        let (xs, ys) = correlated_data(n, rho);
        for &lambda in &[1.0_f32, 0.9] {
            let model = Rls::new(2, lambda, delta).unwrap();
            let state = run_sequence(&model, &xs, &ys);
            for value in state.w.as_slice().iter().chain(state.p.as_slice()) {
                assert!(
                    value.is_finite(),
                    "E4 rho={rho} lambda={lambda}: non-finite"
                );
            }
            assert_eq!(
                symmetry_residual(&state.p),
                0.0,
                "E4 rho={rho} lambda={lambda}"
            );
            assert!(trace(&state.p) > 0.0, "E4 rho={rho} lambda={lambda}");

            let (a, _b, _w) = oracle_solution(f64::from(lambda), f64::from(delta), &xs, &ys);
            let residual = covariance_residual(&a, &state.p);
            assert!(
                residual <= 1e-2,
                "E4 rho={rho} lambda={lambda}: residual {residual}"
            );
            let p_ref = reference_covariance(&a);
            let error = reference_covariance_error(&state.p, &p_ref);
            assert!(
                error <= 1e-2,
                "E4 rho={rho} lambda={lambda}: P_ref error {error}"
            );

            if lambda == 1.0 {
                let expected_norm = 1.0 / (1.0 - rho + 1.0 / f64::from(delta));
                let actual_norm = matrix_infinity_norm(&state.p);
                assert!(
                    (actual_norm - expected_norm).abs() <= 1e-3 * expected_norm,
                    "E4 lambda=1 rho={rho}: ‖P‖∞ {actual_norm} vs {expected_norm}"
                );
                assert!(
                    actual_norm > previous_norm,
                    "E4 monotone in rho: {actual_norm} vs previous {previous_norm} at rho={rho}"
                );
                previous_norm = actual_norm;
            }
        }
    }
}

// --- RLS excitation: E5 weak excitation ---------------------------------------

#[test]
fn excitation_e5_weak_axis_finite_horizon_trend() {
    // H4: over the finite horizon, the weak-axis P peak increases monotonically
    // as epsilon decreases, stays finite for every fixed epsilon > 0, and
    // matches the f64 reference. This is finite-horizon evidence only; it does
    // not prove the asymptotic bound.
    let delta = 1.0e3_f32;
    let lambda = 0.9_f32;
    let n = 200;
    let mut previous_max = 0.0_f64;
    for &epsilon in &[1.0_f32, 1.0e-1, 1.0e-2, 1.0e-3] {
        let (xs, ys) = weak_axis_data(n, epsilon);
        let model = Rls::new(2, lambda, delta).unwrap();
        let mut state = model.initial_state();
        let mut max_weak = 0.0_f64;
        for (x, &y) in xs.iter().zip(&ys) {
            state = model.update(&state, &sample(x, y)).unwrap();
            max_weak = max_weak.max(diagonal(&state.p, 1));
        }
        for value in state.w.as_slice().iter().chain(state.p.as_slice()) {
            assert!(value.is_finite(), "E5 epsilon={epsilon}: non-finite");
        }
        assert!(
            max_weak > previous_max,
            "E5 epsilon={epsilon}: max P_ee {max_weak} not > previous {previous_max}"
        );
        previous_max = max_weak;

        let (a, _b, _w) = oracle_solution(f64::from(lambda), f64::from(delta), &xs, &ys);
        let residual = covariance_residual(&a, &state.p);
        assert!(
            residual <= 1e-2,
            "E5 epsilon={epsilon}: residual {residual}"
        );
        let p_ref = reference_covariance(&a);
        let error = reference_covariance_error(&state.p, &p_ref);
        assert!(error <= 1e-2, "E5 epsilon={epsilon}: P_ref error {error}");
    }
}

// --- RLS excitation: E6 changing/rotating excitation --------------------------

#[test]
fn excitation_e6_rotation_window_interaction() {
    // H6: with effective window L = 1/(1 - lambda) = 10, period 4 is PE
    // (P bounded) and a much longer period is effectively rank-1 within the
    // window (P large). The plan's period 200 gave a measured slow/fast ratio of
    // 8.0 (< 10), because the windowed arc 2*pi*L/P scales lambda_min
    // quadratically; the corrected plan permits changing the period, so the slow
    // period is 400 (it divides N, so the lambda = 1 data Gram stays isotropic
    // and the control is exact). The peak is taken over the final quarter so the
    // initial delta-transient does not mask the regime (for lambda = 1 the
    // growing accumulated data would otherwise dominate the earliest steps).
    let delta = 1.0e3_f32;
    let n = 400;
    let burn_in = 3 * n / 4;
    for &lambda in &[1.0_f32, 0.9] {
        let mut max_norms = Vec::new();
        for &period in &[4_usize, 400] {
            let (xs, ys) = rotating_data(n, period);
            let model = Rls::new(2, lambda, delta).unwrap();
            let mut state = model.initial_state();
            let mut max_norm = 0.0_f64;
            for (step, (x, &y)) in xs.iter().zip(&ys).enumerate() {
                state = model.update(&state, &sample(x, y)).unwrap();
                if step >= burn_in {
                    max_norm = max_norm.max(matrix_infinity_norm(&state.p));
                }
            }
            for value in state.p.as_slice() {
                assert!(
                    value.is_finite(),
                    "E6 lambda={lambda} period={period}: non-finite"
                );
            }
            let (a, _b, _w) = oracle_solution(f64::from(lambda), f64::from(delta), &xs, &ys);
            let residual = covariance_residual(&a, &state.p);
            assert!(
                residual <= 1e-2,
                "E6 lambda={lambda} period={period}: residual {residual}"
            );
            max_norms.push(max_norm);
        }
        let ratio = max_norms[1] / max_norms[0];
        if lambda == 1.0 {
            assert!(ratio < 2.0, "E6 lambda=1: slow/fast ratio {ratio}");
        } else {
            assert!(
                ratio > 10.0,
                "E6 lambda=0.9: slow/fast ratio {ratio} (fast {}, slow {})",
                max_norms[0],
                max_norms[1]
            );
        }
    }
}

#[test]
#[ignore = "release-only diagnostic: N=4000 rotation window/period interaction"]
fn excitation_e6_rotation_long_window_ignored() {
    // Same contrast as E6 over a much longer horizon, so the periodic
    // sawtooth shape is fully resolved. Criterion unchanged (slow/fast > 10 at
    // lambda = 0.9; < 2 at lambda = 1).
    let delta = 1.0e3_f32;
    let n = 4000;
    let burn_in = 3 * n / 4;
    for &lambda in &[1.0_f32, 0.9] {
        let mut max_norms = Vec::new();
        for &period in &[4_usize, 400] {
            let (xs, ys) = rotating_data(n, period);
            let model = Rls::new(2, lambda, delta).unwrap();
            let mut state = model.initial_state();
            let mut max_norm = 0.0_f64;
            for (step, (x, &y)) in xs.iter().zip(&ys).enumerate() {
                state = model.update(&state, &sample(x, y)).unwrap();
                if step >= burn_in {
                    max_norm = max_norm.max(matrix_infinity_norm(&state.p));
                }
            }
            for value in state.p.as_slice() {
                assert!(value.is_finite(), "E6-long lambda={lambda} period={period}");
            }
            max_norms.push(max_norm);
        }
        let ratio = max_norms[1] / max_norms[0];
        if lambda == 1.0 {
            assert!(ratio < 2.0, "E6-long lambda=1: slow/fast ratio {ratio}");
        } else {
            assert!(ratio > 10.0, "E6-long lambda=0.9: slow/fast ratio {ratio}");
        }
    }
}

// --- RLS excitation: D=1 closed forms (R1) ------------------------------------

#[test]
fn excitation_c1_zero_input_closed_form_and_overflow() {
    // C1: with x = 0, P_t = delta / lambda^t and w stays exactly zero. For
    // lambda = 1, P stays exactly delta. The trajectory ends in the same
    // NonFiniteCandidate overflow as E3.
    let delta = 1.0e3_f32;

    let model = Rls::new(1, 1.0, delta).unwrap();
    let mut state = model.initial_state();
    for _ in 0..100 {
        state = model.update(&state, &sample(&[0.0], 5.0)).unwrap();
        assert_f32_bits_eq(state.p.get(0, 0).unwrap(), delta);
        assert_f32_bits_eq(state.w.as_slice()[0], 0.0);
    }

    let lambda = 0.9_f32;
    let model = Rls::new(1, lambda, delta).unwrap();
    let mut state = model.initial_state();
    for t in 1..=100 {
        state = model.update(&state, &sample(&[0.0], 5.0)).unwrap();
        let expected = f64::from(delta) / f64::from(lambda).powi(t);
        let actual = diagonal(&state.p, 0);
        assert!(
            (actual - expected).abs() <= 1e-3 * expected,
            "C1 t={t}: P {actual} vs {expected}"
        );
        assert_f32_bits_eq(state.w.as_slice()[0], 0.0);
    }

    let k = overflow_update_number(lambda, delta);
    let n = k + 50;
    let xs = vec![vec![0.0_f32]; n];
    let ys = vec![5.0_f32; n];
    let (last, failure) = run_until_first_failure(&model, &xs, &ys);
    let (index, error) = failure.expect("C1: expected overflow guard");
    assert_eq!(error, RlsError::NonFiniteCandidate);
    let rejected = model
        .update(&last, &sample(&xs[index], ys[index]))
        .unwrap_err();
    assert_eq!(rejected, error, "C1: re-attempt error");
    assert!(
        index.abs_diff(k.saturating_sub(1)) <= 1,
        "C1 index {index} vs {}",
        k.saturating_sub(1)
    );
    assert_f32_bits_eq(last.w.as_slice()[0], 0.0);
    let expected_last = f64::from(delta) / f64::from(lambda).powi(index as i32);
    assert!((diagonal(&last.p, 0) - expected_last).abs() <= 1e-2 * expected_last);
}

#[test]
fn excitation_c2_constant_input_fixed_point() {
    // C2: with constant nonzero x, P -> (1 - lambda)/x^2 for lambda < 1 and
    // P -> 0 for lambda = 1; w -> y/x in both cases.
    let x = 2.0_f32;
    let y = 6.0_f32;
    let delta = 1.0e3_f32;

    let lambda = 0.9_f32;
    let model = Rls::new(1, lambda, delta).unwrap();
    let mut state = model.initial_state();
    for _ in 0..500 {
        state = model.update(&state, &sample(&[x], y)).unwrap();
    }
    let expected_p = f64::from(1.0 - lambda) / f64::from(x * x);
    let actual_p = diagonal(&state.p, 0);
    assert!(
        (actual_p - expected_p).abs() <= 1e-3 * expected_p,
        "C2 P {actual_p} vs {expected_p}"
    );
    let expected_w = f64::from(y) / f64::from(x);
    let actual_w = f64::from(state.w.as_slice()[0]);
    assert!(
        (actual_w - expected_w).abs() <= 1e-3 * expected_w,
        "C2 w {actual_w} vs {expected_w}"
    );

    let xs: Vec<f64> = vec![f64::from(x); 500];
    let ys: Vec<f64> = vec![f64::from(y); 500];
    let (a, b) = d1_closed_form(f64::from(lambda), f64::from(delta), &xs, &ys);
    assert!((actual_p * a - 1.0).abs() <= 1e-2, "C2 |A P - 1|");
    assert!(
        (actual_w - b / a).abs() <= 1e-3 * (1.0 + (b / a).abs()),
        "C2 w vs closed form"
    );

    let model = Rls::new(1, 1.0, delta).unwrap();
    let mut state = model.initial_state();
    for _ in 0..500 {
        state = model.update(&state, &sample(&[x], y)).unwrap();
    }
    let count = 500.0_f64;
    let expected_p = 1.0 / (count * f64::from(x * x) + 1.0 / f64::from(delta));
    assert!(
        (diagonal(&state.p, 0) - expected_p).abs() <= 1e-3 * expected_p,
        "C2 lambda=1 P"
    );
    assert!(
        (f64::from(state.w.as_slice()[0]) - expected_w).abs() <= 1e-3 * (1.0 + expected_w),
        "C2 lambda=1 w"
    );
}

#[test]
fn excitation_c3_varying_input_closed_form() {
    // C3: a short varying-input D = 1 sequence matches the scalar closed form.
    let delta = 1.0e3_f32;
    let lambda = 0.9_f32;
    let n = 40;
    let xs: Vec<Vec<f32>> = (0..n).map(|t| vec![(t as f32 * 0.4).sin() + 1.5]).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 0.5 * x[0] + 0.25).collect();

    let model = Rls::new(1, lambda, delta).unwrap();
    let state = run_sequence(&model, &xs, &ys);
    let x_values: Vec<f64> = xs.iter().map(|x| f64::from(x[0])).collect();
    let y_values: Vec<f64> = ys.iter().map(|&y| f64::from(y)).collect();
    let (a, b) = d1_closed_form(f64::from(lambda), f64::from(delta), &x_values, &y_values);
    let w_oracle = b / a;
    let w_recurrence = f64::from(state.w.as_slice()[0]);
    assert!(
        (w_recurrence - w_oracle).abs() <= 5e-3 * (1.0 + w_oracle.abs()),
        "C3 w {w_recurrence} vs {w_oracle}"
    );
    let p_recurrence = diagonal(&state.p, 0);
    assert!(
        (p_recurrence * a - 1.0).abs() <= 1e-2,
        "C3 |A P - 1| = {}",
        (p_recurrence * a - 1.0).abs()
    );
}

// --- RLS stage 2: regime shift, noise, prequential, recovery (R2) -------------
//
// Test-local `f64` diagnostics only; reuses the D1/R1 helpers. Serves the R2
// falsification rules: lambda = 1 must pool (not track), lambda < 1 must track
// under persistent excitation, noise must preserve the R1 numerical
// conclusions, and a retained/inflated P must recover once excitation resumes.

/// Persistently exciting design whose target switches from `w_a` to `w_b` at
/// `split`.
fn regime_data(
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

/// Adds bounded uniform noise in `[-sigma, sigma)` (variance `sigma^2 / 3`).
fn noisy_targets(ys: &[f32], sigma: f32, seed: u64) -> Vec<f32> {
    let mut lcg = Lcg::new(seed);
    ys.iter()
        .map(|&y| y + sigma * 2.0 * lcg.next_f32())
        .collect()
}

/// One-step-ahead (prequential) errors: predict from the committed state, then
/// update. Causal and look-ahead-free by construction.
fn prequential_errors(model: &Rls, xs: &[Vec<f32>], ys: &[f32]) -> (Vec<f64>, RlsState) {
    let mut state = model.initial_state();
    let mut errors = Vec::with_capacity(xs.len());
    for (x, &y) in xs.iter().zip(ys) {
        let prediction = model.predict(&state, &Vector::from_slice(x)).unwrap();
        errors.push(f64::from(y) - f64::from(prediction));
        state = model.update(&state, &sample(x, y)).unwrap();
    }
    (errors, state)
}

/// First index whose value is `<= tolerance`.
fn convergence_step(values: &[f64], tolerance: f64) -> Option<usize> {
    values.iter().position(|&value| value <= tolerance)
}

/// `‖w − target‖∞`.
fn weight_distance(state: &RlsState, target: &[f32]) -> f64 {
    state
        .w
        .as_slice()
        .iter()
        .zip(target)
        .fold(0.0_f64, |max, (&actual, &expected)| {
            max.max((f64::from(actual) - f64::from(expected)).abs())
        })
}

/// `‖w − target‖∞` against an `f64` reference such as the pooled oracle.
fn weight_distance_f64(state: &RlsState, target: &[f64]) -> f64 {
    state
        .w
        .as_slice()
        .iter()
        .zip(target)
        .fold(0.0_f64, |max, (&actual, &expected)| {
            max.max((f64::from(actual) - expected).abs())
        })
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    (mean, variance.sqrt())
}

#[test]
fn regime_e7_abrupt_shift_pools_for_lambda_one_and_tracks_for_lambda_below_one() {
    // split = 400, N = 1400, not the plan's split = 200: lambda = 0.99 needs
    // ~8/(1-lambda) = 800 post-shift steps, and with split = 200 the pooled
    // lambda = 1 blend would sit within the 0.2 threshold of w_B, making the
    // negative control degenerate. split = 400 keeps the control meaningful.
    let delta = 1.0e3_f32;
    let split = 400_usize;
    let n = 1400_usize;
    for &d in &[2_usize, 4] {
        let (w_a, w_b) = if d == 2 {
            (vec![0.7_f32, -1.3], vec![-0.5_f32, 1.1])
        } else {
            (vec![0.7, -1.3, 0.4, 2.1], vec![-0.5, 1.1, -1.4, 0.6])
        };
        let shift = w_a.iter().zip(&w_b).fold(0.0_f64, |max, (&a, &b)| {
            max.max((f64::from(a) - f64::from(b)).abs())
        });
        let (xs, ys) = regime_data(d, n, split, &w_a, &w_b);

        // lambda = 1 negative control: the pooled minimizer, no tracking.
        let control = Rls::new(d, 1.0, delta).unwrap();
        let (_, control_state) = prequential_errors(&control, &xs, &ys);
        let (_, _, pooled) = oracle_solution(1.0, f64::from(delta), &xs, &ys);
        let pooled_scale = 1.0 + pooled.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            weight_distance_f64(&control_state, &pooled) <= 5e-3 * pooled_scale,
            "E7 d={d}: lambda=1 does not match the pooled objective"
        );
        assert!(
            weight_distance(&control_state, &w_b) > 0.2 * shift,
            "E7 d={d}: lambda=1 unexpectedly tracks regime B"
        );

        let mut steps = Vec::new();
        for &lambda in &[0.9_f32, 0.95, 0.99] {
            let model = Rls::new(d, lambda, delta).unwrap();
            let mut state = model.initial_state();
            let mut pre_shift = 0.0_f64;
            let mut distances = Vec::new();
            for (index, (x, &y)) in xs.iter().zip(&ys).enumerate() {
                state = model.update(&state, &sample(x, y)).unwrap();
                if index + 1 == split {
                    pre_shift = weight_distance(&state, &w_a);
                }
                if index >= split {
                    distances.push(weight_distance(&state, &w_b));
                }
            }
            assert!(
                pre_shift <= 0.05 * shift,
                "E7 d={d} lambda={lambda}: pre-shift error {pre_shift}"
            );
            let budget = split + (8.0 / (1.0 - f64::from(lambda))).ceil() as usize;
            let step = convergence_step(&distances, 0.05 * shift).map(|offset| split + offset);
            assert!(
                step.is_some_and(|value| value <= budget),
                "E7 d={d} lambda={lambda}: convergence {step:?} exceeds budget {budget}"
            );
            steps.push(step.unwrap());
        }
        assert!(
            steps[0] < steps[1] && steps[1] < steps[2],
            "E7 d={d}: convergence steps {steps:?} do not follow memory ordering"
        );
    }
}

#[test]
fn regime_e8_noise_preserves_bias_and_variance_memory_tradeoff() {
    let delta = 1.0e3_f32;
    let d = 2_usize;
    let split = 400_usize;
    let n = 1400_usize;
    let w_a = vec![0.7_f32, -1.3];
    let w_b = vec![-0.5_f32, 1.1];
    let shift = 2.4_f64;
    let (xs, clean) = regime_data(d, n, split, &w_a, &w_b);
    let seeds: Vec<u64> = (0..24)
        .map(|index| 0x5eed_0000_u64 + index as u64)
        .collect();

    // lambda = 1 mean stays at the pooled blend, not w_B.
    let control = Rls::new(d, 1.0, delta).unwrap();
    let control_means: Vec<f64> = (0..2)
        .map(|component| {
            let values: Vec<f64> = seeds
                .iter()
                .map(|&seed| {
                    let ys = noisy_targets(&clean, 0.2, seed);
                    let (_, state) = prequential_errors(&control, &xs, &ys);
                    f64::from(state.w.as_slice()[component])
                })
                .collect();
            mean_std(&values).0
        })
        .collect();
    let (_, _, pooled) = oracle_solution(1.0, f64::from(delta), &xs, &clean);
    for (component, &pooled_value) in pooled.iter().enumerate() {
        assert!(
            (control_means[component] - pooled_value).abs() <= 0.05,
            "E8 lambda=1 component {component}: mean {} vs pooled {pooled_value}",
            control_means[component]
        );
    }

    for &sigma in &[0.0_f32, 0.05, 0.2] {
        let mut stds = Vec::new();
        for &lambda in &[0.9_f32, 0.99] {
            let model = Rls::new(d, lambda, delta).unwrap();
            let mut values: Vec<Vec<f64>> = vec![Vec::new(), Vec::new()];
            for &seed in &seeds {
                let ys = noisy_targets(&clean, sigma, seed);
                let (_, state) = prequential_errors(&model, &xs, &ys);
                values[0].push(f64::from(state.w.as_slice()[0]));
                values[1].push(f64::from(state.w.as_slice()[1]));
            }
            let mut component_stds = Vec::new();
            for (component, &target) in w_b.iter().enumerate() {
                let (mean, std) = mean_std(&values[component]);
                assert!(
                    (mean - f64::from(target)).abs() <= 0.10 * shift,
                    "E8 sigma={sigma} lambda={lambda} component {component}: bias {mean} vs {}",
                    f64::from(target)
                );
                component_stds.push(std);
            }
            stds.push(component_stds);
        }
        if sigma > 0.0 {
            let long = ((stds[1][0].powi(2) + stds[1][1].powi(2)) / 2.0).sqrt();
            let short = ((stds[0][0].powi(2) + stds[0][1].powi(2)) / 2.0).sqrt();
            assert!(
                long < short,
                "E8 sigma={sigma}: std at lambda=0.99 ({long}) not below lambda=0.9 ({short})"
            );
        }
    }

    // Noise does not change the R1 numerical conclusions: no guard, finite P.
    let model = Rls::new(d, 0.9, delta).unwrap();
    let ys = noisy_targets(&clean, 0.2, 7);
    let (_, state) = prequential_errors(&model, &xs, &ys);
    assert!(state.p.as_slice().iter().all(|value| value.is_finite()));
}

#[test]
fn regime_e9_prequential_error_decays_for_lambda_below_one_only() {
    let delta = 1.0e3_f32;
    let d = 2_usize;
    let split = 400_usize;
    let n = 1400_usize;
    let w_a = vec![0.7_f32, -1.3];
    let w_b = vec![-0.5_f32, 1.1];
    let shift = 2.4_f64;
    let (xs, ys) = regime_data(d, n, split, &w_a, &w_b);

    let mean_abs = |errors: &[f64], range: std::ops::Range<usize>| -> f64 {
        let slice = &errors[range];
        slice.iter().map(|value| value.abs()).sum::<f64>() / slice.len() as f64
    };

    // lambda = 1 follows the *time-varying pooled solution* exactly: the pooled
    // minimizer itself moves toward w_B as regime-B data accumulates, so its
    // prequential error shrinks slowly but never reaches the lambda < 1 level.
    let control = Rls::new(d, 1.0, delta).unwrap();
    let mut control_state = control.initial_state();
    let mut control_errors = Vec::with_capacity(n);
    for (index, (x, &y)) in xs.iter().zip(&ys).enumerate() {
        let prediction = control
            .predict(&control_state, &Vector::from_slice(x))
            .unwrap();
        control_errors.push(f64::from(y) - f64::from(prediction));
        control_state = control.update(&control_state, &sample(x, y)).unwrap();
        if index + 1 == n || (index + 1) % 400 == 0 {
            let (_, _, pooled) =
                oracle_solution(1.0, f64::from(delta), &xs[..index + 1], &ys[..index + 1]);
            let scale = 1.0 + pooled.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                weight_distance_f64(&control_state, &pooled) <= 5e-3 * scale,
                "E9 lambda=1 at t={}: not the pooled solution",
                index + 1
            );
        }
    }
    let control_late = mean_abs(&control_errors, n - 100..n);
    assert!(
        control_late >= 0.1 * shift,
        "E9 lambda=1: post-shift prequential error collapsed to {control_late}"
    );

    let mut late_errors = Vec::new();
    for &lambda in &[0.9_f32, 0.95, 0.99] {
        let model = Rls::new(d, lambda, delta).unwrap();
        let (errors, _) = prequential_errors(&model, &xs, &ys);
        let early = mean_abs(&errors, split..split + 20);
        let late = mean_abs(&errors, n - 100..n);
        assert!(
            late <= 0.1 * early,
            "E9 lambda={lambda}: prequential error did not decay ({early} -> {late})"
        );
        late_errors.push(late);
    }
    for &late in &late_errors {
        assert!(
            control_late > 5.0 * late,
            "E9 lambda=1 error {control_late} not clearly above lambda<1 error {late}"
        );
    }
}

#[test]
fn regime_e10_excitation_loss_freezes_weight_and_recovers_after_resume() {
    // Each observation excites exactly one axis, so P stays exactly diagonal:
    // an unexcited axis is frozen, not merely slow.
    let delta = 1.0e3_f32;
    let lambda = 0.9_f32;
    let w_true = [0.7_f32, -1.3];
    let normal = 200_usize;
    let loss = 60_usize;
    let resume = 120_usize;
    let n = normal + loss + resume;

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for t in 0..n {
        let in_loss = (normal..normal + loss).contains(&t);
        let x = if in_loss || t % 2 == 0 {
            vec![(t as f32 * 0.37).sin(), 0.0]
        } else {
            vec![0.0, (t as f32 * 1.1).sin()]
        };
        let y = x[0] * w_true[0] + x[1] * w_true[1];
        xs.push(x);
        ys.push(y);
    }

    let model = Rls::new(2, lambda, delta).unwrap();
    let mut state = model.initial_state();
    for (x, &y) in xs.iter().zip(&ys).take(normal) {
        state = model.update(&state, &sample(x, y)).unwrap();
        assert_eq!(symmetry_residual(&state.p), 0.0);
        assert_f32_bits_eq(state.p.get(0, 1).unwrap(), 0.0);
    }
    let learned_w1 = f64::from(state.w.as_slice()[1]);
    assert!(
        (learned_w1 - f64::from(w_true[1])).abs() <= 5e-3 * (1.0 + f64::from(w_true[1]).abs()),
        "E10 phase 1 did not learn w_1: {learned_w1}"
    );

    // Phase 2: axis 1 unexcited; frozen weight, P_11 inflates by 1/lambda.
    let frozen_w1 = state.w.as_slice()[1];
    let p11_before = diagonal(&state.p, 1);
    let mut previous = p11_before;
    for (x, &y) in xs.iter().zip(&ys).skip(normal).take(loss) {
        state = model.update(&state, &sample(x, y)).unwrap();
        assert_f32_bits_eq(state.w.as_slice()[1], frozen_w1);
        let current = diagonal(&state.p, 1);
        let ratio = current / previous;
        let expected = 1.0 / f64::from(lambda);
        assert!(
            (ratio - expected).abs() <= 1e-4 * expected,
            "E10 loss step: P_11 ratio {ratio} vs {expected}"
        );
        previous = current;
    }
    let p11_peak = diagonal(&state.p, 1);
    assert!(
        p11_peak > 100.0 * p11_before,
        "E10: expected noticeable inflation"
    );

    // Phase 3: excitation returns; P_11 deflates and w_1 stays correct.
    for (x, &y) in xs.iter().zip(&ys).skip(normal + loss) {
        state = model.update(&state, &sample(x, y)).unwrap();
    }
    assert!(
        diagonal(&state.p, 1) < 0.1 * p11_peak,
        "E10: P_11 did not recover after excitation resumed"
    );
    assert!(
        weight_distance(&state, &w_true) <= 5e-3 * (1.0 + 1.3),
        "E10: w drifted during/after loss"
    );

    // Recovery of a deliberately wrong w_1: inflated P corrects it in one
    // informative step (strictly faster than the 200-step cold start).
    let mut wrong = model.initial_state();
    for (x, &y) in xs.iter().zip(&ys).take(normal) {
        wrong = model.update(&wrong, &sample(x, y)).unwrap();
    }
    let mut perturbed_state = wrong;
    perturbed_state.w = Vector::from_slice(&[perturbed_state.w.as_slice()[0], 0.7]);
    for (x, &y) in xs.iter().zip(&ys).skip(normal).take(loss) {
        perturbed_state = model.update(&perturbed_state, &sample(x, y)).unwrap();
    }
    assert_f32_bits_eq(perturbed_state.w.as_slice()[1], 0.7);
    let mut informative = 0_usize;
    for (x, &y) in xs.iter().zip(&ys).skip(normal + loss) {
        perturbed_state = model.update(&perturbed_state, &sample(x, y)).unwrap();
        if x[1].abs() > 0.0 {
            informative += 1;
            if (f64::from(perturbed_state.w.as_slice()[1]) - f64::from(w_true[1])).abs()
                <= 1e-3 * (1.0 + f64::from(w_true[1]).abs())
            {
                break;
            }
        }
    }
    assert!(
        informative <= 4,
        "E10: recovery took {informative} informative steps"
    );
}

#[test]
fn regime_e11_gradual_drift_lag_grows_with_memory() {
    let delta = 1.0e3_f32;
    let d = 2_usize;
    let n = 1400_usize;
    let w_start = [0.7_f32, -1.3];
    let w_end = [-0.5_f32, 1.1];
    let (xs, _) = phase_shifted_data(d, n);
    let target = |t: usize| -> [f64; 2] {
        let fraction = t as f64 / (n - 1) as f64;
        [
            f64::from(w_start[0]) + (f64::from(w_end[0]) - f64::from(w_start[0])) * fraction,
            f64::from(w_start[1]) + (f64::from(w_end[1]) - f64::from(w_start[1])) * fraction,
        ]
    };
    let ys: Vec<f32> = xs
        .iter()
        .enumerate()
        .map(|(t, x)| {
            let w = target(t);
            (f64::from(x[0]) * w[0] + f64::from(x[1]) * w[1]) as f32
        })
        .collect();

    let mut lags = Vec::new();
    for &lambda in &[0.9_f32, 0.99] {
        let model = Rls::new(d, lambda, delta).unwrap();
        let mut state = model.initial_state();
        let mut late_error = 0.0_f64;
        let mut count = 0_usize;
        for (t, (x, &y)) in xs.iter().zip(&ys).enumerate() {
            state = model.update(&state, &sample(x, y)).unwrap();
            if t >= n - 200 {
                let w = target(t);
                let error = (f64::from(state.w.as_slice()[0]) - w[0])
                    .abs()
                    .max((f64::from(state.w.as_slice()[1]) - w[1]).abs());
                late_error += error;
                count += 1;
            }
        }
        lags.push(late_error / count as f64);
    }
    assert!(
        lags[1] > lags[0],
        "E11: larger memory should lag more: {lags:?}"
    );
}

// --- RLS stage 3: feature scaling and conditioning (R3) -----------------------
//
// Test-local f64 diagnostics only. The literal weighted objective and direct
// solve remain the correctness oracle; the production recurrence is never its
// own oracle. Uniform scaling leaves the condition number unchanged, so an
// exact-similarity failure (with the matched prior) is never excused by
// conditioning. Unequal scaling raises kappa, so a failure beyond the
// pre-registered pass region is recorded as a numerical boundary rather than
// excused; only pass-region and exact-similarity failures fail the test.

struct ScalingRun {
    kappa: f64,
    prediction_error: Option<f64>,
    residual: Option<f64>,
    state: RlsState,
    guard: Option<(usize, RlsError)>,
}

fn scale_features(xs: &[Vec<f32>], factors: &[f32]) -> Vec<Vec<f32>> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(factors)
                .map(|(value, factor)| value * factor)
                .collect()
        })
        .collect()
}

fn oracle_predictions(weights: &[f64], xs: &[Vec<f32>]) -> Vec<f64> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(weights)
                .map(|(&value, &weight)| f64::from(value) * weight)
                .sum()
        })
        .collect()
}

fn prediction_error(state: &RlsState, xs: &[Vec<f32>], weights: &[f64]) -> f64 {
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

/// Prediction-relevant coefficient error: the per-component contribution error
/// normalized by the largest reference contribution. Avoids the meaningless
/// blow-up of relative error on near-zero coefficients.
fn contribution_weight_error(state: &RlsState, expected: &[f64], factors: &[f32]) -> f64 {
    let scale = expected
        .iter()
        .zip(factors)
        .map(|(&weight, &factor)| (weight * f64::from(factor)).abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    state.w.as_slice().iter().zip(expected).zip(factors).fold(
        0.0_f64,
        |max, ((&actual, &target), &factor)| {
            let contribution = (f64::from(actual) - target) * f64::from(factor);
            max.max(contribution.abs() / scale)
        },
    )
}

fn trace_f64(a: &[Vec<f64>]) -> f64 {
    (0..a.len()).map(|index| a[index][index]).sum()
}

/// Exact condition number of a symmetric `2×2` matrix via closed-form
/// eigenvalues (not a generic eigensolver).
fn kappa_2x2(a: &[Vec<f64>]) -> f64 {
    let trace = a[0][0] + a[1][1];
    let determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let half = trace / 2.0;
    let root = (half * half - determinant).max(0.0).sqrt();
    (half + root) / (half - root)
}

/// Design conditioning: exact `kappa_2(A)` for `D = 2`; the rigorous upper bound
/// `kappa_2(A) <= tr(A)·tr(A^{-1})` (documented proxy) for `D > 2`.
fn design_kappa(a: &[Vec<f64>]) -> f64 {
    if a.len() == 2 {
        kappa_2x2(a)
    } else {
        let p_ref = reference_covariance(a);
        trace_f64(a) * trace_f64(&p_ref)
    }
}

fn max_abs_matrix(matrix: &Matrix) -> f64 {
    matrix
        .as_slice()
        .iter()
        .fold(0.0_f64, |max, &value| max.max(f64::from(value).abs()))
}

fn scaling_run(model: &Rls, xs: &[Vec<f32>], ys: &[f32]) -> ScalingRun {
    let (state, guard) = run_until_first_failure(model, xs, ys);
    let (a, _b, w_ref) =
        oracle_solution(f64::from(model.lambda()), f64::from(model.delta()), xs, ys);
    let kappa = design_kappa(&a);
    let (prediction_error, residual) = if guard.is_none() {
        (
            Some(prediction_error(&state, xs, &w_ref)),
            Some(covariance_residual(&a, &state.p)),
        )
    } else {
        (None, None)
    };
    ScalingRun {
        kappa,
        prediction_error,
        residual,
        state,
        guard,
    }
}

/// Invariants hold at every scale. A failure inside the pre-registered pass
/// region is unexplained and fails the test; beyond it, the failure is recorded
/// as a numerical boundary (conditioning or cancellation / dynamic range) and
/// reported, so it cannot mask a pass-region or exact-similarity failure.
fn classify_scaling(label: &str, run: &ScalingRun, within_pass_region: bool) {
    assert_eq!(
        symmetry_residual(&run.state.p),
        0.0,
        "{label}: symmetry residual"
    );
    assert!(
        run.state
            .w
            .as_slice()
            .iter()
            .chain(run.state.p.as_slice())
            .all(|value| value.is_finite()),
        "{label}: non-finite committed state"
    );
    let pass = run.guard.is_none()
        && run.prediction_error.is_some_and(|value| value <= 5e-3)
        && run.residual.is_some_and(|value| value <= 1e-2);
    if pass {
        return;
    }
    if within_pass_region {
        panic!(
            "{label}: failure inside the pre-registered pass region (guard={:?}, kappa={})",
            run.guard, run.kappa
        );
    }
    println!(
        "R3 boundary: {label} guard={:?} pred={:?} res={:?} kappa={}",
        run.guard, run.prediction_error, run.residual, run.kappa
    );
}

#[test]
fn scaling_e12_uniform_scaling_exact_similarity() {
    // Exact similarity: x̃ = s·x with δ_s = δ/s² gives A(x̃) = s²A(x), so
    // w(x̃;δ_s) = w(x;δ)/s and predictions are invariant; uniform scaling also
    // leaves the condition number unchanged. It must therefore hold at every s,
    // and is never excused by conditioning. Fixed-δ runs are a different
    // (prior-shifted) objective and only need to match their own f64 oracle,
    // with conditioning as the accepted explanation beyond the pass region.
    let delta = 1.0e3_f32;
    let n = 200_usize;
    for &d in &[2_usize, 4] {
        let (xs0, ys0) = phase_shifted_data(d, n);
        for &lambda in &[1.0_f32, 0.9] {
            let (_, _, w0) = oracle_solution(f64::from(lambda), f64::from(delta), &xs0, &ys0);
            for &s in &[1.0_f32, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0] {
                let factors = vec![s; d];
                let xs_s = scale_features(&xs0, &factors);
                let delta_s = delta / (s * s);

                let model_s = Rls::new(d, lambda, delta_s).unwrap();
                let run_s = scaling_run(&model_s, &xs_s, &ys0);
                assert!(
                    run_s.guard.is_none(),
                    "E12 exact-similarity guard d={d} lambda={lambda} s={s}: {:?}",
                    run_s.guard
                );
                let w_expected: Vec<f64> = w0.iter().map(|value| value / f64::from(s)).collect();
                let invariance = prediction_error(&run_s.state, &xs_s, &w_expected);
                let contribution = contribution_weight_error(&run_s.state, &w_expected, &factors);
                assert!(
                    invariance <= 5e-3 && contribution <= 1e-2,
                    "E12 exact-similarity d={d} lambda={lambda} s={s}: invariance={invariance} contribution={contribution}"
                );

                let model_fixed = Rls::new(d, lambda, delta).unwrap();
                let run_fixed = scaling_run(&model_fixed, &xs_s, &ys0);
                classify_scaling(
                    &format!("E12 fixed-delta d={d} lambda={lambda} s={s}"),
                    &run_fixed,
                    s <= 10.0,
                );
            }
        }
    }
}

#[test]
fn scaling_e13_unequal_feature_scale_conditioning() {
    // Scale the last axis relative to the others. Pass region (pre-registered):
    // disparity max(ratio, 1/ratio) <= 10. Beyond it, a failure or guard is
    // accepted only when the design conditioning accounts for it.
    let delta = 1.0e3_f32;
    let n = 200_usize;
    let ratios = [
        0.001_f32, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0,
    ];
    for &d in &[2_usize, 4] {
        let (xs0, ys0) = phase_shifted_data(d, n);
        for &lambda in &[1.0_f32, 0.9] {
            for &ratio in &ratios {
                let mut factors = vec![1.0_f32; d];
                factors[d - 1] = ratio;
                let xs = scale_features(&xs0, &factors);
                let model = Rls::new(d, lambda, delta).unwrap();
                let run = scaling_run(&model, &xs, &ys0);
                let disparity = ratio.max(1.0 / ratio);
                classify_scaling(
                    &format!("E13 d={d} lambda={lambda} ratio={ratio}"),
                    &run,
                    disparity <= 10.0,
                );
            }
        }
    }
}

#[test]
fn scaling_e14_scaled_and_correlated_features() {
    // Correlation and scale disparity compound. Pass region: s <= 10 (all rho).
    let delta = 1.0e3_f32;
    let n = 200_usize;
    for &rho in &[0.0_f64, 0.9, 0.99] {
        let (xs0, ys0) = correlated_data(n, rho);
        for &lambda in &[1.0_f32, 0.9] {
            for &s in &[1.0_f32, 10.0, 100.0, 1_000.0] {
                let xs = scale_features(&xs0, &[1.0, s]);
                let model = Rls::new(2, lambda, delta).unwrap();
                let run = scaling_run(&model, &xs, &ys0);
                classify_scaling(
                    &format!("E14 rho={rho} lambda={lambda} s={s}"),
                    &run,
                    s <= 10.0,
                );
            }
        }
    }
}

#[test]
fn scaling_e15_f32_f64_agreement_boundary() {
    // D=2 unequal scaling, lambda=1: record the first ratio where the
    // pre-registered criterion fails or a guard trips. Pass region: ratio <= 10.
    let delta = 1.0e3_f32;
    let n = 200_usize;
    let (xs0, ys0) = phase_shifted_data(2, n);
    let ratios = [1.0_f32, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0];
    let mut boundary: Option<f32> = None;
    for &ratio in &ratios {
        let xs = scale_features(&xs0, &[1.0, ratio]);
        let model = Rls::new(2, 1.0, delta).unwrap();
        let run = scaling_run(&model, &xs, &ys0);
        let pass = run.guard.is_none()
            && run.prediction_error.is_some_and(|value| value <= 5e-3)
            && run.residual.is_some_and(|value| value <= 1e-2);
        if !pass && boundary.is_none() {
            boundary = Some(ratio);
            println!(
                "E15 first boundary ratio={ratio} guard={:?} pred={:?} res={:?} kappa={}",
                run.guard, run.prediction_error, run.residual, run.kappa
            );
        }
        classify_scaling(&format!("E15 ratio={ratio}"), &run, ratio <= 10.0);
    }
    println!("E15 boundary={boundary:?} over ratios {ratios:?}");
}

#[test]
fn scaling_e16_sequential_trajectory_under_exact_scaling() {
    // Per-step (not final-only) prediction trajectory: the exact-similarity
    // scaled run must match the unscaled run at every step, preserving the
    // ordered, causal lambda < 1 trajectory.
    let delta = 1.0e3_f32;
    let d = 2_usize;
    let n = 200_usize;
    let lambda = 0.9_f32;
    let (xs0, ys0) = phase_shifted_data(d, n);
    for &s in &[1.0_f32, 100.0, 1_000.0] {
        let xs_s = scale_features(&xs0, &[s, s]);
        let model0 = Rls::new(d, lambda, delta).unwrap();
        let model_s = Rls::new(d, lambda, delta / (s * s)).unwrap();
        let mut state0 = model0.initial_state();
        let mut state_s = model_s.initial_state();
        let mut max_step_error = 0.0_f64;
        for (index, (x0, &y)) in xs0.iter().zip(&ys0).enumerate() {
            let baseline = model0.predict(&state0, &Vector::from_slice(x0)).unwrap();
            let scaled = model_s
                .predict(&state_s, &Vector::from_slice(&xs_s[index]))
                .unwrap();
            max_step_error = max_step_error.max(
                (f64::from(scaled) - f64::from(baseline)).abs() / (1.0 + f64::from(baseline).abs()),
            );
            state0 = model0.update(&state0, &sample(x0, y)).unwrap();
            state_s = model_s.update(&state_s, &sample(&xs_s[index], y)).unwrap();
        }
        assert!(
            max_step_error <= 5e-3,
            "E16 step-trajectory mismatch at s={s}: {max_step_error}"
        );
    }
}

#[test]
#[ignore = "release-only diagnostic: D=8, s=1e6, N=1e4 scaling accumulation"]
fn scaling_e17_extreme_scaling_long_run_ignored() {
    let delta = 1.0e3_f32;
    let d = 8_usize;
    let n = 10_000_usize;
    let (xs0, ys0) = phase_shifted_data(d, n);
    for &lambda in &[1.0_f32, 0.9] {
        for &ratio in &[1.0_f32, 1.0e6] {
            let mut factors = vec![1.0_f32; d];
            factors[d - 1] = ratio;
            let xs = scale_features(&xs0, &factors);
            let model = Rls::new(d, lambda, delta).unwrap();
            let run = scaling_run(&model, &xs, &ys0);
            println!(
                "E17 d={d} lambda={lambda} ratio={ratio} guard={:?} pred={:?} res={:?} kappa={} maxP={} maxw={}",
                run.guard,
                run.prediction_error,
                run.residual,
                run.kappa,
                max_abs_matrix(&run.state.p),
                run.state
                    .w
                    .as_slice()
                    .iter()
                    .fold(0.0_f64, |max, &value| max.max(f64::from(value).abs())),
            );
        }
    }
}

#[test]
fn scaling_e15_accumulation_at_n_1000() {
    // N-growth: the pre-registered pass region (ratio <= 10) must still pass
    // after 1000 observations, so accumulation must not create a boundary
    // inside the claimed envelope.
    let delta = 1.0e3_f32;
    let (xs0, ys0) = phase_shifted_data(2, 1000);
    for &ratio in &[1.0_f32, 10.0, 100.0, 1_000.0] {
        let xs = scale_features(&xs0, &[1.0, ratio]);
        let model = Rls::new(2, 1.0, delta).unwrap();
        let run = scaling_run(&model, &xs, &ys0);
        classify_scaling(&format!("E15 N=1000 ratio={ratio}"), &run, ratio <= 10.0);
    }
}

// --- RLS stage 4: dimension, horizon, and boundary attribution (R4) -----------
//
// Test-local `f64` diagnostics only. The literal weighted objective plus a
// direct solve remain the correctness oracle; the `A <- lambda A + x x^T`
// recurrence is never the oracle. The one narrow eigensolver exception below
// exists because R3's `tr(A) tr(A^{-1})` proxy is a loose upper bound that
// cannot separate dimension `D` from conditioning `kappa` above `D = 2`, which
// is exactly the R4 attribution question.

/// `max_i |v_i|` for a raw `f32` slice.
fn max_abs_vector(values: &[f32]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |max, &value| max.max(f64::from(value).abs()))
}

/// Test-local `f64` symmetric Jacobi eigensolver for `D <= 64`, returning the
/// eigenvalues in ascending order. It exists only to measure the exact
/// condition number `kappa_2(A) = lambda_max / lambda_min`; it is not a
/// `foundation` primitive, is not exported, and changes no production path. The
/// rotation method is iterative, so the sweep loop is capped and stops once the
/// off-diagonal Frobenius mass falls below `1e-12 * trace`; `lambda_min` is
/// clamped away from zero for ratio safety only. Index arithmetic is inherent
/// to the rotations, hence the range-loop allowance.
#[allow(clippy::needless_range_loop)]
fn symmetric_eigenvalues(a: &[Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    let mut m = a.to_vec();
    for _ in 0..(100 * n.max(1)) {
        let mut off_diagonal = 0.0_f64;
        for row in 0..n {
            for col in (row + 1)..n {
                off_diagonal += m[row][col] * m[row][col];
            }
        }
        let trace_abs: f64 = (0..n).map(|index| m[index][index].abs()).sum();
        if off_diagonal.sqrt() <= 1e-12 * trace_abs.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p][q];
                if apq.abs() <= f64::EPSILON * (m[p][p].abs() + m[q][q].abs()).max(1e-300) {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let app = m[p][p];
                let aqq = m[q][q];
                for k in 0..n {
                    if k != p && k != q {
                        let mkp = m[k][p];
                        let mkq = m[k][q];
                        m[k][p] = c * mkp - s * mkq;
                        m[p][k] = m[k][p];
                        m[k][q] = s * mkp + c * mkq;
                        m[q][k] = m[k][q];
                    }
                }
                m[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                m[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                m[p][q] = 0.0;
                m[q][p] = 0.0;
            }
        }
    }
    let mut eigenvalues: Vec<f64> = (0..n).map(|index| m[index][index]).collect();
    eigenvalues.sort_by(|left, right| left.partial_cmp(right).unwrap());
    eigenvalues
}

/// `(kappa_2(A), lambda_min(A))` from the Jacobi spectrum.
fn jacobi_conditioning(a: &[Vec<f64>]) -> (f64, f64) {
    let eigenvalues = symmetric_eigenvalues(a);
    let largest = eigenvalues.last().copied().unwrap_or(0.0);
    let smallest = eigenvalues.first().copied().unwrap_or(0.0).max(1e-300);
    (largest / smallest, smallest)
}

/// Dense design `x_t = diag(sigma) z_t` with `z_t` dense from the fixed-seed
/// LCG; the intended (pre-measurement) data conditioning is
/// `(sigma_max/sigma_min)^2`. The realized `kappa` is measured from `A` and is
/// never assumed.
fn conditioning_data(d: usize, n: usize, sigma_min: f32, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut lcg = Lcg::new(seed);
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            (0..d)
                .map(|axis| {
                    let sigma = if axis == 0 { sigma_min } else { 1.0 };
                    sigma * 2.0 * lcg.next_f32()
                })
                .collect()
        })
        .collect();
    let ys = linear_targets(&xs, &default_true_weights(d));
    (xs, ys)
}

/// Dense design whose first two axes are correlated by `rho` and whose remaining
/// axes are independent unit-scale draws; the correlated excitation regime.
fn correlated_dense_data(d: usize, n: usize, rho: f64, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut lcg = Lcg::new(seed);
    let complement = (1.0 - rho * rho).sqrt();
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let u = f64::from(2.0 * lcg.next_f32());
            let w = f64::from(2.0 * lcg.next_f32());
            let mut x = Vec::with_capacity(d);
            x.push(u as f32);
            x.push((rho * u + complement * w) as f32);
            for _ in 2..d {
                x.push(2.0 * lcg.next_f32());
            }
            x
        })
        .collect();
    let ys = linear_targets(&xs, &default_true_weights(d));
    (xs, ys)
}

/// Effective exponential weight `sum_{j=0}^{N-1} lambda^j`.
fn effective_weight(lambda: f64, n: usize) -> f64 {
    if lambda == 1.0 {
        n as f64
    } else {
        (1.0 - lambda.powi(n as i32)) / (1.0 - lambda)
    }
}

/// `max_t ||x_t||^2`, the excitation-energy term of the dynamic-range proxy.
fn max_squared_norm(xs: &[Vec<f32>]) -> f64 {
    xs.iter().fold(0.0_f64, |max, x| {
        let norm: f64 = x
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum();
        max.max(norm)
    })
}

/// Per-regime R4 measurements plus the committed state. `kappa`/`lambda_min`
/// and `prior_ratio` are design properties of the literal objective, so they are
/// available even when a guard stops the run; the error terms are `None` then.
struct AttributionRun {
    kappa: f64,
    lambda_min: f64,
    prediction_error: Option<f64>,
    residual: Option<f64>,
    normal_residual: Option<f64>,
    guard: Option<(usize, RlsError)>,
    max_p: f64,
    max_w: f64,
    min_denominator: f64,
    prior_ratio: f64,
    effective_weight: f64,
    dynamic_range: f64,
    state: RlsState,
}

fn attribution_run(model: &Rls, xs: &[Vec<f32>], ys: &[f32]) -> AttributionRun {
    let lambda = f64::from(model.lambda());
    let delta = f64::from(model.delta());
    let (a, b, w_ref) = oracle_solution(lambda, delta, xs, ys);
    let (kappa, lambda_min) = jacobi_conditioning(&a);

    let mut state = model.initial_state();
    let mut guard = None;
    let mut max_p = max_abs_matrix(&state.p);
    let mut max_w = max_abs_vector(state.w.as_slice());
    let mut min_denominator = f64::INFINITY;
    for (index, (x, &y)) in xs.iter().zip(ys).enumerate() {
        min_denominator = min_denominator.min(denominator(model, &state, x));
        match model.update(&state, &sample(x, y)) {
            Ok(next) => {
                state = next;
                max_p = max_p.max(max_abs_matrix(&state.p));
                max_w = max_w.max(max_abs_vector(state.w.as_slice()));
            }
            Err(error) => {
                guard = Some((index, error));
                break;
            }
        }
    }

    let (prediction_error, residual, normal_residual) = if guard.is_none() {
        let w_rec: Vec<f64> = state.w.as_slice().iter().map(|&v| f64::from(v)).collect();
        (
            Some(prediction_error(&state, xs, &w_ref)),
            Some(covariance_residual(&a, &state.p)),
            Some(max_abs_normal_residual(&a, &b, &w_rec)),
        )
    } else {
        (None, None, None)
    };
    AttributionRun {
        kappa,
        lambda_min,
        prediction_error,
        residual,
        normal_residual,
        guard,
        max_p,
        max_w,
        min_denominator,
        prior_ratio: lambda.powi(xs.len() as i32) / delta / lambda_min,
        effective_weight: effective_weight(lambda, xs.len()),
        dynamic_range: max_p * max_squared_norm(xs),
        state,
    }
}

/// Prints the full measurement record for one regime.
fn describe_run(tag: &str, d: usize, n: usize, lambda: f32, run: &AttributionRun) {
    println!(
        "{tag} d={d} n={n} lambda={lambda} kappa={:.4e} lambda_min={:.3e} prior_ratio={:.3e} \
         eff_weight={:.3} N/D={:.2} maxP={:.4e} maxw={:.4e} dr={:.4e} min_d={:.3e} \
         pred={:?} res={:?} norm={:?} guard={:?}",
        run.kappa,
        run.lambda_min,
        run.prior_ratio,
        run.effective_weight,
        n as f64 / d as f64,
        run.max_p,
        run.max_w,
        run.dynamic_range,
        run.min_denominator,
        run.prediction_error,
        run.residual,
        run.normal_residual,
        run.guard,
    );
}

/// Invariants hold at every regime. A failure inside the pre-registered pass
/// region is unexplained and fails the test; beyond it the failure is recorded
/// with its measured `kappa`/prior-ratio/dynamic-range signature and reported, so
/// it cannot mask a pass-region or invariant failure. Thresholds are frozen:
/// prediction `<= 5e-3`, covariance residual `<= 1e-2`.
fn classify_attribution(label: &str, run: &AttributionRun, within_pass_region: bool) {
    assert_eq!(
        symmetry_residual(&run.state.p),
        0.0,
        "{label}: symmetry residual"
    );
    assert!(
        run.state
            .w
            .as_slice()
            .iter()
            .chain(run.state.p.as_slice())
            .all(|value| value.is_finite()),
        "{label}: non-finite committed state"
    );
    let pass = run.guard.is_none()
        && run.prediction_error.is_some_and(|value| value <= 5e-3)
        && run.residual.is_some_and(|value| value <= 1e-2);
    if pass {
        return;
    }
    if within_pass_region {
        panic!(
            "{label}: failure inside the pre-registered pass region (guard={:?}, kappa={}, \
             prior_ratio={:.3e}, eff_weight={:.3})",
            run.guard, run.kappa, run.prior_ratio, run.effective_weight
        );
    }
    println!(
        "R4 boundary: {label} guard={:?} pred={:?} res={:?} kappa={:.4e} prior_ratio={:.3e} \
         dr={:.4e} maxP={:.4e}",
        run.guard,
        run.prediction_error,
        run.residual,
        run.kappa,
        run.prior_ratio,
        run.dynamic_range,
        run.max_p
    );
}

/// Parameter error of the committed state against the literal-objective oracle
/// on the prefix, sampled every `step` observations; the `N`-growth diagnostic
/// for the accumulation question.
fn sampled_prefix_weight_errors(
    model: &Rls,
    xs: &[Vec<f32>],
    ys: &[f32],
    step: usize,
) -> Vec<(usize, f64)> {
    let mut state = model.initial_state();
    let mut samples = Vec::new();
    for index in 0..xs.len() {
        if index > 0 && index % step == 0 {
            let (_, _, w_ref) = oracle_solution(
                f64::from(model.lambda()),
                f64::from(model.delta()),
                &xs[..index],
                &ys[..index],
            );
            let scale = 1.0
                + w_ref
                    .iter()
                    .fold(0.0_f64, |max, value| max.max(value.abs()));
            let error = state
                .w
                .as_slice()
                .iter()
                .zip(&w_ref)
                .fold(0.0_f64, |max, (&actual, &reference)| {
                    max.max((f64::from(actual) - reference).abs())
                })
                / scale;
            samples.push((index, error));
        }
        match model.update(&state, &sample(&xs[index], ys[index])) {
            Ok(next) => state = next,
            Err(_) => break,
        }
    }
    samples
}

#[test]
fn r4_e23_jacobi_eigensolver_self_test() {
    // The solver underpins every R4 kappa measurement, so pin it against exact
    // spectra before use.
    let two = vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]];
    let eigenvalues = symmetric_eigenvalues(&two);
    assert!(
        (eigenvalues[0] - 1.0).abs() <= 1e-12 && (eigenvalues[1] - 3.0).abs() <= 1e-12,
        "2x2 spectrum {eigenvalues:?}"
    );

    let tridiagonal = vec![
        vec![2.0_f64, 1.0, 0.0],
        vec![1.0, 2.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let eigenvalues = symmetric_eigenvalues(&tridiagonal);
    let root_two = 2.0_f64.sqrt();
    for (actual, expected) in eigenvalues
        .iter()
        .zip([2.0 - root_two, 2.0, 2.0 + root_two])
    {
        assert!(
            (actual - expected).abs() <= 1e-10,
            "tridiagonal spectrum {eigenvalues:?}"
        );
    }

    let diagonal = vec![
        vec![3.0_f64, 0.0, 0.0],
        vec![0.0, 5.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    assert_eq!(symmetric_eigenvalues(&diagonal), vec![1.0, 3.0, 5.0]);

    // Dense SPD 2x2: Jacobi kappa must agree with the closed form.
    let dense = vec![vec![4.0_f64, 1.5], vec![1.5, 0.75]];
    let kappa = jacobi_conditioning(&dense).0;
    assert!(
        (kappa - kappa_2x2(&dense)).abs() <= 1e-9 * kappa,
        "Jacobi kappa {kappa} vs closed form {}",
        kappa_2x2(&dense)
    );
}

#[test]
fn r4_e23_deterministic_controls() {
    // Validate the harness and the kappa measurement against designs whose
    // spectra are known exactly.
    let delta = 1.0e3_f32;

    // Scaled identity: x_t = c e_{t mod D} with lambda = 1 gives
    // A = (N/D) c^2 I + (1/delta) I, exactly isotropic => kappa = 1.
    let d = 4_usize;
    let n = 64_usize;
    let c = 2.0_f32;
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|t| {
            let mut x = vec![0.0_f32; d];
            x[t % d] = c;
            x
        })
        .collect();
    let ys = linear_targets(&xs, &default_true_weights(d));
    let model = Rls::new(d, 1.0, delta).unwrap();
    let run = attribution_run(&model, &xs, &ys);
    describe_run("control", d, n, 1.0, &run);
    assert!(
        (run.kappa - 1.0).abs() <= 1e-9,
        "scaled identity kappa {}",
        run.kappa
    );
    classify_attribution("E23 scaled identity", &run, true);

    // Cyclic one-hot with unequal amplitudes: A = diag(n0 c0^2 + 1/delta,
    // n1 c1^2 + 1/delta), a known kappa.
    let n = 64_usize;
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|t| {
            if t % 2 == 0 {
                vec![1.0_f32, 0.0]
            } else {
                vec![0.0_f32, 2.0]
            }
        })
        .collect();
    let ys = linear_targets(&xs, &default_true_weights(2));
    let model = Rls::new(2, 1.0, delta).unwrap();
    let run = attribution_run(&model, &xs, &ys);
    let half = (n / 2) as f64;
    // kappa = lambda_max / lambda_min; the amplitude-2 axis dominates.
    let expected = (half * 4.0 + 1.0 / f64::from(delta)) / (half + 1.0 / f64::from(delta));
    assert!(
        (run.kappa - expected).abs() <= 1e-9 * expected,
        "cyclic one-hot kappa {} vs {expected}",
        run.kappa
    );
    classify_attribution("E23 cyclic one-hot", &run, true);

    // Rank-1 repeated: one live direction, the rest inflate and stay exactly
    // zero; finite, symmetric, no guard, live-axis P pinned by the R1 closed
    // form. The unexcited axes make the objective ill-conditioned, so this is a
    // harness control, not a pass-region case.
    let d = 4_usize;
    let x = 1.5_f32;
    let n = 200_usize;
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut row = vec![0.0_f32; d];
            row[0] = x;
            row
        })
        .collect();
    let ys = vec![2.0_f32; n];
    let model = Rls::new(d, 0.9, delta).unwrap();
    let run = attribution_run(&model, &xs, &ys);
    describe_run("control rank-1", d, n, 0.9, &run);
    assert!(run.guard.is_none(), "rank-1 control guarded");
    assert_eq!(symmetry_residual(&run.state.p), 0.0);
    let expected_live = (1.0_f64 - 0.9_f64) / f64::from(x * x);
    let actual_live = diagonal(&run.state.p, 0);
    assert!(
        (actual_live - expected_live).abs() <= 1e-2 * expected_live,
        "rank-1 live P {actual_live} vs {expected_live}"
    );
    assert_f32_bits_eq(run.state.w.as_slice()[1], 0.0);
    classify_attribution("E23 rank-1 repeated", &run, false);
}

#[test]
fn r4_e18a_dimension_track_a_controlled() {
    // Track A: hold conditioning, feature scale, excitation, and samples per
    // dimension `N/D = 64` fixed, then grow `D`. `delta_D = delta0 * D/2` is the
    // plan's prior control; its realized prior-to-data ratio is recorded (not a
    // threshold) so any `D` degradation cannot be silently blamed on it. Only
    // points whose measured Jacobi kappa lies in `[0.5, 2]x` the intended value
    // are compared; a dimension penalty surviving that gate is a
    // representation-limit candidate, reported not implemented.
    let delta0 = 1.0e3_f32;
    let samples_per_dimension = 64_usize;
    let dial = [(1.0_f32, "kappa1"), (0.1_f32, "kappa100")];
    for &d in &[2_usize, 4, 8, 16] {
        let n = samples_per_dimension * d;
        let delta_d = delta0 * d as f32 / 2.0;
        for &(sigma_min, dial_label) in &dial {
            let (xs, ys) = conditioning_data(d, n, sigma_min, 0xC0FFEE ^ d as u64);
            for &lambda in &[1.0_f32, 0.99, 0.9] {
                let model = Rls::new(d, lambda, delta_d).unwrap();
                let run = attribution_run(&model, &xs, &ys);
                describe_run(&format!("E18a {dial_label}"), d, n, lambda, &run);
                let intended = if sigma_min == 1.0 { 1.0 } else { 100.0 };
                // The measured-kappa band is a *gate* for matched-conditioning
                // comparisons (C2/C8), not a guarantee of the random dial: at
                // lambda < 1 the effective sample size can be too small for the
                // dense design to stay near-isotropic. Out-of-band points are
                // recorded and excluded from any cross-D attribution, but they
                // still sit inside the measured pass region B6 (kappa <= 1e3).
                let in_band = run.kappa >= 0.5 * intended && run.kappa <= 2.0 * intended;
                println!(
                    "E18a band {dial_label} d={d} lambda={lambda} intended={intended} measured={:.4e} in_band={in_band}",
                    run.kappa
                );
                classify_attribution(
                    &format!("E18a {dial_label} d={d} lambda={lambda}"),
                    &run,
                    d <= 16 && run.kappa <= 1e3,
                );
            }
        }
    }
}

#[test]
fn r4_e18b_dimension_track_b_fixed_horizon() {
    // Track B: fix `N`, grow `D`, so `N/D` falls. A failure here is not a
    // representation boundary unless conditioning and sample density are both
    // controlled, so the pass region additionally requires `N/D >= 16`; below
    // that the observation is classified SAMPLE-DENSITY / IDENTIFIABILITY.
    let delta = 1.0e3_f32;
    for &n in &[200_usize, 1000] {
        for &d in &[2_usize, 4, 8, 16] {
            let (xs, ys) = phase_shifted_data(d, n);
            for &lambda in &[1.0_f32, 0.99, 0.9] {
                let model = Rls::new(d, lambda, delta).unwrap();
                let run = attribution_run(&model, &xs, &ys);
                describe_run("E18b", d, n, lambda, &run);
                let density = n as f64 / d as f64;
                classify_attribution(
                    &format!("E18b n={n} d={d} lambda={lambda}"),
                    &run,
                    n <= 1000 && d <= 16 && run.kappa <= 1e3 && density >= 16.0,
                );
            }
        }
    }
}

#[test]
fn r4_e19_horizon_accumulation() {
    // Horizon: `D` fixed, `N` grows at fixed design/scale/kappa. For lambda < 1
    // the weighted objective saturates after `~1/(1-lambda)`, so the parameter
    // error should plateau; for lambda = 1 `P -> 0` and the dynamic range grows.
    // The sampled prefix-oracle parameter error is the N-growth signature.
    let delta = 1.0e3_f32;
    for &d in &[4_usize, 16] {
        for &sigma_min in &[1.0_f32, 0.1] {
            for &n in &[200_usize, 1000] {
                let (xs, ys) = conditioning_data(d, n, sigma_min, 0xA11CE ^ d as u64 ^ n as u64);
                for &lambda in &[1.0_f32, 0.99, 0.9] {
                    let model = Rls::new(d, lambda, delta).unwrap();
                    let run = attribution_run(&model, &xs, &ys);
                    describe_run("E19", d, n, lambda, &run);
                    let step = (n / 20).max(1);
                    let growth = sampled_prefix_weight_errors(&model, &xs, &ys, step);
                    println!("E19 growth d={d} n={n} lambda={lambda} samples={growth:?}");
                    classify_attribution(
                        &format!("E19 d={d} n={n} lambda={lambda}"),
                        &run,
                        n <= 1000 && d <= 16 && run.kappa <= 1e3,
                    );
                }
            }
        }
    }
}

#[test]
fn r4_e20_forgetting_factor_cross() {
    // Forgetting: lambda is already crossed into E18/E19; this pins the
    // effective weight `sum lambda^j` ordering and confirms the pass region holds
    // across the memory range at fixed design.
    let delta = 1.0e3_f32;
    let d = 4_usize;
    let n = 1000_usize;
    let (xs, ys) = conditioning_data(d, n, 1.0, 0xF0F0);
    let mut previous_weight = f64::INFINITY;
    for &lambda in &[1.0_f32, 0.99, 0.9] {
        let model = Rls::new(d, lambda, delta).unwrap();
        let run = attribution_run(&model, &xs, &ys);
        describe_run("E20", d, n, lambda, &run);
        let expected = effective_weight(f64::from(lambda), n);
        assert!(
            (run.effective_weight - expected).abs() <= 1e-9 * expected.max(1.0),
            "E20 lambda={lambda}: effective weight {} vs {expected}",
            run.effective_weight
        );
        assert!(
            run.effective_weight < previous_weight,
            "E20 lambda={lambda}: effective weight {} not below {previous_weight}",
            run.effective_weight
        );
        previous_weight = run.effective_weight;
        classify_attribution(&format!("E20 d={d} n={n} lambda={lambda}"), &run, true);
    }
}

#[test]
fn r4_e21_excitation_regimes() {
    // Excitation: dense full-rank and correlated must pass inside the
    // well-conditioned pass region (B5). Partial (an unexcited axis) and weak
    // (small sigma_min) are deliberately outside the fully-excited envelope, so
    // their degradation is recorded, not failed.
    let delta = 1.0e3_f32;
    let samples_per_dimension = 64_usize;
    for &d in &[4_usize, 16] {
        let n = samples_per_dimension * d;
        let (xs_dense, ys_dense) = conditioning_data(d, n, 1.0, 0xD3D3 ^ d as u64);
        let (xs_partial, ys_partial) = unexcited_axis_data(d, n, d - 1);
        let (xs_correlated, ys_correlated) = correlated_dense_data(d, n, 0.99, 0xD4D4 ^ d as u64);
        let (xs_base, ys_weak) = conditioning_data(d, n, 1.0, 0xD5D5 ^ d as u64);
        let mut factors = vec![1.0_f32; d];
        factors[d - 1] = 1.0e-2;
        let xs_weak = scale_features(&xs_base, &factors);
        let regimes = [
            ("dense", &xs_dense, &ys_dense, true),
            ("correlated", &xs_correlated, &ys_correlated, true),
            ("partial", &xs_partial, &ys_partial, false),
            ("weak", &xs_weak, &ys_weak, false),
        ];
        for &lambda in &[1.0_f32, 0.9] {
            for &(regime, xs, ys, within) in &regimes {
                let model = Rls::new(d, lambda, delta).unwrap();
                let run = attribution_run(&model, xs, ys);
                describe_run(&format!("E21 {regime}"), d, n, lambda, &run);
                classify_attribution(
                    &format!("E21 d={d} lambda={lambda} regime={regime}"),
                    &run,
                    within && run.kappa <= 1e3,
                );
            }
        }
    }
}

#[test]
fn r4_e22_scale_conditioning_dimension() {
    // Scale x conditioning x dimension. The uniform matched-`delta_s = delta/s^2`
    // form is exact similarity (kappa invariant) and must pass for every tested
    // `s`/`D` (B8). Unequal last-axis disparity is recorded beyond `s <= 10`, the
    // R3 passing envelope.
    let delta = 1.0e3_f32;
    let samples_per_dimension = 64_usize;
    for &d in &[2_usize, 8, 16] {
        let n = samples_per_dimension * d;
        let (xs0, ys0) = conditioning_data(d, n, 1.0, 0x5CA1E ^ d as u64);
        for &lambda in &[1.0_f32, 0.9] {
            for &s in &[1.0_f32, 10.0, 100.0] {
                let mut factors = vec![1.0_f32; d];
                factors[d - 1] = s;
                let xs = scale_features(&xs0, &factors);
                let model = Rls::new(d, lambda, delta).unwrap();
                let run = attribution_run(&model, &xs, &ys0);
                describe_run("E22 disparity", d, n, lambda, &run);
                classify_attribution(
                    &format!("E22 disparity d={d} lambda={lambda} s={s}"),
                    &run,
                    s <= 10.0,
                );

                let factors = vec![s; d];
                let xs = scale_features(&xs0, &factors);
                let delta_s = delta / (s * s);
                let model = Rls::new(d, lambda, delta_s).unwrap();
                let run = attribution_run(&model, &xs, &ys0);
                describe_run("E22 uniform-matched", d, n, lambda, &run);
                classify_attribution(
                    &format!("E22 uniform-matched d={d} lambda={lambda} s={s}"),
                    &run,
                    true,
                );
            }
        }
    }
}

#[test]
#[ignore = "release-only diagnostic: Track A D=32,64 at N=64*D"]
fn r4_e18a_dimension_track_a_large_ignored() {
    let delta0 = 1.0e3_f32;
    let samples_per_dimension = 64_usize;
    for &d in &[32_usize, 64] {
        let n = samples_per_dimension * d;
        let delta_d = delta0 * d as f32 / 2.0;
        for &(sigma_min, dial_label) in &[(1.0_f32, "kappa1"), (0.1_f32, "kappa100")] {
            let (xs, ys) = conditioning_data(d, n, sigma_min, 0xC0FFEE ^ d as u64);
            for &lambda in &[1.0_f32, 0.99, 0.9] {
                let model = Rls::new(d, lambda, delta_d).unwrap();
                let run = attribution_run(&model, &xs, &ys);
                describe_run(&format!("E18a-large {dial_label}"), d, n, lambda, &run);
                classify_attribution(
                    &format!("E18a-large {dial_label} d={d} lambda={lambda}"),
                    &run,
                    false,
                );
            }
        }
    }
}

#[test]
#[ignore = "release-only diagnostic: Track B N in {5000,10000}"]
fn r4_e18b_dimension_track_b_large_ignored() {
    let delta = 1.0e3_f32;
    for &n in &[5000_usize, 10_000] {
        for &d in &[2_usize, 4, 8, 16, 32] {
            let (xs, ys) = phase_shifted_data(d, n);
            for &lambda in &[1.0_f32, 0.99, 0.9] {
                let model = Rls::new(d, lambda, delta).unwrap();
                let run = attribution_run(&model, &xs, &ys);
                describe_run("E18b-large", d, n, lambda, &run);
                classify_attribution(
                    &format!("E18b-large n={n} d={d} lambda={lambda}"),
                    &run,
                    false,
                );
            }
        }
    }
}

#[test]
#[ignore = "release-only diagnostic: horizon N in {5000,10000}"]
fn r4_e19_horizon_large_ignored() {
    let delta = 1.0e3_f32;
    for &d in &[4_usize, 16] {
        for &sigma_min in &[1.0_f32, 0.1] {
            for &n in &[5000_usize, 10_000] {
                let (xs, ys) = conditioning_data(d, n, sigma_min, 0xA11CE ^ d as u64 ^ n as u64);
                for &lambda in &[1.0_f32, 0.99, 0.9] {
                    let model = Rls::new(d, lambda, delta).unwrap();
                    let run = attribution_run(&model, &xs, &ys);
                    describe_run("E19-large", d, n, lambda, &run);
                    let step = (n / 20).max(1);
                    let growth = sampled_prefix_weight_errors(&model, &xs, &ys, step);
                    println!("E19-large growth d={d} n={n} lambda={lambda} samples={growth:?}");
                    classify_attribution(
                        &format!("E19-large d={d} n={n} lambda={lambda}"),
                        &run,
                        false,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "release-only diagnostic: E22 s=1000 at D=16"]
fn r4_e22_extreme_scale_ignored() {
    let delta = 1.0e3_f32;
    let d = 16_usize;
    let n = 64 * d;
    let (xs0, ys0) = conditioning_data(d, n, 1.0, 0x5CA1E ^ d as u64);
    for &lambda in &[1.0_f32, 0.9] {
        let s = 1000.0_f32;
        let mut factors = vec![1.0_f32; d];
        factors[d - 1] = s;
        let xs = scale_features(&xs0, &factors);
        let model = Rls::new(d, lambda, delta).unwrap();
        let run = attribution_run(&model, &xs, &ys0);
        describe_run("E22-extreme disparity", d, n, lambda, &run);
        classify_attribution(
            &format!("E22-extreme disparity d={d} lambda={lambda} s={s}"),
            &run,
            false,
        );

        let factors = vec![s; d];
        let xs = scale_features(&xs0, &factors);
        let delta_s = delta / (s * s);
        let model = Rls::new(d, lambda, delta_s).unwrap();
        let run = attribution_run(&model, &xs, &ys0);
        describe_run("E22-extreme uniform-matched", d, n, lambda, &run);
        classify_attribution(
            &format!("E22-extreme uniform-matched d={d} lambda={lambda} s={s}"),
            &run,
            true,
        );
    }
}

#[test]
#[ignore = "release-only diagnostic: lambda in {0.999,0.9999}, N ~ 10/(1-lambda)"]
fn r4_e24_lambda_near_one_ignored() {
    // lambda = 0.999/0.9999 have memory 1e3/1e4, so N ~ 10 memories lets the
    // run separate memory from horizon: the lambda = 1 shrink (P -> 0) is a
    // dynamic-range effect, while a lambda < 1 plateau at fixed N is expected.
    let delta = 1.0e3_f32;
    for &(d, n, lambda) in &[
        (2_usize, 10_000_usize, 0.999_f32),
        (4_usize, 10_000_usize, 0.999_f32),
        (2_usize, 100_000_usize, 0.9999_f32),
    ] {
        let (xs, ys) = conditioning_data(d, n, 1.0, 0x9999 ^ d as u64);
        for &candidate in &[lambda, 1.0_f32] {
            let model = Rls::new(d, candidate, delta).unwrap();
            let run = attribution_run(&model, &xs, &ys);
            describe_run("E24", d, n, candidate, &run);
            classify_attribution(&format!("E24 d={d} n={n} lambda={candidate}"), &run, true);
        }
    }
}
