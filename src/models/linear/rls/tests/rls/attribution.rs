//! RLS R4 dimension, horizon and boundary attribution evidence.

use super::support::*;
use super::*;

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
