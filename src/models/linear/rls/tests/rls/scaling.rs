//! RLS R3 feature-scaling and conditioning evidence.

use super::support::*;
use super::*;

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
