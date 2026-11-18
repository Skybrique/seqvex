//! RLS R1 excitation evidence and test-side diagnostics.

use super::support::*;
use super::*;

// --- RLS excitation: test-side diagnostics (R1) -------------------------------
//
// R1 owns its test-local diagnostics (`trace`, `reference_covariance_error`,
// `overflow_update_number`, `weak_axis_data`, `rotating_data`). The shared
// diagnostics it consumes (`denominator`, `diagonal`, `symmetry_residual`,
// `matrix_infinity_norm`, `oracle_solution`, `covariance_residual`,
// `reference_covariance`, `d1_closed_form`, `run_until_first_failure`) live in
// `support`; the literal-objective `A_ref` comes from the D1 helpers there.
// All are computed in `f64` from the committed `f32` state. No production
// primitive, monitor, or eigensolver is added.

/// Sum of the diagonal of a square matrix.
fn trace(matrix: &Matrix) -> f64 {
    (0..matrix.rows())
        .map(|index| f64::from(matrix.get(index, index).unwrap()))
        .sum()
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
