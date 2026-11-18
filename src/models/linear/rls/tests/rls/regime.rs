//! RLS R2 regime, noise, prequential and recovery evidence.

use super::support::*;
use super::*;

// --- RLS stage 2: regime shift, noise, prequential, recovery (R2) -------------
//
// Test-local `f64` diagnostics only; reuses the D1/R1 helpers. Serves the R2
// falsification rules: lambda = 1 must pool (not track), lambda < 1 must track
// under persistent excitation, noise must preserve the R1 numerical
// conclusions, and a retained/inflated P must recover once excitation resumes.

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
