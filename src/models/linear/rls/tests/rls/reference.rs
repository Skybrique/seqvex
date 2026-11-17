//! RLS core mathematical-reference tests: construction, sequence context,
//! the scalar and literal-objective oracle, long-run stability, and
//! ordering/permutation semantics. Chunking and precision/objective
//! evidence live in `batch` and `precision`.

use super::support::*;
use super::*;

// --- core-reference helpers -----------------------------------------------

pub(super) fn model(dimension: usize) -> Rls {
    Rls::new(dimension, 0.99, 1.0e4).unwrap()
}

pub(super) fn matrix_difference_infinity_norm(left: &Matrix, right: &Matrix) -> f64 {
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
