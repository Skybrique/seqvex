//! RLS precision-profile evidence: objective/quality acceptance (E9), native
//! f64 order controls, strict f32 named prefixes, prior/forgetting checks,
//! hand/rational controls, permutation/conversion semantics, and the native f64
//! discriminator.

mod support;

use self::support::{
    a_objective_for_pairs, a_order_diagnostics, assert_a_objective, assert_c_objective,
    c_objective_for_pairs, c_order_diagnostics, permutation_pairs32, permutation_pairs64, permuted,
    run_a_prefixes, run_c_pairs, run_c_prefixes, sinusoidal_data64,
};
use super::support::*;
use super::*;

#[test]
fn acceptance_checked_prequential_matches_literal_objective_and_e9_quality() {
    // Exact minimization is an exact-arithmetic property: `w_t` is the exact
    // minimiser of `J_t` in exact arithmetic; the `f32` recurrence only
    // approximates it and is checked here within the established tolerance.
    let delta = 1.0e3_f32;
    let d = 2_usize;
    let split = 400_usize;
    let n = 1400_usize;
    let w_a = [0.7_f32, -1.3];
    let w_b = [-0.5_f32, 1.1];
    let shift = w_a.iter().zip(&w_b).fold(0.0_f64, |m, (&a, &b)| {
        m.max((f64::from(a) - f64::from(b)).abs())
    });
    let (xs, ys) = regime_data(d, n, split, &w_a, &w_b);

    let mean_abs = |errors: &[f64], range: std::ops::Range<usize>| -> f64 {
        let slice = &errors[range];
        slice.iter().map(|value| value.abs()).sum::<f64>() / slice.len() as f64
    };

    let mut control_late = f64::NAN;
    let mut late_errors: Vec<f64> = Vec::new();

    for &lambda in &[1.0_f32, 0.9, 0.95, 0.99] {
        let model = Rls::new(d, lambda, delta).unwrap();
        let mut state = model.initial_state();

        // Learn zero-based rows 0..400 through exactly 400 public updates.
        for index in 0..split {
            state = model
                .update(&state, &sample(&xs[index], ys[index]))
                .unwrap_or_else(|error| {
                    panic!("update failed at learn row {index} (lambda {lambda}): {error:?}")
                });
        }
        // Learned-prefix state against the independent literal weighted objective.
        let (_, _, prefix_oracle) = oracle_solution(
            f64::from(lambda),
            f64::from(delta),
            &xs[..split],
            &ys[..split],
        );
        assert_weights_match_oracle(&state, &prefix_oracle, 5e-3);

        let is_control = lambda == 1.0_f32;
        if is_control {
            let scale = 1.0 + prefix_oracle.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                weight_distance_f64(&state, &prefix_oracle) <= 5e-3 * scale,
                "lambda=1 at t={split}: not the pooled solution"
            );
        }

        // Score rows 400..1400: checked prediction from the committed state,
        // then reveal the target and update. The first scored row is index 400,
        // using the state committed after the 400th update.
        let mut errors: Vec<f64> = Vec::with_capacity(n - split);
        for index in split..n {
            let features = Vector::from_slice(&xs[index]);
            let prediction = model
                .predict_checked(&state, &features)
                .unwrap_or_else(|error| {
                    panic!("predict_checked failed at row {index} (lambda {lambda}): {error:?}")
                });
            errors.push(f64::from(ys[index]) - f64::from(prediction));
            state = model
                .update(&state, &sample(&xs[index], ys[index]))
                .unwrap_or_else(|error| {
                    panic!("update failed at row {index} (lambda {lambda}): {error:?}")
                });
            if is_control && (index + 1 == n || (index + 1) % 400 == 0) {
                let (_, _, pooled) = oracle_solution(
                    f64::from(lambda),
                    f64::from(delta),
                    &xs[..index + 1],
                    &ys[..index + 1],
                );
                let scale = 1.0 + pooled.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                assert!(
                    weight_distance_f64(&state, &pooled) <= 5e-3 * scale,
                    "lambda=1 at t={}: not the pooled solution",
                    index + 1
                );
            }
        }

        // Final state against the independent literal weighted objective.
        let (_, _, final_oracle) = oracle_solution(f64::from(lambda), f64::from(delta), &xs, &ys);
        assert_weights_match_oracle(&state, &final_oracle, 5e-3);

        // Retained E9 quality assertions on this same fixture (no new margin).
        let late = mean_abs(&errors, 900..1000);
        if is_control {
            control_late = late;
        } else {
            let early = mean_abs(&errors, 0..20);
            assert!(
                late <= 0.1 * early,
                "lambda={lambda}: prequential error did not decay ({early} -> {late})"
            );
            late_errors.push(late);
        }
    }

    assert!(
        control_late >= 0.1 * shift,
        "lambda=1: post-prefix prequential error collapsed to {control_late}"
    );
    for &late in &late_errors {
        assert!(
            control_late > 5.0 * late,
            "lambda=1 error {control_late} not clearly above lambda<1 error {late}"
        );
    }
}

#[test]
fn native_f64_order4_same_order_controls_and_objective_match() {
    let pairs = [
        ([1.0_f64, 0.0], 1.0),
        ([0.0, 1.0], -2.0),
        ([0.5, 0.5], 3.0),
        ([-0.25, 0.75], 0.5),
    ];
    for lambda in [1.0_f64, 0.9, 0.95] {
        let model = model64(2, lambda, 5.0);
        let mut direct = model.initial_state();
        let mut reference = ref_initial64(2, 5.0);
        let mut expected_states = vec![to_state64(&reference)];
        let observations: Vec<_> = pairs.iter().map(|(x, y)| sample64(x, *y)).collect();
        for (index, (x, y)) in pairs.iter().enumerate() {
            let expected_prediction = ref_predict64(&reference, x);
            let actual_prediction = model.predict(&direct, &Vector64::from_slice(x)).unwrap();
            assert_f64_bits_eq(actual_prediction, expected_prediction);
            direct = model.update(&direct, &sample64(x, *y)).unwrap();
            reference = ref_step64(&reference, x, *y, lambda);
            let expected = to_state64(&reference);
            assert_state64_bitwise_eq(&direct, &expected);
            expected_states.push(expected);
            assert_a_objective(
                lambda,
                5.0,
                &direct,
                &pairs[..=index]
                    .iter()
                    .map(|(x, _)| x.to_vec())
                    .collect::<Vec<_>>(),
                &pairs[..=index].iter().map(|(_, y)| *y).collect::<Vec<_>>(),
                1e-12,
                1e-10,
            );
        }
        let mut executor = StreamingExecutor::new(&model, model.initial_state());
        for (index, observation) in observations.iter().enumerate() {
            executor.process_one(observation).unwrap();
            assert_state64_bitwise_eq(executor.state(), &expected_states[index + 1]);
        }
        assert_state64_bitwise_eq(executor.state(), &direct);
        let mut chunked = model.initial_state();
        for (start, end) in [(0, 0), (0, 2), (2, 2), (2, 4)] {
            chunked = model
                .process_batch_reference(chunked, &observations[start..end], end - start)
                .unwrap();
            assert_state64_bitwise_eq(&chunked, &expected_states[end]);
        }
        assert_state64_bitwise_eq(&chunked, &direct);

        let mut failure_rows = observations.clone();
        failure_rows[2] = sample64(&pairs[2].0, f64::NAN);
        match model.process_batch_reference(model.initial_state(), &failure_rows, 4) {
            Err(RlsBatchError64::Transition {
                failed_index,
                state,
                error,
            }) => {
                assert_eq!(failed_index, 2);
                assert_eq!(error, RlsError64::NonFiniteInput);
                assert_state64_bitwise_eq(&state, &expected_states[2]);
            }
            other => panic!("expected A failure at ORDER4 row 2, got {other:?}"),
        }
    }
}

#[test]
fn strict_f32_order4_and_named_wls_prefixes_keep_both_controls() {
    let pairs = [
        ([1.0_f32, 0.0], 1.0),
        ([0.0, 1.0], -2.0),
        ([0.5, 0.5], 3.0),
        ([-0.25, 0.75], 0.5),
    ];
    for lambda in [1.0_f32, 0.9, 0.95] {
        let model = Rls::new(2, lambda, 5.0).unwrap();
        let mut direct = model.initial_state();
        let mut reference = ref_initial(2, 5.0);
        let mut expected_states = vec![to_state(&reference)];
        let observations: Vec<_> = pairs.iter().map(|(x, y)| sample(x, *y)).collect();
        for (index, (x, y)) in pairs.iter().enumerate() {
            let expected_prediction = ref_predict(&reference, x);
            let actual_prediction = model.predict(&direct, &Vector::from_slice(x)).unwrap();
            assert_f32_bits_eq(actual_prediction, expected_prediction);
            direct = model.update(&direct, &sample(x, *y)).unwrap();
            reference = ref_step(&reference, x, *y, lambda);
            let expected = to_state(&reference);
            assert_state_bitwise_eq(&direct, &expected);
            expected_states.push(expected);
            assert_c_objective(
                f64::from(lambda),
                5.0,
                &direct,
                &pairs[..=index]
                    .iter()
                    .map(|(x, _)| x.to_vec())
                    .collect::<Vec<_>>(),
                &pairs[..=index].iter().map(|(_, y)| *y).collect::<Vec<_>>(),
            );
        }
        let mut executor = StreamingExecutor::new(&model, model.initial_state());
        for (index, observation) in observations.iter().enumerate() {
            executor.process_one(observation).unwrap();
            assert_state_bitwise_eq(executor.state(), &expected_states[index + 1]);
        }
        assert_state_bitwise_eq(executor.state(), &direct);
        let mut chunked = model.initial_state();
        for (start, end) in [(0, 0), (0, 2), (2, 2), (2, 4)] {
            chunked = model
                .process_batch_reference(chunked, &observations[start..end], end - start)
                .unwrap();
            assert_state_bitwise_eq(&chunked, &expected_states[end]);
        }
        assert_state_bitwise_eq(&chunked, &direct);

        let mut failure_rows = observations.clone();
        failure_rows[2] = sample(&pairs[2].0, f32::NAN);
        match model.process_batch_reference(model.initial_state(), &failure_rows, 4) {
            Err(RlsBatchError::Transition {
                failed_index,
                state,
                error,
            }) => {
                assert_eq!(failed_index, 2);
                assert_eq!(error, RlsError::NonFiniteInput);
                assert_state_bitwise_eq(&state, &expected_states[2]);
            }
            other => panic!("expected C failure at ORDER4 row 2, got {other:?}"),
        }
    }

    let cases = [
        (2, 20, 1.0_f64),
        (2, 100, 1.0),
        (4, 20, 1.0),
        (4, 100, 1.0),
        (2, 20, 0.9),
        (2, 60, 0.9),
        (4, 20, 0.9),
        (4, 60, 0.9),
        (2, 20, 0.99),
        (2, 60, 0.99),
        (4, 20, 0.99),
        (4, 60, 0.99),
    ];
    for (dimension, rows, lambda) in cases {
        let (xs64, ys64) = sinusoidal_data64(dimension, rows);
        let model_a = model64(dimension, lambda, 1000.0);
        run_a_prefixes(&model_a, &xs64, &ys64);

        let (xs32, ys32) = sinusoidal_data(dimension, rows);
        let model_c = Rls::new(dimension, lambda as f32, 1000.0_f32).unwrap();
        run_c_prefixes(&model_c, &xs32, &ys32);
    }

    let model_a = model64(2, 0.9, 1.0);
    let xs64 = vec![vec![1.0, 2.0]; 40];
    let ys64 = vec![3.0; 40];
    run_a_prefixes(&model_a, &xs64, &ys64);

    let model_c = Rls::new(2, 0.9, 1.0).unwrap();
    let xs32 = vec![vec![1.0, 2.0]; 40];
    let ys32 = vec![3.0; 40];
    run_c_prefixes(&model_c, &xs32, &ys32);
}

#[test]
fn prior_forgetting_and_initialization_are_checked_for_every_prefix() {
    let xs64 = vec![vec![0.0]; 12];
    let ys64: Vec<f64> = (0..12).map(|j| j as f64).collect();
    let model_a = model64(1, 0.9, 1000.0);
    let mut state_a = model_a.initial_state();
    let mut reference_a = ref_initial64(1, 1000.0);
    assert_f64_bits_eq(state_a.weights().get(0).unwrap(), 0.0);
    assert_f64_bits_eq(state_a.covariance().get(0, 0).unwrap(), 1000.0);
    for index in 0..12 {
        assert_f64_bits_eq(
            model_a
                .predict(&state_a, &Vector64::from_slice(&xs64[index]))
                .unwrap(),
            ref_predict64(&reference_a, &xs64[index]),
        );
        state_a = model_a
            .update(&state_a, &sample64(&xs64[index], ys64[index]))
            .unwrap();
        reference_a = ref_step64(&reference_a, &xs64[index], ys64[index], 0.9);
        assert_state64_bitwise_eq(&state_a, &to_state64(&reference_a));
        assert_f64_bits_eq(state_a.weights().get(0).unwrap(), 0.0);
        assert_a_objective(
            0.9,
            1000.0,
            &state_a,
            &xs64[..=index],
            &ys64[..=index],
            1e-12,
            1e-10,
        );
    }

    let xs32 = vec![vec![0.0_f32]; 12];
    let ys32: Vec<f32> = (0..12).map(|j| j as f32).collect();
    let model_c = Rls::new(1, 0.9, 1000.0).unwrap();
    let mut state_c = model_c.initial_state();
    let mut reference_c = ref_initial(1, 1000.0);
    for index in 0..12 {
        assert_f32_bits_eq(
            model_c
                .predict(&state_c, &Vector::from_slice(&xs32[index]))
                .unwrap(),
            ref_predict(&reference_c, &xs32[index]),
        );
        state_c = model_c
            .update(&state_c, &sample(&xs32[index], ys32[index]))
            .unwrap();
        reference_c = ref_step(&reference_c, &xs32[index], ys32[index], 0.9);
        assert_state_bitwise_eq(&state_c, &to_state(&reference_c));
        assert_f32_bits_eq(state_c.w.get(0).unwrap(), 0.0);
        assert_c_objective(0.9, 1000.0, &state_c, &xs32[..=index], &ys32[..=index]);
    }
}

#[test]
fn native_f64_hand_and_rational_controls_are_exact() {
    let hand_model = model64(2, 1.0, 1.0);
    let mut hand_state = hand_model.initial_state();
    assert_f64_bits_eq(
        hand_model
            .predict(&hand_state, &Vector64::from_slice(&[1.0, 0.0]))
            .unwrap(),
        0.0,
    );
    hand_state = hand_model
        .update(&hand_state, &sample64(&[1.0, 0.0], 1.0))
        .unwrap();
    assert_eq!(hand_state.weights().as_slice(), &[0.5, 0.0]);
    assert_eq!(hand_state.covariance().as_slice(), &[0.5, 0.0, 0.0, 1.0]);
    assert_f64_bits_eq(
        hand_model
            .predict(&hand_state, &Vector64::from_slice(&[0.0, 1.0]))
            .unwrap(),
        0.0,
    );
    hand_state = hand_model
        .update(&hand_state, &sample64(&[0.0, 1.0], 3.0))
        .unwrap();
    assert_eq!(hand_state.weights().as_slice(), &[0.5, 1.5]);
    assert_eq!(hand_state.covariance().as_slice(), &[0.5, 0.0, 0.0, 0.5]);
    assert_f64_bits_eq(
        hand_model
            .predict(&hand_state, &Vector64::from_slice(&[1.0, 1.0]))
            .unwrap(),
        2.0,
    );

    let rational_model = model64(1, 1.0, 1.0);
    let mut rational_state = rational_model.initial_state();
    assert_f64_bits_eq(
        rational_model
            .predict(&rational_state, &Vector64::from_slice(&[1.0]))
            .unwrap(),
        0.0,
    );
    rational_state = rational_model
        .update(&rational_state, &sample64(&[1.0], 2.0))
        .unwrap();
    assert_eq!(rational_state.weights().as_slice(), &[1.0]);
    assert_eq!(rational_state.covariance().as_slice(), &[0.5]);
    assert_f64_bits_eq(
        rational_model
            .predict(&rational_state, &Vector64::from_slice(&[1.0]))
            .unwrap(),
        1.0,
    );
    rational_state = rational_model
        .update(&rational_state, &sample64(&[1.0], 4.0))
        .unwrap();
    assert_eq!(rational_state.weights().as_slice(), &[2.0]);
    let covariance_error = (rational_state.covariance().get(0, 0).unwrap() - 1.0 / 3.0).abs();
    assert!(
        covariance_error <= 1e-15,
        "1/3 covariance error {covariance_error}"
    );
    assert_f64_bits_eq(
        rational_model
            .predict(&rational_state, &Vector64::from_slice(&[1.0]))
            .unwrap(),
        2.0,
    );
}

#[test]
fn permutation_and_conversion_semantics_match_named_bounds() {
    let forward_pairs = permutation_pairs64(2);
    let permuted_pairs = permuted(&forward_pairs);
    for (lambda, order_weight_tolerance, order_covariance_tolerance) in
        [(1.0_f64, 1e-12, 1e-10), (0.5_f64, 1e-10, 1e-9)]
    {
        let model = model64(2, lambda, 1000.0);
        let forward = run_a_pairs(&model, &forward_pairs);
        let permuted_state = run_a_pairs(&model, &permuted_pairs);
        let observations: Vec<_> = forward_pairs.iter().map(|(x, y)| sample64(x, *y)).collect();
        let batched = model
            .process_batch_reference(model.initial_state(), &observations, 30)
            .unwrap();
        assert_state64_bitwise_eq(&batched, &forward);
        let permuted_observations: Vec<_> = permuted_pairs
            .iter()
            .map(|(x, y)| sample64(x, *y))
            .collect();
        let permuted_batched = model
            .process_batch_reference(model.initial_state(), &permuted_observations, 30)
            .unwrap();
        assert_state64_bitwise_eq(&permuted_batched, &permuted_state);

        a_objective_for_pairs(
            &model,
            &forward,
            &forward_pairs,
            order_weight_tolerance,
            order_covariance_tolerance,
        );
        a_objective_for_pairs(
            &model,
            &permuted_state,
            &permuted_pairs,
            order_weight_tolerance,
            order_covariance_tolerance,
        );
        let (d_w, s_w, d_p, s_p) = a_order_diagnostics(&forward, &permuted_state);
        if lambda == 1.0 {
            assert!(d_w <= 1e-10 * s_w, "A λ=1 D_w={d_w}, S_w={s_w}");
            assert!(d_p <= 1e-9 * s_p, "A λ=1 D_P={d_p}, S_P={s_p}");
        } else {
            assert!(d_w > 1e-3, "A λ=0.5 D_w={d_w}");
        }

        let serving_forward = model.to_serving_f32(&forward).unwrap();
        let serving_permuted = model.to_serving_f32(&permuted_state).unwrap();
        let expected_forward: Vec<f32> = forward
            .weights()
            .as_slice()
            .iter()
            .map(|&value| value as f32)
            .collect();
        let expected_permuted: Vec<f32> = permuted_state
            .weights()
            .as_slice()
            .iter()
            .map(|&value| value as f32)
            .collect();
        for (actual, expected) in serving_forward
            .weights()
            .as_slice()
            .iter()
            .zip(&expected_forward)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        for (actual, expected) in serving_permuted
            .weights()
            .as_slice()
            .iter()
            .zip(&expected_permuted)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let serving_features: Vec<Vector> = forward_pairs
            .iter()
            .map(|(x, _)| {
                let narrowed: Vec<f32> = x
                    .iter()
                    .map(|&value| {
                        assert!(value.is_finite());
                        let narrowed = value as f32;
                        assert!(narrowed.is_finite());
                        narrowed
                    })
                    .collect();
                Vector::from_slice(&narrowed)
            })
            .collect();
        let mut q_forward = Vec::with_capacity(serving_features.len());
        let mut q_permuted = Vec::with_capacity(serving_features.len());
        for features in &serving_features {
            let expected: f32 = serving_forward
                .weights()
                .as_slice()
                .iter()
                .zip(features.as_slice())
                .map(|(w, x)| w * x)
                .sum();
            let actual = serving_forward
                .predict_checked(features)
                .expect("forward serving prediction must be finite");
            assert_eq!(actual.to_bits(), expected.to_bits());
            let other = serving_permuted
                .predict_checked(features)
                .expect("permuted serving prediction must be finite");
            let expected_other: f32 = serving_permuted
                .weights()
                .as_slice()
                .iter()
                .zip(features.as_slice())
                .map(|(w, x)| w * x)
                .sum();
            assert_eq!(other.to_bits(), expected_other.to_bits());
            q_forward.push(f64::from(actual));
            q_permuted.push(f64::from(other));
        }
        let d_q = max_abs_difference(&q_forward, &q_permuted);
        let s_q = 1.0 + infinity_norm(&q_forward);
        if lambda == 1.0 {
            assert!(d_q <= 1e-6 * s_q, "B λ=1 D_q={d_q}, S_q={s_q}");
        } else {
            assert!(d_q > 1e-4, "B λ=0.5 D_q={d_q}");
        }
    }

    let forward_pairs = permutation_pairs32();
    let permuted_pairs = permuted(&forward_pairs);
    for lambda in [1.0_f32, 0.5] {
        let model = Rls::new(2, lambda, 1000.0).unwrap();
        let forward = run_c_pairs(&model, &forward_pairs);
        let permuted_state = run_c_pairs(&model, &permuted_pairs);
        let observations: Vec<_> = forward_pairs.iter().map(|(x, y)| sample(x, *y)).collect();
        let batched = model
            .process_batch_reference(model.initial_state(), &observations, 30)
            .unwrap();
        assert_state_bitwise_eq(&batched, &forward);
        let permuted_observations: Vec<_> =
            permuted_pairs.iter().map(|(x, y)| sample(x, *y)).collect();
        let permuted_batched = model
            .process_batch_reference(model.initial_state(), &permuted_observations, 30)
            .unwrap();
        assert_state_bitwise_eq(&permuted_batched, &permuted_state);
        c_objective_for_pairs(&model, &forward, &forward_pairs);
        c_objective_for_pairs(&model, &permuted_state, &permuted_pairs);
        let (d_w, s_w, d_p, s_p) = c_order_diagnostics(&forward, &permuted_state);
        if lambda == 1.0 {
            assert!(d_w <= 1e-4 * s_w, "C λ=1 D_w={d_w}, S_w={s_w}");
            assert!(d_p <= 1e-3 * s_p, "C λ=1 D_P={d_p}, S_P={s_p}");
        } else {
            assert!(d_w > 1e-3, "C λ=0.5 D_w={d_w}");
        }
    }
}

#[test]
fn native_f64_discriminator_is_not_a_quantized_f32_trajectory() {
    let model = model64(1, 1.0, 1.0);
    let first = model
        .update(&model.initial_state(), &sample64(&[1.0], 0.0))
        .unwrap();
    let second_input = 1.0 + 2f64.powi(-24);
    let native = model
        .update(&first, &sample64(&[second_input], 2f64.powi(24)))
        .unwrap();
    let reference = ref_step64(
        &ref_step64(&ref_initial64(1, 1.0), &[1.0], 0.0, 1.0),
        &[second_input],
        2f64.powi(24),
        1.0,
    );
    assert_state64_bitwise_eq(&native, &to_state64(&reference));
    let expected = 5_592_405.444_444_444_f64;
    let actual = native.weights().get(0).unwrap();
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "native f64 weight {actual} is not the expected {expected}"
    );

    let quantized_model = Rls::new(1, 1.0, 1.0).unwrap();
    let first = quantized_model
        .update(&quantized_model.initial_state(), &sample(&[1.0], 0.0))
        .unwrap();
    let quantized = quantized_model
        .update(&first, &sample(&[1.0], 2f32.powi(24)))
        .unwrap();
    assert_eq!(
        quantized.w.get(0).unwrap().to_bits(),
        5_592_405.5_f32.to_bits()
    );
    assert!((actual - f64::from(quantized.w.get(0).unwrap())).abs() > 0.05);

    let serving = model.to_serving_f32(&native).unwrap();
    assert_eq!(
        serving.weights().get(0).unwrap().to_bits(),
        quantized.w.get(0).unwrap().to_bits()
    );
}
