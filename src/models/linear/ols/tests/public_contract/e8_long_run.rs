use super::support::{
    Lcg, assert_bits_eq, assert_matches_f64_oracle, assert_params_unchanged,
    deterministic_features, model, observation, scalar_predict_f32, snapshot_params,
};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;

// --- E.8 long-run determinism -----------------------------------------------

#[test]
fn long_run_repeated_identical_is_stable() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let features = [1.0_f32, -2.0, 0.5];
    let first = model.predict(&Vector::from_slice(&features)).unwrap();
    for step in 0..10_000 {
        let prediction = model.predict(&Vector::from_slice(&features)).unwrap();
        assert!(prediction.is_finite(), "non-finite at step {step}");
        assert_bits_eq(prediction, first);
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn long_run_alternating_and_cyclic_are_stable() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);

    let alternating = [vec![1.0_f32, -2.0, 0.5], vec![-0.5, 0.25, -0.75]];
    let alternating_baseline: Vec<f32> = alternating
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    for step in 0..10_000 {
        let index = step % alternating.len();
        assert_bits_eq(
            model
                .predict(&Vector::from_slice(&alternating[index]))
                .unwrap(),
            alternating_baseline[index],
        );
    }

    let cycle: Vec<Vec<f32>> = (0..7)
        .map(|index| deterministic_features(index, 3))
        .collect();
    let cycle_baseline: Vec<f32> = cycle
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    for step in 0..10_000 {
        let index = step % cycle.len();
        assert_bits_eq(
            model.predict(&Vector::from_slice(&cycle[index])).unwrap(),
            cycle_baseline[index],
        );
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn long_run_pseudorandom_matches_oracle_and_replay() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let mut lcg = Lcg::new(0xfeed_0001);
    let series: Vec<Vec<f32>> = (0..10_000)
        .map(|_| vec![lcg.next_f32(), lcg.next_f32(), lcg.next_f32()])
        .collect();

    let first_pass: Vec<f32> = series
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    let second_pass: Vec<f32> = series
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    for (first, second) in first_pass.iter().zip(&second_pass) {
        assert_bits_eq(*first, *second);
    }
    for (index, features) in series.iter().enumerate() {
        assert!(first_pass[index].is_finite());
        assert_bits_eq(
            first_pass[index],
            scalar_predict_f32(model.weights().as_slice(), model.bias(), features),
        );
        assert_matches_f64_oracle(&model, features);
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn long_run_streaming_state_is_last_prediction_only() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let mut lcg = Lcg::new(0xfeed_0002);
    let series: Vec<Vec<f32>> = (0..10_000)
        .map(|_| vec![lcg.next_f32(), lcg.next_f32(), lcg.next_f32()])
        .collect();

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    let mut last = 0.0_f32;
    for features in &series {
        last = *executor.process_one(&observation(features)).unwrap();
        assert_bits_eq(last, model.predict(&Vector::from_slice(features)).unwrap());
    }
    assert_bits_eq(*executor.state(), last);

    // Reset and replay: the identical trajectory shows no hidden accumulation.
    executor.reset(0.0);
    for features in &series {
        assert_bits_eq(
            *executor.process_one(&observation(features)).unwrap(),
            model.predict(&Vector::from_slice(features)).unwrap(),
        );
    }
    assert_params_unchanged(&model, &snapshot);
}
