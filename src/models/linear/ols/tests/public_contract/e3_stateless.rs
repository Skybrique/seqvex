use super::support::{
    assert_bits_eq, assert_params_unchanged, model, observation, snapshot_params,
};
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::state::StateModel;

// --- E.3 statelessness ------------------------------------------------------

#[test]
fn predict_does_not_mutate_parameters() {
    let model = model(&[0.5, -1.25, 3.0], -0.75);
    let snapshot = snapshot_params(&model);
    let probes = [
        vec![1.0_f32, 2.0, 3.0],
        vec![-1.0, 0.0, 0.5],
        vec![f32::NAN, 0.0, 0.0],
        vec![1.0],
    ];
    for features in probes {
        let _ = model.predict(&Vector::from_slice(&features));
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn repeated_identical_prediction_is_bitwise_stable() {
    let model = model(&[0.25, -0.5, 1.5], 0.125);
    let features = [1.0_f32, -2.0, 0.75];
    let first = model.predict(&Vector::from_slice(&features)).unwrap();
    for _ in 0..1_000 {
        assert_bits_eq(
            model.predict(&Vector::from_slice(&features)).unwrap(),
            first,
        );
    }
}

#[test]
fn prediction_is_independent_of_previous_observations() {
    let model = model(&[0.5, 1.25, -2.0], -0.75);
    let target = [1.0_f32, 2.0, 3.0];
    let baseline = model.predict(&Vector::from_slice(&target)).unwrap();

    for interposed in [
        [0.0_f32, 0.0, 0.0],
        [1.0e30, -1.0e30, 1.0e30],
        [-3.0, 4.0, -5.0],
    ] {
        model.predict(&Vector::from_slice(&interposed)).unwrap();
        assert_bits_eq(
            model.predict(&Vector::from_slice(&target)).unwrap(),
            baseline,
        );
    }
    // A failed call in between leaves the result unchanged too.
    assert!(model.predict(&Vector::from_slice(&[1.0])).is_err());
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );
}

#[test]
fn state_model_update_ignores_incoming_state() {
    // Supporting evidence of state independence: the carried state is not an
    // input to the computation. This is NOT the primary causality test; see
    // prefix_and_suffix_do_not_affect_prediction.
    let model = model(&[1.5, -2.0], 0.25);
    let observation = observation(&[4.0, 5.0]);
    let expected = model.predict(observation.value()).unwrap();
    for state in [0.0_f32, 1.0e30, -3.5, f32::NAN] {
        assert_bits_eq(model.update(&state, &observation).unwrap(), expected);
    }
}
