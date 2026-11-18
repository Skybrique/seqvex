use super::support::{assert_bits_eq, model, observation};
use seqvex::foundation::state::StateModel;

// --- E.13 reference vs production classification ----------------------------

#[test]
fn single_implementation_delegation() {
    // LR has one implementation (`predict`); `StateModel::update` and
    // `predict_batch` delegate to it. LR has no dual reference/production path.
    let model = model(&[1.5, -2.0, 0.25], 0.125);
    let observation = observation(&[4.0, 5.0, -6.0]);
    let reference = model.predict(observation.value()).unwrap();
    for state in [0.0_f32, 7.5, -3.0] {
        assert_bits_eq(model.update(&state, &observation).unwrap(), reference);
    }
    let batch = model
        .predict_batch(std::slice::from_ref(&observation))
        .unwrap();
    assert_bits_eq(batch[0], reference);
}
