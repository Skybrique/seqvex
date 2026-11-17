use super::support::{
    assert_bits_eq, deterministic_features, inverse_permutation, model, observation, permutation,
};
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;

// --- E.4 ordering / permutation ---------------------------------------------

#[test]
fn permutation_does_not_change_per_observation_prediction() {
    let dimension = 4;
    let count = 64;
    let weights = [0.5_f32, -1.25, 2.0, 0.125];
    let model = model(&weights, -0.375);
    let features: Vec<Vec<f32>> = (0..count)
        .map(|index| deterministic_features(index, dimension))
        .collect();

    let in_order: Vec<f32> = features
        .iter()
        .map(|feature| model.predict(&Vector::from_slice(feature)).unwrap())
        .collect();

    let order = permutation(count);
    let permuted: Vec<f32> = order
        .iter()
        .map(|&index| {
            model
                .predict(&Vector::from_slice(&features[index]))
                .unwrap()
        })
        .collect();
    let inverse = inverse_permutation(&order);
    for (index, expected) in in_order.iter().enumerate() {
        // Compare each observation's own prediction across evaluation orders;
        // this is not a position-by-position comparison of the output sequence.
        assert_bits_eq(permuted[inverse[index]], *expected);
    }
}

#[test]
fn predict_batch_permutation_aligns_outputs() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let batch: Vec<Observation<Vector>> = (0..8)
        .map(|index| observation(&deterministic_features(index, 3)))
        .collect();
    let forward = model.predict_batch(&batch).unwrap();

    let order = permutation(8);
    let permuted_batch: Vec<Observation<Vector>> =
        order.iter().map(|&index| batch[index].clone()).collect();
    let permuted = model.predict_batch(&permuted_batch).unwrap();
    for (position, &source) in order.iter().enumerate() {
        assert_bits_eq(permuted[position], forward[source]);
    }
}
