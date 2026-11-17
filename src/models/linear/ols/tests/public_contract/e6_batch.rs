use super::support::{
    assert_bits_eq, assert_params_unchanged, deterministic_features, model, observation,
    snapshot_params,
};
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::RegressionError;

// --- E.6 batch / chunk equivalence ------------------------------------------

#[test]
fn single_observation_batch_equals_predict() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let observation = observation(&[1.0, -2.0, 0.5]);
    let batch = model
        .predict_batch(std::slice::from_ref(&observation))
        .unwrap();
    assert_bits_eq(batch[0], model.predict(observation.value()).unwrap());
}

#[test]
fn arbitrary_chunking_preserves_each_prediction_bitwise() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let observations: Vec<Observation<Vector>> = (0..12)
        .map(|index| observation(&deterministic_features(index, 3)))
        .collect();
    let full = model.predict_batch(&observations).unwrap();

    let chunk_plans: [Vec<usize>; 3] = [
        vec![1, 2, 9],
        vec![6, 6],
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    for plan in &chunk_plans {
        let mut chunked = Vec::new();
        let mut start = 0;
        for &size in plan {
            chunked.extend(
                model
                    .predict_batch(&observations[start..start + size])
                    .unwrap(),
            );
            start += size;
        }
        assert_eq!(chunked.len(), full.len());
        for (index, (chunked, whole)) in chunked.iter().zip(&full).enumerate() {
            // Per-observation comparison: the same observation in a different
            // caller-defined chunk must yield the same prediction.
            assert_bits_eq(*chunked, *whole);
            assert_bits_eq(
                *chunked,
                model.predict(observations[index].value()).unwrap(),
            );
        }
    }
}

#[test]
fn batch_handles_repeated_zero_and_mixed_sign() {
    let model = model(&[0.5, -1.25, 2.0], -0.75);
    let observations = [
        observation(&[1.0, -2.0, 0.5]),
        observation(&[1.0, -2.0, 0.5]),
        observation(&[0.0, 0.0, 0.0]),
        observation(&[-1.0, 2.0, -0.5]),
        observation(&[0.0, 0.0, 0.0]),
    ];
    let batch = model.predict_batch(&observations).unwrap();
    for (index, observation) in observations.iter().enumerate() {
        assert_bits_eq(batch[index], model.predict(observation.value()).unwrap());
    }
}

#[test]
fn batch_element_failure_preserves_model_and_recovers() {
    let model = model(&[1.0, 1.0], -0.5);
    let snapshot = snapshot_params(&model);
    let failing = [
        observation(&[1.0, 1.0]),
        observation(&[1.0]),
        observation(&[1.0, 1.0]),
    ];
    assert_eq!(
        model.predict_batch(&failing),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_params_unchanged(&model, &snapshot);

    let valid = [observation(&[1.0, 1.0]), observation(&[2.0, -2.0])];
    let outputs = model.predict_batch(&valid).unwrap();
    for (index, observation) in valid.iter().enumerate() {
        assert_bits_eq(outputs[index], model.predict(observation.value()).unwrap());
    }
}

#[test]
fn larger_batch_equivalence() {
    let model = model(&[0.5, -1.25, 2.0, 0.125], -0.375);
    let observations: Vec<Observation<Vector>> = (0..1_000)
        .map(|index| observation(&deterministic_features(index, 4)))
        .collect();
    let batch = model.predict_batch(&observations).unwrap();
    for (index, observation) in observations.iter().enumerate() {
        assert_bits_eq(batch[index], model.predict(observation.value()).unwrap());
    }
}
