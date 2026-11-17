use super::support::{
    assert_bits_eq, deterministic_features, model, observation, scalar_predict_f32,
};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;

// --- E.5 causality / prefix / suffix / context ------------------------------

#[test]
fn prefix_and_suffix_do_not_affect_prediction() {
    // Primary causality evidence: predict(x_t) is unchanged by any prefix,
    // suffix, position, or sequence context.
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let target = deterministic_features(5, 3);
    let baseline = model.predict(&Vector::from_slice(&target)).unwrap();

    let prefix: Vec<Vec<f32>> = (0..17)
        .map(|index| deterministic_features(index, 3))
        .collect();
    for features in &prefix {
        model.predict(&Vector::from_slice(features)).unwrap();
    }
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );

    // A different-distribution prefix (large magnitudes) must not matter either.
    for features in &prefix {
        let scaled: Vec<f32> = features.iter().map(|value| value * 1.0e6).collect();
        model.predict(&Vector::from_slice(&scaled)).unwrap();
    }
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );

    // Inside a longer sequence, with an arbitrary suffix after x_t.
    let mut sequence = Vec::new();
    sequence.extend_from_slice(&prefix[..3]);
    sequence.push(target.clone());
    sequence.extend((0..9).map(|index| deterministic_features(index + 100, 3)));
    for features in &sequence {
        model.predict(&Vector::from_slice(features)).unwrap();
    }
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );
    assert_bits_eq(
        scalar_predict_f32(model.weights().as_slice(), model.bias(), &target),
        baseline,
    );
}

#[test]
fn suffix_does_not_affect_committed_stream_state_for_x_t() {
    let prefix = [
        observation(&[0.1, 0.2, 0.3]),
        observation(&[-0.4, 0.5, -0.6]),
    ];
    let target = observation(&[1.0, -2.0, 0.5]);
    let suffix = [
        observation(&[9.0, -8.0, 7.0]),
        observation(&[0.0, 0.0, 0.0]),
    ];
    let reference = model(&[0.5, -1.25, 2.0], -0.375);

    let running_model = reference.clone();
    let mut running = StreamingExecutor::new(&running_model, 0.0);
    for observation in prefix.iter().chain(std::iter::once(&target)) {
        running.process_one(observation).unwrap();
    }
    let committed_for_target = *running.state();
    for observation in &suffix {
        running.process_one(observation).unwrap();
    }

    // The value committed for x_t (snapshotted before any suffix) equals the
    // direct prediction for x_t; the suffix cannot retroactively influence it.
    assert_bits_eq(
        committed_for_target,
        reference.predict(target.value()).unwrap(),
    );
    assert_bits_eq(
        *running.state(),
        reference.predict(suffix.last().unwrap().value()).unwrap(),
    );
}

#[test]
fn sequence_number_context_does_not_change_prediction() {
    use seqvex::foundation::observation::SequenceNumber;

    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let features = [1.0_f32, -2.0, 0.5];
    let plain = model.predict(&Vector::from_slice(&features)).unwrap();
    let with_sequence =
        Observation::new(Vector::from_slice(&features)).with_sequence(SequenceNumber::new(7));
    assert_bits_eq(model.predict(with_sequence.value()).unwrap(), plain);

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    assert_bits_eq(*executor.process_one(&with_sequence).unwrap(), plain);
}
