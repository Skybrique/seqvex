use super::support::{
    assert_bits_eq, assert_params_unchanged, model, observation, snapshot_params,
};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;

// --- E.12 failure behavior --------------------------------------------------

#[test]
fn failure_does_not_corrupt_subsequent_prediction() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let reference = model.clone();
    let valid = [1.0_f32, -2.0, 0.5];

    assert!(model.predict(&Vector::from_slice(&[1.0])).is_err());
    assert!(
        model
            .predict(&Vector::from_slice(&[f32::NAN, 0.0, 0.0]))
            .is_err()
    );
    assert_bits_eq(
        model.predict(&Vector::from_slice(&valid)).unwrap(),
        reference.predict(&Vector::from_slice(&valid)).unwrap(),
    );
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn failed_streaming_prediction_then_valid_continues() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    executor.process_one(&observation(&[3.0])).unwrap();
    assert!(executor.process_one(&observation(&[1.0, 2.0])).is_err());
    assert_bits_eq(*executor.process_one(&observation(&[4.0])).unwrap(), 9.0);
    assert_bits_eq(*executor.state(), 9.0);
}

#[test]
fn failed_batch_then_valid_batch_succeeds() {
    let model = model(&[1.0, 1.0], -0.5);
    let snapshot = snapshot_params(&model);
    assert!(model.predict_batch(&[observation(&[1.0])]).is_err());
    let valid = model.predict_batch(&[observation(&[1.0, 1.0])]).unwrap();
    assert_bits_eq(
        valid[0],
        model.predict(&Vector::from_slice(&[1.0, 1.0])).unwrap(),
    );
    assert_params_unchanged(&model, &snapshot);
}
