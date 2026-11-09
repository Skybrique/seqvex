mod common;

use common::SumModel;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{process_batch, process_one, process_stream};

#[test]
fn streaming_processing_preserves_state() {
    let model = SumModel::new(100);
    let observations = [
        Observation::new(1_i64),
        Observation::new(2_i64),
        Observation::new(3_i64),
    ];
    let state = process_stream(&model, 0_i64, observations).unwrap();
    assert_eq!(state, 6);
}

#[test]
fn sequential_observations_update_state_in_order() {
    let model = SumModel::new(100);
    let observations = [
        Observation::new(10_i64),
        Observation::new(20_i64),
        Observation::new(30_i64),
    ];
    let state = process_stream(&model, 0_i64, observations).unwrap();
    assert_eq!(state, 60);
}

#[test]
fn single_observation_is_smallest_execution_unit() {
    let model = SumModel::new(100);
    let first = process_one(&model, &0_i64, &Observation::new(1_i64)).unwrap();
    let second = process_one(&model, &first, &Observation::new(2_i64)).unwrap();
    assert_eq!(second, 3);
}

#[test]
fn single_observation_updates_current_state() {
    let model = SumModel::new(100);
    let next = process_one(&model, &5_i64, &Observation::new(7_i64)).unwrap();
    assert_eq!(next, 12);
}

#[test]
fn micro_batch_preserves_defined_sequence_semantics() {
    let model = SumModel::new(100);
    let observations = vec![
        Observation::new(1_i64),
        Observation::new(2_i64),
        Observation::new(3_i64),
    ];

    let streamed = process_stream(&model, 0_i64, observations.clone()).unwrap();
    let batched = process_batch(&model, 0_i64, observations).unwrap();

    assert_eq!(streamed, batched);
}
