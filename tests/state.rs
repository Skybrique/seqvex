mod common;

use common::SumModel;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{process_one, process_stream};

#[test]
fn state_persists_across_observations() {
    let model = SumModel::new(100);
    let state = process_stream(
        &model,
        0_i64,
        [
            Observation::new(1_i64),
            Observation::new(2_i64),
            Observation::new(3_i64),
        ],
    )
    .unwrap();
    assert_eq!(state, 6);
}

#[test]
fn next_observation_uses_previous_valid_state() {
    let model = SumModel::new(100);
    let first = process_one(&model, &0_i64, &Observation::new(4_i64)).unwrap();
    let second = process_one(&model, &first, &Observation::new(5_i64)).unwrap();
    assert_eq!(first, 4);
    assert_eq!(second, 9);
}

#[test]
fn state_transition_produces_next_state() {
    let model = SumModel::new(100);
    let state = process_stream(&model, 10_i64, [Observation::new(1_i64)]).unwrap();
    assert_eq!(state, 11);
}
