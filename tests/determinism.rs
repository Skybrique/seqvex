mod common;

use common::{DigitsModel, SumModel};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_stream;

fn additive_replay() -> i64 {
    let model = SumModel::new(100);
    process_stream(
        &model,
        0_i64,
        [
            Observation::new(7_i64),
            Observation::new(11_i64),
            Observation::new(13_i64),
        ],
    )
    .unwrap()
}

#[test]
fn deterministic_replay_produces_same_result() {
    assert_eq!(additive_replay(), additive_replay());
}

#[test]
fn deterministic_replay_is_repeatable() {
    let model = DigitsModel;
    let observations = [Observation::new(4_i64), Observation::new(2_i64)];
    let first = process_stream(&model, 0_i64, observations.clone()).unwrap();
    let second = process_stream(&model, 0_i64, observations).unwrap();
    assert_eq!(first, second);
    assert_eq!(first, 42);
}
