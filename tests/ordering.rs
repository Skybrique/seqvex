mod common;

use common::DigitsModel;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_stream;

#[test]
fn processing_preserves_observation_order() {
    let observations = [
        Observation::new(1_i64),
        Observation::new(2_i64),
        Observation::new(3_i64),
    ];
    let state = process_stream(&DigitsModel, 0_i64, observations).unwrap();
    assert_eq!(state, 123);
}

#[test]
fn sequence_replay_preserves_order() {
    let forward = [
        Observation::new(1_i64),
        Observation::new(2_i64),
        Observation::new(3_i64),
    ];
    let reversed = [
        Observation::new(3_i64),
        Observation::new(2_i64),
        Observation::new(1_i64),
    ];

    let forward_state = process_stream(&DigitsModel, 0_i64, forward).unwrap();
    let reversed_state = process_stream(&DigitsModel, 0_i64, reversed).unwrap();

    assert_eq!(forward_state, 123);
    assert_eq!(reversed_state, 321);
}
