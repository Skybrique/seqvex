mod common;

use common::{FixtureError, FloatModel, SumModel};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{process_one, process_stream_continuing};

#[test]
fn failed_update_preserves_previous_valid_state() {
    let model = SumModel::new(5);
    let state = 3_i64;
    let result = process_one(&model, &state, &Observation::new(3_i64));
    assert_eq!(result, Err(FixtureError::InvariantViolation));
    assert_eq!(state, 3);
}

#[test]
fn failed_update_does_not_commit_partial_state() {
    let model = SumModel::new(5);
    let mut failures = Vec::new();
    let summary = process_stream_continuing(
        &model,
        0_i64,
        [
            Observation::new(2_i64),
            Observation::new(2_i64),
            Observation::new(2_i64),
            Observation::new(1_i64),
        ],
        |position, error| failures.push((position, *error)),
    )
    .unwrap();
    assert_eq!(summary.state, 5);
    assert_eq!(failures, vec![(2, FixtureError::InvariantViolation)]);
}

#[test]
fn invalid_observation_does_not_mutate_state() {
    let model = SumModel::new(5);
    let mut failures = Vec::new();
    let summary = process_stream_continuing(
        &model,
        2_i64,
        [Observation::new(-1_i64)],
        |position, error| {
            failures.push((position, *error));
        },
    )
    .unwrap();
    assert_eq!(summary.state, 2);
    assert_eq!(failures, vec![(0, FixtureError::InvalidInput)]);
}

#[test]
fn numerical_failure_preserves_previous_valid_state() {
    let mut failures = Vec::new();
    let summary = process_stream_continuing(
        &FloatModel,
        1.0_f64,
        [Observation::new(2.0_f64), Observation::new(f64::NAN)],
        |position, error| failures.push((position, *error)),
    )
    .unwrap();
    assert_eq!(summary.state, 3.0);
    assert_eq!(failures, vec![(1, FixtureError::NumericalFailure)]);
}

#[test]
fn invalid_candidate_state_is_not_committed() {
    let model = SumModel::new(10);
    let mut failures = Vec::new();
    let summary = process_stream_continuing(
        &model,
        0_i64,
        [
            Observation::new(4_i64),
            Observation::new(9_i64),
            Observation::new(5_i64),
        ],
        |position, error| failures.push((position, *error)),
    )
    .unwrap();
    assert_eq!(summary.state, 9);
    assert_eq!(failures, vec![(1, FixtureError::InvariantViolation)]);
}

#[test]
fn recoverable_failure_preserves_sequential_continuity() {
    let model = SumModel::new(5);
    let mut failures = Vec::new();
    let summary = process_stream_continuing(
        &model,
        0_i64,
        [
            Observation::new(1_i64),
            Observation::new(2_i64),
            Observation::new(3_i64),
            Observation::new(2_i64),
        ],
        |position, error| failures.push((position, *error)),
    )
    .unwrap();
    // 1 -> 1, 2 -> 3, 3 -> 6 rejected, 2 -> 5 continuing from the valid state 3.
    assert_eq!(summary.state, 5);
    assert_eq!(failures, vec![(2, FixtureError::InvariantViolation)]);
}
