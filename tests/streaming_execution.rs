//! Integration: `observation -> streaming execution -> GRU -> output`.
//!
//! The execution layer is tested first against a trivial foundation fixture
//! (`SumModel`) to show it is model-agnostic, then against the GRU to show the
//! full path preserves streaming state and failure atomicity. The generic
//! reference executor borrows the model immutably; the optimized GRU path is
//! driven by the algorithm-local `GruExecutor`.

mod common;

use common::SumModel;
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::recurrent::gru::{Gru, GruError, GruExecutor, GruParameters};

fn observation(values: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(values))
}

fn params() -> GruParameters {
    GruParameters::deterministic(2, 2)
}

fn assert_bitwise_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
    }
}

#[test]
fn executor_drives_a_non_gru_model() {
    let model = SumModel::new(100);
    let mut executor = StreamingExecutor::new(&model, 0_i64);
    executor.process_one(&Observation::new(3)).unwrap();
    executor.process_one(&Observation::new(4)).unwrap();
    assert_eq!(*executor.state(), 7);
}

#[test]
fn streaming_execution_matches_direct_stepping() {
    let model = Gru::new(2, 2, params()).unwrap();
    let reference = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    let mut reference_state = Vector::zeros(2);

    for input in [[0.1, 0.2], [0.3, -0.4], [-0.5, 0.6]] {
        let observation = observation(&input);
        executor.process_one(&observation).unwrap();
        reference_state = reference.update(&reference_state, &observation).unwrap();
        assert_eq!(executor.state(), &reference_state);
    }
}

#[test]
fn failed_update_preserves_state_and_the_stream_continues() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    executor.process_one(&observation(&[0.1, 0.2])).unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor
            .process_one(&observation(&[f32::NAN, 0.0]))
            .unwrap_err(),
        GruError::NonFiniteInput
    );
    assert_eq!(executor.state(), &committed);

    // The next valid observation continues from the last committed state.
    let reference = Gru::new(2, 2, params()).unwrap();
    let mut reference_state = Vector::zeros(2);
    reference_state = reference
        .update(&reference_state, &observation(&[0.1, 0.2]))
        .unwrap();
    executor.process_one(&observation(&[0.4, -0.1])).unwrap();
    reference_state = reference
        .update(&reference_state, &observation(&[0.4, -0.1]))
        .unwrap();
    assert_eq!(executor.state(), &reference_state);
}

#[test]
fn process_stream_reports_failures_without_stopping() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    let mut failures = 0;

    let final_state = executor
        .process_stream(
            [
                observation(&[0.1, 0.2]),
                observation(&[f32::NAN, 0.0]),
                observation(&[0.3, -0.4]),
            ],
            |error| {
                assert_eq!(*error, GruError::NonFiniteInput);
                failures += 1;
            },
        )
        .clone();

    assert_eq!(failures, 1);
    let reference = Gru::new(2, 2, params()).unwrap();
    let mut reference_state = Vector::zeros(2);
    reference_state = reference
        .update(&reference_state, &observation(&[0.1, 0.2]))
        .unwrap();
    reference_state = reference
        .update(&reference_state, &observation(&[0.3, -0.4]))
        .unwrap();
    assert_eq!(&final_state, &reference_state);
}

#[test]
fn reset_starts_a_new_sequence_at_the_execution_layer() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    executor.process_one(&observation(&[0.1, 0.2])).unwrap();
    executor.reset(Vector::zeros(2));
    executor.process_one(&observation(&[0.3, -0.4])).unwrap();

    let reference = Gru::new(2, 2, params()).unwrap();
    let fresh = reference
        .update(&Vector::zeros(2), &observation(&[0.3, -0.4]))
        .unwrap();
    assert_eq!(executor.state(), &fresh);
}

#[test]
fn optimized_streaming_matches_reference_streaming_bit_for_bit() {
    let reference_model = Gru::new(2, 2, params()).unwrap();
    let production_model = Gru::new(2, 2, params()).unwrap();
    let mut reference = StreamingExecutor::new(&reference_model, Vector::zeros(2));
    let mut production = GruExecutor::new(&production_model, Vector::zeros(2));

    for input in [
        [0.1, 0.2],
        [0.3, -0.4],
        [-0.5, 0.6],
        [0.0, 0.9],
        [0.7, -0.2],
    ] {
        let observation = observation(&input);
        reference.process_one(&observation).unwrap();
        production.process_one_optimized(&observation).unwrap();
        assert_bitwise_eq(production.state().as_slice(), reference.state().as_slice());
    }

    // Reset both and confirm the optimized path continues identically.
    reference.reset(Vector::zeros(2));
    production.reset(Vector::zeros(2));
    for input in [[0.2, -0.7], [-0.4, 0.5]] {
        let observation = observation(&input);
        reference.process_one(&observation).unwrap();
        production.process_one_optimized(&observation).unwrap();
        assert_bitwise_eq(production.state().as_slice(), reference.state().as_slice());
    }
}

#[test]
fn optimized_streaming_preserves_state_after_failure() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = GruExecutor::new(&model, Vector::zeros(2));
    executor
        .process_one_optimized(&observation(&[0.1, 0.2]))
        .unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor
            .process_one_optimized(&observation(&[f32::NAN, 0.0]))
            .unwrap_err(),
        GruError::NonFiniteInput
    );
    assert_eq!(executor.state(), &committed);

    // The next valid observation continues from the committed state and matches
    // the reference streaming path, using the same reusable workspace.
    let reference_model = Gru::new(2, 2, params()).unwrap();
    let mut reference = StreamingExecutor::new(&reference_model, Vector::zeros(2));
    reference.process_one(&observation(&[0.1, 0.2])).unwrap();
    reference.process_one(&observation(&[0.4, -0.1])).unwrap();
    executor
        .process_one_optimized(&observation(&[0.4, -0.1]))
        .unwrap();
    assert_bitwise_eq(executor.state().as_slice(), reference.state().as_slice());
}

#[test]
fn two_executors_share_one_model_with_independent_states() {
    let model = SumModel::new(100);
    let mut first = StreamingExecutor::new(&model, 0_i64);
    let mut second = StreamingExecutor::new(&model, 0_i64);

    first.process_one(&Observation::new(3)).unwrap();
    first.process_one(&Observation::new(4)).unwrap();
    second.process_one(&Observation::new(10)).unwrap();

    assert_eq!(*first.state(), 7);
    assert_eq!(*second.state(), 10);
}

#[test]
fn two_gru_executors_share_one_model_and_keep_states_isolated() {
    let model = Gru::new(2, 2, params()).unwrap();
    let reference = Gru::new(2, 2, params()).unwrap();

    let mut first = GruExecutor::new(&model, Vector::zeros(2));
    let mut second = GruExecutor::new(&model, Vector::zeros(2));

    let sequence_a = [[0.1, 0.2], [0.3, -0.4]];
    let sequence_b = [[0.0, 0.9], [0.7, -0.2]];

    let mut state_a = Vector::zeros(2);
    let mut state_b = Vector::zeros(2);

    // Interleave two independent executions over the one immutable model.
    for (obs_a, obs_b) in sequence_a.iter().zip(sequence_b.iter()) {
        first.process_one_optimized(&observation(obs_a)).unwrap();
        second.process_one_optimized(&observation(obs_b)).unwrap();

        state_a = reference.update(&state_a, &observation(obs_a)).unwrap();
        state_b = reference.update(&state_b, &observation(obs_b)).unwrap();

        assert_eq!(first.state(), &state_a);
        assert_eq!(second.state(), &state_b);
    }

    assert_ne!(first.state(), second.state());
}
