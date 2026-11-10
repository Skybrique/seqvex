//! Production-facing checks for the D5 shared execution contract.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::error::FailureCause;
use seqvex::foundation::state::{
    StateModel, process_batch, process_stream, process_stream_continuing,
};

#[derive(Debug, PartialEq, Eq)]
struct State(i64);

#[derive(Debug, PartialEq, Eq)]
enum ModelError {
    Fail,
}

#[derive(Debug)]
struct ObservationRow {
    value: i64,
    fail: bool,
    panic: bool,
    drops: Rc<Cell<usize>>,
}

impl Drop for ObservationRow {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

struct TestModel {
    calls: Rc<Cell<usize>>,
}

impl StateModel for TestModel {
    type State = State;
    type Observation = ObservationRow;
    type Error = ModelError;

    fn update(&self, state: &State, observation: &ObservationRow) -> Result<State, ModelError> {
        self.calls.set(self.calls.get() + 1);
        if observation.panic {
            panic!("model panic");
        }
        if observation.fail {
            return Err(ModelError::Fail);
        }
        Ok(State(state.0 + observation.value))
    }
}

fn row(value: i64, fail: bool, panic: bool, drops: &Rc<Cell<usize>>) -> ObservationRow {
    ObservationRow {
        value,
        fail,
        panic,
        drops: Rc::clone(drops),
    }
}

fn test_model() -> TestModel {
    TestModel {
        calls: Rc::new(Cell::new(0)),
    }
}

#[test]
fn strict_failure_retains_prefix_and_unattempted_suffix() {
    let drops = Rc::new(Cell::new(0));
    let model = test_model();
    let mut iter = vec![
        row(1, false, false, &drops),
        row(1, false, false, &drops),
        row(0, true, false, &drops),
        row(10, false, false, &drops),
        row(0, false, true, &drops),
        row(100, false, false, &drops),
    ]
    .into_iter();

    let failure = process_stream(&model, State(0), &mut iter).unwrap_err();
    assert_eq!(failure.state, State(2));
    assert_eq!(failure.progress.attempted, 3);
    assert_eq!(failure.progress.committed, 2);
    assert_eq!(failure.progress.failed, 1);
    assert_eq!(failure.progress.first_failed_at, Some(2));
    assert_eq!(failure.progress.last_failed_at, Some(2));
    assert_eq!(failure.cause, FailureCause::Model(ModelError::Fail));
    assert_eq!(model.calls.get(), 3);
    assert_eq!(drops.get(), 3);
    assert_eq!(iter.next().map(|r| r.value), Some(10));

    let retry = process_stream(&model, failure.state, [row(0, true, false, &drops)]).unwrap_err();
    assert_eq!(retry.state, State(2));
    assert_eq!(retry.progress.attempted, 1);
    assert_eq!(retry.progress.committed, 0);

    let corrected = process_stream(
        &model,
        State(2),
        [
            row(10, false, false, &drops),
            row(100, false, false, &drops),
        ],
    )
    .unwrap();
    assert_eq!(corrected, State(112));
}

#[test]
fn continuation_retains_later_successes_without_auto_replay() {
    let drops = Rc::new(Cell::new(0));
    let model = test_model();
    let mut positions = Vec::new();
    let summary = process_stream_continuing(
        &model,
        State(0),
        vec![
            row(1, false, false, &drops),
            row(1, false, false, &drops),
            row(0, true, false, &drops),
            row(10, false, false, &drops),
            row(0, true, false, &drops),
            row(100, false, false, &drops),
        ],
        |position, error| {
            assert_eq!(*error, ModelError::Fail);
            positions.push(position);
        },
    )
    .unwrap();

    assert_eq!(summary.state, State(112));
    assert_eq!(summary.progress.attempted, 6);
    assert_eq!(summary.progress.committed, 4);
    assert_eq!(summary.progress.failed, 2);
    assert_eq!(positions, vec![2, 4]);
    assert_eq!(model.calls.get(), 6);

    let forward = process_stream(&model, summary.state, [row(5, false, false, &drops)]).unwrap();
    assert_eq!(forward, State(117));
}

#[test]
fn owned_and_mut_iterator_remainders_are_distinguished() {
    let mut_drops = Rc::new(Cell::new(0));
    let model = test_model();
    let mut iter = vec![
        row(1, false, false, &mut_drops),
        row(1, false, false, &mut_drops),
        row(0, true, false, &mut_drops),
        row(10, false, false, &mut_drops),
    ]
    .into_iter();
    let mut_failure = process_stream(&model, State(0), &mut iter).unwrap_err();
    assert_eq!(mut_failure.state, State(2));
    assert_eq!(mut_drops.get(), 3);
    assert_eq!(iter.next().map(|r| r.value), Some(10));

    let owned_drops = Rc::new(Cell::new(0));
    let owned_model = test_model();
    let owned_failure = process_stream(
        &owned_model,
        State(0),
        vec![
            row(1, false, false, &owned_drops),
            row(1, false, false, &owned_drops),
            row(0, true, false, &owned_drops),
            row(10, false, false, &owned_drops),
        ],
    )
    .unwrap_err();
    assert_eq!(owned_failure.state, State(2));
    assert_eq!(owned_drops.get(), 4);
}

#[test]
fn batch_failure_carries_prefix_progress_and_cause() {
    let drops = Rc::new(Cell::new(0));
    let model = test_model();
    let failure = process_batch(
        &model,
        State(0),
        vec![
            row(1, false, false, &drops),
            row(0, true, false, &drops),
            row(10, false, false, &drops),
        ],
    )
    .unwrap_err();
    assert_eq!(failure.state, State(1));
    assert_eq!(failure.progress.attempted, 2);
    assert_eq!(failure.progress.committed, 1);
    assert_eq!(failure.cause, FailureCause::Model(ModelError::Fail));
}

#[test]
fn executor_error_and_reset_preserve_borrowing_boundaries() {
    let drops = Rc::new(Cell::new(0));
    let model = test_model();
    let mut executor = StreamingExecutor::new(&model, State(0));
    let failure = executor
        .process_stream(vec![
            row(1, false, false, &drops),
            row(1, false, false, &drops),
            row(0, true, false, &drops),
        ])
        .unwrap_err();
    assert_eq!(failure.progress.attempted, 3);
    assert_eq!(failure.progress.committed, 2);
    assert_eq!(failure.cause, FailureCause::Model(ModelError::Fail));
    assert_eq!(executor.state(), &State(2));
    executor.reset(State(100));
    assert_eq!(executor.state(), &State(100));
}

#[test]
fn model_panic_leaves_last_committed_state() {
    let drops = Rc::new(Cell::new(0));
    let model = test_model();
    let mut executor = StreamingExecutor::new(&model, State(0));
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = executor.process_stream(vec![
            row(1, false, false, &drops),
            row(1, false, false, &drops),
            row(0, false, true, &drops),
            row(100, false, false, &drops),
        ]);
    }));
    assert!(panicked.is_err());
    assert_eq!(executor.state(), &State(2));
    assert_eq!(model.calls.get(), 3);
    assert_eq!(drops.get(), 4);
}

#[test]
fn callback_panic_leaves_last_committed_state_and_remainder() {
    let drops = Rc::new(Cell::new(0));
    let model = test_model();
    let calls = Rc::new(Cell::new(0));
    let callback_calls = Rc::clone(&calls);
    let mut executor = StreamingExecutor::new(&model, State(0));
    let mut iter = vec![
        row(1, false, false, &drops),
        row(0, true, false, &drops),
        row(10, false, false, &drops),
    ]
    .into_iter();
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = executor.process_stream_continuing(&mut iter, move |_, _| {
            callback_calls.set(callback_calls.get() + 1);
            panic!("callback panic");
        });
    }));
    assert!(panicked.is_err());
    assert_eq!(executor.state(), &State(1));
    assert_eq!(model.calls.get(), 2);
    assert_eq!(calls.get(), 1);
    assert_eq!(drops.get(), 2);
    assert_eq!(iter.next().map(|r| r.value), Some(10));
}
