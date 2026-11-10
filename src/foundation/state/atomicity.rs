//! Canonical ordered execution fold and failure-preserving stream wrappers.
//!
//! One fold implements the D5 progress state machine. Strict wrappers stop at
//! the first model or counter failure; the continuation wrapper reports model
//! failures to a callback and stops only on counter overflow. No wrapper retains
//! per-failure states or automatically replays observations.

use crate::foundation::error::{BatchFailure, FailureCause, StreamFailure, StreamProgress};
use crate::foundation::state::transition::StateModel;

/// Execution mode for the canonical fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunMode {
    Strict,
    Continuing,
}

/// Internal failure returned by the canonical fold.
///
/// The fold borrows the state, so this type deliberately carries no state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RunFailure<E> {
    pub(crate) progress: StreamProgress,
    pub(crate) cause: FailureCause<E>,
}

/// A successful continuation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamSummary<S> {
    /// Final state after every attempted observation.
    pub state: S,
    /// Per-invocation progress.
    pub progress: StreamProgress,
}

/// Runs the canonical fold with explicit initial progress.
///
/// Private wrapper helpers call this with zero progress; the explicit parameter
/// is also the test seam used to exercise counter-overflow mapping without a
/// public initial-counter API.
pub(crate) fn run_stream_with_progress<M, I, F>(
    model: &M,
    state: &mut M::State,
    observations: I,
    mode: RunMode,
    initial: StreamProgress,
    mut on_failure: F,
) -> Result<StreamProgress, RunFailure<M::Error>>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
    F: FnMut(u64, &M::Error),
{
    let StreamProgress {
        mut attempted,
        mut committed,
        mut first_failed_at,
        mut last_failed_at,
        ..
    } = initial;
    let mut iter = observations.into_iter();
    loop {
        // D5 §5.2 step 1: headroom before any pull.
        if attempted == u64::MAX || committed == u64::MAX {
            return Err(RunFailure {
                progress: StreamProgress::new(
                    attempted,
                    committed,
                    first_failed_at,
                    last_failed_at,
                ),
                cause: FailureCause::CounterOverflow,
            });
        }

        // D5 §5.2 step 2: pull.
        let observation = match iter.next() {
            Some(observation) => observation,
            None => {
                return Ok(StreamProgress::new(
                    attempted,
                    committed,
                    first_failed_at,
                    last_failed_at,
                ));
            }
        };

        // D5 §5.2 step 3: attempt count.
        attempted = attempted.checked_add(1).expect("headroom checked");

        // D5 §5.2 steps 4-6: update; no commit on failure; commit + count.
        match model.update(&*state, &observation) {
            Ok(next) => {
                *state = next;
                committed = committed.checked_add(1).expect("headroom checked");
            }
            Err(error) => {
                let position = attempted - 1;
                if first_failed_at.is_none() {
                    first_failed_at = Some(position);
                }
                last_failed_at = Some(position);
                match mode {
                    RunMode::Strict => {
                        return Err(RunFailure {
                            progress: StreamProgress::new(
                                attempted,
                                committed,
                                first_failed_at,
                                last_failed_at,
                            ),
                            cause: FailureCause::Model(error),
                        });
                    }
                    RunMode::Continuing => on_failure(position, &error),
                }
            }
        }
    }
}

/// Ordered strict batch fold with explicit initial progress.
///
/// This is the production mapping used by [`process_batch`]; the public wrapper
/// always starts at zero.
fn process_batch_from<M, I>(
    model: &M,
    mut state: M::State,
    batch: I,
    initial: StreamProgress,
) -> Result<M::State, BatchFailure<M::State, M::Error>>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
{
    match run_stream_with_progress(
        model,
        &mut state,
        batch,
        RunMode::Strict,
        initial,
        |_, _| {},
    ) {
        Ok(_) => Ok(state),
        Err(failure) => Err(BatchFailure {
            state,
            progress: failure.progress,
            cause: failure.cause,
        }),
    }
}

/// Ordered strict batch fold.
///
/// Consumes the supplied state and returns either the final committed state or
/// the prefix state with progress and cause. The batch is not bounded here;
/// callers impose their own bounds.
pub fn process_batch<M, I>(
    model: &M,
    state: M::State,
    batch: I,
) -> Result<M::State, BatchFailure<M::State, M::Error>>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
{
    process_batch_from(model, state, batch, StreamProgress::new(0, 0, None, None))
}

/// Strict ordered stream fold with explicit initial progress.
///
/// This is the production mapping used by [`process_stream`]; the public wrapper
/// always starts at zero.
fn process_stream_from<M, I>(
    model: &M,
    mut state: M::State,
    observations: I,
    initial: StreamProgress,
) -> Result<M::State, StreamFailure<M::State, M::Error>>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
{
    match run_stream_with_progress(
        model,
        &mut state,
        observations,
        RunMode::Strict,
        initial,
        |_, _| {},
    ) {
        Ok(_) => Ok(state),
        Err(failure) => Err(StreamFailure {
            state,
            progress: failure.progress,
            cause: failure.cause,
        }),
    }
}

/// Strict ordered stream fold.
///
/// Stops at the first model failure or counter overflow. On a model failure the
/// returned state is the successful prefix; the failed observation was attempted
/// but not committed; the suffix was unattempted.
pub fn process_stream<M, I>(
    model: &M,
    initial: M::State,
    observations: I,
) -> Result<M::State, StreamFailure<M::State, M::Error>>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
{
    process_stream_from(
        model,
        initial,
        observations,
        StreamProgress::new(0, 0, None, None),
    )
}

type StreamContinuationResult<S, E> = Result<StreamSummary<S>, StreamFailure<S, E>>;

/// Explicit continuation fold with explicit initial progress.
///
/// This is the production mapping used by [`process_stream_continuing`]; the
/// public wrapper always starts at zero.
fn process_stream_continuing_from<M, I, F>(
    model: &M,
    mut state: M::State,
    observations: I,
    initial: StreamProgress,
    on_failure: F,
) -> StreamContinuationResult<M::State, M::Error>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
    F: FnMut(u64, &M::Error),
{
    match run_stream_with_progress(
        model,
        &mut state,
        observations,
        RunMode::Continuing,
        initial,
        on_failure,
    ) {
        Ok(progress) => Ok(StreamSummary { state, progress }),
        Err(failure) => Err(StreamFailure {
            state,
            progress: failure.progress,
            cause: failure.cause,
        }),
    }
}

/// Explicit continuation stream fold.
///
/// Model failures are reported to `on_failure` and skipped; successful later
/// observations are committed and included in the returned final state. There is
/// no retrospective retry from that final state and no per-failure state
/// retention. Counter overflow returns the final successful state with progress.
pub fn process_stream_continuing<M, I, F>(
    model: &M,
    initial: M::State,
    observations: I,
    on_failure: F,
) -> StreamContinuationResult<M::State, M::Error>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
    F: FnMut(u64, &M::Error),
{
    process_stream_continuing_from(
        model,
        initial,
        observations,
        StreamProgress::new(0, 0, None, None),
        on_failure,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    struct Model {
        calls: Rc<Cell<usize>>,
    }

    impl StateModel for Model {
        type State = i64;
        type Observation = i64;
        type Error = ();

        fn update(&self, state: &i64, observation: &i64) -> Result<i64, ()> {
            self.calls.set(self.calls.get() + 1);
            Ok(state + observation)
        }
    }

    #[test]
    fn counter_overflow_is_checked_before_pull_and_can_restart() {
        let model = Model {
            calls: Rc::new(Cell::new(0)),
        };
        let mut iter = vec![1_i64, 2_i64].into_iter();
        let mut state = 0_i64;
        let failure = run_stream_with_progress(
            &model,
            &mut state,
            &mut iter,
            RunMode::Strict,
            StreamProgress::new(u64::MAX - 1, u64::MAX - 1, None, None),
            |_, _| {},
        )
        .unwrap_err();

        assert_eq!(failure.cause, FailureCause::CounterOverflow);
        assert_eq!(failure.progress.attempted, u64::MAX);
        assert_eq!(failure.progress.committed, u64::MAX);
        assert_eq!(failure.progress.failed, 0);
        assert_eq!(failure.progress.first_failed_at, None);
        assert_eq!(failure.progress.last_failed_at, None);
        assert_eq!(state, 1);
        assert_eq!(model.calls.get(), 1);

        let resumed = process_stream(&model, state, &mut iter).unwrap();
        assert_eq!(resumed, 3);
        assert_eq!(iter.next(), None);
        assert_eq!(model.calls.get(), 2);
    }

    #[test]
    fn process_batch_maps_counter_overflow_without_pulling() {
        let model = Model {
            calls: Rc::new(Cell::new(0)),
        };
        let mut iter = vec![1_i64, 2_i64].into_iter();
        let initial = StreamProgress::new(u64::MAX, u64::MAX, None, None);
        let failure = process_batch_from(&model, 0_i64, &mut iter, initial.clone()).unwrap_err();

        assert_eq!(failure.cause, FailureCause::CounterOverflow);
        assert_eq!(failure.state, 0);
        assert_eq!(failure.progress, initial);
        assert_eq!(model.calls.get(), 0);
        assert_eq!(iter.next(), Some(1));
    }

    #[test]
    fn process_stream_maps_counter_overflow_without_pulling() {
        let model = Model {
            calls: Rc::new(Cell::new(0)),
        };
        let mut iter = vec![1_i64, 2_i64].into_iter();
        let initial = StreamProgress::new(u64::MAX, u64::MAX, None, None);
        let failure = process_stream_from(&model, 0_i64, &mut iter, initial.clone()).unwrap_err();

        assert_eq!(failure.cause, FailureCause::CounterOverflow);
        assert_eq!(failure.state, 0);
        assert_eq!(failure.progress, initial);
        assert_eq!(model.calls.get(), 0);
        assert_eq!(iter.next(), Some(1));
    }

    #[test]
    fn process_stream_continuing_maps_counter_overflow_with_prior_failures() {
        let model = Model {
            calls: Rc::new(Cell::new(0)),
        };
        let mut iter = vec![1_i64, 2_i64].into_iter();
        let initial = StreamProgress::new(u64::MAX, u64::MAX - 1, Some(0), Some(0));
        let mut callbacks = 0;
        let failure =
            process_stream_continuing_from(&model, 0_i64, &mut iter, initial.clone(), |_, _| {
                callbacks += 1
            })
            .unwrap_err();

        assert_eq!(failure.cause, FailureCause::CounterOverflow);
        assert_eq!(failure.state, 0);
        assert_eq!(failure.progress, initial);
        assert_eq!(failure.progress.failed, 1);
        assert_eq!(failure.progress.first_failed_at, Some(0));
        assert_eq!(failure.progress.last_failed_at, Some(0));
        assert_eq!(model.calls.get(), 0);
        assert_eq!(callbacks, 0);
        assert_eq!(iter.next(), Some(1));
    }
}
