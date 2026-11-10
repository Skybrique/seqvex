//! Streaming execution over an ordered sequence of observations.
//!
//! This layer owns *when and how a model is stepped*, not how it computes. It
//! composes the foundation execution fold and adds only what a model does not
//! own: retaining the committed state across calls and an explicit reset
//! boundary.
//!
//! The model is borrowed immutably: execution reads immutable parameters and
//! carries the per-stream committed state itself. The executor owns that state
//! directly; strict failures leave the committed prefix in place without a state
//! copy in the error.

use crate::foundation::error::{ExecutorStreamFailure, StreamProgress};
use crate::foundation::state::atomicity::{RunMode, run_stream_with_progress};
use crate::foundation::state::{StateModel, process_one};

/// Drives one stateful model, holding its committed state between calls.
///
/// One executor owns exactly one independent execution State and borrows its
/// model immutably, so several executors may share one model while keeping
/// independent States.
pub struct StreamingExecutor<'m, M: StateModel> {
    pub(crate) model: &'m M,
    pub(crate) state: M::State,
}

impl<'m, M: StateModel> StreamingExecutor<'m, M> {
    /// Creates an executor over `model`, starting from `initial`.
    pub fn new(model: &'m M, initial: M::State) -> Self {
        Self {
            model,
            state: initial,
        }
    }

    /// The current committed state.
    pub fn state(&self) -> &M::State {
        &self.state
    }

    /// Processes one observation, committing the next state on success.
    ///
    /// On failure the previous committed state is left intact.
    pub fn process_one(&mut self, observation: &M::Observation) -> Result<&M::State, M::Error> {
        let next = process_one(self.model, &self.state, observation)?;
        self.state = next;
        Ok(&self.state)
    }

    /// Strict ordered stream processing with explicit initial progress.
    ///
    /// This is the production mapping used by
    /// [`StreamingExecutor::process_stream`]; the public method always starts at
    /// zero.
    fn process_stream_with_progress<I>(
        &mut self,
        observations: I,
        initial: StreamProgress,
    ) -> Result<&M::State, ExecutorStreamFailure<M::Error>>
    where
        I: IntoIterator<Item = M::Observation>,
    {
        match run_stream_with_progress(
            self.model,
            &mut self.state,
            observations,
            RunMode::Strict,
            initial,
            |_, _| {},
        ) {
            Ok(_) => Ok(&self.state),
            Err(failure) => Err(ExecutorStreamFailure {
                progress: failure.progress,
                cause: failure.cause,
            }),
        }
    }

    /// Strict ordered stream processing.
    ///
    /// Stops at the first model failure or counter overflow. A strict model
    /// failure leaves the successful prefix in the executor; the failed
    /// observation was attempted but not committed; the suffix was unattempted.
    /// The error carries progress and cause only.
    pub fn process_stream<I>(
        &mut self,
        observations: I,
    ) -> Result<&M::State, ExecutorStreamFailure<M::Error>>
    where
        I: IntoIterator<Item = M::Observation>,
    {
        self.process_stream_with_progress(observations, StreamProgress::new(0, 0, None, None))
    }

    /// Explicit continuation stream processing with explicit initial progress.
    ///
    /// This is the production mapping used by
    /// [`StreamingExecutor::process_stream_continuing`]; the public method always
    /// starts at zero.
    fn process_stream_continuing_with_progress<I, F>(
        &mut self,
        observations: I,
        initial: StreamProgress,
        on_failure: F,
    ) -> Result<StreamProgress, ExecutorStreamFailure<M::Error>>
    where
        I: IntoIterator<Item = M::Observation>,
        F: FnMut(u64, &M::Error),
    {
        run_stream_with_progress(
            self.model,
            &mut self.state,
            observations,
            RunMode::Continuing,
            initial,
            on_failure,
        )
        .map_err(|failure| ExecutorStreamFailure {
            progress: failure.progress,
            cause: failure.cause,
        })
    }

    /// Explicit continuation stream processing.
    ///
    /// Model failures are reported to `on_failure` and skipped; later successful
    /// observations are committed. The final state remains in the executor.
    /// Counter overflow returns progress/cause and leaves the final successful
    /// state in place.
    pub fn process_stream_continuing<I, F>(
        &mut self,
        observations: I,
        on_failure: F,
    ) -> Result<StreamProgress, ExecutorStreamFailure<M::Error>>
    where
        I: IntoIterator<Item = M::Observation>,
        F: FnMut(u64, &M::Error),
    {
        self.process_stream_continuing_with_progress(
            observations,
            StreamProgress::new(0, 0, None, None),
            on_failure,
        )
    }

    /// Resets to an explicit initial state, so the next observation starts a
    /// new sequence rather than continuing the previous one.
    pub fn reset(&mut self, initial: M::State) {
        self.state = initial;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::error::FailureCause;
    use std::cell::Cell;
    use std::rc::Rc;

    struct CountingModel {
        calls: Rc<Cell<usize>>,
    }

    impl StateModel for CountingModel {
        type State = i64;
        type Observation = i64;
        type Error = ();

        fn update(&self, state: &i64, observation: &i64) -> Result<i64, ()> {
            self.calls.set(self.calls.get() + 1);
            Ok(state + observation)
        }
    }

    #[test]
    fn executor_strict_process_stream_maps_counter_overflow_without_pulling() {
        let calls = Rc::new(Cell::new(0));
        let model = CountingModel {
            calls: Rc::clone(&calls),
        };
        let mut executor = StreamingExecutor::new(&model, 0_i64);
        let mut iter = vec![1_i64, 2_i64].into_iter();
        let initial = StreamProgress::new(u64::MAX, u64::MAX, None, None);

        let failure = executor
            .process_stream_with_progress(&mut iter, initial.clone())
            .unwrap_err();

        assert_eq!(failure.cause, FailureCause::CounterOverflow);
        assert_eq!(failure.progress, initial);
        assert_eq!(executor.state(), &0);
        assert_eq!(calls.get(), 0);
        assert_eq!(iter.next(), Some(1));
    }

    #[test]
    fn executor_continuing_process_stream_maps_counter_overflow_with_prior_failures() {
        let calls = Rc::new(Cell::new(0));
        let model = CountingModel {
            calls: Rc::clone(&calls),
        };
        let mut executor = StreamingExecutor::new(&model, 0_i64);
        let mut iter = vec![1_i64, 2_i64].into_iter();
        let initial = StreamProgress::new(u64::MAX, u64::MAX - 1, Some(0), Some(0));
        let mut callbacks = 0;

        let failure = executor
            .process_stream_continuing_with_progress(&mut iter, initial.clone(), |_, _| {
                callbacks += 1;
            })
            .unwrap_err();

        assert_eq!(failure.cause, FailureCause::CounterOverflow);
        assert_eq!(failure.progress, initial);
        assert_eq!(failure.progress.failed, 1);
        assert_eq!(failure.progress.first_failed_at, Some(0));
        assert_eq!(failure.progress.last_failed_at, Some(0));
        assert_eq!(executor.state(), &0);
        assert_eq!(calls.get(), 0);
        assert_eq!(callbacks, 0);
        assert_eq!(iter.next(), Some(1));
    }
}
