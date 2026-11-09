//! Shared execution failure and progress types.
//!
//! `StreamProgress` uses `u64` counters and positions. The canonical fold checks
//! headroom before pulling, so a returned `CounterOverflow` never implies a new
//! observation was pulled, attempted or committed.

/// A progress counter or per-invocation position.
pub type ProgressCount = u64;

/// Progress reported by one execution invocation.
///
/// `failed` is derived as `attempted - committed`; callers must not construct
/// inconsistent combinations. `first_failed_at`/`last_failed_at` are
/// zero-based model-failure positions, not capacity-overflow positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamProgress {
    /// Observations pulled and attempted in this invocation.
    pub attempted: ProgressCount,
    /// Updates committed in this invocation.
    pub committed: ProgressCount,
    /// Model failures observed in this invocation.
    pub failed: ProgressCount,
    /// First model-failure position, if any.
    pub first_failed_at: Option<ProgressCount>,
    /// Last model-failure position, if any.
    pub last_failed_at: Option<ProgressCount>,
}

impl StreamProgress {
    pub(crate) fn new(
        attempted: ProgressCount,
        committed: ProgressCount,
        first_failed_at: Option<ProgressCount>,
        last_failed_at: Option<ProgressCount>,
    ) -> Self {
        Self {
            attempted,
            committed,
            failed: attempted - committed,
            first_failed_at,
            last_failed_at,
        }
    }
}

/// Why an execution invocation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureCause<E> {
    /// A model update returned an error.
    Model(E),
    /// Progress counters could not represent further work.
    CounterOverflow,
}

/// A rejected batch fold.
///
/// `state` is the committed prefix after earlier successful updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFailure<S, E> {
    /// Last committed state after the successful prefix.
    pub state: S,
    /// Progress at the failure boundary.
    pub progress: StreamProgress,
    /// Failure classification.
    pub cause: FailureCause<E>,
}

/// A rejected strict stream.
///
/// For a strict model failure, `state` is the state before the failed
/// observation. For a strict counter overflow, it is the state after the last
/// successful update; no observation was pulled at the overflow boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFailure<S, E> {
    /// Committed state at the return boundary.
    pub state: S,
    /// Progress at the failure boundary.
    pub progress: StreamProgress,
    /// Failure classification.
    pub cause: FailureCause<E>,
}

/// A strict executor failure.
///
/// The committed state remains in the executor; this type never owns a state
/// copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorStreamFailure<E> {
    /// Progress at the failure boundary.
    pub progress: StreamProgress,
    /// Failure classification.
    pub cause: FailureCause<E>,
}
