//! Stochastic-gradient-trained linear regression: ordered single-observation
//! learning of a dense linear model.
//!
//! The model predicts `ŷ = wᵀx` and learns from one supervised observation at a
//! time. With feature dimension `D`, feature vector `x_t`, target `y_t`,
//! parameters `w_t`, and fixed learning rate `η`:
//!
//! ```text
//! ŷ_t = w_{t-1}ᵀ x_t
//! e_t = y_t − ŷ_t
//! w_t = w_{t-1} + η e_t x_t
//! ```
//!
//! The reference dataset objective is `J(w) = Σ(y − wᵀx)² / (2N)`.
//!
//! # Ownership
//!
//! [`Sgd`] holds only immutable configuration (`D`, `η`) and implements
//! [`StateModel`] with a `&self` contract, so one model is shareable across
//! streams. The learned parameters live in [`SgdState`], which is per-execution
//! state. There is no workspace, cache, or model-owned scratch.
//!
//! # Atomicity
//!
//! `update` reads the committed state, computes a candidate `w'` as a local,
//! validates it, and returns it only on success. A failure returns [`SgdError`]
//! and leaves the committed state bitwise unchanged.
//!
//! # Numerics
//!
//! The `f32` substrate ([`Vector`]) is reused. Prediction reduces in
//! feature-index order; the update groups as `s = η·e`, `Δ_j = s·x_j`,
//! `w'_j = w_j + Δ_j`. Validation order is deterministic: dimensions, then
//! feature finiteness, then target finiteness (`update`), then candidate
//! finiteness. [`Sgd::predict`] rejects a non-finite result with
//! [`SgdError::NonFiniteOutput`].
//!
//! The model accepts every finite positive learning rate. Per-sample damping
//! (`η‖x‖² < 1`) is an example-level policy, not a model restriction.
//!
//! # Precision profiles
//!
//! [`Sgd`] remains the strict-`f32` profile. [`Sgd64`] is the native-`f64`
//! learner; [`Sgd64::to_serving_f32`] produces an independently owned
//! [`SgdServing32`] snapshot for fixed `f32` inference. A and C use the single
//! output-validating `predict`; there is no bare/checked pair.
//!
//! # Bounded ordered groups
//!
//! Bounded ordered learning is provided by [`Sgd::update_batch`], which applies
//! the single-step update once per row in increasing order. Grouped prediction
//! is provided by [`Sgd::predict_batch`], which applies the single-step
//! prediction once per row from one pinned state. Group-length bounds are
//! checked before any row validation or transition. Oversized groups return the
//! unchanged supplied state and perform no transition or observation
//! validation. Fixed prediction calculation is separate from caller-owned
//! external publication. Freshness selection, `served_through`/serving-identity
//! bookkeeping, snapshot replacement and external publication remain
//! caller-owned; this module adds no freshness API or automatic replacement.

use crate::foundation::numerical::{DimensionMismatch, Vector};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

mod profiles;

pub use profiles::{
    Sgd64, SgdError64, SgdPredictionError64, SgdSample64, SgdServing32, SgdServingConversionError,
    SgdState64, SgdUpdateBatchError64,
};

/// Failure classes reported by SGD construction, prediction, or update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdError {
    /// A configured dimension is zero.
    ZeroDimension,
    /// A vector has the wrong dimension.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// The learning rate is not finite or not strictly positive.
    InvalidLearningRate,
    /// An observation feature or target is not finite.
    NonFiniteInput,
    /// The computed candidate `w'` is not finite.
    NonFiniteCandidate,
    /// A prediction produced a non-finite result.
    ///
    /// Reported by [`Sgd::predict`]; the supplied state is unchanged, and this
    /// guard neither establishes state finiteness nor diagnoses the cause.
    NonFiniteOutput,
}

impl std::fmt::Display for SgdError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("SGD dimension must be nonzero"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidLearningRate => {
                formatter.write_str("SGD learning rate must be finite and positive")
            }
            Self::NonFiniteInput => {
                formatter.write_str("SGD observation features and target must be finite")
            }
            Self::NonFiniteCandidate => formatter.write_str("SGD candidate state is not finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("SGD prediction produced a non-finite output")
            }
        }
    }
}

impl std::error::Error for SgdError {}

impl From<DimensionMismatch> for SgdError {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// The learned parameters of one SGD stream: `w` (`D`).
#[derive(Debug, Clone, PartialEq)]
pub struct SgdState {
    /// Parameter vector `w` (`D`).
    pub w: Vector,
}

/// One supervised observation `(x, y)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SgdSample {
    /// Feature vector `x` (`D`).
    pub features: Vector,
    /// Target `y`.
    pub target: f32,
}

/// Failure classes reported by bounded ordered SGD learning.
#[derive(Debug, Clone, PartialEq)]
pub enum SgdUpdateBatchError {
    /// The supplied group exceeds the caller's maximum.
    TooLarge {
        /// The caller-supplied maximum group size.
        max: usize,
        /// The supplied group length.
        actual: usize,
        /// The supplied state, bitwise unchanged and unvalidated.
        state: SgdState,
    },
    /// A row failed; the successful prefix is returned.
    Transition {
        /// Zero-based index within the supplied slice.
        failed_index: usize,
        /// Number of committed rows; equal to `failed_index`.
        committed: usize,
        /// The successful prefix after rows `0..failed_index`.
        state: SgdState,
        /// The inner single-step error.
        error: SgdError,
    },
}

impl std::fmt::Display for SgdUpdateBatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { max, actual, .. } => write!(
                formatter,
                "SGD update group is too large: max {max}, got {actual}; no observation was validated or processed and the supplied state was returned unchanged"
            ),
            Self::Transition {
                failed_index,
                committed,
                ..
            } => write!(
                formatter,
                "SGD update failed at index {failed_index} after {committed} committed rows; the failed candidate was not committed and the successful prefix was returned"
            ),
        }
    }
}

impl std::error::Error for SgdUpdateBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge { .. } => None,
            Self::Transition { error, .. } => Some(error),
        }
    }
}

/// Failure classes reported by grouped SGD prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdPredictionError {
    /// The supplied group exceeds the caller's maximum.
    TooLarge {
        /// The caller-supplied maximum prediction count.
        max: usize,
        /// The supplied group length.
        actual: usize,
    },
    /// A row failed; no partial vector is returned.
    At {
        /// Zero-based index within the supplied slice.
        index: usize,
        /// The inner single-step error.
        error: SgdError,
    },
}

impl std::fmt::Display for SgdPredictionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { max, actual } => write!(
                formatter,
                "SGD prediction group is too large: max {max}, got {actual}; no observation was validated or predicted and no output was returned"
            ),
            Self::At { index, .. } => write!(
                formatter,
                "SGD prediction failed at index {index}; no partial prediction vector was returned"
            ),
        }
    }
}

impl std::error::Error for SgdPredictionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge { .. } => None,
            Self::At { error, .. } => Some(error),
        }
    }
}

/// An immutable SGD linear-regression configuration, shareable across streams.
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd {
    dimension: usize,
    learning_rate: f32,
}

impl Sgd {
    /// Builds a configuration for `dimension` features with a fixed learning
    /// rate, rejecting a zero dimension or a non-finite / non-positive rate.
    ///
    /// Every finite positive learning rate is accepted. Per-sample damping is a
    /// caller/example policy, not a model restriction.
    pub fn new(dimension: usize, learning_rate: f32) -> Result<Self, SgdError> {
        if dimension == 0 {
            return Err(SgdError::ZeroDimension);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(SgdError::InvalidLearningRate);
        }
        Ok(Self {
            dimension,
            learning_rate,
        })
    }

    /// The configured feature dimension `D`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The configured fixed learning rate `η`.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// The initial state `w^{(0)} = 0`.
    pub fn initial_state(&self) -> SgdState {
        SgdState {
            w: Vector::zeros(self.dimension),
        }
    }

    /// Prediction `ŷ = wᵀx` using the committed state.
    ///
    /// Validation order is dimensions (`x`, then `w`), then feature finiteness,
    /// then output finiteness: a non-finite result returns
    /// [`SgdError::NonFiniteOutput`].
    pub fn predict(&self, state: &SgdState, features: &Vector) -> Result<f32, SgdError> {
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(SgdError::NonFiniteInput);
        }
        let value = state.w.dot(features)?;
        if !value.is_finite() {
            return Err(SgdError::NonFiniteOutput);
        }
        Ok(value)
    }

    /// Bounded ordered learning: applies [`Sgd::update`] once per observation in
    /// increasing order.
    ///
    /// The group length is checked against `max_batch` before any row is
    /// validated or transitioned. `TooLarge` returns the supplied state
    /// unchanged and unvalidated. On a row failure, `Transition` returns the
    /// successful prefix, the zero-based failing index and the inner error;
    /// rows after the failure are not attempted. Empty input succeeds and
    /// returns the supplied state unchanged.
    pub fn update_batch(
        &self,
        mut state: SgdState,
        observations: &[Observation<SgdSample>],
        max_batch: usize,
    ) -> Result<SgdState, SgdUpdateBatchError> {
        if observations.len() > max_batch {
            return Err(SgdUpdateBatchError::TooLarge {
                max: max_batch,
                actual: observations.len(),
                state,
            });
        }
        for (index, observation) in observations.iter().enumerate() {
            match self.update(&state, observation) {
                Ok(next) => state = next,
                Err(error) => {
                    return Err(SgdUpdateBatchError::Transition {
                        failed_index: index,
                        committed: index,
                        state,
                        error,
                    });
                }
            }
        }
        Ok(state)
    }

    /// Grouped prediction: applies [`Sgd::predict`] once per observation in
    /// increasing order from the pinned state.
    ///
    /// The group length is checked against `max_predictions` before any row is
    /// validated or predicted. `TooLarge` returns no vector. On a row failure,
    /// `At` returns the zero-based failing index and the inner error; no partial
    /// vector is returned. Empty input returns an empty vector.
    pub fn predict_batch(
        &self,
        state: &SgdState,
        observations: &[Observation<Vector>],
        max_predictions: usize,
    ) -> Result<Vec<f32>, SgdPredictionError> {
        if observations.len() > max_predictions {
            return Err(SgdPredictionError::TooLarge {
                max: max_predictions,
                actual: observations.len(),
            });
        }
        let mut outputs = Vec::with_capacity(observations.len());
        for (index, observation) in observations.iter().enumerate() {
            match self.predict(state, observation.value()) {
                Ok(value) => outputs.push(value),
                Err(error) => return Err(SgdPredictionError::At { index, error }),
            }
        }
        Ok(outputs)
    }

    /// Validates the feature and state dimensions, in order `x`, then `w`.
    fn validate(&self, state: &SgdState, features: &Vector) -> Result<(), SgdError> {
        if features.len() != self.dimension {
            return Err(SgdError::DimensionMismatch {
                expected: self.dimension,
                actual: features.len(),
            });
        }
        if state.w.len() != self.dimension {
            return Err(SgdError::DimensionMismatch {
                expected: self.dimension,
                actual: state.w.len(),
            });
        }
        Ok(())
    }
}

impl StateModel for Sgd {
    type State = SgdState;
    type Observation = Observation<SgdSample>;
    type Error = SgdError;

    /// Computes the candidate `w' = w + η·e·x` for one supervised observation.
    ///
    /// The candidate is returned only after it validates, so a failed update
    /// cannot commit partial state.
    fn update(
        &self,
        state: &SgdState,
        observation: &Observation<SgdSample>,
    ) -> Result<SgdState, SgdError> {
        let sample = observation.value();
        let features = &sample.features;
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(SgdError::NonFiniteInput);
        }
        if !sample.target.is_finite() {
            return Err(SgdError::NonFiniteInput);
        }

        // e = y − ŷ ; s = η·e ; w' = w + s·x (grouped as s = fl(η·e),
        // Δ_j = fl(s·x_j), w'_j = fl(w_j + Δ_j)).
        let error = sample.target - state.w.dot(features)?;
        let scale = self.learning_rate * error;
        let next_w = state.w.add(&features.scale(scale))?;

        if !next_w.as_slice().iter().all(|value| value.is_finite()) {
            return Err(SgdError::NonFiniteCandidate);
        }
        Ok(SgdState { w: next_w })
    }
}
