//! Native-`f64` SGD learning and checked `f32` serving.
//!
//! [`Sgd64`] is the native-`f64` learner. It owns no learned state; callers carry
//! [`SgdState64`] per stream. [`Sgd64::to_serving_f32`] produces an independently
//! owned, immutable [`SgdServing32`] snapshot for fixed `f32` inference.
//! [`SgdServing32::predict_batch`] provides grouped fixed-`f32` prediction from
//! one pinned snapshot. The existing [`super::Sgd`] remains the strict-`f32`
//! profile; its existing single-step behavior is preserved and additive grouped
//! methods are introduced.

use crate::foundation::numerical::{DimensionMismatch, Vector, Vector64};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;
use crate::models::linear::serving::{Narrowing, NarrowingError, narrow_f64};

/// Failure classes reported by native-`f64` SGD construction, prediction or
/// update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdError64 {
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
    NonFiniteOutput,
}

impl std::fmt::Display for SgdError64 {
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

impl std::error::Error for SgdError64 {}

impl From<DimensionMismatch> for SgdError64 {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// The native-`f64` learned parameters of one SGD stream: `w` (`D`).
#[derive(Debug, Clone, PartialEq)]
pub struct SgdState64 {
    /// Parameter vector `w` (`D`).
    pub w: Vector64,
}

/// One native-`f64` supervised observation `(x, y)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SgdSample64 {
    /// Feature vector `x` (`D`).
    pub features: Vector64,
    /// Target `y`.
    pub target: f64,
}

/// Failure classes reported by bounded ordered native-`f64` SGD learning.
#[derive(Debug, Clone, PartialEq)]
pub enum SgdUpdateBatchError64 {
    /// The supplied group exceeds the caller's maximum.
    TooLarge {
        /// The caller-supplied maximum group size.
        max: usize,
        /// The supplied group length.
        actual: usize,
        /// The supplied state, bitwise unchanged and unvalidated.
        state: SgdState64,
    },
    /// A row failed; the successful prefix is returned.
    Transition {
        /// Zero-based index within the supplied slice.
        failed_index: usize,
        /// Number of committed rows; equal to `failed_index`.
        committed: usize,
        /// The successful prefix after rows `0..failed_index`.
        state: SgdState64,
        /// The inner single-step error.
        error: SgdError64,
    },
}

impl std::fmt::Display for SgdUpdateBatchError64 {
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

impl std::error::Error for SgdUpdateBatchError64 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge { .. } => None,
            Self::Transition { error, .. } => Some(error),
        }
    }
}

/// Failure classes reported by grouped native-`f64` SGD prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdPredictionError64 {
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
        error: SgdError64,
    },
}

impl std::fmt::Display for SgdPredictionError64 {
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

impl std::error::Error for SgdPredictionError64 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge { .. } => None,
            Self::At { error, .. } => Some(error),
        }
    }
}

/// An immutable native-`f64` SGD linear-regression configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd64 {
    dimension: usize,
    learning_rate: f64,
}

impl Sgd64 {
    /// Builds a native-`f64` configuration, rejecting a zero dimension or a
    /// non-finite / non-positive learning rate.
    pub fn new(dimension: usize, learning_rate: f64) -> Result<Self, SgdError64> {
        if dimension == 0 {
            return Err(SgdError64::ZeroDimension);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(SgdError64::InvalidLearningRate);
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
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// The initial state `w^{(0)} = 0`.
    pub fn initial_state(&self) -> SgdState64 {
        SgdState64 {
            w: Vector64::zeros(self.dimension),
        }
    }

    /// Prediction `ŷ = wᵀx` using the committed state.
    ///
    /// Validation order is dimensions (`x`, then `w`), then feature finiteness,
    /// then output finiteness. A non-finite result returns
    /// [`SgdError64::NonFiniteOutput`].
    pub fn predict(&self, state: &SgdState64, features: &Vector64) -> Result<f64, SgdError64> {
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(SgdError64::NonFiniteInput);
        }
        let value = state.w.dot(features)?;
        if !value.is_finite() {
            return Err(SgdError64::NonFiniteOutput);
        }
        Ok(value)
    }

    /// Bounded ordered native-`f64` learning: applies [`Sgd64::update`] once per
    /// observation in increasing order.
    ///
    /// The group length is checked against `max_batch` before any row is
    /// validated or transitioned. `TooLarge` returns the supplied state
    /// unchanged and unvalidated. On a row failure, `Transition` returns the
    /// successful prefix, the zero-based failing index and the inner error;
    /// rows after the failure are not attempted. Empty input succeeds and
    /// returns the supplied state unchanged.
    pub fn update_batch(
        &self,
        mut state: SgdState64,
        observations: &[Observation<SgdSample64>],
        max_batch: usize,
    ) -> Result<SgdState64, SgdUpdateBatchError64> {
        if observations.len() > max_batch {
            return Err(SgdUpdateBatchError64::TooLarge {
                max: max_batch,
                actual: observations.len(),
                state,
            });
        }
        for (index, observation) in observations.iter().enumerate() {
            match self.update(&state, observation) {
                Ok(next) => state = next,
                Err(error) => {
                    return Err(SgdUpdateBatchError64::Transition {
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

    /// Grouped native-`f64` prediction: applies [`Sgd64::predict`] once per
    /// observation in increasing order from the pinned state.
    ///
    /// The group length is checked against `max_predictions` before any row is
    /// validated or predicted. `TooLarge` returns no vector. On a row failure,
    /// `At` returns the zero-based failing index and the inner error; no partial
    /// vector is returned. Empty input returns an empty vector.
    pub fn predict_batch(
        &self,
        state: &SgdState64,
        observations: &[Observation<Vector64>],
        max_predictions: usize,
    ) -> Result<Vec<f64>, SgdPredictionError64> {
        if observations.len() > max_predictions {
            return Err(SgdPredictionError64::TooLarge {
                max: max_predictions,
                actual: observations.len(),
            });
        }
        let mut outputs = Vec::with_capacity(observations.len());
        for (index, observation) in observations.iter().enumerate() {
            match self.predict(state, observation.value()) {
                Ok(value) => outputs.push(value),
                Err(error) => return Err(SgdPredictionError64::At { index, error }),
            }
        }
        Ok(outputs)
    }

    /// Produces an independently owned `f32` serving snapshot from A's state.
    ///
    /// The state dimension is checked before any weight. Weights are converted
    /// in increasing index order; a non-finite source, cast overflow, or a
    /// nonzero source that rounds to zero is rejected. A finite value that
    /// rounds to [`f32::MAX`] is accepted. Failure returns no partial snapshot
    /// and does not mutate the source state.
    pub fn to_serving_f32(
        &self,
        state: &SgdState64,
    ) -> Result<SgdServing32, SgdServingConversionError> {
        if state.w.len() != self.dimension {
            return Err(SgdServingConversionError::StateDimensionMismatch {
                expected: self.dimension,
                actual: state.w.len(),
            });
        }

        let mut weights = Vec::with_capacity(self.dimension);
        for (index, &source) in state.w.as_slice().iter().enumerate() {
            match narrow_f64(source) {
                Err(NarrowingError::SourceNonFinite) => {
                    return Err(SgdServingConversionError::SourceNonFinite { index });
                }
                Err(NarrowingError::Overflow) => {
                    return Err(SgdServingConversionError::Overflow { index });
                }
                Ok(Narrowing::Underflow(_)) => {
                    return Err(SgdServingConversionError::Underflow { index });
                }
                Ok(Narrowing::Value(value)) => weights.push(value),
            }
        }
        Ok(SgdServing32 {
            weights: Vector::from_vec(weights),
        })
    }

    /// Validates the feature and state dimensions, in order `x`, then `w`.
    fn validate(&self, state: &SgdState64, features: &Vector64) -> Result<(), SgdError64> {
        if features.len() != self.dimension {
            return Err(SgdError64::DimensionMismatch {
                expected: self.dimension,
                actual: features.len(),
            });
        }
        if state.w.len() != self.dimension {
            return Err(SgdError64::DimensionMismatch {
                expected: self.dimension,
                actual: state.w.len(),
            });
        }
        Ok(())
    }
}

impl StateModel for Sgd64 {
    type State = SgdState64;
    type Observation = Observation<SgdSample64>;
    type Error = SgdError64;

    /// Computes the candidate `w' = w + η·e·x` for one supervised observation.
    fn update(
        &self,
        state: &SgdState64,
        observation: &Observation<SgdSample64>,
    ) -> Result<SgdState64, SgdError64> {
        let sample = observation.value();
        let features = &sample.features;
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(SgdError64::NonFiniteInput);
        }
        if !sample.target.is_finite() {
            return Err(SgdError64::NonFiniteInput);
        }

        let error = sample.target - state.w.dot(features)?;
        let scale = self.learning_rate * error;
        let next_w = state.w.add(&features.scale(scale))?;

        if !next_w.as_slice().iter().all(|value| value.is_finite()) {
            return Err(SgdError64::NonFiniteCandidate);
        }
        Ok(SgdState64 { w: next_w })
    }
}

/// An immutable `f32` serving snapshot produced from an [`Sgd64`] state.
///
/// It owns only weights and has no update, resume, or back-conversion path.
/// Freshness metadata (`served_through`, `serving_identity` and the learning
/// watermark) and the `FrozenSelected`/`LatestEligible` policy belong to the
/// caller; this type neither stores nor enforces them. See the SGD README's
/// B freshness and temporal modes section for the caller protocol.
///
/// [`Sgd64::to_serving_f32`] returns a new snapshot on success and leaves the
/// source state unchanged. Installing or replacing the active snapshot is a
/// separate caller action.
#[derive(Debug, Clone, PartialEq)]
pub struct SgdServing32 {
    weights: Vector,
}

impl SgdServing32 {
    /// The snapshot dimension.
    pub fn dimension(&self) -> usize {
        self.weights.len()
    }

    /// The serving weights.
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// Prediction `ŷ = wᵀx` using the fixed serving snapshot.
    ///
    /// Validation order is feature dimension, feature finiteness, then output
    /// finiteness. A non-finite result returns [`super::SgdError::NonFiniteOutput`].
    pub fn predict(&self, features: &Vector) -> Result<f32, super::SgdError> {
        if features.len() != self.weights.len() {
            return Err(super::SgdError::DimensionMismatch {
                expected: self.weights.len(),
                actual: features.len(),
            });
        }
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(super::SgdError::NonFiniteInput);
        }
        let value = self.weights.dot(features)?;
        if !value.is_finite() {
            return Err(super::SgdError::NonFiniteOutput);
        }
        Ok(value)
    }

    /// Grouped fixed-`f32` prediction from this pinned serving snapshot.
    ///
    /// The group length is checked against `max_predictions` before any row is
    /// validated or predicted. `TooLarge` returns no vector. On a row failure,
    /// `At` returns the zero-based failing index and the inner error; no partial
    /// vector is returned. Empty input returns an empty vector. The snapshot is
    /// borrowed immutably and is never mutated.
    pub fn predict_batch(
        &self,
        observations: &[Observation<Vector>],
        max_predictions: usize,
    ) -> Result<Vec<f32>, super::SgdPredictionError> {
        if observations.len() > max_predictions {
            return Err(super::SgdPredictionError::TooLarge {
                max: max_predictions,
                actual: observations.len(),
            });
        }
        let mut outputs = Vec::with_capacity(observations.len());
        for (index, observation) in observations.iter().enumerate() {
            match self.predict(observation.value()) {
                Ok(value) => outputs.push(value),
                Err(error) => return Err(super::SgdPredictionError::At { index, error }),
            }
        }
        Ok(outputs)
    }
}

/// Failure classes reported when converting A's state to a serving snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgdServingConversionError {
    /// The source state does not match the learner's configured dimension.
    StateDimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// A source weight is not finite.
    SourceNonFinite {
        /// The failing weight index.
        index: usize,
    },
    /// A finite source weight rounds to a non-finite `f32`.
    Overflow {
        /// The failing weight index.
        index: usize,
    },
    /// A nonzero source weight rounds to zero in `f32`.
    Underflow {
        /// The failing weight index.
        index: usize,
    },
}

impl std::fmt::Display for SgdServingConversionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateDimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "serving state dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::SourceNonFinite { index } => {
                write!(formatter, "serving source weight {index} is not finite")
            }
            Self::Overflow { index } => {
                write!(formatter, "serving weight {index} overflows f32")
            }
            Self::Underflow { index } => {
                write!(
                    formatter,
                    "serving weight {index} underflows to zero in f32"
                )
            }
        }
    }
}

impl std::error::Error for SgdServingConversionError {}
