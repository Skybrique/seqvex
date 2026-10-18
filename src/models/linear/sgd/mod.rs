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

use crate::foundation::numerical::{DimensionMismatch, Vector};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

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
