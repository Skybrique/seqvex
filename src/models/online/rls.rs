//! Recursive Least Squares (RLS) with a forgetting factor: ordered online
//! adaptation of `(w, P)`.
//!
//! RLS is the online (supervised) slice. Unlike linear-regression prediction,
//! each observation changes the model's parameters, so updates are strictly
//! ordered and state-dependent.
//!
//! # Equations
//!
//! With feature dimension `D`, feature vector `x_t`, target `y_t`, parameter
//! vector `w_t`, inverse-correlation matrix `P_t`, and forgetting factor `λ`:
//!
//! ```text
//! ŷ_t = w_{t-1}ᵀ x_t
//! e_t = y_t − ŷ_t
//! v_t = P_{t-1} x_t
//! d_t = λ + x_tᵀ v_t
//! k_t = v_t / d_t
//! w_t = w_{t-1} + k_t e_t
//! P_t = (P_{t-1} − v_t v_tᵀ / d_t) / λ
//! ```
//!
//! The covariance step uses the transparent rank-one form. Because `P` is
//! symmetric, `k_t x_tᵀ P_{t-1} = v_t v_tᵀ / d_t`, so the two are algebraically
//! identical and the implementation avoids a separate outer-product term.
//!
//! # Ownership
//!
//! [`Rls`] holds only immutable configuration (`D`, `λ`, `δ`) and implements
//! [`StateModel`] with a `&self` contract, so one model is shareable across
//! streams. The adaptive state `(w, P)` is per-stream execution state, carried
//! by [`StateModel::State`]. There is no workspace, cache, or reusable scratch.
//! A future optimization that double-buffered `P` would raise a model-vs-stream
//! ownership question and requires CRITICAL ARCHITECTURE REVIEW; it is not done
//! here.
//!
//! # Atomicity
//!
//! `update` reads the committed state, computes a candidate `(w', P')` as
//! locals, validates it, and returns it only on success. A failure returns
//! [`RlsError`] and leaves the committed state bitwise unchanged; there is no
//! path that commits one component without the other.
//!
//! # Bounded reference batch
//!
//! [`Rls::process_batch_reference`] is an ordered, bounded fold that calls
//! [`Rls::update`] directly over borrowed observations. It preserves the same
//! ordered transition semantics as the foundation fold without calling it,
//! rejects an oversized batch before any transition, and returns the committed
//! `(w, P)` only on success. The foundation `process_batch` remains the
//! **unbounded** reference fold.
//!
//! # Numerics
//!
//! The `f32` substrate ([`Vector`], [`Matrix`]) is reused: observations are
//! already `Observation<Vector>`-shaped and no `f64` matrix representation
//! exists. Whether `f32` suffices for large `D` over long runs is an empirical
//! question, deliberately not resolved here.
//!
//! Validation order is deterministic: dimensions, then observation finiteness,
//! then the denominator `d = λ + xᵀPx` (finite and `> 0`), then candidate
//! finiteness.

use crate::foundation::numerical::{DimensionMismatch, Matrix, Vector};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

/// Failure classes reported by RLS construction, prediction, or update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlsError {
    /// A configured dimension is zero.
    ZeroDimension,
    /// A vector or matrix has the wrong dimension.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// The forgetting factor is not finite or not in `(0, 1]`.
    InvalidForgettingFactor,
    /// The initial covariance scale is not finite or not positive.
    InvalidInitialCovariance,
    /// An observation feature or target is not finite.
    NonFiniteInput,
    /// The denominator `d = λ + xᵀPx` is not finite and positive.
    InvalidDenominator,
    /// The computed candidate `(w', P')` is not finite.
    NonFiniteCandidate,
}

impl From<DimensionMismatch> for RlsError {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// Failure of a bounded RLS reference-batch call.
///
/// The batch API consumes the incoming adaptive state, so both variants return a
/// recoverable state: the caller never needs to clone merely to recover from a
/// rejected bound or a failed transition.
#[derive(Debug, Clone, PartialEq)]
pub enum RlsBatchError {
    /// The supplied batch is longer than the caller-declared bound.
    ///
    /// Checked before any transition or validation, so no observation is
    /// processed. `state` is the original supplied adaptive state, returned
    /// unchanged and unvalidated, so recovery needs no clone.
    BatchTooLarge {
        /// The caller-declared maximum batch length.
        max: usize,
        /// The supplied batch length.
        actual: usize,
        /// The original supplied adaptive state, returned unchanged.
        state: RlsState,
    },
    /// The ordered fold stopped at a failed observation.
    Transition {
        /// Zero-based index of the failed observation.
        failed_index: usize,
        /// The last committed valid state after the successful prefix (the
        /// supplied state when no observation succeeded).
        state: RlsState,
        /// The failure reported by the reference transition.
        error: RlsError,
    },
}

/// The adaptive state of one RLS stream: `w` (`D`) and `P` (`D × D`).
///
/// `P` is expected symmetric positive-semidefinite. The reference does not
/// enforce symmetry (the update preserves it exactly for finite values) and
/// does not run a general positive-definiteness check.
#[derive(Debug, Clone, PartialEq)]
pub struct RlsState {
    /// Parameter vector `w` (`D`).
    pub w: Vector,
    /// Inverse-correlation (covariance) matrix `P` (`D × D`).
    pub p: Matrix,
}

/// One supervised observation `(x, y)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RlsSample {
    /// Feature vector `x` (`D`).
    pub features: Vector,
    /// Target `y`.
    pub target: f32,
}

/// An immutable RLS configuration, shareable across streams.
#[derive(Debug, Clone, PartialEq)]
pub struct Rls {
    dimension: usize,
    lambda: f32,
    delta: f32,
}

impl Rls {
    /// Builds a configuration for `dimension` features, rejecting a zero
    /// dimension, a forgetting factor outside `(0, 1]`, or a non-positive /
    /// non-finite initial covariance scale.
    pub fn new(dimension: usize, lambda: f32, delta: f32) -> Result<Self, RlsError> {
        if dimension == 0 {
            return Err(RlsError::ZeroDimension);
        }
        if !lambda.is_finite() || lambda <= 0.0 || lambda > 1.0 {
            return Err(RlsError::InvalidForgettingFactor);
        }
        if !delta.is_finite() || delta <= 0.0 {
            return Err(RlsError::InvalidInitialCovariance);
        }
        Ok(Self {
            dimension,
            lambda,
            delta,
        })
    }

    /// The configured feature dimension `D`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The configured forgetting factor `λ`.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// The configured initial covariance scale `δ`.
    pub fn delta(&self) -> f32 {
        self.delta
    }

    /// The initial state `w_0 = 0`, `P_0 = δI`.
    ///
    /// `δI` is symmetric positive-definite by construction, so the reference
    /// never needs a general positive-definiteness check.
    pub fn initial_state(&self) -> RlsState {
        RlsState {
            w: Vector::zeros(self.dimension),
            p: Matrix::from_fn(self.dimension, self.dimension, |row, col| {
                if row == col { self.delta } else { 0.0 }
            }),
        }
    }

    /// Reference prediction `ŷ = wᵀx` using the committed state.
    ///
    /// The feature dimension is validated first, then feature finiteness,
    /// matching the validation order used by the other models.
    pub fn predict(&self, state: &RlsState, features: &Vector) -> Result<f32, RlsError> {
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(RlsError::NonFiniteInput);
        }
        Ok(state.w.dot(features)?)
    }

    /// Bounded ordered reference micro-batch over the RLS update path.
    ///
    /// This is the RLS-local bounded reference batching contract: a sequential,
    /// ordered fold — never vectorized and never independent — over a
    /// caller-bounded slice. It calls [`Rls::update`] directly for each borrowed
    /// observation; it does not call or reuse the foundation `process_batch`,
    /// which remains the unbounded reference fold. State stays coupled: success
    /// returns the committed `(w, P)` as one [`RlsState`].
    ///
    /// # Bound
    ///
    /// `batch.len() > max_batch` is rejected *before* any transition or
    /// validation, so an oversized batch processes nothing and returns
    /// [`RlsBatchError::BatchTooLarge`] with the original supplied state,
    /// unchanged and unvalidated. `max_batch == 0` therefore accepts only an
    /// empty batch.
    ///
    /// # Empty input
    ///
    /// An empty batch returns `state` unchanged without validating it. No
    /// additional initial-state validation is performed.
    ///
    /// # Failure
    ///
    /// Observations are processed strictly in slice order. The fold stops at
    /// the first failed transition and returns [`RlsBatchError::Transition`]
    /// with the zero-based `failed_index`, the last committed state after the
    /// successful prefix (the supplied state when nothing succeeded), and the
    /// underlying [`RlsError`]. Later observations are not processed and the
    /// failing candidate is never committed: there is no whole-batch rollback.
    /// Retrying a corrected observation, skipping a failed one, and any further
    /// recovery remain the caller's policy.
    ///
    /// # Continuity
    ///
    /// On success the returned state is the committed state after the final
    /// observation. Passing it to the next call continues the same sequence and
    /// is bitwise-equivalent to repeated single-observation stepping.
    pub fn process_batch_reference(
        &self,
        state: RlsState,
        batch: &[Observation<RlsSample>],
        max_batch: usize,
    ) -> Result<RlsState, RlsBatchError> {
        if batch.len() > max_batch {
            return Err(RlsBatchError::BatchTooLarge {
                max: max_batch,
                actual: batch.len(),
                state,
            });
        }
        let mut current = state;
        for (failed_index, observation) in batch.iter().enumerate() {
            match self.update(&current, observation) {
                Ok(next) => current = next,
                Err(error) => {
                    return Err(RlsBatchError::Transition {
                        failed_index,
                        state: current,
                        error,
                    });
                }
            }
        }
        Ok(current)
    }

    /// Validates the feature and state dimensions, in order `x`, `w`, `P`.
    fn validate(&self, state: &RlsState, features: &Vector) -> Result<(), RlsError> {
        if features.len() != self.dimension {
            return Err(RlsError::DimensionMismatch {
                expected: self.dimension,
                actual: features.len(),
            });
        }
        if state.w.len() != self.dimension {
            return Err(RlsError::DimensionMismatch {
                expected: self.dimension,
                actual: state.w.len(),
            });
        }
        if state.p.rows() != self.dimension {
            return Err(RlsError::DimensionMismatch {
                expected: self.dimension,
                actual: state.p.rows(),
            });
        }
        if state.p.cols() != self.dimension {
            return Err(RlsError::DimensionMismatch {
                expected: self.dimension,
                actual: state.p.cols(),
            });
        }
        Ok(())
    }
}

impl StateModel for Rls {
    type State = RlsState;
    type Observation = Observation<RlsSample>;
    type Error = RlsError;

    /// Computes the candidate `(w', P')` for one supervised observation.
    ///
    /// The candidate is returned as one value only after it validates, so a
    /// failed update cannot commit partial state.
    fn update(
        &self,
        state: &RlsState,
        observation: &Observation<RlsSample>,
    ) -> Result<RlsState, RlsError> {
        let sample = observation.value();
        let features = &sample.features;
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) || !sample.target.is_finite()
        {
            return Err(RlsError::NonFiniteInput);
        }

        // v = P x ; d = λ + xᵀv
        let v = state.p.mul_vector(features)?;
        let denominator = self.lambda + features.dot(&v)?;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(RlsError::InvalidDenominator);
        }

        // e = y − ŷ ; k = v / d ; w' = w + k e
        let error = sample.target - state.w.dot(features)?;
        let gain = v.map(|value| value / denominator);
        let next_w = state.w.add(&gain.scale(error))?;

        // P' = (P − v vᵀ / d) / λ, built row-major without new primitives.
        let p_data = state.p.as_slice();
        let v_data = v.as_slice();
        let dimension = self.dimension;
        let lambda = self.lambda;
        let next_p = Matrix::from_fn(dimension, dimension, |row, col| {
            let p = p_data[row * dimension + col];
            (p - v_data[row] * v_data[col] / denominator) / lambda
        });

        if !next_w.as_slice().iter().all(|value| value.is_finite())
            || !next_p.as_slice().iter().all(|value| value.is_finite())
        {
            return Err(RlsError::NonFiniteCandidate);
        }
        Ok(RlsState {
            w: next_w,
            p: next_p,
        })
    }
}
