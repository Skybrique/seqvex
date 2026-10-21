//! Linear regression prediction: `ŷ = w · x + b`.
//!
//! This is the reference slice for classic ML. It implements **prediction only**.
//! Online parameter updating is a separate, ordered state-update concern and is
//! deliberately not part of this model; see `docs/ML_VERTICAL_SLICES.md`.
//!
//! # Ownership
//!
//! Weights and bias are immutable once constructed. Prediction needs no mutable
//! model state and no reusable scratch, so this model implements
//! [`StateModel`] with a `&self` contract and is freely shareable. It does not
//! use the model-owned-workspace topology that the GRU slice introduced.
//!
//! # State semantics
//!
//! Prediction is stateless: each output depends only on its own observation and
//! the immutable weights. The [`StateModel::State`] carried by the framework is
//! the **most recent prediction**, so the ordered streaming semantics of the
//! execution layer apply unchanged, even though successive predictions do not
//! influence one another.
//!
//! # Micro-batch
//!
//! Because observations are independent under prediction, a bounded micro-batch
//! is admissible: [`LinearRegression::predict_batch`] returns one prediction per
//! observation in input order. It is a concrete, model-local capability — not a
//! generic micro-batch executor, and not a reinterpretation of the foundation's
//! reference `process_batch` fold.
//!
//! # Numerics
//!
//! Features and weights use the existing `f32` substrate ([`Vector::dot`]) so
//! prediction consumes the same [`Observation<Vector>`] the rest of the
//! framework uses. No `f64` conversion or new numerical primitive is added
//! without a measured requirement.
//!
//! ## Operating envelope and caller responsibility
//!
//! Construction rejects non-finite weights and bias
//! ([`RegressionError::NonFiniteParameter`]). Prediction validates the feature
//! dimension first, then feature finiteness ([`RegressionError::NonFiniteInput`]).
//! Prediction does **not** validate its output.
//!
//! Because the substrate is `f32`, finite operands can still overflow during
//! multiplication or summation and produce `±inf`; a subsequent `inf + (-inf)`
//! can produce `NaN`. Finite parameters and finite inputs therefore do **not**
//! guarantee a finite prediction. The caller is responsible for keeping the
//! products and sums representable, for example by scaling features and
//! coefficients.
//!
//! This is expected IEEE-754 `f32` behaviour, not a defect, and no numeric
//! cutoff is imposed or implied: Seqvex does not invent a magnitude threshold
//! beyond which input is rejected. The analogous pure-prediction path
//! `Rls::predict` also computes `wᵀx` without an output-finiteness check; each
//! model documents its own envelope rather than sharing a rule or abstraction.

use crate::foundation::numerical::{DimensionMismatch, Vector};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

/// Failure classes reported by linear-regression construction or prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionError {
    /// A configured dimension is zero.
    ZeroDimension,
    /// The feature vector length differs from the weight count.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// A weight or bias is not finite.
    NonFiniteParameter,
    /// An observation contains a non-finite feature.
    NonFiniteInput,
    /// A **checked** prediction produced an infinite or `NaN` output.
    ///
    /// `observation` is the zero-based position in the supplied group, or
    /// `None` for a single prediction. The bare [`LinearRegression::predict`]
    /// and [`LinearRegression::predict_batch`] paths do not report this.
    NonFiniteOutput {
        /// Zero-based group position, or `None` for a single prediction.
        observation: Option<usize>,
    },
}

impl std::fmt::Display for RegressionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => formatter
                .write_str("linear regression has zero dimension; at least one weight is required"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteParameter => {
                formatter.write_str("linear regression parameters must be finite")
            }
            Self::NonFiniteInput => {
                formatter.write_str("linear regression input features must be finite")
            }
            Self::NonFiniteOutput { observation: None } => {
                formatter.write_str("checked prediction produced a non-finite output")
            }
            Self::NonFiniteOutput {
                observation: Some(index),
            } => write!(
                formatter,
                "checked group prediction produced a non-finite output at observation {index}"
            ),
        }
    }
}

impl std::error::Error for RegressionError {}

impl From<DimensionMismatch> for RegressionError {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// An immutable linear model `ŷ = w · x + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegression {
    weights: Vector,
    bias: f32,
}

impl LinearRegression {
    /// Builds a model from `weights` and `bias`, rejecting an empty weight
    /// vector or non-finite parameters.
    pub fn new(weights: Vector, bias: f32) -> Result<Self, RegressionError> {
        if weights.is_empty() {
            return Err(RegressionError::ZeroDimension);
        }
        if !bias.is_finite() || !weights.as_slice().iter().all(|value| value.is_finite()) {
            return Err(RegressionError::NonFiniteParameter);
        }
        Ok(Self { weights, bias })
    }

    /// The model weights.
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// The model bias.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Reference prediction `ŷ = w · x + b`.
    ///
    /// The dot product accumulates in weight order; this ordering is the
    /// semantic oracle every other prediction path must preserve bitwise.
    ///
    /// The feature dimension is validated first, then feature finiteness,
    /// matching the validation order used by the existing models.
    pub fn predict(&self, features: &Vector) -> Result<f32, RegressionError> {
        if features.len() != self.weights.len() {
            return Err(RegressionError::DimensionMismatch {
                expected: self.weights.len(),
                actual: features.len(),
            });
        }
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(RegressionError::NonFiniteInput);
        }
        Ok(self.weights.dot(features)? + self.bias)
    }

    /// Checked prediction: like [`LinearRegression::predict`], but additionally
    /// rejects a non-finite result.
    ///
    /// Input validation and its error categories are unchanged; a finite result
    /// is returned with identical `f32` bits to [`LinearRegression::predict`].
    /// A non-finite result returns [`RegressionError::NonFiniteOutput`] with
    /// `observation = None`. No saturation, clipping, or second arithmetic path
    /// is introduced.
    pub fn predict_checked(&self, features: &Vector) -> Result<f32, RegressionError> {
        let value = self.predict(features)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(RegressionError::NonFiniteOutput { observation: None })
        }
    }

    /// Bounded micro-batch prediction over independent observations.
    ///
    /// The batch is exactly the supplied slice (the caller bounds it) and the
    /// outputs are returned in input order. Prediction is order-independent, so
    /// aggregating observations cannot change any individual result; the tests
    /// assert bitwise equality with repeated single-observation prediction.
    ///
    /// A straightforward sequential map, not a vectorized kernel.
    /// There is no measured LR bottleneck justifying SIMD, so none is added.
    pub fn predict_batch(
        &self,
        batch: &[Observation<Vector>],
    ) -> Result<Vec<f32>, RegressionError> {
        batch
            .iter()
            .map(|observation| self.predict(observation.value()))
            .collect()
    }

    /// Checked bounded micro-batch prediction.
    ///
    /// Observations are processed in supplied order and the first failure is
    /// returned immediately, so an earlier output failure is never masked by a
    /// later invalid input. Input errors keep their existing categories; a
    /// non-finite output returns [`RegressionError::NonFiniteOutput`] with
    /// `observation = Some(index)` for its zero-based group position. No partial
    /// output vector is returned. An empty group returns an empty success.
    pub fn predict_batch_checked(
        &self,
        batch: &[Observation<Vector>],
    ) -> Result<Vec<f32>, RegressionError> {
        let mut outputs = Vec::with_capacity(batch.len());
        for (index, observation) in batch.iter().enumerate() {
            let value = self.predict(observation.value())?;
            if !value.is_finite() {
                return Err(RegressionError::NonFiniteOutput {
                    observation: Some(index),
                });
            }
            outputs.push(value);
        }
        Ok(outputs)
    }
}

impl StateModel for LinearRegression {
    /// The most recent prediction; see the module documentation.
    type State = f32;
    type Observation = Observation<Vector>;
    type Error = RegressionError;

    fn update(
        &self,
        _state: &f32,
        observation: &Observation<Vector>,
    ) -> Result<f32, RegressionError> {
        self.predict(observation.value())
    }
}

/// A checked view over an immutable [`LinearRegression`].
///
/// It **borrows** the existing predictor and delegates to
/// [`LinearRegression::predict_checked`] through the [`StateModel`] contract, so
/// it can be driven by the existing
/// [`crate::execution::streaming::StreamingExecutor`] with no parameter clone
/// and no hidden mutable state. Only the model's *result classification*
/// changes — a non-finite output becomes an error; how the executor handles
/// errors is unchanged.
#[derive(Debug, Clone)]
pub struct CheckedLinearRegression<'a> {
    inner: &'a LinearRegression,
}

impl<'a> CheckedLinearRegression<'a> {
    /// Wraps an existing predictor by reference.
    pub fn new(inner: &'a LinearRegression) -> Self {
        Self { inner }
    }

    /// The wrapped predictor.
    pub fn predictor(&self) -> &LinearRegression {
        self.inner
    }

    /// Checked single prediction; see [`LinearRegression::predict_checked`].
    pub fn predict_checked(&self, features: &Vector) -> Result<f32, RegressionError> {
        self.inner.predict_checked(features)
    }

    /// Checked grouped prediction; see
    /// [`LinearRegression::predict_batch_checked`].
    pub fn predict_batch_checked(
        &self,
        batch: &[Observation<Vector>],
    ) -> Result<Vec<f32>, RegressionError> {
        self.inner.predict_batch_checked(batch)
    }
}

impl StateModel for CheckedLinearRegression<'_> {
    /// The most recent checked prediction.
    type State = f32;
    type Observation = Observation<Vector>;
    type Error = RegressionError;

    fn update(
        &self,
        _state: &f32,
        observation: &Observation<Vector>,
    ) -> Result<f32, RegressionError> {
        self.inner.predict_checked(observation.value())
    }
}
