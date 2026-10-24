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

use crate::foundation::numerical::{DimensionMismatch, Vector, Vector64};
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

pub(super) fn predict_borrowed(
    weights: &Vector,
    bias: f32,
    features: &Vector,
) -> Result<f32, RegressionError> {
    if features.len() != weights.len() {
        return Err(RegressionError::DimensionMismatch {
            expected: weights.len(),
            actual: features.len(),
        });
    }
    if !features.as_slice().iter().all(|value| value.is_finite()) {
        return Err(RegressionError::NonFiniteInput);
    }
    Ok(weights.dot(features)? + bias)
}

pub(super) fn predict_checked_borrowed(
    weights: &Vector,
    bias: f32,
    features: &Vector,
) -> Result<f32, RegressionError> {
    let value = predict_borrowed(weights, bias, features)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RegressionError::NonFiniteOutput { observation: None })
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
        predict_borrowed(&self.weights, self.bias, features)
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
        predict_checked_borrowed(&self.weights, self.bias, features)
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

pub(super) fn predict_borrowed64(
    weights: &Vector64,
    bias: f64,
    features: &Vector64,
) -> Result<f64, RegressionError> {
    if features.len() != weights.len() {
        return Err(RegressionError::DimensionMismatch {
            expected: weights.len(),
            actual: features.len(),
        });
    }
    if !features.as_slice().iter().all(|value| value.is_finite()) {
        return Err(RegressionError::NonFiniteInput);
    }
    Ok(weights.dot(features)? + bias)
}

pub(super) fn predict_checked_borrowed64(
    weights: &Vector64,
    bias: f64,
    features: &Vector64,
) -> Result<f64, RegressionError> {
    let value = predict_borrowed64(weights, bias, features)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RegressionError::NonFiniteOutput { observation: None })
    }
}

/// An immutable native-`f64` linear model `ŷ = w · x + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegression64 {
    weights: Vector64,
    bias: f64,
}

impl LinearRegression64 {
    /// Builds a model from `weights` and `bias`, rejecting an empty weight
    /// vector or non-finite parameters.
    pub fn new(weights: Vector64, bias: f64) -> Result<Self, RegressionError> {
        if weights.is_empty() {
            return Err(RegressionError::ZeroDimension);
        }
        if !bias.is_finite() || !weights.as_slice().iter().all(|value| value.is_finite()) {
            return Err(RegressionError::NonFiniteParameter);
        }
        Ok(Self { weights, bias })
    }

    /// The model weights.
    pub fn weights(&self) -> &Vector64 {
        &self.weights
    }

    /// The model bias.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Reference native-`f64` prediction `ŷ = w · x + b`.
    ///
    /// The dot product accumulates in weight order. The feature dimension is
    /// validated first, then feature finiteness; the output is not inspected.
    pub fn predict(&self, features: &Vector64) -> Result<f64, RegressionError> {
        predict_borrowed64(&self.weights, self.bias, features)
    }

    /// Checked prediction: rejects a non-finite output without changing the
    /// finite-result arithmetic or bits.
    pub fn predict_checked(&self, features: &Vector64) -> Result<f64, RegressionError> {
        predict_checked_borrowed64(&self.weights, self.bias, features)
    }

    /// Bounded group prediction over independent observations, in input order.
    pub fn predict_batch(
        &self,
        batch: &[Observation<Vector64>],
    ) -> Result<Vec<f64>, RegressionError> {
        batch
            .iter()
            .map(|observation| self.predict(observation.value()))
            .collect()
    }

    /// Checked bounded group prediction.
    ///
    /// Observations are processed in supplied order; the first failure is
    /// returned without a partial output vector. A non-finite output reports
    /// its zero-based group position. Empty groups succeed empty.
    pub fn predict_batch_checked(
        &self,
        batch: &[Observation<Vector64>],
    ) -> Result<Vec<f64>, RegressionError> {
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

impl StateModel for LinearRegression64 {
    /// The most recent prediction.
    type State = f64;
    type Observation = Observation<Vector64>;
    type Error = RegressionError;

    fn update(
        &self,
        _state: &f64,
        observation: &Observation<Vector64>,
    ) -> Result<f64, RegressionError> {
        self.predict(observation.value())
    }
}

/// A checked, borrowing view over an immutable [`LinearRegression64`].
#[derive(Debug, Clone)]
pub struct CheckedLinearRegression64<'a> {
    inner: &'a LinearRegression64,
}

impl<'a> CheckedLinearRegression64<'a> {
    /// Wraps an existing native-`f64` predictor by reference.
    pub fn new(inner: &'a LinearRegression64) -> Self {
        Self { inner }
    }

    /// The wrapped predictor.
    pub fn predictor(&self) -> &LinearRegression64 {
        self.inner
    }

    /// Checked single prediction.
    pub fn predict_checked(&self, features: &Vector64) -> Result<f64, RegressionError> {
        self.inner.predict_checked(features)
    }

    /// Checked grouped prediction.
    pub fn predict_batch_checked(
        &self,
        batch: &[Observation<Vector64>],
    ) -> Result<Vec<f64>, RegressionError> {
        self.inner.predict_batch_checked(batch)
    }
}

impl StateModel for CheckedLinearRegression64<'_> {
    /// The most recent checked prediction.
    type State = f64;
    type Observation = Observation<Vector64>;
    type Error = RegressionError;

    fn update(
        &self,
        _state: &f64,
        observation: &Observation<Vector64>,
    ) -> Result<f64, RegressionError> {
        self.inner.predict_checked(observation.value())
    }
}

#[cfg(test)]
mod tests {
    use super::{LinearRegression, RegressionError, predict_borrowed, predict_checked_borrowed};
    use crate::foundation::numerical::Vector;

    fn pre_refactor_prediction(weights: &[f32], bias: f32, features: &[f32]) -> f32 {
        weights
            .iter()
            .zip(features)
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + bias
    }

    #[test]
    fn borrowed_helpers_match_public_methods_as_delegates() {
        let weights = Vector::from_slice(&[1.0, 2.0]);
        let features = Vector::from_slice(&[3.0, 4.0]);
        let model = LinearRegression::new(weights.clone(), 0.5).unwrap();

        assert_eq!(
            predict_borrowed(&weights, 0.5, &features)
                .unwrap()
                .to_bits(),
            model.predict(&features).unwrap().to_bits()
        );
        assert_eq!(
            predict_checked_borrowed(&weights, 0.5, &features)
                .unwrap()
                .to_bits(),
            model.predict_checked(&features).unwrap().to_bits()
        );
    }

    #[test]
    fn independent_pre_refactor_baseline_keeps_exact_bits() {
        let ordinary = pre_refactor_prediction(&[1.0, 2.0], 0.5, &[3.0, 4.0]);
        assert_eq!(ordinary.to_bits(), 0x4138_0000);

        let ordered = pre_refactor_prediction(&[1.0e20, 3.0, -1.0e20], 1.0, &[1.0; 3]);
        assert_eq!(ordered.to_bits(), 0x3f80_0000);
        let differently_ordered = pre_refactor_prediction(&[1.0e20, -1.0e20, 3.0], 1.0, &[1.0; 3]);
        assert_eq!(differently_ordered.to_bits(), 0x4080_0000);

        let signed_zero = pre_refactor_prediction(&[-0.0], -0.0, &[1.0]);
        assert_eq!(signed_zero.to_bits(), 0x8000_0000);

        let cases = [
            (&[1.0, 2.0][..], 0.5, &[3.0, 4.0][..], 0x4138_0000),
            (&[1.0e20, 3.0, -1.0e20][..], 1.0, &[1.0; 3][..], 0x3f80_0000),
            (&[-0.0][..], -0.0, &[1.0][..], 0x8000_0000),
        ];
        for (weights, bias, features, expected_bits) in cases {
            let weights_vector = Vector::from_slice(weights);
            let features_vector = Vector::from_slice(features);
            let model = LinearRegression::new(weights_vector.clone(), bias).unwrap();
            assert_eq!(
                predict_borrowed(&weights_vector, bias, &features_vector)
                    .unwrap()
                    .to_bits(),
                expected_bits
            );
            assert_eq!(
                predict_checked_borrowed(&weights_vector, bias, &features_vector)
                    .unwrap()
                    .to_bits(),
                expected_bits
            );
            assert_eq!(
                model.predict(&features_vector).unwrap().to_bits(),
                expected_bits
            );
            assert_eq!(
                model.predict_checked(&features_vector).unwrap().to_bits(),
                expected_bits
            );
        }
    }

    #[test]
    fn borrowed_helpers_keep_input_precedence_and_output_policy() {
        let weights = Vector::from_slice(&[2.0]);
        let mismatched_non_finite = Vector::from_slice(&[f32::NAN, 1.0]);
        assert_eq!(
            predict_borrowed(&weights, 0.0, &mismatched_non_finite),
            Err(RegressionError::DimensionMismatch {
                expected: 1,
                actual: 2,
            })
        );
        assert_eq!(
            predict_checked_borrowed(&weights, 0.0, &Vector::from_slice(&[f32::NAN])),
            Err(RegressionError::NonFiniteInput)
        );

        let overflowing = Vector::from_slice(&[2.0e38]);
        let bare = predict_borrowed(&weights, 1.0, &overflowing).unwrap();
        assert!(bare.is_infinite());
        assert_eq!(
            predict_checked_borrowed(&weights, 1.0, &overflowing),
            Err(RegressionError::NonFiniteOutput { observation: None })
        );
    }
}
