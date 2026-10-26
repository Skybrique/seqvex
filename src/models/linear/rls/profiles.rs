//! Precision profiles for recursive least squares.
//!
//! [`Rls64`] is the native-`f64` learner. [`RlsServing32`] is an explicit,
//! independently owned `f32` serving snapshot produced from an [`RlsState64`].
//! The legacy [`Rls`](super::Rls) remains the strict-`f32` profile and gains
//! checked state admission through [`RlsState::admit`](super::RlsState::admit).

use crate::foundation::numerical::{DimensionMismatch, Matrix, Matrix64, Vector, Vector64};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

use super::RlsError;

/// Failure classes reported by native-`f64` RLS construction, prediction, or
/// update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlsError64 {
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
    /// A checked prediction produced a non-finite result.
    NonFiniteOutput,
}

impl std::fmt::Display for RlsError64 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("RLS dimension must be nonzero"),
            Self::DimensionMismatch { expected, actual } => {
                write!(formatter, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidForgettingFactor => {
                formatter.write_str("RLS forgetting factor must be finite and in (0, 1]")
            }
            Self::InvalidInitialCovariance => {
                formatter.write_str("RLS initial covariance scale must be finite and positive")
            }
            Self::NonFiniteInput => {
                formatter.write_str("RLS observation features and target must be finite")
            }
            Self::InvalidDenominator => {
                formatter.write_str("RLS update denominator must be finite and positive")
            }
            Self::NonFiniteCandidate => formatter.write_str(
                "RLS candidate state is not finite. The candidate update was not committed; the supplied state is unchanged.",
            ),
            Self::NonFiniteOutput => {
                formatter.write_str("checked RLS prediction produced a non-finite output")
            }
        }
    }
}

impl std::error::Error for RlsError64 {}

impl From<DimensionMismatch> for RlsError64 {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// Failure of a bounded native-`f64` RLS reference-batch call.
#[derive(Debug, Clone, PartialEq)]
pub enum RlsBatchError64 {
    /// The supplied batch is longer than the caller-declared bound.
    BatchTooLarge {
        /// The caller-declared maximum batch length.
        max: usize,
        /// The supplied batch length.
        actual: usize,
        /// The original supplied state, returned unchanged and unvalidated.
        state: RlsState64,
    },
    /// The ordered fold stopped at a failed observation.
    Transition {
        /// Zero-based index of the failed observation.
        failed_index: usize,
        /// The state before the failed observation.
        state: RlsState64,
        /// The failure reported by the reference transition.
        error: RlsError64,
    },
}

impl std::fmt::Display for RlsBatchError64 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BatchTooLarge { max, actual, .. } => write!(
                formatter,
                "RLS batch length {actual} exceeds maximum {max}; no observation was processed and the supplied state was returned unchanged"
            ),
            Self::Transition {
                failed_index: 0, ..
            } => formatter.write_str(
                "RLS batch transition failed at observation 0; no observation was committed and the supplied state was returned unchanged",
            ),
            Self::Transition {
                failed_index,
                ..
            } => write!(
                formatter,
                "RLS batch transition failed at observation {failed_index}; {failed_index} observations were committed, the failed candidate update was not committed, and the state before the failed observation was returned (no whole-batch rollback)"
            ),
        }
    }
}

impl std::error::Error for RlsBatchError64 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transition { error, .. } => Some(error),
            Self::BatchTooLarge { .. } => None,
        }
    }
}

/// Checked external-state admission failure shared by the `f64` and legacy
/// `f32` RLS profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlsStateAdmissionError {
    /// The covariance has zero dimension.
    ZeroDimension,
    /// The covariance is not square.
    CovarianceNotSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
    /// The weight vector length differs from the covariance dimension.
    WeightDimensionMismatch {
        /// The covariance dimension.
        expected: usize,
        /// The supplied weight length.
        actual: usize,
    },
    /// A weight is not finite.
    NonFiniteWeight {
        /// First offending zero-based index.
        index: usize,
    },
    /// A covariance entry is not finite.
    NonFiniteCovariance {
        /// First offending row in row-major order.
        row: usize,
        /// First offending column in row-major order.
        column: usize,
    },
    /// The covariance is not bitwise symmetric.
    AsymmetricCovariance {
        /// First offending row in row-major order.
        row: usize,
        /// First offending column in row-major order.
        column: usize,
    },
}

impl std::fmt::Display for RlsStateAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("RLS state dimension must be nonzero"),
            Self::CovarianceNotSquare { rows, cols } => {
                write!(
                    formatter,
                    "RLS covariance must be square, got {rows}×{cols}"
                )
            }
            Self::WeightDimensionMismatch { expected, actual } => write!(
                formatter,
                "RLS weight dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteWeight { index } => {
                write!(formatter, "RLS weight at index {index} must be finite")
            }
            Self::NonFiniteCovariance { row, column } => write!(
                formatter,
                "RLS covariance entry at ({row}, {column}) must be finite"
            ),
            Self::AsymmetricCovariance { row, column } => write!(
                formatter,
                "RLS covariance is not bitwise symmetric at ({row}, {column}) versus ({column}, {row})"
            ),
        }
    }
}

impl std::error::Error for RlsStateAdmissionError {}

/// Failure of a checked native-`f64` state conversion into `f32` serving
/// weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlsServingConversionError {
    /// The state weight length differs from the learner dimension.
    StateDimensionMismatch {
        /// The learner dimension.
        expected: usize,
        /// The supplied state weight length.
        actual: usize,
    },
    /// A source weight is not finite.
    SourceNonFinite {
        /// First offending zero-based index.
        index: usize,
    },
    /// A finite source weight rounds to a non-finite `f32`.
    Overflow {
        /// First offending zero-based index.
        index: usize,
    },
    /// A nonzero source weight rounds to zero.
    Underflow {
        /// First offending zero-based index.
        index: usize,
    },
}

impl std::fmt::Display for RlsServingConversionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateDimensionMismatch { expected, actual } => write!(
                formatter,
                "RLS serving conversion dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::SourceNonFinite { index } => write!(
                formatter,
                "RLS serving source weight at index {index} is not finite"
            ),
            Self::Overflow { index } => write!(
                formatter,
                "RLS serving weight at index {index} overflows f32"
            ),
            Self::Underflow { index } => write!(
                formatter,
                "RLS serving weight at index {index} underflows to zero"
            ),
        }
    }
}

impl std::error::Error for RlsServingConversionError {}

/// The adaptive state of one native-`f64` RLS stream.
#[derive(Debug, Clone, PartialEq)]
pub struct RlsState64 {
    w: Vector64,
    p: Matrix64,
}

impl RlsState64 {
    /// Admits an externally supplied state after checking, in order: zero
    /// dimension, square covariance, weight length, finite weights, finite
    /// covariance entries, and bitwise symmetry.
    ///
    /// Admission establishes only those facts. It does not establish positive
    /// semidefiniteness, conditioning, update provenance, learned readiness, or
    /// future denominator positivity, and it performs no repair.
    pub fn admit(weights: Vector64, covariance: Matrix64) -> Result<Self, RlsStateAdmissionError> {
        let dimension = covariance.rows();
        if dimension == 0 {
            return Err(RlsStateAdmissionError::ZeroDimension);
        }
        if dimension != covariance.cols() {
            return Err(RlsStateAdmissionError::CovarianceNotSquare {
                rows: dimension,
                cols: covariance.cols(),
            });
        }
        if weights.len() != dimension {
            return Err(RlsStateAdmissionError::WeightDimensionMismatch {
                expected: dimension,
                actual: weights.len(),
            });
        }
        if let Some(index) = weights
            .as_slice()
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(RlsStateAdmissionError::NonFiniteWeight { index });
        }
        for row in 0..dimension {
            for column in 0..dimension {
                if !covariance.get(row, column).unwrap().is_finite() {
                    return Err(RlsStateAdmissionError::NonFiniteCovariance { row, column });
                }
            }
        }
        for row in 0..dimension {
            for column in 0..dimension {
                if covariance.get(row, column).unwrap().to_bits()
                    != covariance.get(column, row).unwrap().to_bits()
                {
                    return Err(RlsStateAdmissionError::AsymmetricCovariance { row, column });
                }
            }
        }
        Ok(Self {
            w: weights,
            p: covariance,
        })
    }

    /// The admitted parameter vector.
    pub fn weights(&self) -> &Vector64 {
        &self.w
    }

    /// The admitted covariance matrix.
    pub fn covariance(&self) -> &Matrix64 {
        &self.p
    }
}

/// One supervised native-`f64` observation `(x, y)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RlsSample64 {
    /// Feature vector `x` (`D`).
    pub features: Vector64,
    /// Target `y`.
    pub target: f64,
}

/// An immutable native-`f64` RLS configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Rls64 {
    dimension: usize,
    lambda: f64,
    delta: f64,
}

impl Rls64 {
    /// Builds a native-`f64` configuration, rejecting a zero dimension, a
    /// forgetting factor outside `(0, 1]`, or a non-positive / non-finite
    /// initial covariance scale.
    pub fn new(dimension: usize, lambda: f64, delta: f64) -> Result<Self, RlsError64> {
        if dimension == 0 {
            return Err(RlsError64::ZeroDimension);
        }
        if !lambda.is_finite() || lambda <= 0.0 || lambda > 1.0 {
            return Err(RlsError64::InvalidForgettingFactor);
        }
        if !delta.is_finite() || delta <= 0.0 {
            return Err(RlsError64::InvalidInitialCovariance);
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
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The configured initial covariance scale `δ`.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// The initial state `w_0 = 0`, `P_0 = δI`.
    ///
    /// This is a prior, not evidence of a learned model.
    pub fn initial_state(&self) -> RlsState64 {
        RlsState64 {
            w: Vector64::zeros(self.dimension),
            p: Matrix64::from_fn(self.dimension, self.dimension, |row, column| {
                if row == column { self.delta } else { 0.0 }
            }),
        }
    }

    /// Converts the authoritative `f64` weights into an independently owned
    /// immutable `f32` serving snapshot.
    ///
    /// The state dimension is checked before any weight is scanned. Coefficients
    /// are converted in increasing index order with ordinary `as f32` rounding.
    /// Non-finite sources, overflow, and nonzero-to-zero underflow reject the
    /// conversion. The covariance is neither converted nor certified, and the
    /// source state is unchanged.
    pub fn to_serving_f32(
        &self,
        state: &RlsState64,
    ) -> Result<RlsServing32, RlsServingConversionError> {
        if state.w.len() != self.dimension {
            return Err(RlsServingConversionError::StateDimensionMismatch {
                expected: self.dimension,
                actual: state.w.len(),
            });
        }
        let mut weights = Vec::with_capacity(self.dimension);
        for (index, &source) in state.w.as_slice().iter().enumerate() {
            if !source.is_finite() {
                return Err(RlsServingConversionError::SourceNonFinite { index });
            }
            let narrowed = source as f32;
            if !narrowed.is_finite() {
                return Err(RlsServingConversionError::Overflow { index });
            }
            if narrowed == 0.0 && source != 0.0 {
                return Err(RlsServingConversionError::Underflow { index });
            }
            weights.push(narrowed);
        }
        Ok(RlsServing32 {
            weights: Vector::from_slice(&weights),
        })
    }

    /// Reference prediction `ŷ = wᵀx` using the committed native-`f64` state.
    ///
    /// Dimensions are validated in the order features, weights, covariance rows,
    /// covariance columns; feature finiteness is checked afterwards. A bare
    /// prediction may return a non-finite `Ok` value.
    pub fn predict(&self, state: &RlsState64, features: &Vector64) -> Result<f64, RlsError64> {
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(RlsError64::NonFiniteInput);
        }
        Ok(state.w.dot(features)?)
    }

    /// Checked prediction: delegates to [`Rls64::predict`] and rejects a
    /// non-finite output with [`RlsError64::NonFiniteOutput`].
    pub fn predict_checked(
        &self,
        state: &RlsState64,
        features: &Vector64,
    ) -> Result<f64, RlsError64> {
        let value = self.predict(state, features)?;
        if !value.is_finite() {
            return Err(RlsError64::NonFiniteOutput);
        }
        Ok(value)
    }

    /// Bounded ordered native-`f64` reference micro-batch.
    ///
    /// The bound is checked before any transition. An empty batch returns the
    /// supplied state unchanged and unvalidated. The first failed observation
    /// stops the fold and returns the state before that observation plus its
    /// zero-based index.
    pub fn process_batch_reference(
        &self,
        state: RlsState64,
        batch: &[Observation<RlsSample64>],
        max_batch: usize,
    ) -> Result<RlsState64, RlsBatchError64> {
        if batch.len() > max_batch {
            return Err(RlsBatchError64::BatchTooLarge {
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
                    return Err(RlsBatchError64::Transition {
                        failed_index,
                        state: current,
                        error,
                    });
                }
            }
        }
        Ok(current)
    }

    fn validate(&self, state: &RlsState64, features: &Vector64) -> Result<(), RlsError64> {
        if features.len() != self.dimension {
            return Err(RlsError64::DimensionMismatch {
                expected: self.dimension,
                actual: features.len(),
            });
        }
        if state.w.len() != self.dimension {
            return Err(RlsError64::DimensionMismatch {
                expected: self.dimension,
                actual: state.w.len(),
            });
        }
        if state.p.rows() != self.dimension {
            return Err(RlsError64::DimensionMismatch {
                expected: self.dimension,
                actual: state.p.rows(),
            });
        }
        if state.p.cols() != self.dimension {
            return Err(RlsError64::DimensionMismatch {
                expected: self.dimension,
                actual: state.p.cols(),
            });
        }
        Ok(())
    }
}

impl StateModel for Rls64 {
    type State = RlsState64;
    type Observation = Observation<RlsSample64>;
    type Error = RlsError64;

    fn update(
        &self,
        state: &RlsState64,
        observation: &Observation<RlsSample64>,
    ) -> Result<RlsState64, RlsError64> {
        let sample = observation.value();
        let features = &sample.features;
        self.validate(state, features)?;
        if !features.as_slice().iter().all(|value| value.is_finite()) || !sample.target.is_finite()
        {
            return Err(RlsError64::NonFiniteInput);
        }

        let v = state.p.mul_vector(features)?;
        let denominator = self.lambda + features.dot(&v)?;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(RlsError64::InvalidDenominator);
        }

        let error = sample.target - state.w.dot(features)?;
        let gain = v.map(|value| value / denominator);
        let next_w = state.w.add(&gain.scale(error))?;

        let p_data = state.p.as_slice();
        let v_data = v.as_slice();
        let dimension = self.dimension;
        let lambda = self.lambda;
        let next_p = Matrix64::from_fn(dimension, dimension, |row, column| {
            let p = p_data[row * dimension + column];
            (p - v_data[row] * v_data[column] / denominator) / lambda
        });

        if !next_w.as_slice().iter().all(|value| value.is_finite())
            || !next_p.as_slice().iter().all(|value| value.is_finite())
        {
            return Err(RlsError64::NonFiniteCandidate);
        }
        Ok(RlsState64 {
            w: next_w,
            p: next_p,
        })
    }
}

/// An independently owned `f32` serving snapshot of native-`f64` RLS weights.
#[derive(Debug, Clone, PartialEq)]
pub struct RlsServing32 {
    weights: Vector,
}

impl RlsServing32 {
    /// The snapshot feature dimension.
    pub fn dimension(&self) -> usize {
        self.weights.len()
    }

    /// The converted serving weights.
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// Fixed-model prediction using the stored `f32` weights.
    ///
    /// The feature length is checked before feature finiteness. A bare
    /// prediction may return a non-finite `Ok` value.
    pub fn predict(&self, features: &Vector) -> Result<f32, RlsError> {
        if features.len() != self.weights.len() {
            return Err(RlsError::DimensionMismatch {
                expected: self.weights.len(),
                actual: features.len(),
            });
        }
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(RlsError::NonFiniteInput);
        }
        Ok(self.weights.dot(features)?)
    }

    /// Checked fixed-model prediction: delegates to [`RlsServing32::predict`]
    /// and rejects a non-finite output with [`RlsError::NonFiniteOutput`].
    pub fn predict_checked(&self, features: &Vector) -> Result<f32, RlsError> {
        let value = self.predict(features)?;
        if !value.is_finite() {
            return Err(RlsError::NonFiniteOutput);
        }
        Ok(value)
    }
}

impl super::RlsState {
    /// Admits an externally supplied legacy `f32` RLS state after checking, in
    /// order: zero dimension, square covariance, weight length, finite weights,
    /// finite covariance entries, and bitwise symmetry.
    ///
    /// Admission establishes only those facts. It does not establish positive
    /// semidefiniteness, conditioning, update provenance, learned readiness, or
    /// future denominator positivity, and it performs no repair. The legacy
    /// fields remain public: a caller can bypass admission or mutate the state
    /// afterwards, so this is a check at this call, not a permanent invariant.
    pub fn admit(weights: Vector, covariance: Matrix) -> Result<Self, RlsStateAdmissionError> {
        let dimension = covariance.rows();
        if dimension == 0 {
            return Err(RlsStateAdmissionError::ZeroDimension);
        }
        if dimension != covariance.cols() {
            return Err(RlsStateAdmissionError::CovarianceNotSquare {
                rows: dimension,
                cols: covariance.cols(),
            });
        }
        if weights.len() != dimension {
            return Err(RlsStateAdmissionError::WeightDimensionMismatch {
                expected: dimension,
                actual: weights.len(),
            });
        }
        if let Some(index) = weights
            .as_slice()
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(RlsStateAdmissionError::NonFiniteWeight { index });
        }
        for row in 0..dimension {
            for column in 0..dimension {
                if !covariance.get(row, column).unwrap().is_finite() {
                    return Err(RlsStateAdmissionError::NonFiniteCovariance { row, column });
                }
            }
        }
        for row in 0..dimension {
            for column in 0..dimension {
                if covariance.get(row, column).unwrap().to_bits()
                    != covariance.get(column, row).unwrap().to_bits()
                {
                    return Err(RlsStateAdmissionError::AsymmetricCovariance { row, column });
                }
            }
        }
        Ok(Self {
            w: weights,
            p: covariance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn private_state(weights: &[f64], rows: &[&[f64]]) -> RlsState64 {
        RlsState64 {
            w: Vector64::from_slice(weights),
            p: Matrix64::from_rows(rows).unwrap(),
        }
    }

    fn bits(state: &RlsState64) -> (Vec<u64>, Vec<u64>) {
        (
            state
                .w
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            state
                .p
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
        )
    }

    #[test]
    fn conversion_rejects_private_non_finite_sources_without_mutation() {
        let model = Rls64::new(2, 1.0, 1.0).unwrap();
        let active = model.to_serving_f32(&model.initial_state()).unwrap();
        let cases = [
            ([f64::NAN, 1.0], 0),
            ([1.0, f64::INFINITY], 1),
            ([1.0, f64::NEG_INFINITY], 1),
        ];
        for (weights, index) in cases {
            let state = private_state(&weights, &[&[1.0, 0.0], &[0.0, 1.0]]);
            let before = bits(&state);
            assert_eq!(
                model.to_serving_f32(&state).unwrap_err(),
                RlsServingConversionError::SourceNonFinite { index }
            );
            assert_eq!(bits(&state), before);
            assert_eq!(active.dimension(), 2);
        }
    }

    #[test]
    fn conversion_checks_dimension_before_a_non_finite_source() {
        let model = Rls64::new(2, 1.0, 1.0).unwrap();
        let state = private_state(&[f64::NAN], &[&[1.0]]);
        assert_eq!(
            model.to_serving_f32(&state).unwrap_err(),
            RlsServingConversionError::StateDimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn malformed_private_state_shapes_are_reported_by_prediction_and_update() {
        let model = Rls64::new(2, 1.0, 1.0).unwrap();
        let features = Vector64::from_slice(&[1.0, 2.0]);
        let short_weights = private_state(&[1.0], &[&[1.0]]);
        assert_eq!(
            model.predict(&short_weights, &features).unwrap_err(),
            RlsError64::DimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
        let short_rows = private_state(&[1.0, 2.0], &[&[1.0, 0.0]]);
        assert_eq!(
            model.predict(&short_rows, &features).unwrap_err(),
            RlsError64::DimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
        let short_cols = private_state(&[1.0, 2.0], &[&[1.0], &[0.0]]);
        assert_eq!(
            model.predict(&short_cols, &features).unwrap_err(),
            RlsError64::DimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
        let observation = Observation::new(RlsSample64 {
            features,
            target: 1.0,
        });
        assert_eq!(
            model.update(&short_cols, &observation).unwrap_err(),
            RlsError64::DimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }
}
