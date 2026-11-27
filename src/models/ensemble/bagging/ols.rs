//! OLS bags for Profiles A, B, and C.

use std::fmt;

use crate::foundation::numerical::{Vector, Vector64};
use crate::foundation::observation::Observation;
use crate::models::linear::ols::{
    FitConfig32, FitConfig64, FitError32, FitError64, LinearRegression, LinearRegression64,
    RegressionError, ServingConversionError, UnderflowPolicy, fit_f32, fit_f64,
};
use crate::preprocessing::sampling::SampleUnit;

use super::{
    AggregationStage, BaggingError, BaggingSpec, MemberPredictCause, PredictionError, RowError,
    assemble_candidate, mean::mean_f32, mean::mean_f64, publish_batch, retain_ols32, retain_ols64,
    validate_row_32, validate_row_64, validate_units,
};

/// OLS-bag fit or prediction failure.
#[derive(Debug, Clone, PartialEq)]
pub enum OlsBagError {
    Bagging(BaggingError),
    MemberFit64 {
        member: usize,
        source: FitError64,
    },
    MemberFit32 {
        member: usize,
        source: FitError32,
    },
    MemberConversion {
        member: usize,
        source: ServingConversionError,
    },
    Predict(PredictionError<RegressionError>),
}

impl fmt::Display for OlsBagError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bagging(error) => write!(formatter, "{error}"),
            Self::MemberFit64 { member, source } => {
                write!(formatter, "member {member} f64 fit failed: {source}")
            }
            Self::MemberFit32 { member, source } => {
                write!(formatter, "member {member} f32 fit failed: {source}")
            }
            Self::MemberConversion { member, source } => {
                write!(formatter, "member {member} conversion failed: {source}")
            }
            Self::Predict(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for OlsBagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bagging(error) => Some(error),
            Self::MemberFit64 { source, .. } => Some(source),
            Self::MemberFit32 { source, .. } => Some(source),
            Self::MemberConversion { source, .. } => Some(source),
            Self::Predict(error) => Some(error),
        }
    }
}

impl From<BaggingError> for OlsBagError {
    fn from(error: BaggingError) -> Self {
        Self::Bagging(error)
    }
}

/// Native-f64 OLS bag (Profile A).
#[derive(Debug, Clone, PartialEq)]
pub struct OlsBag64 {
    members: Vec<LinearRegression64>,
    feature_dimension: usize,
    retained_bytes: usize,
    max_predictions: usize,
}

impl OlsBag64 {
    pub fn fit(
        features: &[Vector64],
        targets: &[f64],
        units: &[SampleUnit],
        spec: &BaggingSpec,
        config: &FitConfig64,
    ) -> Result<Self, OlsBagError> {
        validate_units(units, features.len(), targets.len(), spec, false)?;
        let candidate = assemble_candidate(
            units,
            spec,
            false,
            OlsBagError::Bagging,
            |member, rows| {
                let fit = fit_f64(features, targets, rows, config)
                    .map_err(|source| OlsBagError::MemberFit64 { member, source })?;
                let predictor = fit.into_predictor();
                let mut bytes = 0;
                retain_ols64(&mut bytes, &predictor).map_err(OlsBagError::Bagging)?;
                Ok((predictor, bytes))
            },
            |model| model.weights().len(),
            |member, expected, actual| {
                OlsBagError::Bagging(BaggingError::MemberDimensionMismatch {
                    member,
                    expected,
                    actual,
                })
            },
            |requested, limit| {
                OlsBagError::Bagging(BaggingError::RetainedBytesExceeded { requested, limit })
            },
        )?;
        Ok(Self {
            members: candidate.members,
            feature_dimension: candidate.feature_dimension,
            retained_bytes: candidate.retained_bytes,
            max_predictions: spec.limits.max_predictions,
        })
    }

    pub fn members(&self) -> &[LinearRegression64] {
        &self.members
    }

    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn predict(&self, row: &Vector64) -> Result<f64, OlsBagError> {
        validate_row_64(row, self.feature_dimension)
            .map_err(|cause| prediction_row_validation(0, cause))?;
        let mut predictions = Vec::with_capacity(self.members.len());
        for (member, model) in self.members.iter().enumerate() {
            match model.predict(row) {
                Ok(value) if value.is_finite() => predictions.push(value),
                Ok(_) => {
                    return Err(OlsBagError::Predict(PredictionError::MemberPredict {
                        row_index: 0,
                        member_index: member,
                        cause: MemberPredictCause::NonFiniteOutput,
                        evaluated_rows: 0,
                    }));
                }
                Err(source) => {
                    return Err(OlsBagError::Predict(PredictionError::MemberPredict {
                        row_index: 0,
                        member_index: member,
                        cause: MemberPredictCause::Predictor(source),
                        evaluated_rows: 0,
                    }));
                }
            }
        }
        mean_f64(&predictions).map_err(|cause| {
            OlsBagError::Predict(PredictionError::Aggregation {
                row_index: 0,
                stage: AggregationStage::Mean,
                cause,
                evaluated_rows: 0,
            })
        })
    }

    pub fn predict_batch(
        &self,
        batch: &[Observation<Vector64>],
        max_predictions: usize,
    ) -> Result<Vec<f64>, OlsBagError> {
        publish_batch::<Vector64, f64, RegressionError>(
            batch,
            max_predictions,
            self.max_predictions,
            self.members.len(),
            |_row_index, row| validate_row_64(row, self.feature_dimension),
            |_row_index, member_index, row| self.members[member_index].predict(row),
            |value| value.is_finite(),
            mean_f64,
        )
        .map_err(OlsBagError::Predict)
    }
}

/// f32-serving OLS bag (Profiles B and C).
#[derive(Debug, Clone, PartialEq)]
pub struct OlsBag {
    members: Vec<LinearRegression>,
    feature_dimension: usize,
    retained_bytes: usize,
    max_predictions: usize,
}

impl OlsBag {
    /// Profile B: f64 member fits, checked f32 conversion.
    pub fn fit_from_f64(
        features: &[Vector64],
        targets: &[f64],
        units: &[SampleUnit],
        spec: &BaggingSpec,
        config: &FitConfig64,
        policy: UnderflowPolicy,
    ) -> Result<Self, OlsBagError> {
        validate_units(units, features.len(), targets.len(), spec, false)?;
        let candidate = assemble_candidate(
            units,
            spec,
            false,
            OlsBagError::Bagging,
            |member, rows| {
                let fit = fit_f64(features, targets, rows, config)
                    .map_err(|source| OlsBagError::MemberFit64 { member, source })?;
                let predictor = fit
                    .to_serving_f32(policy)
                    .map_err(|source| OlsBagError::MemberConversion { member, source })?;
                let mut bytes = 0;
                retain_ols32(&mut bytes, &predictor).map_err(OlsBagError::Bagging)?;
                Ok((predictor, bytes))
            },
            |model| model.weights().len(),
            |member, expected, actual| {
                OlsBagError::Bagging(BaggingError::MemberDimensionMismatch {
                    member,
                    expected,
                    actual,
                })
            },
            |requested, limit| {
                OlsBagError::Bagging(BaggingError::RetainedBytesExceeded { requested, limit })
            },
        )?;
        Ok(Self {
            members: candidate.members,
            feature_dimension: candidate.feature_dimension,
            retained_bytes: candidate.retained_bytes,
            max_predictions: spec.limits.max_predictions,
        })
    }

    /// Profile C: strict f32 member fits.
    pub fn fit_f32(
        features: &[Vector],
        targets: &[f32],
        units: &[SampleUnit],
        spec: &BaggingSpec,
        config: &FitConfig32,
    ) -> Result<Self, OlsBagError> {
        validate_units(units, features.len(), targets.len(), spec, false)?;
        let candidate = assemble_candidate(
            units,
            spec,
            false,
            OlsBagError::Bagging,
            |member, rows| {
                let fit = fit_f32(features, targets, rows, config)
                    .map_err(|source| OlsBagError::MemberFit32 { member, source })?;
                let predictor = fit.into_predictor();
                let mut bytes = 0;
                retain_ols32(&mut bytes, &predictor).map_err(OlsBagError::Bagging)?;
                Ok((predictor, bytes))
            },
            |model| model.weights().len(),
            |member, expected, actual| {
                OlsBagError::Bagging(BaggingError::MemberDimensionMismatch {
                    member,
                    expected,
                    actual,
                })
            },
            |requested, limit| {
                OlsBagError::Bagging(BaggingError::RetainedBytesExceeded { requested, limit })
            },
        )?;
        Ok(Self {
            members: candidate.members,
            feature_dimension: candidate.feature_dimension,
            retained_bytes: candidate.retained_bytes,
            max_predictions: spec.limits.max_predictions,
        })
    }

    pub fn members(&self) -> &[LinearRegression] {
        &self.members
    }

    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn predict(&self, row: &Vector) -> Result<f32, OlsBagError> {
        validate_row_32(row, self.feature_dimension)
            .map_err(|cause| prediction_row_validation(0, cause))?;
        let mut predictions = Vec::with_capacity(self.members.len());
        for (member, model) in self.members.iter().enumerate() {
            match model.predict(row) {
                Ok(value) if value.is_finite() => predictions.push(value),
                Ok(_) => {
                    return Err(OlsBagError::Predict(PredictionError::MemberPredict {
                        row_index: 0,
                        member_index: member,
                        cause: MemberPredictCause::NonFiniteOutput,
                        evaluated_rows: 0,
                    }));
                }
                Err(source) => {
                    return Err(OlsBagError::Predict(PredictionError::MemberPredict {
                        row_index: 0,
                        member_index: member,
                        cause: MemberPredictCause::Predictor(source),
                        evaluated_rows: 0,
                    }));
                }
            }
        }
        mean_f32(&predictions).map_err(|cause| {
            OlsBagError::Predict(PredictionError::Aggregation {
                row_index: 0,
                stage: AggregationStage::Mean,
                cause,
                evaluated_rows: 0,
            })
        })
    }

    pub fn predict_batch(
        &self,
        batch: &[Observation<Vector>],
        max_predictions: usize,
    ) -> Result<Vec<f32>, OlsBagError> {
        publish_batch::<Vector, f32, RegressionError>(
            batch,
            max_predictions,
            self.max_predictions,
            self.members.len(),
            |_row_index, row| validate_row_32(row, self.feature_dimension),
            |_row_index, member_index, row| self.members[member_index].predict(row),
            |value| value.is_finite(),
            mean_f32,
        )
        .map_err(OlsBagError::Predict)
    }
}

fn prediction_row_validation(row_index: usize, cause: RowError) -> OlsBagError {
    OlsBagError::Predict(PredictionError::RowValidation {
        row_index,
        cause,
        evaluated_rows: row_index,
    })
}
