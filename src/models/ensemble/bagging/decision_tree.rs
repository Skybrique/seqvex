//! Decision Tree bags for Profiles A, B, and C.

use std::fmt;

use crate::foundation::numerical::{Vector, Vector64};
use crate::foundation::observation::Observation;
use crate::models::trees::decision_tree::{
    DecisionTree, DecisionTree64, DecisionTreeError, TreeFitConfig, TreeFitError,
    TreeServingConversionError, TreeUnderflowPolicy, fit_tree_f32, fit_tree_f64,
};
use crate::preprocessing::sampling::SampleUnit;

use super::{
    AggregationStage, BaggingError, BaggingSpec, MemberPredictCause, PredictionError, RowError,
    assemble_candidate, mean::mean_f32, mean::mean_f64, publish_batch, retain_tree32,
    retain_tree64, validate_row_32, validate_row_64, validate_units,
};

/// Tree-bag fit or prediction failure.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionTreeBagError {
    Bagging(BaggingError),
    MemberFit {
        member: usize,
        source: TreeFitError,
    },
    MemberConversion {
        member: usize,
        source: TreeServingConversionError,
    },
    Predict(PredictionError<DecisionTreeError>),
}

impl fmt::Display for DecisionTreeBagError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bagging(error) => write!(formatter, "{error}"),
            Self::MemberFit { member, source } => {
                write!(formatter, "member {member} fit failed: {source}")
            }
            Self::MemberConversion { member, source } => {
                write!(formatter, "member {member} conversion failed: {source}")
            }
            Self::Predict(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for DecisionTreeBagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bagging(error) => Some(error),
            Self::MemberFit { source, .. } => Some(source),
            Self::MemberConversion { source, .. } => Some(source),
            Self::Predict(_) => None,
        }
    }
}

impl From<BaggingError> for DecisionTreeBagError {
    fn from(error: BaggingError) -> Self {
        Self::Bagging(error)
    }
}

/// Native-f64 Decision Tree bag (Profile A).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTreeBag64 {
    members: Vec<DecisionTree64>,
    feature_dimension: usize,
    retained_bytes: usize,
    max_predictions: usize,
}

impl DecisionTreeBag64 {
    pub fn fit(
        features: &[Vector64],
        targets: &[f64],
        units: &[SampleUnit],
        spec: &BaggingSpec,
        config: &TreeFitConfig,
    ) -> Result<Self, DecisionTreeBagError> {
        validate_units(units, features.len(), targets.len(), spec, true, false)?;
        let candidate = assemble_candidate(
            units,
            spec,
            DecisionTreeBagError::Bagging,
            |member, rows| {
                let fit = fit_tree_f64(features, targets, rows, config)
                    .map_err(|source| DecisionTreeBagError::MemberFit { member, source })?;
                let predictor = fit.into_predictor();
                let mut bytes = 0;
                retain_tree64(&mut bytes, &predictor).map_err(DecisionTreeBagError::Bagging)?;
                Ok((predictor, bytes))
            },
            DecisionTree64::feature_dimension,
            |member, expected, actual| {
                DecisionTreeBagError::Bagging(BaggingError::MemberDimensionMismatch {
                    member,
                    expected,
                    actual,
                })
            },
            |requested, limit| {
                DecisionTreeBagError::Bagging(BaggingError::RetainedBytesExceeded {
                    requested,
                    limit,
                })
            },
        )?;
        Ok(Self {
            members: candidate.members,
            feature_dimension: candidate.feature_dimension,
            retained_bytes: candidate.retained_bytes,
            max_predictions: spec.limits.max_predictions,
        })
    }

    pub fn members(&self) -> &[DecisionTree64] {
        &self.members
    }

    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn predict(&self, row: &Vector64) -> Result<f64, DecisionTreeBagError> {
        validate_row_64(row, self.feature_dimension)
            .map_err(|cause| prediction_row_validation(0, cause))?;
        let mut predictions = Vec::with_capacity(self.members.len());
        for (member, model) in self.members.iter().enumerate() {
            match model.predict(row) {
                Ok(value) if value.is_finite() => predictions.push(value),
                Ok(_) => {
                    return Err(DecisionTreeBagError::Predict(
                        PredictionError::MemberPredict {
                            row_index: 0,
                            member_index: member,
                            cause: MemberPredictCause::NonFiniteOutput,
                            evaluated_rows: 0,
                        },
                    ));
                }
                Err(source) => {
                    return Err(DecisionTreeBagError::Predict(
                        PredictionError::MemberPredict {
                            row_index: 0,
                            member_index: member,
                            cause: MemberPredictCause::Predictor(source),
                            evaluated_rows: 0,
                        },
                    ));
                }
            }
        }
        mean_f64(&predictions).map_err(|cause| {
            DecisionTreeBagError::Predict(PredictionError::Aggregation {
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
    ) -> Result<Vec<f64>, DecisionTreeBagError> {
        publish_batch::<Vector64, f64, DecisionTreeError>(
            batch,
            max_predictions,
            self.max_predictions,
            self.members.len(),
            |_row_index, row| validate_row_64(row, self.feature_dimension),
            |_row_index, member_index, row| self.members[member_index].predict(row),
            |value| value.is_finite(),
            mean_f64,
        )
        .map_err(DecisionTreeBagError::Predict)
    }
}

/// f32-serving Decision Tree bag (Profiles B and C).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTreeBag {
    members: Vec<DecisionTree>,
    feature_dimension: usize,
    retained_bytes: usize,
    max_predictions: usize,
}

impl DecisionTreeBag {
    /// Profile B: f64 member fits, checked f32 conversion.
    pub fn fit_from_f64(
        features: &[Vector64],
        targets: &[f64],
        units: &[SampleUnit],
        spec: &BaggingSpec,
        config: &TreeFitConfig,
        policy: TreeUnderflowPolicy,
    ) -> Result<Self, DecisionTreeBagError> {
        validate_units(units, features.len(), targets.len(), spec, true, false)?;
        let candidate = assemble_candidate(
            units,
            spec,
            DecisionTreeBagError::Bagging,
            |member, rows| {
                let fit = fit_tree_f64(features, targets, rows, config)
                    .map_err(|source| DecisionTreeBagError::MemberFit { member, source })?;
                let predictor = fit
                    .to_serving_f32(policy)
                    .map_err(|source| DecisionTreeBagError::MemberConversion { member, source })?;
                let mut bytes = 0;
                retain_tree32(&mut bytes, &predictor).map_err(DecisionTreeBagError::Bagging)?;
                Ok((predictor, bytes))
            },
            DecisionTree::feature_dimension,
            |member, expected, actual| {
                DecisionTreeBagError::Bagging(BaggingError::MemberDimensionMismatch {
                    member,
                    expected,
                    actual,
                })
            },
            |requested, limit| {
                DecisionTreeBagError::Bagging(BaggingError::RetainedBytesExceeded {
                    requested,
                    limit,
                })
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
        config: &TreeFitConfig,
    ) -> Result<Self, DecisionTreeBagError> {
        validate_units(units, features.len(), targets.len(), spec, true, true)?;
        let candidate = assemble_candidate(
            units,
            spec,
            DecisionTreeBagError::Bagging,
            |member, rows| {
                let fit = fit_tree_f32(features, targets, rows, config)
                    .map_err(|source| DecisionTreeBagError::MemberFit { member, source })?;
                let predictor = fit.into_predictor();
                let mut bytes = 0;
                retain_tree32(&mut bytes, &predictor).map_err(DecisionTreeBagError::Bagging)?;
                Ok((predictor, bytes))
            },
            DecisionTree::feature_dimension,
            |member, expected, actual| {
                DecisionTreeBagError::Bagging(BaggingError::MemberDimensionMismatch {
                    member,
                    expected,
                    actual,
                })
            },
            |requested, limit| {
                DecisionTreeBagError::Bagging(BaggingError::RetainedBytesExceeded {
                    requested,
                    limit,
                })
            },
        )?;
        Ok(Self {
            members: candidate.members,
            feature_dimension: candidate.feature_dimension,
            retained_bytes: candidate.retained_bytes,
            max_predictions: spec.limits.max_predictions,
        })
    }

    pub fn members(&self) -> &[DecisionTree] {
        &self.members
    }

    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn predict(&self, row: &Vector) -> Result<f32, DecisionTreeBagError> {
        validate_row_32(row, self.feature_dimension)
            .map_err(|cause| prediction_row_validation(0, cause))?;
        let mut predictions = Vec::with_capacity(self.members.len());
        for (member, model) in self.members.iter().enumerate() {
            match model.predict(row) {
                Ok(value) if value.is_finite() => predictions.push(value),
                Ok(_) => {
                    return Err(DecisionTreeBagError::Predict(
                        PredictionError::MemberPredict {
                            row_index: 0,
                            member_index: member,
                            cause: MemberPredictCause::NonFiniteOutput,
                            evaluated_rows: 0,
                        },
                    ));
                }
                Err(source) => {
                    return Err(DecisionTreeBagError::Predict(
                        PredictionError::MemberPredict {
                            row_index: 0,
                            member_index: member,
                            cause: MemberPredictCause::Predictor(source),
                            evaluated_rows: 0,
                        },
                    ));
                }
            }
        }
        mean_f32(&predictions).map_err(|cause| {
            DecisionTreeBagError::Predict(PredictionError::Aggregation {
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
    ) -> Result<Vec<f32>, DecisionTreeBagError> {
        publish_batch::<Vector, f32, DecisionTreeError>(
            batch,
            max_predictions,
            self.max_predictions,
            self.members.len(),
            |_row_index, row| validate_row_32(row, self.feature_dimension),
            |_row_index, member_index, row| self.members[member_index].predict(row),
            |value| value.is_finite(),
            mean_f32,
        )
        .map_err(DecisionTreeBagError::Predict)
    }
}

fn prediction_row_validation(row_index: usize, cause: RowError) -> DecisionTreeBagError {
    DecisionTreeBagError::Predict(PredictionError::RowValidation {
        row_index,
        cause,
        evaluated_rows: row_index,
    })
}
