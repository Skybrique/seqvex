//! Legacy OLS candidate acceptance checks over selected deployed predictions.

use super::super::predict::{RegressionError, predict_checked_borrowed};
use super::legacy::FitResult;
use crate::foundation::numerical::Vector;

/// Summary of an explicit check over selected deployed predictions.
///
/// This is a diagnostic result, not a quality threshold, deployment certificate
/// or installation token. Repeated selection indices contribute repeatedly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceCheckReport {
    /// Number of selected positions checked, including repeated indices.
    pub observations_checked: usize,
    /// Ordered f64 RSS of checked f32 predictions against supplied f32 targets.
    pub rss_of_checked_predictions_f64: f64,
}

/// Failure reported by [`FitResult::check_acceptance`].
///
/// Selection-position fields refer to the zero-based position in the supplied
/// acceptance-index slice; `index` fields refer to the original row. Only the
/// [`Prediction`](Self::Prediction) variant has an underlying error source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceCheckError {
    /// A retained feature coefficient is not finite.
    NonFiniteWeight {
        /// Zero-based coefficient index.
        index: usize,
    },
    /// The retained bias is not finite.
    NonFiniteBias,
    /// The feature row count differs from the target count.
    FeatureTargetLengthMismatch {
        /// Number of feature rows.
        features: usize,
        /// Number of targets.
        targets: usize,
    },
    /// No acceptance rows were selected.
    EmptySelection,
    /// A selection index is outside the supplied rows.
    IndexOutOfRange {
        /// Zero-based position in the acceptance-index slice.
        position: usize,
        /// Supplied original row index.
        index: usize,
        /// Number of available rows.
        rows: usize,
    },
    /// A selected row has a different feature width from the fitted model.
    RowDimensionMismatch {
        /// Zero-based position in the acceptance-index slice.
        position: usize,
        /// Supplied original row index.
        index: usize,
        /// Fitted feature dimension.
        expected: usize,
        /// Selected row's actual feature dimension.
        actual: usize,
    },
    /// A selected feature or target is not finite.
    NonFiniteSelectedValue {
        /// Zero-based position in the acceptance-index slice.
        position: usize,
        /// Supplied original row index.
        index: usize,
        /// Feature component or target that was not finite.
        field: SelectedField,
    },
    /// One of the named stored RSS diagnostics is not finite.
    NonFiniteStoredDiagnostic {
        /// Stored diagnostic that was not finite.
        diagnostic: StoredDiagnostic,
    },
    /// Checked prediction rejected this selected row.
    Prediction {
        /// Zero-based position in the acceptance-index slice.
        position: usize,
        /// Supplied original row index.
        index: usize,
        /// Original checked-prediction failure.
        error: RegressionError,
    },
}

/// Field of a selected observation that failed finiteness validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedField {
    /// Feature component at the given zero-based column.
    Feature {
        /// Zero-based feature column.
        column: usize,
    },
    /// The selected scalar target.
    Target,
}

/// Stored OLS RSS diagnostic checked by candidate admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredDiagnostic {
    /// RSS from the fitted f64 reference solution.
    RssF64Reference,
    /// RSS from the deployed f32 predictor.
    RssF32Predictor,
}

impl std::fmt::Display for AcceptanceCheckError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteWeight { index } => {
                write!(formatter, "non-finite OLS weight at index {index}")
            }
            Self::NonFiniteBias => formatter.write_str("non-finite OLS bias"),
            Self::FeatureTargetLengthMismatch { features, targets } => write!(
                formatter,
                "acceptance feature row count {features} does not match target count {targets}"
            ),
            Self::EmptySelection => formatter.write_str("acceptance selection is empty"),
            Self::IndexOutOfRange {
                position,
                index,
                rows,
            } => write!(
                formatter,
                "acceptance index {index} at position {position} is out of range for {rows} rows"
            ),
            Self::RowDimensionMismatch {
                position,
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "acceptance row {index} at position {position} has width {actual}; expected {expected}"
            ),
            Self::NonFiniteSelectedValue {
                position,
                index,
                field: SelectedField::Feature { column },
            } => write!(
                formatter,
                "non-finite feature column {column} at acceptance position {position}, row {index}"
            ),
            Self::NonFiniteSelectedValue {
                position,
                index,
                field: SelectedField::Target,
            } => write!(
                formatter,
                "non-finite target at acceptance position {position}, row {index}"
            ),
            Self::NonFiniteStoredDiagnostic { diagnostic } => {
                let name = match diagnostic {
                    StoredDiagnostic::RssF64Reference => "rss_f64_reference",
                    StoredDiagnostic::RssF32Predictor => "rss_f32_predictor",
                };
                write!(formatter, "stored fit diagnostic {name} is non-finite")
            }
            Self::Prediction {
                position, index, ..
            } => write!(
                formatter,
                "checked OLS prediction failed at acceptance position {position}, row {index}"
            ),
        }
    }
}

impl std::error::Error for AcceptanceCheckError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Prediction { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl FitResult {
    /// Checks this fitted candidate against an explicit nonempty row selection.
    ///
    /// This method borrows the result and input rows, leaving the candidate
    /// available after either success or failure. It checks parameters, array
    /// lengths, selection, selected row shapes/values, the two stored RSS
    /// diagnostics, then checked predictions, in that order. Selection positions
    /// are processed in supplied order; repeated indices are checked and counted
    /// repeatedly. Only selected rows are inspected.
    ///
    /// Within those phases, weights are checked in index order before bias;
    /// length mismatch precedes empty selection; bounds precede row shapes;
    /// all selected shapes precede selected numeric values; feature columns
    /// precede each target; `rss_f64_reference` precedes `rss_f32_predictor`;
    /// predictions are then checked in selection order.
    ///
    /// The report's RSS is a sequential `f64` accumulation of residuals from the
    /// existing checked `f32` predictor. Its finite range is established for
    /// finite f32 predictions/targets on supported binary64 round-to-nearest
    /// arithmetic; it is not an accuracy or quality guarantee. The method does
    /// not revalidate rank, inspect unselected rows, certify future inputs,
    /// install the candidate or establish application acceptance.
    ///
    /// # Errors
    ///
    /// Returns [`AcceptanceCheckError`] in the precedence described above.
    pub fn check_acceptance(
        &self,
        features: &[Vector],
        targets: &[f32],
        acceptance_indices: &[usize],
    ) -> Result<AcceptanceCheckReport, AcceptanceCheckError> {
        for (index, &weight) in self.weights().as_slice().iter().enumerate() {
            if !weight.is_finite() {
                return Err(AcceptanceCheckError::NonFiniteWeight { index });
            }
        }
        if !self.intercept().is_finite() {
            return Err(AcceptanceCheckError::NonFiniteBias);
        }

        if features.len() != targets.len() {
            return Err(AcceptanceCheckError::FeatureTargetLengthMismatch {
                features: features.len(),
                targets: targets.len(),
            });
        }
        if acceptance_indices.is_empty() {
            return Err(AcceptanceCheckError::EmptySelection);
        }
        for (position, &index) in acceptance_indices.iter().enumerate() {
            if index >= features.len() {
                return Err(AcceptanceCheckError::IndexOutOfRange {
                    position,
                    index,
                    rows: features.len(),
                });
            }
        }

        let expected = self.weights().len();
        for (position, &index) in acceptance_indices.iter().enumerate() {
            let actual = features[index].len();
            if actual != expected {
                return Err(AcceptanceCheckError::RowDimensionMismatch {
                    position,
                    index,
                    expected,
                    actual,
                });
            }
        }
        for (position, &index) in acceptance_indices.iter().enumerate() {
            for (column, &value) in features[index].as_slice().iter().enumerate() {
                if !value.is_finite() {
                    return Err(AcceptanceCheckError::NonFiniteSelectedValue {
                        position,
                        index,
                        field: SelectedField::Feature { column },
                    });
                }
            }
            if !targets[index].is_finite() {
                return Err(AcceptanceCheckError::NonFiniteSelectedValue {
                    position,
                    index,
                    field: SelectedField::Target,
                });
            }
        }

        if !self.diagnostics().rss_f64_reference.is_finite() {
            return Err(AcceptanceCheckError::NonFiniteStoredDiagnostic {
                diagnostic: StoredDiagnostic::RssF64Reference,
            });
        }
        if !self.diagnostics().rss_f32_predictor.is_finite() {
            return Err(AcceptanceCheckError::NonFiniteStoredDiagnostic {
                diagnostic: StoredDiagnostic::RssF32Predictor,
            });
        }

        let mut rss = 0.0f64;
        for (position, &index) in acceptance_indices.iter().enumerate() {
            let prediction =
                predict_checked_borrowed(self.weights(), self.intercept(), &features[index])
                    .map_err(|error| AcceptanceCheckError::Prediction {
                        position,
                        index,
                        error,
                    })?;
            let residual = f64::from(prediction) - f64::from(targets[index]);
            let term = residual * residual;
            rss += term;
        }

        Ok(AcceptanceCheckReport {
            observations_checked: acceptance_indices.len(),
            rss_of_checked_predictions_f64: rss,
        })
    }
}
