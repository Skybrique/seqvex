//! Ordinary least-squares **fitting** provides three explicit precision paths
//! alongside the original compatibility fitter. [`fit_f64`] is native f64
//! learning for profile A; [`FitResult64::to_serving_f32`] converts that same
//! fit into separately owned checked f32 serving for B; [`fit_f32`] performs
//! learning and training diagnostics natively in f32 for C. The original
//! [`fit`] accepts f32 values, widens them for f64 solving, then retains checked
//! f32 parameters and legacy diagnostics.
//!
//! Each call is a dense, selected-row dataset batch fit. A new candidate is a
//! separate refit; OLS does not incrementally update a previous result. There
//! are no epochs, learning rates, automatic scheduling, or generic training
//! framework.
//!
//! # Solver
//!
//! The numerical backend is `faer`'s low-level **column-pivoted Householder
//! QR** (`=0.24.4`, features `std` + `linalg`, `Par::Seq`, no Rayon), used
//! through a small private adapter so the backend can be replaced later without
//! changing the public model API. The adapter follows the compatibility-tested
//! route: caller-owned `A`/`rhs`/`Q_coeff`/permutation buffers, fallible
//! reservation before initialization, a single combined scratch buffer sized by
//! `StackReq::any_of` and reused for the sequential factor→solve, rank read from
//! the factorized `R` diagonal **before** solving, and pivot restoration by the
//! solve routine.
//!
//! # Rank
//!
//! Rank is a **numerical indicator**, not a condition number. The policy is the
//! relative-pivot form `tau = c_rank · max(N, p) · u64 · |R₀₀|` with
//! `u64 = 2^-53` (the `f64` unit roundoff, not `f64::EPSILON = 2^-52`); a pivot
//! whose magnitude does not exceed `tau` does not count toward the rank. A rank
//! below `p = features + intercept` is rejected before solving
//! ([`FitError::RankDeficient`]). An exactly full-rank but nearly dependent
//! matrix may legitimately fall outside this threshold.
//!
//! # Conversion
//!
//! Fitted `f64` coefficients are converted to the existing `f32` predictor with
//! checked conversion: non-finite `f64` coefficients and `f64→f32` overflow are
//! rejected; a nonzero coefficient (or the bias) that becomes exactly zero is
//! rejected by default ([`ConversionCause`]) unless the caller opts in with
//! [`UnderflowPolicy::AllowLoss`]. Legitimate representable subnormal values are
//! preserved. The formally certified cross-precision error-envelope project is
//! **deferred**; no interval arithmetic, compensated dot product, or
//! proof-dependent acceptance gate is implemented here.
//!
//! # Diagnostics
//!
//! Diagnostics are computed on the **selected training rows only** (held-out
//! targets are never read). [`FitDiagnostics::rss_f64_reference`] is a rounded
//! `f64` quantity computed with the fitted `f64` coefficients on the original
//! `f32` features; it is not an exact-real quantity.
//! [`FitDiagnostics::rss_f32_predictor`] is computed from the actual deployed
//! `f32` predictor outputs.

use super::artifact::{self, ArtifactError, TrainedModel, TrainingMeta};
use super::predict::{
    LinearRegression, LinearRegression64, RegressionError, predict_checked_borrowed,
};
use crate::foundation::numerical::{Vector, Vector64};

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::col_pivoting::factor::{
    ColPivQrParams, qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::qr::col_pivoting::solve::{solve_lstsq_in_place, solve_lstsq_in_place_scratch};
use faer::{Mat, Par, Spec, TryReserveError};

/// Default requested-workspace budget (256 MiB), as in the accepted design.
pub const DEFAULT_MAX_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

/// `f64` unit roundoff `u64 = 2^-53`.
const U64_ROUNDOFF: f64 = f64::from_bits((1023u64 - 53) << 52);

/// `f32` unit roundoff `u32 = 2^-24`.
const U32_ROUNDOFF: f32 = f32::from_bits((127u32 - 24) << 23);

/// Which fallible reservation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationSource {
    /// A `faer::Mat::try_reserve` reservation failed.
    Faer,
    /// A `faer::dyn_stack::MemBuffer::try_new` reservation failed.
    DynStack,
    /// A standard `Vec::try_reserve_exact` reservation failed.
    Std,
}

/// Why a fitted coefficient could not be deployed to `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionCause {
    /// A nonzero feature coefficient became exactly zero when rounded to `f32`.
    CoefficientUnderflow,
    /// A nonzero bias became exactly zero when rounded to `f32`.
    BiasUnderflow,
    /// A fitted `f64` coefficient is not finite, or overflows `f32`.
    Overflow,
}

/// Caller policy for a nonzero `f64` value that rounds to a zero `f32` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderflowPolicy {
    /// Reject the fit (default).
    Reject,
    /// Accept the loss and deploy the zero value.
    AllowLoss,
}

/// Configuration for a single OLS fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitConfig {
    /// Include an intercept term (leading ones column).
    pub intercept: bool,
    /// Requested **workspace** budget in bytes (not a physical-memory ceiling).
    pub max_workspace_bytes: usize,
    /// Optional maximum feature count.
    pub max_features: Option<usize>,
    /// Rank policy constant `c_rank` in `tau = c_rank · max(N, p) · u64 · |R₀₀|`.
    pub rank_threshold: f64,
    /// Policy for a nonzero coefficient/bias that rounds to zero.
    pub on_underflow: UnderflowPolicy,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            intercept: true,
            max_workspace_bytes: DEFAULT_MAX_WORKSPACE_BYTES,
            max_features: None,
            rank_threshold: 1.0,
            on_underflow: UnderflowPolicy::Reject,
        }
    }
}

/// Diagnostics for a completed fit, computed on the selected training rows.
#[derive(Debug, Clone, PartialEq)]
pub struct FitDiagnostics {
    /// Number of selected training rows `N`.
    pub rows: usize,
    /// Feature count `d`.
    pub features: usize,
    /// Whether an intercept column was fitted.
    pub intercept: bool,
    /// Numerical rank (count of pivots above the threshold).
    pub rank: usize,
    /// Required rank `p = d + intercept`.
    pub required_rank: usize,
    /// The absolute rank threshold `tau` applied to the R diagonal.
    pub rank_threshold: f64,
    /// Rounded `f64` reference residual sum of squares (fitted `f64`
    /// coefficients, original `f32` features) on the selected training rows.
    pub rss_f64_reference: f64,
    /// Residual sum of squares from the deployed `f32` predictor on the
    /// selected training rows.
    pub rss_f32_predictor: f64,
    /// Requested workspace bytes (budget accounting; not measured memory).
    pub requested_workspace_bytes: usize,
}

/// A successful fit: learned `f32` coefficients, intercept, and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    coefficients: Vector,
    intercept: f32,
    diagnostics: FitDiagnostics,
}

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
        for (index, &weight) in self.coefficients.as_slice().iter().enumerate() {
            if !weight.is_finite() {
                return Err(AcceptanceCheckError::NonFiniteWeight { index });
            }
        }
        if !self.intercept.is_finite() {
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

        let expected = self.coefficients.len();
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

        if !self.diagnostics.rss_f64_reference.is_finite() {
            return Err(AcceptanceCheckError::NonFiniteStoredDiagnostic {
                diagnostic: StoredDiagnostic::RssF64Reference,
            });
        }
        if !self.diagnostics.rss_f32_predictor.is_finite() {
            return Err(AcceptanceCheckError::NonFiniteStoredDiagnostic {
                diagnostic: StoredDiagnostic::RssF32Predictor,
            });
        }

        let mut rss = 0.0f64;
        for (position, &index) in acceptance_indices.iter().enumerate() {
            let prediction =
                predict_checked_borrowed(&self.coefficients, self.intercept, &features[index])
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

    /// The learned feature coefficients (without the intercept).
    pub fn weights(&self) -> &Vector {
        &self.coefficients
    }

    /// The learned intercept (zero when no intercept was fitted).
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// The training diagnostics.
    pub fn diagnostics(&self) -> &FitDiagnostics {
        &self.diagnostics
    }

    /// Moves the learned parameters into an immutable predictor.
    #[must_use]
    pub fn into_predictor(self) -> LinearRegression {
        LinearRegression::new(self.coefficients, self.intercept)
            .expect("fit produces a validated, non-empty, finite predictor")
    }

    /// Builds a predictor by cloning the learned parameters.
    pub fn predictor(&self) -> LinearRegression {
        LinearRegression::new(self.coefficients.clone(), self.intercept)
            .expect("fit produces a validated, non-empty, finite predictor")
    }

    /// Moves the learned parameters into an immutable [`TrainedModel`] carrying
    /// the derived [`super::artifact::ArtifactMetadata`].
    ///
    /// Returned as [`ArtifactError`] because the checked `u32` conversions of
    /// `N`/`d` occur here; parameter finiteness was guaranteed by [`fit`].
    pub fn into_trained(self, meta: &TrainingMeta) -> Result<TrainedModel, ArtifactError> {
        let limit = u32::MAX as usize;
        let rows =
            u32::try_from(self.diagnostics.rows).map_err(|_| ArtifactError::Capacity { limit })?;
        let features = u32::try_from(self.diagnostics.features)
            .map_err(|_| ArtifactError::Capacity { limit })?;
        let intercept_present = self.diagnostics.intercept;
        let coefficients = self.coefficients;
        let bias = self.intercept;
        artifact::build_trained(meta, coefficients, bias, rows, features, intercept_present)
    }
}

/// Failure classes reported by [`fit`].
#[derive(Debug, Clone, PartialEq)]
pub enum FitError {
    /// `features.len() != targets.len()`.
    RowTargetMismatch {
        /// Number of feature rows.
        features: usize,
        /// Number of targets.
        targets: usize,
    },
    /// No training rows were selected.
    TooFewRows,
    /// A training index is out of range.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The number of available rows.
        rows: usize,
    },
    /// The selected rows have zero feature width.
    ZeroFeatures,
    /// The feature width exceeds the configured maximum.
    TooManyFeatures {
        /// Observed feature width.
        features: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Selected feature rows do not all share one width.
    RaggedFeatures {
        /// Original row position of the offending observation.
        row: usize,
        /// Expected width.
        expected: usize,
        /// Observed width.
        actual: usize,
    },
    /// The rank-policy multiplier is negative or not finite.
    ///
    /// A configuration error, not a `RankDeficient` outcome: the supplied
    /// `rank_threshold` cannot define a supported rank policy. Zero remains a
    /// valid multiplier (it still rejects exact zero pivots).
    InvalidRankThreshold {
        /// The rejected `rank_threshold` value.
        value: f64,
    },
    /// A checked size calculation overflowed `usize`.
    SizeOverflow {
        /// Row count.
        rows: usize,
        /// Column count.
        features: usize,
    },
    /// The solver scratch layout is invalid (`StackReq` overflow/invalid).
    ScratchLayout,
    /// The requested workspace exceeds the configured budget.
    Capacity {
        /// Requested workspace bytes.
        requested: usize,
        /// Configured limit.
        limit: usize,
    },
    /// A fallible reservation failed.
    AllocationFailed {
        /// Buffer label.
        buffer: &'static str,
        /// Requested bytes.
        requested_bytes: usize,
        /// Reservation source.
        source: AllocationSource,
    },
    /// Duplicate training indices were supplied.
    DuplicateIndex {
        /// The repeated index.
        index: usize,
    },
    /// A selected training feature or target is not finite.
    NonFiniteTrainingValue {
        /// Original row position.
        row: usize,
        /// Feature column, or `features` for the target.
        column: usize,
    },
    /// The numerical rank is below the required rank.
    RankDeficient {
        /// Computed rank.
        rank: usize,
        /// Required rank.
        required: usize,
    },
    /// The solver produced a non-finite coefficient, or failed internally.
    NumericalFailure,
    /// A fitted coefficient could not be converted to the `f32` predictor.
    ConversionRejected(ConversionCause),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RowTargetMismatch { features, targets } => write!(
                formatter,
                "feature row count {features} does not match target count {targets}"
            ),
            Self::TooFewRows => formatter.write_str("no training rows were selected"),
            Self::IndexOutOfRange { index, rows } => {
                write!(
                    formatter,
                    "training index {index} is out of range for {rows} rows"
                )
            }
            Self::ZeroFeatures => {
                formatter.write_str("selected training rows have zero feature columns")
            }
            Self::TooManyFeatures { features, max } => write!(
                formatter,
                "feature count {features} exceeds configured maximum {max}"
            ),
            Self::RaggedFeatures {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "feature row {row} has width {actual}; expected {expected}"
            ),
            Self::InvalidRankThreshold { value } => {
                write!(formatter, "invalid OLS rank threshold {value}")
            }
            Self::SizeOverflow { rows, features } => write!(
                formatter,
                "OLS size calculation overflowed for {rows} rows and {features} design-matrix columns"
            ),
            Self::ScratchLayout => formatter.write_str("OLS solver scratch layout is invalid"),
            Self::Capacity { requested, limit } => write!(
                formatter,
                "requested OLS workspace {requested} bytes exceeds limit {limit} bytes"
            ),
            Self::AllocationFailed {
                buffer,
                requested_bytes,
                source,
            } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for OLS buffer {buffer} ({})",
                allocation_source_name(*source)
            ),
            Self::DuplicateIndex { index } => {
                write!(
                    formatter,
                    "training index {index} was supplied more than once"
                )
            }
            Self::NonFiniteTrainingValue { row, column } => write!(
                formatter,
                "non-finite training value at row {row}, column {column}"
            ),
            Self::RankDeficient { rank, required } => {
                write!(
                    formatter,
                    "OLS numerical rank {rank} is below required rank {required}"
                )
            }
            Self::NumericalFailure => formatter.write_str("OLS numerical calculation failed"),
            Self::ConversionRejected(cause) => write!(
                formatter,
                "OLS coefficient conversion rejected: {}",
                conversion_cause_name(*cause)
            ),
        }
    }
}

fn allocation_source_name(source: AllocationSource) -> &'static str {
    match source {
        AllocationSource::Faer => "faer allocator",
        AllocationSource::DynStack => "dynamic stack allocator",
        AllocationSource::Std => "standard allocator",
    }
}

fn conversion_cause_name(cause: ConversionCause) -> &'static str {
    match cause {
        ConversionCause::CoefficientUnderflow => "coefficient underflow",
        ConversionCause::BiasUnderflow => "bias underflow",
        ConversionCause::Overflow => "overflow",
    }
}

impl std::error::Error for FitError {}

/// Fits dense, unweighted OLS with one scalar target over the selected training
/// rows.
///
/// Row `k` for `k in 0..train.len()` uses `features[train[k]]` and
/// `targets[train[k]]`; held-out rows and their targets are never read.
pub fn fit(
    features: &[Vector],
    targets: &[f32],
    train: &[usize],
    config: &FitConfig,
) -> Result<FitResult, FitError> {
    // 0. Full-array length equality, from lengths alone.
    if features.len() != targets.len() {
        return Err(FitError::RowTargetMismatch {
            features: features.len(),
            targets: targets.len(),
        });
    }
    let rows = train.len();
    if rows == 0 {
        return Err(FitError::TooFewRows);
    }
    // 1. Selected-index range.
    for &index in train {
        if index >= features.len() {
            return Err(FitError::IndexOutOfRange {
                index,
                rows: features.len(),
            });
        }
    }
    // 2. Feature width and optional maximum.
    let d = features[train[0]].len();
    if d == 0 {
        return Err(FitError::ZeroFeatures);
    }
    if let Some(max) = config.max_features {
        if d > max {
            return Err(FitError::TooManyFeatures { features: d, max });
        }
    }
    // 3. Rectangularity across selected rows (lengths only).
    for &index in train {
        let actual = features[index].len();
        if actual != d {
            return Err(FitError::RaggedFeatures {
                row: index,
                expected: d,
                actual,
            });
        }
    }

    // 4. Rank-policy configuration. A negative or non-finite multiplier cannot
    //    define a supported policy and is rejected as a configuration error
    //    (never reported as `RankDeficient`). This follows the length/range/
    //    feature-shape checks above and precedes size/layout, budget and
    //    allocation failures below. Zero is permitted and still rejects exact
    //    zero pivots; the threshold is a numerical indicator, not certified
    //    algebraic rank.
    if !config.rank_threshold.is_finite() || config.rank_threshold < 0.0 {
        return Err(FitError::InvalidRankThreshold {
            value: config.rank_threshold,
        });
    }

    let p = d + usize::from(config.intercept);
    let size = rows.min(p);
    let block = recommended_block_size::<f64>(rows, p);

    // 5. Checked size products and scratch layout, before any allocation.
    //
    // Requested-workspace accounting: the sum of every **distinct owned backing
    // buffer** whose lifetime overlaps the fit peak. `d <= p`, so parameter
    // buffers are conservatively budgeted at `p` elements. This is a logical
    // request, distinct from allocator traffic, physical resident memory and
    // allocator padding.
    let f64_bytes = std::mem::size_of::<f64>();
    let f32_bytes = std::mem::size_of::<f32>();
    let usize_bytes = std::mem::size_of::<usize>();
    let size_overflow = || FitError::SizeOverflow { rows, features: p };
    let a_bytes = rows
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(f64_bytes))
        .ok_or_else(size_overflow)?;
    let q_bytes = block
        .checked_mul(size)
        .and_then(|cells| cells.checked_mul(f64_bytes))
        .ok_or_else(size_overflow)?;
    let rhs_bytes = rows.checked_mul(f64_bytes).ok_or_else(size_overflow)?;
    // Two permutation vectors (`col_perm_fwd` + `col_perm_inv`).
    let perm_bytes = p
        .checked_mul(2)
        .and_then(|n| n.checked_mul(usize_bytes))
        .ok_or_else(size_overflow)?;
    // `f64` solution vector, live through solve and diagnostics.
    let solution_bytes = p.checked_mul(f64_bytes).ok_or_else(size_overflow)?;
    // Converted-parameter buffer (`weights32` push storage).
    let converted_bytes = p.checked_mul(f32_bytes).ok_or_else(size_overflow)?;
    // Retained coefficient `Vector` moved into `FitResult` (one distinct copy).
    let retained_bytes = p.checked_mul(f32_bytes).ok_or_else(size_overflow)?;
    // Transient diagnostics predictor coefficient `Vector` (a second copy).
    let diagnostics_bytes = p.checked_mul(f32_bytes).ok_or_else(size_overflow)?;
    let uniq_bytes = features.len().div_ceil(8);

    let scratch = StackReq::any_of(&[
        qr_in_place_scratch::<usize, f64>(rows, p, block, Par::Seq, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f64>(rows, p, block, 1, Par::Seq),
    ]);
    if scratch.layout().is_err() || scratch.align_bytes() == 0 {
        return Err(FitError::ScratchLayout);
    }
    let scratch_bytes = scratch.unaligned_bytes_required();

    let requested = a_bytes
        .checked_add(q_bytes)
        .and_then(|v| v.checked_add(rhs_bytes))
        .and_then(|v| v.checked_add(perm_bytes))
        .and_then(|v| v.checked_add(solution_bytes))
        .and_then(|v| v.checked_add(converted_bytes))
        .and_then(|v| v.checked_add(retained_bytes))
        .and_then(|v| v.checked_add(diagnostics_bytes))
        .and_then(|v| v.checked_add(uniq_bytes))
        .and_then(|v| v.checked_add(scratch_bytes))
        .ok_or_else(size_overflow)?;

    // 6. Requested-workspace budget.
    if requested > config.max_workspace_bytes {
        return Err(FitError::Capacity {
            requested,
            limit: config.max_workspace_bytes,
        });
    }

    // 7. Fallible allocation phase. Matrix, scratch, permutation, solution and
    //    bitset reservations are fallible and share the requested accounting.
    //    The retained/diagnostics `Vector` copies in step 12 remain infallible
    //    (disclosed there and in the response).
    let mut a = try_matrix(rows, p).map_err(|_| FitError::AllocationFailed {
        buffer: "design matrix A",
        requested_bytes: a_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut q_coeff = try_matrix(block, size).map_err(|_| FitError::AllocationFailed {
        buffer: "Householder coefficients",
        requested_bytes: q_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut rhs = try_matrix(rows, 1).map_err(|_| FitError::AllocationFailed {
        buffer: "right-hand side",
        requested_bytes: rhs_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut col_perm_fwd = try_vec_filled(p, 0usize, "col_perm_fwd")?;
    let mut col_perm_inv = try_vec_filled(p, 0usize, "col_perm_inv")?;
    let mut solution = try_vec_filled(p, 0.0f64, "f64 solution")?;
    let mut mem = MemBuffer::try_new(scratch).map_err(|_| FitError::AllocationFailed {
        buffer: "solver scratch",
        requested_bytes: scratch_bytes,
        source: AllocationSource::DynStack,
    })?;
    let mut seen: Vec<u8> = Vec::new();
    seen.try_reserve_exact(uniq_bytes)
        .map_err(|_| FitError::AllocationFailed {
            buffer: "uniqueness bitset",
            requested_bytes: uniq_bytes,
            source: AllocationSource::Std,
        })?;
    seen.resize(uniq_bytes, 0u8);

    // 8. Duplicate selected indices, via the budgeted bitset.
    for &index in train {
        let byte = index / 8;
        let bit = 1u8 << (index % 8);
        if seen[byte] & bit != 0 {
            return Err(FitError::DuplicateIndex { index });
        }
        seen[byte] |= bit;
    }

    // 9. Finiteness of selected feature values and targets only.
    for &index in train {
        let row = features[index].as_slice();
        for (column, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(FitError::NonFiniteTrainingValue { row: index, column });
            }
        }
        if !targets[index].is_finite() {
            return Err(FitError::NonFiniteTrainingValue {
                row: index,
                column: d,
            });
        }
    }

    // 10. Fill the design matrix (leading ones column when fitting an intercept).
    for (k, &index) in train.iter().enumerate() {
        let row = features[index].as_slice();
        if config.intercept {
            a[(k, 0)] = 1.0;
            for (j, &value) in row.iter().enumerate() {
                a[(k, j + 1)] = f64::from(value);
            }
        } else {
            for (j, &value) in row.iter().enumerate() {
                a[(k, j)] = f64::from(value);
            }
        }
        rhs[(k, 0)] = f64::from(targets[index]);
    }

    // 11. Factor (sequential), rank from the R diagonal, then solve.
    let applied_tau: f64;
    {
        let (_info, col_perm) = qr_in_place(
            a.as_mut(),
            q_coeff.as_mut(),
            &mut col_perm_fwd,
            &mut col_perm_inv,
            Par::Seq,
            MemStack::new(&mut mem),
            Spec::<ColPivQrParams, f64>::default(),
        );

        let first = a[(0, 0)].abs();
        let tau = config.rank_threshold * (rows.max(p) as f64) * U64_ROUNDOFF * first;
        applied_tau = tau;
        let mut rank = 0usize;
        for k in 0..size {
            if a[(k, k)].abs() > tau {
                rank += 1;
            }
        }
        if rank < p {
            return Err(FitError::RankDeficient { rank, required: p });
        }

        solve_lstsq_in_place(
            a.as_ref(),
            q_coeff.as_ref(),
            a.as_ref(),
            col_perm,
            rhs.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        for (i, value) in solution.iter_mut().enumerate() {
            *value = rhs[(i, 0)];
        }
    }

    // Transient solver buffers are released here (scratch, A, Q_coeff, rhs).
    drop(mem);
    drop(a);
    drop(q_coeff);
    drop(rhs);
    drop(seen);
    drop(col_perm_fwd);
    drop(col_perm_inv);

    // 12. Checked conversion into the f32 predictor parameters.
    let (bias64, weights64) = if config.intercept {
        (solution[0], &solution[1..])
    } else {
        (0.0f64, &solution[..])
    };
    if !solution.iter().all(|value| value.is_finite()) {
        return Err(FitError::NumericalFailure);
    }

    let bias32 = convert_checked(bias64, config.intercept, config)?;
    let mut weights32: Vec<f32> = Vec::new();
    weights32
        .try_reserve_exact(d)
        .map_err(|_| FitError::AllocationFailed {
            buffer: "converted coefficients",
            requested_bytes: converted_bytes,
            source: AllocationSource::Std,
        })?;
    for &value in weights64 {
        weights32.push(convert_checked(value, false, config)?);
    }

    // 13. Training diagnostics on the selected rows only.
    //
    // The two `Vector::from_slice` copies below (the transient diagnostics
    // predictor and the retained result coefficients) are **infallible**
    // allocations. They are budgeted (`diagnostics_bytes` + `retained_bytes`)
    // but are not allocation-failure recoverable. Universal OOM recovery is
    // not claimed.
    let mut rss64 = 0.0f64;
    let mut rss32 = 0.0f64;
    let predictor = LinearRegression::new(Vector::from_slice(&weights32), bias32)
        .map_err(|_| FitError::NumericalFailure)?;
    for &index in train {
        let row = features[index].as_slice();
        let target = f64::from(targets[index]);

        let mut acc = 0.0f64;
        for (j, &value) in row.iter().enumerate() {
            acc += f64::from(value) * weights64[j];
        }
        acc += bias64;
        let residual64 = acc - target;
        rss64 += residual64 * residual64;

        let prediction = predictor
            .predict(&features[index])
            .map_err(|_| FitError::NumericalFailure)?;
        let residual32 = f64::from(prediction) - target;
        rss32 += residual32 * residual32;
    }

    let diagnostics = FitDiagnostics {
        rows,
        features: d,
        intercept: config.intercept,
        rank: p,
        required_rank: p,
        rank_threshold: applied_tau,
        rss_f64_reference: rss64,
        rss_f32_predictor: rss32,
        requested_workspace_bytes: requested,
    };

    Ok(FitResult {
        coefficients: Vector::from_slice(&weights32),
        intercept: bias32,
        diagnostics,
    })
}

/// Converts one fitted `f64` parameter with checked `f64→f32` conversion.
fn convert_checked(value: f64, is_bias: bool, config: &FitConfig) -> Result<f32, FitError> {
    if !value.is_finite() {
        return Err(FitError::ConversionRejected(ConversionCause::Overflow));
    }
    let converted = value as f32;
    if !converted.is_finite() {
        return Err(FitError::ConversionRejected(ConversionCause::Overflow));
    }
    if value != 0.0 && converted == 0.0 && config.on_underflow == UnderflowPolicy::Reject {
        return Err(FitError::ConversionRejected(if is_bias {
            ConversionCause::BiasUnderflow
        } else {
            ConversionCause::CoefficientUnderflow
        }));
    }
    Ok(converted)
}

/// Fallible caller-owned matrix: reserve, then initialize without reallocating.
fn try_matrix(rows: usize, cols: usize) -> Result<Mat<f64>, TryReserveError> {
    let mut matrix: Mat<f64> = Mat::with_capacity(0, 0);
    matrix.try_reserve(rows, cols)?;
    matrix.resize_with(rows, cols, |_, _| 0.0);
    Ok(matrix)
}

/// Fallible standard `Vec` reservation followed by in-capacity initialization.
fn try_vec_filled<T: Clone>(
    len: usize,
    value: T,
    buffer: &'static str,
) -> Result<Vec<T>, FitError> {
    let mut values: Vec<T> = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| FitError::AllocationFailed {
            buffer,
            requested_bytes: len.saturating_mul(std::mem::size_of::<T>()),
            source: AllocationSource::Std,
        })?;
    values.resize(len, value);
    Ok(values)
}

impl From<RegressionError> for FitError {
    fn from(_: RegressionError) -> Self {
        Self::NumericalFailure
    }
}

/// Configuration for a native-`f64` OLS fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitConfig64 {
    /// Include an intercept term (leading ones column).
    pub intercept: bool,
    /// Requested workspace budget in bytes (not a physical-memory ceiling).
    pub max_workspace_bytes: usize,
    /// Optional maximum feature count.
    pub max_features: Option<usize>,
    /// Rank policy constant `c_rank` in `tau = c_rank · max(N, p) · u64 · |R₀₀|`.
    pub rank_threshold: f64,
}

impl Default for FitConfig64 {
    fn default() -> Self {
        Self {
            intercept: true,
            max_workspace_bytes: DEFAULT_MAX_WORKSPACE_BYTES,
            max_features: None,
            rank_threshold: 1.0,
        }
    }
}

/// Configuration for a native-`f32` OLS fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitConfig32 {
    /// Include an intercept term (leading ones column).
    pub intercept: bool,
    /// Requested workspace budget in bytes (not a physical-memory ceiling).
    pub max_workspace_bytes: usize,
    /// Optional maximum feature count.
    pub max_features: Option<usize>,
    /// Rank policy constant `c_rank` in `tau = c_rank · max(N, p) · u32 · |R₀₀|`.
    pub rank_threshold: f32,
}

impl Default for FitConfig32 {
    fn default() -> Self {
        Self {
            intercept: true,
            max_workspace_bytes: DEFAULT_MAX_WORKSPACE_BYTES,
            max_features: None,
            rank_threshold: 1.0,
        }
    }
}

/// Diagnostics for a completed native-`f64` fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitDiagnostics64 {
    /// Number of selected training rows `N`.
    pub rows: usize,
    /// Feature count `d`.
    pub features: usize,
    /// Whether an intercept column was fitted.
    pub intercept: bool,
    /// Numerical rank (count of pivots above the threshold).
    pub rank: usize,
    /// Required rank `p = d + intercept`.
    pub required_rank: usize,
    /// The absolute rank threshold `tau` applied to the R diagonal.
    pub rank_threshold: f64,
    /// Ordered f64 training RSS from the retained parameters.
    pub rss_f64: f64,
    /// Requested workspace bytes (logical accounting).
    pub requested_workspace_bytes: usize,
}

/// Diagnostics for a completed native-`f32` fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitDiagnostics32 {
    /// Number of selected training rows `N`.
    pub rows: usize,
    /// Feature count `d`.
    pub features: usize,
    /// Whether an intercept column was fitted.
    pub intercept: bool,
    /// Numerical rank (count of pivots above the threshold).
    pub rank: usize,
    /// Required rank `p = d + intercept`.
    pub required_rank: usize,
    /// The absolute rank threshold `tau` applied to the R diagonal.
    pub rank_threshold: f32,
    /// Ordered f32 training RSS from the retained parameters.
    pub rss_f32: f32,
    /// Requested workspace bytes (logical accounting).
    pub requested_workspace_bytes: usize,
}

/// A successful native-`f64` fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult64 {
    weights: Vector64,
    intercept: f64,
    diagnostics: FitDiagnostics64,
}

impl FitResult64 {
    /// The learned feature coefficients (without the intercept).
    pub fn weights(&self) -> &Vector64 {
        &self.weights
    }

    /// The learned intercept (zero when no intercept was fitted).
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The training diagnostics.
    pub fn diagnostics(&self) -> &FitDiagnostics64 {
        &self.diagnostics
    }

    /// Moves the learned parameters into an immutable native-`f64` predictor.
    #[must_use]
    pub fn into_predictor(self) -> LinearRegression64 {
        LinearRegression64::new(self.weights, self.intercept)
            .expect("fit produces a validated, non-empty, finite predictor")
    }

    /// Builds a native-`f64` predictor by cloning the learned parameters.
    pub fn predictor(&self) -> LinearRegression64 {
        LinearRegression64::new(self.weights.clone(), self.intercept)
            .expect("fit produces a validated, non-empty, finite predictor")
    }

    /// Converts the f64 parameters into a separately owned f32 predictor.
    ///
    /// Weights are checked in increasing index order, then the intercept when
    /// present. The learner is not mutated and remains the continuation state.
    /// Conversion does not install the predictor or validate transform pairing;
    /// the caller runs its declared acceptance operation and replaces predictor
    /// and preprocessing together. See [`ServingConversionError`] for the checked
    /// failure classes.
    pub fn to_serving_f32(
        &self,
        on_underflow: UnderflowPolicy,
    ) -> Result<LinearRegression, ServingConversionError> {
        let mut converted = Vec::with_capacity(self.weights.len());
        for (index, &value) in self.weights.as_slice().iter().enumerate() {
            converted.push(convert_serving(
                value,
                ServingField::Weight { index },
                on_underflow,
            )?);
        }
        let bias = if self.diagnostics.intercept {
            convert_serving(self.intercept, ServingField::Intercept, on_underflow)?
        } else {
            0.0f32
        };
        let predictor = LinearRegression::new(Vector::from_vec(converted), bias)
            .expect("fit invariants make converted serving parameters valid");
        Ok(predictor)
    }
}

/// A successful native-`f32` fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult32 {
    weights: Vector,
    intercept: f32,
    diagnostics: FitDiagnostics32,
}

impl FitResult32 {
    /// The learned feature coefficients (without the intercept).
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// The learned intercept (zero when no intercept was fitted).
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// The training diagnostics.
    pub fn diagnostics(&self) -> &FitDiagnostics32 {
        &self.diagnostics
    }

    /// Moves the learned parameters into the existing f32 predictor.
    #[must_use]
    pub fn into_predictor(self) -> LinearRegression {
        LinearRegression::new(self.weights, self.intercept)
            .expect("fit produces a validated, non-empty, finite predictor")
    }

    /// Builds the existing f32 predictor by cloning the learned parameters.
    pub fn predictor(&self) -> LinearRegression {
        LinearRegression::new(self.weights.clone(), self.intercept)
            .expect("fit produces a validated, non-empty, finite predictor")
    }
}

/// Parameter field named by a serving-conversion failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingField {
    /// A feature coefficient at the zero-based index.
    Weight {
        /// Zero-based coefficient index.
        index: usize,
    },
    /// The fitted intercept.
    Intercept,
}

impl std::fmt::Display for ServingField {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Weight { index } => write!(formatter, "weight {index}"),
            Self::Intercept => formatter.write_str("intercept"),
        }
    }
}

/// Failure classes for checked f64-to-f32 serving conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingConversionError {
    /// The source parameter is NaN or infinite.
    SourceNonFinite {
        /// Offending parameter.
        field: ServingField,
    },
    /// A finite source rounds to a non-finite f32.
    Overflow {
        /// Offending parameter.
        field: ServingField,
    },
    /// A nonzero source rounds to zero and the policy rejects the loss.
    Underflow {
        /// Offending parameter.
        field: ServingField,
    },
}

impl std::fmt::Display for ServingConversionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, field) = match self {
            Self::SourceNonFinite { field } => ("non-finite source", field),
            Self::Overflow { field } => ("overflow", field),
            Self::Underflow { field } => ("underflow", field),
        };
        write!(formatter, "serving conversion {kind} at {field}")
    }
}

impl std::error::Error for ServingConversionError {}

/// Failure classes reported by [`fit_f64`].
#[derive(Debug, Clone, PartialEq)]
pub enum FitError64 {
    /// `features.len() != targets.len()`.
    RowTargetMismatch {
        /// Number of feature rows.
        features: usize,
        /// Number of targets.
        targets: usize,
    },
    /// No training rows were selected.
    TooFewRows,
    /// A training index is out of range.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The number of available rows.
        rows: usize,
    },
    /// The selected rows have zero feature width.
    ZeroFeatures,
    /// The feature width exceeds the configured maximum.
    TooManyFeatures {
        /// Observed feature width.
        features: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Selected feature rows do not all share one width.
    RaggedFeatures {
        /// Original row position.
        row: usize,
        /// Expected width.
        expected: usize,
        /// Observed width.
        actual: usize,
    },
    /// The rank-policy multiplier is negative or not finite.
    InvalidRankThreshold {
        /// Rejected value.
        value: f64,
    },
    /// A checked size calculation overflowed `usize`.
    SizeOverflow {
        /// Row count.
        rows: usize,
        /// Design-matrix column count.
        features: usize,
    },
    /// The solver scratch layout is invalid.
    ScratchLayout,
    /// The requested workspace exceeds the configured budget.
    Capacity {
        /// Requested workspace bytes.
        requested: usize,
        /// Configured limit.
        limit: usize,
    },
    /// A fallible reservation failed.
    AllocationFailed {
        /// Buffer label.
        buffer: &'static str,
        /// Requested bytes.
        requested_bytes: usize,
        /// Reservation source.
        source: AllocationSource,
    },
    /// Duplicate training indices were supplied.
    DuplicateIndex {
        /// The repeated index.
        index: usize,
    },
    /// A selected training feature or target is not finite.
    NonFiniteTrainingValue {
        /// Original row position.
        row: usize,
        /// Feature column, or `features` for the target.
        column: usize,
    },
    /// The numerical rank is below the required rank.
    RankDeficient {
        /// Computed rank.
        rank: usize,
        /// Required rank.
        required: usize,
    },
    /// The solver produced a non-finite coefficient or failed internally.
    NumericalFailure,
    /// The profile-precision training RSS is not finite.
    NonFiniteRss,
}

/// Failure classes reported by [`fit_f32`].
#[derive(Debug, Clone, PartialEq)]
pub enum FitError32 {
    /// `features.len() != targets.len()`.
    RowTargetMismatch {
        /// Number of feature rows.
        features: usize,
        /// Number of targets.
        targets: usize,
    },
    /// No training rows were selected.
    TooFewRows,
    /// A training index is out of range.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The number of available rows.
        rows: usize,
    },
    /// The selected rows have zero feature width.
    ZeroFeatures,
    /// The feature width exceeds the configured maximum.
    TooManyFeatures {
        /// Observed feature width.
        features: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Selected feature rows do not all share one width.
    RaggedFeatures {
        /// Original row position.
        row: usize,
        /// Expected width.
        expected: usize,
        /// Observed width.
        actual: usize,
    },
    /// The rank-policy multiplier is negative or not finite.
    InvalidRankThreshold {
        /// Rejected value.
        value: f32,
    },
    /// A checked size calculation overflowed `usize`.
    SizeOverflow {
        /// Row count.
        rows: usize,
        /// Design-matrix column count.
        features: usize,
    },
    /// The solver scratch layout is invalid.
    ScratchLayout,
    /// The requested workspace exceeds the configured budget.
    Capacity {
        /// Requested workspace bytes.
        requested: usize,
        /// Configured limit.
        limit: usize,
    },
    /// A fallible reservation failed.
    AllocationFailed {
        /// Buffer label.
        buffer: &'static str,
        /// Requested bytes.
        requested_bytes: usize,
        /// Reservation source.
        source: AllocationSource,
    },
    /// Duplicate training indices were supplied.
    DuplicateIndex {
        /// The repeated index.
        index: usize,
    },
    /// A selected training feature or target is not finite.
    NonFiniteTrainingValue {
        /// Original row position.
        row: usize,
        /// Feature column, or `features` for the target.
        column: usize,
    },
    /// The numerical rank is below the required rank.
    RankDeficient {
        /// Computed rank.
        rank: usize,
        /// Required rank.
        required: usize,
    },
    /// The solver produced a non-finite coefficient or failed internally.
    NumericalFailure,
    /// The profile-precision training RSS is not finite.
    NonFiniteRss,
}

macro_rules! impl_native_fit_error_display {
    ($error:ident) => {
        impl std::fmt::Display for $error {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::RowTargetMismatch { features, targets } => write!(
                        formatter,
                        "feature row count {features} does not match target count {targets}"
                    ),
                    Self::TooFewRows => formatter.write_str("no training rows were selected"),
                    Self::IndexOutOfRange { index, rows } => {
                        write!(
                            formatter,
                            "training index {index} is out of range for {rows} rows"
                        )
                    }
                    Self::ZeroFeatures => {
                        formatter.write_str("selected training rows have zero feature columns")
                    }
                    Self::TooManyFeatures { features, max } => write!(
                        formatter,
                        "feature count {features} exceeds configured maximum {max}"
                    ),
                    Self::RaggedFeatures {
                        row,
                        expected,
                        actual,
                    } => write!(
                        formatter,
                        "feature row {row} has width {actual}; expected {expected}"
                    ),
                    Self::InvalidRankThreshold { value } => {
                        write!(formatter, "invalid OLS rank threshold {value}")
                    }
                    Self::SizeOverflow { rows, features } => write!(
                        formatter,
                        "OLS size calculation overflowed for {rows} rows and {features} columns"
                    ),
                    Self::ScratchLayout => {
                        formatter.write_str("OLS solver scratch layout is invalid")
                    }
                    Self::Capacity { requested, limit } => write!(
                        formatter,
                        "requested OLS workspace {requested} bytes exceeds limit {limit} bytes"
                    ),
                    Self::AllocationFailed {
                        buffer,
                        requested_bytes,
                        source,
                    } => write!(
                        formatter,
                        "could not reserve {requested_bytes} bytes for OLS buffer {buffer} ({})",
                        allocation_source_name(*source)
                    ),
                    Self::DuplicateIndex { index } => {
                        write!(
                            formatter,
                            "training index {index} was supplied more than once"
                        )
                    }
                    Self::NonFiniteTrainingValue { row, column } => write!(
                        formatter,
                        "non-finite training value at row {row}, column {column}"
                    ),
                    Self::RankDeficient { rank, required } => write!(
                        formatter,
                        "OLS numerical rank {rank} is below required rank {required}"
                    ),
                    Self::NumericalFailure => {
                        formatter.write_str("OLS numerical calculation failed")
                    }
                    Self::NonFiniteRss => {
                        formatter.write_str("OLS training RSS is not finite in profile precision")
                    }
                }
            }
        }
    };
}

impl_native_fit_error_display!(FitError64);
impl_native_fit_error_display!(FitError32);
impl std::error::Error for FitError64 {}
impl std::error::Error for FitError32 {}

fn convert_serving(
    value: f64,
    field: ServingField,
    on_underflow: UnderflowPolicy,
) -> Result<f32, ServingConversionError> {
    if !value.is_finite() {
        return Err(ServingConversionError::SourceNonFinite { field });
    }
    let converted = value as f32;
    if !converted.is_finite() {
        return Err(ServingConversionError::Overflow { field });
    }
    if value != 0.0 && converted == 0.0 && on_underflow == UnderflowPolicy::Reject {
        return Err(ServingConversionError::Underflow { field });
    }
    Ok(converted)
}

/// Fits dense, unweighted OLS from native `f64` rows and targets.
pub fn fit_f64(
    features: &[Vector64],
    targets: &[f64],
    train: &[usize],
    config: &FitConfig64,
) -> Result<FitResult64, FitError64> {
    if features.len() != targets.len() {
        return Err(FitError64::RowTargetMismatch {
            features: features.len(),
            targets: targets.len(),
        });
    }
    let rows = train.len();
    if rows == 0 {
        return Err(FitError64::TooFewRows);
    }
    for &index in train {
        if index >= features.len() {
            return Err(FitError64::IndexOutOfRange {
                index,
                rows: features.len(),
            });
        }
    }
    let d = features[train[0]].len();
    if d == 0 {
        return Err(FitError64::ZeroFeatures);
    }
    if let Some(max) = config.max_features {
        if d > max {
            return Err(FitError64::TooManyFeatures { features: d, max });
        }
    }
    for &index in train {
        let actual = features[index].len();
        if actual != d {
            return Err(FitError64::RaggedFeatures {
                row: index,
                expected: d,
                actual,
            });
        }
    }
    if !config.rank_threshold.is_finite() || config.rank_threshold < 0.0 {
        return Err(FitError64::InvalidRankThreshold {
            value: config.rank_threshold,
        });
    }

    let p = d + usize::from(config.intercept);
    let size = rows.min(p);
    let block = recommended_block_size::<f64>(rows, p);
    let f64_bytes = std::mem::size_of::<f64>();
    let usize_bytes = std::mem::size_of::<usize>();
    let size_overflow = || FitError64::SizeOverflow { rows, features: p };
    let a_bytes = rows
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(f64_bytes))
        .ok_or_else(size_overflow)?;
    let q_bytes = block
        .checked_mul(size)
        .and_then(|cells| cells.checked_mul(f64_bytes))
        .ok_or_else(size_overflow)?;
    let rhs_bytes = rows.checked_mul(f64_bytes).ok_or_else(size_overflow)?;
    let perm_bytes = p
        .checked_mul(2)
        .and_then(|n| n.checked_mul(usize_bytes))
        .ok_or_else(size_overflow)?;
    let solution_bytes = p.checked_mul(f64_bytes).ok_or_else(size_overflow)?;
    let uniq_bytes = features.len().div_ceil(8);
    let scratch = StackReq::any_of(&[
        qr_in_place_scratch::<usize, f64>(rows, p, block, Par::Seq, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f64>(rows, p, block, 1, Par::Seq),
    ]);
    if scratch.layout().is_err() || scratch.align_bytes() == 0 {
        return Err(FitError64::ScratchLayout);
    }
    let scratch_bytes = scratch.unaligned_bytes_required();
    let requested = a_bytes
        .checked_add(q_bytes)
        .and_then(|v| v.checked_add(rhs_bytes))
        .and_then(|v| v.checked_add(perm_bytes))
        .and_then(|v| v.checked_add(solution_bytes))
        .and_then(|v| v.checked_add(uniq_bytes))
        .and_then(|v| v.checked_add(scratch_bytes))
        .ok_or_else(size_overflow)?;
    if requested > config.max_workspace_bytes {
        return Err(FitError64::Capacity {
            requested,
            limit: config.max_workspace_bytes,
        });
    }

    let mut a = try_matrix(rows, p).map_err(|_| FitError64::AllocationFailed {
        buffer: "design matrix A",
        requested_bytes: a_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut q_coeff = try_matrix(block, size).map_err(|_| FitError64::AllocationFailed {
        buffer: "Householder coefficients",
        requested_bytes: q_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut rhs = try_matrix(rows, 1).map_err(|_| FitError64::AllocationFailed {
        buffer: "right-hand side",
        requested_bytes: rhs_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut col_perm_fwd = try_vec_native(p, 0usize).map_err(|_| FitError64::AllocationFailed {
        buffer: "col_perm_fwd",
        requested_bytes: p.saturating_mul(usize_bytes),
        source: AllocationSource::Std,
    })?;
    let mut col_perm_inv = try_vec_native(p, 0usize).map_err(|_| FitError64::AllocationFailed {
        buffer: "col_perm_inv",
        requested_bytes: p.saturating_mul(usize_bytes),
        source: AllocationSource::Std,
    })?;
    let mut solution = try_vec_native(p, 0.0f64).map_err(|_| FitError64::AllocationFailed {
        buffer: "f64 solution",
        requested_bytes: solution_bytes,
        source: AllocationSource::Std,
    })?;
    let mut mem = MemBuffer::try_new(scratch).map_err(|_| FitError64::AllocationFailed {
        buffer: "solver scratch",
        requested_bytes: scratch_bytes,
        source: AllocationSource::DynStack,
    })?;
    let mut seen: Vec<u8> = Vec::new();
    seen.try_reserve_exact(uniq_bytes)
        .map_err(|_| FitError64::AllocationFailed {
            buffer: "uniqueness bitset",
            requested_bytes: uniq_bytes,
            source: AllocationSource::Std,
        })?;
    seen.resize(uniq_bytes, 0u8);

    for &index in train {
        let byte = index / 8;
        let bit = 1u8 << (index % 8);
        if seen[byte] & bit != 0 {
            return Err(FitError64::DuplicateIndex { index });
        }
        seen[byte] |= bit;
    }
    for &index in train {
        for (column, &value) in features[index].as_slice().iter().enumerate() {
            if !value.is_finite() {
                return Err(FitError64::NonFiniteTrainingValue { row: index, column });
            }
        }
        if !targets[index].is_finite() {
            return Err(FitError64::NonFiniteTrainingValue {
                row: index,
                column: d,
            });
        }
    }
    for (k, &index) in train.iter().enumerate() {
        let row = features[index].as_slice();
        if config.intercept {
            a[(k, 0)] = 1.0;
            for (j, &value) in row.iter().enumerate() {
                a[(k, j + 1)] = value;
            }
        } else {
            for (j, &value) in row.iter().enumerate() {
                a[(k, j)] = value;
            }
        }
        rhs[(k, 0)] = targets[index];
    }

    let applied_tau: f64;
    {
        let (_info, col_perm) = qr_in_place(
            a.as_mut(),
            q_coeff.as_mut(),
            &mut col_perm_fwd,
            &mut col_perm_inv,
            Par::Seq,
            MemStack::new(&mut mem),
            Spec::<ColPivQrParams, f64>::default(),
        );
        let first = a[(0, 0)].abs();
        let tau = config.rank_threshold * (rows.max(p) as f64) * U64_ROUNDOFF * first;
        applied_tau = tau;
        let mut rank = 0usize;
        for k in 0..size {
            if a[(k, k)].abs() > tau {
                rank += 1;
            }
        }
        if rank < p {
            return Err(FitError64::RankDeficient { rank, required: p });
        }
        solve_lstsq_in_place(
            a.as_ref(),
            q_coeff.as_ref(),
            a.as_ref(),
            col_perm,
            rhs.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        for (i, value) in solution.iter_mut().enumerate() {
            *value = rhs[(i, 0)];
        }
    }
    drop(mem);
    drop(a);
    drop(q_coeff);
    drop(rhs);
    drop(seen);
    drop(col_perm_fwd);
    drop(col_perm_inv);

    if !solution.iter().all(|value| value.is_finite()) {
        return Err(FitError64::NumericalFailure);
    }
    let (bias, weights) = if config.intercept {
        (solution[0], &solution[1..])
    } else {
        (0.0f64, &solution[..])
    };
    let mut rss_f64 = 0.0f64;
    for &index in train {
        let row = features[index].as_slice();
        let prediction = weights
            .iter()
            .zip(row)
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f64>()
            + bias;
        let residual = prediction - targets[index];
        rss_f64 += residual * residual;
    }
    if !rss_f64.is_finite() {
        return Err(FitError64::NonFiniteRss);
    }
    let diagnostics = FitDiagnostics64 {
        rows,
        features: d,
        intercept: config.intercept,
        rank: p,
        required_rank: p,
        rank_threshold: applied_tau,
        rss_f64,
        requested_workspace_bytes: requested,
    };
    let weights = if config.intercept {
        solution.remove(0);
        Vector64::from_vec(solution)
    } else {
        Vector64::from_vec(solution)
    };
    Ok(FitResult64 {
        weights,
        intercept: bias,
        diagnostics,
    })
}

/// Fits dense, unweighted OLS from native `f32` rows and targets.
pub fn fit_f32(
    features: &[Vector],
    targets: &[f32],
    train: &[usize],
    config: &FitConfig32,
) -> Result<FitResult32, FitError32> {
    if features.len() != targets.len() {
        return Err(FitError32::RowTargetMismatch {
            features: features.len(),
            targets: targets.len(),
        });
    }
    let rows = train.len();
    if rows == 0 {
        return Err(FitError32::TooFewRows);
    }
    for &index in train {
        if index >= features.len() {
            return Err(FitError32::IndexOutOfRange {
                index,
                rows: features.len(),
            });
        }
    }
    let d = features[train[0]].len();
    if d == 0 {
        return Err(FitError32::ZeroFeatures);
    }
    if let Some(max) = config.max_features {
        if d > max {
            return Err(FitError32::TooManyFeatures { features: d, max });
        }
    }
    for &index in train {
        let actual = features[index].len();
        if actual != d {
            return Err(FitError32::RaggedFeatures {
                row: index,
                expected: d,
                actual,
            });
        }
    }
    if !config.rank_threshold.is_finite() || config.rank_threshold < 0.0 {
        return Err(FitError32::InvalidRankThreshold {
            value: config.rank_threshold,
        });
    }

    let p = d + usize::from(config.intercept);
    let size = rows.min(p);
    let block = recommended_block_size::<f32>(rows, p);
    let f32_bytes = std::mem::size_of::<f32>();
    let usize_bytes = std::mem::size_of::<usize>();
    let size_overflow = || FitError32::SizeOverflow { rows, features: p };
    let a_bytes = rows
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(f32_bytes))
        .ok_or_else(size_overflow)?;
    let q_bytes = block
        .checked_mul(size)
        .and_then(|cells| cells.checked_mul(f32_bytes))
        .ok_or_else(size_overflow)?;
    let rhs_bytes = rows.checked_mul(f32_bytes).ok_or_else(size_overflow)?;
    let perm_bytes = p
        .checked_mul(2)
        .and_then(|n| n.checked_mul(usize_bytes))
        .ok_or_else(size_overflow)?;
    let solution_bytes = p.checked_mul(f32_bytes).ok_or_else(size_overflow)?;
    let uniq_bytes = features.len().div_ceil(8);
    let scratch = StackReq::any_of(&[
        qr_in_place_scratch::<usize, f32>(rows, p, block, Par::Seq, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f32>(rows, p, block, 1, Par::Seq),
    ]);
    if scratch.layout().is_err() || scratch.align_bytes() == 0 {
        return Err(FitError32::ScratchLayout);
    }
    let scratch_bytes = scratch.unaligned_bytes_required();
    let requested = a_bytes
        .checked_add(q_bytes)
        .and_then(|v| v.checked_add(rhs_bytes))
        .and_then(|v| v.checked_add(perm_bytes))
        .and_then(|v| v.checked_add(solution_bytes))
        .and_then(|v| v.checked_add(uniq_bytes))
        .and_then(|v| v.checked_add(scratch_bytes))
        .ok_or_else(size_overflow)?;
    if requested > config.max_workspace_bytes {
        return Err(FitError32::Capacity {
            requested,
            limit: config.max_workspace_bytes,
        });
    }

    let mut a = try_matrix_f32(rows, p).map_err(|_| FitError32::AllocationFailed {
        buffer: "design matrix A",
        requested_bytes: a_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut q_coeff = try_matrix_f32(block, size).map_err(|_| FitError32::AllocationFailed {
        buffer: "Householder coefficients",
        requested_bytes: q_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut rhs = try_matrix_f32(rows, 1).map_err(|_| FitError32::AllocationFailed {
        buffer: "right-hand side",
        requested_bytes: rhs_bytes,
        source: AllocationSource::Faer,
    })?;
    let mut col_perm_fwd = try_vec_native(p, 0usize).map_err(|_| FitError32::AllocationFailed {
        buffer: "col_perm_fwd",
        requested_bytes: p.saturating_mul(usize_bytes),
        source: AllocationSource::Std,
    })?;
    let mut col_perm_inv = try_vec_native(p, 0usize).map_err(|_| FitError32::AllocationFailed {
        buffer: "col_perm_inv",
        requested_bytes: p.saturating_mul(usize_bytes),
        source: AllocationSource::Std,
    })?;
    let mut solution = try_vec_native(p, 0.0f32).map_err(|_| FitError32::AllocationFailed {
        buffer: "f32 solution",
        requested_bytes: solution_bytes,
        source: AllocationSource::Std,
    })?;
    let mut mem = MemBuffer::try_new(scratch).map_err(|_| FitError32::AllocationFailed {
        buffer: "solver scratch",
        requested_bytes: scratch_bytes,
        source: AllocationSource::DynStack,
    })?;
    let mut seen: Vec<u8> = Vec::new();
    seen.try_reserve_exact(uniq_bytes)
        .map_err(|_| FitError32::AllocationFailed {
            buffer: "uniqueness bitset",
            requested_bytes: uniq_bytes,
            source: AllocationSource::Std,
        })?;
    seen.resize(uniq_bytes, 0u8);

    for &index in train {
        let byte = index / 8;
        let bit = 1u8 << (index % 8);
        if seen[byte] & bit != 0 {
            return Err(FitError32::DuplicateIndex { index });
        }
        seen[byte] |= bit;
    }
    for &index in train {
        for (column, &value) in features[index].as_slice().iter().enumerate() {
            if !value.is_finite() {
                return Err(FitError32::NonFiniteTrainingValue { row: index, column });
            }
        }
        if !targets[index].is_finite() {
            return Err(FitError32::NonFiniteTrainingValue {
                row: index,
                column: d,
            });
        }
    }
    for (k, &index) in train.iter().enumerate() {
        let row = features[index].as_slice();
        if config.intercept {
            a[(k, 0)] = 1.0;
            for (j, &value) in row.iter().enumerate() {
                a[(k, j + 1)] = value;
            }
        } else {
            for (j, &value) in row.iter().enumerate() {
                a[(k, j)] = value;
            }
        }
        rhs[(k, 0)] = targets[index];
    }

    let applied_tau: f32;
    {
        let (_info, col_perm) = qr_in_place(
            a.as_mut(),
            q_coeff.as_mut(),
            &mut col_perm_fwd,
            &mut col_perm_inv,
            Par::Seq,
            MemStack::new(&mut mem),
            Spec::<ColPivQrParams, f32>::default(),
        );
        let first = a[(0, 0)].abs();
        let tau = config.rank_threshold * (rows.max(p) as f32) * U32_ROUNDOFF * first;
        applied_tau = tau;
        let mut rank = 0usize;
        for k in 0..size {
            if a[(k, k)].abs() > tau {
                rank += 1;
            }
        }
        if rank < p {
            return Err(FitError32::RankDeficient { rank, required: p });
        }
        solve_lstsq_in_place(
            a.as_ref(),
            q_coeff.as_ref(),
            a.as_ref(),
            col_perm,
            rhs.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        for (i, value) in solution.iter_mut().enumerate() {
            *value = rhs[(i, 0)];
        }
    }
    drop(mem);
    drop(a);
    drop(q_coeff);
    drop(rhs);
    drop(seen);
    drop(col_perm_fwd);
    drop(col_perm_inv);

    if !solution.iter().all(|value| value.is_finite()) {
        return Err(FitError32::NumericalFailure);
    }
    let (bias, weights) = if config.intercept {
        (solution[0], &solution[1..])
    } else {
        (0.0f32, &solution[..])
    };
    let mut rss_f32 = 0.0f32;
    for &index in train {
        let row = features[index].as_slice();
        let prediction = weights
            .iter()
            .zip(row)
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + bias;
        let residual = prediction - targets[index];
        rss_f32 += residual * residual;
    }
    if !rss_f32.is_finite() {
        return Err(FitError32::NonFiniteRss);
    }
    let diagnostics = FitDiagnostics32 {
        rows,
        features: d,
        intercept: config.intercept,
        rank: p,
        required_rank: p,
        rank_threshold: applied_tau,
        rss_f32,
        requested_workspace_bytes: requested,
    };
    let weights = if config.intercept {
        solution.remove(0);
        Vector::from_vec(solution)
    } else {
        Vector::from_vec(solution)
    };
    Ok(FitResult32 {
        weights,
        intercept: bias,
        diagnostics,
    })
}

fn try_vec_native<T: Clone>(len: usize, value: T) -> Result<Vec<T>, usize> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| len.saturating_mul(std::mem::size_of::<T>()))?;
    values.resize(len, value);
    Ok(values)
}

fn try_matrix_f32(rows: usize, cols: usize) -> Result<Mat<f32>, TryReserveError> {
    let mut matrix: Mat<f32> = Mat::with_capacity(0, 0);
    matrix.try_reserve(rows, cols)?;
    matrix.resize_with(rows, cols, |_, _| 0.0);
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::{
        AcceptanceCheckError, FitDiagnostics, FitDiagnostics64, FitResult, FitResult64,
        ServingConversionError, ServingField, StoredDiagnostic, UnderflowPolicy,
    };
    use crate::foundation::numerical::{Vector, Vector64};

    fn diagnostics() -> FitDiagnostics {
        FitDiagnostics {
            rows: 1,
            features: 1,
            intercept: true,
            rank: 2,
            required_rank: 2,
            rank_threshold: 0.0,
            rss_f64_reference: 0.0,
            rss_f32_predictor: 0.0,
            requested_workspace_bytes: 0,
        }
    }

    fn result(weights: &[f32], intercept: f32, diagnostics: FitDiagnostics) -> FitResult {
        FitResult {
            coefficients: Vector::from_slice(weights),
            intercept,
            diagnostics,
        }
    }

    #[test]
    fn parameter_errors_precede_selection_metadata() {
        let candidate = result(&[f32::NAN], f32::NAN, diagnostics());
        assert_eq!(
            candidate.check_acceptance(&[], &[1.0], &[]),
            Err(AcceptanceCheckError::NonFiniteWeight { index: 0 })
        );

        let candidate = result(&[1.0], f32::NAN, diagnostics());
        assert_eq!(
            candidate.check_acceptance(&[], &[1.0], &[0]),
            Err(AcceptanceCheckError::NonFiniteBias)
        );
    }

    #[test]
    fn stored_rss_diagnostics_are_checked_in_declared_order() {
        let mut stored = diagnostics();
        stored.rss_f64_reference = f64::NAN;
        stored.rss_f32_predictor = f64::INFINITY;
        let candidate = result(&[1.0], 0.0, stored);

        assert_eq!(
            candidate.check_acceptance(&[Vector::from_slice(&[1.0])], &[1.0], &[0]),
            Err(AcceptanceCheckError::NonFiniteStoredDiagnostic {
                diagnostic: StoredDiagnostic::RssF64Reference,
            })
        );
    }

    #[test]
    fn acceptance_does_not_revalidate_fit_rank_policy() {
        let mut stored = diagnostics();
        stored.rank = 0;
        stored.required_rank = 2;
        stored.rank_threshold = f64::NAN;
        let candidate = result(&[1.0], 0.0, stored);

        let report = candidate
            .check_acceptance(&[Vector::from_slice(&[1.0])], &[1.0], &[0])
            .unwrap();
        assert_eq!(report.observations_checked, 1);
        assert_eq!(report.rss_of_checked_predictions_f64, 0.0);
    }

    #[test]
    fn extreme_finite_f32_residual_produces_finite_f64_rss() {
        let candidate = result(&[f32::MAX], 0.0, diagnostics());
        let report = candidate
            .check_acceptance(&[Vector::from_slice(&[1.0])], &[-f32::MAX], &[0])
            .unwrap();

        let residual = f64::from(f32::MAX) - f64::from(-f32::MAX);
        assert_eq!(report.observations_checked, 1);
        assert_eq!(report.rss_of_checked_predictions_f64, residual * residual);
        assert!(report.rss_of_checked_predictions_f64.is_finite());
        assert!(report.rss_of_checked_predictions_f64 > 4.0e77);
    }

    #[test]
    fn serving_conversion_reports_intercept_and_nonfinite_fields() {
        let diagnostics = FitDiagnostics64 {
            rows: 2,
            features: 1,
            intercept: true,
            rank: 2,
            required_rank: 2,
            rank_threshold: 0.0,
            rss_f64: 0.0,
            requested_workspace_bytes: 0,
        };
        let mut result = FitResult64 {
            weights: Vector64::from_slice(&[0.0]),
            intercept: 2.0f64.powi(-200),
            diagnostics,
        };
        assert_eq!(
            result.to_serving_f32(UnderflowPolicy::Reject),
            Err(ServingConversionError::Underflow {
                field: ServingField::Intercept,
            })
        );

        result.intercept = 2.0 * f64::from(f32::MAX);
        assert_eq!(
            result.to_serving_f32(UnderflowPolicy::Reject),
            Err(ServingConversionError::Overflow {
                field: ServingField::Intercept,
            })
        );

        result.weights = Vector64::from_slice(&[f64::NAN]);
        result.intercept = 0.0;
        assert_eq!(
            result.to_serving_f32(UnderflowPolicy::Reject),
            Err(ServingConversionError::SourceNonFinite {
                field: ServingField::Weight { index: 0 },
            })
        );

        result.weights = Vector64::from_slice(&[0.0]);
        result.intercept = f64::NAN;
        assert_eq!(
            result.to_serving_f32(UnderflowPolicy::Reject),
            Err(ServingConversionError::SourceNonFinite {
                field: ServingField::Intercept,
            })
        );
    }
}
