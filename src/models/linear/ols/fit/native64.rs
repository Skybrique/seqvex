//! Native-`f64` OLS fitting and checked `f32` serving conversion.

use super::super::predict::{LinearRegression, LinearRegression64};
use super::common::{
    AllocationSource, DEFAULT_MAX_WORKSPACE_BYTES, U64_ROUNDOFF, UnderflowPolicy, try_matrix,
    try_vec_native,
};
use crate::foundation::numerical::narrowing::{Narrowing, NarrowingError, narrow_f64};
use crate::foundation::numerical::{Vector, Vector64};

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::col_pivoting::factor::{
    ColPivQrParams, qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::qr::col_pivoting::solve::{solve_lstsq_in_place, solve_lstsq_in_place_scratch};
use faer::{Par, Spec};

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

impl_native_fit_error_display!(FitError64);
impl std::error::Error for FitError64 {}

fn convert_serving(
    value: f64,
    field: ServingField,
    on_underflow: UnderflowPolicy,
) -> Result<f32, ServingConversionError> {
    match narrow_f64(value) {
        Err(NarrowingError::SourceNonFinite) => {
            Err(ServingConversionError::SourceNonFinite { field })
        }
        Err(NarrowingError::Overflow) => Err(ServingConversionError::Overflow { field }),
        Ok(Narrowing::Underflow(_)) if on_underflow == UnderflowPolicy::Reject => {
            Err(ServingConversionError::Underflow { field })
        }
        Ok(Narrowing::Underflow(converted)) => Ok(converted),
        Ok(Narrowing::Value(converted)) => Ok(converted),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
