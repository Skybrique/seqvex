//! Native-`f32` OLS fitting and C-only solver helpers.

use super::super::predict::LinearRegression;
use super::common::{AllocationSource, DEFAULT_MAX_WORKSPACE_BYTES, try_vec_native};
use crate::foundation::numerical::Vector;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::col_pivoting::factor::{
    ColPivQrParams, qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::qr::col_pivoting::solve::{solve_lstsq_in_place, solve_lstsq_in_place_scratch};
use faer::{Mat, Par, Spec, TryReserveError};

/// `f32` unit roundoff `u32 = 2^-24`.
const U32_ROUNDOFF: f32 = f32::from_bits((127u32 - 24) << 23);

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

impl_native_fit_error_display!(FitError32);
impl std::error::Error for FitError32 {}

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

fn try_matrix_f32(rows: usize, cols: usize) -> Result<Mat<f32>, TryReserveError> {
    let mut matrix: Mat<f32> = Mat::with_capacity(0, 0);
    matrix.try_reserve(rows, cols)?;
    matrix.resize_with(rows, cols, |_, _| 0.0);
    Ok(matrix)
}
