//! Ordinary least-squares **fitting**: batch dense least squares in `f64`,
//! followed by checked conversion into the existing `f32` predictor.
//!
//! This is the first working fitting workflow for OLS. It is a single batch
//! least-squares solve — no epochs, learning rates, online updates or generic
//! training framework.
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
use super::predict::{LinearRegression, RegressionError};
use crate::foundation::numerical::Vector;

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

impl FitResult {
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
    // allocations: the shared `Vector` API has no move constructor, so they are
    // budgeted (`diagnostics_bytes` + `retained_bytes`) but are not
    // allocation-failure recoverable. Universal OOM recovery is not claimed.
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
