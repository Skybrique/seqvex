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

mod acceptance;
mod common;
mod legacy;

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
                        super::common::allocation_source_name(*source)
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

mod native32;
mod native64;

pub use acceptance::{
    AcceptanceCheckError, AcceptanceCheckReport, SelectedField, StoredDiagnostic,
};
pub use common::{AllocationSource, DEFAULT_MAX_WORKSPACE_BYTES, UnderflowPolicy};
pub use legacy::{ConversionCause, FitConfig, FitDiagnostics, FitError, FitResult, fit};
pub use native32::{FitConfig32, FitDiagnostics32, FitError32, FitResult32, fit_f32};
pub use native64::{
    FitConfig64, FitDiagnostics64, FitError64, FitResult64, ServingConversionError, ServingField,
    fit_f64,
};
