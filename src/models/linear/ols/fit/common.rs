//! Shared OLS-fit allocation, workspace policy and presentation helpers.

use faer::{Mat, TryReserveError};

/// Default requested-workspace budget (256 MiB), as in the accepted design.
pub const DEFAULT_MAX_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

/// `f64` unit roundoff `u64 = 2^-53`.
pub(super) const U64_ROUNDOFF: f64 = f64::from_bits((1023u64 - 53) << 52);

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

/// Caller policy for a nonzero `f64` value that rounds to a zero `f32` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderflowPolicy {
    /// Reject the fit (default).
    Reject,
    /// Accept the loss and deploy the zero value.
    AllowLoss,
}

pub(super) fn allocation_source_name(source: AllocationSource) -> &'static str {
    match source {
        AllocationSource::Faer => "faer allocator",
        AllocationSource::DynStack => "dynamic stack allocator",
        AllocationSource::Std => "standard allocator",
    }
}

/// Fallible caller-owned matrix: reserve, then initialize without reallocating.
pub(super) fn try_matrix(rows: usize, cols: usize) -> Result<Mat<f64>, TryReserveError> {
    let mut matrix: Mat<f64> = Mat::with_capacity(0, 0);
    matrix.try_reserve(rows, cols)?;
    matrix.resize_with(rows, cols, |_, _| 0.0);
    Ok(matrix)
}

pub(super) fn try_vec_native<T: Clone>(len: usize, value: T) -> Result<Vec<T>, usize> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| len.saturating_mul(std::mem::size_of::<T>()))?;
    values.resize(len, value);
    Ok(values)
}
