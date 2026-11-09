//! Shared foundation-owned failure context.
//!
//! These types describe execution progress and failure classification without
//! owning model-specific error variants. Specialized model errors remain in
//! their algorithm modules.

pub mod execution;
pub mod shape;

pub use execution::{
    BatchFailure, ExecutorStreamFailure, FailureCause, StreamFailure, StreamProgress,
};
pub use shape::DimensionMismatch;
