//! Ordered row-position and elapsed-time splits.
//!
//! Stage 1 provides expanding and fixed-width rolling walk-forward selection
//! over caller-owned rows. Stage 2 provides elapsed-time, availability-aware
//! expanding and rolling selection with explicit unusable outcomes. Both return
//! lazy, half-open ranges and do not own observations, fit preprocessing, score
//! labels, or run an execution schedule. Stage 3 blocked splits, purging,
//! embargo, and CPCV remain separate.

mod row;
mod time;

pub use row::{RowFold, RowFolds, RowSplitError, RowWindowSpec, TailPolicy, expanding, rolling};
pub use time::{
    ExclusionCounts, TimeCoverage, TimeFold, TimeFoldOutcome, TimeFolds, TimeRow, TimeSplitError,
    TimeWindowSpec, UnusableReason, expanding_time, rolling_time,
};
