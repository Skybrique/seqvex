//! Ordered row-position splits.
//!
//! Stage 1 provides expanding and fixed-width rolling walk-forward selection
//! over caller-owned rows. It returns lazy, half-open ranges and does not own
//! observations, infer time or label availability, fit preprocessing, or run an
//! execution schedule. Later timestamp, availability, purge, and embargo stages
//! remain separate.

mod row;

pub use row::{RowFold, RowFolds, RowSplitError, RowWindowSpec, TailPolicy, expanding, rolling};
