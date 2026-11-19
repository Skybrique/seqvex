//! Data preparation operations.
//!
//! Stage 1 exposes row-position expanding and rolling walk-forward splits.
//! Stage 2 exposes elapsed-time, availability-aware expanding and rolling
//! splits under the same caller-owned data contract. The caller retains
//! ownership of the ordered data and its features, targets, and metadata.

pub mod split;
