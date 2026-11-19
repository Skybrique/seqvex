//! Data preparation operations.
//!
//! Stage 1 exposes row-position expanding and rolling walk-forward splits.
//! Stage 2 exposes elapsed-time, availability-aware expanding and rolling
//! splits under the same caller-owned data contract. The caller retains
//! ownership of the ordered data and its features, targets, and metadata.
//!
//! [`sampling`] composes with the splitter output: it draws training rows from
//! caller-built eligible units, is opt-in, and is independent of the splitter's
//! cutoffs, diagnostics, and held-out membership.

pub mod sampling;
pub mod split;
