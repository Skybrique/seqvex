//! Data preparation operations.
//!
//! Stage 1 exposes row-position expanding and rolling walk-forward splits. The
//! caller retains ownership of the ordered data and its features, targets, and
//! metadata.

pub mod split;
