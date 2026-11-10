//! Shared shape errors.
//!
//! `DimensionMismatch` is defined once in the numerical vector module; this
//! module provides the common error-layer re-export without duplicating it.

pub use crate::foundation::numerical::DimensionMismatch;
