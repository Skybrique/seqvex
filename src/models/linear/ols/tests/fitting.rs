//! Fitter tests for the first working OLS workflow.
//!
//! Independent references are closed-form / algebraically known values; the
//! fitter's own faer QR path is never reused as the oracle.

#[path = "fitting/acceptance.rs"]
mod acceptance;
#[path = "fitting/errors.rs"]
mod errors;
#[path = "fitting/reference.rs"]
mod reference;
#[path = "fitting/support.rs"]
mod support;
#[path = "fitting/validation.rs"]
mod validation;
#[path = "fitting/workspace.rs"]
mod workspace;
