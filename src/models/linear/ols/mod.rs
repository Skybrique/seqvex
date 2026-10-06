//! Ordinary least-squares (OLS) regression.
//!
//! Supplied-parameter prediction is implemented, relocated unchanged from the
//! classic slice. The OLS fitter, diagnostics, artifact contract and fitting
//! lifecycle are planned under
//! [#23](https://github.com/Skybrique/seqvex/issues/23) and are deliberately
//! absent until the technical design is reviewed and approved.
//!
//! See `README.md` in this directory for the algorithm contract, the four
//! issue-linked workstreams and the implementation design/task plan.

mod predict;

pub use predict::{LinearRegression, RegressionError};
