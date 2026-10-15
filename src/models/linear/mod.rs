//! Linear-model family.
//!
//! Holds the linear-model family: the ordinary-least-squares (OLS) slice and
//! recursive least squares (RLS). The supplied-parameter predictor from the
//! classic slice was relocated here unchanged; OLS fitting and periodic
//! refitting are planned under
//! [#23](https://github.com/Skybrique/seqvex/issues/23) and are not present yet.
//! Future variants (for example ridge, lasso or logistic regression) keep
//! separate algorithm contracts and are not declared here until they exist.
//!
//! See [`ols`] and [`rls`] for the algorithm modules and their canonical READMEs.

pub mod ols;
pub mod rls;
