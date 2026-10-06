//! Linear-model family.
//!
//! Holds the linear-regression family, currently the ordinary-least-squares
//! (OLS) slice. The supplied-parameter predictor from the classic slice was
//! relocated here unchanged; OLS fitting and periodic refitting are planned
//! under [#23](https://github.com/Skybrique/seqvex/issues/23) and are not
//! present yet. Future variants (for example ridge, lasso or logistic
//! regression) keep separate algorithm contracts and are not declared here
//! until they exist.
//!
//! See [`ols`] for the algorithm module and its canonical README.

pub mod ols;
