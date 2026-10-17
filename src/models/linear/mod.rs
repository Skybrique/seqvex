//! Linear-model family.
//!
//! Holds the linear-model family: ordinary least squares (OLS), recursive least
//! squares (RLS), and SGD-trained linear regression. The supplied-parameter
//! predictor from the classic slice was relocated here unchanged, and OLS
//! fitting (with diagnostics, artifact v1, and checked serving) is implemented
//! under [#23](https://github.com/Skybrique/seqvex/issues/23). Future variants
//! (for example ridge, lasso or logistic regression) keep separate algorithm
//! contracts and are not declared here until they exist.
//!
//! See [`ols`] and [`rls`] for the algorithm modules and their canonical
//! READMEs; [`sgd`] holds the SGD-trained linear-regression slice.

pub mod ols;
pub mod rls;
pub mod sgd;
