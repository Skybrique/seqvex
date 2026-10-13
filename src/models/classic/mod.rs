//! Classic ML model namespace.
//!
//! [Decision tree](decision_tree) and [KNN](knn) prediction remain classic
//! slices: immutable, shareable, with no model-owned scratch. The linear
//! regression predictor used as the architectural control case moved to the
//! linear family at [`crate::models::linear::ols`]; the
//! `classic::linear_regression` path is retained below as a compatibility
//! re-export of the same types. See `docs/ML_VERTICAL_SLICES.md`.

pub mod decision_tree;
pub mod knn;

/// Compatibility path for the linear-regression predictor.
///
/// The implementation now lives in [`crate::models::linear::ols`]. This module
/// preserves the historical `models::classic::linear_regression` import path as
/// a re-export of the same public types.
pub mod linear_regression {
    pub use crate::models::linear::ols::{LinearRegression, RegressionError};
}

pub use decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode};
pub use knn::{Knn, KnnError, Reference};
pub use linear_regression::{LinearRegression, RegressionError};
