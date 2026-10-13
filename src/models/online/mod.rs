//! Online ML model namespace.
//!
//! The first online slice is [recursive least squares](rls): an immutable,
//! shareable configuration `(D, λ, δ)` whose ordered adaptive state `(w, P)`
//! lives entirely in [`StateModel::State`]. See `docs/ML_VERTICAL_SLICES.md`.

pub mod rls;

pub use rls::{Rls, RlsBatchError, RlsError, RlsSample, RlsState};
