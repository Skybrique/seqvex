//! Model namespace.
//!
//! Deliberately not a universal model trait: the foundation's `StateModel`
//! already expresses the stateful contract each model needs
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §16).
//!
//! `recurrent` holds the stateful GRU slice. `classic` holds the classic ML
//! slices (decision tree and KNN); `linear` holds the linear-model family (the
//! supplied-parameter predictor relocated from classic, the planned OLS fitting
//! work under #23, and recursive least squares); and `online` is retained as a
//! compatibility namespace re-exporting the relocated RLS types. See
//! `docs/ML_VERTICAL_SLICES.md`.

pub mod classic;
pub mod linear;
pub mod online;
pub mod recurrent;
