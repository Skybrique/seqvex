//! Model namespace.
//!
//! Deliberately not a universal model trait: the foundation's `StateModel`
//! already expresses the stateful contract each model needs
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §16).
//!
//! `recurrent` holds the stateful GRU slice; `linear` holds the linear-model
//! family (OLS, RLS and SGD); `trees` holds the Decision Tree model; and
//! `neighbors` holds the K-Nearest Neighbors model. See
//! `docs/ML_VERTICAL_SLICES.md`.

pub mod linear;
pub mod neighbors;
pub mod recurrent;
pub mod trees;
