//! Seqvex — a streaming-first ML/RL framework for sequential, non-IID data.
//!
//! This crate contains the foundational semantics (observation, ordering,
//! streaming state, state transitions, and failure atomicity) plus the first
//! vertical slice built on them: a small numerical foundation (the initial model
//! slice uses `f32`), streaming execution, and a stateful GRU
//! (`models::recurrent::gru`).
//!
//! The GRU is the slice's first model, not the framework's model abstraction;
//! `models::linear` holds the linear-model family (OLS, RLS and SGD),
//! `models::trees` holds the Decision Tree model, and `models::neighbors`
//! holds the K-Nearest Neighbors model. Hardware placement and storage design
//! remain deliberately out of scope; see `docs/ML_VERTICAL_SLICES.md`.
//!
//! The canonical development contract is `docs/DEVELOPMENT.md`.

#![forbid(unsafe_code)]

pub mod execution;
pub mod foundation;
pub mod models;
