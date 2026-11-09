//! Foundation semantics for Seqvex.
//!
//! These modules establish the smallest explicit foundations later ML/RL
//! components can rely on. They intentionally do not implement models,
//! hardware placement, or storage design.

pub mod error;
pub mod numerical;
pub mod observation;
pub mod state;
