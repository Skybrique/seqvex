//! Minimal numerical foundation.
//!
//! `statistics` holds the established `f64` primitives. `vector` and `linalg`
//! provide concrete `f32` and `f64` containers; the GRU uses the `f32` substrate.
//! `activations` remain scalar `f32`. `random` provides seeded and
//! entropy-seeded uniform parameter-value draws (it does not sample or reorder
//! observations; see `docs/RANDOMNESS.md`). Statistics retain their established
//! panic/IEEE contracts; container shape errors use `DimensionMismatch`
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §9–13).

pub mod activations;
mod kernels;
pub mod linalg;
pub mod random;
pub mod statistics;
pub mod vector;

pub use activations::{sigmoid, tanh};
pub use linalg::{Matrix, Matrix64};
pub use random::{RandomError, RandomGenerator, UniformRange};
pub use statistics::{dot, mean, norm, online_mean, online_variance, sum, variance};
pub use vector::{DimensionMismatch, Vector, Vector64};
