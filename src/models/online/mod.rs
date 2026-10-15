//! Online ML compatibility namespace.
//!
//! Recursive least squares (RLS) moved to [`crate::models::linear::rls`] in the
//! linear-model family; the canonical path is `seqvex::models::linear::rls`.
//! This namespace is retained only as a compatibility re-export of the same
//! canonical types. See `docs/ML_VERTICAL_SLICES.md`.

pub mod rls {
    pub use crate::models::linear::rls::{Rls, RlsBatchError, RlsError, RlsSample, RlsState};
}

pub use rls::{Rls, RlsBatchError, RlsError, RlsSample, RlsState};
