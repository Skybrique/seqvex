//! Precision-profile evidence for the SGD-trained linear-regression slice.
//!
//! `a` covers native-f64 recurrence/ordering/execution, `b` the checked serving
//! conversion/replacement/prediction path, `c` strict-f32 recurrence/execution,
//! and `contracts` construction/failure/state-preservation evidence.

#![allow(clippy::needless_range_loop)]

mod a;
mod b;
mod c;
mod contracts;
mod support;
