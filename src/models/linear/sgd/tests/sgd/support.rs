//! Shared SGD integration-test sample and bit helpers.
//!
//! These helpers have matching contracts across the root and subgroup evidence.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::sgd::{SgdSample, SgdSample64};

pub(crate) fn sample(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

pub(crate) fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

pub(crate) fn bits(vector: &Vector) -> Vec<u32> {
    vector
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

pub(crate) fn bits64(vector: &Vector64) -> Vec<u64> {
    vector
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}
