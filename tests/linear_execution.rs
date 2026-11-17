#[path = "support/linear_execution.rs"]
mod support;

#[path = "linear_execution/batch.rs"]
mod batch;

#[path = "linear_execution/publication.rs"]
mod publication;

#[path = "linear_execution/temporal.rs"]
mod temporal;

#[path = "linear_execution/ols.rs"]
mod ols;

#[path = "linear_execution/progress.rs"]
mod progress;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::{RlsSample, RlsSample64};
use seqvex::models::linear::sgd::{SgdSample, SgdSample64};

fn sample32(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn rls_sample32(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn rls_sample64(features: &[f64], target: f64) -> Observation<RlsSample64> {
    Observation::new(RlsSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn assert_f32_bits(left: f32, right: f32) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left:?} != {right:?}");
}

fn assert_f64_bits(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left:?} != {right:?}");
}
