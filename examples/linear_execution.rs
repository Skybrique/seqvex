//! Deterministic end-to-end caller workflows for OLS, RLS and SGD.
//!
//! This is a caller protocol example, not a quality or performance benchmark.

#[path = "../tests/support/linear_execution.rs"]
mod support;

#[path = "linear_execution/ols.rs"]
mod ols;

#[path = "linear_execution/sgd_rls.rs"]
mod sgd_rls;

#[path = "linear_execution/temporal.rs"]
mod temporal;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::{RlsSample, RlsSample64};
use seqvex::models::linear::sgd::{SgdSample, SgdSample64};

fn sgd_sample(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn sgd_sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn rls_sample(features: &[f32], target: f32) -> Observation<RlsSample> {
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

fn run() -> Result<(), String> {
    ols::run_ols()?;
    sgd_rls::run_sgd_and_rls()?;
    temporal::run_temporal_and_publication()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("linear_execution failed: {error}");
        std::process::exit(1);
    }
    println!("linear_execution completed");
}

#[cfg(test)]
mod tests {
    #[test]
    fn deterministic_example_runs() {
        super::run().expect("example workflow succeeds");
    }
}
