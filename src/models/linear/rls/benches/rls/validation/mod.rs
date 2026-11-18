//! RLS benchmark validation entry point.
//!
//! Owns the debug/release validation coordinator. Legacy path controls live in
//! the `legacy_*` submodules and are re-exported for the benchmark entry point.

mod legacy_controls;
mod legacy_paths;
mod legacy_support;
mod legacy_sweep;
mod legacy_workloads;

pub(crate) use legacy_controls::{
    legacy_horizon, run_legacy_fixed_control, sweep_state_after_outer_calls,
};
pub(crate) use legacy_paths::validate_legacy_paths;
pub(crate) use legacy_sweep::run_legacy_batch_sweep;
pub(crate) use legacy_workloads::{run_legacy_evidence, run_legacy_workloads};

use crate::candidates::{
    Candidate, candidate_row, precheck32, precheck64, record_precheck_failure,
};
use crate::common::comparison::{baseline_first, capture_from_env};
use crate::fixtures::{make_model32, make_model64, observations32, observations64};
use crate::mode::{horizon, resolve_mode, warmup};
use crate::{DELTA_F32, DELTA_F64, DIMS, LAMBDA_F32, LAMBDA_F64};

pub(crate) fn run_validation() {
    let warm = warmup();
    let h = horizon();
    let capture = capture_from_env();
    let order = baseline_first(
        capture,
        &[
            Candidate::Production,
            Candidate::Fused,
            Candidate::Symmetric,
        ],
    );
    println!(
        "VALIDATION mode={:?} warmup={warm} horizon={h} no_timing=true capture={} order={order:?}",
        resolve_mode(),
        capture.number()
    );
    for d in DIMS {
        let Some(model) = make_model32(d, LAMBDA_F32, DELTA_F32) else {
            continue;
        };
        let observations = observations32(d);
        for candidate in [
            Candidate::Production,
            Candidate::Fused,
            Candidate::Symmetric,
        ] {
            match precheck32(&model, &observations, warm, h, candidate) {
                Ok(_) => println!("VALIDATION f32 D={d} candidate={candidate:?} PASS"),
                Err(failure) => record_precheck_failure(
                    candidate_row("f32", candidate),
                    "f32",
                    d,
                    warm,
                    h,
                    failure,
                ),
            }
        }
        let Some(model64) = make_model64(d, LAMBDA_F64, DELTA_F64) else {
            continue;
        };
        let observations64 = observations64(d);
        for candidate in [
            Candidate::Production,
            Candidate::Fused,
            Candidate::Symmetric,
        ] {
            match precheck64(&model64, &observations64, warm, h, candidate) {
                Ok(_) => println!("VALIDATION f64 D={d} candidate={candidate:?} PASS"),
                Err(failure) => record_precheck_failure(
                    candidate_row("f64", candidate),
                    "f64",
                    d,
                    warm,
                    h,
                    failure,
                ),
            }
        }
    }
}
