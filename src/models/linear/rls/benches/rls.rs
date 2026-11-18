//! RLS benchmark: strict successful-update cost attribution plus untimed failure
//! diagnostics.
//!
//! Implements design Rev 4.1 / plan Rev 2. Measurement only; no production
//! optimization. Debug and validation-only modes never enter timing.
//!
//! This is the single benchmark target root. It keeps the common-module
//! inclusion, benchmark configuration constants and entry dispatch; behavioural
//! owners live in the `mode`, `evidence`, `fixtures`, `candidates`,
//! `measurement`, `diagnostics` and `validation` submodules.

use std::process::ExitCode;

use crate::diagnostics::run_failure_diagnostics;
use crate::evidence::percentile_self_check;
use crate::measurement::run_measurement;
use crate::mode::{
    Mode, env_flag, has_unexpected_failure, print_run_identity, resolve_mode,
    set_unexpected_failure,
};
use crate::validation::{
    run_legacy_batch_sweep, run_legacy_evidence, run_legacy_fixed_control, run_legacy_workloads,
    run_validation, validate_legacy_paths,
};

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

#[path = "rls/candidates.rs"]
mod candidates;
#[path = "rls/diagnostics/mod.rs"]
mod diagnostics;
#[path = "rls/evidence.rs"]
mod evidence;
#[path = "rls/fixtures.rs"]
mod fixtures;
#[path = "rls/measurement/mod.rs"]
mod measurement;
#[path = "rls/mode.rs"]
mod mode;
#[path = "rls/validation/mod.rs"]
mod validation;

const LAMBDA_F32: f32 = 0.999;
const LAMBDA_F64: f64 = 0.999;
const DELTA_F32: f32 = 1.0e4;
const DELTA_F64: f64 = 1.0e4;
const DIMS: [usize; 2] = [8, 32];
const WARMUP: usize = 1_000;
const HORIZON: usize = 10_000;
const WINDOWS: usize = 20;
const WARMUP_WINDOWS: usize = 2;
const DEBUG_WARMUP: usize = 64;
const DEBUG_HORIZON: usize = 256;
const CONTROL_ITERS: usize = 10_000;
const LAMBDA_CONTROL: f32 = 1.0;
const DELTA: f32 = 1.0e4;
const BATCH: usize = 8;
const SWEEP_OUTER_RELEASE: usize = 128;
const SWEEP_FIXTURE_LEN: usize = 2048;
const SWEEP_DIMENSIONS: [usize; 2] = [8, 32];
const SWEEP_BATCH_SIZES: [usize; 4] = [1, 8, 32, 128];

fn main() -> ExitCode {
    println!("RLS benchmark — design Rev 4.1 / plan Rev 2");
    common::verify_workloads();
    let mode = resolve_mode();
    print_run_identity(mode);
    println!(
        "mode={mode:?} allows_timing={} debug={}",
        mode.allows_timing(),
        cfg!(debug_assertions)
    );
    assert!(!cfg!(debug_assertions) || !mode.allows_timing());
    if !percentile_self_check() {
        println!("FAILURE percentile self-check failed");
        set_unexpected_failure();
    }
    let mut legacy_raws = Vec::new();
    match mode {
        Mode::DebugValidate | Mode::ReleaseValidate => {
            run_validation();
            if env_flag("SEQVEX_LEGACY_DIAGNOSTICS") {
                validate_legacy_paths();
            }
        }
        Mode::LegacyEvidence => {
            run_legacy_evidence(&mut legacy_raws);
        }
        Mode::MeasureUpdateOnly => {
            run_measurement();
        }
        Mode::MeasureDefault => {
            run_measurement();
            let mut ignored_failure = None;
            let mut ignored_raw = None;
            run_legacy_fixed_control(
                &mut legacy_raws,
                None,
                false,
                &mut ignored_failure,
                &mut ignored_raw,
            );
            run_legacy_batch_sweep(&mut legacy_raws, None);
            if env_flag("SEQVEX_LEGACY_DIAGNOSTICS") {
                run_legacy_workloads(&mut legacy_raws);
                run_legacy_evidence(&mut legacy_raws);
            }
        }
    }
    let _ = run_failure_diagnostics();
    if has_unexpected_failure() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
