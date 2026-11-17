//! Untimed failure diagnostics, shared diagnostic checks and grouped entry.

use std::sync::Mutex;

use seqvex::foundation::numerical::{Matrix, Matrix64, Vector, Vector64};
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{RlsError, RlsError64, RlsState, RlsState64};

use crate::candidates::{update_sym32, update_sym64};
use crate::evidence::{INSTRUMENTATION_VERSION, RAW_RECORD_VERSION};
use crate::fixtures::{
    invalid_state32, invalid_state64, make_model32, make_model64, sample32, sample64, state_eq32,
    state_eq64,
};
use crate::mode::set_unexpected_failure;

use super::operation_count::run_operation_count_diagnostics;
use super::retention::run_retention_diagnostics;
use super::stop::run_stop_diagnostics;

static DIAGNOSTIC_RESULTS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

fn diagnostic_identity(label: &str) -> (&'static str, usize) {
    let precision = if label.contains("f64") { "f64" } else { "f32" };
    let dimension = if label.starts_with("F-Denom")
        || label.starts_with("F-SymDenom")
        || label.starts_with("F-NonFinite")
        || label.starts_with("F-SymNonFinite")
        || label.starts_with("F-SymValid")
    {
        2
    } else {
        1
    };
    (precision, dimension)
}

fn emit_diagnostic_records() {
    let Ok(mut results) = DIAGNOSTIC_RESULTS.lock() else {
        return;
    };
    for (label, passed) in results.drain(..) {
        let (precision, dimension) = diagnostic_identity(&label);
        println!(
            "RAW row={} category=F precision={} D={} unit=none window=None obs_start=None obs_end=None scheduled_calls=0 attempted_calls=0 work_per_call=0 scheduled_units=0 attempted_units=0 successful_units=0 failed_units=0 unattempted_units=0 completed_updates=0 validation_failure=false postcondition_failure=false first_divergence=None stream_a_attempted=None stream_a_committed=None stream_b_attempted=None stream_b_committed=None timed=false elapsed_ns=None allocation_requests=None requested_bytes=None peak_live_requested_bytes=None raw_record_version={} instrumentation_version={} status={} failure_record_id=None",
            label,
            precision,
            dimension,
            RAW_RECORD_VERSION,
            INSTRUMENTATION_VERSION,
            if passed { "valid" } else { "invalid" }
        );
    }
}

pub(super) fn diag_check(label: &str, condition: bool, detail: &str) -> bool {
    if let Ok(mut results) = DIAGNOSTIC_RESULTS.lock() {
        results.push((label.to_owned(), condition));
    }
    if condition {
        println!("DIAGNOSTIC {label} PASS");
        true
    } else {
        println!("DIAGNOSTIC {label} FAIL {detail}");
        set_unexpected_failure();
        false
    }
}

pub(crate) fn run_failure_diagnostics() -> bool {
    let mut ok = true;
    println!("\n== untimed failure diagnostics ==");

    let Some(model) = make_model32(2, 1.0, 1.0) else {
        return false;
    };
    let invalid = invalid_state32();
    let invalid_before = invalid.clone();
    let invalid_obs = sample32(&[1.0, 0.0], 0.0);
    let production = model.update(&invalid, &invalid_obs);
    ok &= diag_check(
        "F-DenomProd",
        matches!(production, Err(RlsError::InvalidDenominator))
            && state_eq32(&invalid, &invalid_before),
        "expected InvalidDenominator and unchanged state",
    );
    let symmetric = update_sym32(&model, &invalid, invalid_obs.value());
    ok &= diag_check(
        "F-SymDenom",
        matches!(symmetric, Err(RlsError::InvalidDenominator))
            && state_eq32(&invalid, &invalid_before),
        "expected corrected symmetric InvalidDenominator and unchanged state",
    );

    let Some(model64) = make_model64(2, 1.0, 1.0) else {
        return false;
    };
    let Some(invalid64) = invalid_state64() else {
        return false;
    };
    let invalid64_before = invalid64.clone();
    let invalid_obs64 = sample64(&[1.0, 0.0], 0.0);
    let production64 = model64.update(&invalid64, &invalid_obs64);
    ok &= diag_check(
        "F-DenomProd-f64",
        matches!(production64, Err(RlsError64::InvalidDenominator))
            && state_eq64(&invalid64, &invalid64_before),
        "expected f64 InvalidDenominator and unchanged state",
    );
    let symmetric64 = update_sym64(&model64, &invalid64, invalid_obs64.value());
    ok &= diag_check(
        "F-SymDenom-f64",
        matches!(symmetric64, Err(RlsError64::InvalidDenominator))
            && state_eq64(&invalid64, &invalid64_before),
        "expected corrected f64 symmetric InvalidDenominator and unchanged state",
    );

    let Some(explosive_model) = make_model32(1, 1.0, 1.0e-30) else {
        return false;
    };
    let explosive = RlsState {
        w: Vector::from_slice(&[f32::MAX]),
        p: Matrix::from_fn(1, 1, |_, _| 1.0e-30),
    };
    let explosive_before = explosive.clone();
    let explosive_obs = sample32(&[1.0], -f32::MAX);
    let production = explosive_model.update(&explosive, &explosive_obs);
    ok &= diag_check(
        "F-NonFinite",
        matches!(production, Err(RlsError::NonFiniteCandidate))
            && state_eq32(&explosive, &explosive_before),
        "expected NonFiniteCandidate and unchanged state",
    );
    let symmetric = update_sym32(&explosive_model, &explosive, explosive_obs.value());
    ok &= diag_check(
        "F-SymNonFinite",
        matches!(symmetric, Err(RlsError::NonFiniteCandidate))
            && state_eq32(&explosive, &explosive_before),
        "expected corrected symmetric NonFiniteCandidate and unchanged state",
    );

    let Some(explosive_model64) = make_model64(1, 1.0, 1.0e-300) else {
        return false;
    };
    let explosive64 = match RlsState64::admit(
        Vector64::from_slice(&[f64::MAX]),
        Matrix64::from_fn(1, 1, |_, _| 1.0e-300),
    ) {
        Ok(state) => state,
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL explosive64 {error:?}");
            set_unexpected_failure();
            return false;
        }
    };
    let explosive64_before = explosive64.clone();
    let explosive_obs64 = sample64(&[1.0], -f64::MAX);
    let production64 = explosive_model64.update(&explosive64, &explosive_obs64);
    ok &= diag_check(
        "F-NonFinite-f64",
        matches!(production64, Err(RlsError64::NonFiniteCandidate))
            && state_eq64(&explosive64, &explosive64_before),
        "expected f64 NonFiniteCandidate and unchanged state",
    );
    let symmetric64 = update_sym64(&explosive_model64, &explosive64, explosive_obs64.value());
    ok &= diag_check(
        "F-SymNonFinite-f64",
        matches!(symmetric64, Err(RlsError64::NonFiniteCandidate))
            && state_eq64(&explosive64, &explosive64_before),
        "expected corrected f64 symmetric NonFiniteCandidate and unchanged state",
    );

    let Some(valid_model) = make_model32(2, 0.999, 100.0) else {
        return false;
    };
    let valid_state = valid_model.initial_state();
    let valid_obs = sample32(&[0.5, -0.25], 1.0);
    let production = valid_model.update(&valid_state, &valid_obs);
    let symmetric = update_sym32(&valid_model, &valid_state, valid_obs.value());
    ok &= diag_check(
        "F-SymValid",
        matches!((production, symmetric), (Ok(left), Ok(right)) if state_eq32(&left, &right)),
        "expected valid symmetric control to match production bitwise",
    );
    let Some(valid_model64) = make_model64(2, 0.999, 100.0) else {
        return false;
    };
    let valid_state64 = valid_model64.initial_state();
    let valid_obs64 = sample64(&[0.5, -0.25], 1.0);
    let production64 = valid_model64.update(&valid_state64, &valid_obs64);
    let symmetric64 = update_sym64(&valid_model64, &valid_state64, valid_obs64.value());
    ok &= diag_check(
        "F-SymValid-f64",
        matches!((production64, symmetric64), (Ok(left), Ok(right)) if state_eq64(&left, &right)),
        "expected valid f64 symmetric control to match production bitwise",
    );

    ok &= run_stop_diagnostics();
    ok &= run_operation_count_diagnostics();
    ok &= run_retention_diagnostics();
    emit_diagnostic_records();
    ok
}
