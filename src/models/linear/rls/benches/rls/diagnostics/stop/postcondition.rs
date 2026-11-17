//! f32/f64 postcondition-only and matching controls.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::models::linear::rls::{RlsState, RlsState64};

use crate::candidates::{Candidate, precheck32, precheck64};
use crate::diagnostics::failure::diag_check;
use crate::evidence::failure_id;
use crate::fixtures::{fingerprint32, fingerprint64, make_model64, observations32, observations64};
use crate::measurement::{UpdateRunStatus, measure_update32, measure_update64};
use crate::mode::{has_unexpected_failure, horizon, set_unexpected_failure, warmup};
use crate::{DELTA_F64, LAMBDA_F64, WINDOWS};

use super::StopDiagnosticContext;

pub(super) fn check_f32_postcondition_controls(ctx: &StopDiagnosticContext) -> Option<bool> {
    let validation_observations = observations32(1);
    let (validation_warm, validation_expected) = match precheck32(
        &ctx.model,
        &validation_observations,
        warmup(),
        horizon(),
        Candidate::Production,
    ) {
        Ok(value) => value,
        Err(failure) => {
            println!("DIAGNOSTIC F-PostconditionOnly setup FAIL {failure:?}");
            set_unexpected_failure();
            return None;
        }
    };
    let mut altered_weights = validation_expected.w.as_slice().to_vec();
    if let Some(first) = altered_weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    let altered_expected = RlsState {
        w: Vector::from_slice(&altered_weights),
        p: validation_expected.p.clone(),
    };
    let h = horizon();
    let expected_failure_id = failure_id(
        "F-PostconditionOnly",
        "postcondition",
        Some(0),
        Some(warmup() + h - 1),
    );
    let failure_before = has_unexpected_failure();
    let mut postcondition_raws = Vec::new();
    let mut postcondition_failure = None;
    let mut postcondition_record = None;
    let mut postcondition_status = UpdateRunStatus::default();
    let postcondition_summary = measure_update32(
        "F-PostconditionOnly",
        &ctx.model,
        &validation_warm,
        &altered_expected,
        &validation_observations,
        Candidate::Production,
        &mut postcondition_raws,
        WINDOWS,
        false,
        false,
        &mut postcondition_failure,
        &mut postcondition_record,
        &mut postcondition_status,
    );
    let measured_windows = postcondition_raws
        .iter()
        .filter(|record| record.status != "warmup" && record.window_index.is_some())
        .count();
    let postcondition_ok = postcondition_summary.is_none()
        && postcondition_status.planned_windows == WINDOWS
        && postcondition_status.valid_windows == 0
        && postcondition_status.invalid_windows == 1
        && postcondition_status.stopped_on_failure
        && !postcondition_status.summary_eligible
        && !postcondition_status.summary_emitted
        && measured_windows == 1
        && postcondition_raws
            .iter()
            .filter(|record| record.status != "warmup" && record.window_index.is_some())
            .all(|record| record.window_index == Some(0))
        && postcondition_failure.as_ref().is_some_and(|failure| {
            failure.kind == "postcondition_failure"
                && !failure.validation_failure
                && failure.postcondition_failure
                && failure.first_divergence.is_none()
                && failure.scheduled_units == h as u64
                && failure.attempted_units == h as u64
                && failure.successful_units == h as u64
                && failure.failed_units == 0
                && failure.unattempted_units == 0
                && failure.committed_progress == h as u64
                && failure.retained_fingerprint == Some(fingerprint32(&validation_expected))
                && failure_id(
                    &failure.row,
                    failure.phase,
                    failure.window,
                    failure.observation_global,
                ) == expected_failure_id
        })
        && postcondition_record.as_ref().is_some_and(|record| {
            record.status == "invalid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && !record.validation_failure
                && record.postcondition_failure
                && record.first_divergence.is_none()
                && record.scheduled_units == h as u64
                && record.attempted_units == h as u64
                && record.successful_units == h as u64
                && record.failed_units == 0
                && record.unattempted_units == 0
                && record.completed_updates == h as u64
                && record.failure_record_id.as_deref() == Some(expected_failure_id.as_str())
        })
        && !failure_before
        && !has_unexpected_failure();
    let mut ok = diag_check(
        "F-PostconditionOnly",
        postcondition_ok,
        "expected the real update runner to emit an unlinked-free invalid postcondition record",
    );

    let mut matching_raws = Vec::new();
    let mut matching_failure = None;
    let mut matching_record = None;
    let mut matching_status = UpdateRunStatus::default();
    let failure_before = has_unexpected_failure();
    let matching_summary = measure_update32(
        "F-PostconditionMatch",
        &ctx.model,
        &validation_warm,
        &validation_expected,
        &validation_observations,
        Candidate::Production,
        &mut matching_raws,
        WINDOWS,
        false,
        false,
        &mut matching_failure,
        &mut matching_record,
        &mut matching_status,
    );
    let matching_ok = matching_summary.is_none()
        && matching_status.planned_windows == WINDOWS
        && matching_status.valid_windows == WINDOWS
        && matching_status.invalid_windows == 0
        && !matching_status.stopped_on_failure
        && matching_status.summary_eligible
        && !matching_status.summary_emitted
        && matching_failure.is_none()
        && matching_record.is_none()
        && matching_raws.last().is_some_and(|record| {
            record.status == "valid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && !record.postcondition_failure
                && !record.validation_failure
                && record.scheduled_units == h as u64
                && record.attempted_units == h as u64
                && record.successful_units == h as u64
                && record.failed_units == 0
                && record.unattempted_units == 0
                && record.completed_updates == h as u64
        })
        && !failure_before
        && !has_unexpected_failure();
    ok &= diag_check(
        "F-PostconditionMatch",
        matching_ok,
        "expected the same runner path to pass a matching final state",
    );
    Some(ok)
}

pub(super) fn check_postcondition64_controls(_ctx: &StopDiagnosticContext) -> Option<bool> {
    let validation_model64 = make_model64(1, LAMBDA_F64, DELTA_F64)?;
    let validation_observations64 = observations64(1);
    let (validation_warm64, validation_expected64) = match precheck64(
        &validation_model64,
        &validation_observations64,
        warmup(),
        horizon(),
        Candidate::Production,
    ) {
        Ok(value) => value,
        Err(failure) => {
            println!("DIAGNOSTIC F-PostconditionOnly-f64 setup FAIL {failure:?}");
            set_unexpected_failure();
            return None;
        }
    };
    let mut altered_weights64 = validation_expected64.weights().as_slice().to_vec();
    if let Some(first) = altered_weights64.first_mut() {
        *first = f64::from_bits(first.to_bits() ^ 1);
    }
    let altered_expected64 = match RlsState64::admit(
        Vector64::from_slice(&altered_weights64),
        validation_expected64.covariance().clone(),
    ) {
        Ok(state) => state,
        Err(error) => {
            println!("DIAGNOSTIC F-PostconditionOnly-f64 setup FAIL {error:?}");
            set_unexpected_failure();
            return None;
        }
    };
    let h = horizon();
    let expected_failure_id64 = failure_id(
        "F-PostconditionOnly-f64",
        "postcondition",
        Some(0),
        Some(warmup() + h - 1),
    );
    let mut f64_raws = Vec::new();
    let mut f64_failure = None;
    let mut f64_record = None;
    let mut f64_status = UpdateRunStatus::default();
    let failure_before = has_unexpected_failure();
    let f64_summary = measure_update64(
        "F-PostconditionOnly-f64",
        &validation_model64,
        &validation_warm64,
        &altered_expected64,
        &validation_observations64,
        Candidate::Production,
        &mut f64_raws,
        WINDOWS,
        false,
        false,
        &mut f64_failure,
        &mut f64_record,
        &mut f64_status,
    );
    let f64_windows = f64_raws
        .iter()
        .filter(|record| record.status != "warmup" && record.window_index.is_some())
        .count();
    let f64_mismatch_ok = f64_summary.is_none()
        && f64_status.planned_windows == WINDOWS
        && f64_status.valid_windows == 0
        && f64_status.invalid_windows == 1
        && f64_status.stopped_on_failure
        && !f64_status.summary_eligible
        && !f64_status.summary_emitted
        && f64_windows == 1
        && f64_raws
            .iter()
            .filter(|record| record.status != "warmup" && record.window_index.is_some())
            .all(|record| record.window_index == Some(0))
        && f64_failure.as_ref().is_some_and(|failure| {
            failure.kind == "postcondition_failure"
                && !failure.validation_failure
                && failure.postcondition_failure
                && failure.first_divergence.is_none()
                && failure.scheduled_units == h as u64
                && failure.attempted_units == h as u64
                && failure.successful_units == h as u64
                && failure.failed_units == 0
                && failure.unattempted_units == 0
                && failure.committed_progress == h as u64
                && failure.retained_fingerprint == Some(fingerprint64(&validation_expected64))
                && failure_id(
                    &failure.row,
                    failure.phase,
                    failure.window,
                    failure.observation_global,
                ) == expected_failure_id64
        })
        && f64_record.as_ref().is_some_and(|record| {
            record.status == "invalid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && !record.validation_failure
                && record.postcondition_failure
                && record.first_divergence.is_none()
                && record.scheduled_units == h as u64
                && record.attempted_units == h as u64
                && record.successful_units == h as u64
                && record.failed_units == 0
                && record.unattempted_units == 0
                && record.completed_updates == h as u64
                && record.failure_record_id.as_deref() == Some(expected_failure_id64.as_str())
        })
        && !failure_before
        && !has_unexpected_failure();

    let mut f64_matching_raws = Vec::new();
    let mut f64_matching_failure = None;
    let mut f64_matching_record = None;
    let mut f64_matching_status = UpdateRunStatus::default();
    let failure_before = has_unexpected_failure();
    let f64_matching_summary = measure_update64(
        "F-PostconditionMatch-f64",
        &validation_model64,
        &validation_warm64,
        &validation_expected64,
        &validation_observations64,
        Candidate::Production,
        &mut f64_matching_raws,
        WINDOWS,
        false,
        false,
        &mut f64_matching_failure,
        &mut f64_matching_record,
        &mut f64_matching_status,
    );
    let f64_match_ok = f64_matching_summary.is_none()
        && f64_matching_status.planned_windows == WINDOWS
        && f64_matching_status.valid_windows == WINDOWS
        && f64_matching_status.invalid_windows == 0
        && !f64_matching_status.stopped_on_failure
        && f64_matching_status.summary_eligible
        && !f64_matching_status.summary_emitted
        && f64_matching_failure.is_none()
        && f64_matching_record.is_none()
        && f64_matching_raws.last().is_some_and(|record| {
            record.status == "valid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && !record.postcondition_failure
                && !record.validation_failure
                && record.scheduled_units == h as u64
                && record.attempted_units == h as u64
                && record.successful_units == h as u64
                && record.failed_units == 0
                && record.unattempted_units == 0
                && record.completed_updates == h as u64
        })
        && !failure_before
        && !has_unexpected_failure();
    let mut ok = diag_check(
        "F-PostconditionOnly-f64",
        f64_mismatch_ok,
        "expected the real f64 runner to emit an invalid postcondition record",
    );
    ok &= diag_check(
        "F-PostconditionMatch-f64",
        f64_match_ok,
        "expected the same f64 runner path to pass a matching final state",
    );
    Some(ok)
}
