//! Legacy-path measurement runners.

use std::time::Instant;

use crate::common;
use crate::evidence::{FailureRecord, RawWindow, Summary, failure_id, push_warmup_raw, summarize};
use crate::mode::legacy_validating;
use crate::{WARMUP_WINDOWS, WINDOWS};

#[derive(Debug)]
pub(crate) struct LegacyFailure {
    pub(crate) attempted: u64,
    pub(crate) committed: u64,
    pub(crate) message: String,
    pub(crate) retained_fingerprint: Option<u64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_legacy(
    row: &str,
    precision: &'static str,
    dimension: usize,
    unit: &'static str,
    work_per_call: u64,
    calls: usize,
    update: bool,
    body: impl FnMut(usize) -> Result<u64, LegacyFailure>,
    raws: &mut Vec<RawWindow>,
) -> Option<Summary> {
    let mut out_failure = None;
    let mut out_raw = None;
    let warmup_windows = if legacy_validating() {
        0
    } else {
        WARMUP_WINDOWS
    };
    measure_legacy_impl(
        row,
        precision,
        dimension,
        unit,
        work_per_call,
        calls,
        update,
        warmup_windows,
        !legacy_validating(),
        false,
        &mut out_failure,
        &mut out_raw,
        body,
        raws,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_legacy_impl(
    row: &str,
    precision: &'static str,
    dimension: usize,
    unit: &'static str,
    work_per_call: u64,
    calls: usize,
    update: bool,
    warmup_windows: usize,
    timed: bool,
    expected_failure: bool,
    out_failure: &mut Option<FailureRecord>,
    out_raw: &mut Option<RawWindow>,
    mut body: impl FnMut(usize) -> Result<u64, LegacyFailure>,
    raws: &mut Vec<RawWindow>,
) -> Option<Summary> {
    for warm_window in 0..warmup_windows {
        push_warmup_raw(
            raws,
            row,
            "legacy",
            precision,
            dimension,
            unit,
            work_per_call,
            calls,
            update,
            warm_window,
        );
    }
    for warm_window in 0..warmup_windows {
        for call in 0..calls {
            if let Err(error) = body(call) {
                let failure_record = FailureRecord {
                    kind: "operation_error",
                    phase: "warmup",
                    row: row.to_owned(),
                    precision,
                    dimension,
                    window: Some(warm_window),
                    observation_global: None,
                    observation_window: Some(call),
                    committed_progress: error.committed,
                    retained_fingerprint: error.retained_fingerprint,
                    scheduled_units: (calls as u64) * work_per_call,
                    attempted_units: error.attempted,
                    successful_units: error.committed,
                    failed_units: 1,
                    unattempted_units: (calls as u64) * work_per_call - error.attempted,
                    validation_failure: false,
                    postcondition_failure: false,
                    first_divergence: None,
                    error: Some(error.message),
                };
                *out_failure = Some(failure_record.clone());
                if expected_failure {
                    failure_record.emit_observed();
                } else {
                    failure_record.emit();
                }
                return None;
            }
        }
    }

    for window in 0..WINDOWS {
        let before = if timed {
            Some(common::allocation_totals())
        } else {
            None
        };
        let start = if timed { Some(Instant::now()) } else { None };
        let mut successful_so_far = 0_u64;
        let mut failure = None;
        for call in 0..calls {
            match body(call) {
                Ok(units) => successful_so_far += units,
                Err(error) => {
                    failure = Some((call, error));
                    break;
                }
            }
        }
        let elapsed_ns = start.map(|start| start.elapsed().as_nanos() as u64);
        let after = if timed {
            Some(common::allocation_totals())
        } else {
            None
        };
        let allocation_requests = match (before, after) {
            (Some(before), Some(after)) => Some(after.0 - before.0),
            _ => None,
        };
        let requested_bytes = match (before, after) {
            (Some(before), Some(after)) => Some(after.1 - before.1),
            _ => None,
        };
        let scheduled_units = (calls as u64) * work_per_call;
        if let Some((call, error)) = failure {
            let attempted_units = successful_so_far + error.attempted;
            let successful_units = successful_so_far + error.committed;
            let record = RawWindow {
                row_id: row.to_owned(),
                category: "legacy",
                precision,
                dimension,
                unit,
                window_index: Some(window),
                observation_start: None,
                observation_end: Some(call),
                scheduled_calls: calls as u64,
                attempted_calls: (call + 1) as u64,
                work_per_call,
                scheduled_units,
                attempted_units,
                successful_units,
                failed_units: attempted_units - successful_units,
                unattempted_units: scheduled_units - attempted_units,
                completed_updates: if update { successful_units } else { 0 },
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                stream_a_attempted: None,
                stream_a_committed: None,
                stream_b_attempted: None,
                stream_b_committed: None,
                timed,
                elapsed_ns,
                allocation_requests,
                requested_bytes,
                peak_live_requested_bytes: None,
                status: "invalid",
                failure_record_id: Some(failure_id(row, "measured_window", Some(window), None)),
                emitted: true,
            };
            let failure_record = FailureRecord {
                kind: "operation_error",
                phase: "measured_window",
                row: row.to_owned(),
                precision,
                dimension,
                window: Some(window),
                observation_global: None,
                observation_window: Some(call),
                committed_progress: successful_units,
                retained_fingerprint: error.retained_fingerprint,
                scheduled_units,
                attempted_units,
                successful_units,
                failed_units: attempted_units - successful_units,
                unattempted_units: scheduled_units - attempted_units,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(error.message),
            };
            *out_failure = Some(failure_record.clone());
            *out_raw = Some(record.clone());
            raws.push(record.clone());
            record.emit();
            if expected_failure {
                failure_record.emit_observed();
            } else {
                failure_record.emit();
            }
            return None;
        }
        raws.push(RawWindow {
            row_id: row.to_owned(),
            category: "legacy",
            precision,
            dimension,
            unit,
            window_index: Some(window),
            observation_start: None,
            observation_end: Some(calls - 1),
            scheduled_calls: calls as u64,
            attempted_calls: calls as u64,
            work_per_call,
            scheduled_units,
            attempted_units: scheduled_units,
            successful_units: scheduled_units,
            failed_units: 0,
            unattempted_units: 0,
            completed_updates: if update { scheduled_units } else { 0 },
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            stream_a_attempted: None,
            stream_a_committed: None,
            stream_b_attempted: None,
            stream_b_committed: None,
            timed,
            elapsed_ns,
            allocation_requests,
            requested_bytes,
            peak_live_requested_bytes: None,
            status: "valid",
            failure_record_id: None,
            emitted: false,
        });
    }
    let summary = summarize(raws, row, precision, dimension, unit, update);
    if let Some(summary) = &summary {
        summary.emit();
    }
    summary
}
