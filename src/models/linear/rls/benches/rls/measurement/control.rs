//! Untimed/timed control runners shared by component and sharing evidence.

use std::time::Instant;

use crate::common;
use crate::evidence::{FailureRecord, RawWindow, Summary, failure_id, push_warmup_raw, summarize};
use crate::mode::legacy_validating;
use crate::{WARMUP_WINDOWS, WINDOWS};

pub(crate) fn measure_control(
    row: &str,
    precision: &'static str,
    dimension: usize,
    unit: &'static str,
    iterations: usize,
    mut op: impl FnMut() -> Result<(), String>,
    raws: &mut Vec<RawWindow>,
) -> Option<Summary> {
    let timed = !legacy_validating();
    for warm_window in 0..WARMUP_WINDOWS {
        push_warmup_raw(
            raws,
            row,
            "C",
            precision,
            dimension,
            unit,
            1,
            iterations,
            false,
            warm_window,
        );
    }
    for warm_window in 0..WARMUP_WINDOWS {
        for index in 0..iterations {
            if let Err(error) = op() {
                FailureRecord {
                    kind: "operation_error",
                    phase: "warmup",
                    row: row.to_owned(),
                    precision,
                    dimension,
                    window: Some(warm_window),
                    observation_global: None,
                    observation_window: Some(index),
                    committed_progress: index as u64,
                    retained_fingerprint: None,
                    scheduled_units: iterations as u64,
                    attempted_units: (index + 1) as u64,
                    successful_units: index as u64,
                    failed_units: 1,
                    unattempted_units: (iterations - index - 1) as u64,
                    validation_failure: false,
                    postcondition_failure: false,
                    first_divergence: None,
                    error: Some(error),
                }
                .emit();
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
        let mut failure = None;
        for index in 0..iterations {
            if let Err(error) = op() {
                failure = Some((index, error));
                break;
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
        if let Some((index, error)) = failure {
            let record = RawWindow {
                row_id: row.to_owned(),
                category: "C",
                precision,
                dimension,
                unit,
                window_index: Some(window),
                observation_start: None,
                observation_end: None,
                scheduled_calls: iterations as u64,
                attempted_calls: (index + 1) as u64,
                work_per_call: 1,
                scheduled_units: iterations as u64,
                attempted_units: (index + 1) as u64,
                successful_units: index as u64,
                failed_units: 1,
                unattempted_units: (iterations - index - 1) as u64,
                completed_updates: 0,
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
                failure_record_id: None,
                emitted: true,
            };
            record.emit();
            FailureRecord {
                kind: "operation_error",
                phase: "measured_window",
                row: row.to_owned(),
                precision,
                dimension,
                window: Some(window),
                observation_global: None,
                observation_window: Some(index),
                committed_progress: index as u64,
                retained_fingerprint: None,
                scheduled_units: iterations as u64,
                attempted_units: (index + 1) as u64,
                successful_units: index as u64,
                failed_units: 1,
                unattempted_units: (iterations - index - 1) as u64,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(error),
            }
            .emit();
            raws.push(record);
            return None;
        }
        raws.push(RawWindow {
            row_id: row.to_owned(),
            category: "C",
            precision,
            dimension,
            unit,
            window_index: Some(window),
            observation_start: None,
            observation_end: None,
            scheduled_calls: iterations as u64,
            attempted_calls: iterations as u64,
            work_per_call: 1,
            scheduled_units: iterations as u64,
            attempted_units: iterations as u64,
            successful_units: iterations as u64,
            failed_units: 0,
            unattempted_units: 0,
            completed_updates: 0,
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
    let summary = summarize(raws, row, precision, dimension, unit, false);
    if let Some(summary) = &summary {
        summary.emit();
    }
    summary
}

#[derive(Debug, Clone)]
pub(crate) struct SharingFailure {
    pub(crate) stream: &'static str,
    pub(crate) a_attempted: u64,
    pub(crate) a_committed: u64,
    pub(crate) b_attempted: u64,
    pub(crate) b_committed: u64,
    pub(crate) retained_a: u64,
    pub(crate) retained_b: u64,
    pub(crate) message: String,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_sharing(
    row: &str,
    precision: &'static str,
    dimension: usize,
    calls: usize,
    warmup_windows: usize,
    timed: bool,
    expected_failure: bool,
    out_failure: &mut Option<SharingFailure>,
    out_record: &mut Option<RawWindow>,
    mut body: impl FnMut(usize) -> Result<(u64, u64), SharingFailure>,
    raws: &mut Vec<RawWindow>,
) -> Option<Summary> {
    for warm_window in 0..warmup_windows {
        push_warmup_raw(
            raws,
            row,
            "legacy",
            precision,
            dimension,
            "update",
            2,
            calls,
            true,
            warm_window,
        );
    }
    for warm_window in 0..warmup_windows {
        for call in 0..calls {
            if let Err(error) = body(call) {
                *out_failure = Some(error.clone());
                FailureRecord {
                    kind: "operation_error",
                    phase: "warmup",
                    row: row.to_owned(),
                    precision,
                    dimension,
                    window: Some(warm_window),
                    observation_global: None,
                    observation_window: Some(call),
                    committed_progress: error.a_committed + error.b_committed,
                    retained_fingerprint: Some(error.retained_a ^ error.retained_b),
                    scheduled_units: (calls as u64) * 2,
                    attempted_units: error.a_attempted + error.b_attempted,
                    successful_units: error.a_committed + error.b_committed,
                    failed_units: 1,
                    unattempted_units: (calls as u64) * 2 - (error.a_attempted + error.b_attempted),
                    validation_failure: false,
                    postcondition_failure: false,
                    first_divergence: None,
                    error: Some(format!("stream={} {}", error.stream, error.message)),
                }
                .emit();
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
        let mut a_committed = 0_u64;
        let mut b_committed = 0_u64;
        let mut failure = None;
        for call in 0..calls {
            match body(call) {
                Ok((a, b)) => {
                    a_committed += a;
                    b_committed += b;
                }
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
        let scheduled_units = (calls as u64) * 2;
        if let Some((call, error)) = failure {
            let attempted_units = error.a_attempted + error.b_attempted;
            let successful_units = error.a_committed + error.b_committed;
            let record = RawWindow {
                row_id: row.to_owned(),
                category: "legacy",
                precision,
                dimension,
                unit: "update",
                window_index: Some(window),
                observation_start: None,
                observation_end: Some(call),
                scheduled_calls: calls as u64,
                attempted_calls: (call + 1) as u64,
                work_per_call: 2,
                scheduled_units,
                attempted_units,
                successful_units,
                failed_units: attempted_units - successful_units,
                unattempted_units: scheduled_units - attempted_units,
                completed_updates: successful_units,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                stream_a_attempted: Some(error.a_attempted),
                stream_a_committed: Some(error.a_committed),
                stream_b_attempted: Some(error.b_attempted),
                stream_b_committed: Some(error.b_committed),
                timed,
                elapsed_ns,
                allocation_requests,
                requested_bytes,
                peak_live_requested_bytes: None,
                status: "invalid",
                failure_record_id: Some(failure_id(row, "sharing_failure", Some(window), None)),
                emitted: true,
            };
            *out_failure = Some(error.clone());
            *out_record = Some(record.clone());
            raws.push(record.clone());
            record.emit();
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
                retained_fingerprint: Some(error.retained_a ^ error.retained_b),
                scheduled_units,
                attempted_units,
                successful_units,
                failed_units: attempted_units - successful_units,
                unattempted_units: scheduled_units - attempted_units,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("stream={} {}", error.stream, error.message)),
            };
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
            unit: "update",
            window_index: Some(window),
            observation_start: None,
            observation_end: Some(calls - 1),
            scheduled_calls: calls as u64,
            attempted_calls: calls as u64,
            work_per_call: 2,
            scheduled_units,
            attempted_units: scheduled_units,
            successful_units: scheduled_units,
            failed_units: 0,
            unattempted_units: 0,
            completed_updates: scheduled_units,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            stream_a_attempted: Some(calls as u64),
            stream_a_committed: Some(a_committed),
            stream_b_attempted: Some(calls as u64),
            stream_b_committed: Some(b_committed),
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
    let summary = summarize(raws, row, precision, dimension, "update", true);
    if let Some(summary) = &summary {
        summary.emit();
    }
    summary
}
