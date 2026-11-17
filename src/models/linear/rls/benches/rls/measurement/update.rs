//! Timed update runners for the f32/f64 production and candidate rows.

use std::hint::black_box;
use std::time::Instant;

use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64, RlsState, RlsState64};

use crate::candidates::{Candidate, candidate_update32, candidate_update64};
use crate::common;
use crate::evidence::{FailureRecord, RawWindow, Summary, failure_id, push_warmup_raw, summarize};
use crate::fixtures::{
    fingerprint32, fingerprint64, postcondition_divergence32, postcondition_divergence64,
};
use crate::mode::{horizon, warmup};
use crate::{WARMUP_WINDOWS, WINDOWS};

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct UpdateRunStatus {
    pub(crate) planned_windows: usize,
    pub(crate) valid_windows: usize,
    pub(crate) invalid_windows: usize,
    pub(crate) stopped_on_failure: bool,
    pub(crate) summary_eligible: bool,
    pub(crate) summary_emitted: bool,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_update32(
    row: &str,
    model: &Rls,
    warm_state: &RlsState,
    expected: &RlsState,
    observations: &[Observation<RlsSample>],
    candidate: Candidate,
    raws: &mut Vec<RawWindow>,
    windows: usize,
    timed: bool,
    emit_failure: bool,
    out_failure: &mut Option<FailureRecord>,
    out_raw: &mut Option<RawWindow>,
    out_status: &mut UpdateRunStatus,
) -> Option<Summary> {
    *out_status = UpdateRunStatus {
        planned_windows: windows,
        ..UpdateRunStatus::default()
    };
    let h = horizon();
    let start_index = warmup();
    for warm_window in 0..WARMUP_WINDOWS {
        push_warmup_raw(
            raws,
            row,
            candidate.category(),
            "f32",
            model.dimension(),
            "update",
            1,
            h,
            true,
            warm_window,
        );
    }
    for warm_window in 0..WARMUP_WINDOWS {
        let mut state = warm_state.clone();
        for k in 0..h {
            match candidate_update32(
                model,
                black_box(&state),
                black_box(&observations[start_index + k]),
                candidate,
            ) {
                Ok(next) => state = next,
                Err(error) => {
                    let foreign = state.clone();
                    FailureRecord {
                        kind: "operation_error",
                        phase: "warmup",
                        row: row.to_owned(),
                        precision: "f32",
                        dimension: model.dimension(),
                        window: Some(warm_window),
                        observation_global: Some(start_index + k),
                        observation_window: Some(k),
                        committed_progress: k as u64,
                        retained_fingerprint: Some(fingerprint32(&state)),
                        scheduled_units: h as u64,
                        attempted_units: (k + 1) as u64,
                        successful_units: k as u64,
                        failed_units: 1,
                        unattempted_units: (h - k - 1) as u64,
                        validation_failure: false,
                        postcondition_failure: false,
                        first_divergence: None,
                        error: Some(format!("{error:?}")),
                    }
                    .emit();
                    black_box(foreign);
                    return None;
                }
            }
        }
    }
    for window in 0..windows {
        let mut state = warm_state.clone();
        let before = if timed {
            Some(common::allocation_totals())
        } else {
            None
        };
        let start = if timed { Some(Instant::now()) } else { None };
        let mut failure = None;
        for k in 0..h {
            match candidate_update32(
                model,
                black_box(&state),
                black_box(&observations[start_index + k]),
                candidate,
            ) {
                Ok(next) => state = next,
                Err(error) => {
                    failure = Some((k, error));
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
        if let Some((k, error)) = failure {
            let record = RawWindow {
                row_id: row.to_owned(),
                category: candidate.category(),
                precision: "f32",
                dimension: model.dimension(),
                unit: "update",
                window_index: Some(window),
                observation_start: Some(start_index),
                observation_end: Some(start_index + k),
                scheduled_calls: h as u64,
                attempted_calls: (k + 1) as u64,
                work_per_call: 1,
                scheduled_units: h as u64,
                attempted_units: (k + 1) as u64,
                successful_units: k as u64,
                failed_units: 1,
                unattempted_units: (h - k - 1) as u64,
                completed_updates: k as u64,
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
            let fingerprint = fingerprint32(&state);
            let failure_record = FailureRecord {
                kind: "operation_error",
                phase: "measured_window",
                row: row.to_owned(),
                precision: "f32",
                dimension: model.dimension(),
                window: Some(window),
                observation_global: Some(start_index + k),
                observation_window: Some(k),
                committed_progress: k as u64,
                retained_fingerprint: Some(fingerprint),
                scheduled_units: h as u64,
                attempted_units: (k + 1) as u64,
                successful_units: k as u64,
                failed_units: 1,
                unattempted_units: (h - k - 1) as u64,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            };
            *out_failure = Some(failure_record.clone());
            *out_raw = Some(record.clone());
            raws.push(record.clone());
            record.emit();
            if emit_failure {
                failure_record.emit();
            } else {
                failure_record.emit_observed();
            }
            out_status.invalid_windows += 1;
            out_status.stopped_on_failure = true;
            out_status.summary_eligible = false;
            return None;
        }
        if let Some(divergence) = postcondition_divergence32(&state, expected) {
            let record = RawWindow {
                row_id: row.to_owned(),
                category: candidate.category(),
                precision: "f32",
                dimension: model.dimension(),
                unit: "update",
                window_index: Some(window),
                observation_start: Some(start_index),
                observation_end: Some(start_index + h - 1),
                scheduled_calls: h as u64,
                attempted_calls: h as u64,
                work_per_call: 1,
                scheduled_units: h as u64,
                attempted_units: h as u64,
                successful_units: h as u64,
                failed_units: 0,
                unattempted_units: 0,
                completed_updates: h as u64,
                validation_failure: false,
                postcondition_failure: true,
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
                failure_record_id: Some(failure_id(
                    row,
                    "postcondition",
                    Some(window),
                    Some(start_index + h - 1),
                )),
                emitted: true,
            };
            let failure_record = FailureRecord {
                kind: "postcondition_failure",
                phase: "postcondition",
                row: row.to_owned(),
                precision: "f32",
                dimension: model.dimension(),
                window: Some(window),
                observation_global: Some(start_index + h - 1),
                observation_window: Some(h - 1),
                committed_progress: h as u64,
                retained_fingerprint: Some(fingerprint32(&state)),
                scheduled_units: h as u64,
                attempted_units: h as u64,
                successful_units: h as u64,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: true,
                first_divergence: None,
                error: Some(divergence),
            };
            *out_failure = Some(failure_record.clone());
            *out_raw = Some(record.clone());
            raws.push(record.clone());
            record.emit();
            if emit_failure {
                failure_record.emit();
            } else {
                failure_record.emit_observed();
            }
            out_status.invalid_windows += 1;
            out_status.stopped_on_failure = true;
            out_status.summary_eligible = false;
            return None;
        }
        raws.push(RawWindow {
            row_id: row.to_owned(),
            category: candidate.category(),
            precision: "f32",
            dimension: model.dimension(),
            unit: "update",
            window_index: Some(window),
            observation_start: Some(start_index),
            observation_end: Some(start_index + h - 1),
            scheduled_calls: h as u64,
            attempted_calls: h as u64,
            work_per_call: 1,
            scheduled_units: h as u64,
            attempted_units: h as u64,
            successful_units: h as u64,
            failed_units: 0,
            unattempted_units: 0,
            completed_updates: h as u64,
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
        out_status.valid_windows += 1;
    }
    out_status.summary_eligible =
        out_status.valid_windows == WINDOWS && out_status.invalid_windows == 0;
    let summary = summarize(raws, row, "f32", model.dimension(), "update", true);
    if timed {
        if let Some(summary) = &summary {
            summary.emit();
            out_status.summary_emitted = true;
        }
    }
    summary
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_update64(
    row: &str,
    model: &Rls64,
    warm_state: &RlsState64,
    expected: &RlsState64,
    observations: &[Observation<RlsSample64>],
    candidate: Candidate,
    raws: &mut Vec<RawWindow>,
    windows: usize,
    timed: bool,
    emit_failure: bool,
    out_failure: &mut Option<FailureRecord>,
    out_raw: &mut Option<RawWindow>,
    out_status: &mut UpdateRunStatus,
) -> Option<Summary> {
    *out_status = UpdateRunStatus {
        planned_windows: windows,
        ..UpdateRunStatus::default()
    };
    let h = horizon();
    let start_index = warmup();
    for warm_window in 0..WARMUP_WINDOWS {
        push_warmup_raw(
            raws,
            row,
            candidate.category(),
            "f64",
            model.dimension(),
            "update",
            1,
            h,
            true,
            warm_window,
        );
    }
    for warm_window in 0..WARMUP_WINDOWS {
        let mut state = warm_state.clone();
        for k in 0..h {
            match candidate_update64(
                model,
                black_box(&state),
                black_box(&observations[start_index + k]),
                candidate,
            ) {
                Ok(next) => state = next,
                Err(error) => {
                    FailureRecord {
                        kind: "operation_error",
                        phase: "warmup",
                        row: row.to_owned(),
                        precision: "f64",
                        dimension: model.dimension(),
                        window: Some(warm_window),
                        observation_global: Some(start_index + k),
                        observation_window: Some(k),
                        committed_progress: k as u64,
                        retained_fingerprint: Some(fingerprint64(&state)),
                        scheduled_units: h as u64,
                        attempted_units: (k + 1) as u64,
                        successful_units: k as u64,
                        failed_units: 1,
                        unattempted_units: (h - k - 1) as u64,
                        validation_failure: false,
                        postcondition_failure: false,
                        first_divergence: None,
                        error: Some(format!("{error:?}")),
                    }
                    .emit();
                    return None;
                }
            }
        }
    }
    for window in 0..windows {
        let mut state = warm_state.clone();
        let before = if timed {
            Some(common::allocation_totals())
        } else {
            None
        };
        let start = if timed { Some(Instant::now()) } else { None };
        let mut failure = None;
        for k in 0..h {
            match candidate_update64(
                model,
                black_box(&state),
                black_box(&observations[start_index + k]),
                candidate,
            ) {
                Ok(next) => state = next,
                Err(error) => {
                    failure = Some((k, error));
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
        if let Some((k, error)) = failure {
            let record = RawWindow {
                row_id: row.to_owned(),
                category: candidate.category(),
                precision: "f64",
                dimension: model.dimension(),
                unit: "update",
                window_index: Some(window),
                observation_start: Some(start_index),
                observation_end: Some(start_index + k),
                scheduled_calls: h as u64,
                attempted_calls: (k + 1) as u64,
                work_per_call: 1,
                scheduled_units: h as u64,
                attempted_units: (k + 1) as u64,
                successful_units: k as u64,
                failed_units: 1,
                unattempted_units: (h - k - 1) as u64,
                completed_updates: k as u64,
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
            let failure_record = FailureRecord {
                kind: "operation_error",
                phase: "measured_window",
                row: row.to_owned(),
                precision: "f64",
                dimension: model.dimension(),
                window: Some(window),
                observation_global: Some(start_index + k),
                observation_window: Some(k),
                committed_progress: k as u64,
                retained_fingerprint: Some(fingerprint64(&state)),
                scheduled_units: h as u64,
                attempted_units: (k + 1) as u64,
                successful_units: k as u64,
                failed_units: 1,
                unattempted_units: (h - k - 1) as u64,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            };
            *out_failure = Some(failure_record.clone());
            *out_raw = Some(record.clone());
            raws.push(record.clone());
            record.emit();
            if emit_failure {
                failure_record.emit();
            } else {
                failure_record.emit_observed();
            }
            out_status.invalid_windows += 1;
            out_status.stopped_on_failure = true;
            out_status.summary_eligible = false;
            return None;
        }
        if let Some(divergence) = postcondition_divergence64(&state, expected) {
            let record = RawWindow {
                row_id: row.to_owned(),
                category: candidate.category(),
                precision: "f64",
                dimension: model.dimension(),
                unit: "update",
                window_index: Some(window),
                observation_start: Some(start_index),
                observation_end: Some(start_index + h - 1),
                scheduled_calls: h as u64,
                attempted_calls: h as u64,
                work_per_call: 1,
                scheduled_units: h as u64,
                attempted_units: h as u64,
                successful_units: h as u64,
                failed_units: 0,
                unattempted_units: 0,
                completed_updates: h as u64,
                validation_failure: false,
                postcondition_failure: true,
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
                failure_record_id: Some(failure_id(
                    row,
                    "postcondition",
                    Some(window),
                    Some(start_index + h - 1),
                )),
                emitted: true,
            };
            let failure_record = FailureRecord {
                kind: "postcondition_failure",
                phase: "postcondition",
                row: row.to_owned(),
                precision: "f64",
                dimension: model.dimension(),
                window: Some(window),
                observation_global: Some(start_index + h - 1),
                observation_window: Some(h - 1),
                committed_progress: h as u64,
                retained_fingerprint: Some(fingerprint64(&state)),
                scheduled_units: h as u64,
                attempted_units: h as u64,
                successful_units: h as u64,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: true,
                first_divergence: None,
                error: Some(divergence),
            };
            *out_failure = Some(failure_record.clone());
            *out_raw = Some(record.clone());
            raws.push(record.clone());
            record.emit();
            if emit_failure {
                failure_record.emit();
            } else {
                failure_record.emit_observed();
            }
            out_status.invalid_windows += 1;
            out_status.stopped_on_failure = true;
            out_status.summary_eligible = false;
            return None;
        }
        raws.push(RawWindow {
            row_id: row.to_owned(),
            category: candidate.category(),
            precision: "f64",
            dimension: model.dimension(),
            unit: "update",
            window_index: Some(window),
            observation_start: Some(start_index),
            observation_end: Some(start_index + h - 1),
            scheduled_calls: h as u64,
            attempted_calls: h as u64,
            work_per_call: 1,
            scheduled_units: h as u64,
            attempted_units: h as u64,
            successful_units: h as u64,
            failed_units: 0,
            unattempted_units: 0,
            completed_updates: h as u64,
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
        out_status.valid_windows += 1;
    }
    out_status.summary_eligible =
        out_status.valid_windows == WINDOWS && out_status.invalid_windows == 0;
    let summary = summarize(raws, row, "f64", model.dimension(), "update", true);
    if timed {
        if let Some(summary) = &summary {
            summary.emit();
            out_status.summary_emitted = true;
        }
    }
    summary
}
