//! Benchmark records, summaries and failure/comparison evidence constructors.
//!
//! Shared output schema only: raw windows, failure records, summaries,
//! fingerprints are imported from `fixtures`, and record emission stays here.

use crate::WINDOWS;
use crate::fixtures::{fingerprint32, first_divergence32};
use crate::mode::set_unexpected_failure;

use seqvex::models::linear::rls::RlsState;

pub(crate) const RAW_RECORD_VERSION: u16 = 1;

pub(crate) const INSTRUMENTATION_VERSION: &str = "rls-bench-v1";

#[derive(Debug, Clone)]
pub(crate) struct FailureRecord {
    pub(crate) kind: &'static str,
    pub(crate) phase: &'static str,
    pub(crate) row: String,
    pub(crate) precision: &'static str,
    pub(crate) dimension: usize,
    pub(crate) window: Option<usize>,
    pub(crate) observation_global: Option<usize>,
    pub(crate) observation_window: Option<usize>,
    pub(crate) committed_progress: u64,
    pub(crate) retained_fingerprint: Option<u64>,
    pub(crate) scheduled_units: u64,
    pub(crate) attempted_units: u64,
    pub(crate) successful_units: u64,
    pub(crate) failed_units: u64,
    pub(crate) unattempted_units: u64,
    pub(crate) validation_failure: bool,
    pub(crate) postcondition_failure: bool,
    pub(crate) first_divergence: Option<String>,
    pub(crate) error: Option<String>,
}

impl FailureRecord {
    fn print_observed(&self) {
        println!(
            "FAILURE kind={} failure_record_id={} phase={} row={} precision={} D={} window={:?} observation_global={:?} observation_window={:?} committed_progress={} retained_fingerprint={:?} scheduled={} attempted={} successful={} failed={} unattempted={} validation_failure={} postcondition_failure={} first_divergence={:?} error={:?}",
            self.kind,
            failure_id(&self.row, self.phase, self.window, self.observation_global),
            self.phase,
            self.row,
            self.precision,
            self.dimension,
            self.window,
            self.observation_global,
            self.observation_window,
            self.committed_progress,
            self.retained_fingerprint,
            self.scheduled_units,
            self.attempted_units,
            self.successful_units,
            self.failed_units,
            self.unattempted_units,
            self.validation_failure,
            self.postcondition_failure,
            self.first_divergence,
            self.error
        );
    }

    pub(crate) fn emit(&self) {
        self.print_observed();
        set_unexpected_failure();
    }

    pub(crate) fn emit_observed(&self) {
        self.print_observed();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RawWindow {
    pub(crate) row_id: String,
    pub(crate) category: &'static str,
    pub(crate) precision: &'static str,
    pub(crate) dimension: usize,
    pub(crate) unit: &'static str,
    pub(crate) window_index: Option<usize>,
    pub(crate) observation_start: Option<usize>,
    pub(crate) observation_end: Option<usize>,
    pub(crate) scheduled_calls: u64,
    pub(crate) attempted_calls: u64,
    pub(crate) work_per_call: u64,
    pub(crate) scheduled_units: u64,
    pub(crate) attempted_units: u64,
    pub(crate) successful_units: u64,
    pub(crate) failed_units: u64,
    pub(crate) unattempted_units: u64,
    pub(crate) completed_updates: u64,
    pub(crate) validation_failure: bool,
    pub(crate) postcondition_failure: bool,
    pub(crate) first_divergence: Option<String>,
    pub(crate) stream_a_attempted: Option<u64>,
    pub(crate) stream_a_committed: Option<u64>,
    pub(crate) stream_b_attempted: Option<u64>,
    pub(crate) stream_b_committed: Option<u64>,
    pub(crate) timed: bool,
    pub(crate) elapsed_ns: Option<u64>,
    pub(crate) allocation_requests: Option<u64>,
    pub(crate) requested_bytes: Option<u64>,
    pub(crate) peak_live_requested_bytes: Option<u64>,
    pub(crate) status: &'static str,
    pub(crate) failure_record_id: Option<String>,
    pub(crate) emitted: bool,
}

impl RawWindow {
    pub(crate) fn emit_once(&mut self) -> bool {
        if self.emitted {
            return false;
        }
        self.emit();
        self.emitted = true;
        true
    }

    pub(crate) fn emit(&self) {
        if self.scheduled_units != self.scheduled_calls.saturating_mul(self.work_per_call)
            || self.attempted_units > self.scheduled_units
            || self.unattempted_units != self.scheduled_units.saturating_sub(self.attempted_units)
            || self.successful_units.saturating_add(self.failed_units) != self.attempted_units
        {
            println!(
                "FAILURE count_inconsistency row={} scheduled_units={} scheduled_calls={} work_per_call={} attempted_units={} unattempted_units={} successful_units={} failed_units={}",
                self.row_id,
                self.scheduled_units,
                self.scheduled_calls,
                self.work_per_call,
                self.attempted_units,
                self.unattempted_units,
                self.successful_units,
                self.failed_units
            );
            set_unexpected_failure();
        }
        let linked_failure_id = self.failure_record_id.clone().or_else(|| {
            if self.status == "invalid" {
                let phase = if self.postcondition_failure {
                    "postcondition"
                } else if self.validation_failure {
                    "validation"
                } else {
                    "measured_window"
                };
                Some(failure_id(
                    &self.row_id,
                    phase,
                    self.window_index,
                    self.observation_end,
                ))
            } else {
                None
            }
        });
        println!(
            "RAW row={} category={} precision={} D={} unit={} window={:?} obs_start={:?} obs_end={:?} scheduled_calls={} attempted_calls={} work_per_call={} scheduled_units={} attempted_units={} successful_units={} failed_units={} unattempted_units={} completed_updates={} validation_failure={} postcondition_failure={} first_divergence={:?} stream_a_attempted={:?} stream_a_committed={:?} stream_b_attempted={:?} stream_b_committed={:?} timed={} elapsed_ns={:?} allocation_requests={:?} requested_bytes={:?} peak_live_requested_bytes={:?} raw_record_version={} instrumentation_version={} status={} failure_record_id={:?}",
            self.row_id,
            self.category,
            self.precision,
            self.dimension,
            self.unit,
            self.window_index,
            self.observation_start,
            self.observation_end,
            self.scheduled_calls,
            self.attempted_calls,
            self.work_per_call,
            self.scheduled_units,
            self.attempted_units,
            self.successful_units,
            self.failed_units,
            self.unattempted_units,
            self.completed_updates,
            self.validation_failure,
            self.postcondition_failure,
            self.first_divergence,
            self.stream_a_attempted,
            self.stream_a_committed,
            self.stream_b_attempted,
            self.stream_b_committed,
            self.timed,
            self.elapsed_ns,
            self.allocation_requests,
            self.requested_bytes,
            self.peak_live_requested_bytes,
            RAW_RECORD_VERSION,
            INSTRUMENTATION_VERSION,
            self.status,
            linked_failure_id
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn push_warmup_raw(
    raws: &mut Vec<RawWindow>,
    row: &str,
    category: &'static str,
    precision: &'static str,
    dimension: usize,
    unit: &'static str,
    work_per_call: u64,
    calls: usize,
    update: bool,
    window: usize,
) {
    let units = (calls as u64) * work_per_call;
    raws.push(RawWindow {
        row_id: row.to_owned(),
        category,
        precision,
        dimension,
        unit,
        window_index: Some(window),
        observation_start: None,
        observation_end: None,
        scheduled_calls: calls as u64,
        attempted_calls: calls as u64,
        work_per_call,
        scheduled_units: units,
        attempted_units: units,
        successful_units: units,
        failed_units: 0,
        unattempted_units: 0,
        completed_updates: if update { units } else { 0 },
        validation_failure: false,
        postcondition_failure: false,
        first_divergence: None,
        stream_a_attempted: None,
        stream_a_committed: None,
        stream_b_attempted: None,
        stream_b_committed: None,
        timed: false,
        elapsed_ns: None,
        allocation_requests: None,
        requested_bytes: None,
        peak_live_requested_bytes: None,
        status: "warmup",
        failure_record_id: None,
        emitted: false,
    });
}

pub(crate) fn emit_memory_record(
    raws: &mut Vec<RawWindow>,
    row: &str,
    precision: &'static str,
    dimension: usize,
    peak: Option<u64>,
    status: &'static str,
) {
    raws.push(RawWindow {
        row_id: row.to_owned(),
        category: "memory",
        precision,
        dimension,
        unit: "memory_probe",
        window_index: None,
        observation_start: None,
        observation_end: None,
        scheduled_calls: 0,
        attempted_calls: 0,
        work_per_call: 0,
        scheduled_units: 0,
        attempted_units: 0,
        successful_units: 0,
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
        timed: false,
        elapsed_ns: None,
        allocation_requests: None,
        requested_bytes: None,
        peak_live_requested_bytes: peak,
        status,
        failure_record_id: None,
        emitted: false,
    });
}

#[derive(Debug, Clone)]
pub(crate) struct Summary {
    pub(crate) row_id: String,
    pub(crate) unit: &'static str,
    pub(crate) median: f64,
    pub(crate) iqr: f64,
    pub(crate) p95: f64,
    pub(crate) requests_per_unit: f64,
    pub(crate) bytes_per_unit: f64,
    pub(crate) windows: usize,
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    assert!(!sorted.is_empty(), "percentile requires a non-empty sample");
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

pub(crate) fn percentile_self_check() -> bool {
    let sample: Vec<f64> = (0..20).map(|value| value as f64).collect();
    let single = [7.0_f64];
    percentile(&sample, 0.5) == 10.0
        && percentile(&sample, 0.95) == 18.0
        && (percentile(&sample, 0.75) - percentile(&sample, 0.25)) == 9.0
        && percentile(&single, 0.5) == 7.0
}

pub(crate) fn summarize(
    raws: &[RawWindow],
    row_id: &str,
    precision: &'static str,
    dimension: usize,
    unit: &'static str,
    update: bool,
) -> Option<Summary> {
    let valid: Vec<&RawWindow> = raws
        .iter()
        .filter(|record| {
            record.status == "valid"
                && record.row_id == row_id
                && record.precision == precision
                && record.dimension == dimension
        })
        .collect();
    if valid.len() != WINDOWS {
        return None;
    }
    let mut values = Vec::with_capacity(valid.len());
    let mut requests = 0_u64;
    let mut bytes = 0_u64;
    let mut units = 0_u64;
    for record in &valid {
        let denominator = if update {
            record.completed_updates
        } else {
            record.successful_units
        };
        if denominator == 0 {
            return None;
        }
        let elapsed = record.elapsed_ns?;
        values.push(elapsed as f64 / denominator as f64);
        requests += record.allocation_requests?;
        bytes += record.requested_bytes?;
        units += denominator;
    }
    if units == 0 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(Summary {
        row_id: row_id.to_owned(),
        unit,
        median: percentile(&values, 0.5),
        iqr: percentile(&values, 0.75) - percentile(&values, 0.25),
        p95: percentile(&values, 0.95),
        requests_per_unit: requests as f64 / units as f64,
        bytes_per_unit: bytes as f64 / units as f64,
        windows: valid.len(),
    })
}

impl Summary {
    pub(crate) fn emit(&self) {
        println!(
            "SUMMARY row={} unit={} windows={} median={:.1}ns/{} iqr={:.1} p95={:.1} allocation-requests/{}={:.3} requested-bytes/{}={:.1}",
            self.row_id,
            self.unit,
            self.windows,
            self.median,
            self.unit,
            self.iqr,
            self.p95,
            self.unit,
            self.requests_per_unit,
            self.unit,
            self.bytes_per_unit
        );
    }
}

pub(crate) fn failure_id(
    row: &str,
    phase: &str,
    window: Option<usize>,
    observation: Option<usize>,
) -> String {
    format!("{row}:{phase}:w={window:?}:o={observation:?}")
}

#[derive(Debug, Clone)]
pub(crate) struct ComparisonEvidence {
    pub(crate) left_path: String,
    pub(crate) left_fingerprint: u64,
    pub(crate) right_path: String,
    pub(crate) right_fingerprint: u64,
    pub(crate) first_difference: String,
    pub(crate) context: String,
    pub(crate) scheduled: u64,
    pub(crate) committed: u64,
}

impl ComparisonEvidence {
    pub(crate) fn message(&self) -> String {
        format!(
            "{} fp=0x{:016x} vs {} fp=0x{:016x}; {}; first_difference={}",
            self.left_path,
            self.left_fingerprint,
            self.right_path,
            self.right_fingerprint,
            self.context,
            self.first_difference
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct OperationEvidence {
    pub(crate) scheduled: u64,
    pub(crate) attempted: u64,
    pub(crate) committed: u64,
    pub(crate) retained_fingerprint: Option<u64>,
    pub(crate) message: String,
}

#[derive(Debug)]
pub(crate) enum SweepFailure {
    Operation(OperationEvidence),
    Comparison(Box<ComparisonEvidence>),
}

impl From<OperationEvidence> for SweepFailure {
    fn from(evidence: OperationEvidence) -> Self {
        SweepFailure::Operation(evidence)
    }
}

impl From<ComparisonEvidence> for SweepFailure {
    fn from(evidence: ComparisonEvidence) -> Self {
        SweepFailure::Comparison(Box::new(evidence))
    }
}

/// Builds structured two-state comparison evidence. Both trajectories returned
/// `Ok`, so every planned unit completed: `attempted = successful = scheduled`,
/// `failed = 0`. `retained_fingerprint` stays `None` because no single retained
/// state represents the comparison; both state fingerprints are in the message
/// and `first_divergence`.
pub(crate) fn sweep_comparison_failure(
    left_path: &str,
    left: &RlsState,
    right_path: &str,
    right: &RlsState,
    context: &str,
    scheduled: u64,
) -> ComparisonEvidence {
    ComparisonEvidence {
        left_path: left_path.to_owned(),
        left_fingerprint: fingerprint32(left),
        right_path: right_path.to_owned(),
        right_fingerprint: fingerprint32(right),
        first_difference: first_divergence32(left, right)
            .unwrap_or_else(|| "states are bitwise equal".to_owned()),
        context: context.to_owned(),
        scheduled,
        committed: scheduled,
    }
}

pub(crate) fn sweep_failure_record(
    row: &str,
    features: usize,
    failure: SweepFailure,
) -> FailureRecord {
    match failure {
        SweepFailure::Comparison(evidence) => comparison_failure_record(row, features, *evidence),
        SweepFailure::Operation(failure) => operation_failure_record(row, features, failure),
    }
}

pub(crate) fn comparison_failure_record(
    row: &str,
    features: usize,
    evidence: ComparisonEvidence,
) -> FailureRecord {
    let error = evidence.message();
    FailureRecord {
        kind: "comparison_failure",
        phase: "precheck",
        row: row.to_owned(),
        precision: "f32",
        dimension: features,
        window: None,
        observation_global: None,
        observation_window: None,
        committed_progress: evidence.committed,
        retained_fingerprint: None,
        scheduled_units: evidence.scheduled,
        attempted_units: evidence.scheduled,
        successful_units: evidence.committed,
        failed_units: 0,
        unattempted_units: 0,
        validation_failure: true,
        postcondition_failure: false,
        first_divergence: Some(evidence.first_difference),
        error: Some(error),
    }
}

pub(crate) fn operation_failure_record(
    row: &str,
    features: usize,
    evidence: OperationEvidence,
) -> FailureRecord {
    let successful = evidence.committed;
    let failed = evidence.attempted.saturating_sub(successful);
    FailureRecord {
        kind: "operation_error",
        phase: "precheck",
        row: row.to_owned(),
        precision: "f32",
        dimension: features,
        window: None,
        observation_global: None,
        observation_window: None,
        committed_progress: successful,
        retained_fingerprint: evidence.retained_fingerprint,
        scheduled_units: evidence.scheduled,
        attempted_units: evidence.attempted,
        successful_units: successful,
        failed_units: failed,
        unattempted_units: evidence.scheduled.saturating_sub(evidence.attempted),
        validation_failure: false,
        postcondition_failure: false,
        first_divergence: None,
        error: Some(evidence.message),
    }
}

pub(crate) fn emit_retained_records(raws: &mut [RawWindow]) -> usize {
    let mut emitted = 0;
    for record in raws {
        if record.emit_once() {
            emitted += 1;
        }
    }
    emitted
}
