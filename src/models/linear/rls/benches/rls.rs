//! RLS benchmark: strict successful-update cost attribution plus untimed failure
//! diagnostics.
//!
//! Implements design Rev 4.1 / plan Rev 2. Measurement only; no production
//! optimization. Debug and validation-only modes never enter timing.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::error::FailureCause;
use seqvex::foundation::numerical::{Matrix, Matrix64, Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{
    BatchFailure, StateModel, process_batch, process_one, process_stream_continuing,
};
use seqvex::models::linear::rls::{
    Rls, Rls64, RlsBatchError, RlsError, RlsError64, RlsSample, RlsSample64, RlsState, RlsState64,
};

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

use common::comparison::{baseline_first, capture_from_env};

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
const RAW_RECORD_VERSION: u16 = 1;
const INSTRUMENTATION_VERSION: &str = "rls-bench-v1";

static UNEXPECTED_FAILURE: AtomicBool = AtomicBool::new(false);
static LEGACY_VALIDATE_ONLY: AtomicBool = AtomicBool::new(false);

fn legacy_validating() -> bool {
    LEGACY_VALIDATE_ONLY.load(Ordering::Relaxed)
}
static DIAGNOSTIC_RESULTS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

fn set_unexpected_failure() {
    UNEXPECTED_FAILURE.store(true, Ordering::Relaxed);
}

fn has_unexpected_failure() -> bool {
    UNEXPECTED_FAILURE.load(Ordering::Relaxed)
}

fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    DebugValidate,
    ReleaseValidate,
    LegacyEvidence,
    MeasureUpdateOnly,
    MeasureDefault,
}

impl Mode {
    fn allows_timing(self) -> bool {
        matches!(
            self,
            Self::MeasureUpdateOnly | Self::MeasureDefault | Self::LegacyEvidence
        )
    }
}

fn print_run_identity(mode: Mode) {
    let capture = capture_from_env();
    println!(
        "RUN_IDENTITY mode={mode:?} capture={} update_only={} validate_only={} evidence_only={} legacy_diagnostics={} debug={} profile={} env={}",
        capture.number(),
        env_flag("SEQVEX_RLS_UPDATE_ONLY"),
        env_flag("SEQVEX_RLS_VALIDATE_ONLY"),
        env_flag("SEQVEX_EVIDENCE_ONLY"),
        env_flag("SEQVEX_LEGACY_DIAGNOSTICS"),
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        common::env_summary()
    );
}

fn resolve_mode() -> Mode {
    if cfg!(debug_assertions) {
        Mode::DebugValidate
    } else if env_flag("SEQVEX_RLS_VALIDATE_ONLY") {
        Mode::ReleaseValidate
    } else if env_flag("SEQVEX_EVIDENCE_ONLY") {
        Mode::LegacyEvidence
    } else if env_flag("SEQVEX_RLS_UPDATE_ONLY") {
        Mode::MeasureUpdateOnly
    } else {
        Mode::MeasureDefault
    }
}

fn warmup() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_WARMUP
    } else {
        WARMUP
    }
}

fn horizon() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_HORIZON
    } else {
        HORIZON
    }
}

#[derive(Debug, Clone)]
struct FailureRecord {
    kind: &'static str,
    phase: &'static str,
    row: String,
    precision: &'static str,
    dimension: usize,
    window: Option<usize>,
    observation_global: Option<usize>,
    observation_window: Option<usize>,
    committed_progress: u64,
    retained_fingerprint: Option<u64>,
    scheduled_units: u64,
    attempted_units: u64,
    successful_units: u64,
    failed_units: u64,
    unattempted_units: u64,
    validation_failure: bool,
    postcondition_failure: bool,
    first_divergence: Option<String>,
    error: Option<String>,
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

    fn emit(&self) {
        self.print_observed();
        set_unexpected_failure();
    }

    fn emit_observed(&self) {
        self.print_observed();
    }
}

#[derive(Debug, Clone)]
struct RawWindow {
    row_id: String,
    category: &'static str,
    precision: &'static str,
    dimension: usize,
    unit: &'static str,
    window_index: Option<usize>,
    observation_start: Option<usize>,
    observation_end: Option<usize>,
    scheduled_calls: u64,
    attempted_calls: u64,
    work_per_call: u64,
    scheduled_units: u64,
    attempted_units: u64,
    successful_units: u64,
    failed_units: u64,
    unattempted_units: u64,
    completed_updates: u64,
    validation_failure: bool,
    postcondition_failure: bool,
    first_divergence: Option<String>,
    stream_a_attempted: Option<u64>,
    stream_a_committed: Option<u64>,
    stream_b_attempted: Option<u64>,
    stream_b_committed: Option<u64>,
    timed: bool,
    elapsed_ns: Option<u64>,
    allocation_requests: Option<u64>,
    requested_bytes: Option<u64>,
    peak_live_requested_bytes: Option<u64>,
    status: &'static str,
    failure_record_id: Option<String>,
    emitted: bool,
}

impl RawWindow {
    fn emit_once(&mut self) -> bool {
        if self.emitted {
            return false;
        }
        self.emit();
        self.emitted = true;
        true
    }

    fn emit(&self) {
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
fn push_warmup_raw(
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

fn emit_memory_record(
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

#[derive(Debug, Clone)]
struct Summary {
    row_id: String,
    unit: &'static str,
    median: f64,
    iqr: f64,
    p95: f64,
    requests_per_unit: f64,
    bytes_per_unit: f64,
    windows: usize,
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    assert!(!sorted.is_empty(), "percentile requires a non-empty sample");
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

fn percentile_self_check() -> bool {
    let sample: Vec<f64> = (0..20).map(|value| value as f64).collect();
    let single = [7.0_f64];
    percentile(&sample, 0.5) == 10.0
        && percentile(&sample, 0.95) == 18.0
        && (percentile(&sample, 0.75) - percentile(&sample, 0.25)) == 9.0
        && percentile(&single, 0.5) == 7.0
}

fn summarize(
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
    fn emit(&self) {
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

fn fingerprint32(state: &RlsState) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in state.w.as_slice().iter().chain(state.p.as_slice().iter()) {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fingerprint64(state: &RlsState64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in state
        .weights()
        .as_slice()
        .iter()
        .chain(state.covariance().as_slice().iter())
    {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn state_eq32(left: &RlsState, right: &RlsState) -> bool {
    left.w.as_slice().len() == right.w.as_slice().len()
        && left
            .w
            .as_slice()
            .iter()
            .zip(right.w.as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left.p.as_slice().len() == right.p.as_slice().len()
        && left
            .p
            .as_slice()
            .iter()
            .zip(right.p.as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn state_eq64(left: &RlsState64, right: &RlsState64) -> bool {
    left.weights().as_slice().len() == right.weights().as_slice().len()
        && left
            .weights()
            .as_slice()
            .iter()
            .zip(right.weights().as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left.covariance().as_slice().len() == right.covariance().as_slice().len()
        && left
            .covariance()
            .as_slice()
            .iter()
            .zip(right.covariance().as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn symmetric32(matrix: &Matrix) -> bool {
    let d = matrix.rows();
    matrix.cols() == d
        && (0..d).all(|row| {
            (0..d).all(|col| {
                matrix.as_slice()[row * d + col].to_bits()
                    == matrix.as_slice()[col * d + row].to_bits()
            })
        })
}

fn symmetric64(matrix: &Matrix64) -> bool {
    let d = matrix.rows();
    matrix.cols() == d
        && (0..d).all(|row| {
            (0..d).all(|col| {
                matrix.as_slice()[row * d + col].to_bits()
                    == matrix.as_slice()[col * d + row].to_bits()
            })
        })
}

fn first_divergence32(left: &RlsState, right: &RlsState) -> Option<String> {
    for (index, (a, b)) in left.w.as_slice().iter().zip(right.w.as_slice()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "w[{index}] left={:#010x} right={:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    for (index, (a, b)) in left.p.as_slice().iter().zip(right.p.as_slice()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "p[{index}] left={:#010x} right={:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    None
}

fn postcondition_divergence32(state: &RlsState, expected: &RlsState) -> Option<String> {
    if state_eq32(state, expected) {
        None
    } else {
        first_divergence32(state, expected)
    }
}

fn postcondition_divergence64(state: &RlsState64, expected: &RlsState64) -> Option<String> {
    if state_eq64(state, expected) {
        None
    } else {
        first_divergence64(state, expected)
    }
}

fn first_divergence64(left: &RlsState64, right: &RlsState64) -> Option<String> {
    for (index, (a, b)) in left
        .weights()
        .as_slice()
        .iter()
        .zip(right.weights().as_slice())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "w[{index}] left={:#018x} right={:#018x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    for (index, (a, b)) in left
        .covariance()
        .as_slice()
        .iter()
        .zip(right.covariance().as_slice())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "p[{index}] left={:#018x} right={:#018x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    None
}

struct F64Lcg {
    state: u64,
}

impl F64Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 40) as f64 / (1_u64 << 24) as f64) - 0.5
    }
}

fn observations32(d: usize) -> Vec<Observation<RlsSample>> {
    let n = warmup() + horizon();
    let xs = common::persistent_excitation_features(d, n);
    let weights = common::fixed_weights(d);
    let ys = common::linear_targets(&xs, &weights);
    xs.iter()
        .zip(ys)
        .map(|(x, target)| {
            Observation::new(RlsSample {
                features: Vector::from_slice(x),
                target,
            })
        })
        .collect()
}

fn observations64(d: usize) -> Vec<Observation<RlsSample64>> {
    let n = warmup() + horizon();
    let mut lcg = F64Lcg::new(common::EXCITATION_SEED);
    let weights: Vec<f64> = (0..d)
        .map(|index| (index as f64 * 0.9).sin() + 0.3)
        .collect();
    (0..n)
        .map(|_| {
            let features: Vec<f64> = (0..d).map(|_| 2.0 * lcg.next()).collect();
            let target = features
                .iter()
                .zip(&weights)
                .map(|(value, weight)| value * weight)
                .sum();
            Observation::new(RlsSample64 {
                features: Vector64::from_slice(&features),
                target,
            })
        })
        .collect()
}

fn p_full_into32(out: &mut [f32], p: &[f32], v: &[f32], d: usize, lambda: f32, denom: f32) {
    for row in 0..d {
        for col in 0..d {
            out[row * d + col] = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
        }
    }
}

fn p_sym_into32(out: &mut [f32], p: &[f32], v: &[f32], d: usize, lambda: f32, denom: f32) {
    for row in 0..d {
        for col in row..d {
            let value = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
            out[row * d + col] = value;
            out[col * d + row] = value;
        }
    }
}

fn p_full_into64(out: &mut [f64], p: &[f64], v: &[f64], d: usize, lambda: f64, denom: f64) {
    for row in 0..d {
        for col in 0..d {
            out[row * d + col] = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
        }
    }
}

fn p_sym_into64(out: &mut [f64], p: &[f64], v: &[f64], d: usize, lambda: f64, denom: f64) {
    for row in 0..d {
        for col in row..d {
            let value = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
            out[row * d + col] = value;
            out[col * d + row] = value;
        }
    }
}

fn validate32(model: &Rls, state: &RlsState, sample: &RlsSample) -> Result<(), RlsError> {
    let d = model.dimension();
    let features = &sample.features;
    if features.len() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: features.len(),
        });
    }
    if state.w.len() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: state.w.len(),
        });
    }
    if state.p.rows() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: state.p.rows(),
        });
    }
    if state.p.cols() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: state.p.cols(),
        });
    }
    if !features.as_slice().iter().all(|value| value.is_finite()) || !sample.target.is_finite() {
        return Err(RlsError::NonFiniteInput);
    }
    Ok(())
}

fn validate64(model: &Rls64, state: &RlsState64, sample: &RlsSample64) -> Result<(), RlsError64> {
    let d = model.dimension();
    let features = &sample.features;
    let weights = state.weights();
    let covariance = state.covariance();
    if features.len() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: features.len(),
        });
    }
    if weights.len() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: weights.len(),
        });
    }
    if covariance.rows() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: covariance.rows(),
        });
    }
    if covariance.cols() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: covariance.cols(),
        });
    }
    if !features.as_slice().iter().all(|value| value.is_finite()) || !sample.target.is_finite() {
        return Err(RlsError64::NonFiniteInput);
    }
    Ok(())
}

fn update_fused32(model: &Rls, state: &RlsState, sample: &RlsSample) -> Result<RlsState, RlsError> {
    validate32(model, state, sample)?;
    let features = &sample.features;
    let v = state.p.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError::InvalidDenominator);
    }
    let error = sample.target - state.w.dot(features)?;
    let d = model.dimension();
    let next_w = Vector::from_fn(d, |index| {
        state.w.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    let p_data = state.p.as_slice();
    let v_data = v.as_slice();
    let lambda = model.lambda();
    let next_p = Matrix::from_fn(d, d, |row, col| {
        (p_data[row * d + col] - v_data[row] * v_data[col] / denom) / lambda
    });
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError::NonFiniteCandidate);
    }
    Ok(RlsState {
        w: next_w,
        p: next_p,
    })
}

fn update_sym32(model: &Rls, state: &RlsState, sample: &RlsSample) -> Result<RlsState, RlsError> {
    validate32(model, state, sample)?;
    let features = &sample.features;
    let v = state.p.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError::InvalidDenominator);
    }
    let error = sample.target - state.w.dot(features)?;
    let gain = v.map(|value| value / denom);
    let next_w = state.w.add(&gain.scale(error))?;
    let d = model.dimension();
    let mut buffer = vec![0.0_f32; d * d];
    p_sym_into32(
        &mut buffer,
        state.p.as_slice(),
        v.as_slice(),
        d,
        model.lambda(),
        denom,
    );
    let next_p = Matrix::from_fn(d, d, |row, col| buffer[row * d + col]);
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError::NonFiniteCandidate);
    }
    Ok(RlsState {
        w: next_w,
        p: next_p,
    })
}

fn admit64(weights: Vector64, covariance: Matrix64) -> Result<RlsState64, RlsError64> {
    RlsState64::admit(weights, covariance).map_err(|_| RlsError64::NonFiniteCandidate)
}

fn update_fused64(
    model: &Rls64,
    state: &RlsState64,
    sample: &RlsSample64,
) -> Result<RlsState64, RlsError64> {
    validate64(model, state, sample)?;
    let features = &sample.features;
    let weights = state.weights();
    let covariance = state.covariance();
    let v = covariance.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError64::InvalidDenominator);
    }
    let error = sample.target - weights.dot(features)?;
    let d = model.dimension();
    let next_w = Vector64::from_fn(d, |index| {
        weights.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    let p_data = covariance.as_slice();
    let v_data = v.as_slice();
    let lambda = model.lambda();
    let next_p = Matrix64::from_fn(d, d, |row, col| {
        (p_data[row * d + col] - v_data[row] * v_data[col] / denom) / lambda
    });
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError64::NonFiniteCandidate);
    }
    admit64(next_w, next_p)
}

fn update_sym64(
    model: &Rls64,
    state: &RlsState64,
    sample: &RlsSample64,
) -> Result<RlsState64, RlsError64> {
    validate64(model, state, sample)?;
    let features = &sample.features;
    let weights = state.weights();
    let covariance = state.covariance();
    let v = covariance.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError64::InvalidDenominator);
    }
    let error = sample.target - weights.dot(features)?;
    let gain = v.map(|value| value / denom);
    let next_w = weights.add(&gain.scale(error))?;
    let d = model.dimension();
    let mut buffer = vec![0.0_f64; d * d];
    p_sym_into64(
        &mut buffer,
        covariance.as_slice(),
        v.as_slice(),
        d,
        model.lambda(),
        denom,
    );
    let next_p = Matrix64::from_fn(d, d, |row, col| buffer[row * d + col]);
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError64::NonFiniteCandidate);
    }
    admit64(next_w, next_p)
}

fn vcur_alloc32(w: &Vector, v: &Vector, error: f32, denom: f32) -> Result<(), String> {
    let gain = v.map(|value| value / denom);
    let scaled = gain.scale(error);
    let next = w.add(&scaled).map_err(|error| format!("{error:?}"))?;
    black_box(&next);
    Ok(())
}

fn vcur_pre32(
    w: &Vector,
    v: &Vector,
    error: f32,
    denom: f32,
    gain: &mut [f32],
    scaled: &mut [f32],
    next: &mut [f32],
) -> Result<(), String> {
    for index in 0..w.len() {
        gain[index] = v.as_slice()[index] / denom;
        scaled[index] = gain[index] * error;
        next[index] = w.as_slice()[index] + scaled[index];
    }
    black_box(&next);
    Ok(())
}

fn vfused_alloc32(w: &Vector, v: &Vector, error: f32, denom: f32) -> Result<(), String> {
    let next = Vector::from_fn(w.len(), |index| {
        w.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    black_box(&next);
    Ok(())
}

fn vfused_pre32(
    w: &Vector,
    v: &Vector,
    error: f32,
    denom: f32,
    next: &mut [f32],
) -> Result<(), String> {
    for (index, slot) in next.iter_mut().enumerate().take(w.len()) {
        *slot = w.as_slice()[index] + (v.as_slice()[index] / denom) * error;
    }
    black_box(&next);
    Ok(())
}

fn vcur_alloc64(w: &Vector64, v: &Vector64, error: f64, denom: f64) -> Result<(), String> {
    let gain = v.map(|value| value / denom);
    let scaled = gain.scale(error);
    let next = w.add(&scaled).map_err(|error| format!("{error:?}"))?;
    black_box(&next);
    Ok(())
}

fn vcur_pre64(
    w: &Vector64,
    v: &Vector64,
    error: f64,
    denom: f64,
    gain: &mut [f64],
    scaled: &mut [f64],
    next: &mut [f64],
) -> Result<(), String> {
    for index in 0..w.len() {
        gain[index] = v.as_slice()[index] / denom;
        scaled[index] = gain[index] * error;
        next[index] = w.as_slice()[index] + scaled[index];
    }
    black_box(&next);
    Ok(())
}

fn vfused_alloc64(w: &Vector64, v: &Vector64, error: f64, denom: f64) -> Result<(), String> {
    let next = Vector64::from_fn(w.len(), |index| {
        w.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    black_box(&next);
    Ok(())
}

fn vfused_pre64(
    w: &Vector64,
    v: &Vector64,
    error: f64,
    denom: f64,
    next: &mut [f64],
) -> Result<(), String> {
    for (index, slot) in next.iter_mut().enumerate().take(w.len()) {
        *slot = w.as_slice()[index] + (v.as_slice()[index] / denom) * error;
    }
    black_box(&next);
    Ok(())
}

#[derive(Debug)]
struct PrecheckFailure {
    index: usize,
    kind: &'static str,
    error: String,
    fingerprint: u64,
    validation: bool,
    first_divergence: Option<String>,
}

fn precheck32(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    warm: usize,
    horizon: usize,
    candidate: Candidate,
) -> Result<(RlsState, RlsState), PrecheckFailure> {
    let total = warm + horizon;
    let mut production = model.initial_state();
    let mut candidate_state = production.clone();
    let mut warm_state = None;
    for (index, observation) in observations.iter().enumerate().take(total) {
        let production_result = model.update(&production, observation);
        let candidate_result = match candidate {
            Candidate::Production => production_result.clone(),
            Candidate::Fused => update_fused32(model, &candidate_state, observation.value()),
            Candidate::Symmetric => update_sym32(model, &candidate_state, observation.value()),
            Candidate::Mismatch => update_mismatch32(model, &candidate_state, observation),
        };
        if let Err(error) = &production_result {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: format!("{error:?}"),
                fingerprint: fingerprint32(&production),
                validation: false,
                first_divergence: None,
            });
        }
        let production_next = match production_result {
            Ok(next) => next,
            Err(_) => unreachable!("production error already handled"),
        };
        let candidate_next = match candidate_result {
            Ok(next) => next,
            Err(error) => {
                return Err(PrecheckFailure {
                    index,
                    kind: "comparison_failure",
                    error: format!("candidate returned {error:?}"),
                    fingerprint: fingerprint32(&production),
                    validation: true,
                    first_divergence: Some(format!("candidate Err at observation {index}")),
                });
            }
        };
        if !state_eq32(&production_next, &candidate_next) {
            return Err(PrecheckFailure {
                index,
                kind: "comparison_failure",
                error: "coupled state mismatch".to_owned(),
                fingerprint: fingerprint32(&production),
                validation: true,
                first_divergence: first_divergence32(&production_next, &candidate_next),
            });
        }
        if !production_next
            .w
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
            || !production_next
                .p
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "non-finite committed state".to_owned(),
                fingerprint: fingerprint32(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        if index % 1_000 == 0 && !symmetric32(&production_next.p) {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "covariance is not bitwise symmetric".to_owned(),
                fingerprint: fingerprint32(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        production = production_next;
        candidate_state = candidate_next;
        if index + 1 == warm {
            warm_state = Some(production.clone());
        }
    }
    let warm_state = match warm_state {
        Some(state) => state,
        None => {
            return Err(PrecheckFailure {
                index: total,
                kind: "operation_error",
                error: "warm state was not captured".to_owned(),
                fingerprint: fingerprint32(&production),
                validation: false,
                first_divergence: None,
            });
        }
    };
    Ok((warm_state, production))
}

fn precheck64(
    model: &Rls64,
    observations: &[Observation<RlsSample64>],
    warm: usize,
    horizon: usize,
    candidate: Candidate,
) -> Result<(RlsState64, RlsState64), PrecheckFailure> {
    let total = warm + horizon;
    let mut production = model.initial_state();
    let mut candidate_state = production.clone();
    let mut warm_state = None;
    for (index, observation) in observations.iter().enumerate().take(total) {
        let production_result = model.update(&production, observation);
        let candidate_result = match candidate {
            Candidate::Production => production_result.clone(),
            Candidate::Fused => update_fused64(model, &candidate_state, observation.value()),
            Candidate::Symmetric => update_sym64(model, &candidate_state, observation.value()),
            Candidate::Mismatch => production_result.clone(),
        };
        if let Err(error) = &production_result {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: format!("{error:?}"),
                fingerprint: fingerprint64(&production),
                validation: false,
                first_divergence: None,
            });
        }
        let production_next = match production_result {
            Ok(next) => next,
            Err(_) => unreachable!("production error already handled"),
        };
        let candidate_next = match candidate_result {
            Ok(next) => next,
            Err(error) => {
                return Err(PrecheckFailure {
                    index,
                    kind: "comparison_failure",
                    error: format!("candidate returned {error:?}"),
                    fingerprint: fingerprint64(&production),
                    validation: true,
                    first_divergence: Some(format!("candidate Err at observation {index}")),
                });
            }
        };
        if !state_eq64(&production_next, &candidate_next) {
            return Err(PrecheckFailure {
                index,
                kind: "comparison_failure",
                error: "coupled state mismatch".to_owned(),
                fingerprint: fingerprint64(&production),
                validation: true,
                first_divergence: first_divergence64(&production_next, &candidate_next),
            });
        }
        if !production_next
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
            || !production_next
                .covariance()
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "non-finite committed state".to_owned(),
                fingerprint: fingerprint64(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        if index % 1_000 == 0 && !symmetric64(production_next.covariance()) {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "covariance is not bitwise symmetric".to_owned(),
                fingerprint: fingerprint64(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        production = production_next;
        candidate_state = candidate_next;
        if index + 1 == warm {
            warm_state = Some(production.clone());
        }
    }
    let warm_state = match warm_state {
        Some(state) => state,
        None => {
            return Err(PrecheckFailure {
                index: total,
                kind: "operation_error",
                error: "warm state was not captured".to_owned(),
                fingerprint: fingerprint64(&production),
                validation: false,
                first_divergence: None,
            });
        }
    };
    Ok((warm_state, production))
}

fn record_precheck_failure(
    row: &str,
    precision: &'static str,
    dimension: usize,
    warm: usize,
    horizon: usize,
    failure: PrecheckFailure,
) {
    let attempted = (failure.index + 1) as u64;
    let successful = if failure.validation {
        attempted
    } else {
        failure.index as u64
    };
    let failed = if failure.validation { 0 } else { 1 };
    let scheduled = (warm + horizon) as u64;
    FailureRecord {
        kind: failure.kind,
        phase: "precheck",
        row: row.to_owned(),
        precision,
        dimension,
        window: None,
        observation_global: Some(failure.index),
        observation_window: None,
        committed_progress: failure.index as u64,
        retained_fingerprint: Some(failure.fingerprint),
        scheduled_units: scheduled,
        attempted_units: attempted,
        successful_units: successful,
        failed_units: failed,
        unattempted_units: scheduled - attempted,
        validation_failure: failure.validation,
        postcondition_failure: false,
        first_divergence: failure.first_divergence,
        error: Some(failure.error),
    }
    .emit();
}

#[derive(Debug, Default, Clone, Copy)]
struct UpdateRunStatus {
    planned_windows: usize,
    valid_windows: usize,
    invalid_windows: usize,
    stopped_on_failure: bool,
    summary_eligible: bool,
    summary_emitted: bool,
}

#[allow(clippy::too_many_arguments)]
fn measure_update32(
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

fn candidate_update32(
    model: &Rls,
    state: &RlsState,
    observation: &Observation<RlsSample>,
    candidate: Candidate,
) -> Result<RlsState, RlsError> {
    match candidate {
        Candidate::Production => model.update(state, observation),
        Candidate::Fused => update_fused32(model, state, observation.value()),
        Candidate::Symmetric => update_sym32(model, state, observation.value()),
        Candidate::Mismatch => update_mismatch32(model, state, observation),
    }
}

fn candidate_update64(
    model: &Rls64,
    state: &RlsState64,
    observation: &Observation<RlsSample64>,
    candidate: Candidate,
) -> Result<RlsState64, RlsError64> {
    match candidate {
        Candidate::Production => model.update(state, observation),
        Candidate::Fused => update_fused64(model, state, observation.value()),
        Candidate::Symmetric => update_sym64(model, state, observation.value()),
        Candidate::Mismatch => model.update(state, observation),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Production,
    Fused,
    Symmetric,
    Mismatch,
}

impl Candidate {
    fn category(self) -> &'static str {
        match self {
            Self::Production => "S",
            Self::Fused | Self::Symmetric => "U",
            Self::Mismatch => "F",
        }
    }
}

fn update_mismatch32(
    model: &Rls,
    state: &RlsState,
    observation: &Observation<RlsSample>,
) -> Result<RlsState, RlsError> {
    let next = model.update(state, observation)?;
    let mut weights = next.w.as_slice().to_vec();
    if let Some(first) = weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    Ok(RlsState {
        w: Vector::from_slice(&weights),
        p: next.p,
    })
}

#[allow(clippy::too_many_arguments)]
fn measure_update64(
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

fn measure_control(
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

fn run_components32(
    model: &Rls,
    expected: &RlsState,
    last: &Observation<RlsSample>,
    raws: &mut Vec<RawWindow>,
) {
    let d = model.dimension();
    let sample = last.value();
    let v = match expected.p.mul_vector(&sample.features) {
        Ok(value) => value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f32",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint32(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    let denom = match sample.features.dot(&v) {
        Ok(value) => model.lambda() + value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f32",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint32(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    if !denom.is_finite() || denom <= 0.0 {
        FailureRecord {
            kind: "operation_error",
            phase: "control_setup",
            row: "C-Pfull".to_owned(),
            precision: "f32",
            dimension: d,
            window: None,
            observation_global: Some(warmup() + horizon() - 1),
            observation_window: Some(horizon() - 1),
            committed_progress: 0,
            retained_fingerprint: Some(fingerprint32(expected)),
            scheduled_units: 0,
            attempted_units: 0,
            successful_units: 0,
            failed_units: 0,
            unattempted_units: 0,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            error: Some("invalid control denominator".to_owned()),
        }
        .emit();
        return;
    }
    let prediction = match expected.w.dot(&sample.features) {
        Ok(value) => value,
        Err(error) => {
            control_setup_failure("C-Pfull", "f32", d, format!("{error:?}"));
            return;
        }
    };
    let error = sample.target - prediction;
    let lambda = model.lambda();
    let mut full = vec![0.0_f32; d * d];
    let _ = measure_control(
        "C-Pfull",
        "f32",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_full_into32(
                &mut full,
                expected.p.as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&full);
            Ok(())
        },
        raws,
    );
    let mut sym = vec![0.0_f32; d * d];
    let _ = measure_control(
        "C-Psym",
        "f32",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_sym_into32(
                &mut sym,
                expected.p.as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&sym);
            Ok(())
        },
        raws,
    );
    let _ = measure_control(
        "C-Vcur-alloc",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vcur_alloc32(&expected.w, &v, error, denom),
        raws,
    );
    let mut gain = vec![0.0_f32; d];
    let mut scaled = vec![0.0_f32; d];
    let mut next = vec![0.0_f32; d];
    let _ = measure_control(
        "C-Vcur-pre",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || {
            vcur_pre32(
                &expected.w,
                &v,
                error,
                denom,
                &mut gain,
                &mut scaled,
                &mut next,
            )
        },
        raws,
    );
    let _ = measure_control(
        "C-Vfused-alloc",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_alloc32(&expected.w, &v, error, denom),
        raws,
    );
    let mut fused = vec![0.0_f32; d];
    let _ = measure_control(
        "C-Vfused-pre",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_pre32(&expected.w, &v, error, denom, &mut fused),
        raws,
    );
    let _ = measure_control(
        "C-Alloc",
        "f32",
        d,
        "allocation_set",
        CONTROL_ITERS,
        || {
            let a = Vec::<f32>::with_capacity(d);
            let b = Vec::<f32>::with_capacity(d);
            let c = Vec::<f32>::with_capacity(d);
            let e = Vec::<f32>::with_capacity(d);
            let p = Vec::<f32>::with_capacity(d * d);
            black_box((&a, &b, &c, &e, &p));
            Ok(())
        },
        raws,
    );
}

fn run_components64(
    model: &Rls64,
    expected: &RlsState64,
    last: &Observation<RlsSample64>,
    raws: &mut Vec<RawWindow>,
) {
    let d = model.dimension();
    let sample = last.value();
    let v = match expected.covariance().mul_vector(&sample.features) {
        Ok(value) => value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f64",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint64(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    let denom = match sample.features.dot(&v) {
        Ok(value) => model.lambda() + value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f64",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint64(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    if !denom.is_finite() || denom <= 0.0 {
        FailureRecord {
            kind: "operation_error",
            phase: "control_setup",
            row: "C-Pfull".to_owned(),
            precision: "f64",
            dimension: d,
            window: None,
            observation_global: Some(warmup() + horizon() - 1),
            observation_window: Some(horizon() - 1),
            committed_progress: 0,
            retained_fingerprint: Some(fingerprint64(expected)),
            scheduled_units: 0,
            attempted_units: 0,
            successful_units: 0,
            failed_units: 0,
            unattempted_units: 0,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            error: Some("invalid control denominator".to_owned()),
        }
        .emit();
        return;
    }
    let error = sample.target
        - match expected.weights().dot(&sample.features) {
            Ok(value) => value,
            Err(error) => {
                FailureRecord {
                    kind: "operation_error",
                    phase: "control_setup",
                    row: "C-Pfull".to_owned(),
                    precision: "f64",
                    dimension: d,
                    window: None,
                    observation_global: Some(warmup() + horizon() - 1),
                    observation_window: Some(horizon() - 1),
                    committed_progress: 0,
                    retained_fingerprint: Some(fingerprint64(expected)),
                    scheduled_units: 0,
                    attempted_units: 0,
                    successful_units: 0,
                    failed_units: 0,
                    unattempted_units: 0,
                    validation_failure: false,
                    postcondition_failure: false,
                    first_divergence: None,
                    error: Some(format!("{error:?}")),
                }
                .emit();
                return;
            }
        };
    let lambda = model.lambda();
    let mut full = vec![0.0_f64; d * d];
    let _ = measure_control(
        "C-Pfull",
        "f64",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_full_into64(
                &mut full,
                expected.covariance().as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&full);
            Ok(())
        },
        raws,
    );
    let mut sym = vec![0.0_f64; d * d];
    let _ = measure_control(
        "C-Psym",
        "f64",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_sym_into64(
                &mut sym,
                expected.covariance().as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&sym);
            Ok(())
        },
        raws,
    );
    let _ = measure_control(
        "C-Vcur-alloc",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vcur_alloc64(expected.weights(), &v, error, denom),
        raws,
    );
    let mut gain = vec![0.0_f64; d];
    let mut scaled = vec![0.0_f64; d];
    let mut next = vec![0.0_f64; d];
    let _ = measure_control(
        "C-Vcur-pre",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || {
            vcur_pre64(
                expected.weights(),
                &v,
                error,
                denom,
                &mut gain,
                &mut scaled,
                &mut next,
            )
        },
        raws,
    );
    let _ = measure_control(
        "C-Vfused-alloc",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_alloc64(expected.weights(), &v, error, denom),
        raws,
    );
    let mut fused = vec![0.0_f64; d];
    let _ = measure_control(
        "C-Vfused-pre",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_pre64(expected.weights(), &v, error, denom, &mut fused),
        raws,
    );
    let _ = measure_control(
        "C-Alloc",
        "f64",
        d,
        "allocation_set",
        CONTROL_ITERS,
        || {
            let a = Vec::<f64>::with_capacity(d);
            let b = Vec::<f64>::with_capacity(d);
            let c = Vec::<f64>::with_capacity(d);
            let e = Vec::<f64>::with_capacity(d);
            let p = Vec::<f64>::with_capacity(d * d);
            black_box((&a, &b, &c, &e, &p));
            Ok(())
        },
        raws,
    );
}

fn expected_identities() -> BTreeSet<(String, &'static str, usize, usize)> {
    let mut set = BTreeSet::new();
    for (precision, s_row, d) in [
        ("f32", "S-C", DIMS[0]),
        ("f32", "S-C", DIMS[1]),
        ("f64", "S-A", DIMS[0]),
        ("f64", "S-A", DIMS[1]),
    ] {
        for window in 0..WINDOWS {
            set.insert((s_row.to_owned(), precision, d, window));
        }
    }
    for precision in ["f32", "f64"] {
        for d in DIMS {
            for row in [
                "U-Fused",
                "U-Sym",
                "C-Pfull",
                "C-Psym",
                "C-Vcur-alloc",
                "C-Vcur-pre",
                "C-Vfused-alloc",
                "C-Vfused-pre",
                "C-Alloc",
            ] {
                for window in 0..WINDOWS {
                    set.insert((row.to_owned(), precision, d, window));
                }
            }
        }
    }
    set
}

type RowIdentity = (String, &'static str, usize, usize);

fn catalog_report(
    expected: &BTreeSet<RowIdentity>,
    raws: &[RawWindow],
) -> (Vec<RowIdentity>, Vec<RowIdentity>, Vec<(RowIdentity, u64)>) {
    let mut counts: BTreeMap<RowIdentity, u64> = BTreeMap::new();
    for record in raws {
        if record.status != "warmup" && matches!(record.category, "S" | "U" | "C") {
            if let Some(window) = record.window_index {
                *counts
                    .entry((
                        record.row_id.clone(),
                        record.precision,
                        record.dimension,
                        window,
                    ))
                    .or_insert(0) += 1;
            }
        }
    }
    let missing = expected
        .iter()
        .filter(|identity| !counts.contains_key(*identity))
        .cloned()
        .collect();
    let extra = counts
        .keys()
        .filter(|identity| !expected.contains(*identity))
        .cloned()
        .collect();
    let duplicates = counts
        .iter()
        .filter(|(_, count)| **count != 1)
        .map(|(identity, count)| (identity.clone(), *count))
        .collect();
    (missing, extra, duplicates)
}

fn check_catalog(raws: &[RawWindow]) -> bool {
    let expected = expected_identities();
    let (missing, extra, duplicates) = catalog_report(&expected, raws);
    if !missing.is_empty() || !extra.is_empty() || !duplicates.is_empty() {
        println!("CATALOG missing={missing:?} extra={extra:?} duplicates={duplicates:?}");
        set_unexpected_failure();
        false
    } else {
        println!(
            "CATALOG expected={} observed={} status=ok",
            expected.len(),
            expected.len()
        );
        true
    }
}

fn emit_retained_records(raws: &mut [RawWindow]) -> usize {
    let mut emitted = 0;
    for record in raws {
        if record.emit_once() {
            emitted += 1;
        }
    }
    emitted
}

fn run_memory_probes32(
    model: &Rls,
    warm: &RlsState,
    last: &Observation<RlsSample>,
    raws: &mut Vec<RawWindow>,
) {
    let committed = common::comparison::memory_probe(|| model.initial_state());
    println!(
        "MEMORY precision=f32 D={} committed retained-requested-bytes={} peak-live-requested-bytes-above-baseline={}",
        model.dimension(),
        committed.retained_bytes,
        committed.peak_above_baseline
    );
    emit_memory_record(
        raws,
        "memory-committed",
        "f32",
        model.dimension(),
        Some(committed.peak_above_baseline),
        "valid",
    );
    if let Err(error) = model.update(warm, last) {
        FailureRecord {
            kind: "operation_error",
            phase: "memory_probe",
            row: "memory-candidate".to_owned(),
            precision: "f32",
            dimension: model.dimension(),
            window: None,
            observation_global: Some(warmup() + horizon() - 1),
            observation_window: Some(horizon() - 1),
            committed_progress: 0,
            retained_fingerprint: Some(fingerprint32(warm)),
            scheduled_units: 1,
            attempted_units: 1,
            successful_units: 0,
            failed_units: 1,
            unattempted_units: 0,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            error: Some(format!("{error:?}")),
        }
        .emit();
        emit_memory_record(
            raws,
            "memory-candidate",
            "f32",
            model.dimension(),
            None,
            "invalid",
        );
        return;
    }
    let candidate = common::comparison::memory_probe(|| model.update(warm, last));
    println!(
        "MEMORY precision=f32 D={} candidate retained-requested-bytes={} peak-live-requested-bytes-above-baseline={}",
        model.dimension(),
        candidate.retained_bytes,
        candidate.peak_above_baseline
    );
    emit_memory_record(
        raws,
        "memory-candidate",
        "f32",
        model.dimension(),
        Some(candidate.peak_above_baseline),
        "valid",
    );
}

fn run_memory_probes64(
    model: &Rls64,
    warm: &RlsState64,
    last: &Observation<RlsSample64>,
    raws: &mut Vec<RawWindow>,
) {
    let committed = common::comparison::memory_probe(|| model.initial_state());
    println!(
        "MEMORY precision=f64 D={} committed retained-requested-bytes={} peak-live-requested-bytes-above-baseline={}",
        model.dimension(),
        committed.retained_bytes,
        committed.peak_above_baseline
    );
    emit_memory_record(
        raws,
        "memory-committed",
        "f64",
        model.dimension(),
        Some(committed.peak_above_baseline),
        "valid",
    );
    if let Err(error) = model.update(warm, last) {
        FailureRecord {
            kind: "operation_error",
            phase: "memory_probe",
            row: "memory-candidate".to_owned(),
            precision: "f64",
            dimension: model.dimension(),
            window: None,
            observation_global: Some(warmup() + horizon() - 1),
            observation_window: Some(horizon() - 1),
            committed_progress: 0,
            retained_fingerprint: Some(fingerprint64(warm)),
            scheduled_units: 1,
            attempted_units: 1,
            successful_units: 0,
            failed_units: 1,
            unattempted_units: 0,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            error: Some(format!("{error:?}")),
        }
        .emit();
        emit_memory_record(
            raws,
            "memory-candidate",
            "f64",
            model.dimension(),
            None,
            "invalid",
        );
        return;
    }
    let candidate = common::comparison::memory_probe(|| model.update(warm, last));
    println!(
        "MEMORY precision=f64 D={} candidate retained-requested-bytes={} peak-live-requested-bytes-above-baseline={}",
        model.dimension(),
        candidate.retained_bytes,
        candidate.peak_above_baseline
    );
    emit_memory_record(
        raws,
        "memory-candidate",
        "f64",
        model.dimension(),
        Some(candidate.peak_above_baseline),
        "valid",
    );
}

fn control_setup_failure(row: &str, precision: &'static str, dimension: usize, error: String) {
    FailureRecord {
        kind: "operation_error",
        phase: "control_setup",
        row: row.to_owned(),
        precision,
        dimension,
        window: None,
        observation_global: Some(warmup() + horizon() - 1),
        observation_window: Some(horizon() - 1),
        committed_progress: 0,
        retained_fingerprint: None,
        scheduled_units: 0,
        attempted_units: 0,
        successful_units: 0,
        failed_units: 0,
        unattempted_units: 0,
        validation_failure: false,
        postcondition_failure: false,
        first_divergence: None,
        error: Some(error),
    }
    .emit();
}

fn diag_check(label: &str, condition: bool, detail: &str) -> bool {
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

fn make_model32(d: usize, lambda: f32, delta: f32) -> Option<Rls> {
    match Rls::new(d, lambda, delta) {
        Ok(model) => Some(model),
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL Rls::new {error:?}");
            set_unexpected_failure();
            None
        }
    }
}

fn make_model64(d: usize, lambda: f64, delta: f64) -> Option<Rls64> {
    match Rls64::new(d, lambda, delta) {
        Ok(model) => Some(model),
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL Rls64::new {error:?}");
            set_unexpected_failure();
            None
        }
    }
}

fn sample32(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn sample64(features: &[f64], target: f64) -> Observation<RlsSample64> {
    Observation::new(RlsSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

fn invalid_state32() -> RlsState {
    RlsState {
        w: Vector::zeros(2),
        p: Matrix::from_fn(2, 2, |row, col| {
            if row == col {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            }
        }),
    }
}

fn invalid_state64() -> Option<RlsState64> {
    match RlsState64::admit(
        Vector64::zeros(2),
        Matrix64::from_fn(2, 2, |row, col| {
            if row == col {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            }
        }),
    ) {
        Ok(state) => Some(state),
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL invalid_state64 {error:?}");
            set_unexpected_failure();
            None
        }
    }
}

fn legacy_model(features: usize, lambda: f32) -> Rls {
    match Rls::new(features, lambda, DELTA) {
        Ok(model) => model,
        Err(error) => {
            println!("FAILURE legacy model construction {error:?}");
            set_unexpected_failure();
            std::process::exit(1);
        }
    }
}

fn control_observation(features: usize) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_fn(features, |index| (index as f32 * 0.31).sin()),
        target: 0.25,
    })
}

fn workload_observations(xs: &[Vec<f32>], ys: &[f32]) -> Vec<Observation<RlsSample>> {
    xs.iter()
        .zip(ys)
        .map(|(x, &target)| {
            Observation::new(RlsSample {
                features: Vector::from_slice(x),
                target,
            })
        })
        .collect()
}

fn control_steps(features: usize) -> u32 {
    if legacy_validating() {
        return BATCH as u32;
    }
    common::timed_steps(match features {
        8 => 200_000,
        32 => 50_000,
        128 => 5_000,
        _ => 2_000,
    })
}

fn workload_steps(features: usize) -> u32 {
    if legacy_validating() {
        return BATCH as u32;
    }
    common::timed_steps(match features {
        8 => 200_000,
        32 => 50_000,
        64 => 20_000,
        128 => 5_000,
        _ => 2_000,
    })
}

fn evidence_steps(features: usize) -> u32 {
    if legacy_validating() {
        return 2;
    }
    match features {
        32 => 50_000,
        64 => 20_000,
        128 => 5_000,
        256 => 2_000,
        _ => 1_000,
    }
}

#[derive(Debug)]
struct LegacyFailure {
    attempted: u64,
    committed: u64,
    message: String,
    retained_fingerprint: Option<u64>,
}

fn failure_id(row: &str, phase: &str, window: Option<usize>, observation: Option<usize>) -> String {
    format!("{row}:{phase}:w={window:?}:o={observation:?}")
}

#[allow(clippy::too_many_arguments)]
fn measure_legacy(
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
fn measure_legacy_impl(
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

#[derive(Debug, Clone)]
struct SharingFailure {
    stream: &'static str,
    a_attempted: u64,
    a_committed: u64,
    b_attempted: u64,
    b_committed: u64,
    retained_a: u64,
    retained_b: u64,
    message: String,
}

#[allow(clippy::too_many_arguments)]
fn measure_sharing(
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

fn sweep_outer() -> usize {
    if legacy_validating() {
        return 8;
    }
    if cfg!(debug_assertions) {
        8
    } else {
        SWEEP_OUTER_RELEASE
    }
}

fn sweep_batch_sizes() -> &'static [usize] {
    if legacy_validating() {
        return &SWEEP_BATCH_SIZES[..2];
    }
    if cfg!(debug_assertions) {
        &SWEEP_BATCH_SIZES[..2]
    } else {
        &SWEEP_BATCH_SIZES
    }
}

fn sweep_steps(features: usize) -> u32 {
    if legacy_validating() {
        return 1;
    }
    if cfg!(debug_assertions) {
        4
    } else {
        match features {
            8 => 32,
            _ => 16,
        }
    }
}

fn sweep_fixture(features: usize) -> Vec<Observation<RlsSample>> {
    let xs = common::persistent_excitation_features(features, SWEEP_FIXTURE_LEN);
    let weights = common::fixed_weights(features);
    let ys = common::linear_targets(&xs, &weights);
    workload_observations(&xs, &ys)
}

fn legacy_horizon_plan(steps: u32) -> usize {
    let warm_up_calls = (steps as usize / 10).max(1000);
    warm_up_calls + common::RUNS * steps as usize
}

fn legacy_horizon(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    outer: usize,
    total_outer_calls: usize,
) -> Result<(), OperationEvidence> {
    let scheduled = (total_outer_calls * outer) as u64;
    let mut state = model.initial_state();
    let mut index = 0_usize;
    for call in 0..total_outer_calls {
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        let committed_state = state.clone();
        state = match model.process_batch_reference(state, window, outer) {
            Ok(next) => next,
            Err(error) => {
                let (committed, attempted, fingerprint, message) = match &error {
                    RlsBatchError::Transition {
                        failed_index,
                        state,
                        error,
                    } => {
                        let committed = (call * outer + failed_index) as u64;
                        (
                            committed,
                            committed + 1,
                            fingerprint32(state),
                            format!("observation={committed} {error:?} (outer call {call})"),
                        )
                    }
                    other => {
                        let committed = (call * outer) as u64;
                        (
                            committed,
                            committed,
                            fingerprint32(&committed_state),
                            format!("observation={committed} {other:?} (outer call {call})"),
                        )
                    }
                };
                return Err(OperationEvidence {
                    scheduled,
                    attempted,
                    committed,
                    retained_fingerprint: Some(fingerprint),
                    message,
                });
            }
        };
        if !state.w.as_slice().iter().all(|value| value.is_finite())
            || !state.p.as_slice().iter().all(|value| value.is_finite())
        {
            let committed = ((call + 1) * outer) as u64;
            return Err(OperationEvidence {
                scheduled,
                attempted: committed,
                committed,
                retained_fingerprint: Some(fingerprint32(&state)),
                message: format!("non-finite state at outer call {call}"),
            });
        }
        index += outer;
    }
    println!(
        "  legacy horizon validation PASS: D={} obs={} horizon={}",
        model.dimension(),
        scheduled,
        if cfg!(debug_assertions) || legacy_validating() {
            "shortened"
        } else {
            "declared"
        }
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_legacy_fixed_control(
    raws: &mut Vec<RawWindow>,
    fail_at: Option<usize>,
    expected_failure: bool,
    out_failure: &mut Option<FailureRecord>,
    out_raw: &mut Option<RawWindow>,
) {
    println!("\n== control (fixed observation, lambda=1) ==");
    for features in [8_usize, 32, 128, 256] {
        let steps = control_steps(features) as usize;
        let observation = control_observation(features);
        let batch = vec![observation.clone(); BATCH];
        println!("features={features} bounded_batch={BATCH} steps={steps}");

        let reference = legacy_model(features, LAMBDA_CONTROL);
        let mut reference_state = reference.initial_state();
        let invalid_control = Observation::new(RlsSample {
            features: Vector::from_fn(features, |_| f32::NAN),
            target: 0.0,
        });
        let reference_result = measure_legacy_impl(
            "control-reference",
            "f32",
            features,
            "update",
            1,
            steps,
            true,
            if legacy_validating() {
                0
            } else {
                WARMUP_WINDOWS
            },
            !legacy_validating(),
            expected_failure,
            out_failure,
            out_raw,
            |call| {
                if fail_at == Some(call) {
                    return match reference.update(&reference_state, &invalid_control) {
                        Ok(next) => {
                            reference_state = next;
                            Ok(1)
                        }
                        Err(error) => Err(LegacyFailure {
                            retained_fingerprint: Some(fingerprint32(&reference_state)),
                            attempted: 1,
                            committed: 0,
                            message: format!("{error:?}"),
                        }),
                    };
                }
                match reference.update(&reference_state, &observation) {
                    Ok(next) => {
                        reference_state = next;
                        Ok(1)
                    }
                    Err(error) => Err(LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&reference_state)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    }),
                }
            },
            raws,
        );
        if fail_at.is_some() && reference_result.is_none() {
            return;
        }

        let streaming_model = legacy_model(features, LAMBDA_CONTROL);
        let mut executor =
            StreamingExecutor::new(&streaming_model, streaming_model.initial_state());
        let _ = measure_legacy(
            "control-streaming",
            "f32",
            features,
            "update",
            1,
            steps,
            true,
            |_| match executor.process_one(&observation) {
                Ok(_) => Ok(1),
                Err(error) => Err(LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(executor.state())),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                }),
            },
            raws,
        );

        let fold_model = legacy_model(features, LAMBDA_CONTROL);
        let mut fold_state = fold_model.initial_state();
        let _ = measure_legacy(
            "control-bounded-fold",
            "f32",
            features,
            "update",
            BATCH as u64,
            steps / BATCH,
            true,
            |_| match process_batch(&fold_model, fold_state.clone(), batch.iter().cloned()) {
                Ok(next) => {
                    fold_state = next;
                    Ok(BATCH as u64)
                }
                Err(error) => Err(LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(&fold_state)),
                    attempted: BATCH as u64,
                    committed: 0,
                    message: format!("{error:?}"),
                }),
            },
            raws,
        );
    }
}

fn sweep_state_after_outer_calls(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    outer: usize,
    batch: usize,
    outer_calls: usize,
    foundation: bool,
) -> Result<RlsState, OperationEvidence> {
    let scheduled = (outer_calls * outer) as u64;
    let mut state = model.initial_state();
    let mut index = 0_usize;
    for _ in 0..outer_calls {
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        let mut window_committed = 0_usize;
        for chunk in window.chunks(batch) {
            let chunk_start = index + window_committed;
            let committed_state = state.clone();
            if foundation {
                let attempted_in_chunk = Cell::new(0_u64);
                let observations_in_chunk = chunk
                    .iter()
                    .inspect(|_| attempted_in_chunk.set(attempted_in_chunk.get() + 1))
                    .cloned();
                match process_batch(model, state, observations_in_chunk) {
                    Ok(next) => {
                        state = next;
                        window_committed += chunk.len();
                    }
                    Err(error) => {
                        let BatchFailure { state, .. } = &error;
                        let committed =
                            chunk_start as u64 + attempted_in_chunk.get().saturating_sub(1);
                        return Err(OperationEvidence {
                            scheduled,
                            attempted: committed + 1,
                            committed,
                            retained_fingerprint: Some(fingerprint32(state)),
                            message: format!(
                                "observation={committed} {error:?} (foundation chunk start={chunk_start})"
                            ),
                        });
                    }
                }
            } else {
                match model.process_batch_reference(state, chunk, batch) {
                    Ok(next) => {
                        state = next;
                        window_committed += chunk.len();
                    }
                    Err(error) => {
                        let (committed, attempted, fingerprint, message) = match &error {
                            RlsBatchError::Transition {
                                failed_index,
                                state,
                                error,
                            } => {
                                let committed = (chunk_start + failed_index) as u64;
                                (
                                    committed,
                                    committed + 1,
                                    fingerprint32(state),
                                    format!(
                                        "observation={committed} {error:?} (chunk start={chunk_start})"
                                    ),
                                )
                            }
                            other => {
                                let committed = chunk_start as u64;
                                (
                                    committed,
                                    committed,
                                    fingerprint32(&committed_state),
                                    format!("observation={committed} {other:?}"),
                                )
                            }
                        };
                        return Err(OperationEvidence {
                            scheduled,
                            attempted,
                            committed,
                            retained_fingerprint: Some(fingerprint),
                            message,
                        });
                    }
                }
            }
        }
        index += outer;
    }
    Ok(state)
}

#[derive(Debug, Clone)]
struct ComparisonEvidence {
    left_path: String,
    left_fingerprint: u64,
    right_path: String,
    right_fingerprint: u64,
    first_difference: String,
    context: String,
    scheduled: u64,
    committed: u64,
}

impl ComparisonEvidence {
    fn message(&self) -> String {
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
struct OperationEvidence {
    scheduled: u64,
    attempted: u64,
    committed: u64,
    retained_fingerprint: Option<u64>,
    message: String,
}

#[derive(Debug)]
enum SweepFailure {
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
fn sweep_comparison_failure(
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

fn sweep_failure_record(row: &str, features: usize, failure: SweepFailure) -> FailureRecord {
    match failure {
        SweepFailure::Comparison(evidence) => comparison_failure_record(row, features, *evidence),
        SweepFailure::Operation(failure) => operation_failure_record(row, features, failure),
    }
}

fn comparison_failure_record(
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

fn operation_failure_record(
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

#[derive(Debug, Default)]
struct SweepInjection {
    feature: usize,
    expected: bool,
    fail_observation: Option<usize>,
    comparison_invocations: Cell<u64>,
    measurement_invocations: Cell<u64>,
    stopped_on_precheck: Cell<bool>,
    failure_record: RefCell<Option<FailureRecord>>,
}

fn count_comparison(injection: Option<&SweepInjection>) {
    if let Some(injection) = injection {
        injection
            .comparison_invocations
            .set(injection.comparison_invocations.get() + 1);
    }
}

/// Diagnostic-only controlled mismatch: flips the lowest weight bit so the
/// bitwise comparison fails while both trajectories remain finite.
fn one_bit_mismatch(state: &RlsState) -> RlsState {
    let mut weights = state.w.as_slice().to_vec();
    if let Some(first) = weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    RlsState {
        w: Vector::from_slice(&weights),
        p: state.p.clone(),
    }
}

fn sweep_equivalence(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    outer: usize,
    batch_sizes: &[usize],
    outer_calls: usize,
    injection: Option<&SweepInjection>,
) -> Result<(), SweepFailure> {
    let mut executor = StreamingExecutor::new(model, model.initial_state());
    let scheduled_grouped = (outer_calls * outer) as u64;
    let mut committed = 0_u64;
    let mut index = 0_usize;
    for _ in 0..outer_calls {
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        for observation in window {
            if let Err(error) = executor.process_one(observation) {
                return Err(OperationEvidence {
                    scheduled: scheduled_grouped,
                    attempted: committed + 1,
                    committed,
                    retained_fingerprint: Some(fingerprint32(executor.state())),
                    message: format!("observation={committed} {error:?} (grouped)"),
                }
                .into());
            }
            committed += 1;
        }
        index += outer;
    }
    let grouped = executor.state().clone();
    let mut first_bounded: Option<(usize, RlsState)> = None;
    for &batch in batch_sizes {
        let batched =
            sweep_state_after_outer_calls(model, observations, outer, batch, outer_calls, false)?;
        let folded =
            sweep_state_after_outer_calls(model, observations, outer, batch, outer_calls, true)?;
        count_comparison(injection);
        let compared_batched = match injection {
            Some(_) => one_bit_mismatch(&batched),
            None => batched.clone(),
        };
        let context = if injection.is_some() {
            format!("B={batch} outer_calls={outer_calls} diagnostic one-bit mismatch injection")
        } else {
            format!("B={batch} outer_calls={outer_calls}")
        };
        if !state_eq32(&grouped, &compared_batched) {
            return Err(sweep_comparison_failure(
                "grouped",
                &grouped,
                "bounded-reference",
                &compared_batched,
                &context,
                (outer_calls * outer) as u64,
            )
            .into());
        }
        count_comparison(injection);
        if !state_eq32(&grouped, &folded) {
            return Err(sweep_comparison_failure(
                "grouped",
                &grouped,
                "foundation-fold",
                &folded,
                &format!("B={batch} outer_calls={outer_calls}"),
                (outer_calls * outer) as u64,
            )
            .into());
        }
        match &first_bounded {
            None => first_bounded = Some((batch, batched)),
            Some((first_batch, reference)) => {
                count_comparison(injection);
                if !state_eq32(reference, &batched) {
                    return Err(sweep_comparison_failure(
                        &format!("bounded-reference B={first_batch}"),
                        reference,
                        &format!("bounded-reference B={batch}"),
                        &batched,
                        &format!("cross-batch continuity outer_calls={outer_calls}"),
                        (outer_calls * outer) as u64,
                    )
                    .into());
                }
            }
        }
    }

    let segment = &observations[..32.min(observations.len())];
    let segment_scheduled = segment.len() as u64;
    let segment_committed = model.initial_state();
    let single = model
        .process_batch_reference(segment_committed.clone(), segment, segment.len())
        .map_err(|error| match &error {
            RlsBatchError::Transition {
                failed_index,
                state,
                error,
            } => OperationEvidence {
                scheduled: segment_scheduled,
                attempted: *failed_index as u64 + 1,
                committed: *failed_index as u64,
                retained_fingerprint: Some(fingerprint32(state)),
                message: format!("observation={failed_index} {error:?} (single segment)"),
            },
            other => OperationEvidence {
                scheduled: segment_scheduled,
                attempted: 0,
                committed: 0,
                retained_fingerprint: Some(fingerprint32(&segment_committed)),
                message: format!("observation=0 {other:?} (single segment)"),
            },
        })?;
    let mut chunked = model.initial_state();
    let mut start = 0_usize;
    for &size in &[2_usize, 3, 5, 7, 15] {
        let end = (start + size).min(segment.len());
        let committed_state = chunked.clone();
        chunked = model
            .process_batch_reference(chunked, &segment[start..end], end - start)
            .map_err(|error| match &error {
                RlsBatchError::Transition {
                    failed_index,
                    state,
                    error,
                } => {
                    let committed = (start + failed_index) as u64;
                    OperationEvidence {
                        scheduled: segment_scheduled,
                        attempted: committed + 1,
                        committed,
                        retained_fingerprint: Some(fingerprint32(state)),
                        message: format!("observation={committed} {error:?} (chunk start={start})"),
                    }
                }
                other => OperationEvidence {
                    scheduled: segment_scheduled,
                    attempted: start as u64,
                    committed: start as u64,
                    retained_fingerprint: Some(fingerprint32(&committed_state)),
                    message: format!("observation={start} {other:?} (chunk start={start})"),
                },
            })?;
        start = end;
        if start == segment.len() {
            break;
        }
    }
    count_comparison(injection);
    if !state_eq32(&single, &chunked) {
        return Err(sweep_comparison_failure(
            "single-pass segment",
            &single,
            "chunked segment",
            &chunked,
            "segment=0..32 chunk_sizes=[2,3,5,7,15]",
            segment.len() as u64,
        )
        .into());
    }

    let wrap_scheduled = (2 * outer) as u64;
    let mut wrap_grouped = StreamingExecutor::new(model, model.initial_state());
    let mut wrap_batched = model.initial_state();
    let mut index = observations.len().saturating_sub(outer);
    let mut wrap_committed = 0_u64;
    for _ in 0..2 {
        let window_base = wrap_committed;
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        for observation in window {
            if let Err(error) = wrap_grouped.process_one(observation) {
                return Err(OperationEvidence {
                    scheduled: wrap_scheduled,
                    attempted: wrap_committed + 1,
                    committed: wrap_committed,
                    retained_fingerprint: Some(fingerprint32(wrap_grouped.state())),
                    message: format!("observation={wrap_committed} {error:?} (wrap grouped)"),
                }
                .into());
            }
            wrap_committed += 1;
        }
        let committed_state = wrap_batched.clone();
        wrap_batched = match model.process_batch_reference(wrap_batched, window, outer) {
            Ok(next) => next,
            Err(error) => {
                let (committed, attempted, fingerprint, message) = match &error {
                    RlsBatchError::Transition {
                        failed_index,
                        state,
                        error,
                    } => {
                        let committed = window_base + *failed_index as u64;
                        (
                            committed,
                            committed + 1,
                            fingerprint32(state),
                            format!("observation={committed} {error:?} (wrap bounded)"),
                        )
                    }
                    other => (
                        window_base,
                        window_base,
                        fingerprint32(&committed_state),
                        format!("observation={window_base} {other:?} (wrap bounded)"),
                    ),
                };
                return Err(OperationEvidence {
                    scheduled: wrap_scheduled,
                    attempted,
                    committed,
                    retained_fingerprint: Some(fingerprint),
                    message,
                }
                .into());
            }
        };
        index += outer;
    }
    count_comparison(injection);
    if !state_eq32(wrap_grouped.state(), &wrap_batched) {
        return Err(sweep_comparison_failure(
            "grouped wrap",
            wrap_grouped.state(),
            "bounded wrap",
            &wrap_batched,
            &format!(
                "outer={outer} wrap_start={} crossings=2",
                observations.len().saturating_sub(outer)
            ),
            (2 * outer) as u64,
        )
        .into());
    }
    Ok(())
}

fn run_legacy_batch_sweep(raws: &mut Vec<RawWindow>, injection: Option<&SweepInjection>) {
    let outer = sweep_outer();
    let batch_sizes = sweep_batch_sizes();
    println!(
        "\n== bounded reference batch-size sensitivity (lambda={}, outer={outer}) ==",
        common::LAMBDA_PE
    );
    for features in SWEEP_DIMENSIONS {
        let inject_here = injection.filter(|injection| injection.feature == features);
        if injection.is_some() && inject_here.is_none() {
            continue;
        }
        let model = legacy_model(features, common::LAMBDA_PE);
        let mut observations = sweep_fixture(features);
        if let Some(index) = injection.and_then(|injection| injection.fail_observation) {
            if index < observations.len() {
                observations[index] = sample32(&vec![f32::NAN; features], 0.0);
            }
        }
        if let Err(failure) = sweep_equivalence(
            &model,
            &observations,
            outer,
            batch_sizes,
            if cfg!(debug_assertions) { 2 } else { 4 },
            inject_here,
        ) {
            let record = sweep_failure_record("batch-sweep-equivalence", features, failure);
            if let Some(injection) = inject_here {
                *injection.failure_record.borrow_mut() = Some(record.clone());
                injection.stopped_on_precheck.set(true);
            }
            if injection.is_some_and(|injection| injection.expected) {
                record.emit_observed();
            } else {
                record.emit();
            }
            continue;
        }
        if let Err(failure) = legacy_horizon(
            &model,
            &observations,
            outer,
            legacy_horizon_plan(sweep_steps(features)),
        ) {
            sweep_failure_record("batch-sweep", features, failure.into()).emit();
            continue;
        }
        let steps = sweep_steps(features) as usize;

        if let Some(injection) = injection {
            injection
                .measurement_invocations
                .set(injection.measurement_invocations.get() + 1);
        }
        let mut grouped = StreamingExecutor::new(&model, model.initial_state());
        let mut grouped_index = 0_usize;
        let _ = measure_legacy(
            "sweep-grouped",
            "f32",
            features,
            "update",
            outer as u64,
            steps,
            true,
            |_| {
                let offset = grouped_index % observations.len();
                let window = &observations[offset..offset + outer];
                for observation in window {
                    match grouped.process_one(black_box(observation)) {
                        Ok(_) => {}
                        Err(error) => {
                            return Err(LegacyFailure {
                                retained_fingerprint: Some(fingerprint32(grouped.state())),
                                attempted: outer as u64,
                                committed: 0,
                                message: format!("{error:?}"),
                            });
                        }
                    }
                }
                grouped_index += outer;
                Ok(outer as u64)
            },
            raws,
        );

        for &batch in batch_sizes {
            let mut state = Some(model.initial_state());
            let mut index = 0_usize;
            if let Some(injection) = injection {
                injection
                    .measurement_invocations
                    .set(injection.measurement_invocations.get() + 1);
            }
            let _ = measure_legacy(
                "sweep-bounded-ref",
                "f32",
                features,
                "update",
                outer as u64,
                steps,
                true,
                |_| {
                    let offset = index % observations.len();
                    let window = &observations[offset..offset + outer];
                    let mut carried = match state.take() {
                        Some(state) => state,
                        None => {
                            return Err(LegacyFailure {
                                retained_fingerprint: None,
                                attempted: outer as u64,
                                committed: 0,
                                message: "missing bounded state".to_owned(),
                            });
                        }
                    };
                    for chunk in window.chunks(batch) {
                        let committed = carried.clone();
                        match model.process_batch_reference(carried, chunk, batch) {
                            Ok(next) => carried = next,
                            Err(error) => {
                                let fingerprint = match &error {
                                    RlsBatchError::Transition { state, .. } => fingerprint32(state),
                                    _ => fingerprint32(&committed),
                                };
                                return Err(LegacyFailure {
                                    retained_fingerprint: Some(fingerprint),
                                    attempted: outer as u64,
                                    committed: 0,
                                    message: format!("{error:?}"),
                                });
                            }
                        }
                    }
                    state = Some(carried);
                    index += outer;
                    Ok(outer as u64)
                },
                raws,
            );
            let mut foundation_state = Some(model.initial_state());
            let mut foundation_index = 0_usize;
            if let Some(injection) = injection {
                injection
                    .measurement_invocations
                    .set(injection.measurement_invocations.get() + 1);
            }
            let _ = measure_legacy(
                "sweep-foundation-fold",
                "f32",
                features,
                "update",
                outer as u64,
                steps,
                true,
                |_| {
                    let offset = foundation_index % observations.len();
                    let window = &observations[offset..offset + outer];
                    let mut carried = match foundation_state.take() {
                        Some(state) => state,
                        None => {
                            return Err(LegacyFailure {
                                retained_fingerprint: None,
                                attempted: outer as u64,
                                committed: 0,
                                message: "missing foundation state".to_owned(),
                            });
                        }
                    };
                    for chunk in window.chunks(batch) {
                        match process_batch(&model, carried, chunk.iter().cloned()) {
                            Ok(next) => carried = next,
                            Err(error) => {
                                let BatchFailure { state, .. } = &error;
                                let fingerprint = fingerprint32(state);
                                return Err(LegacyFailure {
                                    retained_fingerprint: Some(fingerprint),
                                    attempted: outer as u64,
                                    committed: 0,
                                    message: format!("{error:?}"),
                                });
                            }
                        }
                    }
                    foundation_state = Some(carried);
                    foundation_index += outer;
                    Ok(outer as u64)
                },
                raws,
            );
        }
    }
}

fn run_legacy_workloads(raws: &mut Vec<RawWindow>) {
    let dimensions: Vec<usize> = if legacy_validating() {
        vec![8, 32]
    } else {
        vec![8, 32, 64, 128, 256]
    };
    for workload in ["A", "B", "C"] {
        println!("\n== workload {workload} (lambda={}) ==", common::LAMBDA_PE);
        for &features in &dimensions {
            let steps = workload_steps(features) as usize;
            let xs = match workload {
                "A" => common::persistent_excitation_features(features, 2048),
                "B" => common::structured_features(
                    features,
                    2048,
                    common::AR_COEFFICIENT,
                    common::STRUCTURED_SEED,
                ),
                _ => common::persistent_excitation_features(features, steps),
            };
            let weights = common::fixed_weights(features);
            let ys = if workload == "B" {
                let mut shifted = weights.clone();
                shifted[0] += 4.0;
                common::regime_targets(&xs, &weights, &shifted, xs.len() / 2)
            } else {
                common::linear_targets(&xs, &weights)
            };
            let observations = workload_observations(&xs, &ys);
            let reference = legacy_model(features, common::LAMBDA_PE);
            let mut state = reference.initial_state();
            let mut index = 0_usize;
            let _ = measure_legacy(
                &format!("workload-{workload}-reference"),
                "f32",
                features,
                "update",
                1,
                steps,
                true,
                |_| {
                    let observation = &observations[index % observations.len()];
                    index += 1;
                    match reference.update(&state, observation) {
                        Ok(next) => {
                            state = next;
                            Ok(1)
                        }
                        Err(error) => Err(LegacyFailure {
                            retained_fingerprint: Some(fingerprint32(&state)),
                            attempted: 1,
                            committed: 0,
                            message: format!("{error:?}"),
                        }),
                    }
                },
                raws,
            );

            let streaming_model = legacy_model(features, common::LAMBDA_PE);
            let mut executor =
                StreamingExecutor::new(&streaming_model, streaming_model.initial_state());
            let mut stream_index = 0_usize;
            let _ = measure_legacy(
                &format!("workload-{workload}-streaming"),
                "f32",
                features,
                "update",
                1,
                steps,
                true,
                |_| {
                    let observation = &observations[stream_index % observations.len()];
                    stream_index += 1;
                    match executor.process_one(observation) {
                        Ok(_) => Ok(1),
                        Err(error) => Err(LegacyFailure {
                            retained_fingerprint: Some(fingerprint32(executor.state())),
                            attempted: 1,
                            committed: 0,
                            message: format!("{error:?}"),
                        }),
                    }
                },
                raws,
            );

            if workload == "A" {
                let fold_model = legacy_model(features, common::LAMBDA_PE);
                let mut fold_state = fold_model.initial_state();
                let mut fold_index = 0_usize;
                let _ = measure_legacy(
                    "workload-A-bounded-fold",
                    "f32",
                    features,
                    "update",
                    BATCH as u64,
                    steps / BATCH,
                    true,
                    |_| {
                        let total = observations.len();
                        let mut window = Vec::with_capacity(BATCH);
                        for offset in 0..BATCH {
                            window.push(observations[(fold_index + offset) % total].clone());
                        }
                        fold_index += BATCH;
                        match process_batch(&fold_model, fold_state.clone(), window) {
                            Ok(next) => {
                                fold_state = next;
                                Ok(BATCH as u64)
                            }
                            Err(error) => Err(LegacyFailure {
                                retained_fingerprint: Some(fingerprint32(&fold_state)),
                                attempted: BATCH as u64,
                                committed: 0,
                                message: format!("{error:?}"),
                            }),
                        }
                    },
                    raws,
                );

                let shared = legacy_model(features, common::LAMBDA_PE);
                let mut shared_a = shared.initial_state();
                let mut shared_b = shared.initial_state();
                let mut shared_index = 0_usize;
                let mut ignored_failure = None;
                let mut ignored_record = None;
                let _ = measure_sharing(
                    "sharing-1model",
                    "f32",
                    features,
                    steps,
                    if legacy_validating() {
                        0
                    } else {
                        WARMUP_WINDOWS
                    },
                    !legacy_validating(),
                    false,
                    &mut ignored_failure,
                    &mut ignored_record,
                    |_| {
                        let observation = &observations[shared_index % observations.len()];
                        shared_index += 1;
                        match process_one(&shared, &shared_a, observation) {
                            Ok(next_a) => {
                                shared_a = next_a;
                                match process_one(&shared, &shared_b, observation) {
                                    Ok(next_b) => {
                                        shared_b = next_b;
                                        Ok((1, 1))
                                    }
                                    Err(error) => Err(SharingFailure {
                                        stream: "B",
                                        a_attempted: 1,
                                        a_committed: 1,
                                        b_attempted: 1,
                                        b_committed: 0,
                                        retained_a: fingerprint32(&shared_a),
                                        retained_b: fingerprint32(&shared_b),
                                        message: format!("{error:?}"),
                                    }),
                                }
                            }
                            Err(error) => Err(SharingFailure {
                                stream: "A",
                                a_attempted: 1,
                                a_committed: 0,
                                b_attempted: 0,
                                b_committed: 0,
                                retained_a: fingerprint32(&shared_a),
                                retained_b: fingerprint32(&shared_b),
                                message: format!("{error:?}"),
                            }),
                        }
                    },
                    raws,
                );

                let model_a = legacy_model(features, common::LAMBDA_PE);
                let model_b = legacy_model(features, common::LAMBDA_PE);
                let mut separate_a = model_a.initial_state();
                let mut separate_b = model_b.initial_state();
                let mut separate_index = 0_usize;
                let mut ignored_failure = None;
                let mut ignored_record = None;
                let _ = measure_sharing(
                    "sharing-2models",
                    "f32",
                    features,
                    steps,
                    if legacy_validating() {
                        0
                    } else {
                        WARMUP_WINDOWS
                    },
                    !legacy_validating(),
                    false,
                    &mut ignored_failure,
                    &mut ignored_record,
                    |_| {
                        let observation = &observations[separate_index % observations.len()];
                        separate_index += 1;
                        match process_one(&model_a, &separate_a, observation) {
                            Ok(next_a) => {
                                separate_a = next_a;
                                match process_one(&model_b, &separate_b, observation) {
                                    Ok(next_b) => {
                                        separate_b = next_b;
                                        Ok((1, 1))
                                    }
                                    Err(error) => Err(SharingFailure {
                                        stream: "B",
                                        a_attempted: 1,
                                        a_committed: 1,
                                        b_attempted: 1,
                                        b_committed: 0,
                                        retained_a: fingerprint32(&separate_a),
                                        retained_b: fingerprint32(&separate_b),
                                        message: format!("{error:?}"),
                                    }),
                                }
                            }
                            Err(error) => Err(SharingFailure {
                                stream: "A",
                                a_attempted: 1,
                                a_committed: 0,
                                b_attempted: 0,
                                b_committed: 0,
                                retained_a: fingerprint32(&separate_a),
                                retained_b: fingerprint32(&separate_b),
                                message: format!("{error:?}"),
                            }),
                        }
                    },
                    raws,
                );
            }
        }
    }
}

fn run_legacy_evidence(raws: &mut Vec<RawWindow>) {
    println!("\n== evidence: RLS P' allocation vs computation ==");
    let evidence_dimensions: Vec<usize> = if legacy_validating() {
        vec![32]
    } else {
        vec![32, 64, 128, 256, 512]
    };
    for features in evidence_dimensions {
        let steps = evidence_steps(features) as usize;
        let model = legacy_model(features, common::LAMBDA_PE);
        let xs = common::persistent_excitation_features(features, 64);
        let weights = common::fixed_weights(features);
        let samples: Vec<Observation<RlsSample>> = xs
            .iter()
            .map(|x| {
                let target = x.iter().zip(&weights).map(|(a, b)| a * b).sum();
                Observation::new(RlsSample {
                    features: Vector::from_slice(x),
                    target,
                })
            })
            .collect();
        let mut state = model.initial_state();
        let mut warm_failed = false;
        for (index, sample) in samples.iter().take(16).enumerate() {
            match model.update(&state, sample) {
                Ok(next) => state = next,
                Err(error) => {
                    FailureRecord {
                        kind: "operation_error",
                        phase: "precheck",
                        row: "evidence-warmup".to_owned(),
                        precision: "f32",
                        dimension: features,
                        window: None,
                        observation_global: None,
                        observation_window: Some(index),
                        committed_progress: index as u64,
                        retained_fingerprint: Some(fingerprint32(&state)),
                        scheduled_units: 16,
                        attempted_units: (index + 1) as u64,
                        successful_units: index as u64,
                        failed_units: 1,
                        unattempted_units: 15 - index as u64,
                        validation_failure: false,
                        postcondition_failure: false,
                        first_divergence: None,
                        error: Some(format!("{error:?}")),
                    }
                    .emit();
                    warm_failed = true;
                    break;
                }
            }
        }
        if warm_failed {
            continue;
        }
        let p_fixed = state.p.clone();
        let sample_features = Vector::from_slice(&xs[0]);
        let v_fixed = match p_fixed.mul_vector(&sample_features) {
            Ok(value) => value,
            Err(error) => {
                println!("FAILURE evidence mul_vector {error:?}");
                set_unexpected_failure();
                continue;
            }
        };
        let denominator_fixed = match sample_features.dot(&v_fixed) {
            Ok(value) => model.lambda() + value,
            Err(error) => {
                println!("FAILURE evidence dot {error:?}");
                set_unexpected_failure();
                continue;
            }
        };
        let mut index = 0_usize;
        let full = measure_legacy(
            "rls-full",
            "f32",
            features,
            "update",
            1,
            steps,
            true,
            |_| {
                let observation = &samples[index % samples.len()];
                index += 1;
                match model.update(&state, observation) {
                    Ok(next) => {
                        state = next;
                        Ok(1)
                    }
                    Err(error) => Err(LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&state)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    }),
                }
            },
            raws,
        );
        let mut out_full = vec![0.0_f32; features * features];
        let _ = measure_control(
            "p-full-compute",
            "f32",
            features,
            "buffer_computation",
            steps,
            || {
                p_full_into32(
                    &mut out_full,
                    p_fixed.as_slice(),
                    v_fixed.as_slice(),
                    features,
                    common::LAMBDA_PE,
                    denominator_fixed,
                );
                black_box(&out_full);
                Ok(())
            },
            raws,
        );
        let mut out_symmetric = vec![0.0_f32; features * features];
        let _ = measure_control(
            "p-symmetric-compute",
            "f32",
            features,
            "buffer_computation",
            steps,
            || {
                p_sym_into32(
                    &mut out_symmetric,
                    p_fixed.as_slice(),
                    v_fixed.as_slice(),
                    features,
                    common::LAMBDA_PE,
                    denominator_fixed,
                );
                black_box(&out_symmetric);
                Ok(())
            },
            raws,
        );
        let _ = measure_control(
            "alloc-control",
            "f32",
            features,
            "allocation_set",
            steps,
            || {
                black_box(Vec::<f32>::with_capacity(features * features));
                Ok(())
            },
            raws,
        );
        let mut symmetric_state = model.initial_state();
        let mut symmetric_index = 0_usize;
        let symmetric = measure_legacy(
            "rls-symmetric-p",
            "f32",
            features,
            "update",
            1,
            steps,
            true,
            |_| {
                let observation = &samples[symmetric_index % samples.len()];
                symmetric_index += 1;
                match update_sym32(&model, &symmetric_state, observation.value()) {
                    Ok(next) => {
                        symmetric_state = next;
                        Ok(1)
                    }
                    Err(error) => Err(LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&symmetric_state)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    }),
                }
            },
            raws,
        );
        if let (Some(full), Some(symmetric)) = (full, symmetric) {
            println!(
                "    D={features} full={:.1} ns/obs symmetric-p={:.1} ns/obs delta={:.1} -> {}",
                full.median,
                symmetric.median,
                (full.median - symmetric.median).abs(),
                common::materiality(full.median, full.iqr, symmetric.median, symmetric.iqr)
            );
        }
    }
}

fn run_failure_diagnostics() -> bool {
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

struct GroupedRun {
    attempted: u64,
    successful: u64,
    failed: u64,
    suffix_a_attempted: u64,
    group_b_calls: u64,
    retained: Option<RlsState>,
    error: Option<RlsError>,
}

fn run_grouped32(
    model: &Rls,
    initial: RlsState,
    group_a: &[Observation<RlsSample>],
    suffix_a_len: usize,
    group_b: &[Observation<RlsSample>],
) -> GroupedRun {
    let mut result = GroupedRun {
        attempted: 0,
        successful: 0,
        failed: 0,
        suffix_a_attempted: 0,
        group_b_calls: 0,
        retained: None,
        error: None,
    };
    let mut state = initial;
    for observation in group_a {
        result.attempted += 1;
        match model.update(&state, observation) {
            Ok(next) => {
                state = next;
                result.successful += 1;
            }
            Err(error) => {
                result.failed += 1;
                result.retained = Some(state.clone());
                result.error = Some(error);
                break;
            }
        }
    }
    let prefix_and_failure_len = group_a.len().saturating_sub(suffix_a_len) as u64;
    result.suffix_a_attempted = if result.failed == 0 {
        suffix_a_len as u64
    } else {
        result
            .attempted
            .saturating_sub(prefix_and_failure_len)
            .min(suffix_a_len as u64)
    };
    if result.failed == 0 {
        for observation in group_b {
            result.group_b_calls += 1;
            match model.update(&state, observation) {
                Ok(next) => state = next,
                Err(error) => {
                    result.failed += 1;
                    result.retained = Some(state.clone());
                    result.error = Some(error);
                    break;
                }
            }
        }
    }
    result
}

fn run_stop_diagnostics() -> bool {
    let Some(model) = make_model32(1, 1.0, 1.0) else {
        return false;
    };
    let prefix_a = sample32(&[1.0], 1.0);
    let prefix_b = sample32(&[0.5], 0.5);
    let invalid = sample32(&[f32::NAN], 0.0);
    let suffix_a = sample32(&[0.25], 0.25);
    let corrected = sample32(&[0.75], 0.75);
    let group_b = sample32(&[0.125], 0.125);
    let initial = model.initial_state();

    let mut prefix_state = initial.clone();
    for observation in [&prefix_a, &prefix_b] {
        match model.update(&prefix_state, observation) {
            Ok(next) => prefix_state = next,
            Err(error) => {
                println!("DIAGNOSTIC F-StrictStop setup FAIL {error:?}");
                set_unexpected_failure();
                return false;
            }
        }
    }
    let prefix_fingerprint = fingerprint32(&prefix_state);
    let group_a_direct = vec![
        prefix_a.clone(),
        prefix_b.clone(),
        invalid.clone(),
        suffix_a.clone(),
    ];
    let direct = run_grouped32(
        &model,
        initial.clone(),
        &group_a_direct,
        1,
        std::slice::from_ref(&group_b),
    );
    let direct_scheduled = 5_u64;
    let direct_unattempted = direct_scheduled - direct.attempted;
    let stop_ok = direct.attempted == 3
        && direct.successful == 2
        && direct.failed == 1
        && direct_unattempted == 2
        && direct.suffix_a_attempted == 0
        && direct.group_b_calls == 0
        && matches!(direct.error, Some(RlsError::NonFiniteInput))
        && direct
            .retained
            .as_ref()
            .is_some_and(|state| state_eq32(state, &prefix_state))
        && direct
            .retained
            .as_ref()
            .is_some_and(|state| fingerprint32(state) == prefix_fingerprint);
    let mut ok = diag_check(
        "F-StrictStop-direct",
        stop_ok,
        "expected failed observation, preserved prefix, unattempted suffix_a/group_b",
    );

    let retry = run_grouped32(
        &model,
        prefix_state.clone(),
        &[corrected.clone(), suffix_a.clone()],
        1,
        std::slice::from_ref(&group_b),
    );
    let retry_ok = retry.attempted == 2
        && retry.successful == 2
        && retry.failed == 0
        && retry.suffix_a_attempted == 1
        && retry.group_b_calls == 1
        && retry.attempted + retry.group_b_calls == 3
        && retry.retained.is_none();
    ok &= diag_check(
        "F-Retry",
        retry_ok,
        "expected caller-driven corrected retry from preserved prefix without replay",
    );

    let group_a = vec![
        prefix_a.clone(),
        prefix_b.clone(),
        invalid.clone(),
        suffix_a.clone(),
    ];
    let batch = process_batch(&model, initial.clone(), group_a.iter().cloned());
    let batch_group_b_calls = match &batch {
        Ok(state) => {
            let _ = model.update(state, &group_b);
            1
        }
        Err(_) => 0,
    };
    let batch_ok = match batch {
        Err(BatchFailure { state, cause, .. }) => {
            state_eq32(&state, &prefix_state)
                && matches!(cause, FailureCause::Model(RlsError::NonFiniteInput))
                && batch_group_b_calls == 0
        }
        Ok(_) => false,
    };
    ok &= diag_check(
        "F-BatchContrast",
        batch_ok,
        "expected BatchFailure with prefix state and no later-group call",
    );

    let bounded = model.process_batch_reference(initial.clone(), &group_a, group_a.len());
    let bounded_group_b_calls = match &bounded {
        Ok(state) => {
            let _ = model.update(state, &group_b);
            1
        }
        Err(_) => 0,
    };
    let bounded_ok = match bounded {
        Err(RlsBatchError::Transition {
            failed_index,
            state,
            error,
        }) => {
            failed_index == 2
                && state_eq32(&state, &prefix_state)
                && matches!(error, RlsError::NonFiniteInput)
                && bounded_group_b_calls == 0
        }
        Ok(_) | Err(_) => false,
    };
    ok &= diag_check(
        "F-BoundedContrast",
        bounded_ok,
        "expected Transition with failed_index 2 and prefix state",
    );

    let mut executor = StreamingExecutor::new(&model, initial.clone());
    let mut executor_attempts = 0_u64;
    for observation in [&prefix_a, &prefix_b] {
        executor_attempts += 1;
        if executor.process_one(observation).is_err() {
            println!("DIAGNOSTIC F-ExecutorContrast setup FAIL");
            set_unexpected_failure();
            return false;
        }
    }
    executor_attempts += 1;
    let executor_before = executor.state().clone();
    let executor_failure = executor.process_one(&invalid);
    let executor_failed = executor_failure.is_err();
    let executor_state_ok = state_eq32(&executor_before, executor.state());
    let executor_group_b_calls = if executor_failed {
        0
    } else {
        let _ = executor.process_one(&group_b);
        1
    };
    ok &= diag_check(
        "F-ExecutorContrast",
        executor_failed
            && executor_state_ok
            && executor_attempts == 3
            && executor_group_b_calls == 0,
        "expected executor Err, retained committed state, and no later-group call",
    );

    let prefix_before_symmetric = prefix_state.clone();
    let symmetric_failure = update_sym32(&model, &prefix_state, invalid.value());
    ok &= diag_check(
        "F-SymStop",
        matches!(symmetric_failure, Err(RlsError::NonFiniteInput))
            && state_eq32(&prefix_state, &prefix_before_symmetric),
        "expected corrected symmetric control Err and unchanged prefix",
    );

    let sharing_valid = sample32(&[1.0], 1.0);
    let sharing_invalid = sample32(&[f32::NAN], 0.0);
    let initial_fingerprint = fingerprint32(&initial);

    let mut stream_a = initial.clone();
    let stream_b = initial.clone();
    let mut sharing_raws = Vec::new();
    let mut sharing_failure = None;
    let mut sharing_record = None;
    let failure_before = has_unexpected_failure();
    let sharing_summary = measure_sharing(
        "F-Sharing-A",
        "f32",
        1,
        1,
        0,
        false,
        true,
        &mut sharing_failure,
        &mut sharing_record,
        |_| match process_one(&model, &stream_a, &sharing_invalid) {
            Ok(next) => {
                stream_a = next;
                Ok((1, 1))
            }
            Err(error) => Err(SharingFailure {
                stream: "A",
                a_attempted: 1,
                a_committed: 0,
                b_attempted: 0,
                b_committed: 0,
                retained_a: fingerprint32(&stream_a),
                retained_b: fingerprint32(&stream_b),
                message: format!("{error:?}"),
            }),
        },
        &mut sharing_raws,
    );
    let a_failure_ok = sharing_summary.is_none()
        && sharing_failure.as_ref().is_some_and(|failure| {
            failure.stream == "A"
                && failure.a_attempted == 1
                && failure.a_committed == 0
                && failure.b_attempted == 0
                && failure.b_committed == 0
                && failure.message.contains("NonFiniteInput")
                && failure.retained_a == initial_fingerprint
                && failure.retained_b == initial_fingerprint
        })
        && sharing_record.as_ref().is_some_and(|record| {
            record.status == "invalid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && record.failure_record_id.is_some()
                && record.scheduled_units == 2
                && record.attempted_units == 1
                && record.successful_units == 0
                && record.failed_units == 1
                && record.unattempted_units == 1
                && record.completed_updates == 0
                && record.stream_a_attempted == Some(1)
                && record.stream_a_committed == Some(0)
                && record.stream_b_attempted == Some(0)
                && record.stream_b_committed == Some(0)
        })
        && !failure_before
        && !has_unexpected_failure();
    ok &= diag_check(
        "F-Sharing-A",
        a_failure_ok,
        "expected stream A runner failure with invalid record and no summary",
    );

    let mut stream_a = initial.clone();
    let mut stream_b = initial.clone();
    let mut sharing_raws = Vec::new();
    let mut sharing_failure = None;
    let mut sharing_record = None;
    let failure_before = has_unexpected_failure();
    let sharing_summary = measure_sharing(
        "F-Sharing-B",
        "f32",
        1,
        1,
        0,
        false,
        true,
        &mut sharing_failure,
        &mut sharing_record,
        |_| match process_one(&model, &stream_a, &sharing_valid) {
            Ok(next) => {
                stream_a = next;
                match process_one(&model, &stream_b, &sharing_invalid) {
                    Ok(next_b) => {
                        stream_b = next_b;
                        Ok((1, 1))
                    }
                    Err(error) => Err(SharingFailure {
                        stream: "B",
                        a_attempted: 1,
                        a_committed: 1,
                        b_attempted: 1,
                        b_committed: 0,
                        retained_a: fingerprint32(&stream_a),
                        retained_b: fingerprint32(&stream_b),
                        message: format!("{error:?}"),
                    }),
                }
            }
            Err(error) => Err(SharingFailure {
                stream: "A",
                a_attempted: 1,
                a_committed: 0,
                b_attempted: 0,
                b_committed: 0,
                retained_a: fingerprint32(&stream_a),
                retained_b: fingerprint32(&stream_b),
                message: format!("{error:?}"),
            }),
        },
        &mut sharing_raws,
    );
    let b_failure_ok = sharing_summary.is_none()
        && sharing_failure.as_ref().is_some_and(|failure| {
            failure.stream == "B"
                && failure.a_attempted == 1
                && failure.a_committed == 1
                && failure.b_attempted == 1
                && failure.b_committed == 0
                && failure.message.contains("NonFiniteInput")
                && failure.retained_a != initial_fingerprint
                && failure.retained_b == initial_fingerprint
        })
        && sharing_record.as_ref().is_some_and(|record| {
            record.status == "invalid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && record.failure_record_id.is_some()
                && record.scheduled_units == 2
                && record.attempted_units == 2
                && record.successful_units == 1
                && record.failed_units == 1
                && record.unattempted_units == 0
                && record.completed_updates == 1
                && record.stream_a_attempted == Some(1)
                && record.stream_a_committed == Some(1)
                && record.stream_b_attempted == Some(1)
                && record.stream_b_committed == Some(0)
        })
        && !failure_before
        && !has_unexpected_failure();
    ok &= diag_check(
        "F-Sharing-B",
        b_failure_ok,
        "expected stream B runner failure with A committed, invalid record, no summary",
    );

    let foundation_before = prefix_state.clone();
    let foundation_failure = process_one(&model, &prefix_state, &invalid);
    ok &= diag_check(
        "F-FoundationStop",
        matches!(foundation_failure, Err(RlsError::NonFiniteInput))
            && state_eq32(&prefix_state, &foundation_before),
        "expected foundation process_one Err with unchanged retained state",
    );

    let mut continue_failures = 0_u64;
    let continue_summary = process_stream_continuing(
        &model,
        initial.clone(),
        vec![
            prefix_a.clone(),
            prefix_b.clone(),
            invalid.clone(),
            suffix_a.clone(),
            group_b.clone(),
        ],
        |position, error| {
            if position == 2 && matches!(*error, RlsError::NonFiniteInput) {
                continue_failures += 1;
            }
        },
    )
    .unwrap();
    let continue_state = continue_summary.state;
    let mut expected = prefix_state.clone();
    for observation in [&suffix_a, &group_b] {
        match model.update(&expected, observation) {
            Ok(next) => expected = next,
            Err(error) => {
                println!("DIAGNOSTIC F-ContinueContrast setup FAIL {error:?}");
                set_unexpected_failure();
                return false;
            }
        }
    }
    ok &= diag_check(
        "F-ContinueContrast",
        continue_failures == 1 && state_eq32(&continue_state, &expected),
        "expected process_stream to continue from the last valid state",
    );

    let comparison_observations = vec![prefix_a.clone(), prefix_b.clone()];
    let comparison = precheck32(&model, &comparison_observations, 1, 1, Candidate::Mismatch);
    let comparison_ok = matches!(
        comparison,
        Err(ref failure)
            if failure.kind == "comparison_failure"
                && failure.validation
                && failure.index == 0
                && failure.first_divergence.is_some()
    );
    ok &= diag_check(
        "F-ComparisonOnly",
        comparison_ok,
        "expected the real precheck comparison path to localize a boundary mismatch",
    );

    let validation_observations = observations32(1);
    let (validation_warm, validation_expected) = match precheck32(
        &model,
        &validation_observations,
        warmup(),
        horizon(),
        Candidate::Production,
    ) {
        Ok(value) => value,
        Err(failure) => {
            println!("DIAGNOSTIC F-PostconditionOnly setup FAIL {failure:?}");
            set_unexpected_failure();
            return false;
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
        &model,
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
    ok &= diag_check(
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
        &model,
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

    let Some(validation_model64) = make_model64(1, LAMBDA_F64, DELTA_F64) else {
        return false;
    };
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
            return false;
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
            return false;
        }
    };
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
    ok &= diag_check(
        "F-PostconditionOnly-f64",
        f64_mismatch_ok,
        "expected the real f64 runner to emit an invalid postcondition record",
    );
    ok &= diag_check(
        "F-PostconditionMatch-f64",
        f64_match_ok,
        "expected the same f64 runner path to pass a matching final state",
    );

    let mut altered_weights = prefix_state.w.as_slice().to_vec();
    if let Some(first) = altered_weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    let altered = RlsState {
        w: Vector::from_slice(&altered_weights),
        p: prefix_state.p.clone(),
    };
    let left_fingerprint = fingerprint32(&prefix_state);
    let right_fingerprint = fingerprint32(&altered);
    let difference = first_divergence32(&prefix_state, &altered)
        .unwrap_or_else(|| "states are bitwise equal".to_owned());
    // prefix_a and prefix_b both committed, so the compared trajectory's plan
    // and completed work are both 2 units.
    let comparison_failure = sweep_comparison_failure(
        "left-controlled",
        &prefix_state,
        "right-controlled",
        &altered,
        "diagnostic one-bit mismatch",
        2,
    );
    let message = comparison_failure.message();
    let evidence_ok = comparison_failure.left_path == "left-controlled"
        && comparison_failure.right_path == "right-controlled"
        && comparison_failure.left_fingerprint == left_fingerprint
        && comparison_failure.right_fingerprint == right_fingerprint
        && comparison_failure.scheduled == 2
        && comparison_failure.committed == 2
        && comparison_failure.first_difference == difference
        && message.contains(&format!("left-controlled fp=0x{left_fingerprint:016x}"))
        && message.contains(&format!("right-controlled fp=0x{right_fingerprint:016x}"))
        && message.contains("diagnostic one-bit mismatch")
        && message.contains(&difference)
        && difference != "states are bitwise equal";
    let failure_before = has_unexpected_failure();
    let record = sweep_failure_record("F-ComparisonEvidence", 1, comparison_failure.into());
    let expected_failure_id = failure_id("F-ComparisonEvidence", "precheck", None, None);
    let record_ok = record.kind == "comparison_failure"
        && record.phase == "precheck"
        && record.dimension == 1
        && record.retained_fingerprint.is_none()
        && record.committed_progress == 2
        && record.scheduled_units == 2
        && record.attempted_units == 2
        && record.successful_units == 2
        && record.failed_units == 0
        && record.unattempted_units == 0
        && record.validation_failure
        && !record.postcondition_failure
        && record.first_divergence.as_deref() == Some(difference.as_str())
        && record.attempted_units <= record.scheduled_units
        && record.successful_units + record.failed_units == record.attempted_units
        && record.unattempted_units == record.scheduled_units - record.attempted_units
        && record
            .error
            .as_deref()
            .is_some_and(|error| error.contains("left-controlled fp=0x"))
        && record
            .error
            .as_deref()
            .is_some_and(|error| error.contains("right-controlled fp=0x"))
        && record
            .error
            .as_deref()
            .is_some_and(|error| error.contains(&difference))
        && failure_id(
            &record.row,
            record.phase,
            record.window,
            record.observation_global,
        ) == expected_failure_id;
    record.emit_observed();
    ok &= diag_check(
        "F-ComparisonEvidence",
        evidence_ok && record_ok && !failure_before && !has_unexpected_failure(),
        "expected comparison_failure classification, both fingerprints, exact divergence, consistent counts and linkage",
    );

    ok
}

fn run_operation_count_diagnostics() -> bool {
    let Some(model) = make_model32(4, common::LAMBDA_PE, DELTA) else {
        return false;
    };
    // Fixed plan: 4 outer calls x 4 observations. Index 11 fails: groups 0 and
    // 1 commit (8), group 2 commits its first 3 observations, the 4th fails,
    // and 4 observations remain unattempted.
    let make_observation =
        |offset: usize| sample32(&[0.5 + offset as f32 * 0.01, 0.25, 0.75, 0.125], 0.5);
    let mut observations: Vec<_> = (0..16).map(make_observation).collect();
    observations[11] = sample32(&[f32::NAN, 0.0, 0.0, 0.0], 0.0);

    let mut expected_prefix = model.initial_state();
    for observation in &observations[..11] {
        match model.update(&expected_prefix, observation) {
            Ok(next) => expected_prefix = next,
            Err(error) => {
                println!("DIAGNOSTIC F-OperationCounts setup FAIL {error:?}");
                set_unexpected_failure();
                return false;
            }
        }
    }
    let expected_fingerprint = fingerprint32(&expected_prefix);

    let mut ok = true;
    for &batch in &[2_usize, 3, 4] {
        let label = format!("F-OperationCounts-batch{batch}");
        let evidence =
            match sweep_state_after_outer_calls(&model, &observations, 4, batch, 4, false) {
                Ok(_) => {
                    ok &= diag_check(
                        &label,
                        false,
                        "expected a bounded-reference precheck failure at observation 11",
                    );
                    continue;
                }
                Err(evidence) => evidence,
            };
        ok &= diag_check(
            &label,
            evidence.scheduled == 16
                && evidence.committed == 11
                && evidence.attempted == 12
                && evidence.retained_fingerprint == Some(expected_fingerprint)
                && evidence.message.contains("observation=11")
                && evidence.message.contains("NonFiniteInput"),
            "expected scheduled=16 committed=11 attempted=12 and the index-11 prefix fingerprint",
        );
    }

    for &batch in &[2_usize, 3] {
        let label = format!("F-OperationCounts-foundation-batch{batch}");
        let evidence = match sweep_state_after_outer_calls(&model, &observations, 4, batch, 4, true)
        {
            Ok(_) => {
                ok &= diag_check(
                    &label,
                    false,
                    "expected a foundation-fold precheck failure at observation 11",
                );
                continue;
            }
            Err(evidence) => evidence,
        };
        ok &= diag_check(
            &label,
            evidence.scheduled == 16
                && evidence.committed == 11
                && evidence.attempted == 12
                && evidence.retained_fingerprint == Some(expected_fingerprint)
                && evidence.message.contains("observation=11")
                && evidence.message.contains("NonFiniteInput"),
            "expected the foundation fold to report the same committed prefix",
        );
    }

    let horizon_evidence = match legacy_horizon(&model, &observations, 4, 4) {
        Ok(()) => {
            ok &= diag_check(
                "F-OperationCounts-horizon",
                false,
                "expected a horizon precheck failure at observation 11",
            );
            return ok;
        }
        Err(evidence) => evidence,
    };
    let record = sweep_failure_record("F-OperationCounts", 4, horizon_evidence.clone().into());
    let expected_failure_id = failure_id("F-OperationCounts", "precheck", None, None);
    ok &= diag_check(
        "F-OperationCounts-horizon",
        horizon_evidence.scheduled == 16
            && horizon_evidence.committed == 11
            && horizon_evidence.attempted == 12
            && horizon_evidence.retained_fingerprint == Some(expected_fingerprint)
            && horizon_evidence.message.contains("observation=11")
            && horizon_evidence.message.contains("NonFiniteInput")
            && record.kind == "operation_error"
            && record.committed_progress == 11
            && record.scheduled_units == 16
            && record.attempted_units == 12
            && record.successful_units == 11
            && record.failed_units == 1
            && record.unattempted_units == 4
            && record.retained_fingerprint == Some(expected_fingerprint)
            && record.attempted_units <= record.scheduled_units
            && record.successful_units + record.failed_units == record.attempted_units
            && record.unattempted_units == record.scheduled_units - record.attempted_units
            && failure_id(
                &record.row,
                record.phase,
                record.window,
                record.observation_global,
            ) == expected_failure_id,
        "expected scheduled=16 committed=11 attempted=12 failed=1 unattempted=4 and linkage",
    );
    record.emit_observed();
    ok
}

fn retention_fixture(
    row: &str,
    category: &'static str,
    unit: &'static str,
    status: &'static str,
    window: Option<usize>,
    peak: Option<u64>,
) -> RawWindow {
    let update = unit == "update";
    let units = u64::from(update);
    RawWindow {
        row_id: row.to_owned(),
        category,
        precision: "f32",
        dimension: 1,
        unit,
        window_index: window,
        observation_start: None,
        observation_end: None,
        scheduled_calls: units,
        attempted_calls: units,
        work_per_call: units,
        scheduled_units: units,
        attempted_units: units,
        successful_units: units,
        failed_units: 0,
        unattempted_units: 0,
        completed_updates: units,
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
    }
}

fn run_retention_diagnostics() -> bool {
    let mut records = vec![
        retention_fixture(
            "F-Retention-Measured",
            "S",
            "update",
            "valid",
            Some(0),
            None,
        ),
        retention_fixture("F-Retention-Warmup", "S", "update", "warmup", Some(0), None),
        retention_fixture(
            "F-Retention-Memory",
            "memory",
            "memory_probe",
            "valid",
            None,
            Some(64),
        ),
    ];
    let first_emission = emit_retained_records(&mut records);
    let second_emission = emit_retained_records(&mut records);
    let emitted_once =
        first_emission == 3 && second_emission == 0 && records.iter().all(|record| record.emitted);
    let output_ok = diag_check(
        "F-RetentionOutput",
        emitted_once,
        "expected the shared output path to serialize measured-schema, warm-up and memory records exactly once",
    );

    let identity = |row: &str| -> RowIdentity { (row.to_owned(), "f32", 1, 0) };
    let expected: BTreeSet<RowIdentity> = ["F-Retention-A", "F-Retention-B", "F-Retention-C"]
        .into_iter()
        .map(identity)
        .collect();
    let report = |rows: &[&str]| {
        let raws: Vec<RawWindow> = rows
            .iter()
            .map(|row| retention_fixture(row, "S", "update", "valid", Some(0), None))
            .collect();
        catalog_report(&expected, &raws)
    };

    let (missing, extra, duplicates) = report(&["F-Retention-A", "F-Retention-B"]);
    let missing_ok =
        missing == vec![identity("F-Retention-C")] && extra.is_empty() && duplicates.is_empty();
    let (missing, extra, duplicates) = report(&[
        "F-Retention-A",
        "F-Retention-B",
        "F-Retention-C",
        "F-Retention-D",
    ]);
    let extra_ok =
        missing.is_empty() && extra == vec![identity("F-Retention-D")] && duplicates.is_empty();
    let (missing, extra, duplicates) = report(&[
        "F-Retention-A",
        "F-Retention-A",
        "F-Retention-B",
        "F-Retention-C",
    ]);
    let duplicate_ok = missing.is_empty()
        && extra.is_empty()
        && duplicates == vec![(identity("F-Retention-A"), 2)];
    let (missing, extra, duplicates) = report(&["F-Retention-A", "F-Retention-B", "F-Retention-C"]);
    let exact_ok = missing.is_empty() && extra.is_empty() && duplicates.is_empty();
    let catalog_ok = diag_check(
        "F-RetentionCatalog",
        missing_ok && extra_ok && duplicate_ok && exact_ok,
        "expected missing, unexpected, duplicate and exact identity sets to be classified correctly",
    );

    output_ok && catalog_ok
}

fn legacy_rejection(label: &str, failure: &LegacyFailure, fatal: bool) {
    println!(
        "LEGACY_VALIDATION {label} REJECTED attempted={} committed={} error={} fatal={fatal}",
        failure.attempted, failure.committed, failure.message
    );
    if fatal {
        set_unexpected_failure();
    }
}

fn legacy_rejection_operation(label: &str, failure: &OperationEvidence, fatal: bool) {
    println!(
        "LEGACY_VALIDATION {label} REJECTED scheduled={} attempted={} committed={} error={} fatal={fatal}",
        failure.scheduled, failure.attempted, failure.committed, failure.message
    );
    if fatal {
        set_unexpected_failure();
    }
}

fn legacy_check(label: &str, condition: bool, detail: &str) {
    println!(
        "LEGACY_VALIDATION {label} {} {detail}",
        if condition { "PASS" } else { "FAIL" }
    );
    if !condition {
        set_unexpected_failure();
    }
}

fn validate_legacy_paths() {
    println!("LEGACY_VALIDATION no_timing=true");
    LEGACY_VALIDATE_ONLY.store(true, Ordering::Relaxed);
    let mut wrapper_raws = Vec::new();
    let mut ignored_failure = None;
    let mut ignored_raw = None;
    run_legacy_fixed_control(
        &mut wrapper_raws,
        None,
        false,
        &mut ignored_failure,
        &mut ignored_raw,
    );
    run_legacy_batch_sweep(&mut wrapper_raws, None);
    run_legacy_workloads(&mut wrapper_raws);
    run_legacy_evidence(&mut wrapper_raws);

    // Diagnostic-only: inject a controlled one-bit comparison mismatch into the
    // ordinary batch-sweep precheck path and prove that measurement and the
    // dependent comparisons are never invoked once the precheck fails.
    let injection = SweepInjection {
        feature: SWEEP_DIMENSIONS[0],
        expected: true,
        ..SweepInjection::default()
    };
    let mut injected_raws = Vec::new();
    let injection_failure_before = has_unexpected_failure();
    run_legacy_batch_sweep(&mut injected_raws, Some(&injection));
    let injected_record = injection.failure_record.borrow().clone();
    let injection_ok = !injection_failure_before
        && injection.stopped_on_precheck.get()
        && injection.comparison_invocations.get() == 1
        && injection.measurement_invocations.get() == 0
        && injected_raws.is_empty()
        && injected_record.as_ref().is_some_and(|record| {
            record.kind == "comparison_failure"
                && record.validation_failure
                && !record.postcondition_failure
                && record.failed_units == 0
                && record.attempted_units == record.scheduled_units
                && record.successful_units == record.attempted_units
                && record.unattempted_units == 0
                && record
                    .first_divergence
                    .as_deref()
                    .is_some_and(|difference| difference.starts_with("w[0] left=0x"))
                && record.error.as_deref().is_some_and(|error| {
                    error.contains("grouped fp=0x")
                        && error.contains("bounded-reference fp=0x")
                        && error.contains("diagnostic one-bit mismatch injection")
                })
        })
        && !has_unexpected_failure();
    legacy_check(
        "batch-sweep-precheck-suppression",
        injection_ok,
        "expected one injected comparison failure, zero measurement/dependent comparisons and no measured raw window",
    );

    // Diagnostic-only: inject one invalid observation into the ordinary
    // batch-sweep precheck and check operation-error counts plus suppression.
    let operation_injection = SweepInjection {
        feature: SWEEP_DIMENSIONS[0],
        expected: true,
        fail_observation: Some(11),
        ..SweepInjection::default()
    };
    let driver_model = legacy_model(SWEEP_DIMENSIONS[0], common::LAMBDA_PE);
    let driver_observations = sweep_fixture(SWEEP_DIMENSIONS[0]);
    let mut driver_prefix = driver_model.initial_state();
    let mut driver_prefix_ok = true;
    for observation in driver_observations.iter().take(11) {
        match driver_model.update(&driver_prefix, observation) {
            Ok(next) => driver_prefix = next,
            Err(_) => {
                driver_prefix_ok = false;
                break;
            }
        }
    }
    let driver_fingerprint = fingerprint32(&driver_prefix);
    let expected_plan = if cfg!(debug_assertions) { 16 } else { 32 };
    let mut operation_raws = Vec::new();
    let operation_failure_before = has_unexpected_failure();
    run_legacy_batch_sweep(&mut operation_raws, Some(&operation_injection));
    let operation_record = operation_injection.failure_record.borrow().clone();
    let operation_ok = !operation_failure_before
        && driver_prefix_ok
        && operation_injection.stopped_on_precheck.get()
        && operation_injection.measurement_invocations.get() == 0
        && operation_injection.comparison_invocations.get() == 0
        && operation_raws.is_empty()
        && operation_record.as_ref().is_some_and(|record| {
            record.kind == "operation_error"
                && record.committed_progress == 11
                && record.scheduled_units == expected_plan
                && record.attempted_units == 12
                && record.successful_units == 11
                && record.failed_units == 1
                && record.unattempted_units == expected_plan - 12
                && record.retained_fingerprint == Some(driver_fingerprint)
                && record.error.as_deref().is_some_and(|error| {
                    error.contains("NonFiniteInput") && error.contains("observation=11")
                })
        })
        && !has_unexpected_failure();
    legacy_check(
        "batch-sweep-operation-suppression",
        operation_ok,
        "expected operation_error committed=11 attempted=12 failed=1, zero measurement/dependent comparisons and no raw window",
    );

    let wrapper_prefix_model = legacy_model(8, LAMBDA_CONTROL);
    let wrapper_prefix_state = match wrapper_prefix_model.update(
        &wrapper_prefix_model.initial_state(),
        &control_observation(8),
    ) {
        Ok(state) => state,
        Err(error) => {
            legacy_rejection(
                "wrapper-prefix",
                &LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(
                        &wrapper_prefix_model.initial_state(),
                    )),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                },
                true,
            );
            return;
        }
    };
    let wrapper_prefix_fingerprint = fingerprint32(&wrapper_prefix_state);
    let mut wrapper_failure = None;
    let mut wrapper_failure_raw = None;
    let mut wrapper_failure_raws = Vec::new();
    run_legacy_fixed_control(
        &mut wrapper_failure_raws,
        Some(1),
        true,
        &mut wrapper_failure,
        &mut wrapper_failure_raw,
    );
    let wrapper_failure_id = failure_id("control-reference", "measured_window", Some(0), None);
    let wrapper_failure_ok = wrapper_failure.as_ref().is_some_and(|failure| {
        failure.kind == "operation_error"
            && failure.row == "control-reference"
            && failure
                .error
                .as_deref()
                .is_some_and(|error| error.contains("NonFiniteInput"))
            && failure.window == Some(0)
            && failure.observation_window == Some(1)
            && failure.committed_progress == 1
            && failure.scheduled_units == BATCH as u64
            && failure.attempted_units == 2
            && failure.successful_units == 1
            && failure.failed_units == 1
            && failure.unattempted_units == (BATCH as u64 - 2)
            && failure.retained_fingerprint == Some(wrapper_prefix_fingerprint)
            && failure_id(
                &failure.row,
                failure.phase,
                failure.window,
                failure.observation_global,
            ) == wrapper_failure_id
    }) && wrapper_failure_raw.as_ref().is_some_and(|record| {
        record.row_id == "control-reference"
            && record.status == "invalid"
            && !record.timed
            && record.elapsed_ns.is_none()
            && record.allocation_requests.is_none()
            && record.requested_bytes.is_none()
            && record.failure_record_id.as_deref() == Some(wrapper_failure_id.as_str())
            && record.scheduled_units == BATCH as u64
            && record.attempted_units == 2
            && record.successful_units == 1
            && record.failed_units == 1
            && record.unattempted_units == (BATCH as u64 - 2)
            && record.completed_updates == 1
    }) && wrapper_failure_raws
        .iter()
        .filter(|record| record.status != "warmup" && record.window_index.is_some())
        .count()
        == 1
        && wrapper_failure_raws
            .iter()
            .all(|record| record.row_id == "control-reference");
    legacy_check(
        "wrapper-failure",
        wrapper_failure_ok && !has_unexpected_failure(),
        "expected actual fixed-control wrapper prefix failure with no suffix/later rows",
    );
    LEGACY_VALIDATE_ONLY.store(false, Ordering::Relaxed);
    let model = legacy_model(8, LAMBDA_CONTROL);
    let observation = control_observation(8);
    let mut reference_state = model.initial_state();
    for _ in 0..2 {
        match model.update(&reference_state, &observation) {
            Ok(next) => reference_state = next,
            Err(error) => {
                legacy_rejection(
                    "fixed-reference",
                    &LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&reference_state)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    },
                    true,
                );
                break;
            }
        }
    }
    let mut executor = StreamingExecutor::new(&model, model.initial_state());
    for _ in 0..2 {
        if let Err(error) = executor.process_one(&observation) {
            legacy_rejection(
                "fixed-streaming",
                &LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(executor.state())),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                },
                true,
            );
            break;
        }
    }
    let mut fold_state = model.initial_state();
    match process_batch(&model, fold_state.clone(), vec![observation.clone(); BATCH]) {
        Ok(next) => fold_state = next,
        Err(error) => legacy_rejection(
            "fixed-bounded-fold",
            &LegacyFailure {
                retained_fingerprint: Some(fingerprint32(&fold_state)),
                attempted: BATCH as u64,
                committed: 0,
                message: format!("{error:?}"),
            },
            true,
        ),
    }
    println!(
        "LEGACY_VALIDATION fixed-control committed_progress={}",
        fold_state.w.len()
    );

    let mut runner_state = model.initial_state();
    let mut runner_raws = Vec::new();
    let mut runner_failure = None;
    let mut runner_record = None;
    let runner_summary = measure_legacy_impl(
        "legacy-runner-success",
        "f32",
        8,
        "update",
        1,
        1,
        true,
        0,
        false,
        false,
        &mut runner_failure,
        &mut runner_record,
        |_| match model.update(&runner_state, &observation) {
            Ok(next) => {
                runner_state = next;
                Ok(1)
            }
            Err(error) => Err(LegacyFailure {
                retained_fingerprint: Some(fingerprint32(&runner_state)),
                attempted: 1,
                committed: 0,
                message: format!("{error:?}"),
            }),
        },
        &mut runner_raws,
    );
    legacy_check(
        "legacy-runner-success",
        runner_summary.is_none()
            && runner_failure.is_none()
            && runner_record.is_none()
            && runner_raws.len() == WINDOWS
            && runner_raws.iter().all(|record| {
                record.status == "valid"
                    && !record.timed
                    && record.elapsed_ns.is_none()
                    && record.allocation_requests.is_none()
                    && record.requested_bytes.is_none()
                    && record.completed_updates == 1
            }),
        "expected 20 untimed valid windows and no summary",
    );

    let invalid_legacy_features = [f32::NAN; 8];
    let invalid_legacy = sample32(&invalid_legacy_features, 0.0);
    let mut runner_state = model.initial_state();
    let mut runner_raws = Vec::new();
    let mut runner_failure = None;
    let mut runner_record = None;
    let failure_before = has_unexpected_failure();
    let runner_summary = measure_legacy_impl(
        "legacy-runner-failure",
        "f32",
        8,
        "update",
        1,
        3,
        true,
        0,
        false,
        true,
        &mut runner_failure,
        &mut runner_record,
        |call| match call {
            0 => match model.update(&runner_state, &observation) {
                Ok(next) => {
                    runner_state = next;
                    Ok(1)
                }
                Err(error) => Err(LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(&runner_state)),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                }),
            },
            1 => match model.update(&runner_state, &invalid_legacy) {
                Ok(next) => {
                    runner_state = next;
                    Ok(1)
                }
                Err(error) => Err(LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(&runner_state)),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                }),
            },
            _ => match model.update(&runner_state, &observation) {
                Ok(next) => {
                    runner_state = next;
                    Ok(1)
                }
                Err(error) => Err(LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(&runner_state)),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                }),
            },
        },
        &mut runner_raws,
    );
    let legacy_failure_id = failure_id("legacy-runner-failure", "measured_window", Some(0), None);
    let prefix_fingerprint = fingerprint32(&runner_state);
    legacy_check(
        "legacy-runner-failure",
        runner_summary.is_none()
            && runner_failure.as_ref().is_some_and(|failure| {
                failure.kind == "operation_error"
                    && failure_id(
                        &failure.row,
                        failure.phase,
                        failure.window,
                        failure.observation_global,
                    ) == legacy_failure_id
                    && failure
                        .error
                        .as_deref()
                        .is_some_and(|error| error.contains("NonFiniteInput"))
                    && failure.scheduled_units == 3
                    && failure.attempted_units == 2
                    && failure.successful_units == 1
                    && failure.failed_units == 1
                    && failure.unattempted_units == 1
                    && failure.committed_progress == 1
                    && failure.retained_fingerprint == Some(prefix_fingerprint)
            })
            && runner_record.as_ref().is_some_and(|record| {
                record.status == "invalid"
                    && !record.timed
                    && record.elapsed_ns.is_none()
                    && record.allocation_requests.is_none()
                    && record.requested_bytes.is_none()
                    && record.failure_record_id.as_deref() == Some(legacy_failure_id.as_str())
                    && record.scheduled_units == 3
                    && record.attempted_units == 2
                    && record.successful_units == 1
                    && record.failed_units == 1
                    && record.unattempted_units == 1
                    && record.completed_updates == 1
            })
            && !failure_before
            && !has_unexpected_failure(),
        "expected prefix commit, immediate failure, retained state, invalid record and no summary",
    );

    for features in SWEEP_DIMENSIONS {
        let sweep_model = legacy_model(features, common::LAMBDA_PE);
        let observations = sweep_fixture(features);
        if let Err(failure) = sweep_equivalence(
            &sweep_model,
            &observations,
            sweep_outer(),
            sweep_batch_sizes(),
            if cfg!(debug_assertions) { 2 } else { 4 },
            None,
        ) {
            match &failure {
                SweepFailure::Operation(operation) => {
                    legacy_rejection_operation("batch-equivalence", operation, false)
                }
                SweepFailure::Comparison(evidence) => println!(
                    "LEGACY_VALIDATION batch-equivalence REJECTED comparison {} fp=0x{:016x} vs {} fp=0x{:016x}",
                    evidence.context,
                    evidence.left_fingerprint,
                    evidence.right_path,
                    evidence.right_fingerprint
                ),
            }
        }
        if let Err(failure) = legacy_horizon(
            &sweep_model,
            &observations,
            sweep_outer(),
            legacy_horizon_plan(sweep_steps(features)),
        ) {
            legacy_rejection_operation("batch-horizon", &failure, false);
        }
    }

    let workload_model = legacy_model(8, common::LAMBDA_PE);
    let xs = common::persistent_excitation_features(8, 4);
    let weights = common::fixed_weights(8);
    let ys = common::linear_targets(&xs, &weights);
    let observations = workload_observations(&xs, &ys);
    let mut reference_state = workload_model.initial_state();
    for observation in observations.iter().take(2) {
        match workload_model.update(&reference_state, observation) {
            Ok(next) => reference_state = next,
            Err(error) => {
                legacy_rejection(
                    "workload-reference",
                    &LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&reference_state)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    },
                    true,
                );
                break;
            }
        }
    }
    let mut executor = StreamingExecutor::new(&workload_model, workload_model.initial_state());
    for observation in observations.iter().take(2) {
        if let Err(error) = executor.process_one(observation) {
            legacy_rejection(
                "workload-streaming",
                &LegacyFailure {
                    retained_fingerprint: Some(fingerprint32(executor.state())),
                    attempted: 1,
                    committed: 0,
                    message: format!("{error:?}"),
                },
                true,
            );
            break;
        }
    }
    let mut shared_a = workload_model.initial_state();
    let mut shared_b = workload_model.initial_state();
    for observation in observations.iter().take(2) {
        match process_one(&workload_model, &shared_a, observation) {
            Ok(next_a) => {
                shared_a = next_a;
                match process_one(&workload_model, &shared_b, observation) {
                    Ok(next_b) => shared_b = next_b,
                    Err(error) => {
                        legacy_rejection(
                            "sharing-B",
                            &LegacyFailure {
                                retained_fingerprint: Some(fingerprint32(&shared_b)),
                                attempted: 1,
                                committed: 0,
                                message: format!("{error:?}"),
                            },
                            true,
                        );
                        break;
                    }
                }
            }
            Err(error) => {
                legacy_rejection(
                    "sharing-A",
                    &LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&shared_a)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    },
                    true,
                );
                break;
            }
        }
    }
    println!(
        "LEGACY_VALIDATION workload-sharing a_fp=0x{:016x} b_fp=0x{:016x}",
        fingerprint32(&shared_a),
        fingerprint32(&shared_b)
    );

    for dimension in [128_usize, 256] {
        let high_model = legacy_model(dimension, common::LAMBDA_PE);
        let high_xs = common::persistent_excitation_features(dimension, 4);
        let high_weights = common::fixed_weights(dimension);
        let high_ys = common::linear_targets(&high_xs, &high_weights);
        let high_observations = workload_observations(&high_xs, &high_ys);
        let mut high_state = high_model.initial_state();
        let mut rejected = false;
        for observation in high_observations.iter().take(2) {
            match high_model.update(&high_state, observation) {
                Ok(next) => high_state = next,
                Err(error) => {
                    legacy_rejection(
                        &format!("high-D-{dimension}"),
                        &LegacyFailure {
                            retained_fingerprint: Some(fingerprint32(&high_state)),
                            attempted: 1,
                            committed: 0,
                            message: format!("{error:?}"),
                        },
                        false,
                    );
                    rejected = true;
                    break;
                }
            }
        }
        if !rejected {
            println!("LEGACY_VALIDATION high-D-{dimension} accepted");
        }
    }

    let evidence_model = legacy_model(32, common::LAMBDA_PE);
    let evidence_xs = common::persistent_excitation_features(32, 4);
    let evidence_weights = common::fixed_weights(32);
    let evidence_ys = common::linear_targets(&evidence_xs, &evidence_weights);
    let evidence_observations = workload_observations(&evidence_xs, &evidence_ys);
    let mut evidence_state = evidence_model.initial_state();
    for observation in evidence_observations.iter().take(2) {
        match evidence_model.update(&evidence_state, observation) {
            Ok(next) => evidence_state = next,
            Err(error) => {
                legacy_rejection(
                    "evidence-warmup",
                    &LegacyFailure {
                        retained_fingerprint: Some(fingerprint32(&evidence_state)),
                        attempted: 1,
                        committed: 0,
                        message: format!("{error:?}"),
                    },
                    true,
                );
                break;
            }
        }
    }
    match update_sym32(
        &evidence_model,
        &evidence_state,
        evidence_observations[0].value(),
    ) {
        Ok(_) => println!("LEGACY_VALIDATION evidence-symmetric PASS"),
        Err(error) => legacy_rejection(
            "evidence-symmetric",
            &LegacyFailure {
                retained_fingerprint: Some(fingerprint32(&evidence_state)),
                attempted: 1,
                committed: 0,
                message: format!("{error:?}"),
            },
            true,
        ),
    }
}

fn run_validation() {
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

fn candidate_row(precision: &'static str, candidate: Candidate) -> &'static str {
    match (precision, candidate) {
        ("f32", Candidate::Production) => "S-C",
        ("f64", Candidate::Production) => "S-A",
        (_, Candidate::Fused) => "U-Fused",
        (_, Candidate::Symmetric) => "U-Sym",
        (_, Candidate::Mismatch) => "F-ComparisonOnly",
        (_, Candidate::Production) => "S-C",
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_candidate32(
    row: &str,
    model: &Rls,
    warm_state: &RlsState,
    expected: &RlsState,
    observations: &[Observation<RlsSample>],
    candidate: Candidate,
    warm: usize,
    horizon: usize,
    raws: &mut Vec<RawWindow>,
) {
    match precheck32(model, observations, warm, horizon, candidate) {
        Ok(_) => {
            let mut ignored_failure = None;
            let mut ignored_raw = None;
            let mut ignored_status = UpdateRunStatus::default();
            let _ = measure_update32(
                row,
                model,
                warm_state,
                expected,
                observations,
                candidate,
                raws,
                WINDOWS,
                true,
                true,
                &mut ignored_failure,
                &mut ignored_raw,
                &mut ignored_status,
            );
        }
        Err(failure) => {
            record_precheck_failure(row, "f32", model.dimension(), warm, horizon, failure)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_candidate64(
    row: &str,
    model: &Rls64,
    warm_state: &RlsState64,
    expected: &RlsState64,
    observations: &[Observation<RlsSample64>],
    candidate: Candidate,
    warm: usize,
    horizon: usize,
    raws: &mut Vec<RawWindow>,
) {
    match precheck64(model, observations, warm, horizon, candidate) {
        Ok(_) => {
            let mut ignored_failure = None;
            let mut ignored_raw = None;
            let mut ignored_status = UpdateRunStatus::default();
            let _ = measure_update64(
                row,
                model,
                warm_state,
                expected,
                observations,
                candidate,
                raws,
                WINDOWS,
                true,
                true,
                &mut ignored_failure,
                &mut ignored_raw,
                &mut ignored_status,
            );
        }
        Err(failure) => {
            record_precheck_failure(row, "f64", model.dimension(), warm, horizon, failure)
        }
    }
}

fn run_measurement() {
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
    let u_order: Vec<Candidate> = order
        .into_iter()
        .filter(|candidate| *candidate != Candidate::Production)
        .collect();
    println!("capture={} candidate_order={u_order:?}", capture.number());
    println!(
        "UNITS allocation-requests/<unit>=wrapper alloc+realloc requests; requested-bytes/<unit>=requested payload bytes; MEMORY retained-requested-bytes and peak-live-requested-bytes-above-baseline are live requested payload, not allocator overhead"
    );
    let mut raws = Vec::new();
    for d in DIMS {
        let Some(model) = make_model32(d, LAMBDA_F32, DELTA_F32) else {
            continue;
        };
        let observations = observations32(d);
        let production = match precheck32(&model, &observations, warm, h, Candidate::Production) {
            Ok(value) => value,
            Err(failure) => {
                record_precheck_failure("S-C", "f32", d, warm, h, failure);
                continue;
            }
        };
        let (warm_state, expected) = production;
        let mut ignored_failure = None;
        let mut ignored_raw = None;
        let mut ignored_status = UpdateRunStatus::default();
        let _ = measure_update32(
            "S-C",
            &model,
            &warm_state,
            &expected,
            &observations,
            Candidate::Production,
            &mut raws,
            WINDOWS,
            true,
            true,
            &mut ignored_failure,
            &mut ignored_raw,
            &mut ignored_status,
        );
        for candidate in &u_order {
            let row = candidate_row("f32", *candidate);
            measure_candidate32(
                row,
                &model,
                &warm_state,
                &expected,
                &observations,
                *candidate,
                warm,
                h,
                &mut raws,
            );
        }
        let last = &observations[warm + h - 1];
        run_components32(&model, &expected, last, &mut raws);
        run_memory_probes32(&model, &warm_state, last, &mut raws);

        let Some(model64) = make_model64(d, LAMBDA_F64, DELTA_F64) else {
            continue;
        };
        let observations64 = observations64(d);
        let production64 =
            match precheck64(&model64, &observations64, warm, h, Candidate::Production) {
                Ok(value) => value,
                Err(failure) => {
                    record_precheck_failure("S-A", "f64", d, warm, h, failure);
                    continue;
                }
            };
        let (warm_state64, expected64) = production64;
        let mut ignored_failure = None;
        let mut ignored_raw = None;
        let mut ignored_status = UpdateRunStatus::default();
        let _ = measure_update64(
            "S-A",
            &model64,
            &warm_state64,
            &expected64,
            &observations64,
            Candidate::Production,
            &mut raws,
            WINDOWS,
            true,
            true,
            &mut ignored_failure,
            &mut ignored_raw,
            &mut ignored_status,
        );
        for candidate in &u_order {
            let row = candidate_row("f64", *candidate);
            measure_candidate64(
                row,
                &model64,
                &warm_state64,
                &expected64,
                &observations64,
                *candidate,
                warm,
                h,
                &mut raws,
            );
        }
        let last64 = &observations64[warm + h - 1];
        run_components64(&model64, &expected64, last64, &mut raws);
        run_memory_probes64(&model64, &warm_state64, last64, &mut raws);
    }
    check_catalog(&raws);
    let emitted = emit_retained_records(&mut raws);
    println!(
        "RETAINED records_emitted={emitted} records_total={}",
        raws.len()
    );
}

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
