//! Evidence-only memory probes and memory records.

use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64, RlsState, RlsState64};

use crate::common;
use crate::evidence::{FailureRecord, RawWindow, emit_memory_record};
use crate::fixtures::{fingerprint32, fingerprint64};
use crate::mode::{horizon, warmup};

pub(super) fn run_memory_probes32(
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

pub(super) fn run_memory_probes64(
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
