//! Shared legacy-validation output and workload-plan helpers.

use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::RlsSample;

use crate::common;
use crate::evidence::OperationEvidence;
use crate::fixtures::workload_observations;
use crate::measurement::LegacyFailure;
use crate::mode::{legacy_validating, set_unexpected_failure};
use crate::{BATCH, SWEEP_BATCH_SIZES, SWEEP_FIXTURE_LEN, SWEEP_OUTER_RELEASE};

pub(super) fn control_steps(features: usize) -> u32 {
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

pub(super) fn workload_steps(features: usize) -> u32 {
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

pub(super) fn evidence_steps(features: usize) -> u32 {
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

pub(super) fn sweep_outer() -> usize {
    if legacy_validating() {
        return 8;
    }
    if cfg!(debug_assertions) {
        8
    } else {
        SWEEP_OUTER_RELEASE
    }
}

pub(super) fn sweep_batch_sizes() -> &'static [usize] {
    if legacy_validating() {
        return &SWEEP_BATCH_SIZES[..2];
    }
    if cfg!(debug_assertions) {
        &SWEEP_BATCH_SIZES[..2]
    } else {
        &SWEEP_BATCH_SIZES
    }
}

pub(super) fn sweep_steps(features: usize) -> u32 {
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

pub(super) fn sweep_fixture(features: usize) -> Vec<Observation<RlsSample>> {
    let xs = common::persistent_excitation_features(features, SWEEP_FIXTURE_LEN);
    let weights = common::fixed_weights(features);
    let ys = common::linear_targets(&xs, &weights);
    workload_observations(&xs, &ys)
}

pub(super) fn legacy_horizon_plan(steps: u32) -> usize {
    let warm_up_calls = (steps as usize / 10).max(1000);
    warm_up_calls + common::RUNS * steps as usize
}

pub(super) fn legacy_rejection(label: &str, failure: &LegacyFailure, fatal: bool) {
    println!(
        "LEGACY_VALIDATION {label} REJECTED attempted={} committed={} error={} fatal={fatal}",
        failure.attempted, failure.committed, failure.message
    );
    if fatal {
        set_unexpected_failure();
    }
}

pub(super) fn legacy_rejection_operation(label: &str, failure: &OperationEvidence, fatal: bool) {
    println!(
        "LEGACY_VALIDATION {label} REJECTED scheduled={} attempted={} committed={} error={} fatal={fatal}",
        failure.scheduled, failure.attempted, failure.committed, failure.message
    );
    if fatal {
        set_unexpected_failure();
    }
}

pub(super) fn legacy_check(label: &str, condition: bool, detail: &str) {
    println!(
        "LEGACY_VALIDATION {label} {} {detail}",
        if condition { "PASS" } else { "FAIL" }
    );
    if !condition {
        set_unexpected_failure();
    }
}
