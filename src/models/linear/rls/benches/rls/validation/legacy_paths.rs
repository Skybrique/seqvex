//! Legacy-path coordinator and ordered control groups.
//!
//! The coordinator preserves the original `LEGACY_VALIDATE_ONLY` transitions:
//! the flag is set before the wrapper/injection group, cleared only after that
//! group completes without its wrapper-prefix early return, and left set when
//! the original early return fires.

use std::sync::atomic::Ordering;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{StateModel, process_batch, process_one};
use seqvex::models::linear::rls::{Rls, RlsSample};

use crate::candidates::update_sym32;
use crate::common;
use crate::evidence::failure_id;
use crate::fixtures::{control_observation, fingerprint32, legacy_model, workload_observations};
use crate::measurement::{LegacyFailure, measure_legacy_impl};
use crate::mode::LEGACY_VALIDATE_ONLY;
use crate::validation::legacy_controls::{legacy_horizon, run_legacy_fixed_control};
use crate::validation::legacy_support::{
    legacy_check, legacy_rejection, legacy_rejection_operation, sweep_batch_sizes, sweep_fixture,
    sweep_outer, sweep_steps,
};
use crate::validation::legacy_sweep::{SweepInjection, run_legacy_batch_sweep, sweep_equivalence};
use crate::validation::legacy_workloads::{run_legacy_evidence, run_legacy_workloads};
use crate::{BATCH, LAMBDA_CONTROL, SWEEP_DIMENSIONS};

pub(crate) fn validate_legacy_paths() {
    println!("LEGACY_VALIDATION no_timing=true");
    LEGACY_VALIDATE_ONLY.store(true, Ordering::Relaxed);
    if !run_wrapper_and_injection_controls() {
        return;
    }
    LEGACY_VALIDATE_ONLY.store(false, Ordering::Relaxed);
    let control_model = legacy_model(8, LAMBDA_CONTROL);
    let control_observation = control_observation(8);
    run_fixed_reference_controls(&control_model, &control_observation);
    run_runner_controls(&control_model, &control_observation);
    run_sweep_equivalence_controls();
    run_workload_sharing_controls();
    run_high_dimension_controls();
    run_evidence_controls();
}

fn run_wrapper_and_injection_controls() -> bool {
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
    let injection_failure_before = crate::mode::has_unexpected_failure();
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
        && !crate::mode::has_unexpected_failure();
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
    let operation_failure_before = crate::mode::has_unexpected_failure();
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
        && !crate::mode::has_unexpected_failure();
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
            return false;
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
        wrapper_failure_ok && !crate::mode::has_unexpected_failure(),
        "expected actual fixed-control wrapper prefix failure with no suffix/later rows",
    );
    true
}

fn run_fixed_reference_controls(model: &Rls, observation: &Observation<RlsSample>) {
    let mut reference_state = model.initial_state();
    for _ in 0..2 {
        match model.update(&reference_state, observation) {
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
    let mut executor = StreamingExecutor::new(model, model.initial_state());
    for _ in 0..2 {
        if let Err(error) = executor.process_one(observation) {
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
    match process_batch(model, fold_state.clone(), vec![observation.clone(); BATCH]) {
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
}

fn run_runner_controls(model: &Rls, observation: &Observation<RlsSample>) {
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
        |_| match model.update(&runner_state, observation) {
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
            && runner_raws.len() == crate::WINDOWS
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
    let invalid_legacy = crate::fixtures::sample32(&invalid_legacy_features, 0.0);
    let mut runner_state = model.initial_state();
    let mut runner_raws = Vec::new();
    let mut runner_failure = None;
    let mut runner_record = None;
    let failure_before = crate::mode::has_unexpected_failure();
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
            0 => match model.update(&runner_state, observation) {
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
            _ => match model.update(&runner_state, observation) {
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
            && !crate::mode::has_unexpected_failure(),
        "expected prefix commit, immediate failure, retained state, invalid record and no summary",
    );
}

fn run_sweep_equivalence_controls() {
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
                crate::evidence::SweepFailure::Operation(operation) => {
                    legacy_rejection_operation("batch-equivalence", operation, false)
                }
                crate::evidence::SweepFailure::Comparison(evidence) => println!(
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
            crate::validation::legacy_support::legacy_horizon_plan(sweep_steps(features)),
        ) {
            legacy_rejection_operation("batch-horizon", &failure, false);
        }
    }
}

fn run_workload_sharing_controls() {
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
}

fn run_high_dimension_controls() {
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
}

fn run_evidence_controls() {
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
