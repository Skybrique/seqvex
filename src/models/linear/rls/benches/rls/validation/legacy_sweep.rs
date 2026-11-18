//! Batch-sweep equivalence and injection controls.

use std::cell::{Cell, RefCell};
use std::hint::black_box;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{BatchFailure, process_batch};
use seqvex::models::linear::rls::{Rls, RlsBatchError, RlsSample, RlsState};

use crate::SWEEP_DIMENSIONS;
use crate::common;
use crate::evidence::{
    FailureRecord, OperationEvidence, RawWindow, SweepFailure, sweep_comparison_failure,
    sweep_failure_record,
};
use crate::fixtures::{fingerprint32, legacy_model, one_bit_mismatch, sample32, state_eq32};
use crate::measurement::{LegacyFailure, measure_legacy};
use crate::validation::legacy_controls::{legacy_horizon, sweep_state_after_outer_calls};
use crate::validation::legacy_support::{
    legacy_horizon_plan, sweep_batch_sizes, sweep_fixture, sweep_outer, sweep_steps,
};

#[derive(Debug, Default)]
pub(crate) struct SweepInjection {
    pub(super) feature: usize,
    pub(super) expected: bool,
    pub(super) fail_observation: Option<usize>,
    pub(super) comparison_invocations: Cell<u64>,
    pub(super) measurement_invocations: Cell<u64>,
    pub(super) stopped_on_precheck: Cell<bool>,
    pub(super) failure_record: RefCell<Option<FailureRecord>>,
}

fn count_comparison(injection: Option<&SweepInjection>) {
    if let Some(injection) = injection {
        injection
            .comparison_invocations
            .set(injection.comparison_invocations.get() + 1);
    }
}

pub(super) fn sweep_equivalence(
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

pub(crate) fn run_legacy_batch_sweep(
    raws: &mut Vec<RawWindow>,
    injection: Option<&SweepInjection>,
) {
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
