//! Legacy fixed-control, horizon and chunked-prefix runners.

use std::cell::Cell;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{BatchFailure, StateModel, process_batch};
use seqvex::models::linear::rls::{Rls, RlsBatchError, RlsSample, RlsState};

use crate::evidence::{FailureRecord, OperationEvidence, RawWindow};
use crate::fixtures::{control_observation, fingerprint32, legacy_model};
use crate::measurement::{LegacyFailure, measure_legacy, measure_legacy_impl};
use crate::mode::legacy_validating;
use crate::validation::legacy_support::control_steps;
use crate::{BATCH, LAMBDA_CONTROL, WARMUP_WINDOWS};

pub(crate) fn legacy_horizon(
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
pub(crate) fn run_legacy_fixed_control(
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

pub(crate) fn sweep_state_after_outer_calls(
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
