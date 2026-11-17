//! Legacy workload and evidence controls.

use std::hint::black_box;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_one;
use seqvex::foundation::state::{StateModel, process_batch};
use seqvex::models::linear::rls::RlsSample;

use crate::candidates::{p_full_into32, p_sym_into32, update_sym32};
use crate::common;
use crate::evidence::{FailureRecord, RawWindow};
use crate::fixtures::{fingerprint32, legacy_model, workload_observations};
use crate::measurement::{
    LegacyFailure, SharingFailure, measure_control, measure_legacy, measure_sharing,
};
use crate::mode::{legacy_validating, set_unexpected_failure};
use crate::validation::legacy_support::{evidence_steps, workload_steps};
use crate::{BATCH, WARMUP_WINDOWS};

pub(crate) fn run_legacy_workloads(raws: &mut Vec<RawWindow>) {
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

pub(crate) fn run_legacy_evidence(raws: &mut Vec<RawWindow>) {
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
