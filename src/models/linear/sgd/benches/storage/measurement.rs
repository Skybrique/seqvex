//! Storage guards, row execution and memory probes.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdError, SgdSample, SgdSample64};
use tinyvec::TinyVec;

use super::candidates::{
    Candidate, ordered_candidates, states_equal32, states_equal64, update32, update64,
};
use super::common::comparison::{
    RowEvidence, TimingMetadata, memory_probe as shared_memory_probe, run_raw_row, timed_inputs,
};
use super::fixtures::{
    expected_manifest32, expected_manifest64, hex, manifest32, manifest64, observations32,
    observations64, warm32, warm64,
};
use super::metadata::{detached_id, sequence_id};
use super::{DETACHED_DIMS, DIMS, ETA32, ETA64, FIXTURE_LEN, WARM_START};

pub(super) fn guard_storage() {
    for d in DIMS {
        let observations = observations32(d);
        let model = Sgd::new(d, ETA32).unwrap();
        let warm = warm32(&model, &observations);
        let manifest = hex(&manifest32(d, &observations, &warm));
        assert_eq!(
            manifest,
            expected_manifest32(d),
            "f32 manifest mismatch D={d}"
        );
        println!("manifest f32 D={d} {manifest}");
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            for sequence in [32usize, 256] {
                let mut reference = warm.clone();
                let mut state = warm.clone();
                for observation in observations.iter().skip(WARM_START).take(sequence) {
                    reference = model.update(&reference, observation).unwrap();
                    state = update32(candidate, &model, &state, observation, &mut scratch).unwrap();
                    assert!(
                        states_equal32(&reference, &state),
                        "f32 {} D={d} sequence={sequence} bit mismatch",
                        candidate.name()
                    );
                }
                let final_index = WARM_START + sequence;
                let final_features = &observations[final_index].value().features;
                let reference_prediction = model.predict(&reference, final_features).unwrap();
                let candidate_prediction = model.predict(&state, final_features).unwrap();
                assert!(candidate_prediction.is_finite());
                assert_eq!(
                    candidate_prediction.to_bits(),
                    reference_prediction.to_bits(),
                    "f32 {} D={d} sequence={sequence} final prediction mismatch",
                    candidate.name()
                );
            }
        }
    }
    for d in DIMS {
        let observations = observations64(d);
        let model = Sgd64::new(d, ETA64).unwrap();
        let warm = warm64(&model, &observations);
        let manifest = hex(&manifest64(d, &observations, &warm));
        assert_eq!(
            manifest,
            expected_manifest64(d),
            "f64 manifest mismatch D={d}"
        );
        println!("manifest f64 D={d} {manifest}");
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            for sequence in [32usize, 256] {
                let mut reference = warm.clone();
                let mut state = warm.clone();
                for observation in observations.iter().skip(WARM_START).take(sequence) {
                    reference = model.update(&reference, observation).unwrap();
                    state = update64(candidate, &model, &state, observation, &mut scratch).unwrap();
                    assert!(
                        states_equal64(&reference, &state),
                        "f64 {} D={d} sequence={sequence} bit mismatch",
                        candidate.name()
                    );
                }
                let final_index = WARM_START + sequence;
                let final_features = &observations[final_index].value().features;
                let reference_prediction = model.predict(&reference, final_features).unwrap();
                let candidate_prediction = model.predict(&state, final_features).unwrap();
                assert!(candidate_prediction.is_finite());
                assert_eq!(
                    candidate_prediction.to_bits(),
                    reference_prediction.to_bits(),
                    "f64 {} D={d} sequence={sequence} final prediction mismatch",
                    candidate.name()
                );
            }
        }
    }

    for d in [3usize, 6, 8, 9] {
        let observations = observations32(d);
        let model = Sgd::new(d, ETA32).unwrap();
        let warm = warm32(&model, &observations);
        let sample = observations[WARM_START].value();
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            let state = update32(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut scratch,
            )
            .expect("f32 lifecycle update");
            assert_eq!(state.w.len(), d);
            let cloned = state.clone();
            drop(state);
            assert!(states_equal32(&cloned, &cloned));
            let moved = cloned;
            assert_eq!(moved.w.len(), d);
            drop(moved);
            let _ = sample;
            let empty: TinyVec<[f32; 8]> = TinyVec::new();
            assert!(empty.is_empty());
            let mut rejected_scratch = vec![0.0f32; d];
            let bad = Observation::new(SgdSample {
                features: Vector::from_fn(d, |j| if j == 0 { f32::NAN } else { 1.0 }),
                target: 1.0,
            });
            assert!(update32(candidate, &model, &warm, &bad, &mut rejected_scratch).is_err());
            let retried = update32(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut rejected_scratch,
            )
            .expect("f32 retry");
            let mut fresh = Vec::new();
            let fresh_state = update32(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut fresh,
            )
            .expect("f32 fresh");
            assert!(states_equal32(&retried, &fresh_state));
        }
    }
    for d in [3usize, 6, 8, 9] {
        let observations = observations64(d);
        let model = Sgd64::new(d, ETA64).unwrap();
        let warm = warm64(&model, &observations);
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            let state = update64(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut scratch,
            )
            .expect("f64 lifecycle update");
            assert_eq!(state.w.len(), d);
            let cloned = state.clone();
            drop(state);
            let moved = cloned;
            assert_eq!(moved.w.len(), d);
            drop(moved);
            let empty: TinyVec<[f64; 8]> = TinyVec::new();
            assert!(empty.is_empty());
            let bad = Observation::new(SgdSample64 {
                features: Vector64::from_fn(d, |j| if j == 0 { f64::NAN } else { 1.0 }),
                target: 1.0,
            });
            let mut rejected_scratch = vec![0.0f64; d];
            assert!(update64(candidate, &model, &warm, &bad, &mut rejected_scratch).is_err());
            let retried = update64(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut rejected_scratch,
            )
            .expect("f64 retry");
            let mut fresh = Vec::new();
            let fresh_state = update64(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut fresh,
            )
            .expect("f64 fresh");
            assert!(states_equal64(&retried, &fresh_state));
        }
    }

    let mut tiny: TinyVec<[f32; 8]> = TinyVec::new();
    assert!(tiny.is_empty());
    for (index, value) in [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
        tiny.push(*value);
        assert_eq!(tiny.len(), index + 1);
        assert_eq!(tiny[index].to_bits(), value.to_bits());
    }
    tiny.push(7.0);
    assert!(tiny.is_inline());
    tiny.push(8.0);
    assert!(!tiny.is_inline());
    let moved = tiny;
    assert_eq!(moved[8].to_bits(), 8.0f32.to_bits());
    let cloned = moved.clone();
    assert_eq!(cloned.as_slice()[0].to_bits(), 0.0f32.to_bits());
    let mut shrunk = moved;
    shrunk.truncate(3);
    assert!(!shrunk.is_inline());
    shrunk.push(10.0);
    assert_eq!(shrunk[3].to_bits(), 10.0f32.to_bits());
    drop(shrunk);
    assert!(
        DIMS.iter()
            .all(|&d| !Candidate::Array.applicable(d) || d <= 8)
    );
    assert!(!Candidate::Array.applicable(9));
    assert!(!Candidate::Array.applicable(32));
    assert!(!Candidate::Array.applicable(128));

    let model = Sgd::new(2, ETA32).unwrap();
    let state = model.initial_state();
    let sample = Observation::new(SgdSample {
        features: Vector::from_slice(&[1.0, 2.0]),
        target: 3.0,
    });
    let mut scratch = Vec::new();
    let before = state.clone();
    assert!(update32(Candidate::Direct, &model, &state, &sample, &mut scratch).is_ok());
    assert!(states_equal32(&state, &before));
    let bad = Observation::new(SgdSample {
        features: Vector::from_slice(&[f32::NAN, 0.0]),
        target: 1.0,
    });
    assert_eq!(
        update32(Candidate::Reuse, &model, &state, &bad, &mut scratch),
        Err(SgdError::NonFiniteInput)
    );
    assert!(states_equal32(&state, &before));
}

fn run_row(timing: &BTreeMap<String, TimingMetadata>, id: &str, body: impl FnMut()) -> RowEvidence {
    let registration = timing
        .get(id)
        .unwrap_or_else(|| panic!("unregistered SGD timing row {id}"));
    run_raw_row(id, registration.calls, registration.units, "update", body)
}

fn memory_probe<T>(id: &str, body: impl FnOnce() -> T) {
    let evidence = shared_memory_probe(body);
    println!(
        "{id}/mem before={} live={} peak={} after={}",
        evidence.before, evidence.live, evidence.peak, evidence.after
    );
}

pub(super) fn run_storage_rows(
    timing: &BTreeMap<String, TimingMetadata>,
    probes: &BTreeSet<String>,
    evidence: &mut Vec<RowEvidence>,
    probe_ids: &mut Vec<String>,
) {
    for d in DIMS {
        let observations = observations32(d);
        let model = Sgd::new(d, ETA32).unwrap();
        let warm = warm32(&model, &observations);
        println!(
            "manifest f32 D={d} {:02x?}",
            manifest32(d, &observations, &warm)
        );
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            if DETACHED_DIMS.contains(&d) {
                let id = detached_id("f32", candidate, d);
                let mut index = 0usize;
                evidence.push(run_row(timing, &id, || {
                    let sample_index = WARM_START + (index % (FIXTURE_LEN - WARM_START));
                    index += 1;
                    timed_inputs(
                        (
                            candidate,
                            &model,
                            &warm,
                            &observations[sample_index],
                            &mut scratch,
                        ),
                        |(candidate, model, warm, observation, scratch)| {
                            black_box(
                                update32(candidate, model, warm, observation, scratch).unwrap(),
                            );
                        },
                    );
                }));
            }
            for sequence in [32usize, 256] {
                let id = sequence_id(sequence, "f32", candidate, d);
                evidence.push(run_row(timing, &id, || {
                    timed_inputs(
                        (
                            candidate,
                            &model,
                            &warm,
                            observations.as_slice(),
                            &mut scratch,
                        ),
                        |(candidate, model, warm, observations, scratch)| {
                            let mut state = warm.clone();
                            for observation in observations.iter().skip(WARM_START).take(sequence) {
                                state = update32(candidate, model, &state, observation, scratch)
                                    .unwrap();
                            }
                            // Complete sequence includes one final prediction, not another update.
                            let final_index = WARM_START + sequence;
                            let final_observation = &observations[final_index];
                            let prediction = model
                                .predict(&state, &final_observation.value().features)
                                .unwrap();
                            black_box(prediction);
                            black_box(&state);
                        },
                    );
                }));
            }
            let id = sequence_id(256, "f32", candidate, d);
            assert!(probes.contains(&id), "unregistered SGD probe {id}");
            memory_probe(&id, || {
                let mut state = warm.clone();
                for observation in observations.iter().skip(WARM_START).take(256) {
                    state = update32(candidate, &model, &state, observation, &mut scratch).unwrap();
                }
                state
            });
            probe_ids.push(id);
        }
    }
    for d in DIMS {
        let observations = observations64(d);
        let model = Sgd64::new(d, ETA64).unwrap();
        let warm = warm64(&model, &observations);
        println!(
            "manifest f64 D={d} {:02x?}",
            manifest64(d, &observations, &warm)
        );
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            if DETACHED_DIMS.contains(&d) {
                let id = detached_id("f64", candidate, d);
                let mut index = 0usize;
                evidence.push(run_row(timing, &id, || {
                    let sample_index = WARM_START + (index % (FIXTURE_LEN - WARM_START));
                    index += 1;
                    timed_inputs(
                        (
                            candidate,
                            &model,
                            &warm,
                            &observations[sample_index],
                            &mut scratch,
                        ),
                        |(candidate, model, warm, observation, scratch)| {
                            black_box(
                                update64(candidate, model, warm, observation, scratch).unwrap(),
                            );
                        },
                    );
                }));
            }
            for sequence in [32usize, 256] {
                let id = sequence_id(sequence, "f64", candidate, d);
                evidence.push(run_row(timing, &id, || {
                    timed_inputs(
                        (
                            candidate,
                            &model,
                            &warm,
                            observations.as_slice(),
                            &mut scratch,
                        ),
                        |(candidate, model, warm, observations, scratch)| {
                            let mut state = warm.clone();
                            for observation in observations.iter().skip(WARM_START).take(sequence) {
                                state = update64(candidate, model, &state, observation, scratch)
                                    .unwrap();
                            }
                            // Complete sequence includes one final prediction, not another update.
                            let final_index = WARM_START + sequence;
                            let final_observation = &observations[final_index];
                            let prediction = model
                                .predict(&state, &final_observation.value().features)
                                .unwrap();
                            black_box(prediction);
                            black_box(&state);
                        },
                    );
                }));
            }
            let id = sequence_id(256, "f64", candidate, d);
            assert!(probes.contains(&id), "unregistered SGD probe {id}");
            memory_probe(&id, || {
                let mut state = warm.clone();
                for observation in observations.iter().skip(WARM_START).take(256) {
                    state = update64(candidate, &model, &state, observation, &mut scratch).unwrap();
                }
                state
            });
            probe_ids.push(id);
        }
    }
}
