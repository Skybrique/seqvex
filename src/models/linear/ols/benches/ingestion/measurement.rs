use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;

use crate::common;
use crate::common::comparison::{
    RowEvidence, TimingMetadata, baseline_first, capture_from_env,
    memory_probe as shared_memory_probe, run_raw_row, timed_inputs,
};

use seqvex::models::linear::ols::{FitConfig32, FitConfig64, UnderflowPolicy, fit_f32, fit_f64};

use super::baseline::{black_box_dataset, materialize, parse_candidate, source_for};
use super::candidates::{Candidate, parse_rows, parse_rows_baseline};
use super::fit_profiles_support;
use super::fixtures::fixtures;
use super::reporting::{WORKFLOW_NAMES, complete_id, materialize_id, parse_id, workflow_id};

fn ordered_candidates() -> Vec<Candidate> {
    baseline_first(capture_from_env(), &Candidate::ALL)
}

fn run_row(timing: &BTreeMap<String, TimingMetadata>, id: &str, body: impl FnMut()) -> RowEvidence {
    let registration = timing
        .get(id)
        .unwrap_or_else(|| panic!("unregistered OLS timing row {id}"));
    run_raw_row(id, registration.calls, registration.units, "unit", body)
}

fn memory_probe<T>(id: &str, body: impl FnOnce() -> T) {
    let evidence = shared_memory_probe(body);
    println!(
        "{id}/mem before={} live={} peak={} after={}",
        evidence.before, evidence.live, evidence.peak, evidence.after
    );
}

pub(super) fn run_ingestion_rows(
    timing: &BTreeMap<String, TimingMetadata>,
    probes: &BTreeSet<String>,
    evidence: &mut Vec<RowEvidence>,
    probe_ids: &mut Vec<String>,
) {
    let fixtures = fixtures();
    for candidate in ordered_candidates() {
        for fixture in &fixtures {
            let id = parse_id(candidate, fixture.name);
            evidence.push(run_row(timing, &id, || {
                drop(black_box(timed_inputs(
                    (candidate, fixture.bytes.as_slice()),
                    |(candidate, input)| parse_rows(candidate, input).unwrap(),
                )));
            }));
            let id = complete_id(candidate, fixture.name);
            evidence.push(run_row(timing, &id, || {
                timed_inputs(
                    (candidate, fixture.bytes.as_slice()),
                    |(candidate, input)| {
                        let data = parse_candidate(candidate, input).unwrap();
                        black_box_dataset(&data);
                    },
                );
            }));
        }
    }
    for fixture in &fixtures {
        let rows = parse_rows_baseline(&fixture.bytes).unwrap();
        let id = materialize_id(fixture.name);
        evidence.push(run_row(timing, &id, || {
            timed_inputs(
                (rows.as_slice(), fixture.bytes.as_slice()),
                |(rows, bytes)| {
                    let data = materialize(rows, bytes.to_vec(), source_for(bytes));
                    black_box_dataset(&data);
                },
            );
        }));
    }
    for candidate in ordered_candidates() {
        for fixture in &fixtures {
            let id = complete_id(candidate, fixture.name);
            assert!(probes.contains(&id), "unregistered OLS probe {id}");
            memory_probe(&id, || parse_candidate(candidate, &fixture.bytes).unwrap());
            probe_ids.push(id);
        }
    }
    for fixture in &fixtures {
        let rows = parse_rows_baseline(&fixture.bytes).unwrap();
        let id = materialize_id(fixture.name);
        assert!(probes.contains(&id), "unregistered OLS probe {id}");
        memory_probe(&id, || {
            materialize(&rows, fixture.bytes.clone(), source_for(&fixture.bytes))
        });
        probe_ids.push(id);
    }
    for candidate in ordered_candidates() {
        for workflow in WORKFLOW_NAMES {
            let id = workflow_id(candidate, workflow);
            assert!(probes.contains(&id), "unregistered OLS probe {id}");
            workflow_memory_probe(candidate, workflow);
            probe_ids.push(id);
        }
    }
}

fn narrow_rows(
    features: &[seqvex::foundation::numerical::Vector64],
) -> Vec<seqvex::foundation::numerical::Vector> {
    features
        .iter()
        .enumerate()
        .map(|(row, feature)| {
            let value = feature.as_slice()[0];
            assert!(value.is_finite(), "B serving input row {row}: non-finite");
            let rounded = value as f32;
            assert!(rounded.is_finite(), "B serving input row {row}: overflow");
            assert!(
                !(value != 0.0 && rounded == 0.0),
                "B serving input row {row}: underflow"
            );
            seqvex::foundation::numerical::Vector::from_slice(&[rounded])
        })
        .collect()
}

pub(super) fn run_model_rows(
    timing: &BTreeMap<String, TimingMetadata>,
    evidence: &mut Vec<RowEvidence>,
) {
    let data = parse_candidate(Candidate::Baseline, fit_profiles_support::CANONICAL_CSV).unwrap();
    let train: Vec<usize> = (0..fit_profiles_support::TRAIN_ROWS).collect();
    let held: Vec<usize> = (fit_profiles_support::TRAIN_ROWS..data.features64.len()).collect();
    let fit_config64 = FitConfig64::default();
    let fit_config32 = FitConfig32::default();
    let a_fit = fit_f64(&data.features64, &data.targets64, &train, &fit_config64).unwrap();
    let b_model = a_fit.to_serving_f32(UnderflowPolicy::Reject).unwrap();
    let a_model = fit_f64(&data.features64, &data.targets64, &train, &fit_config64)
        .unwrap()
        .into_predictor();
    let c_model = fit_f32(&data.features32, &data.targets32, &train, &fit_config32)
        .unwrap()
        .into_predictor();
    let b_inputs = narrow_rows(&data.features64);

    let id = "model/afit/csv-can-8";
    evidence.push(run_row(timing, id, || {
        drop(black_box(timed_inputs(
            (&data, &train, &fit_config64),
            |(data, train, config)| {
                fit_f64(&data.features64, &data.targets64, train, config).unwrap()
            },
        )));
    }));
    let id = "model/cfit/csv-can-8";
    evidence.push(run_row(timing, id, || {
        drop(black_box(timed_inputs(
            (&data, &train, &fit_config32),
            |(data, train, config)| {
                fit_f32(&data.features32, &data.targets32, train, config).unwrap()
            },
        )));
    }));
    let id = "model/bconvert/csv-can-8";
    evidence.push(run_row(timing, id, || {
        drop(black_box(timed_inputs(
            (&a_fit, UnderflowPolicy::Reject),
            |(a_fit, policy)| a_fit.to_serving_f32(policy).unwrap(),
        )));
    }));
    let id = "model/aserve/csv-can-8";
    evidence.push(run_row(timing, id, || {
        timed_inputs((&a_model, &data, &held), |(model, data, held)| {
            for &index in held {
                black_box(model.predict_checked(&data.features64[index]).unwrap());
            }
        });
    }));
    let id = "model/bserve/csv-can-8";
    evidence.push(run_row(timing, id, || {
        timed_inputs((&b_model, &b_inputs, &held), |(model, inputs, held)| {
            for &index in held {
                black_box(model.predict_checked(&inputs[index]).unwrap());
            }
        });
    }));
    let id = "model/cserve/csv-can-8";
    evidence.push(run_row(timing, id, || {
        timed_inputs((&c_model, &data, &held), |(model, data, held)| {
            for &index in held {
                black_box(model.predict_checked(&data.features32[index]).unwrap());
            }
        });
    }));
}

struct WorkflowRetained {
    data: fit_profiles_support::Dataset,
    model: Box<dyn std::any::Any>,
    inputs: Option<Vec<seqvex::foundation::numerical::Vector>>,
}

fn workflow_once(candidate: Candidate, workflow: &str, bytes: &[u8]) -> WorkflowRetained {
    let data = parse_candidate(candidate, bytes).unwrap();
    let train: Vec<usize> = (0..fit_profiles_support::TRAIN_ROWS).collect();
    match workflow {
        "a" => {
            let held: Vec<usize> =
                (fit_profiles_support::TRAIN_ROWS..data.features64.len()).collect();
            let model = fit_f64(
                &data.features64,
                &data.targets64,
                &train,
                &FitConfig64::default(),
            )
            .unwrap()
            .into_predictor();
            for index in &held {
                black_box(model.predict_checked(&data.features64[*index]).unwrap());
            }
            WorkflowRetained {
                data,
                model: Box::new(model),
                inputs: None,
            }
        }
        "c" => {
            let held: Vec<usize> =
                (fit_profiles_support::TRAIN_ROWS..data.features32.len()).collect();
            let model = fit_f32(
                &data.features32,
                &data.targets32,
                &train,
                &FitConfig32::default(),
            )
            .unwrap()
            .into_predictor();
            for index in &held {
                black_box(model.predict_checked(&data.features32[*index]).unwrap());
            }
            WorkflowRetained {
                data,
                model: Box::new(model),
                inputs: None,
            }
        }
        "ab" => {
            let held: Vec<usize> =
                (fit_profiles_support::TRAIN_ROWS..data.features64.len()).collect();
            let a = fit_f64(
                &data.features64,
                &data.targets64,
                &train,
                &FitConfig64::default(),
            )
            .unwrap();
            let b = a.to_serving_f32(UnderflowPolicy::Reject).unwrap();
            let inputs = narrow_rows(&data.features64);
            for index in &held {
                black_box(b.predict_checked(&inputs[*index]).unwrap());
            }
            WorkflowRetained {
                data,
                model: Box::new(b),
                inputs: Some(inputs),
            }
        }
        _ => panic!("unknown workflow"),
    }
}

fn above_baseline(value: u64, baseline: u64, label: &str) -> u64 {
    value
        .checked_sub(baseline)
        .unwrap_or_else(|| panic!("{label} below baseline: value={value}, baseline={baseline}"))
}

fn workflow_memory_probe(candidate: Candidate, workflow: &str) {
    let before = common::live_bytes();
    common::reset_peak();
    let retained = workflow_once(candidate, workflow, fit_profiles_support::CANONICAL_CSV);
    let aggregate_live = common::live_bytes();
    let aggregate_peak = common::peak_bytes();
    black_box(&retained);
    let aggregate_retained = above_baseline(aggregate_live, before, "aggregate retained");
    let aggregate_peak_above_baseline = above_baseline(aggregate_peak, before, "aggregate peak");

    let WorkflowRetained {
        data,
        model,
        inputs,
    } = retained;
    drop(data);
    drop(inputs);
    let predictor_live = common::live_bytes();
    let predictor_retained = above_baseline(predictor_live, before, "predictor retained");

    common::reset_peak();
    let predictor_retained_peak = common::peak_bytes();
    let predictor_retained_peak_above_baseline =
        above_baseline(predictor_retained_peak, before, "predictor retained peak");

    drop(model);
    let after = common::live_bytes();
    assert_eq!(
        after,
        before,
        "workflow/{workflow}/{} after-drop baseline mismatch",
        candidate.name()
    );

    println!(
        "workflow/{workflow}/{}/aggregate/mem before={before} aggregate_live={aggregate_live} aggregate_peak={aggregate_peak} aggregate_retained={aggregate_retained} aggregate_peak_above_baseline={aggregate_peak_above_baseline}",
        candidate.name()
    );
    println!(
        "workflow/{workflow}/{}/predictor-retained/mem before={before} predictor_live={predictor_live} predictor_retained={predictor_retained} predictor_retained_peak={predictor_retained_peak} predictor_retained_peak_above_baseline={predictor_retained_peak_above_baseline}",
        candidate.name()
    );
    println!(
        "workflow/{workflow}/{}/drop/mem before={before} after={after}",
        candidate.name()
    );
}

pub(super) fn run_workflow_rows(
    timing: &BTreeMap<String, TimingMetadata>,
    evidence: &mut Vec<RowEvidence>,
) {
    for candidate in ordered_candidates() {
        for workflow in WORKFLOW_NAMES {
            let id = workflow_id(candidate, workflow);
            evidence.push(run_row(timing, &id, || {
                drop(black_box(timed_inputs(
                    (candidate, workflow, fit_profiles_support::CANONICAL_CSV),
                    |(candidate, workflow, bytes)| workflow_once(candidate, workflow, bytes),
                )));
            }));
        }
    }
}
