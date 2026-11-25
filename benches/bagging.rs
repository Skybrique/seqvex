//! Regression-bagging cost/memory matrix for the pre-#103 checkpoint.
//!
//! Release: `cargo bench --bench bagging`. Debug/all-target: bounded, not evidence.

use std::hint::black_box;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::ensemble::bagging::{
    BaggingSpec, DecisionTreeBag, DecisionTreeBag64, OlsBag, OlsBag64,
};
use seqvex::models::linear::ols::{FitConfig32, FitConfig64, UnderflowPolicy};
use seqvex::models::trees::decision_tree::{TreeFitConfig, TreeUnderflowPolicy};
use seqvex::preprocessing::sampling::{SampleMethod, SampleUnit, SamplingMode};

#[allow(dead_code)]
mod common;

fn debug() -> bool {
    cfg!(debug_assertions)
}
fn rows() -> usize {
    if debug() { 128 } else { 2048 }
}
fn train_rows() -> usize {
    rows() * 3 / 4
}
fn member_counts() -> Vec<usize> {
    if debug() { vec![1, 2] } else { vec![1, 8] }
}
fn batch_sizes() -> Vec<usize> {
    if debug() {
        vec![1, 16, 32]
    } else {
        vec![1, 32, 128]
    }
}
fn runs() -> usize {
    if debug() { 1 } else { 3 }
}

#[allow(clippy::single_range_in_vec_init)]
fn units(count: usize) -> Vec<SampleUnit> {
    (0..count)
        .map(|index| SampleUnit {
            rows: vec![index..index + 1],
            outcomes: Vec::new(),
            sequence: 0,
        })
        .collect()
}

fn tree_data() -> (Vec<Vector64>, Vec<f64>, Vec<Vector>, Vec<f32>) {
    let n = rows();
    let mut f64_rows = Vec::with_capacity(n);
    let mut f32_rows = Vec::with_capacity(n);
    let mut y64 = Vec::with_capacity(n);
    let mut y32 = Vec::with_capacity(n);
    for index in 0..n {
        let mut row = [0.0f32; 8];
        for (feature, value) in row.iter_mut().enumerate() {
            *value = (((index * 131 + feature * 977) % 1009) as f32) / 1008.0;
        }
        let target = 10.0 * f32::from(row[0] > 0.5)
            + 5.0 * f32::from(row[1] > 0.25)
            + 2.0 * f32::from(row[2] > 0.75)
            + f32::from(row[3] > 0.5);
        f32_rows.push(Vector::from_slice(&row));
        f64_rows.push(Vector64::from_slice(
            &row.iter().map(|value| *value as f64).collect::<Vec<_>>(),
        ));
        y32.push(target);
        y64.push(target as f64);
    }
    (f64_rows, y64, f32_rows, y32)
}

fn ols_data() -> (Vec<Vector64>, Vec<f64>, Vec<Vector>, Vec<f32>) {
    let n = rows();
    let mut f64_rows = Vec::with_capacity(n);
    let mut f32_rows = Vec::with_capacity(n);
    let mut y64 = Vec::with_capacity(n);
    let mut y32 = Vec::with_capacity(n);
    for index in 0..n {
        let mut row = [0.0f32; 8];
        for (feature, value) in row.iter_mut().enumerate() {
            *value = (((index * 17 + feature * 31) % 257) as f32) / 256.0;
        }
        let target = 1.0 + row.iter().map(|value| value * 0.5).sum::<f32>();
        f32_rows.push(Vector::from_slice(&row));
        f64_rows.push(Vector64::from_slice(
            &row.iter().map(|value| *value as f64).collect::<Vec<_>>(),
        ));
        y32.push(target);
        y64.push(target as f64);
    }
    (f64_rows, y64, f32_rows, y32)
}

fn peak_of(fit: impl FnOnce()) -> u64 {
    let base = common::live_bytes();
    common::reset_peak();
    fit();
    common::peak_bytes().saturating_sub(base)
}

fn replacement_peak<T>(baseline: u64, mut active: T, candidate: T) -> u64 {
    common::reset_peak();
    let old = std::mem::replace(&mut active, candidate);
    let peak = common::peak_bytes();
    drop(old);
    drop(active);
    peak.saturating_sub(baseline)
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    estimator: &str,
    profile: &str,
    mode: &str,
    members: usize,
    batch: usize,
    timed_rows: usize,
    fit_ns: f64,
    predict_ns: f64,
    fit_peak: u64,
    replacement_peak: u64,
    retained: usize,
) {
    println!(
        "{estimator}_{profile} estimator={estimator} profile={profile} mode={mode} members={members} batch={batch} timed_rows={timed_rows} fit_ns={fit_ns:.0} predict_ns={predict_ns:.1} fit_peak={fit_peak} replacement_peak={replacement_peak} retained={retained}"
    );
}

fn run_tree_a(
    features: &[Vector64],
    targets: &[f64],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector64>],
) {
    for members in member_counts() {
        let spec = BaggingSpec::new(
            members,
            SamplingMode::Method(SampleMethod::WithReplacement),
            train_rows(),
            0x1060,
            0,
        );
        let fit = || {
            DecisionTreeBag64::fit(features, targets, units, &spec, &TreeFitConfig::default())
                .expect("tree A fit")
        };
        let bag = fit();
        let fit_stats = common::measure_runs("tree_a_fit", 1, 1, runs(), || {
            black_box(fit());
        });
        let fit_peak = peak_of(|| {
            fit();
        });
        let replacement = {
            let base = common::live_bytes();
            replacement_peak(base, fit(), fit())
        };
        for batch in batch_sizes() {
            let stats = common::measure("tree_a_predict", 1, batch_rows.len(), || {
                for chunk in batch_rows.chunks(batch) {
                    black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                }
            });
            print_row(
                "tree",
                "A",
                "with_replacement",
                members,
                batch,
                batch_rows.len(),
                fit_stats.median,
                stats.median,
                fit_peak,
                replacement,
                bag.retained_bytes(),
            );
        }
    }
}

fn run_tree_b(
    features: &[Vector64],
    targets: &[f64],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector>],
) {
    for members in member_counts() {
        let spec = BaggingSpec::new(
            members,
            SamplingMode::Method(SampleMethod::WithReplacement),
            train_rows(),
            0x1061,
            0,
        );
        let fit = || {
            DecisionTreeBag::fit_from_f64(
                features,
                targets,
                units,
                &spec,
                &TreeFitConfig::default(),
                TreeUnderflowPolicy::Reject,
            )
            .expect("tree B fit")
        };
        let bag = fit();
        let fit_stats = common::measure_runs("tree_b_fit", 1, 1, runs(), || {
            black_box(fit());
        });
        let fit_peak = peak_of(|| {
            fit();
        });
        let replacement = {
            let base = common::live_bytes();
            replacement_peak(base, fit(), fit())
        };
        for batch in batch_sizes() {
            let stats = common::measure("tree_b_predict", 1, batch_rows.len(), || {
                for chunk in batch_rows.chunks(batch) {
                    black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                }
            });
            print_row(
                "tree",
                "B",
                "with_replacement",
                members,
                batch,
                batch_rows.len(),
                fit_stats.median,
                stats.median,
                fit_peak,
                replacement,
                bag.retained_bytes(),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ols_a(
    features: &[Vector64],
    targets: &[f64],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector64>],
    mode: SamplingMode,
    sample_size: usize,
    label: &str,
    members_list: &[usize],
    batches: &[usize],
) {
    for &members in members_list {
        let spec = BaggingSpec::new(members, mode, sample_size, 0x77, 0);
        let fit = || {
            OlsBag64::fit(features, targets, units, &spec, &FitConfig64::default())
                .expect("OLS A fit")
        };
        let bag = fit();
        let fit_stats = common::measure_runs("ols_a_fit", 1, 1, runs(), || {
            black_box(fit());
        });
        let fit_peak = peak_of(|| {
            fit();
        });
        let replacement = {
            let base = common::live_bytes();
            replacement_peak(base, fit(), fit())
        };
        for &batch in batches {
            let stats = common::measure("ols_a_predict", 1, batch_rows.len(), || {
                for chunk in batch_rows.chunks(batch) {
                    black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                }
            });
            print_row(
                "ols",
                "A",
                label,
                members,
                batch,
                batch_rows.len(),
                fit_stats.median,
                stats.median,
                fit_peak,
                replacement,
                bag.retained_bytes(),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ols_b(
    features: &[Vector64],
    targets: &[f64],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector>],
    mode: SamplingMode,
    sample_size: usize,
    label: &str,
    members_list: &[usize],
    batches: &[usize],
) {
    for &members in members_list {
        let spec = BaggingSpec::new(members, mode, sample_size, 0x78, 0);
        let fit = || {
            OlsBag::fit_from_f64(
                features,
                targets,
                units,
                &spec,
                &FitConfig64::default(),
                UnderflowPolicy::Reject,
            )
            .expect("OLS B fit")
        };
        let bag = fit();
        let fit_stats = common::measure_runs("ols_b_fit", 1, 1, runs(), || {
            black_box(fit());
        });
        let fit_peak = peak_of(|| {
            fit();
        });
        let replacement = {
            let base = common::live_bytes();
            replacement_peak(base, fit(), fit())
        };
        for &batch in batches {
            let stats = common::measure("ols_b_predict", 1, batch_rows.len(), || {
                for chunk in batch_rows.chunks(batch) {
                    black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                }
            });
            print_row(
                "ols",
                "B",
                label,
                members,
                batch,
                batch_rows.len(),
                fit_stats.median,
                stats.median,
                fit_peak,
                replacement,
                bag.retained_bytes(),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ols_c(
    features: &[Vector],
    targets: &[f32],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector>],
    mode: SamplingMode,
    sample_size: usize,
    label: &str,
    members_list: &[usize],
    batches: &[usize],
) {
    for &members in members_list {
        let spec = BaggingSpec::new(members, mode, sample_size, 0x79, 0);
        let fit = || {
            OlsBag::fit_f32(features, targets, units, &spec, &FitConfig32::default())
                .expect("OLS C fit")
        };
        let bag = fit();
        let fit_stats = common::measure_runs("ols_c_fit", 1, 1, runs(), || {
            black_box(fit());
        });
        let fit_peak = peak_of(|| {
            fit();
        });
        let replacement = {
            let base = common::live_bytes();
            replacement_peak(base, fit(), fit())
        };
        for &batch in batches {
            let stats = common::measure("ols_c_predict", 1, batch_rows.len(), || {
                for chunk in batch_rows.chunks(batch) {
                    black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                }
            });
            print_row(
                "ols",
                "C",
                label,
                members,
                batch,
                batch_rows.len(),
                fit_stats.median,
                stats.median,
                fit_peak,
                replacement,
                bag.retained_bytes(),
            );
        }
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let (tree64, y64, tree32, _y32) = tree_data();
    let tree_units = units(train_rows());
    let tree_batch64: Vec<_> = tree64
        .iter()
        .skip(train_rows())
        .map(|row| Observation::new(row.clone()))
        .collect();
    let tree_batch32: Vec<_> = tree32
        .iter()
        .skip(train_rows())
        .map(|row| Observation::new(row.clone()))
        .collect();
    let (ols64, ols_y64, ols32, ols_y32) = ols_data();
    let ols_units = units(train_rows());
    let ols_batch64: Vec<_> = ols64
        .iter()
        .skip(train_rows())
        .map(|row| Observation::new(row.clone()))
        .collect();
    let ols_batch32: Vec<_> = ols32
        .iter()
        .skip(train_rows())
        .map(|row| Observation::new(row.clone()))
        .collect();

    println!(
        "workload rows={} train={} test={} members={:?} batches={:?}",
        rows(),
        train_rows(),
        rows() - train_rows(),
        member_counts(),
        batch_sizes()
    );

    run_tree_a(&tree64, &y64, &tree_units, &tree_batch64);
    run_tree_b(&tree64, &y64, &tree_units, &tree_batch32);

    // Planned OLS matrix: Disabled.
    run_ols_a(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch64,
        SamplingMode::Disabled,
        0,
        "disabled",
        &member_counts(),
        &batch_sizes(),
    );
    run_ols_b(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch32,
        SamplingMode::Disabled,
        0,
        "disabled",
        &member_counts(),
        &batch_sizes(),
    );
    run_ols_c(
        &ols32,
        &ols_y32,
        &ols_units,
        &ols_batch32,
        SamplingMode::Disabled,
        0,
        "disabled",
        &member_counts(),
        &batch_sizes(),
    );

    // Supplementary previous evidence: without-replacement OLS at members=8, batch=128.
    let supplement_members = if debug() { vec![2] } else { vec![8] };
    let supplement_batch = if debug() { vec![32] } else { vec![128] };
    let supplement_size = train_rows() * 2 / 3;
    run_ols_a(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch64,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        supplement_size,
        "without_replacement_supplementary",
        &supplement_members,
        &supplement_batch,
    );
    run_ols_b(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch32,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        supplement_size,
        "without_replacement_supplementary",
        &supplement_members,
        &supplement_batch,
    );
    run_ols_c(
        &ols32,
        &ols_y32,
        &ols_units,
        &ols_batch32,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        supplement_size,
        "without_replacement_supplementary",
        &supplement_members,
        &supplement_batch,
    );
}
