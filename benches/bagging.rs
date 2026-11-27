//! Decision-grade regression-bagging matrix for #106 S9b.
//!
//! Release: `CARGO_BUILD_JOBS=1 cargo +1.100.0 bench --locked --bench bagging`.
//! Debug/all-target: bounded smoke only; debug numbers are never evidence.
//!
//! Timing boundaries: a fit window is timed until the ready-to-serve bag is
//! returned, including checked Profile B conversion and internal fit temporaries;
//! elapsed and the closing allocation snapshot are read with that bag still
//! live, and only then is the bag dropped outside the window. Prediction times
//! the full ordered `predict_batch` chunk loop over the holdout, including output
//! allocation and destruction. Per-window allocation deltas come from
//! `common::allocation_totals()` snapshots taken outside the clock. Those totals
//! are cumulative requested allocation count and bytes since process start:
//! each allocation/realloc request adds `layout.size()`/full `new_size`
//! (realloc on shrink is not offset), deallocation does not subtract, and the
//! counters are process-wide. They are not net growth, peak live bytes, logical
//! payload, or measured RSS. Peak/retained fields are absolute tracked live bytes
//! (or deltas explicitly named) and include resident fixtures.

use std::hint::black_box;
use std::time::Instant;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::ensemble::bagging::{
    BaggingSpec, DecisionTreeBag, DecisionTreeBag64, OlsBag, OlsBag64,
};
use seqvex::models::linear::ols::{FitConfig32, FitConfig64, UnderflowPolicy, fit_f64};
use seqvex::models::trees::decision_tree::{TreeFitConfig, TreeUnderflowPolicy, fit_tree_f64};
use seqvex::preprocessing::sampling::{OutcomeInterval, SampleMethod, SampleUnit, SamplingMode};

#[allow(dead_code)]
mod common;

const FIT_WINDOWS: usize = 9;
const PREDICT_WINDOWS: usize = 20;
const WARM_UPS: usize = 2;
const TREE_C_ROWS_PER_UNIT: usize = 8;
const RELEASE_ROWS: usize = 2048;
const RELEASE_TRAIN: usize = RELEASE_ROWS * 3 / 4;
const RELEASE_MEMBERS: [usize; 2] = [1, 8];
const RELEASE_BATCHES: [usize; 3] = [1, 32, 128];
const SUPPLEMENTARY_SCENARIOS: usize = 3;

fn debug() -> bool {
    cfg!(debug_assertions)
}
fn rows() -> usize {
    if debug() { 128 } else { RELEASE_ROWS }
}
fn train_rows() -> usize {
    rows() * 3 / 4
}
fn holdout_rows() -> usize {
    rows() - train_rows()
}
fn member_counts() -> Vec<usize> {
    if debug() { vec![2] } else { vec![1, 8] }
}
fn batch_sizes() -> Vec<usize> {
    if debug() { vec![32] } else { vec![1, 32, 128] }
}
fn fit_windows() -> usize {
    if debug() { 1 } else { FIT_WINDOWS }
}
fn predict_windows() -> usize {
    if debug() { 1 } else { PREDICT_WINDOWS }
}
fn warm_ups() -> usize {
    if debug() { 0 } else { WARM_UPS }
}
fn tree_c_unit_count() -> usize {
    train_rows() / TREE_C_ROWS_PER_UNIT
}
fn tree_c_draws() -> usize {
    tree_c_unit_count()
}
fn tree_c_cutoff() -> i64 {
    train_rows() as i64
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

#[allow(clippy::single_range_in_vec_init)]
fn tree_c_units() -> Vec<SampleUnit> {
    (0..tree_c_unit_count())
        .map(|unit| {
            let start = unit * TREE_C_ROWS_PER_UNIT;
            let end = start + TREE_C_ROWS_PER_UNIT;
            SampleUnit {
                rows: vec![start..end],
                outcomes: vec![OutcomeInterval {
                    start: start as i64,
                    end: end as i64,
                }],
                sequence: 0,
            }
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

#[derive(Clone, Copy)]
struct Timing {
    median: f64,
    p95: f64,
    min: f64,
    max: f64,
    iqr: f64,
}

fn summarize(mut samples: Vec<u64>) -> Timing {
    samples.sort_unstable();
    let percentile = |fraction: f64| -> f64 {
        let last = samples.len() - 1;
        let index = (((last as f64) * fraction).round() as usize).min(last);
        samples[index] as f64
    };
    Timing {
        median: percentile(0.5),
        p95: percentile(0.95),
        min: samples[0] as f64,
        max: samples[samples.len() - 1] as f64,
        iqr: percentile(0.75) - percentile(0.25),
    }
}

/// Times a fit until the ready-to-serve bag is returned, then drops that bag
/// outside the window. The closing allocation snapshot is taken before the drop
/// so the timing and allocation boundaries agree.
fn measure_fit<T>(
    fit_case: &str,
    eligible_train_rows: usize,
    fit_rows_per_member: usize,
    mut fit: impl FnMut() -> T,
) -> Timing {
    for _ in 0..warm_ups() {
        drop(black_box(fit()));
    }
    let mut samples = Vec::with_capacity(fit_windows());
    for window in 0..fit_windows() {
        let (start_requests, start_bytes) = common::allocation_totals();
        let start = Instant::now();
        let candidate = fit();
        let elapsed = start.elapsed().as_nanos() as u64;
        let (end_requests, end_bytes) = common::allocation_totals();
        let requests = end_requests
            .checked_sub(start_requests)
            .expect("allocation request totals must be monotonic");
        let bytes = end_bytes
            .checked_sub(start_bytes)
            .expect("allocated byte totals must be monotonic");
        drop(black_box(candidate));
        samples.push(elapsed);
        println!(
            "bench_fit_window fit_case={fit_case} window={window} elapsed_ns={elapsed} calls=1 \
             eligible_train_rows={eligible_train_rows} fit_rows_per_member={fit_rows_per_member} \
             alloc_requests_delta={requests} alloc_requested_bytes_delta={bytes}"
        );
    }
    summarize(samples)
}

fn measure_predict(
    scenario: &str,
    fit_case: &str,
    rows: usize,
    batch: usize,
    mut predict: impl FnMut(),
) -> Timing {
    let calls = rows.div_ceil(batch);
    for _ in 0..warm_ups() {
        predict();
    }
    let mut samples = Vec::with_capacity(predict_windows());
    for window in 0..predict_windows() {
        let (start_requests, start_bytes) = common::allocation_totals();
        let start = Instant::now();
        predict();
        let elapsed = start.elapsed().as_nanos() as u64;
        let (end_requests, end_bytes) = common::allocation_totals();
        let requests = end_requests
            .checked_sub(start_requests)
            .expect("allocation request totals must be monotonic");
        let bytes = end_bytes
            .checked_sub(start_bytes)
            .expect("allocated byte totals must be monotonic");
        samples.push(elapsed);
        println!(
            "bench_predict_window scenario={scenario} fit_case={fit_case} window={window} \
             elapsed_ns={elapsed} calls={calls} rows={rows} alloc_requests_delta={requests} \
             alloc_requested_bytes_delta={bytes}"
        );
    }
    summarize(samples)
}

/// Isolated fit peak: fixtures resident, no active bag, peak read while the
/// fitted result is live. The result is dropped after the peak is read.
fn peak_while_live<T>(build: impl FnOnce() -> T) -> u64 {
    let base = common::live_bytes();
    common::reset_peak();
    let result = build();
    let peak = common::peak_bytes();
    let delta = peak
        .checked_sub(base)
        .expect("peak live bytes must be at least the pre-fit base");
    drop(black_box(result));
    delta
}

struct Replacement {
    start_live_abs: u64,
    peak_live_abs: u64,
    retained_after_old_drop_live_abs: u64,
}

/// Tracks the candidate fit while `active` is still live, then replaces it.
/// Peak is read while both the old and candidate bags are live; retained live is
/// read after the old bag is dropped and before scoring/timing outputs exist.
fn measure_replacement<T>(mut active: T, build: impl FnOnce() -> T) -> (T, Replacement) {
    let start_live_abs = common::live_bytes();
    common::reset_peak();
    let candidate = build();
    let peak_live_abs = common::peak_bytes();
    let old = std::mem::replace(&mut active, candidate);
    drop(old);
    let retained_after_old_drop_live_abs = common::live_bytes();
    (
        active,
        Replacement {
            start_live_abs,
            peak_live_abs,
            retained_after_old_drop_live_abs,
        },
    )
}

fn metrics(predictions: &[f64], actual: &[f64]) -> (f64, f64) {
    assert_eq!(predictions.len(), actual.len());
    let n = predictions.len() as f64;
    let mut absolute = 0.0;
    let mut squared = 0.0;
    for (prediction, target) in predictions.iter().zip(actual) {
        let error = prediction - target;
        absolute += error.abs();
        squared += error * error;
    }
    (absolute / n, (squared / n).sqrt())
}

fn print_quality(scenario: &str, predictions: &[f64], actual: &[f64]) {
    let (mae, rmse) = metrics(predictions, actual);
    println!(
        "bench_quality scenario={scenario} mae={mae:.6} rmse={rmse:.6} rows={}",
        actual.len()
    );
}

fn verify_batch64(
    batch_rows: &[Observation<Vector64>],
    scalar: &[f64],
    batch: usize,
    mut predict: impl FnMut(&[Observation<Vector64>]) -> Vec<f64>,
) {
    let mut offset = 0;
    for chunk in batch_rows.chunks(batch) {
        let values = predict(chunk);
        assert_eq!(
            values.len(),
            chunk.len(),
            "batch {batch} output length at offset {offset}"
        );
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                scalar[offset + index].to_bits(),
                "batch {batch} row {}",
                offset + index
            );
        }
        offset += chunk.len();
    }
    assert_eq!(offset, batch_rows.len());
}

fn verify_batch32(
    batch_rows: &[Observation<Vector>],
    scalar: &[f32],
    batch: usize,
    mut predict: impl FnMut(&[Observation<Vector>]) -> Vec<f32>,
) {
    let mut offset = 0;
    for chunk in batch_rows.chunks(batch) {
        let values = predict(chunk);
        assert_eq!(
            values.len(),
            chunk.len(),
            "batch {batch} output length at offset {offset}"
        );
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                scalar[offset + index].to_bits(),
                "batch {batch} row {}",
                offset + index
            );
        }
        offset += chunk.len();
    }
    assert_eq!(offset, batch_rows.len());
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    scenario: &str,
    fit_case: &str,
    estimator: &str,
    profile: &str,
    mode: &str,
    members: usize,
    batch: usize,
    rows: usize,
    fit: Timing,
    predict: Timing,
    fit_peak_delta: u64,
    replacement: &Replacement,
    retained_logical: usize,
) {
    println!(
        "bench_row scenario={scenario} fit_case={fit_case} estimator={estimator} profile={profile} mode={mode} members={members} batch={batch} rows={rows} calls={} \
         fit_ns_median={:.0} fit_ns_p95={:.0} fit_ns_iqr={:.0} fit_ns_min={:.0} fit_ns_max={:.0} \
         predict_ns_median={:.0} predict_ns_p95={:.0} predict_ns_iqr={:.0} predict_ns_min={:.0} predict_ns_max={:.0} \
         fit_peak_delta_bytes={fit_peak_delta} replacement_start_live_abs_bytes={} replacement_peak_live_abs_bytes={} retained_after_old_drop_live_abs_bytes={} retained_logical_bytes={retained_logical}",
        rows.div_ceil(batch),
        fit.median,
        fit.p95,
        fit.iqr,
        fit.min,
        fit.max,
        predict.median,
        predict.p95,
        predict.iqr,
        predict.min,
        predict.max,
        replacement.start_live_abs,
        replacement.peak_live_abs,
        replacement.retained_after_old_drop_live_abs,
    );
}

fn run_tree_a(
    features: &[Vector64],
    targets: &[f64],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector64>],
    actual: &[f64],
) {
    for members in member_counts() {
        let spec = BaggingSpec::new(
            members,
            SamplingMode::Method(SampleMethod::WithReplacement),
            train_rows(),
            0x1060_u64,
            0,
        );
        let fit_case = format!("tree_A_with_replacement_m{members}");
        let fit = || {
            DecisionTreeBag64::fit(features, targets, units, &spec, &TreeFitConfig::default())
                .expect("tree A fit")
        };
        let fit_timing = measure_fit(&fit_case, train_rows(), train_rows(), fit);
        let fit_peak = peak_while_live(fit);
        let bag = fit();
        let (bag, replacement) = measure_replacement(bag, fit);
        let scalar: Vec<f64> = batch_rows
            .iter()
            .map(|row| bag.predict(row.value()).expect("tree A predict"))
            .collect();
        print_quality(&fit_case, &scalar, actual);
        for batch in batch_sizes() {
            let scenario = format!("{fit_case}_b{batch}");
            verify_batch64(batch_rows, &scalar, batch, |chunk| {
                bag.predict_batch(chunk, chunk.len()).expect("tree A batch")
            });
            let predict_timing =
                measure_predict(&scenario, &fit_case, batch_rows.len(), batch, || {
                    for chunk in batch_rows.chunks(batch) {
                        black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                    }
                });
            print_row(
                &scenario,
                &fit_case,
                "tree",
                "A",
                "with_replacement",
                members,
                batch,
                batch_rows.len(),
                fit_timing,
                predict_timing,
                fit_peak,
                &replacement,
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
    actual: &[f64],
) {
    for members in member_counts() {
        let spec = BaggingSpec::new(
            members,
            SamplingMode::Method(SampleMethod::WithReplacement),
            train_rows(),
            0x1061_u64,
            0,
        );
        let fit_case = format!("tree_B_with_replacement_m{members}");
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
        let fit_timing = measure_fit(&fit_case, train_rows(), train_rows(), fit);
        let fit_peak = peak_while_live(fit);
        let bag = fit();
        let (bag, replacement) = measure_replacement(bag, fit);
        let scalar32: Vec<f32> = batch_rows
            .iter()
            .map(|row| bag.predict(row.value()).expect("tree B predict"))
            .collect();
        let scalar: Vec<f64> = scalar32.iter().map(|value| f64::from(*value)).collect();
        print_quality(&fit_case, &scalar, actual);
        for batch in batch_sizes() {
            let scenario = format!("{fit_case}_b{batch}");
            verify_batch32(batch_rows, &scalar32, batch, |chunk| {
                bag.predict_batch(chunk, chunk.len()).expect("tree B batch")
            });
            let predict_timing =
                measure_predict(&scenario, &fit_case, batch_rows.len(), batch, || {
                    for chunk in batch_rows.chunks(batch) {
                        black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                    }
                });
            print_row(
                &scenario,
                &fit_case,
                "tree",
                "B",
                "with_replacement",
                members,
                batch,
                batch_rows.len(),
                fit_timing,
                predict_timing,
                fit_peak,
                &replacement,
                bag.retained_bytes(),
            );
        }
    }
}

fn run_tree_c(
    features: &[Vector],
    targets: &[f32],
    units: &[SampleUnit],
    batch_rows: &[Observation<Vector>],
    actual: &[f64],
) {
    for members in member_counts() {
        let spec = BaggingSpec::new(
            members,
            SamplingMode::Default,
            tree_c_draws(),
            0x1062_u64,
            0,
        )
        .with_support_cutoff(Some(tree_c_cutoff()));
        let fit_case = format!("tree_C_sequential_m{members}");
        let fit = || {
            DecisionTreeBag::fit_f32(features, targets, units, &spec, &TreeFitConfig::default())
                .expect("tree C fit")
        };
        let fit_timing = measure_fit(&fit_case, train_rows(), train_rows(), fit);
        let fit_peak = peak_while_live(fit);
        let bag = fit();
        let (bag, replacement) = measure_replacement(bag, fit);
        let scalar32: Vec<f32> = batch_rows
            .iter()
            .map(|row| bag.predict(row.value()).expect("tree C predict"))
            .collect();
        let scalar: Vec<f64> = scalar32.iter().map(|value| f64::from(*value)).collect();
        print_quality(&fit_case, &scalar, actual);
        for batch in batch_sizes() {
            let scenario = format!("{fit_case}_b{batch}");
            verify_batch32(batch_rows, &scalar32, batch, |chunk| {
                bag.predict_batch(chunk, chunk.len()).expect("tree C batch")
            });
            let predict_timing =
                measure_predict(&scenario, &fit_case, batch_rows.len(), batch, || {
                    for chunk in batch_rows.chunks(batch) {
                        black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                    }
                });
            print_row(
                &scenario,
                &fit_case,
                "tree",
                "C",
                "sequential",
                members,
                batch,
                batch_rows.len(),
                fit_timing,
                predict_timing,
                fit_peak,
                &replacement,
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
    actual: &[f64],
    mode: SamplingMode,
    sample_size: usize,
    label: &str,
    members_list: &[usize],
    batches: &[usize],
) {
    for &members in members_list {
        let spec = BaggingSpec::new(members, mode, sample_size, 0x77_u64, 0);
        let fit_case = format!("ols_A_{label}_m{members}");
        let fit = || {
            OlsBag64::fit(features, targets, units, &spec, &FitConfig64::default())
                .expect("OLS A fit")
        };
        let fit_rows_per_member = if sample_size == 0 {
            train_rows()
        } else {
            sample_size
        };
        let fit_timing = measure_fit(&fit_case, train_rows(), fit_rows_per_member, fit);
        let fit_peak = peak_while_live(fit);
        let bag = fit();
        let (bag, replacement) = measure_replacement(bag, fit);
        let scalar: Vec<f64> = batch_rows
            .iter()
            .map(|row| bag.predict(row.value()).expect("OLS A predict"))
            .collect();
        print_quality(&fit_case, &scalar, actual);
        for &batch in batches {
            let scenario = format!("{fit_case}_b{batch}");
            verify_batch64(batch_rows, &scalar, batch, |chunk| {
                bag.predict_batch(chunk, chunk.len()).expect("OLS A batch")
            });
            let predict_timing =
                measure_predict(&scenario, &fit_case, batch_rows.len(), batch, || {
                    for chunk in batch_rows.chunks(batch) {
                        black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                    }
                });
            print_row(
                &scenario,
                &fit_case,
                "ols",
                "A",
                label,
                members,
                batch,
                batch_rows.len(),
                fit_timing,
                predict_timing,
                fit_peak,
                &replacement,
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
    actual: &[f64],
    mode: SamplingMode,
    sample_size: usize,
    label: &str,
    members_list: &[usize],
    batches: &[usize],
) {
    for &members in members_list {
        let spec = BaggingSpec::new(members, mode, sample_size, 0x78_u64, 0);
        let fit_case = format!("ols_B_{label}_m{members}");
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
        let fit_rows_per_member = if sample_size == 0 {
            train_rows()
        } else {
            sample_size
        };
        let fit_timing = measure_fit(&fit_case, train_rows(), fit_rows_per_member, fit);
        let fit_peak = peak_while_live(fit);
        let bag = fit();
        let (bag, replacement) = measure_replacement(bag, fit);
        let scalar32: Vec<f32> = batch_rows
            .iter()
            .map(|row| bag.predict(row.value()).expect("OLS B predict"))
            .collect();
        let scalar: Vec<f64> = scalar32.iter().map(|value| f64::from(*value)).collect();
        print_quality(&fit_case, &scalar, actual);
        for &batch in batches {
            let scenario = format!("{fit_case}_b{batch}");
            verify_batch32(batch_rows, &scalar32, batch, |chunk| {
                bag.predict_batch(chunk, chunk.len()).expect("OLS B batch")
            });
            let predict_timing =
                measure_predict(&scenario, &fit_case, batch_rows.len(), batch, || {
                    for chunk in batch_rows.chunks(batch) {
                        black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                    }
                });
            print_row(
                &scenario,
                &fit_case,
                "ols",
                "B",
                label,
                members,
                batch,
                batch_rows.len(),
                fit_timing,
                predict_timing,
                fit_peak,
                &replacement,
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
    actual: &[f64],
    mode: SamplingMode,
    sample_size: usize,
    label: &str,
    members_list: &[usize],
    batches: &[usize],
) {
    for &members in members_list {
        let spec = BaggingSpec::new(members, mode, sample_size, 0x79_u64, 0);
        let fit_case = format!("ols_C_{label}_m{members}");
        let fit = || {
            OlsBag::fit_f32(features, targets, units, &spec, &FitConfig32::default())
                .expect("OLS C fit")
        };
        let fit_rows_per_member = if sample_size == 0 {
            train_rows()
        } else {
            sample_size
        };
        let fit_timing = measure_fit(&fit_case, train_rows(), fit_rows_per_member, fit);
        let fit_peak = peak_while_live(fit);
        let bag = fit();
        let (bag, replacement) = measure_replacement(bag, fit);
        let scalar32: Vec<f32> = batch_rows
            .iter()
            .map(|row| bag.predict(row.value()).expect("OLS C predict"))
            .collect();
        let scalar: Vec<f64> = scalar32.iter().map(|value| f64::from(*value)).collect();
        print_quality(&fit_case, &scalar, actual);
        for &batch in batches {
            let scenario = format!("{fit_case}_b{batch}");
            verify_batch32(batch_rows, &scalar32, batch, |chunk| {
                bag.predict_batch(chunk, chunk.len()).expect("OLS C batch")
            });
            let predict_timing =
                measure_predict(&scenario, &fit_case, batch_rows.len(), batch, || {
                    for chunk in batch_rows.chunks(batch) {
                        black_box(bag.predict_batch(chunk, chunk.len()).unwrap());
                    }
                });
            print_row(
                &scenario,
                &fit_case,
                "ols",
                "C",
                label,
                members,
                batch,
                batch_rows.len(),
                fit_timing,
                predict_timing,
                fit_peak,
                &replacement,
                bag.retained_bytes(),
            );
        }
    }
}

/// Release-matrix arithmetic, asserted on every run including the debug smoke.
fn print_release_matrix() {
    let per_profile = RELEASE_MEMBERS.len() * RELEASE_BATCHES.len();
    let planned = 5 * per_profile;
    let tree_c = per_profile;
    let total = planned + SUPPLEMENTARY_SCENARIOS + tree_c;
    let holdout = RELEASE_ROWS - RELEASE_TRAIN;
    let calls: Vec<usize> = RELEASE_BATCHES
        .iter()
        .map(|batch| holdout.div_ceil(*batch))
        .collect();
    let units = RELEASE_TRAIN / TREE_C_ROWS_PER_UNIT;
    let expanded = units * TREE_C_ROWS_PER_UNIT;
    let scratch = 64 * units + 64 * units + 256;
    let work = units * units * (2 * units);
    let aggregate = work * RELEASE_MEMBERS.iter().copied().max().unwrap();
    let main_fit_cases = 6 * RELEASE_MEMBERS.len();
    let supplementary_fit_cases = SUPPLEMENTARY_SCENARIOS;
    let fit_cases = main_fit_cases + supplementary_fit_cases;
    let planned_fit_windows = main_fit_cases * FIT_WINDOWS;
    let supplementary_fit_windows = supplementary_fit_cases * FIT_WINDOWS;
    let fit_windows_total = fit_cases * FIT_WINDOWS;
    let planned_scenarios = 6 * RELEASE_MEMBERS.len() * RELEASE_BATCHES.len();
    let scenario_rows = planned_scenarios + SUPPLEMENTARY_SCENARIOS;
    let predict_windows_total = scenario_rows * PREDICT_WINDOWS;

    assert_eq!(RELEASE_TRAIN, 1536);
    assert_eq!(holdout, 512);
    assert_eq!(per_profile, 6);
    assert_eq!(planned, 30);
    assert_eq!(tree_c, 6);
    assert_eq!(total, 39);
    assert_eq!(calls, vec![512, 16, 4]);
    assert_eq!(units, 192);
    assert_eq!(expanded, 1536);
    assert_eq!(scratch, 24_832);
    assert_eq!(work, 14_155_776);
    assert_eq!(aggregate, 113_246_208);
    assert_eq!(fit_cases, 15);
    assert_eq!(planned_fit_windows, 108);
    assert_eq!(supplementary_fit_windows, 27);
    assert_eq!(fit_windows_total, 135);
    assert_eq!(scenario_rows, 39);
    assert_eq!(predict_windows_total, 780);

    println!(
        "bench_release_matrix planned={planned} supplementary={SUPPLEMENTARY_SCENARIOS} tree_c={tree_c} total={total} holdout_rows={holdout} calls_b1={} calls_b32={} calls_b128={} tree_c_units={units} expanded_rows={expanded} scratch_bytes={scratch} work_per_call={work} aggregate_work={aggregate} fit_cases={fit_cases} planned_fit_windows={planned_fit_windows} supplementary_fit_windows={supplementary_fit_windows} fit_windows={fit_windows_total} scenario_rows={scenario_rows} predict_windows={predict_windows_total}",
        calls[0], calls[1], calls[2]
    );
}

fn print_tree_c_fixture() {
    let units = tree_c_unit_count();
    let draws = tree_c_draws();
    let expanded = draws * TREE_C_ROWS_PER_UNIT;
    let scratch = 64 * units + 64 * units + 256;
    let work = draws as u128 * units as u128 * (2 * units) as u128;
    let members = member_counts().iter().copied().max().unwrap_or(0);
    let aggregate = work * members as u128;
    println!(
        "bench_fixture name=tree_c units={units} draws={draws} cutoff={} expanded_rows={expanded} scratch_bytes={scratch} work_per_call={work} aggregate_work={aggregate}",
        tree_c_cutoff()
    );
}

#[allow(clippy::too_many_arguments)]
fn print_baselines(
    tree64: &[Vector64],
    y64: &[f64],
    tree_holdout: &[Observation<Vector64>],
    tree_actual: &[f64],
    ols64: &[Vector64],
    ols_y64: &[f64],
    ols_holdout: &[Observation<Vector64>],
    ols_actual: &[f64],
) {
    let indices: Vec<usize> = (0..train_rows()).collect();
    let tree = fit_tree_f64(tree64, y64, &indices, &TreeFitConfig::default())
        .expect("baseline tree fit")
        .into_predictor();
    let predictions: Vec<f64> = tree_holdout
        .iter()
        .map(|row| tree.predict(row.value()).expect("baseline tree predict"))
        .collect();
    print_quality("tree_single", &predictions, tree_actual);

    let mean = y64[..train_rows()].iter().sum::<f64>() / train_rows() as f64;
    print_quality(
        "tree_training_mean",
        &vec![mean; holdout_rows()],
        tree_actual,
    );

    let ols = fit_f64(ols64, ols_y64, &indices, &FitConfig64::default())
        .expect("baseline OLS fit")
        .into_predictor();
    let predictions: Vec<f64> = ols_holdout
        .iter()
        .map(|row| ols.predict(row.value()).expect("baseline OLS predict"))
        .collect();
    print_quality("ols_full_fit64", &predictions, ols_actual);
}

fn main() {
    let main_fit_cases = 6 * member_counts().len();
    let supplementary_fit_cases = SUPPLEMENTARY_SCENARIOS;
    let fit_cases = main_fit_cases + supplementary_fit_cases;
    let planned_fit_windows = main_fit_cases * fit_windows();
    let supplementary_fit_windows = supplementary_fit_cases * fit_windows();
    let fit_windows_total = fit_cases * fit_windows();
    let planned_scenarios = 6 * member_counts().len() * batch_sizes().len();
    let scenario_rows = planned_scenarios + SUPPLEMENTARY_SCENARIOS;
    let predict_windows_total = scenario_rows * predict_windows();

    if debug() {
        assert_eq!(
            (rows(), train_rows(), holdout_rows()),
            (128, 96, 32),
            "debug smoke bounds"
        );
        assert_eq!(member_counts(), vec![2], "debug member count");
        assert_eq!(batch_sizes(), vec![32], "debug batch size");
        assert_eq!(
            (fit_windows(), predict_windows(), warm_ups()),
            (1, 1, 0),
            "debug windows"
        );
        assert_eq!(
            (fit_cases, fit_windows_total),
            (9, 9),
            "debug fit cases/windows"
        );
        assert_eq!(
            (scenario_rows, predict_windows_total),
            (9, 9),
            "debug scenarios/prediction windows"
        );
        println!(
            "bench_debug_bounds rows=128 train=96 holdout=32 members=2 batch=32 fit_windows=1 predict_windows=1 warm_ups=0"
        );
    }

    print_release_matrix();

    println!(
        "bench_scenario_count fit_cases={fit_cases} planned_fit_cases={main_fit_cases} supplementary_fit_cases={supplementary_fit_cases} scenario_rows={scenario_rows} planned_scenarios={planned_scenarios} supplementary_scenarios={SUPPLEMENTARY_SCENARIOS}"
    );
    println!(
        "bench_protocol fit_windows_per_case={} predict_windows_per_scenario={} fit_windows={fit_windows_total} planned_fit_windows={planned_fit_windows} supplementary_fit_windows={supplementary_fit_windows} predict_windows={predict_windows_total} planned_process_captures=3 holdout_rows={} warm_ups={}",
        fit_windows(),
        predict_windows(),
        holdout_rows(),
        warm_ups(),
    );
    print_tree_c_fixture();

    let (tree64, y64, tree32, y32) = tree_data();
    let tree_units = units(train_rows());
    let tree_c_unit_set = tree_c_units();
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
    let tree_actual64: Vec<f64> = y64[train_rows()..].to_vec();
    let tree_actual32: Vec<f64> = y32[train_rows()..].iter().map(|v| f64::from(*v)).collect();

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
    let ols_actual64: Vec<f64> = ols_y64[train_rows()..].to_vec();
    let ols_actual32: Vec<f64> = ols_y32[train_rows()..]
        .iter()
        .map(|v| f64::from(*v))
        .collect();

    println!(
        "workload rows={} train={} test={} members={:?} batches={:?}",
        rows(),
        train_rows(),
        holdout_rows(),
        member_counts(),
        batch_sizes()
    );

    print_baselines(
        &tree64,
        &y64,
        &tree_batch64,
        &tree_actual64,
        &ols64,
        &ols_y64,
        &ols_batch64,
        &ols_actual64,
    );

    run_tree_a(&tree64, &y64, &tree_units, &tree_batch64, &tree_actual64);
    run_tree_b(&tree64, &y64, &tree_units, &tree_batch32, &tree_actual64);
    run_tree_c(
        &tree32,
        &y32,
        &tree_c_unit_set,
        &tree_batch32,
        &tree_actual32,
    );

    // Planned OLS matrix: Disabled.
    let all_members = member_counts();
    let all_batches = batch_sizes();
    run_ols_a(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch64,
        &ols_actual64,
        SamplingMode::Disabled,
        0,
        "disabled",
        &all_members,
        &all_batches,
    );
    run_ols_b(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch32,
        &ols_actual64,
        SamplingMode::Disabled,
        0,
        "disabled",
        &all_members,
        &all_batches,
    );
    run_ols_c(
        &ols32,
        &ols_y32,
        &ols_units,
        &ols_batch32,
        &ols_actual32,
        SamplingMode::Disabled,
        0,
        "disabled",
        &all_members,
        &all_batches,
    );

    // Supplementary previous evidence: without-replacement OLS, labeled separately.
    let supplement_members = if debug() { vec![2] } else { vec![8] };
    let supplement_batch = if debug() { vec![32] } else { vec![128] };
    let supplement_size = train_rows() * 2 / 3;
    run_ols_a(
        &ols64,
        &ols_y64,
        &ols_units,
        &ols_batch64,
        &ols_actual64,
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
        &ols_actual64,
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
        &ols_actual32,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        supplement_size,
        "without_replacement_supplementary",
        &supplement_members,
        &supplement_batch,
    );
}
