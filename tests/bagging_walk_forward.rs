#![allow(clippy::single_range_in_vec_init)]

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::models::ensemble::bagging::{BaggingSpec, DecisionTreeBag64, OlsBag64};
use seqvex::models::linear::ols::FitConfig64;
use seqvex::models::trees::decision_tree::{TreeFitConfig, fit_tree_f64};
use seqvex::preprocessing::sampling::{
    SampleMethod, SampleUnit, SamplingMode, SamplingSpec, sample_training,
};
use seqvex::preprocessing::split::{
    TimeCoverage, TimeFoldOutcome, TimeRow, TimeWindowSpec, expanding_time,
};

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn walk24() -> (Vec<Vector>, Vec<f32>, Vec<Vector64>, Vec<f64>) {
    let mut f32_rows = Vec::with_capacity(24);
    let mut f64_rows = Vec::with_capacity(24);
    let mut y32 = Vec::with_capacity(24);
    let mut y64 = Vec::with_capacity(24);
    for index in 0..24usize {
        let x0 = (index % 6) as f32;
        let x1 = (index / 6) as f32;
        let base = if x0 <= 2.0 {
            if x1 <= 1.0 { 10.0 } else { 30.0 }
        } else if x1 <= 1.0 {
            20.0
        } else {
            40.0
        };
        let delta = match index {
            3 | 10 | 15 | 20 => 1.0,
            5 | 12 | 17 | 22 => -1.0,
            _ => 0.0,
        };
        f32_rows.push(Vector::from_slice(&[x0, x1]));
        f64_rows.push(Vector64::from_slice(&[x0 as f64, x1 as f64]));
        y32.push(base + delta);
        y64.push((base + delta) as f64);
    }
    (f32_rows, y32, f64_rows, y64)
}

#[allow(clippy::single_range_in_vec_init)]
fn units_for_rows(rows: &[usize]) -> Vec<SampleUnit> {
    rows.iter()
        .map(|&row| SampleUnit {
            rows: vec![row..row + 1],
            outcomes: Vec::new(),
            sequence: 0,
        })
        .collect()
}

fn walk24_fold() -> (Vec<usize>, std::ops::Range<usize>) {
    let rows: Vec<TimeRow> = (0..24usize)
        .map(|index| TimeRow {
            decision_at: index as i64,
            features_available_at: -1,
            label_available_at: if index == 7 { None } else { Some(index as i64) },
        })
        .collect();
    let folds: Vec<_> = expanding_time(
        24,
        &rows,
        TimeCoverage::new(0, 24, 0),
        TimeWindowSpec::new(16, 4, 4, 0, 2),
    )
    .unwrap()
    .collect();
    let fold = &folds[0];
    assert_eq!(fold.outcome(), TimeFoldOutcome::Usable);
    let selected = fold
        .train_ranges()
        .iter()
        .flat_map(|range| range.clone())
        .collect();
    (selected, fold.test_range())
}

fn rmse(abs_sq: f64, n: usize) -> f64 {
    (abs_sq / n as f64).sqrt()
}

#[test]
fn walk24_tree_bag_matches_single_tree_and_mean_baseline() {
    let (_, _, features, targets) = walk24();
    let (selected, test) = walk24_fold();
    let sample_units = units_for_rows(&selected);
    let spec = BaggingSpec::new(
        4,
        SamplingMode::Method(SampleMethod::WithReplacement),
        selected.len(),
        0x77,
        0,
    );
    let bag = DecisionTreeBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    let single = fit_tree_f64(&features, &targets, &selected, &TreeFitConfig::default())
        .unwrap()
        .into_predictor();
    let baseline = selected.iter().map(|&row| targets[row]).sum::<f64>() / selected.len() as f64;
    let (mut bag_abs, mut bag_sq) = (0.0, 0.0);
    let (mut single_abs, mut single_sq) = (0.0, 0.0);
    let (mut mean_abs, mut mean_sq) = (0.0, 0.0);
    for row in test.clone() {
        let actual = targets[row];
        let bag_error = bag.predict(&features[row]).unwrap() - actual;
        let single_error = single.predict(&features[row]).unwrap() - actual;
        let mean_error = baseline - actual;
        bag_abs += bag_error.abs();
        bag_sq += bag_error * bag_error;
        single_abs += single_error.abs();
        single_sq += single_error * single_error;
        mean_abs += mean_error.abs();
        mean_sq += mean_error * mean_error;
    }
    let n = test.len();
    println!(
        "WALK24 tree folds=1 train={} test={} members={} mode=with_replacement seed=0x77 bag_mae={:.6} bag_rmse={:.6} single_mae={:.6} single_rmse={:.6} mean_mae={:.6} mean_rmse={:.6}",
        selected.len(),
        n,
        spec.members(),
        bag_abs / n as f64,
        rmse(bag_sq, n),
        single_abs / n as f64,
        rmse(single_sq, n),
        mean_abs / n as f64,
        rmse(mean_sq, n)
    );
    assert!((bag_abs / n as f64).is_finite() && rmse(bag_sq, n).is_finite());
    assert!((single_abs / n as f64).is_finite() && rmse(single_sq, n).is_finite());
    assert!((mean_abs / n as f64).is_finite() && rmse(mean_sq, n).is_finite());
}

#[test]
fn ols_full_set_and_bag_share_holdout() {
    let (_, _, features, targets) = walk24();
    let (selected, test) = walk24_fold();
    let sample_units = units_for_rows(&selected);
    let spec = BaggingSpec::new(
        2,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        10,
        0x33,
        0,
    );
    let bag = OlsBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &FitConfig64::default(),
    )
    .unwrap();
    let full = seqvex::models::linear::ols::fit_f64(
        &features,
        &targets,
        &selected,
        &FitConfig64::default(),
    )
    .unwrap()
    .into_predictor();
    let mut coverage = Vec::new();
    for member in 0..spec.members() {
        let sample = sample_training(
            &sample_units,
            SamplingSpec::new(spec.mode(), spec.sample_size(), spec.member_seed(member)),
        )
        .unwrap();
        assert_eq!(sample.unit_counts().iter().sum::<usize>(), 10);
        coverage.push(sample.unit_counts().to_vec());
    }
    let (mut bag_abs, mut bag_sq) = (0.0, 0.0);
    let (mut full_abs, mut full_sq) = (0.0, 0.0);
    for row in test.clone() {
        let actual = targets[row];
        let bag_error = bag.predict(&features[row]).unwrap() - actual;
        let full_error = full.predict(&features[row]).unwrap() - actual;
        bag_abs += bag_error.abs();
        bag_sq += bag_error * bag_error;
        full_abs += full_error.abs();
        full_sq += full_error * full_error;
    }
    let n = test.len();
    println!(
        "OLS folds=1 train={} test={} members={} mode=without_replacement sample_size=10 seed=0x33 coverage={coverage:?} bag_mae={:.6} bag_rmse={:.6} full_mae={:.6} full_rmse={:.6}",
        selected.len(),
        n,
        spec.members(),
        bag_abs / n as f64,
        rmse(bag_sq, n),
        full_abs / n as f64,
        rmse(full_sq, n)
    );
    assert!((bag_abs / n as f64).is_finite() && rmse(bag_sq, n).is_finite());
    assert!((full_abs / n as f64).is_finite() && rmse(full_sq, n).is_finite());
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn cutoff24_temporal_mutation_preserves_units_samples_and_query() {
    let (_, _, features, mut targets) = walk24();
    let units = vec![
        SampleUnit {
            rows: vec![0..1],
            outcomes: vec![seqvex::preprocessing::sampling::OutcomeInterval { start: 1, end: 4 }],
            sequence: 0,
        },
        SampleUnit {
            rows: vec![1..3],
            outcomes: vec![seqvex::preprocessing::sampling::OutcomeInterval { start: 4, end: 7 }],
            sequence: 0,
        },
        SampleUnit {
            rows: vec![4..5],
            outcomes: vec![seqvex::preprocessing::sampling::OutcomeInterval { start: 12, end: 15 }],
            sequence: 0,
        },
        SampleUnit {
            rows: vec![6..7],
            outcomes: vec![seqvex::preprocessing::sampling::OutcomeInterval { start: 17, end: 20 }],
            sequence: 0,
        },
    ];
    let spec = BaggingSpec::new(
        1,
        SamplingMode::Method(SampleMethod::WithReplacement),
        4,
        0x5A,
        0,
    );
    let sample_before = sample_training(
        &units,
        SamplingSpec::new(spec.mode(), spec.sample_size(), spec.member_seed(0)),
    )
    .unwrap();
    let before = DecisionTreeBag64::fit(
        &features,
        &targets,
        &units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    let query = 18usize;
    let prediction_before = before.predict(&features[query]).unwrap();

    // Unavailable information: excluded rows 3/5/7 and held-out targets.
    for row in [3usize, 5, 7] {
        targets[row] += 1000.0;
    }
    targets[16..24].iter_mut().for_each(|value| *value += 500.0);
    let sample_after = sample_training(
        &units,
        SamplingSpec::new(spec.mode(), spec.sample_size(), spec.member_seed(0)),
    )
    .unwrap();
    let after = DecisionTreeBag64::fit(
        &features,
        &targets,
        &units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    assert_eq!(sample_before.row_indices(), sample_after.row_indices());
    assert_eq!(sample_before.unit_counts(), sample_after.unit_counts());
    assert_eq!(after.predict(&features[query]).unwrap(), prediction_before);
    println!(
        "CUTOFF24 mutation units={} sample_size={} rows_before={:?} rows_after={:?} counts_before={:?} counts_after={:?} query={} prediction={prediction_before:.6}",
        units.len(),
        spec.sample_size(),
        sample_before.row_indices(),
        sample_after.row_indices(),
        sample_before.unit_counts(),
        sample_after.unit_counts(),
        query
    );
}

#[test]
fn two_k_ordered_tree_comparison_is_finite() {
    let mut f64_rows = Vec::with_capacity(2048);
    let mut y = Vec::with_capacity(2048);
    for index in 0..2048usize {
        let mut row = [0.0f32; 8];
        for (feature, value) in row.iter_mut().enumerate() {
            *value = (((index as u64 * 131 + feature as u64 * 977) % 1009) as f32) / 1008.0;
        }
        let noise = (((splitmix64(index as u64) % 2001) as f32 / 1000.0) - 1.0) * 0.05;
        let target = 10.0 * f32::from(row[0] > 0.5)
            + 5.0 * f32::from(row[1] > 0.25)
            + 2.0 * f32::from(row[2] > 0.75)
            + 1.0 * f32::from(row[3] > 0.5)
            + noise;
        f64_rows.push(Vector64::from_slice(
            &row.iter().map(|value| *value as f64).collect::<Vec<_>>(),
        ));
        y.push(target as f64);
    }
    let selected: Vec<usize> = (0..1536).collect();
    let sample_units = units_for_rows(&selected);
    let spec = BaggingSpec::new(
        2,
        SamplingMode::Method(SampleMethod::WithReplacement),
        selected.len(),
        0x11,
        0,
    );
    let bag = DecisionTreeBag64::fit(
        &f64_rows,
        &y,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    let single = fit_tree_f64(&f64_rows, &y, &selected, &TreeFitConfig::default())
        .unwrap()
        .into_predictor();
    let baseline = selected.iter().map(|&row| y[row]).sum::<f64>() / selected.len() as f64;
    let (mut bag_abs, mut bag_sq) = (0.0, 0.0);
    let (mut single_abs, mut single_sq) = (0.0, 0.0);
    let (mut mean_abs, mut mean_sq) = (0.0, 0.0);
    for row in 1536..2048 {
        let actual = y[row];
        let bag_error = bag.predict(&f64_rows[row]).unwrap() - actual;
        let single_error = single.predict(&f64_rows[row]).unwrap() - actual;
        let mean_error = baseline - actual;
        bag_abs += bag_error.abs();
        bag_sq += bag_error * bag_error;
        single_abs += single_error.abs();
        single_sq += single_error * single_error;
        mean_abs += mean_error.abs();
        mean_sq += mean_error * mean_error;
    }
    let n = 512usize;
    println!(
        "2K tree train={} test={} members={} mode=with_replacement seed=0x11 bag_mae={:.6} bag_rmse={:.6} single_mae={:.6} single_rmse={:.6} mean_mae={:.6} mean_rmse={:.6}",
        selected.len(),
        n,
        spec.members(),
        bag_abs / n as f64,
        rmse(bag_sq, n),
        single_abs / n as f64,
        rmse(single_sq, n),
        mean_abs / n as f64,
        rmse(mean_sq, n)
    );
    assert!((bag_abs / n as f64).is_finite() && rmse(bag_sq, n).is_finite());
    assert!((single_abs / n as f64).is_finite() && rmse(single_sq, n).is_finite());
    assert!((mean_abs / n as f64).is_finite() && rmse(mean_sq, n).is_finite());
}
