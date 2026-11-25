#![allow(clippy::single_range_in_vec_init)]

use seqvex::foundation::numerical::Vector64;
use seqvex::models::ensemble::bagging::{BaggingSpec, DecisionTreeBag64, OlsBag64};
use seqvex::models::linear::ols::FitConfig64;
use seqvex::models::trees::decision_tree::TreeFitConfig;
use seqvex::preprocessing::sampling::{
    OutcomeInterval, SampleMethod, SampleUnit, SamplingMode, SamplingSpec, sample_training,
};
use seqvex::preprocessing::split::{
    TimeCoverage, TimeFold, TimeRow, TimeWindowSpec, expanding_time,
};

const DECISIONS: [i64; 8] = [1, 4, 4, 9, 12, 12, 17, 24];
const FEATURES_AVAILABLE: [i64; 8] = [0, 2, 3, 9, 11, 11, 15, 23];
const LABELS: [Option<i64>; 8] = [
    Some(3),
    Some(12),
    Some(11),
    Some(10),
    Some(16),
    None,
    Some(21),
    Some(28),
];

fn rows() -> Vec<TimeRow> {
    (0..8)
        .map(|index| TimeRow {
            decision_at: DECISIONS[index],
            features_available_at: FEATURES_AVAILABLE[index],
            label_available_at: LABELS[index],
        })
        .collect()
}

fn folds(rows: &[TimeRow]) -> Vec<TimeFold> {
    expanding_time(
        8,
        rows,
        TimeCoverage::new(0, 30, 0),
        TimeWindowSpec::new(10, 5, 6, 2, 3),
    )
    .unwrap()
    .collect()
}

fn support_map() -> Vec<Vec<(i64, i64)>> {
    vec![
        vec![(1, 4)],
        vec![(4, 7)],
        vec![(4, 7)],
        vec![(9, 12)],
        vec![(12, 15)],
        vec![(12, 15)],
        vec![(17, 20)],
        vec![(24, 27)],
    ]
}

fn merged(supports: &[Vec<(i64, i64)>], rows: std::ops::Range<usize>) -> Vec<OutcomeInterval> {
    let mut spans: Vec<(i64, i64)> = rows
        .into_iter()
        .flat_map(|row| supports[row].iter().copied())
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (start, end) in spans {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
        .into_iter()
        .map(|(start, end)| OutcomeInterval { start, end })
        .collect()
}

fn units(fold: &TimeFold, supports: &[Vec<(i64, i64)>]) -> Vec<SampleUnit> {
    let mut units = Vec::new();
    for range in fold.train_ranges() {
        let mut start = range.start;
        while start < range.end {
            let mut end = start + 1;
            while end < range.end && DECISIONS[end] == DECISIONS[start] {
                end += 1;
            }
            units.push(SampleUnit {
                rows: vec![start..end],
                outcomes: merged(supports, start..end),
                sequence: 0,
            });
            start = end;
        }
    }
    units
}

fn shapes(units: &[SampleUnit]) -> Vec<(Vec<std::ops::Range<usize>>, Vec<OutcomeInterval>)> {
    units
        .iter()
        .map(|unit| (unit.rows.clone(), unit.outcomes.clone()))
        .collect()
}

#[test]
fn cutoff24_mutation_rebuilds_units_and_preserves_evidence() {
    let rows = rows();
    let folds = folds(&rows);
    let fold = &folds[2];
    assert_eq!(fold.train_ranges(), &[0..3, 4..5, 6..7]);
    assert_eq!(fold.test_range(), 7..8);

    let base_supports = support_map();
    let base_units = units(fold, &base_supports);
    assert_eq!(
        shapes(&base_units),
        vec![
            (vec![0..1], vec![OutcomeInterval { start: 1, end: 4 }]),
            (vec![1..3], vec![OutcomeInterval { start: 4, end: 7 }]),
            (vec![4..5], vec![OutcomeInterval { start: 12, end: 15 }]),
            (vec![6..7], vec![OutcomeInterval { start: 17, end: 20 }]),
        ]
    );

    let features: Vec<Vector64> = (0..8)
        .map(|index| Vector64::from_slice(&[index as f64, (index % 3) as f64]))
        .collect();
    let mut targets: Vec<f64> = (0..8).map(|index| index as f64).collect();
    let spec = BaggingSpec::new(
        1,
        SamplingMode::Method(SampleMethod::WithReplacement),
        4,
        0x5A,
        0,
    );
    let sample_spec = SamplingSpec::new(spec.mode(), spec.sample_size(), spec.member_seed(0));
    let before_sample = sample_training(&base_units, sample_spec).unwrap();
    let before = DecisionTreeBag64::fit(
        &features,
        &targets,
        &base_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    let query = 7usize;
    let before_prediction = before.predict(&features[query]).unwrap();

    let mut mutated_supports = support_map();
    mutated_supports[3] = vec![(100, 140)];
    mutated_supports[5] = vec![(200, 260)];
    mutated_supports[7] = vec![(300, 330)];
    targets[7] += 500.0;
    let mutated_units = units(fold, &mutated_supports);
    assert_eq!(shapes(&base_units), shapes(&mutated_units));

    let after_sample = sample_training(&mutated_units, sample_spec).unwrap();
    let after = DecisionTreeBag64::fit(
        &features,
        &targets,
        &mutated_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    assert_eq!(before_sample.row_indices(), after_sample.row_indices());
    assert_eq!(before_sample.unit_counts(), after_sample.unit_counts());
    assert_eq!(after.predict(&features[query]).unwrap(), before_prediction);
    println!(
        "CUTOFF24 mutation eligible_units={} rows_before={:?} rows_after={:?} counts_before={:?} counts_after={:?} query={} prediction={before_prediction:.6}",
        base_units.len(),
        before_sample.row_indices(),
        after_sample.row_indices(),
        before_sample.unit_counts(),
        after_sample.unit_counts(),
        query
    );
}

#[test]
fn ols_equal_timestamp_group_travels_together() {
    let rows = rows();
    let folds = folds(&rows);
    let fold = &folds[2];
    let units = units(fold, &support_map());
    assert_eq!(units.len(), 4);
    let eligible_rows: Vec<usize> = fold
        .train_ranges()
        .iter()
        .flat_map(|range| range.clone())
        .collect();
    assert_eq!(eligible_rows, vec![0, 1, 2, 4, 6]);

    let features: Vec<Vector64> = (0..8)
        .map(|row| Vector64::from_slice(&[row as f64]))
        .collect();
    let targets: Vec<f64> = (0..8).map(|row| 1.0 + 2.0 * row as f64).collect();
    let spec = BaggingSpec::new(
        2,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        2,
        0x1060_0000_0000_0002,
        0,
    );
    assert!(spec.sample_size() < units.len());

    let bag = OlsBag64::fit(&features, &targets, &units, &spec, &FitConfig64::default()).unwrap();
    let full = seqvex::models::linear::ols::fit_f64(
        &features,
        &targets,
        &eligible_rows,
        &FitConfig64::default(),
    )
    .unwrap()
    .into_predictor();

    let mut coverage = Vec::new();
    let mut group_selected = false;
    for member in 0..spec.members() {
        let sample = sample_training(
            &units,
            SamplingSpec::new(spec.mode(), spec.sample_size(), spec.member_seed(member)),
        )
        .unwrap();
        assert_eq!(sample.unit_counts().len(), units.len());
        assert!(sample.unit_counts().iter().all(|&count| count <= 1));
        assert_eq!(
            sample.unit_counts().iter().sum::<usize>(),
            spec.sample_size()
        );
        if sample.unit_counts()[1] == 1 {
            group_selected = true;
            assert!(sample.row_indices().contains(&1));
            assert!(sample.row_indices().contains(&2));
        }
        coverage.push(sample.unit_counts().to_vec());
    }
    assert!(
        group_selected,
        "fixture must select the equal-timestamp group"
    );

    let query = fold.test_range().start;
    let bag_error = bag.predict(&features[query]).unwrap() - targets[query];
    let full_error = full.predict(&features[query]).unwrap() - targets[query];
    println!(
        "OLS timestamp_group eligible_units={} sample_size={} seed=0x{:016x} coverage={coverage:?} group_rows=[1,2] group_selected={group_selected} bag_mae={:.6} bag_rmse={:.6} full_mae={:.6} full_rmse={:.6}",
        units.len(),
        spec.sample_size(),
        spec.member_seed(0),
        bag_error.abs(),
        (bag_error * bag_error).sqrt(),
        full_error.abs(),
        (full_error * full_error).sqrt()
    );
}
