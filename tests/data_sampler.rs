//! Public-surface contract tests for training-fold sampling.
//!
//! Scripted draws and zero-draw assertions live in the module's private test
//! suite, where the `sample_with_next` seam is reachable. These tests cover the
//! public behavior a caller observes: identity, multiplicity, ordering, seeded
//! replay, cutoff containment, splitter composition, the fragmented-group
//! fixture, and the returned overflow error.

#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use seqvex::preprocessing::sampling::{
    OutcomeInterval, SampleMethod, SampleUnit, SamplingError, SamplingMode, SamplingSpec,
    SequentialLimits, sample_training, sample_training_sequential_f32,
};
use seqvex::preprocessing::split::{
    RowWindowSpec, TailPolicy, TimeCoverage, TimeFold, TimeFoldOutcome, TimeRow, TimeWindowSpec,
    expanding, expanding_time,
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

fn fixture_rows() -> Vec<TimeRow> {
    (0..8)
        .map(|index| TimeRow {
            decision_at: DECISIONS[index],
            features_available_at: FEATURES_AVAILABLE[index],
            label_available_at: LABELS[index],
        })
        .collect()
}

fn expanding_folds(rows: &[TimeRow]) -> Vec<TimeFold> {
    expanding_time(
        8,
        rows,
        TimeCoverage::new(0, 30, 0),
        TimeWindowSpec::new(10, 5, 6, 2, 3),
    )
    .expect("fixture schedule")
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

fn merged_outcomes(supports: &[Vec<(i64, i64)>], rows: Range<usize>) -> Vec<OutcomeInterval> {
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

fn units_from_fold(
    fold: &TimeFold,
    decisions: &[i64],
    supports: &[Vec<(i64, i64)>],
) -> Vec<SampleUnit> {
    let mut units = Vec::new();
    for range in fold.train_ranges() {
        let mut start = range.start;
        while start < range.end {
            let mut end = start + 1;
            while end < range.end && decisions[end] == decisions[start] {
                end += 1;
            }
            units.push(SampleUnit {
                rows: vec![start..end],
                outcomes: merged_outcomes(supports, start..end),
                sequence: 0,
            });
            start = end;
        }
    }
    units
}

fn single(row: usize) -> SampleUnit {
    SampleUnit {
        rows: vec![row..row + 1],
        outcomes: vec![],
        sequence: 0,
    }
}

fn strict_unit(rows: &[Range<usize>], outcomes: &[(i64, i64)], sequence: u32) -> SampleUnit {
    SampleUnit {
        rows: rows.to_vec(),
        outcomes: outcomes
            .iter()
            .map(|&(start, end)| OutcomeInterval { start, end })
            .collect(),
        sequence,
    }
}

fn strict_single(row: usize, outcomes: &[(i64, i64)], sequence: u32) -> SampleUnit {
    strict_unit(&[row..row + 1], outcomes, sequence)
}

fn strict_normalized_units() -> Vec<SampleUnit> {
    (0..4)
        .map(|index| strict_single(index, &[(index as i64, index as i64 + 1)], 0))
        .collect()
}

fn expand_counts(units: &[SampleUnit], counts: &[usize]) -> Vec<usize> {
    let mut rows = Vec::new();
    for (unit, &count) in units.iter().zip(counts) {
        for _ in 0..count {
            for range in &unit.rows {
                rows.extend(range.clone());
            }
        }
    }
    rows
}

#[test]
fn cutoff_12_identity_and_without_replacement_keep_eligible_rows() {
    let rows = fixture_rows();
    let folds = expanding_folds(&rows);
    let fold = &folds[0];
    assert_eq!(fold.outcome(), TimeFoldOutcome::Usable);
    assert_eq!(fold.cutoff(), 12);
    assert_eq!(fold.train_ranges(), &[0..1, 2..3]);

    let units = [single(0), single(2)];
    let identity = sample_training(&units, SamplingSpec::disabled()).expect("identity");
    assert_eq!(identity.row_indices(), &[0, 2]);
    assert_eq!(identity.unit_counts(), &[1, 1]);

    let sampled = sample_training(
        &units,
        SamplingSpec::new(
            SamplingMode::Method(SampleMethod::WithoutReplacement),
            2,
            17,
        ),
    )
    .expect("sampled");
    assert_eq!(sampled.row_indices(), &[0, 2]);
    for &row in sampled.row_indices() {
        assert!(matches!(row, 0 | 2));
        assert!(!fold.test_range().contains(&row));
    }
}

#[test]
fn seeded_replay_and_grouped_multiplicity() {
    let units = [
        single(0),
        SampleUnit {
            rows: vec![1..3],
            outcomes: vec![],
            sequence: 0,
        },
    ];
    let spec = SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 5, 7);
    let first = sample_training(&units, spec).expect("first");
    let second = sample_training(&units, spec).expect("second");
    assert_eq!(first, second);
    assert_eq!(first.unit_counts().iter().sum::<usize>(), 5);

    let mut expanded = Vec::new();
    for (unit, &count) in first.unit_counts().iter().enumerate() {
        for _ in 0..count {
            expanded.extend(units[unit].rows.iter().cloned().flatten());
        }
    }
    assert_eq!(first.row_indices(), expanded.as_slice());
}

#[test]
fn cutoff_24_groups_equal_decisions_and_replays() {
    let rows = fixture_rows();
    let folds = expanding_folds(&rows);
    let fold = &folds[2];
    assert_eq!(fold.cutoff(), 24);
    assert_eq!(fold.train_ranges(), &[0..3, 4..5, 6..7]);

    let supports = support_map();
    let units = units_from_fold(fold, &DECISIONS, &supports);
    let shapes: Vec<(Vec<Range<usize>>, Vec<OutcomeInterval>)> = units
        .iter()
        .map(|unit| (unit.rows.clone(), unit.outcomes.clone()))
        .collect();
    assert_eq!(
        shapes,
        vec![
            (vec![0..1], vec![OutcomeInterval { start: 1, end: 4 }]),
            (vec![1..3], vec![OutcomeInterval { start: 4, end: 7 }]),
            (vec![4..5], vec![OutcomeInterval { start: 12, end: 15 }]),
            (vec![6..7], vec![OutcomeInterval { start: 17, end: 20 }]),
        ]
    );

    let spec = SamplingSpec::new(SamplingMode::Default, 4, 9);
    let first = sample_training(&units, spec).expect("first");
    let second = sample_training(&units, spec).expect("second");
    assert_eq!(first, second);
    assert_eq!(first.unit_counts().iter().sum::<usize>(), 4);
    for &row in first.row_indices() {
        assert!(matches!(row, 0 | 1 | 2 | 4 | 6), "unexpected row {row}");
        assert_ne!(row, 3);
        assert_ne!(row, 5);
        assert_ne!(row, 7);
    }
}

#[test]
fn frozen_cutoff_24_sample_matches_seed_31() {
    let rows = fixture_rows();
    let folds = expanding_folds(&rows);
    let supports = support_map();
    let units = units_from_fold(&folds[2], &DECISIONS, &supports);
    let sample = sample_training(&units, SamplingSpec::new(SamplingMode::Default, 4, 31))
        .expect("seeded sample");
    assert_eq!(sample.row_indices(), &[0, 1, 2, 4, 4]);
    assert_eq!(sample.unit_counts(), &[1, 1, 2, 0]);
}

#[test]
fn excluded_support_mutation_leaves_seeded_sample_unchanged() {
    let rows = fixture_rows();
    let folds = expanding_folds(&rows);
    let fold = &folds[2];

    let baseline_supports = support_map();
    let baseline_units = units_from_fold(fold, &DECISIONS, &baseline_supports);
    let spec = SamplingSpec::new(SamplingMode::Default, 4, 21);
    let baseline = sample_training(&baseline_units, spec).expect("baseline");

    let mut mutated_supports = support_map();
    mutated_supports[3] = vec![(100, 140)];
    mutated_supports[5] = vec![(200, 260)];
    mutated_supports[7] = vec![(300, 330)];

    assert_eq!(fold.train_ranges(), &[0..3, 4..5, 6..7]);
    let mutated_units = units_from_fold(fold, &DECISIONS, &mutated_supports);
    let eligible_shapes = |units: &[SampleUnit]| -> Vec<(Vec<Range<usize>>, Vec<OutcomeInterval>)> {
        units
            .iter()
            .map(|unit| (unit.rows.clone(), unit.outcomes.clone()))
            .collect()
    };
    assert_eq!(
        eligible_shapes(&baseline_units),
        eligible_shapes(&mutated_units)
    );
    assert_eq!(fold.test_range(), 7..8);

    let mutated = sample_training(&mutated_units, spec).expect("mutated");
    assert_eq!(baseline.row_indices(), mutated.row_indices());
    assert_eq!(baseline.unit_counts(), mutated.unit_counts());
}

#[test]
fn fragmented_group_at_one_timestamp_stays_one_unit() {
    let rows = vec![
        TimeRow {
            decision_at: 4,
            features_available_at: 3,
            label_available_at: Some(4),
        },
        TimeRow {
            decision_at: 4,
            features_available_at: 3,
            label_available_at: None,
        },
        TimeRow {
            decision_at: 4,
            features_available_at: 3,
            label_available_at: Some(4),
        },
        TimeRow {
            decision_at: 6,
            features_available_at: 3,
            label_available_at: None,
        },
    ];
    let fold = expanding_time(
        4,
        &rows,
        TimeCoverage::new(4, 20, 4),
        TimeWindowSpec::new(1, 5, 20, 0, 1),
    )
    .expect("snapshot schedule")
    .next()
    .expect("one fold");
    assert_eq!(fold.outcome(), TimeFoldOutcome::Usable);
    assert_eq!(fold.train_ranges(), &[0..1, 2..3]);

    let units = [SampleUnit {
        rows: vec![0..1, 2..3],
        outcomes: vec![],
        sequence: 0,
    }];
    let identity = sample_training(&units, SamplingSpec::disabled()).expect("identity");
    assert_eq!(identity.row_indices(), &[0, 2]);

    let sampled = sample_training(
        &units,
        SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 2, 3),
    )
    .expect("sampled");
    assert_eq!(sampled.unit_counts(), &[2]);
    assert_eq!(sampled.row_indices().len(), 4);
    let mut rows = sampled.row_indices().to_vec();
    rows.sort_unstable();
    assert_eq!(rows, vec![0, 0, 2, 2]);
}

#[test]
fn stage_1_composition_keeps_holdout_rows_out() {
    let folds = expanding(10, RowWindowSpec::new(0, 3, 4, 1, 0, TailPolicy::Drop))
        .expect("row schedule")
        .collect::<Vec<_>>();
    let fold = &folds[0];
    assert_eq!(fold.train_range(), 0..3);
    assert_eq!(fold.test_range(), 3..7);

    let units = [single(0), single(1), single(2)];
    let sample = sample_training(
        &units,
        SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 3, 5),
    )
    .expect("sampled");
    assert_eq!(sample.row_indices(), &[0, 1, 2]);
    for &row in sample.row_indices() {
        assert!(!fold.test_range().contains(&row));
    }
}

#[test]
fn shuffle_preserves_multiset_and_sequence_groups() {
    let units = [
        single(0),
        single(1),
        SampleUnit {
            rows: vec![2..3],
            outcomes: vec![],
            sequence: 1,
        },
    ];
    let base = SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 5, 13);
    let ordered = sample_training(&units, base).expect("ordered");
    let shuffled = sample_training(&units, base.with_shuffle(true)).expect("shuffled");
    assert_eq!(ordered.unit_counts(), shuffled.unit_counts());
    let mut left = ordered.row_indices().to_vec();
    let mut right = shuffled.row_indices().to_vec();
    left.sort_unstable();
    right.sort_unstable();
    assert_eq!(left, right);
    if shuffled.unit_counts()[2] > 0 {
        let tail = &shuffled.row_indices()[shuffled.len() - shuffled.unit_counts()[2]..];
        assert!(tail.iter().all(|&row| row == 2));
    }
}

#[test]
fn public_errors_are_typed() {
    assert_eq!(
        sample_training(&[], SamplingSpec::new(SamplingMode::Default, 0, 1)),
        Err(SamplingError::ZeroSampleSize)
    );
    assert_eq!(
        sample_training(&[], SamplingSpec::disabled()),
        Err(SamplingError::NoEligibleUnits)
    );
    let two_rows = [SampleUnit {
        rows: vec![0..2],
        outcomes: vec![],
        sequence: 0,
    }];
    assert_eq!(
        sample_training(
            &two_rows,
            SamplingSpec::new(
                SamplingMode::Method(SampleMethod::WithReplacement),
                usize::MAX,
                1,
            ),
        ),
        Err(SamplingError::SampleLengthOverflow {
            computed: (usize::MAX as u128) * 2,
            limit: usize::MAX as u128,
        })
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn public_sequential_work_overflow_is_typed() {
    let units = [
        SampleUnit {
            rows: vec![0..1],
            outcomes: vec![OutcomeInterval { start: 1, end: 3 }],
            sequence: 0,
        },
        SampleUnit {
            rows: vec![1..2],
            outcomes: vec![OutcomeInterval { start: 5, end: 7 }],
            sequence: 0,
        },
        SampleUnit {
            rows: vec![2..3],
            outcomes: vec![OutcomeInterval { start: 9, end: 11 }],
            sequence: 0,
        },
        SampleUnit {
            rows: vec![3..4],
            outcomes: vec![OutcomeInterval { start: 13, end: 15 }],
            sequence: 0,
        },
    ];
    let error = sample_training(
        &units,
        SamplingSpec::new(SamplingMode::Default, 1_usize << 59, 1),
    )
    .expect_err("work overflow");
    assert_eq!(
        error,
        SamplingError::SamplingWorkOverflow {
            computed: 1_u128 << 64,
            limit: u64::MAX as u128,
        }
    );
}

#[test]
fn training_sample_accessors_are_available() {
    let units = [single(0)];
    let sample = sample_training(&units, SamplingSpec::disabled()).expect("identity");
    assert_eq!(sample.len(), 1);
    assert!(!sample.is_empty());
    assert_eq!(sample.unit_counts().len(), 1);
}

#[test]
fn strict_f32_seeded_replay_matches_e0_37() {
    let units = strict_normalized_units();
    let limits = SequentialLimits::default();
    for mode in [
        SamplingMode::Default,
        SamplingMode::Method(SampleMethod::SequentialBootstrap),
    ] {
        let spec = SamplingSpec::new(mode, 5, 0x1060_0000_0000_0001);
        let first = sample_training_sequential_f32(&units, spec, &limits).expect("first strict");
        let second = sample_training_sequential_f32(&units, spec, &limits).expect("second strict");
        assert_eq!(first, second, "strict replay for {mode:?}");
        assert_eq!(first.row_indices(), &[0, 0, 2, 2, 3]);
        assert_eq!(first.unit_counts(), &[2, 0, 2, 1]);
        assert_eq!(first.unit_counts().iter().sum::<usize>(), 5);
    }
}

#[test]
fn strict_f32_preflight_contract() {
    const L: u128 = 1 << 24;
    let defaults = SequentialLimits::default();
    let valid = [strict_single(0, &[(0, 1)], 0)];

    let mut limits = defaults;
    limits.max_units = 0;
    assert_eq!(
        sample_training_sequential_f32(
            &valid,
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &limits
        ),
        Err(SamplingError::InvalidLimits {
            field: "max_units",
            value: 0,
            min: 1,
            max: 1_048_576,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &valid,
            SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 1, 1,),
            &defaults,
        ),
        Err(SamplingError::UnsupportedMode {
            mode: SamplingMode::Method(SampleMethod::WithReplacement),
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &valid,
            SamplingSpec::new(SamplingMode::Default, 0, 1),
            &defaults,
        ),
        Err(SamplingError::ZeroSampleSize)
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::NoEligibleUnits)
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_unit(&[], &[(0, 1)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::EmptyUnitRows {
            unit: 0,
            span: None,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_unit(&[0..2, 1..3], &[(0, 2)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::UnitRowsOutOfOrder {
            unit: 0,
            previous_end: 2,
            start: 1,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[
                strict_single(0, &[(0, 1)], 1),
                strict_single(1, &[(1, 2)], 0),
            ],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::SequenceOutOfOrder {
            unit: 1,
            previous: 1,
            actual: 0,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_single(0, &[(5, 5)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::InvalidOutcomeInterval {
            unit: 0,
            span: 0,
            start: 5,
            end: 5,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_unit(&[0..1], &[(1, 4), (3, 6)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::OutcomeOutOfOrder {
            unit: 0,
            span: 1,
            previous_end: 4,
            start: 3,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_single(0, &[], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::MissingOutcomeSupport { unit: 0 })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_single(0, &[(0, (1_i64 << 53) + 1)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::OutcomeSpanOverflow {
            computed: (1_u128 << 53) + 1,
            limit: 1_u128 << 53,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &valid,
            SamplingSpec::new(SamplingMode::Default, (L + 1) as usize, 1),
            &defaults,
        ),
        Err(SamplingError::F32CountUnsupported {
            computed: L + 1,
            limit: L,
        })
    );
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_single(0, &[(0, (L + 1) as i64)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &defaults,
        ),
        Err(SamplingError::F32SpanUnsupported {
            unit: 0,
            computed: L + 1,
            limit: L,
        })
    );

    let two = [
        strict_single(0, &[(0, 1)], 0),
        strict_single(1, &[(1, 2)], 0),
    ];
    let mut limits = defaults;
    limits.max_units = 1;
    assert_eq!(
        sample_training_sequential_f32(
            &two,
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &limits
        ),
        Err(SamplingError::UnitLimitExceeded {
            computed: 2,
            limit: 1,
        })
    );
    let mut limits = defaults;
    limits.max_support_spans = 1;
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_unit(&[0..1], &[(0, 1), (2, 3)], 0)],
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &limits,
        ),
        Err(SamplingError::SupportSpanLimitExceeded {
            computed: 2,
            limit: 1,
        })
    );
    let identical = [
        strict_single(0, &[(0, 1)], 0),
        strict_single(1, &[(0, 1)], 0),
    ];
    let mut limits = defaults;
    limits.max_endpoint_scratch_bytes = 511;
    assert_eq!(
        sample_training_sequential_f32(
            &identical,
            SamplingSpec::new(SamplingMode::Default, 1, 1),
            &limits
        ),
        Err(SamplingError::EndpointScratchExceeded {
            computed: 512,
            limit: 511,
        })
    );
    let mut limits = defaults;
    limits.max_expanded_rows = 3;
    assert_eq!(
        sample_training_sequential_f32(
            &[strict_unit(&[0..1, 2..3], &[(0, 1)], 0)],
            SamplingSpec::new(SamplingMode::Default, 2, 1),
            &limits,
        ),
        Err(SamplingError::ExpandedRowsExceeded {
            computed: 4,
            limit: 3,
        })
    );
    let mut limits = defaults;
    limits.max_work_cells = 2;
    assert_eq!(
        sample_training_sequential_f32(
            &valid,
            SamplingSpec::new(SamplingMode::Default, 2, 1),
            &limits,
        ),
        Err(SamplingError::WorkLimitExceeded {
            computed: 4,
            limit: 2,
        })
    );
}

#[test]
fn strict_f32_admits_long_unused_gap() {
    const L: u128 = 1 << 24;
    let units = [
        strict_single(0, &[(0, 1)], 0),
        strict_single(1, &[((L + 2) as i64, (L + 3) as i64)], 0),
    ];
    let spec = SamplingSpec::new(SamplingMode::Default, 2, 5);
    let limits = SequentialLimits::default();
    let first = sample_training_sequential_f32(&units, spec, &limits).expect("gap first");
    let second = sample_training_sequential_f32(&units, spec, &limits).expect("gap second");
    assert_eq!(first, second);
    assert_eq!(first.unit_counts().iter().sum::<usize>(), 2);
    assert_eq!(first.row_indices().len(), 2);
    assert!(first.row_indices().iter().all(|&row| matches!(row, 0 | 1)));
}

#[test]
fn strict_f32_cutoff_24_matches_and_replays() {
    let rows = fixture_rows();
    let folds = expanding_folds(&rows);
    let fold = &folds[2];
    let supports = support_map();
    let units = units_from_fold(fold, &DECISIONS, &supports);
    assert_eq!(fold.outcome(), TimeFoldOutcome::Usable);
    assert_eq!(fold.cutoff(), 24);
    assert_eq!(fold.train_ranges(), &[0..3, 4..5, 6..7]);
    assert_eq!(fold.test_range(), 7..8, "holdout rows");
    let expected_rows = [0..1, 1..3, 4..5, 6..7];
    let expected_supports = [vec![(1, 4)], vec![(4, 7)], vec![(12, 15)], vec![(17, 20)]];
    assert_eq!(units.len(), expected_rows.len());
    for (unit, (rows, support)) in units
        .iter()
        .zip(expected_rows.iter().zip(expected_supports.iter()))
    {
        assert_eq!(unit.rows, vec![rows.clone()], "cutoff row group");
        assert_eq!(unit.sequence, 0, "cutoff sequence");
        let expected: Vec<OutcomeInterval> = support
            .iter()
            .map(|&(start, end)| OutcomeInterval { start, end })
            .collect();
        assert_eq!(unit.outcomes, expected, "cutoff support");
    }
    let eligible: Vec<usize> = units
        .iter()
        .flat_map(|unit| unit.rows.iter().flat_map(|range| range.clone()))
        .collect();
    assert_eq!(eligible, vec![0, 1, 2, 4, 6], "eligible training rows");
    assert!(
        units
            .iter()
            .flat_map(|unit| unit.rows.iter())
            .all(|range| !range.contains(&7)),
        "holdout row must stay out of the training units"
    );
    let spec = SamplingSpec::new(SamplingMode::Default, 4, 21);
    let limits = SequentialLimits::default();
    let first = sample_training_sequential_f32(&units, spec, &limits).expect("strict cutoff first");
    let second =
        sample_training_sequential_f32(&units, spec, &limits).expect("strict cutoff second");
    assert_eq!(first, second);
    assert_eq!(first.unit_counts().iter().sum::<usize>(), 4);
    assert_eq!(
        first.row_indices(),
        expand_counts(&units, first.unit_counts()).as_slice()
    );
    for &row in first.row_indices() {
        assert!(matches!(row, 0 | 1 | 2 | 4 | 6), "unexpected row {row}");
        assert!(!matches!(row, 3 | 5 | 7), "ineligible row {row}");
    }

    let mut mutated_supports = support_map();
    mutated_supports[3] = vec![(100, 140)];
    mutated_supports[5] = vec![(200, 260)];
    mutated_supports[7] = vec![(300, 330)];
    let mutated_units = units_from_fold(fold, &DECISIONS, &mutated_supports);
    let shapes = |units: &[SampleUnit]| -> Vec<(Vec<Range<usize>>, Vec<OutcomeInterval>)> {
        units
            .iter()
            .map(|unit| (unit.rows.clone(), unit.outcomes.clone()))
            .collect()
    };
    assert_eq!(shapes(&units), shapes(&mutated_units));
    let mutated = sample_training_sequential_f32(&mutated_units, spec, &limits)
        .expect("strict cutoff mutated");
    assert_eq!(first.row_indices(), mutated.row_indices());
    assert_eq!(first.unit_counts(), mutated.unit_counts());
}

#[test]
fn strict_f32_fragmented_and_sequence_controls() {
    let units = [
        strict_unit(&[0..1, 2..3], &[(0, 1), (2, 3)], 0),
        strict_single(3, &[(3, 4)], 0),
        strict_single(4, &[(4, 5)], 1),
    ];
    let spec = SamplingSpec::new(SamplingMode::Default, 4, 7);
    let limits = SequentialLimits::default();
    let first = sample_training_sequential_f32(&units, spec, &limits).expect("fragment first");
    let second = sample_training_sequential_f32(&units, spec, &limits).expect("fragment second");
    assert_eq!(first, second);
    assert_eq!(first.unit_counts().iter().sum::<usize>(), 4);
    assert_eq!(
        first.row_indices(),
        expand_counts(&units, first.unit_counts()).as_slice()
    );
    assert!(
        first
            .row_indices()
            .iter()
            .all(|&row| matches!(row, 0 | 2 | 3 | 4))
    );
    if first.unit_counts()[2] > 0 {
        let tail = &first.row_indices()[first.len() - first.unit_counts()[2]..];
        assert!(tail.iter().all(|&row| row == 4));
    }

    let shuffled_spec = spec.with_shuffle(true);
    let shuffled =
        sample_training_sequential_f32(&units, shuffled_spec, &limits).expect("fragment shuffled");
    let shuffled_replay = sample_training_sequential_f32(&units, shuffled_spec, &limits)
        .expect("fragment shuffled replay");
    assert_eq!(shuffled, shuffled_replay);
    assert_eq!(first.unit_counts(), shuffled.unit_counts());
    let mut ordered_rows = first.row_indices().to_vec();
    let mut reordered_rows = shuffled.row_indices().to_vec();
    ordered_rows.sort_unstable();
    reordered_rows.sort_unstable();
    assert_eq!(ordered_rows, reordered_rows);
    if shuffled.unit_counts()[2] > 0 {
        let tail = &shuffled.row_indices()[shuffled.len() - shuffled.unit_counts()[2]..];
        assert!(tail.iter().all(|&row| row == 4));
    }
}
