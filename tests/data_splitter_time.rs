#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use seqvex::foundation::error::DimensionMismatch;
use seqvex::preprocessing::split::{
    ExclusionCounts, TimeCoverage, TimeFold, TimeFoldOutcome, TimeFolds, TimeRow, TimeSplitError,
    TimeWindowSpec, UnusableReason, expanding_time, rolling_time,
};

fn row(decision_at: i64, features_available_at: i64, label_available_at: Option<i64>) -> TimeRow {
    TimeRow {
        decision_at,
        features_available_at,
        label_available_at,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedFold {
    id: usize,
    cutoff: i64,
    test: Range<usize>,
    train: Vec<Range<usize>>,
    counts: ExclusionCounts,
    outcome: TimeFoldOutcome,
}

#[derive(Clone)]
struct Fixture {
    rows: Vec<TimeRow>,
    start: i64,
    end: i64,
    anchor: i64,
    train: u64,
    test: u64,
    step: u64,
    gap: u64,
    cap: usize,
}

impl Fixture {
    fn coverage(&self) -> TimeCoverage {
        TimeCoverage::new(self.start, self.end, self.anchor)
    }

    fn spec(&self) -> TimeWindowSpec {
        TimeWindowSpec::new(self.train, self.test, self.step, self.gap, self.cap)
    }

    fn expanding(&self) -> Result<TimeFolds<'_>, TimeSplitError> {
        expanding_time(self.rows.len(), &self.rows, self.coverage(), self.spec())
    }

    fn rolling(&self) -> Result<TimeFolds<'_>, TimeSplitError> {
        rolling_time(self.rows.len(), &self.rows, self.coverage(), self.spec())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Expanding,
    Rolling,
}

fn counts(
    train_candidates: usize,
    train_feature_unavailable: usize,
    train_label_unavailable: usize,
    train_eligible: usize,
    test_candidates: usize,
    test_feature_unavailable: usize,
    test_eligible: usize,
) -> ExclusionCounts {
    ExclusionCounts {
        train_candidates,
        train_feature_unavailable,
        train_label_unavailable,
        train_eligible,
        test_candidates,
        test_feature_unavailable,
        test_eligible,
    }
}

fn expected_folds(folds: TimeFolds<'_>) -> Vec<ExpectedFold> {
    folds
        .map(|fold: TimeFold| ExpectedFold {
            id: fold.scheduled_id(),
            cutoff: fold.cutoff(),
            test: fold.test_range(),
            train: fold.train_ranges().to_vec(),
            counts: fold.exclusion_counts(),
            outcome: fold.outcome(),
        })
        .collect()
}

fn oracle(fixture: &Fixture, mode: Mode) -> Vec<ExpectedFold> {
    let mut folds = Vec::new();
    let mut id = 0usize;
    let mut p = i128::from(fixture.anchor) + i128::from(fixture.train) + i128::from(fixture.gap);
    let end = i128::from(fixture.end);
    let test_width = i128::from(fixture.test);
    let step = i128::from(fixture.step);

    while p + test_width <= end {
        let cutoff = i64::try_from(p).expect("oracle cutoff fits i64");
        let q = p - i128::from(fixture.gap);
        let train_start = match mode {
            Mode::Expanding => i128::from(fixture.anchor),
            Mode::Rolling => q - i128::from(fixture.train),
        };

        let mut fold_counts = ExclusionCounts::default();
        let mut train: Vec<Range<usize>> = Vec::new();
        for (index, entry) in fixture.rows.iter().enumerate() {
            let decision = i128::from(entry.decision_at);
            if decision >= train_start && decision < q {
                fold_counts.train_candidates += 1;
                if entry.features_available_at >= entry.decision_at {
                    fold_counts.train_feature_unavailable += 1;
                    continue;
                }
                match entry.label_available_at {
                    Some(label) if label < cutoff => {
                        fold_counts.train_eligible += 1;
                        if let Some(last) = train.last_mut() {
                            if last.end == index {
                                last.end += 1;
                            } else {
                                train.push(index..index + 1);
                            }
                        } else {
                            train.push(index..index + 1);
                        }
                    }
                    _ => fold_counts.train_label_unavailable += 1,
                }
            }
        }

        let mut first = None;
        let mut last = None;
        for (index, entry) in fixture.rows.iter().enumerate() {
            let decision = i128::from(entry.decision_at);
            if decision >= p && decision < p + test_width {
                if first.is_none() {
                    first = Some(index);
                }
                last = Some(index);
            }
        }
        let test = match (first, last) {
            (Some(first), Some(last)) => first..last + 1,
            _ => {
                let mut insertion = 0usize;
                while insertion < fixture.rows.len()
                    && i128::from(fixture.rows[insertion].decision_at) < p
                {
                    insertion += 1;
                }
                insertion..insertion
            }
        };

        let mut first_failure = None;
        for index in test.clone() {
            let entry = &fixture.rows[index];
            fold_counts.test_candidates += 1;
            if entry.features_available_at >= entry.decision_at {
                fold_counts.test_feature_unavailable += 1;
                if first_failure.is_none() {
                    first_failure = Some(UnusableReason::UnavailableTestFeature {
                        row: index,
                        decision_at: entry.decision_at,
                        features_available_at: entry.features_available_at,
                    });
                }
            }
        }
        fold_counts.test_eligible =
            fold_counts.test_candidates - fold_counts.test_feature_unavailable;

        let outcome = if test.is_empty() {
            TimeFoldOutcome::Unusable(UnusableReason::EmptyTest)
        } else if let Some(reason) = first_failure {
            TimeFoldOutcome::Unusable(reason)
        } else if train.is_empty() {
            TimeFoldOutcome::Unusable(UnusableReason::EmptyEligibleTraining)
        } else {
            TimeFoldOutcome::Usable
        };

        folds.push(ExpectedFold {
            id,
            cutoff,
            test,
            train,
            counts: fold_counts,
            outcome,
        });

        p += step;
        id += 1;
    }

    folds
}

fn assert_schedule(fixture: &Fixture, mode: Mode) {
    let actual = match mode {
        Mode::Expanding => expected_folds(fixture.expanding().expect("expanding schedule")),
        Mode::Rolling => expected_folds(fixture.rolling().expect("rolling schedule")),
    };
    let expected = oracle(fixture, mode);
    assert_eq!(actual, expected);
}

fn assert_error(result: Result<TimeFolds<'_>, TimeSplitError>, expected: TimeSplitError) {
    match result {
        Err(actual) => assert_eq!(actual, expected),
        Ok(_) => panic!("expected {expected:?}, got a schedule"),
    }
}

fn primary() -> Fixture {
    Fixture {
        rows: vec![
            row(1, 0, Some(3)),
            row(4, 2, Some(12)),
            row(4, 3, Some(11)),
            row(9, 9, Some(10)),
            row(12, 11, Some(16)),
            row(12, 11, None),
            row(17, 15, Some(21)),
            row(24, 23, Some(28)),
        ],
        start: 0,
        end: 30,
        anchor: 0,
        train: 10,
        test: 5,
        step: 6,
        gap: 2,
        cap: 3,
    }
}

#[test]
fn primary_fixture_matches_pinned_memberships() {
    let fixture = primary();
    let expanding = expected_folds(fixture.expanding().expect("expanding schedule"));
    assert_eq!(
        expanding,
        vec![
            ExpectedFold {
                id: 0,
                cutoff: 12,
                test: 4..6,
                train: vec![0..1, 2..3],
                counts: counts(4, 1, 1, 2, 2, 0, 2),
                outcome: TimeFoldOutcome::Usable,
            },
            ExpectedFold {
                id: 1,
                cutoff: 18,
                test: 7..7,
                train: vec![0..3, 4..5],
                counts: counts(6, 1, 1, 4, 0, 0, 0),
                outcome: TimeFoldOutcome::Unusable(UnusableReason::EmptyTest),
            },
            ExpectedFold {
                id: 2,
                cutoff: 24,
                test: 7..8,
                train: vec![0..3, 4..5, 6..7],
                counts: counts(7, 1, 1, 5, 1, 0, 1),
                outcome: TimeFoldOutcome::Usable,
            },
        ]
    );

    let rolling = expected_folds(fixture.rolling().expect("rolling schedule"));
    assert_eq!(rolling.len(), 3);
    assert_eq!(rolling[0], expanding[0]);
    assert_eq!(
        rolling[1],
        ExpectedFold {
            id: 1,
            cutoff: 18,
            test: 7..7,
            train: vec![4..5],
            counts: counts(3, 1, 1, 1, 0, 0, 0),
            outcome: TimeFoldOutcome::Unusable(UnusableReason::EmptyTest),
        }
    );
    assert_eq!(
        rolling[2],
        ExpectedFold {
            id: 2,
            cutoff: 24,
            test: 7..8,
            train: vec![4..5, 6..7],
            counts: counts(3, 0, 1, 2, 1, 0, 1),
            outcome: TimeFoldOutcome::Usable,
        }
    );

    assert_schedule(&fixture, Mode::Expanding);
    assert_schedule(&fixture, Mode::Rolling);
}

#[test]
fn zero_gap_shared_tick_belongs_to_test() {
    let fixture = Fixture {
        rows: vec![
            row(9, 0, Some(9)),
            row(10, 0, Some(11)),
            row(12, 0, Some(13)),
        ],
        start: 0,
        end: 20,
        anchor: 0,
        train: 10,
        test: 5,
        step: 6,
        gap: 0,
        cap: 1,
    };
    let folds = expected_folds(fixture.expanding().expect("zero-gap schedule"));
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].test, 1..3);
    assert_eq!(folds[0].train, vec![0..1]);
    assert_eq!(folds[0].outcome, TimeFoldOutcome::Usable);
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn negative_shift_preserves_schedule() {
    let base = primary();
    let shift = -100i64;
    let shifted = Fixture {
        rows: base
            .rows
            .iter()
            .map(|entry| {
                row(
                    entry.decision_at + shift,
                    entry.features_available_at + shift,
                    entry.label_available_at.map(|label| label + shift),
                )
            })
            .collect(),
        start: base.start + shift,
        end: base.end + shift,
        anchor: base.anchor + shift,
        train: base.train,
        test: base.test,
        step: base.step,
        gap: base.gap,
        cap: base.cap,
    };

    let base_folds = expected_folds(base.expanding().expect("base schedule"));
    let shifted_folds = expected_folds(shifted.expanding().expect("shifted schedule"));
    assert_eq!(base_folds.len(), shifted_folds.len());
    for (base_fold, shifted_fold) in base_folds.iter().zip(&shifted_folds) {
        assert_eq!(base_fold.id, shifted_fold.id);
        assert_eq!(base_fold.cutoff + shift, shifted_fold.cutoff);
        assert_eq!(base_fold.test, shifted_fold.test);
        assert_eq!(base_fold.train, shifted_fold.train);
        assert_eq!(base_fold.counts, shifted_fold.counts);
        assert_eq!(base_fold.outcome, shifted_fold.outcome);
    }
    assert_schedule(&shifted, Mode::Expanding);
    assert_schedule(&shifted, Mode::Rolling);
}

#[test]
fn empty_dataset_yields_scheduled_empty_tests() {
    let fixture = Fixture {
        rows: Vec::new(),
        start: 0,
        end: 10,
        anchor: 0,
        train: 1,
        test: 2,
        step: 1,
        gap: 0,
        cap: 8,
    };
    let folds = expected_folds(fixture.expanding().expect("empty schedule"));
    assert_eq!(folds.len(), 8);
    for (index, fold) in folds.iter().enumerate() {
        assert_eq!(fold.id, index);
        assert_eq!(fold.cutoff, 1 + index as i64);
        assert_eq!(fold.test, 0..0);
        assert!(fold.train.is_empty());
        assert_eq!(fold.counts, ExclusionCounts::default());
        assert_eq!(
            fold.outcome,
            TimeFoldOutcome::Unusable(UnusableReason::EmptyTest)
        );
    }
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn empty_test_wins_with_no_eligible_training() {
    let mut fixture = primary();
    for entry in &mut fixture.rows[..=5] {
        entry.features_available_at = entry.decision_at;
    }
    let folds = expected_folds(fixture.expanding().expect("empty-test schedule"));
    let fold = &folds[1];
    assert_eq!(fold.id, 1);
    assert_eq!(fold.cutoff, 18);
    assert_eq!(fold.test, 7..7);
    assert!(fold.train.is_empty());
    assert_eq!(fold.counts, counts(6, 6, 0, 0, 0, 0, 0));
    assert_eq!(
        fold.outcome,
        TimeFoldOutcome::Unusable(UnusableReason::EmptyTest)
    );
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn unavailable_test_feature_wins_with_no_eligible_training() {
    let mut fixture = primary();
    for entry in &mut fixture.rows[..=3] {
        entry.features_available_at = entry.decision_at;
    }
    fixture.rows[4].features_available_at = 11;
    fixture.rows[5].features_available_at = 12;
    let folds = expected_folds(fixture.expanding().expect("test-feature schedule"));
    let fold = &folds[0];
    assert_eq!(fold.test, 4..6);
    assert!(fold.train.is_empty());
    assert_eq!(fold.counts, counts(4, 4, 0, 0, 2, 1, 1));
    assert_eq!(
        fold.outcome,
        TimeFoldOutcome::Unusable(UnusableReason::UnavailableTestFeature {
            row: 5,
            decision_at: 12,
            features_available_at: 12,
        })
    );
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn empty_eligible_training_is_explicit() {
    let mut fixture = primary();
    for entry in &mut fixture.rows[..=3] {
        entry.label_available_at = None;
    }
    fixture.rows[3].features_available_at = 8;
    let folds = expected_folds(fixture.expanding().expect("empty-training schedule"));
    let fold = &folds[0];
    assert_eq!(fold.test, 4..6);
    assert!(fold.train.is_empty());
    assert_eq!(fold.counts, counts(4, 0, 4, 0, 2, 0, 2));
    assert_eq!(
        fold.outcome,
        TimeFoldOutcome::Unusable(UnusableReason::EmptyEligibleTraining)
    );
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn mixed_training_failure_counts_feature_only() {
    let fixture = Fixture {
        rows: vec![row(1, 1, None), row(12, 11, Some(16))],
        start: 0,
        end: 30,
        anchor: 0,
        train: 10,
        test: 5,
        step: 6,
        gap: 2,
        cap: 3,
    };
    let folds = expected_folds(fixture.expanding().expect("mixed-reason schedule"));
    let fold = &folds[0];
    assert_eq!(fold.test, 1..2);
    assert_eq!(fold.counts, counts(1, 1, 0, 0, 1, 0, 1));
    assert_eq!(
        fold.outcome,
        TimeFoldOutcome::Unusable(UnusableReason::EmptyEligibleTraining)
    );
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn test_feature_equality_makes_fold_unusable() {
    let mut fixture = primary();
    fixture.rows[5].features_available_at = 12;
    let folds = expected_folds(fixture.expanding().expect("equality schedule"));
    let fold = &folds[0];
    assert_eq!(fold.test, 4..6);
    assert_eq!(fold.train, vec![0..1, 2..3]);
    assert_eq!(fold.counts, counts(4, 1, 1, 2, 2, 1, 1));
    assert_eq!(
        fold.outcome,
        TimeFoldOutcome::Unusable(UnusableReason::UnavailableTestFeature {
            row: 5,
            decision_at: 12,
            features_available_at: 12,
        })
    );
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn fragmented_training_ranges_preserve_order() {
    let fixture = Fixture {
        rows: vec![
            row(1, 0, Some(5)),
            row(2, 2, Some(5)),
            row(3, 0, Some(5)),
            row(4, 4, Some(5)),
            row(5, 0, Some(5)),
            row(6, 0, Some(10)),
        ],
        start: 0,
        end: 10,
        anchor: 1,
        train: 5,
        test: 2,
        step: 10,
        gap: 0,
        cap: 1,
    };
    let folds = expected_folds(fixture.expanding().expect("fragmented schedule"));
    let fold = &folds[0];
    assert_eq!(fold.test, 5..6);
    assert_eq!(fold.train, vec![0..1, 2..3, 4..5]);
    assert_eq!(fold.counts, counts(5, 2, 0, 3, 1, 0, 1));
    assert_eq!(fold.outcome, TimeFoldOutcome::Usable);
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn exact_coverage_end_is_one_window() {
    let fixture = Fixture {
        rows: primary().rows[..6].to_vec(),
        start: 0,
        end: 17,
        anchor: 0,
        train: 10,
        test: 5,
        step: 6,
        gap: 2,
        cap: 1,
    };
    let folds = expected_folds(fixture.expanding().expect("exact-end schedule"));
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].cutoff, 12);
    assert_eq!(folds[0].test, 4..6);
    assert_eq!(folds[0].train, vec![0..1, 2..3]);
    assert_eq!(folds[0].counts, counts(4, 1, 1, 2, 2, 0, 2));
    assert_eq!(folds[0].outcome, TimeFoldOutcome::Usable);
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn long_gap_excludes_training_and_test_roles() {
    let fixture = Fixture {
        rows: vec![
            row(2, 0, Some(4)),
            row(20, 0, Some(25)),
            row(55, 0, Some(60)),
            row(60, 0, Some(65)),
        ],
        start: 0,
        end: 100,
        anchor: 0,
        train: 5,
        test: 5,
        step: 100,
        gap: 50,
        cap: 1,
    };
    let folds = expected_folds(fixture.expanding().expect("long-gap schedule"));
    let fold = &folds[0];
    assert_eq!(fold.test, 2..3);
    assert_eq!(fold.train, vec![0..1]);
    assert_eq!(fold.counts, counts(1, 0, 0, 1, 1, 0, 1));
    assert_eq!(fold.outcome, TimeFoldOutcome::Usable);
    assert_schedule(&fixture, Mode::Expanding);
}

#[test]
fn iteration_is_fused_and_count_is_stable() {
    fn assert_fused<I: std::iter::FusedIterator>(_: &I) {}

    let fixture = primary();
    let mut folds = fixture.expanding().expect("fused schedule");
    assert_fused(&folds);
    assert_eq!(folds.scheduled_fold_count(), 3);
    assert_eq!(folds.size_hint(), (3, Some(3)));
    assert_eq!(folds.next().expect("first fold").scheduled_id(), 0);
    assert_eq!(folds.scheduled_fold_count(), 3);
    assert_eq!(folds.size_hint(), (2, Some(2)));
    assert_eq!(folds.by_ref().count(), 2);
    assert_eq!(folds.size_hint(), (0, Some(0)));
    assert!(folds.next().is_none());
    assert!(folds.next().is_none());
}

#[test]
fn configuration_precedence_advances_field_by_field() {
    let coverage = TimeCoverage::new(10, 0, 0);
    let rows = vec![row(0, 0, Some(0))];
    let row_count = rows.len() + 1;
    let cases = [
        (
            TimeWindowSpec::new(0, 0, 0, 0, 0),
            TimeSplitError::ZeroTrainDuration,
        ),
        (
            TimeWindowSpec::new(1, 0, 0, 0, 0),
            TimeSplitError::ZeroTestDuration,
        ),
        (
            TimeWindowSpec::new(1, 1, 0, 0, 0),
            TimeSplitError::ZeroStepDuration,
        ),
        (
            TimeWindowSpec::new(1, 1, 1, 0, 0),
            TimeSplitError::ZeroScheduleCap,
        ),
        (
            TimeWindowSpec::new(1, 1, 1, 0, 1),
            TimeSplitError::CoverageOrder { start: 10, end: 0 },
        ),
    ];
    for (spec, expected) in cases {
        assert_error(expanding_time(row_count, &rows, coverage, spec), expected);
    }
    assert_error(
        expanding_time(
            row_count,
            &rows,
            TimeCoverage::new(0, 10, 0),
            TimeWindowSpec::new(1, 1, 1, 0, 1),
        ),
        TimeSplitError::MetadataLengthMismatch {
            shape: DimensionMismatch {
                expected: row_count,
                actual: rows.len(),
            },
        },
    );
}

#[test]
fn anchor_boundaries_are_half_open() {
    let rows: [TimeRow; 0] = [];
    let spec = TimeWindowSpec::new(1, 1, 1, 0, 99);
    let coverage = TimeCoverage::new(0, 100, 0);
    assert!(expanding_time(0, &rows, coverage, spec).is_ok());

    let coverage = TimeCoverage::new(0, 100, 100);
    assert_error(
        expanding_time(0, &rows, coverage, spec),
        TimeSplitError::AnchorOutsideCoverage {
            anchor: 100,
            start: 0,
            end: 100,
        },
    );
}

#[test]
fn metadata_length_mismatch_uses_dimension_mismatch() {
    let rows = vec![row(0, 0, Some(0)); 7];
    let coverage = TimeCoverage::new(0, 10, 0);
    let spec = TimeWindowSpec::new(1, 1, 1, 0, 10);
    let error = match expanding_time(8, &rows, coverage, spec) {
        Err(error) => error,
        Ok(_) => panic!("expected metadata length mismatch"),
    };
    assert_eq!(
        error,
        TimeSplitError::MetadataLengthMismatch {
            shape: DimensionMismatch {
                expected: 8,
                actual: 7,
            },
        }
    );
    assert!(std::error::Error::source(&error).is_some());
    assert!(format!("{error}").contains("metadata"));
}

#[test]
fn metadata_classes_are_reported_across_the_snapshot() {
    let coverage = TimeCoverage::new(0, 100, 0);
    let spec = TimeWindowSpec::new(1, 1, 1, 0, 1);

    let ordering = vec![row(-1, 0, Some(0)), row(3, 0, Some(3)), row(2, 0, Some(2))];
    assert_error(
        expanding_time(3, &ordering, coverage, spec),
        TimeSplitError::DecisionOutOfOrder {
            row: 2,
            previous: 3,
            actual: 2,
        },
    );

    let coverage_rows = vec![row(1, 0, Some(0)), row(100, 0, Some(100))];
    assert_error(
        expanding_time(2, &coverage_rows, coverage, spec),
        TimeSplitError::DecisionOutsideCoverage {
            row: 1,
            decision_at: 100,
            start: 0,
            end: 100,
        },
    );

    let label = vec![row(1, 0, Some(0))];
    assert_error(
        expanding_time(1, &label, TimeCoverage::new(0, 10, 0), spec),
        TimeSplitError::LabelBeforeDecision {
            row: 0,
            decision_at: 1,
            label_available_at: 0,
        },
    );
}

#[test]
fn arithmetic_gap_and_overflow_edges() {
    let rows: [TimeRow; 0] = [];
    let spec = TimeWindowSpec::new(10, 5, 1, 1000, 1);
    assert_error(
        expanding_time(0, &rows, TimeCoverage::new(0, 100, 0), spec),
        TimeSplitError::NoFullWindows {
            first_test_start: 1010,
            test_end: 1015,
            coverage_end: 100,
        },
    );

    let huge_step = TimeWindowSpec::new(10, 5, u64::MAX, 2, 1);
    let folds = expected_folds(
        expanding_time(0, &rows, TimeCoverage::new(0, 100, 0), huge_step)
            .expect("huge-step schedule"),
    );
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].cutoff, 12);
    assert_eq!(folds[0].test, 0..0);
    assert_eq!(
        folds[0].outcome,
        TimeFoldOutcome::Unusable(UnusableReason::EmptyTest)
    );

    let max_train = TimeWindowSpec::new(u64::MAX, 1, 1, 0, 1);
    assert_error(
        expanding_time(0, &rows, TimeCoverage::new(0, 100, 0), max_train),
        TimeSplitError::FirstTestStartOverflow,
    );

    let max_test = TimeWindowSpec::new(10, u64::MAX, 1, 2, 1);
    assert_error(
        expanding_time(0, &rows, TimeCoverage::new(0, 100, 0), max_test),
        TimeSplitError::FirstTestEndOverflow,
    );

    let start_overflow = TimeWindowSpec::new(10, 1, 1, 0, 1);
    let coverage = TimeCoverage::new(i64::MAX - 5, i64::MAX, i64::MAX - 5);
    assert_error(
        expanding_time(0, &rows, coverage, start_overflow),
        TimeSplitError::FirstTestStartOverflow,
    );

    let end_overflow = TimeWindowSpec::new(5, 10, 1, 0, 1);
    let coverage = TimeCoverage::new(i64::MAX - 10, i64::MAX, i64::MAX - 10);
    assert_error(
        expanding_time(0, &rows, coverage, end_overflow),
        TimeSplitError::FirstTestEndOverflow,
    );
}

#[test]
fn target_width_count_fixture_is_exact() {
    let rows: [TimeRow; 0] = [];
    let coverage = TimeCoverage::new(0, 4_294_967_297, 0);
    let spec = TimeWindowSpec::new(1, 1, 1, 0, 1);

    #[cfg(target_pointer_width = "64")]
    assert_error(
        expanding_time(0, &rows, coverage, spec),
        TimeSplitError::ScheduleCapExceeded {
            scheduled: 4_294_967_296,
            cap: 1,
        },
    );

    #[cfg(target_pointer_width = "32")]
    assert_error(
        expanding_time(0, &rows, coverage, spec),
        TimeSplitError::ScheduleCountOverflow,
    );
}

#[test]
fn cap_boundaries_and_source_contract() {
    let base = primary();
    assert!(base.expanding().is_ok());

    let mut capped = base.clone();
    capped.cap = 2;
    assert_error(
        capped.expanding(),
        TimeSplitError::ScheduleCapExceeded {
            scheduled: 3,
            cap: 2,
        },
    );

    let mut zero_cap = base;
    zero_cap.cap = 0;
    assert_error(zero_cap.expanding(), TimeSplitError::ZeroScheduleCap);

    let shape = DimensionMismatch {
        expected: 8,
        actual: 7,
    };
    let metadata = TimeSplitError::MetadataLengthMismatch { shape };
    assert!(std::error::Error::source(&metadata).is_some());
    assert!(std::error::Error::source(&TimeSplitError::ZeroTrainDuration).is_none());
}
