//! Elapsed-time expanding and rolling walk-forward splits.
//!
//! Stage 2 selects caller-owned rows on a normalized signed tick axis. It
//! validates snapshot metadata, returns stable scheduled IDs, ordered coalesced
//! training ranges, one contiguous test range, and explicit unusable outcomes.
//! It does not fit preprocessing, score labels, purge outcomes, or embargo.

use std::error::Error;
use std::fmt;
use std::iter::FusedIterator;
use std::ops::Range;

use crate::foundation::error::DimensionMismatch;

/// One aligned row's supplied point-in-time metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRow {
    /// Caller-normalized decision tick.
    pub decision_at: i64,
    /// Tick at which the exact feature version used by this row was available.
    pub features_available_at: i64,
    /// Tick at which the realized training label was available; `None` if unknown.
    pub label_available_at: Option<i64>,
}

/// Half-open decision coverage plus the earliest training anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeCoverage {
    start: i64,
    end: i64,
    anchor: i64,
}

impl TimeCoverage {
    /// Stores coverage and anchor data; validation happens when a schedule is built.
    pub fn new(start: i64, end: i64, anchor: i64) -> Self {
        Self { start, end, anchor }
    }
}

/// Elapsed-time window configuration. Data-only; the schedule validates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeWindowSpec {
    train_duration: u64,
    test_duration: u64,
    step: u64,
    gap: u64,
    max_scheduled_folds: usize,
}

impl TimeWindowSpec {
    /// Stores the window configuration; validation happens when a schedule is built.
    pub fn new(
        train_duration: u64,
        test_duration: u64,
        step: u64,
        gap: u64,
        max_scheduled_folds: usize,
    ) -> Self {
        Self {
            train_duration,
            test_duration,
            step,
            gap,
            max_scheduled_folds,
        }
    }
}

/// Construction failure for a time schedule.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSplitError {
    ZeroTrainDuration,
    ZeroTestDuration,
    ZeroStepDuration,
    ZeroScheduleCap,
    CoverageOrder {
        start: i64,
        end: i64,
    },
    AnchorOutsideCoverage {
        anchor: i64,
        start: i64,
        end: i64,
    },
    MetadataLengthMismatch {
        shape: DimensionMismatch,
    },
    DecisionOutOfOrder {
        row: usize,
        previous: i64,
        actual: i64,
    },
    DecisionOutsideCoverage {
        row: usize,
        decision_at: i64,
        start: i64,
        end: i64,
    },
    LabelBeforeDecision {
        row: usize,
        decision_at: i64,
        label_available_at: i64,
    },
    FirstTestStartOverflow,
    FirstTestEndOverflow,
    ScheduleCountOverflow,
    NoFullWindows {
        first_test_start: i64,
        test_end: i64,
        coverage_end: i64,
    },
    ScheduleCapExceeded {
        scheduled: usize,
        cap: usize,
    },
}

impl fmt::Display for TimeSplitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTrainDuration => {
                formatter.write_str("time split train duration must be positive")
            }
            Self::ZeroTestDuration => {
                formatter.write_str("time split test duration must be positive")
            }
            Self::ZeroStepDuration => formatter.write_str("time split step must be positive"),
            Self::ZeroScheduleCap => {
                formatter.write_str("time split schedule cap must be positive")
            }
            Self::CoverageOrder { start, end } => write!(
                formatter,
                "time split coverage start must be less than end (start {start}, end {end})"
            ),
            Self::AnchorOutsideCoverage { anchor, start, end } => write!(
                formatter,
                "time split anchor {anchor} is outside coverage [{start}, {end})"
            ),
            Self::MetadataLengthMismatch { shape } => {
                write!(formatter, "time split metadata length mismatch: {shape}")
            }
            Self::DecisionOutOfOrder {
                row,
                previous,
                actual,
            } => write!(
                formatter,
                "time split decisions must be nondecreasing: row {row} has {actual} after {previous}"
            ),
            Self::DecisionOutsideCoverage {
                row,
                decision_at,
                start,
                end,
            } => write!(
                formatter,
                "time split row {row} decision {decision_at} is outside coverage [{start}, {end})"
            ),
            Self::LabelBeforeDecision {
                row,
                decision_at,
                label_available_at,
            } => write!(
                formatter,
                "time split row {row} label available at {label_available_at} before decision {decision_at}"
            ),
            Self::FirstTestStartOverflow => {
                formatter.write_str("time split first test start overflows i64")
            }
            Self::FirstTestEndOverflow => {
                formatter.write_str("time split first test end overflows i64")
            }
            Self::ScheduleCountOverflow => {
                formatter.write_str("time split schedule count overflows usize")
            }
            Self::NoFullWindows {
                first_test_start,
                test_end,
                coverage_end,
            } => write!(
                formatter,
                "time split has no full windows: first test [{first_test_start}, {test_end}) exceeds coverage end {coverage_end}"
            ),
            Self::ScheduleCapExceeded { scheduled, cap } => write!(
                formatter,
                "time split schedule has {scheduled} windows, exceeding cap {cap}"
            ),
        }
    }
}

impl Error for TimeSplitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MetadataLengthMismatch { shape } => Some(shape),
            _ => None,
        }
    }
}

/// The usability of one scheduled window.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeFoldOutcome {
    Usable,
    Unusable(UnusableReason),
}

/// Why a scheduled window is unusable.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnusableReason {
    EmptyTest,
    UnavailableTestFeature {
        row: usize,
        decision_at: i64,
        features_available_at: i64,
    },
    EmptyEligibleTraining,
}

/// Disjoint counts for one scheduled fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExclusionCounts {
    pub train_candidates: usize,
    pub train_feature_unavailable: usize,
    pub train_label_unavailable: usize,
    pub train_eligible: usize,
    pub test_candidates: usize,
    pub test_feature_unavailable: usize,
    pub test_eligible: usize,
}

/// One scheduled half-open time window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeFold {
    scheduled_id: usize,
    cutoff: i64,
    test: Range<usize>,
    train: Vec<Range<usize>>,
    excluded: ExclusionCounts,
    outcome: TimeFoldOutcome,
}

impl TimeFold {
    /// Zero-based nominal schedule ID, unchanged by usability.
    pub fn scheduled_id(&self) -> usize {
        self.scheduled_id
    }

    /// Fit cutoff and test start tick.
    pub fn cutoff(&self) -> i64 {
        self.cutoff
    }

    /// Contiguous half-open test candidate range.
    pub fn test_range(&self) -> Range<usize> {
        self.test.clone()
    }

    /// Ordered, disjoint, coalesced eligible training ranges.
    pub fn train_ranges(&self) -> &[Range<usize>] {
        &self.train
    }

    /// Disjoint exclusion counts for this fold.
    pub fn exclusion_counts(&self) -> ExclusionCounts {
        self.excluded
    }

    /// Usability and, when unusable, the primary reason.
    pub fn outcome(&self) -> TimeFoldOutcome {
        self.outcome
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeMode {
    Expanding,
    Rolling,
}

/// Lazy, fused schedule produced by [`expanding_time`] or [`rolling_time`].
#[derive(Debug)]
pub struct TimeFolds<'a> {
    rows: &'a [TimeRow],
    spec: TimeWindowSpec,
    anchor: i64,
    p0: i128,
    mode: TimeMode,
    next_index: usize,
    remaining: usize,
    total: usize,
}

impl TimeFolds<'_> {
    /// Total nominal count fixed at construction.
    pub fn scheduled_fold_count(&self) -> usize {
        self.total
    }
}

impl Iterator for TimeFolds<'_> {
    type Item = TimeFold;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let index = self.next_index;
        let p_i128 = self.p0 + index as i128 * i128::from(self.spec.step);
        let p = i64::try_from(p_i128).expect("validated time schedule start");
        let q_i128 = p_i128 - i128::from(self.spec.gap);
        let q = i64::try_from(q_i128).expect("validated time schedule train end");
        let train_start = match self.mode {
            TimeMode::Expanding => self.anchor,
            TimeMode::Rolling => {
                let start_i128 = q_i128 - i128::from(self.spec.train_duration);
                i64::try_from(start_i128).expect("validated rolling training start")
            }
        };
        let test_end = i64::try_from(p_i128 + i128::from(self.spec.test_duration))
            .expect("validated test end");

        let train_lo = lower_bound(self.rows, train_start);
        let train_hi = lower_bound(self.rows, q);
        let test_lo = lower_bound(self.rows, p);
        let test_hi = lower_bound(self.rows, test_end);

        let mut counts = ExclusionCounts {
            train_candidates: train_hi - train_lo,
            ..ExclusionCounts::default()
        };
        let mut train_ranges: Vec<Range<usize>> = Vec::new();
        for (offset, row) in self.rows[train_lo..train_hi].iter().enumerate() {
            let row_index = train_lo + offset;
            if row.features_available_at >= row.decision_at {
                counts.train_feature_unavailable += 1;
                continue;
            }
            match row.label_available_at {
                Some(label) if label < p => {
                    counts.train_eligible += 1;
                    let contiguous = train_ranges
                        .last()
                        .is_some_and(|last| last.end == row_index);
                    if contiguous {
                        if let Some(last) = train_ranges.last_mut() {
                            last.end += 1;
                        }
                    } else {
                        train_ranges.push(row_index..row_index + 1);
                    }
                }
                _ => counts.train_label_unavailable += 1,
            }
        }

        counts.test_candidates = test_hi - test_lo;
        let mut first_test_failure = None;
        for (offset, row) in self.rows[test_lo..test_hi].iter().enumerate() {
            let row_index = test_lo + offset;
            if row.features_available_at >= row.decision_at {
                counts.test_feature_unavailable += 1;
                if first_test_failure.is_none() {
                    first_test_failure = Some(UnusableReason::UnavailableTestFeature {
                        row: row_index,
                        decision_at: row.decision_at,
                        features_available_at: row.features_available_at,
                    });
                }
            }
        }
        counts.test_eligible = counts.test_candidates - counts.test_feature_unavailable;

        let test = test_lo..test_hi;
        let outcome = if test.is_empty() {
            TimeFoldOutcome::Unusable(UnusableReason::EmptyTest)
        } else if let Some(reason) = first_test_failure {
            TimeFoldOutcome::Unusable(reason)
        } else if train_ranges.is_empty() {
            TimeFoldOutcome::Unusable(UnusableReason::EmptyEligibleTraining)
        } else {
            TimeFoldOutcome::Usable
        };

        self.next_index += 1;
        self.remaining -= 1;

        Some(TimeFold {
            scheduled_id: index,
            cutoff: p,
            test,
            train: train_ranges,
            excluded: counts,
            outcome,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl FusedIterator for TimeFolds<'_> {}

fn lower_bound(rows: &[TimeRow], tick: i64) -> usize {
    rows.partition_point(|row| row.decision_at < tick)
}

fn count_to_usize(count: i128) -> Result<usize, TimeSplitError> {
    usize::try_from(count).map_err(|_| TimeSplitError::ScheduleCountOverflow)
}

fn build<'a>(
    row_count: usize,
    rows: &'a [TimeRow],
    coverage: TimeCoverage,
    spec: TimeWindowSpec,
    mode: TimeMode,
) -> Result<TimeFolds<'a>, TimeSplitError> {
    if spec.train_duration == 0 {
        return Err(TimeSplitError::ZeroTrainDuration);
    }
    if spec.test_duration == 0 {
        return Err(TimeSplitError::ZeroTestDuration);
    }
    if spec.step == 0 {
        return Err(TimeSplitError::ZeroStepDuration);
    }
    if spec.max_scheduled_folds == 0 {
        return Err(TimeSplitError::ZeroScheduleCap);
    }
    if coverage.start >= coverage.end {
        return Err(TimeSplitError::CoverageOrder {
            start: coverage.start,
            end: coverage.end,
        });
    }
    if coverage.anchor < coverage.start || coverage.anchor >= coverage.end {
        return Err(TimeSplitError::AnchorOutsideCoverage {
            anchor: coverage.anchor,
            start: coverage.start,
            end: coverage.end,
        });
    }
    if rows.len() != row_count {
        return Err(TimeSplitError::MetadataLengthMismatch {
            shape: DimensionMismatch {
                expected: row_count,
                actual: rows.len(),
            },
        });
    }

    for index in 1..rows.len() {
        let previous = rows[index - 1].decision_at;
        let actual = rows[index].decision_at;
        if actual < previous {
            return Err(TimeSplitError::DecisionOutOfOrder {
                row: index,
                previous,
                actual,
            });
        }
    }
    for (row, entry) in rows.iter().enumerate() {
        if entry.decision_at < coverage.start || entry.decision_at >= coverage.end {
            return Err(TimeSplitError::DecisionOutsideCoverage {
                row,
                decision_at: entry.decision_at,
                start: coverage.start,
                end: coverage.end,
            });
        }
    }
    for (row, entry) in rows.iter().enumerate() {
        if let Some(label_available_at) = entry.label_available_at {
            if label_available_at < entry.decision_at {
                return Err(TimeSplitError::LabelBeforeDecision {
                    row,
                    decision_at: entry.decision_at,
                    label_available_at,
                });
            }
        }
    }

    let p0_i128 =
        i128::from(coverage.anchor) + i128::from(spec.train_duration) + i128::from(spec.gap);
    let p0 = i64::try_from(p0_i128).map_err(|_| TimeSplitError::FirstTestStartOverflow)?;
    let test_end = i64::try_from(p0_i128 + i128::from(spec.test_duration))
        .map_err(|_| TimeSplitError::FirstTestEndOverflow)?;
    if test_end > coverage.end {
        return Err(TimeSplitError::NoFullWindows {
            first_test_start: p0,
            test_end,
            coverage_end: coverage.end,
        });
    }

    let remaining_span = i128::from(coverage.end) - i128::from(test_end);
    let count = count_to_usize(1 + remaining_span / i128::from(spec.step))?;
    if count > spec.max_scheduled_folds {
        return Err(TimeSplitError::ScheduleCapExceeded {
            scheduled: count,
            cap: spec.max_scheduled_folds,
        });
    }

    Ok(TimeFolds {
        rows,
        spec,
        anchor: coverage.anchor,
        p0: p0_i128,
        mode,
        next_index: 0,
        remaining: count,
        total: count,
    })
}

/// Builds an expanding walk-forward schedule.
pub fn expanding_time<'a>(
    row_count: usize,
    rows: &'a [TimeRow],
    coverage: TimeCoverage,
    spec: TimeWindowSpec,
) -> Result<TimeFolds<'a>, TimeSplitError> {
    build(row_count, rows, coverage, spec, TimeMode::Expanding)
}

/// Builds a fixed-width rolling walk-forward schedule.
pub fn rolling_time<'a>(
    row_count: usize,
    rows: &'a [TimeRow],
    coverage: TimeCoverage,
    spec: TimeWindowSpec,
) -> Result<TimeFolds<'a>, TimeSplitError> {
    build(row_count, rows, coverage, spec, TimeMode::Rolling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_conversion_rejects_unrepresentable_count() {
        assert_eq!(count_to_usize(3), Ok(3));
        assert_eq!(
            count_to_usize(usize::MAX as i128 + 1),
            Err(TimeSplitError::ScheduleCountOverflow)
        );
    }
}
