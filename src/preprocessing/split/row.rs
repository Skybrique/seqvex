//! Row-position walk-forward scheduling.
//!
//! A schedule selects ordered, half-open row ranges from caller-owned data. It
//! does not own observations, infer time or label availability, fit
//! preprocessing, or execute a model.

use std::fmt;
use std::iter::FusedIterator;
use std::ops::Range;

/// Policy for a final test window shorter than `test_rows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailPolicy {
    /// Omit a short final test window.
    Drop,
    /// Emit a short final test window once.
    KeepShort,
}

/// Configuration shared by expanding and rolling row schedules.
///
/// The specification is data-only. Validation that depends on the dataset row
/// count happens in [`expanding`] or [`rolling`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowWindowSpec {
    first_train: usize,
    train_rows: usize,
    test_rows: usize,
    step_rows: usize,
    row_gap: usize,
    tail: TailPolicy,
}

impl RowWindowSpec {
    /// Creates a row-window specification.
    ///
    /// `first_train` is the earliest selected training position;
    /// `train_rows` is the initial expanding width or fixed rolling width;
    /// `test_rows` is the normal test width; `step_rows` advances the test
    /// start; `row_gap` separates training from the test start.
    pub fn new(
        first_train: usize,
        train_rows: usize,
        test_rows: usize,
        step_rows: usize,
        row_gap: usize,
        tail: TailPolicy,
    ) -> Self {
        Self {
            first_train,
            train_rows,
            test_rows,
            step_rows,
            row_gap,
            tail,
        }
    }
}

/// Failure classes returned when a row schedule cannot be constructed.
///
/// The enum is non-exhaustive so later row-scheduling diagnostics can be added
/// without breaking callers. It carries no underlying source error.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSplitError {
    /// `train_rows` was zero.
    ZeroTrainRows,
    /// `test_rows` was zero.
    ZeroTestRows,
    /// `step_rows` was zero.
    ZeroStepRows,
    /// The anchor is beyond the exclusive row count.
    AnchorOutOfBounds {
        /// The supplied first training position.
        first_train: usize,
        /// The exclusive dataset row count.
        row_count: usize,
    },
    /// Forming the first test start overflowed `usize`.
    FirstTestStartOverflow,
    /// The configuration emits no fold.
    NoFolds,
}

impl fmt::Display for RowSplitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTrainRows => write!(formatter, "train_rows must be positive"),
            Self::ZeroTestRows => write!(formatter, "test_rows must be positive"),
            Self::ZeroStepRows => write!(formatter, "step_rows must be positive"),
            Self::AnchorOutOfBounds {
                first_train,
                row_count,
            } => write!(
                formatter,
                "first_train {first_train} exceeds row_count {row_count}"
            ),
            Self::FirstTestStartOverflow => {
                write!(formatter, "the first test start exceeds usize::MAX")
            }
            Self::NoFolds => write!(formatter, "configuration produces no row folds"),
        }
    }
}

impl std::error::Error for RowSplitError {}

/// One row-position fold.
///
/// The ranges are zero-based and half-open. The caller uses them to slice
/// aligned feature, target, and metadata views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowFold {
    train: Range<usize>,
    test: Range<usize>,
}

impl RowFold {
    /// Returns the half-open training range.
    pub fn train_range(&self) -> Range<usize> {
        self.train.clone()
    }

    /// Returns the half-open test range.
    pub fn test_range(&self) -> Range<usize> {
        self.test.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowMode {
    Expanding,
    Rolling,
}

/// Lazy row-fold iterator with constant-size state.
///
/// The iterator is permanently exhausted after the last fold.
#[derive(Debug)]
pub struct RowFolds {
    mode: RowMode,
    row_count: usize,
    first_train: usize,
    train_rows: usize,
    test_rows: usize,
    step_rows: usize,
    row_gap: usize,
    tail: TailPolicy,
    next_start: usize,
    done: bool,
}

impl Iterator for RowFolds {
    type Item = RowFold;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let start = self.next_start;
        debug_assert!(start < self.row_count);
        let train_end = start - self.row_gap;
        let train = match self.mode {
            RowMode::Expanding => self.first_train..train_end,
            RowMode::Rolling => (train_end - self.train_rows)..train_end,
        };

        if self.test_rows <= self.row_count - start {
            let test_end = start + self.test_rows;
            let test = start..test_end;
            if test_end == self.row_count || self.step_rows >= self.row_count - start {
                self.done = true;
            } else {
                self.next_start = start + self.step_rows;
            }
            Some(RowFold { train, test })
        } else {
            self.done = true;
            if self.tail == TailPolicy::KeepShort {
                Some(RowFold {
                    train,
                    test: start..self.row_count,
                })
            } else {
                None
            }
        }
    }
}

impl FusedIterator for RowFolds {}

impl RowFolds {
    fn new(mode: RowMode, row_count: usize, spec: RowWindowSpec) -> Result<Self, RowSplitError> {
        if spec.train_rows == 0 {
            return Err(RowSplitError::ZeroTrainRows);
        }
        if spec.test_rows == 0 {
            return Err(RowSplitError::ZeroTestRows);
        }
        if spec.step_rows == 0 {
            return Err(RowSplitError::ZeroStepRows);
        }
        if spec.first_train > row_count {
            return Err(RowSplitError::AnchorOutOfBounds {
                first_train: spec.first_train,
                row_count,
            });
        }

        let first_train_end = spec
            .first_train
            .checked_add(spec.train_rows)
            .ok_or(RowSplitError::FirstTestStartOverflow)?;
        let first_test_start = first_train_end
            .checked_add(spec.row_gap)
            .ok_or(RowSplitError::FirstTestStartOverflow)?;

        if first_test_start >= row_count {
            return Err(RowSplitError::NoFolds);
        }
        if spec.test_rows > row_count - first_test_start && spec.tail == TailPolicy::Drop {
            return Err(RowSplitError::NoFolds);
        }

        Ok(Self {
            mode,
            row_count,
            first_train: spec.first_train,
            train_rows: spec.train_rows,
            test_rows: spec.test_rows,
            step_rows: spec.step_rows,
            row_gap: spec.row_gap,
            tail: spec.tail,
            next_start: first_test_start,
            done: false,
        })
    }
}

/// Creates an expanding walk-forward row schedule.
///
/// Training starts at `spec.first_train` and grows. The test schedule is the
/// same as [`rolling`] for matching inputs.
pub fn expanding(row_count: usize, spec: RowWindowSpec) -> Result<RowFolds, RowSplitError> {
    RowFolds::new(RowMode::Expanding, row_count, spec)
}

/// Creates a fixed-width rolling walk-forward row schedule.
///
/// Every training range has width `spec.train_rows`; the first fold already has
/// the full width. The test schedule is the same as [`expanding`] for matching
/// inputs.
pub fn rolling(row_count: usize, spec: RowWindowSpec) -> Result<RowFolds, RowSplitError> {
    RowFolds::new(RowMode::Rolling, row_count, spec)
}
