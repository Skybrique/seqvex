//! External Stage 1 contract tests for row-position walk-forward splits.

use std::iter::FusedIterator;
use std::ops::Range;

use seqvex::preprocessing::split::{
    RowFold, RowFolds, RowSplitError, RowWindowSpec, TailPolicy, expanding, rolling,
};

type FoldRanges = Vec<(Range<usize>, Range<usize>)>;
type SplitResult = Result<FoldRanges, RowSplitError>;

fn spec(
    first_train: usize,
    train_rows: usize,
    test_rows: usize,
    step_rows: usize,
    row_gap: usize,
    tail: TailPolicy,
) -> RowWindowSpec {
    RowWindowSpec::new(first_train, train_rows, test_rows, step_rows, row_gap, tail)
}

fn collect(result: Result<RowFolds, RowSplitError>) -> SplitResult {
    result.map(|folds| {
        folds
            .map(|fold| (fold.train_range(), fold.test_range()))
            .collect()
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Expanding,
    Rolling,
}

fn run(mode: Mode, row_count: usize, window: RowWindowSpec) -> SplitResult {
    match mode {
        Mode::Expanding => collect(expanding(row_count, window)),
        Mode::Rolling => collect(rolling(row_count, window)),
    }
}

/// Independent oracle: scan every candidate start and accept the stride class.
///
/// It deliberately does not use the production iterator, a fold-count formula,
/// or production endpoint helpers. It stops at an exact-end full fold or at the
/// first short candidate.
struct OracleCase {
    mode: Mode,
    row_count: usize,
    first_train: usize,
    train_rows: usize,
    test_rows: usize,
    step_rows: usize,
    row_gap: usize,
    tail: TailPolicy,
}

fn oracle(case: OracleCase) -> SplitResult {
    let OracleCase {
        mode,
        row_count,
        first_train,
        train_rows,
        test_rows,
        step_rows,
        row_gap,
        tail,
    } = case;

    if train_rows == 0 {
        return Err(RowSplitError::ZeroTrainRows);
    }
    if test_rows == 0 {
        return Err(RowSplitError::ZeroTestRows);
    }
    if step_rows == 0 {
        return Err(RowSplitError::ZeroStepRows);
    }
    if first_train > row_count {
        return Err(RowSplitError::AnchorOutOfBounds {
            first_train,
            row_count,
        });
    }

    let first_start = first_train as u128 + train_rows as u128 + row_gap as u128;
    if first_start > usize::MAX as u128 {
        return Err(RowSplitError::FirstTestStartOverflow);
    }
    let first_start = first_start as usize;
    if first_start >= row_count {
        return Err(RowSplitError::NoFolds);
    }

    let mut folds = Vec::new();
    for p in 0..row_count {
        if p < first_start || (p - first_start) % step_rows != 0 {
            continue;
        }

        let train_end = p - row_gap;
        let train = match mode {
            Mode::Expanding => first_train..train_end,
            Mode::Rolling => (train_end - train_rows)..train_end,
        };

        if test_rows <= row_count - p {
            let test = p..(p + test_rows);
            folds.push((train, test));
            if p + test_rows == row_count {
                break;
            }
        } else {
            if tail == TailPolicy::KeepShort {
                folds.push((train, p..row_count));
            }
            break;
        }
    }

    if folds.is_empty() {
        Err(RowSplitError::NoFolds)
    } else {
        Ok(folds)
    }
}

fn assert_fused<I: FusedIterator<Item = RowFold>>(_: &I) {}

#[test]
fn public_fused_bound_and_exhaustion() {
    let mut folds = expanding(10, spec(0, 3, 4, 1, 0, TailPolicy::KeepShort)).unwrap();
    assert_fused(&folds);

    for _ in 0..4 {
        assert!(folds.next().is_some());
    }
    assert!(folds.next().is_none());
    assert!(folds.next().is_none());
}

#[test]
fn pinned_geometry_for_both_methods() {
    let keep = spec(2, 4, 3, 4, 1, TailPolicy::KeepShort);
    assert_eq!(
        collect(expanding(17, keep)).unwrap(),
        vec![(2..6, 7..10), (2..10, 11..14), (2..14, 15..17),]
    );
    assert_eq!(
        collect(rolling(17, keep)).unwrap(),
        vec![(2..6, 7..10), (6..10, 11..14), (10..14, 15..17),]
    );

    let drop = spec(2, 4, 3, 4, 1, TailPolicy::Drop);
    assert_eq!(
        collect(expanding(17, drop)).unwrap(),
        vec![(2..6, 7..10), (2..10, 11..14)]
    );
    assert_eq!(
        collect(rolling(17, drop)).unwrap(),
        vec![(2..6, 7..10), (6..10, 11..14)]
    );

    let exact = spec(0, 3, 4, 1, 0, TailPolicy::KeepShort);
    assert_eq!(
        collect(expanding(10, exact)).unwrap(),
        vec![(0..3, 3..7), (0..4, 4..8), (0..5, 5..9), (0..6, 6..10),]
    );
    assert_eq!(
        collect(rolling(10, exact)).unwrap(),
        vec![(0..3, 3..7), (1..4, 4..8), (2..5, 5..9), (3..6, 6..10),]
    );

    let stride = spec(0, 3, 2, 10, 0, TailPolicy::KeepShort);
    assert_eq!(collect(expanding(12, stride)).unwrap(), vec![(0..3, 3..5)]);
    assert_eq!(collect(rolling(12, stride)).unwrap(), vec![(0..3, 3..5)]);
}

#[test]
fn empty_input_and_error_precedence() {
    let drop = TailPolicy::Drop;

    assert_eq!(
        expanding(10, spec(0, 0, 0, 0, 0, drop)).unwrap_err(),
        RowSplitError::ZeroTrainRows
    );
    assert_eq!(
        expanding(10, spec(0, 1, 0, 0, 0, drop)).unwrap_err(),
        RowSplitError::ZeroTestRows
    );
    assert_eq!(
        expanding(10, spec(0, 1, 1, 0, 0, drop)).unwrap_err(),
        RowSplitError::ZeroStepRows
    );
    assert_eq!(
        expanding(0, spec(usize::MAX, 0, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::ZeroTrainRows
    );
    assert_eq!(
        expanding(0, spec(usize::MAX, 1, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::AnchorOutOfBounds {
            first_train: usize::MAX,
            row_count: 0,
        }
    );

    // Empty input: overflow forming A + W + G wins over NoFolds.
    assert_eq!(
        expanding(0, spec(0, 1, 1, 1, usize::MAX, drop)).unwrap_err(),
        RowSplitError::FirstTestStartOverflow
    );
    // Empty input with representable arithmetic.
    assert_eq!(
        expanding(0, spec(0, 1, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::NoFolds
    );
    // A == N is a permitted anchor with no room for a fold.
    assert_eq!(
        expanding(10, spec(10, 1, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::NoFolds
    );
    // A + W overflow.
    assert_eq!(
        expanding(usize::MAX, spec(usize::MAX, 1, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::FirstTestStartOverflow
    );
    // Adding the gap overflows after A + W.
    assert_eq!(
        expanding(usize::MAX, spec(0, usize::MAX, 1, 1, 1, drop)).unwrap_err(),
        RowSplitError::FirstTestStartOverflow
    );
    // Overflow wins over NoFolds even when the first start would exceed N.
    assert_eq!(
        expanding(10, spec(10, usize::MAX, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::FirstTestStartOverflow
    );
    // A representable but huge width needs no allocation and yields NoFolds.
    assert_eq!(
        expanding(10, spec(0, usize::MAX, 1, 1, 0, drop)).unwrap_err(),
        RowSplitError::NoFolds
    );

    // Short first test: Drop is NoFolds; KeepShort emits once.
    assert_eq!(
        expanding(5, spec(0, 3, 4, 1, 0, drop)).unwrap_err(),
        RowSplitError::NoFolds
    );
    assert_eq!(
        collect(expanding(5, spec(0, 3, 4, 1, 0, TailPolicy::KeepShort))).unwrap(),
        vec![(0..3, 3..5)]
    );
}

#[test]
fn huge_step_and_huge_test_width_stop_safely() {
    let huge_step = spec(0, 3, 2, usize::MAX, 0, TailPolicy::KeepShort);
    assert_eq!(
        collect(expanding(10, huge_step)).unwrap(),
        vec![(0..3, 3..5)]
    );
    assert_eq!(collect(rolling(10, huge_step)).unwrap(), vec![(0..3, 3..5)]);

    let huge_test = spec(0, 1, usize::MAX, 1, 0, TailPolicy::KeepShort);
    assert_eq!(
        collect(expanding(10, huge_test)).unwrap(),
        vec![(0..1, 1..10)]
    );
    let huge_test_drop = spec(0, 1, usize::MAX, 1, 0, TailPolicy::Drop);
    assert_eq!(
        expanding(10, huge_test_drop).unwrap_err(),
        RowSplitError::NoFolds
    );
}

#[test]
fn error_display_and_source() {
    let error = RowSplitError::AnchorOutOfBounds {
        first_train: 11,
        row_count: 10,
    };
    let text = error.to_string();
    assert!(text.contains("11"));
    assert!(text.contains("10"));
    assert!(std::error::Error::source(&error).is_none());
}

#[test]
fn small_domain_matches_stride_oracle() {
    for row_count in 0..=16 {
        for first_train in 0..=(row_count + 1) {
            for train_rows in 1..=6 {
                for test_rows in 1..=6 {
                    for step_rows in 1..=6 {
                        for row_gap in 0..=3 {
                            for tail in [TailPolicy::Drop, TailPolicy::KeepShort] {
                                for mode in [Mode::Expanding, Mode::Rolling] {
                                    let window = spec(
                                        first_train,
                                        train_rows,
                                        test_rows,
                                        step_rows,
                                        row_gap,
                                        tail,
                                    );
                                    let actual = run(mode, row_count, window);
                                    let expected = oracle(OracleCase {
                                        mode,
                                        row_count,
                                        first_train,
                                        train_rows,
                                        test_rows,
                                        step_rows,
                                        row_gap,
                                        tail,
                                    });
                                    assert_eq!(
                                        actual, expected,
                                        "mode={mode:?} N={row_count} A={first_train} W={train_rows} H={test_rows} S={step_rows} G={row_gap} tail={tail:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
