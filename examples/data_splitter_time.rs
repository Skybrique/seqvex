//! Runnable Stage 2 time-split consumer using the existing public OLS path.
//!
//! The eight-row fixture is the DS-TD-0.5 cutoff fixture. Preprocessing is
//! centered on the eligible training rows only and frozen before prediction.
//! The example checks both time splitters, the pinned memberships, analytical
//! predictions, row-5 scoring exclusion, and future-information mutation parity.

#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;
use std::process::ExitCode;

use seqvex::foundation::error::DimensionMismatch;
use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::{FitConfig, fit};
use seqvex::preprocessing::split::{
    ExclusionCounts, TimeCoverage, TimeFold, TimeFoldOutcome, TimeFolds, TimeRow, TimeSplitError,
    TimeWindowSpec, UnusableReason, expanding_time, rolling_time,
};

const TOLERANCE: f32 = 1e-5;
const SCORING_CUTOFF: i64 = 30;

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
const X: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
const Y: [f32; 8] = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];

fn fixture_rows() -> Vec<TimeRow> {
    (0..8)
        .map(|index| TimeRow {
            decision_at: DECISIONS[index],
            features_available_at: FEATURES_AVAILABLE[index],
            label_available_at: LABELS[index],
        })
        .collect()
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

fn check_eq<T: std::fmt::Debug + PartialEq>(
    label: &str,
    actual: &T,
    expected: &T,
) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{label}: expected {expected:?}, got {actual:?}"))
    }
}

fn check_close(label: &str, actual: f32, expected: f32) -> Result<(), String> {
    let difference = (actual - expected).abs();
    if difference <= TOLERANCE {
        Ok(())
    } else {
        Err(format!(
            "{label}: expected {expected}, got {actual} (difference {difference}, tolerance {TOLERANCE})"
        ))
    }
}

fn check_bits(label: &str, actual: f32, expected: f32) -> Result<(), String> {
    if actual.to_bits() == expected.to_bits() {
        Ok(())
    } else {
        Err(format!(
            "{label}: expected bits {:08x}, got {:08x}",
            expected.to_bits(),
            actual.to_bits()
        ))
    }
}

fn schedule(rolling: bool, rows: &[TimeRow]) -> Result<Vec<TimeFold>, String> {
    let coverage = TimeCoverage::new(0, 30, 0);
    let spec = TimeWindowSpec::new(10, 5, 6, 2, 3);
    let result: Result<TimeFolds<'_>, TimeSplitError> = if rolling {
        rolling_time(rows.len(), rows, coverage, spec)
    } else {
        expanding_time(rows.len(), rows, coverage, spec)
    };
    result
        .map(|folds| folds.collect())
        .map_err(|error| format!("time schedule failed: {error}"))
}

fn assert_fold(
    fold: &TimeFold,
    id: usize,
    cutoff: i64,
    test: Range<usize>,
    train: &[Range<usize>],
    counts: ExclusionCounts,
    outcome: TimeFoldOutcome,
) -> Result<(), String> {
    check_eq("fold id", &fold.scheduled_id(), &id)?;
    check_eq("fold cutoff", &fold.cutoff(), &cutoff)?;
    check_eq("fold test range", &fold.test_range(), &test)?;
    check_eq(
        "fold train ranges",
        &fold.train_ranges().to_vec(),
        &train.to_vec(),
    )?;
    check_eq("fold counts", &fold.exclusion_counts(), &counts)?;
    check_eq("fold outcome", &fold.outcome(), &outcome)
}

fn assert_primary_folds(folds: &[TimeFold], rolling: bool) -> Result<(), String> {
    if folds.len() != 3 {
        return Err(format!("expected 3 scheduled folds, got {}", folds.len()));
    }
    assert_fold(
        &folds[0],
        0,
        12,
        4..6,
        &[0..1, 2..3],
        counts(4, 1, 1, 2, 2, 0, 2),
        TimeFoldOutcome::Usable,
    )?;
    if rolling {
        assert_fold(
            &folds[1],
            1,
            18,
            7..7,
            &[4..5],
            counts(3, 1, 1, 1, 0, 0, 0),
            TimeFoldOutcome::Unusable(UnusableReason::EmptyTest),
        )?;
        assert_fold(
            &folds[2],
            2,
            24,
            7..8,
            &[4..5, 6..7],
            counts(3, 0, 1, 2, 1, 0, 1),
            TimeFoldOutcome::Usable,
        )
    } else {
        assert_fold(
            &folds[1],
            1,
            18,
            7..7,
            &[0..3, 4..5],
            counts(6, 1, 1, 4, 0, 0, 0),
            TimeFoldOutcome::Unusable(UnusableReason::EmptyTest),
        )?;
        assert_fold(
            &folds[2],
            2,
            24,
            7..8,
            &[0..3, 4..5, 6..7],
            counts(7, 1, 1, 5, 1, 0, 1),
            TimeFoldOutcome::Usable,
        )
    }
}

fn fit_predict(folds: &[TimeFold], x: &[f32], y: &[f32]) -> Result<(f32, [f32; 2]), String> {
    let fold = &folds[0];
    let mut train_indices = Vec::new();
    for range in fold.train_ranges() {
        train_indices.extend(range.clone());
    }
    if train_indices != vec![0, 2] {
        return Err(format!(
            "cutoff-12 eligible training rows: expected [0, 2], got {train_indices:?}"
        ));
    }
    let mean =
        train_indices.iter().map(|&index| x[index]).sum::<f32>() / train_indices.len() as f32;
    let transformed: Vec<Vector> = x
        .iter()
        .map(|&value| Vector::from_slice(&[value - mean]))
        .collect();
    let result = fit(&transformed, y, &train_indices, &FitConfig::default())
        .map_err(|error| format!("OLS fit failed: {error:?}"))?;
    let model = result.into_predictor();
    let row_4 = model
        .predict_checked(&transformed[4])
        .map_err(|error| format!("row 4 prediction failed: {error:?}"))?;
    let row_5 = model
        .predict_checked(&transformed[5])
        .map_err(|error| format!("row 5 prediction failed: {error:?}"))?;
    Ok((mean, [row_4, row_5]))
}

struct MethodRun {
    folds: Vec<TimeFold>,
    mean: f32,
    predictions: [f32; 2],
}

fn run_case(rolling: bool, rows: &[TimeRow], x: &[f32], y: &[f32]) -> Result<MethodRun, String> {
    let folds = schedule(rolling, rows)?;
    assert_primary_folds(&folds, rolling)?;
    let (mean, predictions) = fit_predict(&folds, x, y)?;
    Ok(MethodRun {
        folds,
        mean,
        predictions,
    })
}

fn run() -> Result<(), String> {
    let rows = fixture_rows();
    if rows.len() != X.len() {
        return Err(format!(
            "fixture row/feature mismatch: {}",
            DimensionMismatch {
                expected: rows.len(),
                actual: X.len(),
            }
        ));
    }
    if X.len() != Y.len() {
        return Err(format!(
            "fixture feature/target mismatch: {}",
            DimensionMismatch {
                expected: X.len(),
                actual: Y.len(),
            }
        ));
    }

    let expanding = run_case(false, &rows, &X, &Y)?;
    let rolling = run_case(true, &rows, &X, &Y)?;

    for (name, run) in [("expanding", &expanding), ("rolling", &rolling)] {
        check_close(&format!("{name} frozen mean"), run.mean, 1.0)?;
        check_close(&format!("{name} row 4 prediction"), run.predictions[0], 9.0)?;
        check_close(
            &format!("{name} row 5 prediction"),
            run.predictions[1],
            11.0,
        )?;
        check_close(
            &format!("{name} row 4 residual"),
            run.predictions[0] - Y[4],
            0.0,
        )?;
    }
    check_bits("expanding vs rolling mean", expanding.mean, rolling.mean)?;
    check_bits(
        "expanding vs rolling row 4",
        expanding.predictions[0],
        rolling.predictions[0],
    )?;
    check_bits(
        "expanding vs rolling row 5",
        expanding.predictions[1],
        rolling.predictions[1],
    )?;

    if rows[5].features_available_at >= rows[5].decision_at {
        return Err("row 5 should be eligible for prediction".into());
    }
    if rows[5].label_available_at.is_some() {
        return Err("row 5 should have an unknown label".into());
    }
    if rows[4]
        .label_available_at
        .is_none_or(|label| label >= SCORING_CUTOFF)
    {
        return Err("row 4 should be scoreable at strict cutoff 30".into());
    }

    let mut mutated_rows = rows.clone();
    mutated_rows[4].label_available_at = Some(17);
    mutated_rows[5].label_available_at = Some(31);
    let mut mutated_x = X;
    mutated_x[6] = 1_000.0;
    mutated_x[7] = 1_000.0;
    let mut mutated_y = Y;
    mutated_y[1] = 1_000.0;
    mutated_y[3] = -1_000.0;
    mutated_y[5] = 1_000.0;
    mutated_y[6] = 1_000.0;
    mutated_y[7] = 1_000.0;

    let mutated_expanding = run_case(false, &mutated_rows, &mutated_x, &mutated_y)?;
    let mutated_rolling = run_case(true, &mutated_rows, &mutated_x, &mutated_y)?;

    let eligible_training = [0usize, 2usize];
    let changed_features: Vec<usize> = (0..X.len())
        .filter(|&index| mutated_x[index].to_bits() != X[index].to_bits())
        .collect();
    let changed_targets: Vec<usize> = (0..Y.len())
        .filter(|&index| mutated_y[index].to_bits() != Y[index].to_bits())
        .collect();
    check_eq("changed feature indices", &changed_features, &vec![6, 7])?;
    check_eq(
        "changed target indices",
        &changed_targets,
        &vec![1, 3, 5, 6, 7],
    )?;
    for &index in changed_features.iter().chain(changed_targets.iter()) {
        if index == eligible_training[0] || index == eligible_training[1] {
            return Err(format!(
                "changed feature or target at cutoff-12 eligible training row {index}"
            ));
        }
    }
    for &index in &eligible_training {
        check_eq(
            &format!("row {index} fit metadata"),
            &mutated_rows[index],
            &rows[index],
        )?;
        check_bits(
            &format!("row {index} fit feature"),
            mutated_x[index],
            X[index],
        )?;
        check_bits(
            &format!("row {index} fit target"),
            mutated_y[index],
            Y[index],
        )?;
    }
    for &index in &[4usize, 5usize] {
        if mutated_rows[index].decision_at != rows[index].decision_at
            || mutated_rows[index].features_available_at != rows[index].features_available_at
        {
            return Err(format!(
                "prediction row {index} decision or feature availability changed"
            ));
        }
        check_bits(
            &format!("prediction row {index} feature"),
            mutated_x[index],
            X[index],
        )?;
    }
    check_bits("row 4 prediction target", mutated_y[4], Y[4])?;
    if mutated_y[5].to_bits() == Y[5].to_bits() {
        return Err("row 5 target should change".into());
    }

    for (name, base, mutated) in [
        ("expanding", &expanding, &mutated_expanding),
        ("rolling", &rolling, &mutated_rolling),
    ] {
        check_eq(
            &format!("{name} cutoff-12 test range after mutation"),
            &mutated.folds[0].test_range(),
            &(4..6),
        )?;
        check_eq(
            &format!("{name} cutoff-12 fold after mutation"),
            &base.folds[0],
            &mutated.folds[0],
        )?;
        check_bits(
            &format!("{name} frozen mean after mutation"),
            base.mean,
            mutated.mean,
        )?;
        check_bits(
            &format!("{name} row 4 after mutation"),
            base.predictions[0],
            mutated.predictions[0],
        )?;
        check_bits(
            &format!("{name} row 5 after mutation"),
            base.predictions[1],
            mutated.predictions[1],
        )?;
        check_close(
            &format!("{name} row 4 mutated residual"),
            mutated.predictions[0] - mutated_y[4],
            0.0,
        )?;
    }

    if mutated_rows[5]
        .label_available_at
        .is_none_or(|label| label < SCORING_CUTOFF)
    {
        return Err("mutated row 5 should remain unscored at cutoff 30".into());
    }

    println!("coverage = [0, 30), anchor = 0, train = 10, gap = 2, test = 5, step = 6");
    println!(
        "expanding folds = {}; rolling folds = {}",
        expanding.folds.len(),
        rolling.folds.len()
    );
    println!("cutoff 12 eligible training = [0, 2]");
    println!("frozen mean = {}", expanding.mean);
    println!("predictions = {:?}", expanding.predictions);
    println!("row 4 residual = {:.6}", expanding.predictions[0] - Y[4]);
    println!("row 5 label = unknown; scored = false");
    println!("mutation parity = true");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("data_splitter_time: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn data_splitter_time_workflow_succeeds() {
        super::run().expect("data_splitter_time workflow");
    }
}
