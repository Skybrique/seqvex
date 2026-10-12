//! Synthetic temporal-**selection** tests for the OLS fitter.
//!
//! These tests demonstrate that a caller can build a chronological partition
//! with a purge/embargo gap and that the fitter honors the explicit training
//! indices it is given. They do **not** claim the fitter infers timestamp
//! eligibility, label availability, or overlapping label intervals: the fitter
//! receives numeric arrays and training indices, so all partitioning is
//! caller-owned.
//!
//! All data here is synthetic and constructed with known relationships.

use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::{FitConfig, FitError, fit};

// ---- test-local interval helper (not production preparation) ----

/// Inclusive label interval `[origin, origin + HORIZON]`.
const HORIZON: i64 = 2;
/// Label cutoff; a label is eligible when its interval end is `<= CUTOFF`.
const CUTOFF: i64 = 100;

fn label_interval(origin: i64) -> (i64, i64) {
    (origin, origin + HORIZON)
}

fn eligible(origin: i64) -> bool {
    origin + HORIZON <= CUTOFF
}

fn overlaps(a: (i64, i64), b: (i64, i64)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

// ---- synthetic data helpers ----

fn features(values: &[f32]) -> Vec<Vector> {
    values
        .iter()
        .map(|value| Vector::from_slice(&[*value]))
        .collect()
}

fn linear_features(count: usize) -> Vec<Vector> {
    features(&(0..count).map(|t| t as f32).collect::<Vec<_>>())
}

fn fit_line(features: &[Vector], targets: &[f32], train: &[usize]) -> (f32, f32) {
    let result = fit(features, targets, train, &FitConfig::default()).expect("fit");
    (result.intercept(), result.weights().as_slice()[0])
}

fn approx(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

#[test]
fn chronological_eligibility_and_purge_partition() {
    let initial: Vec<i64> = (0..=90).collect();
    let evaluation: Vec<i64> = (91..=98).collect();
    let purged = [89i64, 90];
    let final_training: Vec<i64> = (0..=88).collect();

    // Eligibility boundaries: the label interval must end at or before CUTOFF.
    assert!(eligible(88));
    assert!(eligible(98)); // evaluation origins are eligible
    assert!(!eligible(99));
    assert!(!eligible(100));
    assert_eq!(label_interval(98), (98, 100));

    // The initial candidate set is exactly final training plus the purged pair.
    let mut rebuilt = final_training.clone();
    rebuilt.extend_from_slice(&purged);
    assert_eq!(rebuilt, initial);

    // Training and evaluation identities are disjoint.
    for origin in &final_training {
        assert!(!evaluation.contains(origin));
    }

    // Evaluation label intervals cover [91, 100].
    let eval_start = *evaluation.first().unwrap();
    let eval_end = label_interval(*evaluation.last().unwrap()).1;
    assert_eq!((eval_start, eval_end), (91, 100));

    // Purge reason: the two purged training intervals overlap the evaluation union.
    let eval_union = (eval_start, eval_end);
    assert!(overlaps(label_interval(89), eval_union));
    assert!(overlaps(label_interval(90), eval_union));
    assert!(!overlaps(label_interval(88), eval_union));

    // After purging, no training interval overlaps any evaluation interval.
    for &train_origin in &final_training {
        for &eval_origin in &evaluation {
            assert!(
                !overlaps(label_interval(train_origin), label_interval(eval_origin)),
                "training origin {train_origin} overlaps evaluation origin {eval_origin}"
            );
        }
    }

    // Ineligible origins are excluded for the cutoff reason, not the purge reason.
    for origin in [99i64, 100] {
        assert!(!eligible(origin));
        assert!(!final_training.contains(&origin));
        assert!(!evaluation.contains(&origin));
    }

    // Fit only the final training selection: y = 3x + 5 is recovered exactly.
    let train: Vec<usize> = final_training.iter().map(|&t| t as usize).collect();
    let x = linear_features(101);
    let y: Vec<f32> = (0..101).map(|t| 3.0 * t as f32 + 5.0).collect();
    let (intercept, slope) = fit_line(&x, &y, &train);
    approx(intercept, 5.0, 1e-3);
    approx(slope, 3.0, 1e-3);
}

#[test]
fn unselected_rows_do_not_affect_the_fit() {
    let train: Vec<usize> = (0..=88).collect();

    let x_base = linear_features(101);
    let y_base: Vec<f32> = (0..101).map(|t| 3.0 * t as f32 + 5.0).collect();
    let base = fit(&x_base, &y_base, &train, &FitConfig::default()).expect("base fit");

    // Change only unselected rows: evaluation (91..=98), purged (89, 90),
    // after-cutoff (99, 100) targets, and unselected feature values, including
    // non-finite values. Array lengths are preserved.
    let mut x_changed = x_base.clone();
    let mut y_changed = y_base.clone();
    for &t in &[89usize, 90] {
        y_changed[t] = 1.0e9;
    }
    for slot in &mut y_changed[91..=98] {
        *slot = -1.0e9;
    }
    y_changed[99] = f32::INFINITY;
    y_changed[100] = f32::NAN;
    for slot in &mut x_changed[89..=100] {
        *slot = Vector::from_slice(&[f32::NAN]);
    }

    let changed = fit(&x_changed, &y_changed, &train, &FitConfig::default()).expect("changed fit");

    assert_eq!(
        base.weights().as_slice()[0].to_bits(),
        changed.weights().as_slice()[0].to_bits()
    );
    assert_eq!(base.intercept().to_bits(), changed.intercept().to_bits());
    assert_eq!(base.diagnostics(), changed.diagnostics());
}

#[test]
fn selected_target_change_updates_the_fit_predictably() {
    // Two-point selection through the origin: y = w x, so w is determined exactly.
    let x = features(&[0.0, 1.0]);
    let base = fit(&x, &[0.0, 2.0], &[0, 1], &FitConfig::default()).expect("base");
    approx(base.weights().as_slice()[0], 2.0, 1e-6);
    approx(base.intercept(), 0.0, 1e-6);

    let changed = fit(&x, &[0.0, 4.0], &[0, 1], &FitConfig::default()).expect("changed");
    approx(changed.weights().as_slice()[0], 4.0, 1e-6);
    approx(changed.intercept(), 0.0, 1e-6);
    assert_ne!(
        base.weights().as_slice()[0].to_bits(),
        changed.weights().as_slice()[0].to_bits()
    );
}

#[test]
fn non_finite_selected_training_value_is_rejected() {
    let x = linear_features(3);
    let mut y_non_finite = vec![5.0f32, 8.0, 11.0];
    y_non_finite[0] = f32::NAN;
    assert_eq!(
        fit(&x, &y_non_finite, &[0, 1, 2], &FitConfig::default()),
        Err(FitError::NonFiniteTrainingValue { row: 0, column: 1 })
    );

    let mut x_non_finite = x;
    x_non_finite[0] = Vector::from_slice(&[f32::INFINITY]);
    assert_eq!(
        fit(
            &x_non_finite,
            &[5.0f32, 8.0, 11.0],
            &[0, 1, 2],
            &FitConfig::default()
        ),
        Err(FitError::NonFiniteTrainingValue { row: 0, column: 0 })
    );
}

#[test]
fn distinct_regimes_are_recovered_independently() {
    // Synthetic regime change: early y = 2x + 1 for origins 0..=49, later
    // y = -x + 10 for origins 50..=99.
    let x = linear_features(100);
    let y: Vec<f32> = (0..100)
        .map(|t| {
            if t <= 49 {
                2.0 * t as f32 + 1.0
            } else {
                -(t as f32) + 10.0
            }
        })
        .collect();

    let early_train: Vec<usize> = (0..=49).collect();
    let late_train: Vec<usize> = (50..=99).collect();

    let (early_intercept, early_slope) = fit_line(&x, &y, &early_train);
    approx(early_intercept, 1.0, 1e-3);
    approx(early_slope, 2.0, 1e-3);
    let early_bits = (early_intercept.to_bits(), early_slope.to_bits());

    let (late_intercept, late_slope) = fit_line(&x, &y, &late_train);
    approx(late_intercept, 10.0, 1e-3);
    approx(late_slope, -1.0, 1e-3);

    // The later window does not blend in the earlier regime.
    assert!(late_slope < 0.0);
    assert_ne!(late_slope.to_bits(), early_slope.to_bits());

    // Refitting the early window reproduces the captured early result: building
    // the later candidate left the earlier fit unchanged.
    let (early_intercept_again, early_slope_again) = fit_line(&x, &y, &early_train);
    assert_eq!(
        (early_intercept_again.to_bits(), early_slope_again.to_bits()),
        early_bits
    );
}
