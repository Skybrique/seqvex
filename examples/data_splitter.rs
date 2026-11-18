//! Runnable Stage 1 row splitter example.
//!
//! The caller owns the aligned rows. This example checks alignment before
//! slicing, contrasts dropped and kept tails, demonstrates one constructor
//! error, shows that a NaN feature does not change membership, and fits a
//! simple preprocessing statistic on training rows only.

use std::error::Error;
use std::ops::Range;

use seqvex::foundation::error::DimensionMismatch;
use seqvex::foundation::numerical::Vector;
use seqvex::preprocessing::split::{RowWindowSpec, TailPolicy, expanding, rolling};

type FoldRanges = Vec<(Range<usize>, Range<usize>)>;

fn main() -> Result<(), Box<dyn Error>> {
    let features: Vec<Vector> = (0..17)
        .map(|index| Vector::from_slice(&[index as f32]))
        .collect();
    let targets: Vec<f32> = (0..17).map(|index| 3.0 * index as f32 + 5.0).collect();

    check_alignment(&features, &targets)?;
    match check_alignment(&features, &targets[..16]) {
        Err(error) => println!("alignment error before slicing: {error}"),
        Ok(()) => return Err("expected a feature/target length mismatch".into()),
    }

    let keep = RowWindowSpec::new(2, 4, 3, 4, 1, TailPolicy::KeepShort);
    let drop = RowWindowSpec::new(2, 4, 3, 4, 1, TailPolicy::Drop);

    let expanding_folds: Vec<_> = expanding(17, keep)?.collect();
    let rolling_folds: Vec<_> = rolling(17, keep)?.collect();
    let dropped_folds: Vec<_> = expanding(17, drop)?.collect();

    println!("KeepShort expanding folds: {}", expanding_folds.len());
    println!("KeepShort rolling folds: {}", rolling_folds.len());
    println!("Drop expanding folds: {}", dropped_folds.len());

    for fold in &expanding_folds {
        let train = fold.train_range();
        let test = fold.test_range();
        let train_features = &features[train.clone()];
        let train_targets = &targets[train.clone()];
        let test_features = &features[test.clone()];
        let test_targets = &targets[test.clone()];
        println!(
            "expanding train {train:?} test {test:?} lengths {} {} {} {}",
            train_features.len(),
            train_targets.len(),
            test_features.len(),
            test_targets.len()
        );
    }
    for fold in &rolling_folds {
        println!(
            "rolling train {:?} test {:?}",
            fold.train_range(),
            fold.test_range()
        );
    }

    match expanding(17, RowWindowSpec::new(0, 3, 3, 0, 0, TailPolicy::Drop)) {
        Err(error) => println!("constructor error: {error}"),
        Ok(_) => return Err("expected ZeroStepRows".into()),
    }

    let first = expanding_folds.first().expect("at least one fold");
    let train_mean_before = mean_feature(&features, first.train_range());
    let mut changed = features.clone();
    changed[first.test_range().start] = Vector::from_slice(&[999.0]);
    let train_mean_after = mean_feature(&changed, first.train_range());
    println!(
        "training-only mean before/after held-out mutation: {train_mean_before} {train_mean_after}"
    );
    if train_mean_before.to_bits() != train_mean_after.to_bits() {
        return Err("held-out mutation changed the training-fitted statistic".into());
    }

    let membership_before = split_membership(&features, &targets, keep)?;
    let mut nan_features = features.clone();
    nan_features[5] = Vector::from_slice(&[f32::NAN]);
    let membership_after = split_membership(&nan_features, &targets, keep)?;
    if membership_before != membership_after {
        return Err("a NaN feature changed row membership".into());
    }
    println!("NaN feature leaves membership unchanged");

    Ok(())
}

fn check_alignment(features: &[Vector], targets: &[f32]) -> Result<(), DimensionMismatch> {
    if features.len() != targets.len() {
        return Err(DimensionMismatch {
            expected: features.len(),
            actual: targets.len(),
        });
    }
    Ok(())
}

fn split_membership(
    features: &[Vector],
    targets: &[f32],
    keep: RowWindowSpec,
) -> Result<FoldRanges, Box<dyn Error>> {
    check_alignment(features, targets)?;
    Ok(expanding(features.len(), keep)?
        .map(|fold| (fold.train_range(), fold.test_range()))
        .collect())
}

fn mean_feature(features: &[Vector], range: Range<usize>) -> f32 {
    let sum: f32 = features[range.clone()]
        .iter()
        .map(|value| value.as_slice()[0])
        .sum();
    sum / (range.end - range.start) as f32
}
