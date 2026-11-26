//! Runnable splitter → sampler → OLS consumer for training-fold sampling.
//!
//! The eight-row fixture is the delivered Stage 2 cutoff fixture. Units are
//! built from `TimeFold::train_ranges()` only; outcome supports are declared
//! from eligible training observations and end at or before the cutoff. The
//! example checks cutoff containment, grouped multiplicity, seeded replay, the
//! excluded-support mutation control, a fragmented group, and Stage 1
//! composition. Preprocessing is centered on the sampled rows only and frozen
//! before prediction.

#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;
use std::process::ExitCode;

use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::{FitConfig, fit};
use seqvex::preprocessing::sampling::{
    OutcomeInterval, SampleMethod, SampleUnit, SamplingMode, SamplingSpec, SequentialLimits,
    sample_training, sample_training_sequential_f32,
};
use seqvex::preprocessing::split::{
    RowWindowSpec, TailPolicy, TimeCoverage, TimeFold, TimeFoldOutcome, TimeRow, TimeWindowSpec,
    expanding, expanding_time,
};

const TOLERANCE: f32 = 1e-5;

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

fn check_eq<T: std::fmt::Debug + PartialEq + ?Sized>(
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
            "{label}: expected {expected}, got {actual} (difference {difference})"
        ))
    }
}

fn single(row: usize) -> SampleUnit {
    SampleUnit {
        rows: vec![row..row + 1],
        outcomes: vec![],
        sequence: 0,
    }
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

fn units_from_fold(fold: &TimeFold, supports: &[Vec<(i64, i64)>]) -> Vec<SampleUnit> {
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
                outcomes: merged_outcomes(supports, start..end),
                sequence: 0,
            });
            start = end;
        }
    }
    units
}

fn run() -> Result<(), String> {
    let rows = fixture_rows();
    let folds = expanding_time(
        8,
        &rows,
        TimeCoverage::new(0, 30, 0),
        TimeWindowSpec::new(10, 5, 6, 2, 3),
    )
    .map_err(|error| format!("time schedule failed: {error}"))?
    .collect::<Vec<_>>();
    check_eq("fold count", &folds.len(), &3)?;

    let cutoff_12 = &folds[0];
    check_eq("cutoff 12", &cutoff_12.cutoff(), &12)?;
    check_eq(
        "cutoff 12 outcome",
        &cutoff_12.outcome(),
        &TimeFoldOutcome::Usable,
    )?;
    check_eq(
        "cutoff 12 train",
        &cutoff_12.train_ranges().to_vec(),
        &vec![0..1, 2..3],
    )?;

    let cutoff_12_units = [single(0), single(2)];
    let identity = sample_training(&cutoff_12_units, SamplingSpec::disabled())
        .map_err(|error| format!("identity sampling failed: {error}"))?;
    check_eq("identity rows", identity.row_indices(), &[0, 2])?;

    let sampled = sample_training(
        &cutoff_12_units,
        SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 2, 5),
    )
    .map_err(|error| format!("cutoff-12 sampling failed: {error}"))?;
    check_eq("sampled rows", sampled.row_indices(), &[0, 2])?;
    for &row in sampled.row_indices() {
        if cutoff_12.test_range().contains(&row) {
            return Err(format!("cutoff-12 sample leaked held-out row {row}"));
        }
    }

    let training = sampled.row_indices().to_vec();
    let mean = training.iter().map(|&row| X[row]).sum::<f32>() / training.len() as f32;
    let transformed: Vec<Vector> = X
        .iter()
        .map(|&value| Vector::from_slice(&[value - mean]))
        .collect();
    let model = fit(&transformed, &Y, &training, &FitConfig::default())
        .map_err(|error| format!("OLS fit failed: {error:?}"))?
        .into_predictor();
    let row_4 = model
        .predict_checked(&transformed[4])
        .map_err(|error| format!("row 4 prediction failed: {error:?}"))?;
    let row_5 = model
        .predict_checked(&transformed[5])
        .map_err(|error| format!("row 5 prediction failed: {error:?}"))?;
    check_close("frozen mean", mean, 1.0)?;
    check_close("row 4 prediction", row_4, 9.0)?;
    check_close("row 5 prediction", row_5, 11.0)?;
    check_close("row 4 residual", row_4 - Y[4], 0.0)?;

    let cutoff_24 = &folds[2];
    check_eq("cutoff 24", &cutoff_24.cutoff(), &24)?;
    check_eq(
        "cutoff 24 outcome",
        &cutoff_24.outcome(),
        &TimeFoldOutcome::Usable,
    )?;
    check_eq(
        "cutoff 24 train",
        &cutoff_24.train_ranges().to_vec(),
        &vec![0..3, 4..5, 6..7],
    )?;

    let supports = support_map();
    let units = units_from_fold(cutoff_24, &supports);
    check_eq("cutoff-24 unit count", &units.len(), &4)?;
    let shapes: Vec<(Vec<Range<usize>>, Vec<OutcomeInterval>)> = units
        .iter()
        .map(|unit| (unit.rows.clone(), unit.outcomes.clone()))
        .collect();
    check_eq(
        "cutoff-24 units",
        &shapes,
        &vec![
            (vec![0..1], vec![OutcomeInterval { start: 1, end: 4 }]),
            (vec![1..3], vec![OutcomeInterval { start: 4, end: 7 }]),
            (vec![4..5], vec![OutcomeInterval { start: 12, end: 15 }]),
            (vec![6..7], vec![OutcomeInterval { start: 17, end: 20 }]),
        ],
    )?;

    let sequential = SamplingSpec::new(SamplingMode::Default, 4, 31);
    let baseline = sample_training(&units, sequential)
        .map_err(|error| format!("cutoff-24 sampling failed: {error}"))?;
    let replay = sample_training(&units, sequential)
        .map_err(|error| format!("cutoff-24 replay failed: {error}"))?;
    check_eq(
        "cutoff-24 replay rows",
        baseline.row_indices(),
        replay.row_indices(),
    )?;
    check_eq(
        "cutoff-24 replay counts",
        baseline.unit_counts(),
        replay.unit_counts(),
    )?;
    check_eq(
        "cutoff-24 draw count",
        &baseline.unit_counts().iter().sum::<usize>(),
        &4,
    )?;
    for &row in baseline.row_indices() {
        if !matches!(row, 0 | 1 | 2 | 4 | 6) {
            return Err(format!("cutoff-24 sample contains ineligible row {row}"));
        }
    }

    let mut mutated_supports = support_map();
    mutated_supports[3] = vec![(100, 140)];
    mutated_supports[5] = vec![(200, 260)];
    mutated_supports[7] = vec![(300, 330)];
    let mutated_units = units_from_fold(cutoff_24, &mutated_supports);
    let mutated_shapes: Vec<(Vec<Range<usize>>, Vec<OutcomeInterval>)> = mutated_units
        .iter()
        .map(|unit| (unit.rows.clone(), unit.outcomes.clone()))
        .collect();
    check_eq("mutation units", &shapes, &mutated_shapes)?;
    check_eq(
        "mutation train ranges",
        &cutoff_24.train_ranges().to_vec(),
        &vec![0..3, 4..5, 6..7],
    )?;
    check_eq("mutation test range", &cutoff_24.test_range(), &(7..8))?;
    let mutated = sample_training(&mutated_units, sequential)
        .map_err(|error| format!("mutated sampling failed: {error}"))?;
    check_eq(
        "mutation rows",
        baseline.row_indices(),
        mutated.row_indices(),
    )?;
    check_eq(
        "mutation counts",
        baseline.unit_counts(),
        mutated.unit_counts(),
    )?;

    let strict_spec = SamplingSpec::new(SamplingMode::Default, 4, 21);
    let strict_limits = SequentialLimits::default();
    let strict = sample_training_sequential_f32(&units, strict_spec, &strict_limits)
        .map_err(|error| format!("strict cutoff-24 sampling failed: {error}"))?;
    let strict_replay = sample_training_sequential_f32(&units, strict_spec, &strict_limits)
        .map_err(|error| format!("strict cutoff-24 replay failed: {error}"))?;
    check_eq(
        "strict cutoff-24 replay rows",
        strict.row_indices(),
        strict_replay.row_indices(),
    )?;
    check_eq(
        "strict cutoff-24 replay counts",
        strict.unit_counts(),
        strict_replay.unit_counts(),
    )?;
    check_eq(
        "strict cutoff-24 draw count",
        &strict.unit_counts().iter().sum::<usize>(),
        &4,
    )?;
    for &row in strict.row_indices() {
        if !matches!(row, 0 | 1 | 2 | 4 | 6) {
            return Err(format!(
                "strict cutoff-24 sample contains ineligible row {row}"
            ));
        }
    }
    let strict_mutated =
        sample_training_sequential_f32(&mutated_units, strict_spec, &strict_limits)
            .map_err(|error| format!("strict mutated sampling failed: {error}"))?;
    check_eq(
        "strict mutation rows",
        strict.row_indices(),
        strict_mutated.row_indices(),
    )?;
    check_eq(
        "strict mutation counts",
        strict.unit_counts(),
        strict_mutated.unit_counts(),
    )?;

    let snapshot = [
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
    let fragmented_fold = expanding_time(
        4,
        &snapshot,
        TimeCoverage::new(4, 20, 4),
        TimeWindowSpec::new(1, 5, 20, 0, 1),
    )
    .map_err(|error| format!("fragmented schedule failed: {error}"))?
    .next()
    .expect("one fragmented fold");
    check_eq(
        "fragmented outcome",
        &fragmented_fold.outcome(),
        &TimeFoldOutcome::Usable,
    )?;
    check_eq(
        "fragmented train",
        &fragmented_fold.train_ranges().to_vec(),
        &vec![0..1, 2..3],
    )?;
    let fragmented = [SampleUnit {
        rows: vec![0..1, 2..3],
        outcomes: vec![],
        sequence: 0,
    }];
    let identity = sample_training(&fragmented, SamplingSpec::disabled())
        .map_err(|error| format!("fragmented identity failed: {error}"))?;
    check_eq("fragmented identity rows", identity.row_indices(), &[0, 2])?;
    let repeated = sample_training(
        &fragmented,
        SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 2, 4),
    )
    .map_err(|error| format!("fragmented repeat failed: {error}"))?;
    check_eq("fragmented counts", repeated.unit_counts(), &[2])?;
    let mut fragmented_rows = repeated.row_indices().to_vec();
    fragmented_rows.sort_unstable();
    check_eq("fragmented rows", &fragmented_rows, &vec![0, 0, 2, 2])?;

    let row_folds = expanding(10, RowWindowSpec::new(0, 3, 4, 1, 0, TailPolicy::Drop))
        .map_err(|error| format!("row schedule failed: {error}"))?
        .collect::<Vec<_>>();
    let row_fold = &row_folds[0];
    check_eq("stage 1 train", &row_fold.train_range(), &(0..3))?;
    check_eq("stage 1 test", &row_fold.test_range(), &(3..7))?;
    let row_units = [single(0), single(1), single(2)];
    let row_sample = sample_training(
        &row_units,
        SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 3, 5),
    )
    .map_err(|error| format!("stage 1 sampling failed: {error}"))?;
    check_eq("stage 1 rows", row_sample.row_indices(), &[0, 1, 2])?;
    for &row in row_sample.row_indices() {
        if row_fold.test_range().contains(&row) {
            return Err(format!("stage 1 sample leaked held-out row {row}"));
        }
    }

    let shuffle_units = [
        single(0),
        single(1),
        SampleUnit {
            rows: vec![2..3],
            outcomes: vec![],
            sequence: 1,
        },
    ];
    let shuffle_spec =
        SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 5, 17);
    let ordered = sample_training(&shuffle_units, shuffle_spec)
        .map_err(|error| format!("ordered sampling failed: {error}"))?;
    let shuffled = sample_training(&shuffle_units, shuffle_spec.with_shuffle(true))
        .map_err(|error| format!("shuffled sampling failed: {error}"))?;
    check_eq(
        "shuffle counts",
        ordered.unit_counts(),
        shuffled.unit_counts(),
    )?;
    let mut ordered_rows = ordered.row_indices().to_vec();
    let mut shuffled_rows = shuffled.row_indices().to_vec();
    ordered_rows.sort_unstable();
    shuffled_rows.sort_unstable();
    check_eq("shuffle multiset", &ordered_rows, &shuffled_rows)?;

    println!("cutoff 12 eligible training = [0, 2]; sample = [0, 2]");
    println!("frozen mean = {mean}");
    println!("predictions = [{row_4}, {row_5}]");
    println!("row 4 residual = {:.6}", row_4 - Y[4]);
    println!("row 5 label = unknown; scored = false");
    println!("cutoff 24 units = 4; supports = [1,4) [4,7) [12,15) [17,20)");
    println!("cutoff 24 seeded counts = {:?}", baseline.unit_counts());
    println!(
        "cutoff 24 strict rows = {:?}; counts = {:?}",
        strict.row_indices(),
        strict.unit_counts()
    );
    println!("excluded-support mutation parity = true");
    println!(
        "fragmented group rows = [0, 2]; repeated rows = {:?}",
        fragmented_rows
    );
    println!("stage 1 sample = [0, 1, 2]");
    println!("shuffle multiset parity = true");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("data_sampler: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn data_sampler_workflow_succeeds() {
        super::run().expect("data_sampler workflow");
    }
}
