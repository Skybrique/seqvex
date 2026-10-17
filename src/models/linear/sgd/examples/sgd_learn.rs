//! Deterministic synthetic demonstration of SGD-trained linear regression.
//!
//! The example generates data from a known linear relation with a declared
//! regime change, learns a prefix through ordered single-observation updates,
//! predicts from the learned weights, and scores the remaining rows with a
//! predict-before-update (prequential) protocol. Frozen evaluation (weights
//! frozen after the prefix) is reported separately from adaptive evaluation.
//!
//! Run with:
//!
//! ```text
//! cargo run --example sgd_learn
//! ```
//!
//! No files or network are used. The generator is bespoke to this example and is
//! deliberately distinct from the frozen correctness fixture in the `sgd` test
//! target. Expected improvements are not measured results until this example has
//! actually executed.

use std::error::Error;
use std::process::ExitCode;

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, SgdSample, SgdState};

/// Learning-prefix length.
const PREFIX: usize = 400;
/// Total number of generated rows.
const TOTAL: usize = 1000;
/// Regime-change position (inside the scoring window).
const SPLIT: usize = 500;
/// Model dimension: two features plus an appended constant.
const DIMENSION: usize = 3;
/// Example-specific fixed learning rate.
const ETA: f32 = 0.05;
/// Ordered passes over the learning prefix.
const EPOCHS: usize = 20;
/// Feature-stream seed (advance-before-draw).
const FEATURE_SEED: u32 = 20_261_018;
/// Noise-stream seed (advance-before-draw).
const NOISE_SEED: u32 = 20_261_019;
/// Coefficients for rows `< SPLIT` (intercept last).
const W_A: [f32; 3] = [0.7, -1.3, 0.5];
/// Coefficients for rows `>= SPLIT` (intercept last).
const W_B: [f32; 3] = [-0.5, 1.1, 2.0];

/// One generated row in original units.
struct Row {
    features: [f32; 3],
    target: f32,
}

/// Advance-before-draw LCG: advance first, then read `u = s / 2^32`.
fn unit(state: &mut u32) -> f64 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    f64::from(*state) / 4_294_967_296.0
}

/// Generates the fixed 1000-row fixture.
fn generate() -> Vec<Row> {
    let mut feature_state = FEATURE_SEED;
    let mut noise_state = NOISE_SEED;
    (0..TOTAL)
        .map(|i| {
            let mut features = [0.0f32; DIMENSION];
            for value in features.iter_mut().take(2) {
                *value = (2.0 * unit(&mut feature_state) - 1.0) as f32;
            }
            features[2] = 1.0;
            let noise = (0.05 * (2.0 * unit(&mut noise_state) - 1.0)) as f32;
            let coefficients = if i < SPLIT { W_A } else { W_B };
            let mut linear = 0.0f64;
            for j in 0..DIMENSION {
                linear += f64::from(coefficients[j]) * f64::from(features[j]);
            }
            Row {
                features,
                target: (linear + f64::from(noise)) as f32,
            }
        })
        .collect()
}

fn sample(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Metrics {
    mae: f64,
    rmse: f64,
}

struct Prediction {
    row: usize,
    actual: f32,
    prediction: f32,
    error: f64,
}

struct Evaluation {
    frozen: Vec<Prediction>,
    adaptive: Vec<Prediction>,
    expanding: Vec<Prediction>,
    previous: Vec<Prediction>,
}

fn record(row: usize, actual: f32, prediction: f32) -> Prediction {
    Prediction {
        row,
        actual,
        prediction,
        error: f64::from(actual) - f64::from(prediction),
    }
}

/// Runs the fixed workflow and returns the scored predictions. No expected
/// improvement is asserted; the example prints what actually happened.
fn evaluate() -> Result<Evaluation, Box<dyn Error>> {
    let rows = generate();
    let model = Sgd::new(DIMENSION, ETA).map_err(|error| format!("Sgd::new failed: {error:?}"))?;

    let mut trained = model.initial_state();
    for epoch in 0..EPOCHS {
        for (row_index, row) in rows.iter().enumerate().take(PREFIX) {
            trained = model
                .update(&trained, &sample(&row.features, row.target))
                .map_err(|error| {
                    format!("epoch {epoch}: learning update failed at row {row_index}: {error:?}")
                })?;
        }
    }

    let mut frozen = Vec::with_capacity(TOTAL - PREFIX);
    for (row_index, row) in rows.iter().enumerate().skip(PREFIX) {
        let features = Vector::from_slice(&row.features);
        let prediction = model
            .predict(&trained, &features)
            .map_err(|error| format!("frozen prediction failed at row {row_index}: {error:?}"))?;
        frozen.push(record(row_index, row.target, prediction));
    }

    let mut state: SgdState = trained.clone();
    let mut adaptive = Vec::with_capacity(TOTAL - PREFIX);
    for (row_index, row) in rows.iter().enumerate().skip(PREFIX) {
        let features = Vector::from_slice(&row.features);
        let prediction = model
            .predict(&state, &features)
            .map_err(|error| format!("adaptive prediction failed at row {row_index}: {error:?}"))?;
        adaptive.push(record(row_index, row.target, prediction));
        state = model
            .update(&state, &sample(&row.features, row.target))
            .map_err(|error| format!("adaptive update failed at row {row_index}: {error:?}"))?;
    }

    let mut running_sum: f64 = rows[..PREFIX].iter().map(|row| f64::from(row.target)).sum();
    let mut running_count = PREFIX as f64;
    let mut previous_target = f64::from(rows[PREFIX - 1].target);
    let mut expanding = Vec::with_capacity(TOTAL - PREFIX);
    let mut previous = Vec::with_capacity(TOTAL - PREFIX);
    for (row_index, row) in rows.iter().enumerate().skip(PREFIX) {
        let expanding_prediction = (running_sum / running_count) as f32;
        let previous_prediction = previous_target as f32;
        expanding.push(record(row_index, row.target, expanding_prediction));
        previous.push(record(row_index, row.target, previous_prediction));
        running_sum += f64::from(row.target);
        running_count += 1.0;
        previous_target = f64::from(row.target);
    }

    Ok(Evaluation {
        frozen,
        adaptive,
        expanding,
        previous,
    })
}

fn metrics(
    predictions: &[Prediction],
    start: usize,
    end: usize,
) -> Result<Metrics, Box<dyn Error>> {
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for prediction in predictions {
        if prediction.row >= start && prediction.row < end {
            sum_abs += prediction.error.abs();
            sum_sq += prediction.error * prediction.error;
            count += 1;
        }
    }
    if count == 0 {
        return Err(format!("empty scoring segment [{start}, {end})").into());
    }
    let metrics = Metrics {
        mae: sum_abs / count as f64,
        rmse: (sum_sq / count as f64).sqrt(),
    };
    if !metrics.mae.is_finite() || !metrics.rmse.is_finite() {
        return Err(format!("non-finite metrics for segment [{start}, {end})").into());
    }
    Ok(metrics)
}

fn run() -> Result<(), Box<dyn Error>> {
    let evaluation = evaluate()?;
    let scored = TOTAL - PREFIX;
    println!("SGD synthetic learning demonstration (bespoke deterministic generator)");
    println!(
        "rows={TOTAL} prefix={PREFIX} scored={scored} regime_split={SPLIT} \
         dimension={DIMENSION} eta={ETA} epochs={EPOCHS}"
    );
    println!("feature order = [f1, f2, 1.0]; w_A={W_A:?}; w_B={W_B:?}; noise=0.05*(2v-1)");
    println!("metrics (identical scoring rows; frozen and adaptive are separate):");
    println!(
        "  {:<18} {:<12} {:>12} {:>12}",
        "predictor", "segment", "MAE", "RMSE"
    );
    let segments = [
        ("overall", PREFIX, TOTAL),
        ("pre-change", PREFIX, SPLIT),
        ("post-change", SPLIT, TOTAL),
    ];
    for (name, predictions) in [
        ("sgd frozen", &evaluation.frozen),
        ("sgd adaptive", &evaluation.adaptive),
        ("expanding mean", &evaluation.expanding),
        ("previous target", &evaluation.previous),
    ] {
        for (segment, start, end) in segments {
            let metrics = metrics(predictions, start, end)?;
            println!(
                "  {:<18} {:<12} {:>12.6} {:>12.6}",
                name, segment, metrics.mae, metrics.rmse
            );
        }
    }

    println!("sample adaptive predictions: row  actual  prediction  error");
    let mut selected: Vec<usize> = (0..evaluation.adaptive.len().min(3)).collect();
    for index in evaluation.adaptive.len().saturating_sub(3)..evaluation.adaptive.len() {
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    for index in selected {
        let entry = &evaluation.adaptive[index];
        println!(
            "  {}  {:.6}  {:.6}  {:+.6e}",
            entry.row, entry.actual, entry.prediction, entry.error
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sgd_learn: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::evaluate;

    #[test]
    fn synthetic_workflow_is_deterministic() {
        let first = evaluate().expect("first evaluation");
        let second = evaluate().expect("second evaluation");
        for (left, right) in [
            (&first.frozen, &second.frozen),
            (&first.adaptive, &second.adaptive),
            (&first.expanding, &second.expanding),
            (&first.previous, &second.previous),
        ] {
            assert_eq!(left.len(), right.len());
            for (left, right) in left.iter().zip(right) {
                assert_eq!(left.prediction.to_bits(), right.prediction.to_bits());
                assert_eq!(left.error.to_bits(), right.error.to_bits());
            }
        }
    }

    #[test]
    fn all_metrics_are_finite() {
        let evaluation = evaluate().expect("evaluation");
        for predictions in [
            &evaluation.frozen,
            &evaluation.adaptive,
            &evaluation.expanding,
            &evaluation.previous,
        ] {
            for prediction in predictions {
                assert!(prediction.prediction.is_finite());
                assert!(prediction.error.is_finite());
            }
        }
    }
}
