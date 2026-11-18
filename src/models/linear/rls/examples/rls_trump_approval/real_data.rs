use std::error::Error;

use seqvex::models::linear::rls::Rls;

use super::lambda::{LambdaRun, run_lambda};
use super::trump::{
    DIMENSION, FEATURE_COLUMNS, PREFIX, TARGET_COLUMN, baseline_metrics, format_weights,
    read_dataset_with_hash,
};
use super::{DELTA, LAMBDAS};

/// Runs the full CSV workflow for a caller-supplied path.
pub(crate) fn run_with_path(path: &str) -> Result<(), Box<dyn Error>> {
    let (rows, _) = read_dataset_with_hash(path)?;
    let scaler = super::trump::Standardizer::from_prefix(&rows)?;
    let scored = rows.len() - PREFIX;

    println!("RLS real-data validation: {path}");
    println!(
        "features = [gallup, ipsos, morning_consult, rasmussen, you_gov, 1.0] \
         (dimension {DIMENSION}); target = {TARGET_COLUMN}; ordinal_date = metadata only"
    );
    println!(
        "rows={} prefix={PREFIX} scored={scored} delta={DELTA} lambdas={LAMBDAS:?}",
        rows.len()
    );
    for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
        println!(
            "scaler(prefix) {name}: mean={:.6} scale={:.6}",
            scaler.mean[j], scaler.scale[j]
        );
    }

    let mut runs: Vec<(f32, LambdaRun)> = Vec::with_capacity(LAMBDAS.len());
    for &lambda in &LAMBDAS {
        let model = Rls::new(DIMENSION, lambda, DELTA)
            .map_err(|error| format!("Rls::new failed for lambda={lambda}: {error:?}"))?;
        let run = run_lambda(&model, &rows, &scaler, lambda)?;
        println!(
            "lambda={lambda}: weights at prefix boundary (standardized space) = [{}]",
            format_weights(&run.weights_at_boundary)
        );
        runs.push((lambda, run));
    }

    let baselines = baseline_metrics(&rows)?;

    println!("metrics (scored rows only, n={scored}):");
    println!("  {:<24} {:>14} {:>14}", "predictor", "MAE", "RMSE");
    for (lambda, run) in &runs {
        println!(
            "  {:<24} {:>14.6} {:>14.6}",
            format!("rls lambda={lambda}"),
            run.metrics.mae,
            run.metrics.rmse
        );
    }
    for (name, metrics) in &baselines {
        println!(
            "  {:<24} {:>14.6} {:>14.6}",
            name, metrics.mae, metrics.rmse
        );
    }

    for (lambda, run) in &runs {
        println!("sample predictions (lambda={lambda}): date  actual  prediction  error");
        let predictions = &run.predictions;
        let mut selected: Vec<usize> = (0..predictions.len().min(5)).collect();
        for index in predictions.len().saturating_sub(5)..predictions.len() {
            if !selected.contains(&index) {
                selected.push(index);
            }
        }
        for index in selected {
            let entry = &predictions[index];
            println!(
                "  {}  {:.6}  {:.6}  {:+.6e}",
                entry.ordinal_date, entry.actual, entry.prediction, entry.error
            );
        }
    }

    Ok(())
}
