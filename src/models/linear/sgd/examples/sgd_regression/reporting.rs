use std::error::Error;

use seqvex::models::linear::sgd::Sgd;

use super::training::{
    evaluate_adaptive, evaluate_frozen, max_prefix_squared_norm, metrics_of, train_prefix,
};
use super::trump::{
    DIMENSION, PREFIX, TARGET_COLUMN, baseline_metrics, format_weights, read_dataset_with_hash,
};
use super::{EPOCHS, ETA, EXPECTED_SHA256};

/// Runs the full CSV workflow for a caller-supplied path.
pub(crate) fn run_with_path(path: &str) -> Result<(), Box<dyn Error>> {
    let (rows, sha256) = read_dataset_with_hash(path)?;
    let scaler = super::trump::Standardizer::from_prefix(&rows)?;
    let scored = rows.len() - PREFIX;

    println!("SGD real-data learning/prediction: {path}");
    println!("dataset sha256: {sha256}");
    if sha256 == EXPECTED_SHA256 {
        println!("  matches the recorded snapshot");
    } else {
        println!("  differs from the recorded snapshot {EXPECTED_SHA256}; record and review");
    }
    println!(
        "features = [gallup, ipsos, morning_consult, rasmussen, you_gov, 1.0] \
         (dimension {DIMENSION}); target = {TARGET_COLUMN}; ordinal_date = metadata only"
    );
    println!(
        "rows={} prefix={PREFIX} scored={scored} eta={ETA} epochs={EPOCHS}",
        rows.len()
    );

    let model = Sgd::new(DIMENSION, ETA).map_err(|error| format!("Sgd::new failed: {error:?}"))?;

    let m_prefix = max_prefix_squared_norm(&rows, &scaler)?;
    let product = f64::from(ETA) * m_prefix;
    if !product.is_finite() || product >= 1.0 {
        return Err(format!(
            "prefix damping check failed: eta*M_prefix = {product} >= 1 \
             (M_prefix = {m_prefix}); report the discrepancy for review; \
             do not lower eta, skip rows, or change preprocessing"
        )
        .into());
    }
    println!("prefix M_prefix={m_prefix:.9} eta*M_prefix={product:.9} (< 1)");

    let trained = train_prefix(&model, &rows, &scaler)?;
    println!(
        "weights at prefix boundary (standardized space) = [{}]",
        format_weights(trained.w.as_slice())
    );

    let frozen = evaluate_frozen(&model, &trained, &rows, &scaler)?;
    let adaptive = evaluate_adaptive(&model, &trained, &rows, &scaler)?;
    let baselines = baseline_metrics(&rows)?;

    let frozen_metrics = metrics_of(&frozen)?;
    let adaptive_metrics = metrics_of(&adaptive)?;

    println!("metrics (scored rows only, n={scored}):");
    println!("  {:<26} {:>14} {:>14}", "predictor", "MAE", "RMSE");
    println!(
        "  {:<26} {:>14.6} {:>14.6}",
        "sgd frozen", frozen_metrics.mae, frozen_metrics.rmse
    );
    println!(
        "  {:<26} {:>14.6} {:>14.6}",
        "sgd adaptive", adaptive_metrics.mae, adaptive_metrics.rmse
    );
    for (name, metrics) in &baselines {
        println!(
            "  {:<26} {:>14.6} {:>14.6}",
            name, metrics.mae, metrics.rmse
        );
    }

    println!("sample adaptive predictions: date  actual  prediction  error");
    let mut selected: Vec<usize> = (0..adaptive.len().min(3)).collect();
    for index in adaptive.len().saturating_sub(3)..adaptive.len() {
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    for index in selected {
        let entry = &adaptive[index];
        println!(
            "  {}  {:.6}  {:.6}  {:+.6e}",
            entry.ordinal_date, entry.actual, entry.prediction, entry.error
        );
    }

    Ok(())
}
