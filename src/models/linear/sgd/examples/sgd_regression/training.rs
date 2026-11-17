use std::error::Error;

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, SgdSample, SgdState};

use super::trump::{Metrics, PREFIX, Prediction, Row, Standardizer};
use super::{EPOCHS, ETA};

fn sample(features: Vector, target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample { features, target })
}

pub(crate) fn squared_norm(features: &Vector) -> f64 {
    features
        .as_slice()
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum()
}

/// Measures `M_prefix = max_i ||x_i||^2` from the actual transformed `f32`
/// prefix vectors (including the appended intercept).
pub(crate) fn max_prefix_squared_norm(
    rows: &[Row],
    scaler: &Standardizer,
) -> Result<f64, Box<dyn Error>> {
    let mut maximum = 0.0f64;
    for (i, row) in rows.iter().enumerate().take(PREFIX) {
        let features = scaler.apply(row, i)?;
        let norm_sq = squared_norm(&features);
        if !norm_sq.is_finite() {
            return Err(format!(
                "preprocessing: row {i} (date {}): non-finite squared norm {norm_sq}",
                row.ordinal_date
            )
            .into());
        }
        if norm_sq > maximum {
            maximum = norm_sq;
        }
    }
    Ok(maximum)
}

/// Learns the prefix with [`EPOCHS`] ordered passes.
pub(crate) fn train_prefix(
    model: &Sgd,
    rows: &[Row],
    scaler: &Standardizer,
) -> Result<SgdState, Box<dyn Error>> {
    let mut state = model.initial_state();
    for epoch in 0..EPOCHS {
        for (i, row) in rows.iter().enumerate().take(PREFIX) {
            let features = scaler.apply(row, i)?;
            state = model
                .update(&state, &sample(features, row.target))
                .map_err(|error| -> Box<dyn Error> {
                    format!(
                        "epoch {epoch}: learning update failed at row {i} (date {}): {error:?}",
                        row.ordinal_date
                    )
                    .into()
                })?;
        }
    }
    Ok(state)
}

/// Frozen evaluation: weights are frozen after the prefix; only prediction is
/// performed on the scoring rows.
pub(crate) fn evaluate_frozen(
    model: &Sgd,
    state: &SgdState,
    rows: &[Row],
    scaler: &Standardizer,
) -> Result<Vec<Prediction>, Box<dyn Error>> {
    let mut predictions = Vec::with_capacity(rows.len() - PREFIX);
    for (i, row) in rows.iter().enumerate().skip(PREFIX) {
        let features = scaler.apply(row, i)?;
        let prediction = model
            .predict(state, &features)
            .map_err(|error| -> Box<dyn Error> {
                format!(
                    "frozen prediction failed at row {i} (date {}): {error:?}",
                    row.ordinal_date
                )
                .into()
            })?;
        predictions.push(Prediction {
            ordinal_date: row.ordinal_date,
            actual: row.target,
            prediction,
            error: f64::from(row.target) - f64::from(prediction),
        });
    }
    Ok(predictions)
}

/// Adaptive (prequential) evaluation: predict, then update. Before each update,
/// the example-level `eta * ||x||^2 < 1` guard is checked on the actual
/// transformed `f32` vector. On failure the run stops with row context, the last
/// committed state is preserved, and execution is reported as incomplete.
pub(crate) fn evaluate_adaptive(
    model: &Sgd,
    state: &SgdState,
    rows: &[Row],
    scaler: &Standardizer,
) -> Result<Vec<Prediction>, Box<dyn Error>> {
    let mut state = state.clone();
    let mut predictions = Vec::with_capacity(rows.len() - PREFIX);
    for (i, row) in rows.iter().enumerate().skip(PREFIX) {
        let features = scaler.apply(row, i)?;
        let prediction = model
            .predict(&state, &features)
            .map_err(|error| -> Box<dyn Error> {
                format!(
                    "adaptive prediction failed at row {i} (date {}): {error:?}",
                    row.ordinal_date
                )
                .into()
            })?;
        predictions.push(Prediction {
            ordinal_date: row.ordinal_date,
            actual: row.target,
            prediction,
            error: f64::from(row.target) - f64::from(prediction),
        });

        let product = f64::from(ETA) * squared_norm(&features);
        if !product.is_finite() || product >= 1.0 {
            return Err(format!(
                "adaptive-scoring guard failed at row {i} (date {}): eta*||x||^2 = {product} >= 1; \
                 incomplete execution; last committed state preserved; \
                 no row was skipped and eta was not retuned",
                row.ordinal_date
            )
            .into());
        }
        state = model
            .update(&state, &sample(features, row.target))
            .map_err(|error| -> Box<dyn Error> {
                format!(
                    "adaptive update failed at row {i} (date {}): {error:?}",
                    row.ordinal_date
                )
                .into()
            })?;
    }
    Ok(predictions)
}

pub(crate) fn metrics_of(predictions: &[Prediction]) -> Result<Metrics, Box<dyn Error>> {
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for prediction in predictions {
        sum_abs += prediction.error.abs();
        sum_sq += prediction.error * prediction.error;
    }
    let count = predictions.len() as f64;
    if count == 0.0 {
        return Err("no scored rows".into());
    }
    let metrics = Metrics {
        mae: sum_abs / count,
        rmse: (sum_sq / count).sqrt(),
    };
    if !metrics.mae.is_finite() || !metrics.rmse.is_finite() {
        return Err("non-finite metrics".into());
    }
    Ok(metrics)
}
