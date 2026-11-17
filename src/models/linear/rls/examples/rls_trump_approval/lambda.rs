use std::error::Error;

use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::Rls;

use super::sample;
use super::trump::{Metrics, PREFIX, Prediction, Row, Standardizer};

/// Outcome of one lambda case.
pub(crate) struct LambdaRun {
    pub(crate) weights_at_boundary: Vec<f32>,
    pub(crate) predictions: Vec<Prediction>,
    pub(crate) metrics: Metrics,
}

/// Learns the prefix and scores every remaining row prequentially for one lambda.
pub(crate) fn run_lambda(
    model: &Rls,
    rows: &[Row],
    scaler: &Standardizer,
    lambda: f32,
) -> Result<LambdaRun, Box<dyn Error>> {
    let mut state = model.initial_state();

    for (i, row) in rows.iter().enumerate().take(PREFIX) {
        let features = scaler.apply(row, i)?;
        let observation = sample(features, row.target);
        state = model.update(&state, &observation).map_err(|error| {
            format!(
                "lambda={lambda}: learn update failed at row {i} (date {}): {error:?}",
                row.ordinal_date
            )
        })?;
    }
    let weights_at_boundary = state.w.as_slice().to_vec();

    let mut predictions = Vec::with_capacity(rows.len() - PREFIX);
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (i, row) in rows.iter().enumerate().skip(PREFIX) {
        let features = scaler.apply(row, i)?;

        // Predict from the previously committed state, before revealing the target.
        let prediction = model.predict_checked(&state, &features).map_err(|error| {
            format!(
                "lambda={lambda}: checked prediction failed at row {i} (date {}): {error:?}",
                row.ordinal_date
            )
        })?;
        let actual = row.target;
        let error = f64::from(actual) - f64::from(prediction);
        sum_abs += error.abs();
        sum_sq += error * error;
        predictions.push(Prediction {
            ordinal_date: row.ordinal_date,
            actual,
            prediction,
            error,
        });

        // Commit the next state only after the update succeeds.
        let observation = sample(features, actual);
        state = model.update(&state, &observation).map_err(|error| {
            format!(
                "lambda={lambda}: scoring update failed at row {i} (date {}): {error:?}",
                row.ordinal_date
            )
        })?;
    }

    let count = predictions.len() as f64;
    let metrics = Metrics {
        mae: sum_abs / count,
        rmse: (sum_sq / count).sqrt(),
    };
    if !metrics.mae.is_finite() || !metrics.rmse.is_finite() {
        return Err(format!("lambda={lambda}: non-finite metrics").into());
    }
    Ok(LambdaRun {
        weights_at_boundary,
        predictions,
        metrics,
    })
}
