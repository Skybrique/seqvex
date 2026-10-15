//! Bounded real-data validation of the **existing** RLS implementation on the
//! supplied `trump_approval.csv`.
//!
//! This example demonstrates that the accepted RLS slice learns from a real CSV
//! and produces *checked* predictions under a chronological (prequential)
//! evaluation protocol. It reads a caller-supplied path, selects columns by
//! header name, freezes a prefix-only standardization, learns a fixed learning
//! prefix through public `update` calls, and scores every remaining row with
//! `Rls::predict_checked` *before* revealing that row's target.
//!
//! Run with:
//!
//! ```text
//! cargo run --example rls_trump_approval --all-features -- \
//!   src/models/linear/rls/examples/data/trump_approval.csv
//! ```
//!
//! The example performs no network access; dataset acquisition is a separate,
//! manual step.
//!
//! # Scope and honesty
//!
//! The target `five_thirty_eight` is itself an aggregate of the same polling
//! series used as features, so this reconstructs an **aggregate approval
//! estimate from polling features**. It is not political forecasting skill, not
//! an exact reproduction of a River model benchmark, and not a production
//! readiness claim. This is River's records with our declared feature selection.
//!
//! # Fixed contracts
//!
//! * Dimension 6: `[gallup, ipsos, morning_consult, rasmussen, you_gov, 1.0]`;
//!   the intercept is generated, never read from CSV. Target `five_thirty_eight`.
//!   `ordinal_date` is metadata only.
//! * Standardization uses the first 400 rows only, in `f64`, frozen for the whole
//!   run: `mean = Σx/400`, `variance = Σ(x−mean)²/400`, `scale = sqrt(variance)`.
//!   Standardized values are computed in `f64`, converted to `f32`, and
//!   re-checked for finiteness. This is a preprocessing/conversion failure when
//!   it fails, reported before any RLS call.
//! * Learning prefix `0..400`; scoring `400..n`; `delta = 1000`; lambdas
//!   `1.0` and `0.95`. Each lambda starts from its own `initial_state()`.
//!
//! No known targets or fitted coefficients are installed into state; no fallback
//! output, skipping, retry, reset, guard bypass, or parameter tuning.

use std::error::Error;
use std::process::ExitCode;

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, RlsSample};

/// Date column (metadata only; never a model input).
const DATE_COLUMN: &str = "ordinal_date";
/// Target column.
const TARGET_COLUMN: &str = "five_thirty_eight";
/// Poll feature columns, in fixed model order.
const FEATURE_COLUMNS: [&str; 5] = ["gallup", "ipsos", "morning_consult", "rasmussen", "you_gov"];
/// Model dimension: five poll features plus a generated intercept.
const DIMENSION: usize = 6;
/// Learning-prefix length in data rows.
const PREFIX: usize = 400;
/// Initial covariance scale.
const DELTA: f32 = 1000.0;
/// Forgetting-factor cases, evaluated independently.
const LAMBDAS: [f32; 2] = [1.0, 0.95];

/// One parsed, validated observation in original units.
struct Row {
    ordinal_date: i64,
    features: [f32; 5],
    target: f32,
}

/// Frozen prefix-only standardization of the five poll features.
struct Standardizer {
    mean: [f64; 5],
    scale: [f64; 5],
}

impl Standardizer {
    /// Computes the frozen transform from the first [`PREFIX`] rows only.
    fn from_prefix(rows: &[Row]) -> Result<Self, Box<dyn Error>> {
        let mut mean = [0.0f64; 5];
        for row in &rows[..PREFIX] {
            for (j, m) in mean.iter_mut().enumerate() {
                *m += f64::from(row.features[j]);
            }
        }
        for m in &mut mean {
            *m /= PREFIX as f64;
        }

        let mut variance = [0.0f64; 5];
        for row in &rows[..PREFIX] {
            for (j, m) in mean.iter().enumerate() {
                let delta = f64::from(row.features[j]) - m;
                variance[j] += delta * delta;
            }
        }

        let mut scale = [0.0f64; 5];
        for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
            scale[j] = (variance[j] / PREFIX as f64).sqrt();
            if !mean[j].is_finite() {
                return Err(format!(
                    "preprocessing: column {name}: non-finite prefix mean {}",
                    mean[j]
                )
                .into());
            }
            if !scale[j].is_finite() || scale[j] == 0.0 {
                return Err(format!(
                    "preprocessing: column {name}: non-finite or zero prefix scale {} \
                     (no epsilon or 1.0 substitution)",
                    scale[j]
                )
                .into());
            }
        }
        Ok(Self { mean, scale })
    }

    /// Standardizes one row with the frozen transform, re-checking finiteness
    /// after `f64` -> `f32` conversion. The intercept stays exactly `1.0`.
    fn apply(&self, row: &Row, row_index: usize) -> Result<Vector, Box<dyn Error>> {
        let mut values = [0.0f32; DIMENSION];
        for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
            let standardized = (f64::from(row.features[j]) - self.mean[j]) / self.scale[j];
            if !standardized.is_finite() {
                return Err(format!(
                    "preprocessing: row {row_index} (date {}): column {name}: \
                     non-finite standardized value {standardized}",
                    row.ordinal_date
                )
                .into());
            }
            let value = standardized as f32;
            if !value.is_finite() {
                return Err(format!(
                    "preprocessing: row {row_index} (date {}): column {name}: \
                     value {standardized} is non-finite after f32 conversion",
                    row.ordinal_date
                )
                .into());
            }
            values[j] = value;
        }
        values[5] = 1.0;
        Ok(Vector::from_slice(&values))
    }
}

/// Parsed metrics over a set of scored rows.
struct Metrics {
    mae: f64,
    rmse: f64,
}

/// One scored prediction recorded before its target was revealed to the model.
struct Prediction {
    ordinal_date: i64,
    actual: f32,
    prediction: f32,
    error: f64,
}

/// Outcome of one lambda case.
struct LambdaRun {
    weights_at_boundary: Vec<f32>,
    predictions: Vec<Prediction>,
    metrics: Metrics,
}

fn sample(features: Vector, target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample { features, target })
}

/// Reads and validates the dataset: exactly the seven required headers, each
/// once; finite numeric fields; strictly increasing `ordinal_date`; at least
/// `PREFIX + 1` rows. Values are never reordered, dropped, imputed, or clamped.
fn read_dataset(path: &str) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let mut date_idx: Option<usize> = None;
    let mut target_idx: Option<usize> = None;
    let mut feature_idx: [Option<usize>; 5] = [None; 5];
    for (index, header) in headers.iter().enumerate() {
        let name = header.trim();
        if name == DATE_COLUMN {
            if date_idx.replace(index).is_some() {
                return Err(format!("duplicate required header {DATE_COLUMN}").into());
            }
        } else if name == TARGET_COLUMN {
            if target_idx.replace(index).is_some() {
                return Err(format!("duplicate required header {TARGET_COLUMN}").into());
            }
        } else if let Some(j) = FEATURE_COLUMNS.iter().position(|column| *column == name) {
            if feature_idx[j].replace(index).is_some() {
                return Err(format!("duplicate required header {name}").into());
            }
        } else {
            return Err(format!("unexpected header {name}").into());
        }
    }

    let date_idx = date_idx.ok_or_else(|| -> Box<dyn Error> {
        format!("missing required header {DATE_COLUMN}").into()
    })?;
    let target_idx = target_idx.ok_or_else(|| -> Box<dyn Error> {
        format!("missing required header {TARGET_COLUMN}").into()
    })?;
    let mut feature_cols = [0usize; 5];
    for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
        feature_cols[j] = feature_idx[j].ok_or_else(|| -> Box<dyn Error> {
            format!("missing required header {name}").into()
        })?;
    }

    let mut rows: Vec<Row> = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        let record =
            record.map_err(|error| format!("row {row_index}: malformed CSV record: {error}"))?;

        let raw_date = record.get(date_idx).ok_or_else(|| -> Box<dyn Error> {
            format!("row {row_index}: missing field for {DATE_COLUMN}").into()
        })?;
        let ordinal_date: i64 = raw_date.trim().parse().map_err(|error| {
            format!(
                "row {row_index}: column {DATE_COLUMN}: invalid integer {}: {error}",
                raw_date.trim()
            )
        })?;

        let mut features = [0.0f32; 5];
        for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
            let raw = record
                .get(feature_cols[j])
                .ok_or_else(|| -> Box<dyn Error> {
                    format!("row {row_index}: missing field for {name}").into()
                })?;
            let value: f32 = raw.trim().parse().map_err(|error| {
                format!(
                    "row {row_index}: column {name}: invalid f32 {}: {error}",
                    raw.trim()
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "row {row_index}: column {name}: non-finite value {}",
                    raw.trim()
                )
                .into());
            }
            features[j] = value;
        }

        let raw_target = record.get(target_idx).ok_or_else(|| -> Box<dyn Error> {
            format!("row {row_index}: missing field for {TARGET_COLUMN}").into()
        })?;
        let target: f32 = raw_target.trim().parse().map_err(|error| {
            format!(
                "row {row_index}: column {TARGET_COLUMN}: invalid f32 {}: {error}",
                raw_target.trim()
            )
        })?;
        if !target.is_finite() {
            return Err(format!(
                "row {row_index}: column {TARGET_COLUMN}: non-finite value {}",
                raw_target.trim()
            )
            .into());
        }

        rows.push(Row {
            ordinal_date,
            features,
            target,
        });
    }

    if rows.len() < PREFIX + 1 {
        return Err(format!(
            "dataset has {} data rows; need at least {} for the {PREFIX}-row learning prefix",
            rows.len(),
            PREFIX + 1
        )
        .into());
    }
    for i in 1..rows.len() {
        if rows[i].ordinal_date <= rows[i - 1].ordinal_date {
            return Err(format!(
                "chronology violation at row {i}: ordinal_date {} is not strictly greater than previous {}",
                rows[i].ordinal_date,
                rows[i - 1].ordinal_date
            )
            .into());
        }
    }
    Ok(rows)
}

/// Learns the prefix and scores every remaining row prequentially for one lambda.
fn run_lambda(
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

/// Metrics for the three lambda-independent baselines over the scoring rows.
/// Historical baselines use only targets strictly preceding the current row.
fn baseline_metrics(rows: &[Row]) -> Result<[(&'static str, Metrics); 3], Box<dyn Error>> {
    let mut running_sum: f64 = rows[..PREFIX].iter().map(|row| f64::from(row.target)).sum();
    let mut running_count = PREFIX as f64;
    let mut expanding = (0.0f64, 0.0f64);
    let mut previous = (0.0f64, 0.0f64);
    let mut raw_feature_mean = (0.0f64, 0.0f64);

    let mut previous_target = f64::from(rows[PREFIX - 1].target);
    for row in rows.iter().skip(PREFIX) {
        let actual = f64::from(row.target);
        let expanding_prediction = running_sum / running_count;
        let previous_prediction = previous_target;
        let raw_prediction = row
            .features
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / 5.0;

        for (accumulator, prediction) in [
            (&mut expanding, expanding_prediction),
            (&mut previous, previous_prediction),
            (&mut raw_feature_mean, raw_prediction),
        ] {
            let error = actual - prediction;
            accumulator.0 += error.abs();
            accumulator.1 += error * error;
        }

        running_sum += actual;
        running_count += 1.0;
        previous_target = actual;
    }

    let count = (rows.len() - PREFIX) as f64;
    let finish = |(sum_abs, sum_sq): (f64, f64)| Metrics {
        mae: sum_abs / count,
        rmse: (sum_sq / count).sqrt(),
    };
    let result = [
        ("expanding target mean", finish(expanding)),
        ("previous target", finish(previous)),
        ("raw five-feature mean", finish(raw_feature_mean)),
    ];
    for (name, metrics) in &result {
        if !metrics.mae.is_finite() || !metrics.rmse.is_finite() {
            return Err(format!("baseline {name}: non-finite metrics").into());
        }
    }
    Ok(result)
}

fn format_weights(weights: &[f32]) -> String {
    weights
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Runs the full CSV workflow for a caller-supplied path.
fn run_with_path(path: &str) -> Result<(), Box<dyn Error>> {
    let rows = read_dataset(path)?;
    let scaler = Standardizer::from_prefix(&rows)?;
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

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!(
                "rls_trump_approval: missing CSV path; usage: \
                 cargo run --example rls_trump_approval --all-features -- <csv>"
            );
            return ExitCode::FAILURE;
        }
    };
    match run_with_path(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rls_trump_approval: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{PREFIX, run_with_path};

    const HEADER: &str =
        "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov";

    /// Deterministic fixture row in original units.
    struct Fixture {
        date: i64,
        features: [f64; 5],
        target: f64,
    }

    /// Small deterministic LCG so fixtures never depend on external randomness.
    struct Lcg(u64);

    impl Lcg {
        fn next_unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn fixtures(count: usize, seed: u64) -> Vec<Fixture> {
        let mut rng = Lcg(seed);
        (0..count)
            .map(|i| {
                let mut features = [0.0f64; 5];
                for value in &mut features {
                    *value = 30.0 + 20.0 * rng.next_unit();
                }
                let target =
                    1.0 + 2.0 * features[0] - 1.5 * features[1] + 0.5 * features[2] + features[3]
                        - features[4];
                Fixture {
                    date: 736_389 + i as i64,
                    features,
                    target,
                }
            })
            .collect()
    }

    fn to_csv(rows: &[Fixture]) -> String {
        let mut content = String::from(HEADER);
        content.push('\n');
        for row in rows {
            content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                row.date,
                row.target,
                row.features[0],
                row.features[1],
                row.features[2],
                row.features[3],
                row.features[4]
            ));
        }
        content
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "seqvex_rls_trump_{name}_{}.csv",
            std::process::id()
        ));
        path
    }

    fn write_temp(name: &str, content: &str) -> PathBuf {
        let path = temp_path(name);
        std::fs::write(&path, content).expect("write temp csv");
        path
    }

    fn run(content: &str, name: &str) -> Result<(), String> {
        let path = write_temp(name, content);
        let result = run_with_path(path.to_str().unwrap()).map_err(|error| error.to_string());
        let _ = std::fs::remove_file(path);
        result
    }

    #[test]
    fn valid_ingestion_and_evaluation_yields_finite_results() {
        run(&to_csv(&fixtures(405, 0x1234_5678_9abc_def0)), "valid").expect("valid fixture");
    }

    #[test]
    fn header_order_is_independent() {
        let rows = fixtures(405, 0x0bad_c0de_0bad_c0de);
        let mut content = String::from(
            "five_thirty_eight,gallup,ordinal_date,you_gov,ipsos,rasmussen,morning_consult\n",
        );
        for row in &rows {
            content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                row.target,
                row.features[0],
                row.date,
                row.features[4],
                row.features[1],
                row.features[3],
                row.features[2]
            ));
        }
        run(&content, "reordered").expect("reordered header fixture");
    }

    #[test]
    fn missing_header_is_rejected() {
        let content = "ordinal_date,five_thirty_eight,ipsos,morning_consult,rasmussen,you_gov\n\
                       736389,1.0,30,30,30,30\n";
        let error = run(content, "missing_header").expect_err("missing gallup must fail");
        assert!(error.contains("missing required header"), "{error}");
        assert!(error.contains("gallup"), "{error}");
    }

    #[test]
    fn duplicate_header_is_rejected() {
        let content = "ordinal_date,five_thirty_eight,gallup,gallup,ipsos,morning_consult,rasmussen,you_gov\n\
                       736389,1.0,30,31,30,30,30,30\n";
        let error = run(content, "duplicate_header").expect_err("duplicate gallup must fail");
        assert!(error.contains("duplicate required header"), "{error}");
        assert!(error.contains("gallup"), "{error}");
    }

    #[test]
    fn unexpected_header_is_rejected() {
        let content = "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov,extra\n\
             736389,1.0,30,30,30,30,30,0\n";
        let error = run(content, "unexpected_header").expect_err("extra header must fail");
        assert!(error.contains("unexpected header"), "{error}");
        assert!(error.contains("extra"), "{error}");
    }

    #[test]
    fn malformed_numeric_is_rejected() {
        let content = "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov\n\
                       736389,1.0,abc,30,30,30,30\n";
        let error = run(content, "malformed").expect_err("malformed gallup must fail");
        assert!(error.contains("invalid f32"), "{error}");
        assert!(error.contains("gallup"), "{error}");
    }

    #[test]
    fn non_finite_numeric_is_rejected() {
        let content = "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov\n\
                       736389,1.0,inf,30,30,30,30\n";
        let error = run(content, "non_finite").expect_err("non-finite gallup must fail");
        assert!(error.contains("non-finite value"), "{error}");
        assert!(error.contains("gallup"), "{error}");
    }

    #[test]
    fn duplicate_date_is_rejected() {
        let mut rows = fixtures(405, 0xdead_beef_dead_beef);
        rows[2].date = rows[1].date;
        let error = run(&to_csv(&rows), "duplicate_date").expect_err("duplicate date must fail");
        assert!(error.contains("chronology violation"), "{error}");
    }

    #[test]
    fn non_increasing_date_is_rejected() {
        let mut rows = fixtures(405, 0xfeed_face_feed_face);
        rows[3].date = rows[2].date - 1;
        let error = run(&to_csv(&rows), "non_increasing").expect_err("decreasing date must fail");
        assert!(error.contains("chronology violation"), "{error}");
    }

    #[test]
    fn zero_scale_is_rejected() {
        let mut rows = fixtures(405, 0x00c0_ffee_00c0_ffee);
        for row in &mut rows {
            row.features[0] = 42.0;
        }
        let error = run(&to_csv(&rows), "zero_scale").expect_err("zero scale must fail");
        assert!(error.contains("preprocessing"), "{error}");
        assert!(error.contains("zero prefix scale"), "{error}");
        assert!(error.contains("gallup"), "{error}");
    }

    #[test]
    fn insufficient_rows_is_rejected() {
        let error = run(
            &to_csv(&fixtures(400, 0x5151_5151_5151_5151)),
            "insufficient",
        )
        .expect_err("400 rows must fail");
        assert!(error.contains("need at least"), "{error}");
    }

    #[test]
    fn standardized_f32_conversion_overflow_is_rejected() {
        let mut rows = fixtures(401, 0x7f7f_7f7f_7f7f_7f7f);
        for (i, row) in rows.iter_mut().enumerate().take(PREFIX) {
            row.features[0] = if i % 2 == 0 { 0.0 } else { 1.0e-6 };
        }
        rows[PREFIX].features[0] = 3.0e38;
        let error = run(&to_csv(&rows), "conversion_overflow")
            .expect_err("standardized f32 overflow must fail");
        assert!(error.contains("f32 conversion"), "{error}");
        assert!(error.contains("gallup"), "{error}");
    }
}
