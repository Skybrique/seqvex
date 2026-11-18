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
//!   examples/data/trump_approval.csv
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

use std::process::ExitCode;

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::RlsSample;

#[path = "rls_trump_approval/lambda.rs"]
mod lambda;
#[path = "rls_trump_approval/real_data.rs"]
mod real_data;
#[path = "../../../../../examples/support/trump.rs"]
mod trump;

#[cfg(test)]
#[path = "../../../../../benches/common/fixture_rng.rs"]
mod fixture_rng;

#[cfg(test)]
#[path = "../../../../../examples/support/trump_fixtures.rs"]
mod trump_fixtures;

/// Initial covariance scale.
const DELTA: f32 = 1000.0;
/// Forgetting-factor cases, evaluated independently.
const LAMBDAS: [f32; 2] = [1.0, 0.95];

fn sample(features: Vector, target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample { features, target })
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
    match real_data::run_with_path(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rls_trump_approval: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::path::PathBuf;

    use seqvex::foundation::numerical::Vector;
    use seqvex::foundation::observation::Observation;
    use seqvex::foundation::state::StateModel;
    use seqvex::models::linear::rls::{Rls, RlsBatchError, RlsError, RlsSample, RlsState};

    use super::real_data::run_with_path;
    use super::trump::{DIMENSION, PREFIX, Row, Standardizer, read_dataset_with_hash};
    use super::{DELTA, LAMBDAS, sample};

    use super::trump_fixtures::{fixtures, to_csv};

    /// Bounded chunk size for the real-prefix equivalence check.
    const BOUNDED_CHUNK: usize = 50;
    /// Caller prediction-group limit for the frozen-prediction check.
    const MAX_HELD_OUT_PREDICTIONS: usize = 50;
    /// Optional dataset path override for the opt-in real-data tests.
    const DATASET_ENV: &str = "SEQVEX_TRUMP_CSV";
    /// Default dataset path, relative to the crate root.
    const DEFAULT_DATASET: &str = "examples/data/trump_approval.csv";

    fn real_dataset_path() -> PathBuf {
        std::env::var_os(DATASET_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DATASET))
    }

    /// Compares state shape and every weight/covariance entry bitwise.
    fn compare_state_bits(
        lambda: f32,
        committed_rows: usize,
        repeated: &RlsState,
        bounded: &RlsState,
    ) -> Result<(), Box<dyn Error>> {
        let dim = repeated.w.len();
        if repeated.p.rows() != dim
            || repeated.p.cols() != dim
            || bounded.w.len() != dim
            || bounded.p.rows() != dim
            || bounded.p.cols() != dim
        {
            return Err(format!(
                "lambda={lambda}, after {committed_rows} committed rows: state shape mismatch \
                 (repeated w={} P={}x{}; bounded w={} P={}x{})",
                dim,
                repeated.p.rows(),
                repeated.p.cols(),
                bounded.w.len(),
                bounded.p.rows(),
                bounded.p.cols()
            )
            .into());
        }
        for j in 0..dim {
            let a = repeated.w.as_slice()[j];
            let b = bounded.w.as_slice()[j];
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "bounded-learning mismatch: lambda={lambda}, after {committed_rows} committed \
                     rows, first differing entry w[{j}]: repeated=0x{:08x} bounded=0x{:08x} \
                     (repeated={a:?} bounded={b:?})",
                    a.to_bits(),
                    b.to_bits()
                )
                .into());
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                let a = repeated.p.get(i, j).expect("in-bounds P entry");
                let b = bounded.p.get(i, j).expect("in-bounds P entry");
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "bounded-learning mismatch: lambda={lambda}, after {committed_rows} \
                         committed rows, first differing entry P[{i}][{j}]: repeated=0x{:08x} \
                         bounded=0x{:08x} (repeated={a:?} bounded={b:?})",
                        a.to_bits(),
                        b.to_bits()
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn run_bounded_lambda_learning(
        lambda: f32,
        observations: &[Observation<RlsSample>],
    ) -> Result<(), Box<dyn Error>> {
        let model = Rls::new(DIMENSION, lambda, DELTA)
            .map_err(|error| format!("lambda={lambda}: Rls::new failed: {error:?}"))?;
        let initial = model.initial_state();
        let mut repeated = initial.clone();
        let mut bounded = initial;
        compare_state_bits(lambda, 0, &repeated, &bounded)?;

        for (chunk_index, chunk) in observations.chunks(BOUNDED_CHUNK).enumerate() {
            for (offset, observation) in chunk.iter().enumerate() {
                let row = chunk_index * BOUNDED_CHUNK + offset + 1;
                repeated = model.update(&repeated, observation).map_err(|error| {
                    format!(
                        "lambda={lambda}: repeated single-row update failed at training row \
                         {row}: {error:?}"
                    )
                })?;
            }
            bounded = model
                .process_batch_reference(bounded, chunk, BOUNDED_CHUNK)
                .map_err(|error| match error {
                    RlsBatchError::Transition {
                        failed_index,
                        error,
                        ..
                    } => format!(
                        "lambda={lambda}: bounded transition failed at training row {}, after {} \
                         committed rows: {error:?}",
                        chunk_index * BOUNDED_CHUNK + failed_index + 1,
                        chunk_index * BOUNDED_CHUNK + failed_index
                    ),
                    RlsBatchError::BatchTooLarge { max, actual, .. } => format!(
                        "lambda={lambda}: bounded chunk exceeded max_batch (max={max} \
                         actual={actual})"
                    ),
                })?;

            let committed_rows = (chunk_index + 1) * BOUNDED_CHUNK;
            compare_state_bits(lambda, committed_rows, &repeated, &bounded)?;
            eprintln!("lambda={lambda}: rows {committed_rows} match bitwise (w and P)");
        }
        Ok(())
    }

    fn run_bounded_learning_check(path: &str) -> Result<(), Box<dyn Error>> {
        let (rows, _) = read_dataset_with_hash(path)?;
        let scaler = Standardizer::from_prefix(&rows)?;
        let observations: Vec<Observation<RlsSample>> = rows[..PREFIX]
            .iter()
            .enumerate()
            .map(|(index, row)| {
                scaler
                    .apply(row, index)
                    .map(|features| sample(features, row.target))
            })
            .collect::<Result<_, _>>()?;

        eprintln!(
            "bounded-learning check: path={path} prefix={PREFIX} chunk={BOUNDED_CHUNK} \
             lambdas={LAMBDAS:?}"
        );
        for &lambda in &LAMBDAS {
            run_bounded_lambda_learning(lambda, &observations)?;
        }
        eprintln!(
            "bounded-learning check passed: {} lambdas x {} boundaries",
            LAMBDAS.len(),
            PREFIX / BOUNDED_CHUNK
        );
        Ok(())
    }

    #[derive(Debug, Clone)]
    struct HeldOutRow {
        csv_row: usize,
        ordinal_date: i64,
        features: Vector,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct HeldOutPrediction {
        csv_row: usize,
        ordinal_date: i64,
        value: f32,
    }

    #[derive(Debug, PartialEq)]
    enum GroupPredictionError {
        TooLarge {
            group_index: usize,
            max: usize,
            actual: usize,
            group_start_csv_row: usize,
            group_start_date: i64,
        },
        Prediction {
            group_index: usize,
            relative_index: usize,
            csv_row: usize,
            ordinal_date: i64,
            error: RlsError,
        },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PredictionPath {
        Single,
        Grouped,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum PredictionComparisonError {
        Count {
            path: PredictionPath,
            expected: usize,
            actual: usize,
            index: usize,
            expected_identity: Option<(usize, i64)>,
            actual_identity: Option<(usize, i64)>,
        },
        Identity {
            path: PredictionPath,
            index: usize,
            expected: (usize, i64),
            actual: (usize, i64),
        },
        Bits {
            index: usize,
            csv_row: usize,
            ordinal_date: i64,
            group_index: usize,
            relative_index: usize,
            single_bits: u32,
            grouped_bits: u32,
        },
    }

    fn held_out_row(csv_row: usize, ordinal_date: i64) -> HeldOutRow {
        HeldOutRow {
            csv_row,
            ordinal_date,
            features: Vector::from_slice(&[0.0]),
        }
    }

    fn collect_group_checked<F>(
        rows: &[HeldOutRow],
        max_predictions: usize,
        group_index: usize,
        mut predict: F,
    ) -> Result<Vec<HeldOutPrediction>, GroupPredictionError>
    where
        F: FnMut(&HeldOutRow) -> Result<f32, RlsError>,
    {
        if rows.len() > max_predictions {
            let first = &rows[0];
            return Err(GroupPredictionError::TooLarge {
                group_index,
                max: max_predictions,
                actual: rows.len(),
                group_start_csv_row: first.csv_row,
                group_start_date: first.ordinal_date,
            });
        }

        let mut outputs = Vec::with_capacity(rows.len());
        for (relative_index, row) in rows.iter().enumerate() {
            match predict(row) {
                Ok(value) => outputs.push(HeldOutPrediction {
                    csv_row: row.csv_row,
                    ordinal_date: row.ordinal_date,
                    value,
                }),
                Err(error) => {
                    return Err(GroupPredictionError::Prediction {
                        group_index,
                        relative_index,
                        csv_row: row.csv_row,
                        ordinal_date: row.ordinal_date,
                        error,
                    });
                }
            }
        }
        Ok(outputs)
    }

    fn verify_path(
        expected: &[HeldOutRow],
        actual: &[HeldOutPrediction],
        path: PredictionPath,
    ) -> Result<(), PredictionComparisonError> {
        if actual.len() != expected.len() {
            let first_divergence =
                expected
                    .iter()
                    .zip(actual.iter())
                    .position(|(expected, actual)| {
                        (expected.csv_row, expected.ordinal_date)
                            != (actual.csv_row, actual.ordinal_date)
                    });
            let index = first_divergence.unwrap_or_else(|| expected.len().min(actual.len()));
            return Err(PredictionComparisonError::Count {
                path,
                expected: expected.len(),
                actual: actual.len(),
                index,
                expected_identity: expected
                    .get(index)
                    .map(|row| (row.csv_row, row.ordinal_date)),
                actual_identity: actual
                    .get(index)
                    .map(|prediction| (prediction.csv_row, prediction.ordinal_date)),
            });
        }

        for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
            let expected_identity = (expected.csv_row, expected.ordinal_date);
            let actual_identity = (actual.csv_row, actual.ordinal_date);
            if expected_identity != actual_identity {
                return Err(PredictionComparisonError::Identity {
                    path,
                    index,
                    expected: expected_identity,
                    actual: actual_identity,
                });
            }
        }
        Ok(())
    }

    fn verify_frozen_predictions(
        expected: &[HeldOutRow],
        single: &[HeldOutPrediction],
        grouped: &[HeldOutPrediction],
    ) -> Result<(), PredictionComparisonError> {
        verify_path(expected, single, PredictionPath::Single)?;
        verify_path(expected, grouped, PredictionPath::Grouped)?;

        for (index, (single, grouped)) in single.iter().zip(grouped.iter()).enumerate() {
            let single_bits = single.value.to_bits();
            let grouped_bits = grouped.value.to_bits();
            if single_bits != grouped_bits {
                return Err(PredictionComparisonError::Bits {
                    index,
                    csv_row: single.csv_row,
                    ordinal_date: single.ordinal_date,
                    group_index: index / MAX_HELD_OUT_PREDICTIONS,
                    relative_index: index % MAX_HELD_OUT_PREDICTIONS,
                    single_bits,
                    grouped_bits,
                });
            }
        }
        Ok(())
    }

    fn format_comparison_error(lambda: f32, error: PredictionComparisonError) -> String {
        match error {
            PredictionComparisonError::Bits {
                index,
                csv_row,
                ordinal_date,
                group_index,
                relative_index,
                single_bits,
                grouped_bits,
            } => format!(
                "lambda={lambda}: frozen-prediction bit mismatch at output index {index}, \
                 csv row {csv_row} (date {ordinal_date}), group {group_index} relative \
                 {relative_index}: single=0x{single_bits:08x} ({:?}) grouped=0x{grouped_bits:08x} \
                 ({:?})",
                f32::from_bits(single_bits),
                f32::from_bits(grouped_bits)
            ),
            other => format!("lambda={lambda}: frozen-prediction comparison failed: {other:?}"),
        }
    }

    fn prepare_held_out(
        rows: &[Row],
        scaler: &Standardizer,
    ) -> Result<Vec<HeldOutRow>, Box<dyn Error>> {
        rows.iter()
            .enumerate()
            .skip(PREFIX)
            .map(|(csv_row, row)| {
                scaler.apply(row, csv_row).map(|features| HeldOutRow {
                    csv_row,
                    ordinal_date: row.ordinal_date,
                    features,
                })
            })
            .collect()
    }

    fn learn_prefix_state(
        model: &Rls,
        rows: &[Row],
        scaler: &Standardizer,
        lambda: f32,
    ) -> Result<RlsState, Box<dyn Error>> {
        let mut state = model.initial_state();
        for (csv_row, row) in rows[..PREFIX].iter().enumerate() {
            let features = scaler.apply(row, csv_row)?;
            let observation = sample(features, row.target);
            state = model.update(&state, &observation).map_err(|error| {
                format!(
                    "lambda={lambda}: prefix learning failed at csv row {csv_row} \
                     (date {}): {error:?}",
                    row.ordinal_date
                )
            })?;
        }
        Ok(state)
    }

    fn run_frozen_prediction_check(path: &str) -> Result<(), Box<dyn Error>> {
        let (rows, _) = read_dataset_with_hash(path)?;
        let scaler = Standardizer::from_prefix(&rows)?;
        let held_out = prepare_held_out(&rows, &scaler)?;
        if held_out.is_empty() {
            return Err("frozen-prediction check: held-out set is empty".into());
        }

        let group_lengths: Vec<usize> = held_out
            .chunks(MAX_HELD_OUT_PREDICTIONS)
            .map(|group| group.len())
            .collect();
        if group_lengths.iter().sum::<usize>() != held_out.len()
            || group_lengths.len() != held_out.len().div_ceil(MAX_HELD_OUT_PREDICTIONS)
            || group_lengths
                .iter()
                .any(|&length| length == 0 || length > MAX_HELD_OUT_PREDICTIONS)
        {
            return Err(format!(
                "frozen-prediction check: group-size invariant failed: held_out={} \
                 groups={group_lengths:?}",
                held_out.len()
            )
            .into());
        }

        eprintln!(
            "frozen-prediction check: path={path} rows={} prefix={PREFIX} held_out={} \
             groups={} limit={MAX_HELD_OUT_PREDICTIONS} lambdas={LAMBDAS:?}",
            rows.len(),
            held_out.len(),
            group_lengths.len()
        );

        for &lambda in &LAMBDAS {
            let model = Rls::new(DIMENSION, lambda, DELTA)
                .map_err(|error| format!("lambda={lambda}: Rls::new failed: {error:?}"))?;
            let state = learn_prefix_state(&model, &rows, &scaler, lambda)?;

            let mut single = Vec::with_capacity(held_out.len());
            for row in &held_out {
                let value = model
                    .predict_checked(&state, &row.features)
                    .map_err(|error| {
                        format!(
                            "lambda={lambda}: single prediction failed at csv row {} \
                             (date {}): {error:?}",
                            row.csv_row, row.ordinal_date
                        )
                    })?;
                single.push(HeldOutPrediction {
                    csv_row: row.csv_row,
                    ordinal_date: row.ordinal_date,
                    value,
                });
            }

            let mut grouped = Vec::with_capacity(held_out.len());
            for (group_index, group) in held_out.chunks(MAX_HELD_OUT_PREDICTIONS).enumerate() {
                let records =
                    collect_group_checked(group, MAX_HELD_OUT_PREDICTIONS, group_index, |row| {
                        model.predict_checked(&state, &row.features)
                    })
                    .map_err(|error| {
                        format!("lambda={lambda}: grouped prediction failed: {error:?}")
                    })?;
                grouped.extend(records);
            }

            verify_frozen_predictions(&held_out, &single, &grouped)
                .map_err(|error| format_comparison_error(lambda, error))?;

            eprintln!(
                "lambda={lambda}: frozen predictions match: rows={} held_out={} groups={}",
                rows.len(),
                held_out.len(),
                group_lengths.len()
            );
        }
        eprintln!("frozen-prediction check passed: {} lambdas", LAMBDAS.len());
        Ok(())
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

    #[test]
    fn grouped_callback_bound_and_first_error_are_observable() {
        let rows = vec![
            held_out_row(10, 1_000),
            held_out_row(11, 1_001),
            held_out_row(12, 1_002),
            held_out_row(13, 1_003),
        ];

        let mut calls = Vec::new();
        let result = collect_group_checked(&rows[..3], 2, 0, |row| {
            calls.push(row.csv_row);
            Ok(0.0)
        });
        assert!(matches!(
            result,
            Err(GroupPredictionError::TooLarge {
                group_index: 0,
                max: 2,
                actual: 3,
                group_start_csv_row: 10,
                group_start_date: 1_000,
            })
        ));
        assert!(calls.is_empty());

        let mut calls = Vec::new();
        let result = collect_group_checked(&rows, 4, 0, |row| {
            calls.push(row.csv_row);
            if row.csv_row == 12 {
                Err(RlsError::NonFiniteInput)
            } else {
                Ok(0.0)
            }
        });
        assert_eq!(calls, vec![10, 11, 12]);
        assert!(matches!(
            result,
            Err(GroupPredictionError::Prediction {
                group_index: 0,
                relative_index: 2,
                csv_row: 12,
                ordinal_date: 1_002,
                error: RlsError::NonFiniteInput,
            })
        ));

        let empty: Vec<HeldOutRow> = Vec::new();
        let mut calls = Vec::new();
        let result = collect_group_checked(&empty, 0, 0, |row| {
            calls.push(row.csv_row);
            Ok(0.0)
        });
        assert!(matches!(result, Ok(ref records) if records.is_empty()));
        assert!(calls.is_empty());

        let mut calls = Vec::new();
        let result = collect_group_checked(&rows[..1], 0, 0, |row| {
            calls.push(row.csv_row);
            Ok(0.0)
        });
        assert!(matches!(
            result,
            Err(GroupPredictionError::TooLarge {
                group_index: 0,
                max: 0,
                actual: 1,
                group_start_csv_row: 10,
                group_start_date: 1_000,
            })
        ));
        assert!(calls.is_empty());

        let mut calls = Vec::new();
        let result = collect_group_checked(&empty, 5, 0, |row| {
            calls.push(row.csv_row);
            Ok(0.0)
        });
        assert!(matches!(result, Ok(ref records) if records.is_empty()));
        assert!(calls.is_empty());
    }

    #[test]
    fn equal_bit_reordering_and_comparison_mismatches_are_detected() {
        let rows = vec![
            held_out_row(10, 1_000),
            held_out_row(11, 1_001),
            held_out_row(12, 1_002),
        ];

        let mut calls = Vec::new();
        let grouped = collect_group_checked(&rows, 3, 0, |row| {
            calls.push(row.csv_row);
            Ok(0.0)
        })
        .expect("grouped predictions");
        assert_eq!(calls, vec![10, 11, 12]);
        assert_eq!(
            grouped.iter().map(|row| row.csv_row).collect::<Vec<_>>(),
            vec![10, 11, 12]
        );
        assert!(
            grouped
                .iter()
                .all(|row| row.value.to_bits() == 0.0f32.to_bits())
        );

        let single: Vec<HeldOutPrediction> = rows
            .iter()
            .map(|row| HeldOutPrediction {
                csv_row: row.csv_row,
                ordinal_date: row.ordinal_date,
                value: 0.0,
            })
            .collect();
        verify_frozen_predictions(&rows, &single, &grouped).expect("matching predictions");

        let mut reordered = grouped.clone();
        reordered.swap(0, 1);
        assert!(matches!(
            verify_frozen_predictions(&rows, &single, &reordered),
            Err(PredictionComparisonError::Identity {
                path: PredictionPath::Grouped,
                index: 0,
                expected: (10, 1_000),
                actual: (11, 1_001),
            })
        ));

        let missing = grouped[..2].to_vec();
        assert!(matches!(
            verify_frozen_predictions(&rows, &single, &missing),
            Err(PredictionComparisonError::Count {
                path: PredictionPath::Grouped,
                expected: 3,
                actual: 2,
                index: 2,
                expected_identity: Some((12, 1_002)),
                actual_identity: None,
            })
        ));

        let mut extra = grouped.clone();
        extra.push(HeldOutPrediction {
            csv_row: 13,
            ordinal_date: 1_003,
            value: 0.0,
        });
        assert!(matches!(
            verify_frozen_predictions(&rows, &single, &extra),
            Err(PredictionComparisonError::Count {
                path: PredictionPath::Grouped,
                expected: 3,
                actual: 4,
                index: 3,
                expected_identity: None,
                actual_identity: Some((13, 1_003)),
            })
        ));

        let mut signed_single = single.clone();
        signed_single[0].value = -0.0;
        match verify_frozen_predictions(&rows, &signed_single, &grouped) {
            Err(PredictionComparisonError::Bits {
                single_bits,
                grouped_bits,
                ..
            }) => {
                assert_eq!(single_bits, 0x8000_0000);
                assert_eq!(grouped_bits, 0x0000_0000);
                assert!(f32::from_bits(single_bits).is_sign_negative());
                assert!(!f32::from_bits(grouped_bits).is_sign_negative());
            }
            other => panic!("expected signed-zero bit mismatch, got {other:?}"),
        }
    }

    #[test]
    #[ignore = "requires the unpublished trump_approval.csv; set SEQVEX_TRUMP_CSV or provide examples/data/trump_approval.csv"]
    fn frozen_prediction_matches_between_single_and_grouped_on_real_holdout() {
        let path = real_dataset_path();
        if !path.is_file() {
            panic!(
                "frozen-prediction check requires the real dataset; {} is not a file. Set \
                 {DATASET_ENV} or place it at {DEFAULT_DATASET}.",
                path.display()
            );
        }
        run_frozen_prediction_check(path.to_str().expect("dataset path must be UTF-8"))
            .unwrap_or_else(|error| panic!("frozen-prediction check failed: {error}"));
    }

    #[test]
    #[ignore = "requires the unpublished trump_approval.csv; set SEQVEX_TRUMP_CSV or provide examples/data/trump_approval.csv"]
    fn bounded_reference_matches_repeated_updates_on_real_prefix() {
        let path = real_dataset_path();
        if !path.is_file() {
            panic!(
                "bounded-learning check requires the real dataset; {} is not a file. Set \
                 {DATASET_ENV} or place it at {DEFAULT_DATASET}.",
                path.display()
            );
        }
        run_bounded_learning_check(path.to_str().expect("dataset path must be UTF-8"))
            .unwrap_or_else(|error| panic!("bounded-learning real-prefix check failed: {error}"));
    }
}
