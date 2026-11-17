//! Offline real-data learning and prediction for SGD-trained linear regression
//! on the supplied `trump_approval.csv`.
//!
//! The example reads a caller-supplied path, selects columns by header name,
//! freezes a prefix-only standardization, learns the first [`PREFIX`] rows with
//! ordered single-observation updates, and then scores every remaining row
//! **before** revealing that row's target.
//!
//! Run with:
//!
//! ```text
//! cargo run --example sgd_regression --all-features -- \
//!   examples/data/trump_approval.csv
//! ```
//!
//! The example performs no network access; dataset acquisition is a separate
//! manual step. Compilation and synthetic parser tests do not substitute for
//! actually running this example against the supplied CSV.
//!
//! # Fixed contracts
//!
//! * Seven required columns, parsed by header name (order-independent):
//!   `ordinal_date`, `five_thirty_eight`, `gallup`, `ipsos`,
//!   `morning_consult`, `rasmussen`, `you_gov`.
//! * Model inputs are the five poll features plus an appended `1.0` intercept;
//!   the target is `five_thirty_eight`; `ordinal_date` is metadata only.
//! * Population mean/std are computed from the first `PREFIX` rows only, in
//!   `f64`, and frozen for the whole run. `M_prefix = max_i ||x_i||^2` is
//!   measured from the actual transformed `f32` prefix vectors and
//!   `eta * M_prefix < 1` is required.
//! * Learning: [`EPOCHS`] ordered passes over the prefix at [`ETA`].
//! * Scoring: predict first, then (adaptive) update. Before each adaptive
//!   update, `eta * ||x||^2 < 1` is checked on the actual transformed `f32`
//!   vector; a failure stops with row context, preserves the last committed
//!   state, and reports incomplete execution. Rows are never skipped and `eta`
//!   is never retuned. This guard is example-level only: `Sgd` accepts every
//!   finite positive learning rate.
//!
//! # Scope and honesty
//!
//! The target `five_thirty_eight` is an aggregate of the same polling series
//! used as features, so this reconstructs an **aggregate approval estimate from
//! polling features**. It is not political forecasting, not a River benchmark
//! reproduction, and not a production-readiness claim. A previous-target
//! baseline may win; results are reported honestly.

use std::process::ExitCode;

#[path = "sgd_regression/reporting.rs"]
mod reporting;
#[path = "sgd_regression/training.rs"]
mod training;
#[path = "../../../../../examples/support/trump.rs"]
mod trump;

#[cfg(test)]
#[path = "../../../../../benches/common/fixture_rng.rs"]
mod fixture_rng;

/// Fixed learning rate for this example's standardized prefix.
const ETA: f32 = 0.01;
/// Ordered passes over the learning prefix.
const EPOCHS: usize = 50;
/// Recorded snapshot checksum (provenance only; never enforced).
const EXPECTED_SHA256: &str = "aef188d04b606270e9590013282e2ede3f5f92edae1106d85d3defa483189cb0";

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!(
                "sgd_regression: missing CSV path; usage: \
                 cargo run --example sgd_regression --all-features -- <csv>"
            );
            return ExitCode::FAILURE;
        }
    };
    match reporting::run_with_path(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sgd_regression: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::reporting::run_with_path;
    use super::trump::PREFIX;

    use super::fixture_rng::Lcg64;

    const HEADER: &str =
        "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov";

    /// Deterministic fixture row in original units.
    struct Fixture {
        date: i64,
        features: [f64; 5],
        target: f64,
    }

    fn fixtures(count: usize, seed: u64) -> Vec<Fixture> {
        let mut rng = Lcg64::new(seed);
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
            "seqvex_sgd_regression_{name}_{}.csv",
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
    fn adaptive_guard_stops_with_row_context() {
        // Tiny nonzero prefix scale makes a moderate scoring outlier transform to
        // a very large standardized norm, so eta*||x||^2 >= 1 before its update.
        let mut rows = fixtures(405, 0xabcd_abcd_abcd_abcd);
        for (i, row) in rows.iter_mut().enumerate().take(PREFIX) {
            row.features[0] = if i % 2 == 0 { 0.0 } else { 1.0e-6 };
        }
        rows[PREFIX].features[0] = 2.0e-4;
        let error = run(&to_csv(&rows), "adaptive_guard").expect_err("guard must stop");
        assert!(error.contains("adaptive-scoring guard failed"), "{error}");
        assert!(error.contains("incomplete execution"), "{error}");
        assert!(error.contains("last committed state preserved"), "{error}");
        assert!(error.contains("eta was not retuned"), "{error}");
    }
}
