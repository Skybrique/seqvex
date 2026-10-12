//! End-to-end OLS workflow on the Salary dataset.
//!
//! borrowed training data → OLS fit → learned predictor → predictions and
//! diagnostics → held-out evaluation.
//!
//! Reads a caller-supplied CSV path, selects columns by header
//! (`YearsExperience` feature, `Salary` target; any index/`Unnamed` column is
//! ignored), applies a deterministic split, fits with an intercept, and reports
//! held-out RMSE against a training-mean baseline.
//!
//! Run with:
//!
//! ```text
//! cargo +1.85.1 run --example fit_salary -- examples/Salary_dataset.csv
//! ```
//!
//! The sample has no time column. The split below is deterministic but **not**
//! chronological, so this example demonstrates the workflow and arithmetic — it
//! does not establish temporal validity or general predictive superiority.

use std::error::Error;
use std::process::ExitCode;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    ArtifactExpectation, ArtifactLimits, CheckedLinearRegression, Expect, FitConfig, FitError,
    PublicationId, TrainedModel, TrainingMeta, fit,
};

/// Deterministic split: every fifth row by position is held out.
fn is_held_out(position: usize) -> bool {
    position % 5 == 4
}

fn observation(row: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(row))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!(
                "fit_salary: missing CSV path; usage: \
                 cargo +1.85.1 run --example fit_salary -- <csv>"
            );
            return ExitCode::FAILURE;
        }
    };
    match run_with_path(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fit_salary: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Runs the full CSV workflow for a caller-supplied path.
///
/// Validation and serving failures are returned as [`Err`] with row context;
/// non-finite parsed values and non-finite evaluation metrics never reach a
/// successful exit. The fitter itself is unchanged: it reads only selected
/// training rows, so value validation here is intentionally example-local.
fn run_with_path(path: &str) -> Result<(), Box<dyn Error>> {
    // 1. Parse by header; ignore index / Unnamed columns.
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let feature_col = headers
        .iter()
        .position(|h| h == "YearsExperience")
        .ok_or("CSV is missing the `YearsExperience` column")?;
    let target_col = headers
        .iter()
        .position(|h| h == "Salary")
        .ok_or("CSV is missing the `Salary` column")?;
    println!("columns: YearsExperience={feature_col}, Salary={target_col} (index/Unnamed ignored)");

    let mut features: Vec<Vector> = Vec::new();
    let mut targets: Vec<f32> = Vec::new();
    for (line, record) in reader.records().enumerate() {
        let record = record?;
        let experience: f32 = record
            .get(feature_col)
            .ok_or("short row")?
            .trim()
            .parse()
            .map_err(|e| format!("row {line}: invalid YearsExperience: {e}"))?;
        let salary: f32 = record
            .get(target_col)
            .ok_or("short row")?
            .trim()
            .parse()
            .map_err(|e| format!("row {line}: invalid Salary: {e}"))?;
        features.push(Vector::from_slice(&[experience]));
        targets.push(salary);
    }
    if features.len() != targets.len() || features.is_empty() {
        return Err("no rows parsed".into());
    }
    // Reject non-finite parsed values with row/column context, for **all** rows
    // (including held-out rows). This is example-local validation; the fitter
    // still inspects only selected training rows.
    for (row, (feature, &target)) in features.iter().zip(&targets).enumerate() {
        if !feature.as_slice()[0].is_finite() {
            return Err(format!("row {row}: non-finite YearsExperience").into());
        }
        if !target.is_finite() {
            return Err(format!("row {row}: non-finite Salary").into());
        }
    }
    println!("rows parsed: {}", features.len());

    // 2. Deterministic split (every fifth row held out).
    let train: Vec<usize> = (0..features.len()).filter(|i| !is_held_out(*i)).collect();
    let held: Vec<usize> = (0..features.len()).filter(|i| is_held_out(*i)).collect();
    if held.is_empty() {
        return Err(
            "deterministic split produced no held-out rows; this example requires at least \
             five data rows"
                .into(),
        );
    }
    println!(
        "split: train={} held-out={} (rule: every 5th row by position)",
        train.len(),
        held.len()
    );

    // 3. Fit on borrowed training data only, with an intercept.
    let config = FitConfig::default();
    let result =
        fit(&features, &targets, &train, &config).map_err(|e| format!("fit failed: {e:?}"))?;
    let diagnostics = result.diagnostics().clone();
    println!(
        "fit: intercept={:.6}, slope={:.6}",
        result.intercept(),
        result.weights().as_slice()[0]
    );
    println!(
        "diagnostics: rank={}/{} tau={:.3e} requested_bytes={}",
        diagnostics.rank,
        diagnostics.required_rank,
        diagnostics.rank_threshold,
        diagnostics.requested_workspace_bytes
    );
    println!(
        "training RSS: f64_reference={:.6}, f32_predictor={:.6}",
        diagnostics.rss_f64_reference, diagnostics.rss_f32_predictor
    );

    // 4. Learned predictor via immutable trained-model metadata.
    let meta = TrainingMeta {
        schema_id: *b"salary-schema-v1",
        transform_link: None,
        cutoff: None,
        fit_revision: [1, 0, 0, 0, 0, 0, 0, 0],
    };
    let trained = result
        .into_trained(&meta)
        .map_err(|e| format!("into_trained failed: {e:?}"))?;
    let model = trained.predictor();
    println!("model_id = {}", hex(trained.model_id()));
    println!("parameter_hash = {}", hex(&trained.metadata().param_hash));

    // 5. Single checked prediction.
    let first = held[0];
    let single = model
        .predict_checked(&features[first])
        .map_err(|e| format!("predict failed: {e:?}"))?;
    println!(
        "single prediction: x={:.4} -> {:.2} (actual {:.2})",
        features[first].as_slice()[0],
        single,
        targets[first]
    );

    // 6. Checked stream and grouped predictions through the existing interfaces.
    let checked = CheckedLinearRegression::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let mut streamed: Vec<f32> = Vec::with_capacity(held.len());
    for &i in &held {
        let value = *executor
            .process_one(&observation(features[i].as_slice()))
            .map_err(|e| format!("checked streaming failed at row {i}: {e:?}"))?;
        streamed.push(value);
    }
    let grouped = model
        .predict_batch_checked(
            &held
                .iter()
                .map(|&i| observation(features[i].as_slice()))
                .collect::<Vec<_>>(),
        )
        .map_err(|e| format!("predict_batch failed: {e:?}"))?;
    assert_eq!(streamed, grouped, "stream and grouped must agree");
    println!("held-out predictions (stream == grouped): {streamed:?}");

    // 7. Held-out RMSE vs training-mean baseline.
    //    The baseline mean is computed from training targets only.
    let train_mean =
        train.iter().map(|&i| f64::from(targets[i])).sum::<f64>() / (train.len() as f64);

    let mut model_sse = 0.0f64;
    let mut baseline_sse = 0.0f64;
    for (&i, &prediction) in held.iter().zip(&streamed) {
        let actual = f64::from(targets[i]);
        model_sse += (f64::from(prediction) - actual).powi(2);
        baseline_sse += (train_mean - actual).powi(2);
    }
    let n = held.len() as f64;
    let model_rmse = (model_sse / n).sqrt();
    let baseline_rmse = (baseline_sse / n).sqrt();
    // Never report a non-finite metric as a successful evaluation.
    if !train_mean.is_finite() || !model_rmse.is_finite() || !baseline_rmse.is_finite() {
        return Err("non-finite evaluation metric".into());
    }
    println!("training-mean baseline = {train_mean:.2}");
    println!("held-out RMSE: model={model_rmse:.2}, baseline={baseline_rmse:.2}");
    if model_rmse <= baseline_rmse {
        println!("model beats the training-mean baseline on this sample");
    } else {
        println!("model does not beat the training-mean baseline on this sample");
    }

    // 8. Artifact v1 round-trip with matching compatibility expectations.
    let limits = ArtifactLimits {
        max_features: 1024,
        max_bytes: 1 << 20,
    };
    let publication: PublicationId = *b"salary-pub-0001\0";
    let mut buffer: Vec<u8> = Vec::new();
    trained
        .export_to(&mut buffer, limits, Some(publication))
        .map_err(|e| format!("export failed: {e:?}"))?;
    let expectation = ArtifactExpectation {
        feature_count: Expect::Required(1),
        intercept: Expect::Required(true),
        schema_id: Expect::Required(meta.schema_id),
        transform_link: Expect::Absent,
        cutoff: Expect::Absent,
    };
    let mut cursor: &[u8] = &buffer;
    let loaded = TrainedModel::import_from(&mut cursor, &expectation, limits)
        .map_err(|e| format!("import failed: {e:?}"))?;
    assert_eq!(loaded.publication, Some(publication));
    assert_eq!(loaded.model.model_id(), trained.model_id());
    assert_eq!(
        loaded.model.predictor().weights().as_slice(),
        model.weights().as_slice()
    );
    assert_eq!(
        loaded.model.predictor().bias().to_bits(),
        model.bias().to_bits()
    );
    for &i in &held {
        let before = model
            .predict(&features[i])
            .map_err(|e| format!("predict failed: {e:?}"))?;
        let after = loaded
            .model
            .predictor()
            .predict(&features[i])
            .map_err(|e| format!("predict failed: {e:?}"))?;
        assert_eq!(
            before.to_bits(),
            after.to_bits(),
            "round-trip prediction bits"
        );
    }
    println!(
        "artifact v1 round-trip: {} bytes; model_id/parameters/predictions preserved",
        buffer.len()
    );

    // 9. Deliberate refit handover via a new executor; the active model is kept.
    //    The candidate is a clearly labeled synthetic modification (targets x2).
    //    All-zero targets can legitimately produce an identical candidate, which
    //    is handled explicitly as a no-op handover rather than asserted as
    //    different.
    let synthetic_targets: Vec<f32> = targets.iter().map(|&value| value * 2.0).collect();
    let candidate = fit(&features, &synthetic_targets, &train, &config)
        .map_err(|e| format!("candidate fit failed: {e:?}"))?
        .into_trained(&meta)
        .map_err(|e| format!("candidate into_trained failed: {e:?}"))?;
    let identical = candidate.model_id() == trained.model_id();

    let candidate_checked = CheckedLinearRegression::new(candidate.predictor());
    let mut candidate_executor = StreamingExecutor::new(&candidate_checked, 0.0f32);
    let mut candidate_streamed: Vec<f32> = Vec::with_capacity(held.len());
    for &i in &held {
        let value = *candidate_executor
            .process_one(&observation(features[i].as_slice()))
            .map_err(|e| format!("candidate checked streaming failed at row {i}: {e:?}"))?;
        candidate_streamed.push(value);
    }
    for (position, &i) in held.iter().enumerate() {
        let direct = candidate
            .predictor()
            .predict_checked(&features[i])
            .map_err(|e| format!("candidate predict failed: {e:?}"))?;
        assert_eq!(
            candidate_streamed[position].to_bits(),
            direct.to_bits(),
            "handover executor must serve the candidate"
        );
        if !identical {
            let active = model
                .predict_checked(&features[i])
                .map_err(|e| format!("predict failed: {e:?}"))?;
            let tolerance = 1e-2 * active.abs().max(1.0);
            assert!(
                (candidate_streamed[position] - 2.0 * active).abs() <= tolerance,
                "synthetic candidate should scale active predictions"
            );
        }
    }
    if identical {
        println!(
            "refit handover: candidate identical to active (no-op handover); active model retained"
        );
    } else {
        println!(
            "refit handover: candidate model_id = {} (new executor serves the candidate)",
            hex(candidate.model_id())
        );
    }

    // 10. Failed candidate: a rank-deficient fit is never installed.
    let duplicate_features = vec![
        Vector::from_slice(&[1.0, 2.0]),
        Vector::from_slice(&[2.0, 4.0]),
        Vector::from_slice(&[3.0, 6.0]),
    ];
    let duplicate_targets = [1.0f32, 2.0, 3.0];
    let duplicate_config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    match fit(
        &duplicate_features,
        &duplicate_targets,
        &[0, 1, 2],
        &duplicate_config,
    ) {
        Err(FitError::RankDeficient { rank, required }) => {
            println!("rank-deficient candidate rejected: rank={rank} required={required}");
        }
        other => return Err(format!("expected RankDeficient, got {other:?}").into()),
    }

    // 11. Rejected artifact import does not affect the active model.
    let mut corrupted = buffer.clone();
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 0xff;
    let mut bad_cursor: &[u8] = &corrupted;
    match TrainedModel::import_from(&mut bad_cursor, &expectation, limits) {
        Err(error) => println!("corrupted artifact rejected: {error:?}"),
        Ok(_) => return Err("corrupted artifact unexpectedly imported".into()),
    }

    // 12. Active model identity, parameters and predictions are unchanged.
    assert_eq!(trained.model_id(), loaded.model.model_id());
    assert_eq!(
        model.weights().as_slice(),
        trained.predictor().weights().as_slice()
    );
    assert_eq!(model.bias().to_bits(), trained.predictor().bias().to_bits());
    for &i in &held {
        let expected = model
            .predict(&features[i])
            .map_err(|e| format!("predict failed: {e:?}"))?;
        let now = trained
            .predictor()
            .predict(&features[i])
            .map_err(|e| format!("predict failed: {e:?}"))?;
        assert_eq!(expected.to_bits(), now.to_bits(), "active model unchanged");
    }
    println!("active model unchanged after failed candidate and rejected import");

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::run_with_path;

    /// Writes a synthetic `,YearsExperience,Salary` CSV to the system temp dir
    /// (never inside the repository) and returns its path.
    fn write_csv(name: &str, rows: &[(f32, f32)]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("seqvex_fit_salary_{name}.csv"));
        let mut content = String::from(",YearsExperience,Salary\n");
        for (row, (feature, target)) in rows.iter().enumerate() {
            content.push_str(&format!("{row},{feature},{target}\n"));
        }
        std::fs::write(&path, content).expect("write temp csv");
        path
    }

    fn toy_rows() -> Vec<(f32, f32)> {
        (0..10).map(|x| (x as f32, 2.0 * x as f32 + 1.0)).collect()
    }

    #[test]
    fn held_out_nan_target_is_rejected() {
        let mut rows = toy_rows();
        rows[9].1 = f32::NAN;
        let path = write_csv("nan_target", &rows);
        let error = run_with_path(path.to_str().unwrap()).expect_err("NaN target must fail");
        let message = error.to_string();
        assert!(message.contains("non-finite Salary"), "{message}");
        assert!(message.contains("row 9"), "{message}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn held_out_nan_feature_is_rejected_without_panic() {
        let mut rows = toy_rows();
        rows[9].0 = f32::NAN;
        let path = write_csv("nan_feature", &rows);
        let error = run_with_path(path.to_str().unwrap()).expect_err("NaN feature must fail");
        let message = error.to_string();
        assert!(message.contains("non-finite YearsExperience"), "{message}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn finite_active_prediction_overflow_is_propagated() {
        // Held-out row 9 uses a finite feature that overflows the active model's
        // kept-splitting slope-2 prediction (2 * 3e38 = inf).
        let mut rows = toy_rows();
        rows[9] = (3.0e38, 1.0);
        let path = write_csv("active_overflow", &rows);
        let error = run_with_path(path.to_str().unwrap()).expect_err("overflow must fail");
        let message = error.to_string();
        assert!(message.contains("checked streaming failed"), "{message}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn candidate_only_overflow_is_propagated() {
        // Active prediction stays finite (2e38); the doubled-target candidate
        // overflows (4e38) and must surface a clear checked error.
        let mut rows = toy_rows();
        rows[9] = (1.0e38, 0.0);
        let path = write_csv("candidate_overflow", &rows);
        let error = run_with_path(path.to_str().unwrap()).expect_err("candidate must fail");
        let message = error.to_string();
        assert!(
            message.contains("candidate checked streaming failed"),
            "{message}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn all_zero_targets_yield_identical_candidate_without_panic() {
        let rows: Vec<(f32, f32)> = (0..10).map(|x| (x as f32, 0.0)).collect();
        let path = write_csv("all_zero", &rows);
        run_with_path(path.to_str().unwrap()).expect("all-zero targets must complete");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn fewer_than_five_rows_is_rejected() {
        let rows: Vec<(f32, f32)> = (0..4).map(|x| (x as f32, 2.0 * x as f32 + 1.0)).collect();
        let path = write_csv("short", &rows);
        let error = run_with_path(path.to_str().unwrap()).expect_err("short split must fail");
        assert!(error.to_string().contains("no held-out rows"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn actual_salary_dataset_workflow_succeeds_when_present() {
        let path = "examples/Salary_dataset.csv";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        run_with_path(path).expect("Salary workflow must succeed");
    }
}
