use std::env;

use sha2::{Digest, Sha256};

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, FitConfig32, FitConfig64, FitError64,
    LinearRegression, UnderflowPolicy, fit_f32, fit_f64,
};

use super::reporting::{report_a_fit, report_c_fit, report_metrics};
use super::support::{
    Dataset, FEATURE_HEADER, MAX_GROUP_SIZE, TARGET_HEADER, TRAIN_ROWS, hex, metrics,
    narrow_b_input, narrow_b_inputs, parse_dataset, read_input,
};

pub(super) fn run() -> Result<(), String> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (bytes, source) = read_input(&args)?;
    let data = parse_dataset(bytes, source)?;
    let rows = data.features64.len();
    let train: Vec<usize> = (0..TRAIN_ROWS).collect();
    let held_out: Vec<usize> = (TRAIN_ROWS..rows).collect();

    println!("source: {}", data.source);
    println!("sha256: {}", hex(&Sha256::digest(&data.bytes)));
    println!("canonical fixture: {}", data.canonical);
    println!("columns: {FEATURE_HEADER} [m] -> {TARGET_HEADER} [s]");
    println!("rows: {rows}; training: {train:?}; evaluation: {held_out:?}");
    println!("preprocessing: identity; feature order: [{FEATURE_HEADER}]");
    println!("split: static first-five training, remaining rows held out (not temporal)");

    let a_fit = fit_f64(
        &data.features64,
        &data.targets64,
        &train,
        &FitConfig64::default(),
    )
    .map_err(|error| format!("A fit failed: {error}"))?;
    report_a_fit(&a_fit);
    assert_fit_dimensions(
        a_fit.weights().len(),
        a_fit.diagnostics().features,
        a_fit.diagnostics().rank,
        a_fit.diagnostics().required_rank,
    );

    let a_weights_before: Vec<u64> = a_fit
        .weights()
        .as_slice()
        .iter()
        .map(|weight| weight.to_bits())
        .collect();
    let a_intercept_before = a_fit.intercept().to_bits();
    let a_diagnostics_before = a_fit.diagnostics().clone();
    let a_training_rss = a_fit.diagnostics().rss_f64;
    if data.canonical {
        assert_close(a_fit.weights().as_slice()[0], 2.0, 1.0e-12, "A weight");
        assert_close(a_fit.intercept(), 1.0, 1.0e-12, "A intercept");
    }

    let b_model = a_fit
        .to_serving_f32(UnderflowPolicy::Reject)
        .map_err(|error| format!("B parameter conversion failed: {error}"))?;
    let b_reference_weights: Vec<f32> = a_fit
        .weights()
        .as_slice()
        .iter()
        .map(|&weight| weight as f32)
        .collect();
    let b_reference_bias = a_fit.intercept() as f32;
    assert_eq!(b_model.weights().len(), a_weights_before.len());
    for (index, (&actual, &expected)) in b_model
        .weights()
        .as_slice()
        .iter()
        .zip(&b_reference_weights)
        .enumerate()
    {
        assert_eq!(actual.to_bits(), expected.to_bits(), "B weight {index}");
    }
    assert_eq!(b_model.bias().to_bits(), b_reference_bias.to_bits());
    assert_eq!(
        a_fit
            .weights()
            .as_slice()
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>(),
        a_weights_before,
        "B conversion must preserve the A source weights"
    );
    assert_eq!(a_fit.intercept().to_bits(), a_intercept_before);
    assert_eq!(a_fit.diagnostics(), &a_diagnostics_before);

    let a_model = a_fit.into_predictor();
    assert_eq!(a_model.weights().len(), 1);
    assert_eq!(b_model.weights().len(), 1);
    assert_single_feature_dimension_a(&a_model);
    assert_single_feature_dimension_f32(&b_model);

    let c_fit = fit_f32(
        &data.features32,
        &data.targets32,
        &train,
        &FitConfig32::default(),
    )
    .map_err(|error| format!("C fit failed: {error}"))?;
    report_c_fit(&c_fit);
    assert_fit_dimensions(
        c_fit.weights().len(),
        c_fit.diagnostics().features,
        c_fit.diagnostics().rank,
        c_fit.diagnostics().required_rank,
    );
    if data.canonical {
        assert_close(
            f64::from(c_fit.weights().as_slice()[0]),
            2.0,
            1.0e-5,
            "C weight",
        );
        assert_close(f64::from(c_fit.intercept()), 1.0, 1.0e-5, "C intercept");
        assert_close(
            f64::from(c_fit.diagnostics().rss_f32),
            7.0 / 8.0,
            1.0e-3,
            "canonical C native training RSS",
        );
    }
    let c_model = c_fit.into_predictor();
    assert_eq!(c_model.weights().len(), 1);
    assert_single_feature_dimension_f32(&c_model);

    let a_predictions = predict_a(&a_model, &data.features64)?;
    let b_input_rows = narrow_b_inputs(&data.features64)?;
    let b_predictions = predict_b(&b_model, &b_input_rows)?;
    let c_predictions = predict_c(&c_model, &data.features32)?;
    let b_reference =
        LinearRegression::new(Vector::from_slice(&b_reference_weights), b_reference_bias)
            .map_err(|error| format!("independent B parameter reference invalid: {error}"))?;
    verify_b_predictions(&b_model, &b_reference, &b_input_rows)?;

    let new_a_prediction = a_model
        .predict_checked(&Vector64::from_slice(&[2.5]))
        .map_err(|error| format!("A unlabelled prediction: {error}"))?;
    let new_b_input = Vector::from_slice(&[narrow_b_input(2.5, rows)?]);
    let new_b_prediction = b_model
        .predict_checked(&new_b_input)
        .map_err(|error| format!("B unlabelled prediction: {error}"))?;
    let new_c_prediction = c_model
        .predict_checked(&Vector::from_slice(&[2.5]))
        .map_err(|error| format!("C unlabelled prediction: {error}"))?;

    for (label, prediction) in [
        ("A", new_a_prediction),
        ("B-from-A", f64::from(new_b_prediction)),
        ("C", f64::from(new_c_prediction)),
    ] {
        println!("frozen version csv-v1/{label}: x=2.5 -> {prediction:.12} seconds");
    }
    if data.canonical {
        assert_close(new_a_prediction, 6.0, 1.0e-11, "A unlabelled prediction");
        assert_close(
            f64::from(new_b_prediction),
            6.0,
            1.0e-4,
            "B unlabelled prediction",
        );
        assert_close(
            f64::from(new_c_prediction),
            6.0,
            1.0e-4,
            "C unlabelled prediction",
        );
    }

    let targets = &data.targets64[TRAIN_ROWS..];
    let a_metrics = metrics(&a_predictions[TRAIN_ROWS..], targets);
    let b_outputs: Vec<f64> = b_predictions[TRAIN_ROWS..]
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    let b_metrics = metrics(&b_outputs, targets);
    let c_outputs: Vec<f64> = c_predictions[TRAIN_ROWS..]
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    let c_metrics = metrics(&c_outputs, targets);
    let training_mean = data.targets64[..TRAIN_ROWS].iter().sum::<f64>() / TRAIN_ROWS as f64;
    let baseline_predictions = vec![training_mean; targets.len()];
    let baseline = metrics(&baseline_predictions, targets);
    report_metrics("A", a_metrics);
    report_metrics("B-from-A", b_metrics);
    report_metrics("C", c_metrics);
    report_metrics("training-target-mean baseline", baseline);

    if data.canonical {
        for (index, expected) in [11.2, 13.2, 15.2].into_iter().enumerate() {
            assert_close(
                a_predictions[TRAIN_ROWS + index],
                expected,
                1.0e-11,
                "A held-out prediction",
            );
            assert_close(
                f64::from(b_predictions[TRAIN_ROWS + index]),
                expected,
                1.0e-4,
                "B held-out prediction",
            );
            assert_close(
                f64::from(c_predictions[TRAIN_ROWS + index]),
                expected,
                1.0e-4,
                "C held-out prediction",
            );
        }
        assert_close(a_training_rss, 7.0 / 8.0, 1e-10, "A training RSS");
        assert_close(a_metrics.rss, 3.0 / 8.0, 1e-10, "A held-out RSS");
        assert_close(a_metrics.mae, 1.0 / 3.0, 1e-11, "A held-out MAE");
        assert_close(
            a_metrics.rmse,
            (1.0_f64 / 8.0).sqrt(),
            1e-11,
            "A held-out RMSE",
        );
        assert_close(b_metrics.rss, 3.0 / 8.0, 1e-3, "B held-out RSS");
        assert_close(c_metrics.rss, 3.0 / 8.0, 1e-3, "C held-out RSS");
        assert_close(b_metrics.mae, 1.0 / 3.0, 1e-4, "B held-out MAE");
        assert_close(
            b_metrics.rmse,
            (1.0_f64 / 8.0).sqrt(),
            1e-4,
            "B held-out RMSE",
        );
        assert_close(c_metrics.mae, 1.0 / 3.0, 1e-4, "C held-out MAE");
        assert_close(
            c_metrics.rmse,
            (1.0_f64 / 8.0).sqrt(),
            1e-4,
            "C held-out RMSE",
        );
        assert_close(baseline.rss, 1603.0 / 8.0, 1e-10, "baseline held-out RSS");
        assert_close(baseline.mae, 8.0, 1e-10, "baseline held-out MAE");
        assert_close(
            baseline.rmse,
            (1603.0_f64 / 24.0).sqrt(),
            1e-10,
            "baseline held-out RMSE",
        );
    }

    verify_delivery_equivalence(&a_model, &b_model, &c_model)?;
    verify_invalid_input(&a_model, &b_model, &c_model);
    if data.canonical {
        verify_candidate_rejection(&data, &a_model, &b_model, &c_model)?;
    } else {
        println!("candidate rejection/freeze demonstration: canonical fixture only");
    }

    println!("persistence: not demonstrated for A/B/C; typed artifact persistence is unsupported");
    println!("workflow complete; no speedup or production-readiness claim");
    Ok(())
}

fn assert_fit_dimensions(weights: usize, features: usize, rank: usize, required_rank: usize) {
    assert_eq!(weights, 1, "one learned weight for one input feature");
    assert_eq!(features, 1, "one feature, intercept stored separately");
    assert_eq!(
        features + 1,
        2,
        "two columns in the intercept-augmented fit matrix"
    );
    assert_eq!(rank, 2);
    assert_eq!(required_rank, 2);
}

fn assert_single_feature_dimension_a(model: &seqvex::models::linear::ols::LinearRegression64) {
    assert!(matches!(
        model.predict_checked(&Vector64::from_slice(&[1.0, 2.0])),
        Err(
            seqvex::models::linear::ols::RegressionError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        )
    ));
}

fn assert_single_feature_dimension_f32(model: &LinearRegression) {
    assert!(matches!(
        model.predict_checked(&Vector::from_slice(&[1.0, 2.0])),
        Err(
            seqvex::models::linear::ols::RegressionError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        )
    ));
}

fn predict_a(
    model: &seqvex::models::linear::ols::LinearRegression64,
    features: &[Vector64],
) -> Result<Vec<f64>, String> {
    features
        .iter()
        .enumerate()
        .map(|(row, feature)| {
            model
                .predict_checked(feature)
                .map_err(|error| format!("A checked prediction row {row}: {error}"))
        })
        .collect()
}

fn predict_b(model: &LinearRegression, features: &[Vector]) -> Result<Vec<f32>, String> {
    features
        .iter()
        .enumerate()
        .map(|(row, feature)| {
            model
                .predict_checked(feature)
                .map_err(|error| format!("B checked prediction row {row}: {error}"))
        })
        .collect()
}

fn predict_c(model: &LinearRegression, features: &[Vector]) -> Result<Vec<f32>, String> {
    features
        .iter()
        .enumerate()
        .map(|(row, feature)| {
            model
                .predict_checked(feature)
                .map_err(|error| format!("C checked prediction row {row}: {error}"))
        })
        .collect()
}

fn independent_predict_f32(model: &LinearRegression, feature: &[f32]) -> f32 {
    let mut accumulator = -0.0_f32;
    for (&weight, &value) in model.weights().as_slice().iter().zip(feature) {
        accumulator += weight * value;
    }
    accumulator + model.bias()
}

fn verify_b_predictions(
    actual: &LinearRegression,
    rounded_parameters: &LinearRegression,
    features: &[Vector],
) -> Result<(), String> {
    assert_eq!(actual.weights().len(), rounded_parameters.weights().len());
    for (row, feature) in features.iter().enumerate() {
        let expected = independent_predict_f32(rounded_parameters, feature.as_slice());
        let prediction = actual
            .predict_checked(feature)
            .map_err(|error| format!("B reference prediction row {row}: {error}"))?;
        assert_eq!(
            prediction.to_bits(),
            expected.to_bits(),
            "B prediction row {row}"
        );
    }
    Ok(())
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{label}: expected {expected:.16} ± {tolerance:e}, got {actual:.16}"
    );
}

fn verify_delivery_equivalence(
    a: &seqvex::models::linear::ols::LinearRegression64,
    b: &LinearRegression,
    c: &LinearRegression,
) -> Result<(), String> {
    let probes = [0.1, 2.5, 7.1];
    assert!(probes.len() <= MAX_GROUP_SIZE);
    let rows64: Vec<_> = probes
        .iter()
        .map(|&value| Observation::new(Vector64::from_slice(&[value])))
        .collect();
    let rows32: Vec<_> = probes
        .iter()
        .map(|&value| Observation::new(Vector::from_slice(&[value as f32])))
        .collect();

    let a_group = a
        .predict_batch_checked(&rows64)
        .map_err(|error| format!("A bounded group: {error}"))?;
    let a_single: Vec<f64> = rows64
        .iter()
        .map(|row| {
            a.predict_checked(row.value())
                .map_err(|error| format!("A single prediction: {error}"))
        })
        .collect::<Result<_, _>>()?;
    assert_bits_equal(&a_group, &a_single, "A group/single");
    let a_checked = CheckedLinearRegression64::new(a);
    let mut a_executor = StreamingExecutor::new(&a_checked, 0.0f64);
    let mut a_stream = Vec::new();
    for row in &rows64 {
        a_stream.push(
            *a_executor
                .process_one(row)
                .map_err(|error| format!("A checked stream: {error}"))?,
        );
    }
    assert_bits_equal(&a_group, &a_stream, "A group/stream");
    assert_eq!(
        a_executor.state().to_bits(),
        a_stream.last().unwrap().to_bits()
    );

    let b_group = b
        .predict_batch_checked(&rows32)
        .map_err(|error| format!("B bounded group: {error}"))?;
    let b_single: Vec<f32> = rows32
        .iter()
        .map(|row| {
            b.predict_checked(row.value())
                .map_err(|error| format!("B single prediction: {error}"))
        })
        .collect::<Result<_, _>>()?;
    assert_bits_equal(&b_group, &b_single, "B group/single");
    let b_checked = CheckedLinearRegression::new(b);
    let mut b_executor = StreamingExecutor::new(&b_checked, 0.0f32);
    let mut b_stream = Vec::new();
    for row in &rows32 {
        b_stream.push(
            *b_executor
                .process_one(row)
                .map_err(|error| format!("B checked stream: {error}"))?,
        );
    }
    assert_bits_equal(&b_group, &b_stream, "B group/stream");
    assert_eq!(
        b_executor.state().to_bits(),
        b_stream.last().unwrap().to_bits()
    );

    let c_group = c
        .predict_batch_checked(&rows32)
        .map_err(|error| format!("C bounded group: {error}"))?;
    let c_single: Vec<f32> = rows32
        .iter()
        .map(|row| {
            c.predict_checked(row.value())
                .map_err(|error| format!("C single prediction: {error}"))
        })
        .collect::<Result<_, _>>()?;
    assert_bits_equal(&c_group, &c_single, "C group/single");
    let c_checked = CheckedLinearRegression::new(c);
    let mut c_executor = StreamingExecutor::new(&c_checked, 0.0f32);
    let mut c_stream = Vec::new();
    for row in &rows32 {
        c_stream.push(
            *c_executor
                .process_one(row)
                .map_err(|error| format!("C checked stream: {error}"))?,
        );
    }
    assert_bits_equal(&c_group, &c_stream, "C group/stream");
    assert_eq!(
        c_executor.state().to_bits(),
        c_stream.last().unwrap().to_bits()
    );
    Ok(())
}

fn assert_bits_equal<T: Copy + Into<f64>>(left: &[T], right: &[T], label: &str) {
    assert_eq!(left.len(), right.len(), "{label} lengths");
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left.into().to_bits(),
            right.into().to_bits(),
            "{label} row {index}"
        );
    }
}

fn verify_invalid_input(
    a: &seqvex::models::linear::ols::LinearRegression64,
    b: &LinearRegression,
    c: &LinearRegression,
) {
    assert!(
        a.predict_checked(&Vector64::from_slice(&[f64::NAN]))
            .is_err()
    );
    assert!(b.predict_checked(&Vector::from_slice(&[f32::NAN])).is_err());
    assert!(c.predict_checked(&Vector::from_slice(&[f32::NAN])).is_err());
}

fn verify_candidate_rejection(
    data: &Dataset,
    selected_a: &seqvex::models::linear::ols::LinearRegression64,
    selected_b: &LinearRegression,
    selected_c: &LinearRegression,
) -> Result<(), String> {
    let probe = Vector::from_slice(&[2.5]);
    let selected_before = selected_b
        .predict_checked(&probe)
        .map_err(|error| format!("selected B probe failed: {error}"))?;
    let selected_weights: Vec<u32> = selected_b
        .weights()
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let a_before = selected_a
        .predict_checked(&Vector64::from_slice(&[2.5]))
        .map_err(|error| format!("selected A probe failed: {error}"))?;
    let c_before = selected_c
        .predict_checked(&probe)
        .map_err(|error| format!("selected C probe failed: {error}"))?;

    let shifted_targets: Vec<f64> = data.targets64.iter().map(|target| target + 100.0).collect();
    let train: Vec<usize> = (0..TRAIN_ROWS).collect();
    let shifted_a = fit_f64(
        &data.features64,
        &shifted_targets,
        &train,
        &FitConfig64::default(),
    )
    .map_err(|error| format!("shifted candidate fit failed: {error}"))?;
    let candidate = shifted_a
        .to_serving_f32(UnderflowPolicy::Reject)
        .map_err(|error| format!("shifted candidate conversion failed: {error}"))?;
    let candidate_prediction = candidate
        .predict_checked(&probe)
        .map_err(|error| format!("shifted candidate prediction failed: {error}"))?;
    assert!(candidate_prediction.is_finite());
    assert_ne!(candidate_prediction.to_bits(), selected_before.to_bits());

    let rank_deficient = fit_f64(
        &[Vector64::from_slice(&[1.0]), Vector64::from_slice(&[1.0])],
        &[1.0, 2.0],
        &[0, 1],
        &FitConfig64::default(),
    );
    assert!(matches!(
        rank_deficient,
        Err(FitError64::RankDeficient { .. })
    ));

    let selected_after = selected_b
        .predict_checked(&probe)
        .map_err(|error| format!("selected B after rejection: {error}"))?;
    assert_eq!(selected_after.to_bits(), selected_before.to_bits());
    assert_eq!(
        selected_b
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        selected_weights
    );
    assert_eq!(
        selected_a
            .predict_checked(&Vector64::from_slice(&[2.5]))
            .map_err(|error| format!("selected A after rejection: {error}"))?
            .to_bits(),
        a_before.to_bits()
    );
    assert_eq!(
        selected_c
            .predict_checked(&probe)
            .map_err(|error| format!("selected C after rejection: {error}"))?
            .to_bits(),
        c_before.to_bits()
    );
    println!(
        "candidate csv-v2/B-from-A rejected by the example's explicit no-auto-install policy; csv-v1/B-from-A remains selected"
    );
    Ok(())
}
