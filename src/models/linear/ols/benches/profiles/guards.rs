use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, FitConfig, FitConfig32, FitConfig64,
    FitResult, FitResult32, FitResult64, LinearRegression, LinearRegression64, UnderflowPolicy,
    fit, fit_f32, fit_f64,
};

use super::fixture::{Fixture, widen};

pub(super) fn fit_a(fixture: &Fixture) -> FitResult64 {
    fit_f64(
        &fixture.features64,
        &fixture.targets64,
        &fixture.selection,
        &FitConfig64::default(),
    )
    .expect("A fit succeeds")
}

pub(super) fn fit_c(fixture: &Fixture) -> FitResult32 {
    fit_f32(
        &fixture.features32,
        &fixture.targets32,
        &fixture.selection,
        &FitConfig32::default(),
    )
    .expect("C fit succeeds")
}

pub(super) fn fit_legacy(fixture: &Fixture) -> FitResult {
    fit(
        &fixture.features32,
        &fixture.targets32,
        &fixture.selection,
        &FitConfig::default(),
    )
    .expect("legacy fit succeeds")
}

pub(super) fn check_fit_a(result: &FitResult64, fixture: &Fixture) {
    let diagnostics = result.diagnostics();
    assert_eq!(diagnostics.rows, fixture.n);
    assert_eq!(diagnostics.features, fixture.d);
    assert!(diagnostics.intercept);
    assert_eq!(diagnostics.rank, diagnostics.required_rank);
    assert!(diagnostics.rank_threshold.is_finite());
    assert!(diagnostics.rss_f64.is_finite());
    assert!(
        result
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.intercept().is_finite());
    let mut rss = 0.0f64;
    for (row, &target) in fixture.features64.iter().zip(&fixture.targets64) {
        let prediction = result
            .weights()
            .as_slice()
            .iter()
            .zip(row.as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f64>()
            + result.intercept();
        let residual = prediction - target;
        rss += residual * residual;
    }
    assert_eq!(rss.to_bits(), diagnostics.rss_f64.to_bits());
    let predictor = result.predictor();
    for probe in &fixture.probes64 {
        assert!(
            predictor
                .predict_checked(probe)
                .expect("A measured prediction")
                .is_finite()
        );
    }
}

pub(super) fn check_fit_c(result: &FitResult32, fixture: &Fixture) {
    let diagnostics = result.diagnostics();
    assert_eq!(diagnostics.rows, fixture.n);
    assert_eq!(diagnostics.features, fixture.d);
    assert!(diagnostics.intercept);
    assert_eq!(diagnostics.rank, diagnostics.required_rank);
    assert!(diagnostics.rank_threshold.is_finite());
    assert!(diagnostics.rss_f32.is_finite());
    assert!(
        result
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.intercept().is_finite());
    let mut rss = 0.0f32;
    for (row, &target) in fixture.features32.iter().zip(&fixture.targets32) {
        let prediction = result
            .weights()
            .as_slice()
            .iter()
            .zip(row.as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + result.intercept();
        let residual = prediction - target;
        rss += residual * residual;
    }
    assert_eq!(rss.to_bits(), diagnostics.rss_f32.to_bits());
    let predictor = result.predictor();
    for probe in &fixture.probes32 {
        assert!(
            predictor
                .predict_checked(probe)
                .expect("C measured prediction")
                .is_finite()
        );
    }
}

pub(super) fn check_fit_legacy(result: &FitResult, fixture: &Fixture) {
    let diagnostics = result.diagnostics();
    assert_eq!(diagnostics.rows, fixture.n);
    assert_eq!(diagnostics.features, fixture.d);
    assert!(diagnostics.intercept);
    assert_eq!(diagnostics.rank, diagnostics.required_rank);
    assert!(diagnostics.rank_threshold.is_finite());
    assert!(diagnostics.rss_f64_reference.is_finite());
    assert!(diagnostics.rss_f32_predictor.is_finite());
    assert!(
        result
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.intercept().is_finite());
    let predictor = result.predictor();
    let mut rss = 0.0f64;
    for (row, &target) in fixture.features32.iter().zip(&fixture.targets32) {
        let prediction = predictor.predict(row).expect("legacy RSS prediction");
        let residual = f64::from(prediction) - f64::from(target);
        rss += residual * residual;
    }
    assert_eq!(rss.to_bits(), diagnostics.rss_f32_predictor.to_bits());
    for probe in &fixture.probes32 {
        assert!(
            predictor
                .predict_checked(probe)
                .expect("legacy measured prediction")
                .is_finite()
        );
    }
}

fn abs32(actual: f32, expected: f32, tolerance: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, got {actual}"
    );
}

fn abs64(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, got {actual}"
    );
}

pub(super) fn check_serving32(model: &LinearRegression, probes: &[Vector]) {
    let observations: Vec<Observation<Vector>> =
        probes.iter().cloned().map(Observation::new).collect();
    let singles: Vec<u32> = probes
        .iter()
        .map(|probe| {
            model
                .predict_checked(probe)
                .expect("single succeeds")
                .to_bits()
        })
        .collect();
    let checked = CheckedLinearRegression::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let streamed: Vec<u32> = observations
        .iter()
        .map(|observation| (*executor.process_one(observation).expect("stream succeeds")).to_bits())
        .collect();
    assert_eq!(streamed, singles, "A/B/C f32 stream vs single");
    assert_eq!(
        executor.state().to_bits(),
        *singles.last().expect("nonempty probes"),
        "f32 final executor state"
    );
    for &batch in &[1usize, 32, 128] {
        let group: Vec<Observation<Vector>> = (0..batch)
            .map(|index| observations[index % observations.len()].clone())
            .collect();
        let grouped: Vec<u32> = model
            .predict_batch_checked(&group)
            .expect("group succeeds")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected: Vec<u32> = (0..batch)
            .map(|index| singles[index % singles.len()])
            .collect();
        assert_eq!(grouped, expected, "f32 group {batch} vs single");
    }
}

pub(super) fn check_serving64(model: &LinearRegression64, probes: &[Vector64]) {
    let observations: Vec<Observation<Vector64>> =
        probes.iter().cloned().map(Observation::new).collect();
    let singles: Vec<u64> = probes
        .iter()
        .map(|probe| {
            model
                .predict_checked(probe)
                .expect("single succeeds")
                .to_bits()
        })
        .collect();
    let checked = CheckedLinearRegression64::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f64);
    let streamed: Vec<u64> = observations
        .iter()
        .map(|observation| (*executor.process_one(observation).expect("stream succeeds")).to_bits())
        .collect();
    assert_eq!(streamed, singles, "A stream vs single");
    assert_eq!(
        executor.state().to_bits(),
        *singles.last().expect("nonempty probes"),
        "A final executor state"
    );
    for &batch in &[1usize, 32, 128] {
        let group: Vec<Observation<Vector64>> = (0..batch)
            .map(|index| observations[index % observations.len()].clone())
            .collect();
        let grouped: Vec<u64> = model
            .predict_batch_checked(&group)
            .expect("group succeeds")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected: Vec<u64> = (0..batch)
            .map(|index| singles[index % singles.len()])
            .collect();
        assert_eq!(grouped, expected, "A group {batch} vs single");
    }
}

pub(super) fn verify_profiles() {
    let features32 = vec![
        Vector::from_slice(&[0.0, 0.0]),
        Vector::from_slice(&[1.0, 0.0]),
        Vector::from_slice(&[0.0, 1.0]),
        Vector::from_slice(&[1.0, 1.0]),
        Vector::from_slice(&[2.0, 1.0]),
        Vector::from_slice(&[1.0, 2.0]),
    ];
    let features64: Vec<Vector64> = features32.iter().map(|row| widen(row.as_slice())).collect();
    let targets32 = [1.0f32, 3.0, -2.0, 0.0, 2.0, -3.0];
    let targets64: Vec<f64> = targets32.iter().map(|&value| f64::from(value)).collect();

    let a_result = fit_f64(
        &features64,
        &targets64,
        &[0, 1, 2, 3, 4, 5],
        &FitConfig64::default(),
    )
    .expect("analytical A fit");
    abs64(a_result.weights().as_slice()[0], 2.0, 1.0e-12, "A w0");
    abs64(a_result.weights().as_slice()[1], -3.0, 1.0e-12, "A w1");
    abs64(a_result.intercept(), 1.0, 1.0e-12, "A bias");
    abs64(a_result.diagnostics().rss_f64, 0.0, 1.0e-12, "A rss");

    let c_result = fit_f32(
        &features32,
        &targets32,
        &[0, 1, 2, 3, 4, 5],
        &FitConfig32::default(),
    )
    .expect("analytical C fit");
    abs32(c_result.weights().as_slice()[0], 2.0, 1.0e-5, "C w0");
    abs32(c_result.weights().as_slice()[1], -3.0, 1.0e-5, "C w1");
    abs32(c_result.intercept(), 1.0, 1.0e-5, "C bias");
    let mut c_rss = 0.0f32;
    for (index, &target) in targets32.iter().enumerate() {
        let prediction = c_result
            .weights()
            .as_slice()
            .iter()
            .zip(features32[index].as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + c_result.intercept();
        let residual = prediction - target;
        c_rss += residual * residual;
    }
    assert_eq!(c_result.diagnostics().rss_f32.to_bits(), c_rss.to_bits());
    let c_model = c_result.into_predictor();
    for (index, probe) in features32.iter().enumerate() {
        abs32(
            c_model
                .predict_checked(probe)
                .expect("C analytical checked prediction"),
            targets32[index],
            1.0e-5,
            "C analytical prediction",
        );
    }

    let source_weights: Vec<u64> = a_result
        .weights()
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let source_bias = a_result.intercept().to_bits();
    let b_model = a_result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion succeeds");
    let expected_weights: Vec<f32> = a_result
        .weights()
        .as_slice()
        .iter()
        .map(|&value| value as f32)
        .collect();
    for (actual, expected) in b_model.weights().as_slice().iter().zip(&expected_weights) {
        assert_eq!(actual.to_bits(), expected.to_bits(), "B weight bits");
    }
    assert_eq!(
        b_model.bias().to_bits(),
        (a_result.intercept() as f32).to_bits(),
        "B bias bits"
    );
    let expected_b = LinearRegression::new(
        Vector::from_slice(&expected_weights),
        a_result.intercept() as f32,
    )
    .expect("expected B model");
    for probe in &features32 {
        assert_eq!(
            b_model
                .predict_checked(probe)
                .expect("B prediction")
                .to_bits(),
            expected_b
                .predict_checked(probe)
                .expect("expected B prediction")
                .to_bits(),
            "B prediction bits"
        );
    }
    assert_eq!(
        source_weights,
        a_result
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "B conversion source weights"
    );
    assert_eq!(source_bias, a_result.intercept().to_bits());

    let a_model = a_result.into_predictor();
    check_serving64(&a_model, &features64);
    check_serving32(&b_model, &features32);
    check_serving32(&c_model, &features32);
}

pub(super) fn source_bits64(result: &FitResult64) -> (Vec<u64>, u64) {
    (
        result
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        result.intercept().to_bits(),
    )
}

pub(super) fn model_bits64(model: &LinearRegression64) -> (Vec<u64>, u64) {
    (
        model
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        model.bias().to_bits(),
    )
}

pub(super) fn model_bits32(model: &LinearRegression) -> (Vec<u32>, u32) {
    (
        model
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        model.bias().to_bits(),
    )
}

pub(super) fn check_b_conversion(
    source: &FitResult64,
    model: &LinearRegression,
    probes: &[Vector],
) {
    let source_weights: Vec<f32> = source
        .weights()
        .as_slice()
        .iter()
        .map(|&value| value as f32)
        .collect();
    assert_eq!(
        model.weights().as_slice().len(),
        source_weights.len(),
        "measured B weight length"
    );
    for (actual, expected) in model.weights().as_slice().iter().zip(&source_weights) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "measured B weight bits"
        );
    }
    assert_eq!(
        model.bias().to_bits(),
        (source.intercept() as f32).to_bits(),
        "measured B bias bits"
    );
    let reference = LinearRegression::new(
        Vector::from_slice(&source_weights),
        source.intercept() as f32,
    )
    .expect("measured B reference");
    for probe in probes {
        assert_eq!(
            model
                .predict_checked(probe)
                .expect("measured B prediction")
                .to_bits(),
            reference
                .predict_checked(probe)
                .expect("measured B reference prediction")
                .to_bits(),
            "measured B prediction bits"
        );
    }
}
