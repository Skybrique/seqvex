//! Independent evidence for the native-f64 (A), checked-serving (B) and
//! strict-f32 (C) OLS profiles.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression64, FitConfig32, FitConfig64, FitError32, FitError64, RegressionError,
    ServingConversionError, ServingField, UnderflowPolicy, fit_f32, fit_f64,
};

fn row64(values: &[f64]) -> Vector64 {
    Vector64::from_slice(values)
}

fn row32(values: &[f32]) -> Vector {
    Vector::from_slice(values)
}

fn obs64(values: &[f64]) -> Observation<Vector64> {
    Observation::new(row64(values))
}

fn obs32(values: &[f32]) -> Observation<Vector> {
    Observation::new(row32(values))
}

fn fit_a(
    features: &[Vector64],
    targets: &[f64],
    train: &[usize],
    config: &FitConfig64,
) -> seqvex::models::linear::ols::FitResult64 {
    fit_f64(features, targets, train, config).expect("A fit succeeds")
}

fn fit_c(
    features: &[Vector],
    targets: &[f32],
    train: &[usize],
    config: &FitConfig32,
) -> seqvex::models::linear::ols::FitResult32 {
    fit_f32(features, targets, train, config).expect("C fit succeeds")
}

fn one_weight(value: f64) -> seqvex::models::linear::ols::FitResult64 {
    fit_a(
        &[row64(&[1.0])],
        &[value],
        &[0],
        &FitConfig64 {
            intercept: false,
            ..FitConfig64::default()
        },
    )
}

#[test]
fn a_recovers_exact_and_noisy_lines_with_independent_references() {
    let exact = fit_a(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[2.0])],
        &[1.0, 3.0, 5.0],
        &[0, 1, 2],
        &FitConfig64::default(),
    );
    assert!((exact.weights().as_slice()[0] - 2.0).abs() <= 2.0e-12);
    assert!((exact.intercept() - 1.0).abs() <= 1.0e-12);
    assert!(exact.diagnostics().rss_f64.abs() <= 1.0e-12);

    let noisy = fit_a(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[2.0])],
        &[1.0, 2.0, 2.0],
        &[0, 1, 2],
        &FitConfig64::default(),
    );
    assert!((noisy.weights().as_slice()[0] - 0.5).abs() <= 1.0e-12);
    assert!((noisy.intercept() - 7.0 / 6.0).abs() <= 1.0e-12);
    assert!((noisy.diagnostics().rss_f64 - 1.0 / 6.0).abs() <= 1.0e-12);
}

#[test]
fn a_recovers_multifeature_and_no_intercept_models() {
    let features = [
        row64(&[0.0, 0.0]),
        row64(&[1.0, 0.0]),
        row64(&[0.0, 1.0]),
        row64(&[1.0, 1.0]),
        row64(&[2.0, 1.0]),
        row64(&[1.0, 2.0]),
    ];
    let targets = [1.0, 3.0, -2.0, 0.0, 2.0, -3.0];
    let result = fit_a(
        &features,
        &targets,
        &[0, 1, 2, 3, 4, 5],
        &FitConfig64::default(),
    );
    assert!((result.weights().as_slice()[0] - 2.0).abs() <= 1.0e-12);
    assert!((result.weights().as_slice()[1] + 3.0).abs() <= 1.0e-12);
    assert!((result.intercept() - 1.0).abs() <= 1.0e-12);
    assert!(result.diagnostics().rss_f64.abs() <= 1.0e-12);

    let no_intercept = fit_a(
        &[row64(&[1.0]), row64(&[2.0])],
        &[2.0, 4.0],
        &[0, 1],
        &FitConfig64 {
            intercept: false,
            ..FitConfig64::default()
        },
    );
    assert!((no_intercept.weights().as_slice()[0] - 2.0).abs() <= 1.0e-12);
    assert_eq!(no_intercept.intercept(), 0.0);
}

#[test]
fn a_native_f64_discrimination_uses_the_actual_fitted_predictor() {
    let features = [row64(&[16777216.0]), row64(&[16777217.0])];
    let result = fit_a(&features, &[0.0, 1.0], &[0, 1], &FitConfig64::default());
    assert_eq!(result.diagnostics().rank, 2);
    assert_eq!(result.diagnostics().required_rank, 2);

    let predictor = result.predictor();
    let references = [0.0f64, 1.0f64];
    for (index, reference) in references.iter().enumerate() {
        let prediction = predictor
            .predict_checked(&features[index])
            .expect("actual fit prediction is finite");
        assert!(
            (prediction - reference).abs() < 0.5,
            "prediction {prediction} vs {reference}"
        );
    }
}

#[test]
fn a_validation_precedence_and_selected_only_reads() {
    let features = [row64(&[0.0]), row64(&[1.0]), row64(&[f64::NAN])];
    let targets = [1.0, 3.0, f64::NAN];
    let result = fit_a(&features, &targets, &[0, 1], &FitConfig64::default());
    assert_eq!(result.diagnostics().rows, 2);

    let duplicate = fit_f64(&features, &targets, &[0, 0], &FitConfig64::default());
    assert_eq!(duplicate, Err(FitError64::DuplicateIndex { index: 0 }));

    let ragged = fit_f64(
        &[row64(&[1.0]), row64(&[1.0, 2.0])],
        &[1.0, 2.0],
        &[0, 1],
        &FitConfig64::default(),
    );
    assert_eq!(
        ragged,
        Err(FitError64::RaggedFeatures {
            row: 1,
            expected: 1,
            actual: 2,
        })
    );

    let rank_deficient = fit_f64(
        &[row64(&[1.0, 1.0]), row64(&[1.0, 1.0]), row64(&[1.0, 1.0])],
        &[1.0, 2.0, 3.0],
        &[0, 1, 2],
        &FitConfig64::default(),
    );
    assert!(matches!(
        rank_deficient,
        Err(FitError64::RankDeficient { .. })
    ));
}

#[test]
fn a_rank_controls_straddle_threshold_and_scaled_full_rank() {
    let below = fit_f64(
        &[row64(&[0.0]), row64(&[1.0e-17])],
        &[0.0, 1.0e-17],
        &[0, 1],
        &FitConfig64::default(),
    );
    assert!(matches!(below, Err(FitError64::RankDeficient { .. })));

    let above = fit_f64(
        &[row64(&[0.0]), row64(&[1.0e-14])],
        &[0.0, 1.0e-14],
        &[0, 1],
        &FitConfig64::default(),
    )
    .expect("above-threshold fixture is full rank");
    assert_eq!(above.diagnostics().rank, 2);

    let scaled = fit_f64(
        &[row64(&[0.0]), row64(&[1.0e-8])],
        &[1.0, 2.0],
        &[0, 1],
        &FitConfig64::default(),
    )
    .expect("scaled full-rank control succeeds");
    assert_eq!(scaled.diagnostics().rank, 2);
}

#[test]
fn a_overflow_is_a_checked_failure() {
    let overflow = fit_f64(
        &[row64(&[1.0e-308])],
        &[1.0e308],
        &[0],
        &FitConfig64 {
            intercept: false,
            ..FitConfig64::default()
        },
    );
    assert_eq!(overflow, Err(FitError64::NumericalFailure));

    let rss_overflow = fit_f64(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[2.0])],
        &[1.0e155, -1.0e155, 1.0e155],
        &[0, 1, 2],
        &FitConfig64::default(),
    );
    assert_eq!(rss_overflow, Err(FitError64::NonFiniteRss));
}

#[test]
fn a_single_stream_and_group_inference_preserve_bits_and_failure_state() {
    let result = fit_a(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[2.0])],
        &[1.0, 3.0, 5.0],
        &[0, 1, 2],
        &FitConfig64::default(),
    );
    let model = result.into_predictor();
    let observations = [obs64(&[3.0]), obs64(&[4.0]), obs64(&[-1.0])];
    let group = model.predict_batch(&observations).unwrap();
    let checked_group = model.predict_batch_checked(&observations).unwrap();
    for (left, right) in group.iter().zip(&checked_group) {
        assert_eq!(left.to_bits(), right.to_bits());
    }

    let checked = CheckedLinearRegression64::new(&model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f64);
    let streamed: Vec<f64> = observations
        .iter()
        .map(|observation| *executor.process_one(observation).unwrap())
        .collect();
    for (single, streamed) in group.iter().zip(&streamed) {
        assert_eq!(single.to_bits(), streamed.to_bits());
    }

    let overflow_group = [obs64(&[3.0]), obs64(&[1.0e308]), obs64(&[4.0])];
    assert_eq!(
        model.predict_batch_checked(&overflow_group),
        Err(RegressionError::NonFiniteOutput {
            observation: Some(1),
        })
    );
    assert_eq!(model.predict_batch_checked(&[]), Ok(Vec::<f64>::new()));

    let before = *executor.state();
    let overflow = obs64(&[1.0e308]);
    let error = executor.process_one(&overflow).unwrap_err();
    assert_eq!(
        error,
        RegressionError::NonFiniteOutput { observation: None }
    );
    assert_eq!(executor.state().to_bits(), before.to_bits());
}

#[test]
fn b_conversion_preserves_rounding_ties_and_subnormals() {
    let tie_low = one_weight(1.0 + 2.0f64.powi(-24));
    let serving = tie_low
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("tie-low conversion");
    assert_eq!(serving.weights().as_slice()[0].to_bits(), 0x3f80_0000);

    let tie_high = one_weight(1.0 + 3.0 * 2.0f64.powi(-24));
    let serving = tie_high
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("tie-high conversion");
    assert_eq!(serving.weights().as_slice()[0].to_bits(), 0x3f80_0002);

    let subnormal = f32::from_bits(1);
    let result = one_weight(f64::from(subnormal));
    let serving = result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("subnormal conversion");
    assert_eq!(serving.weights().as_slice()[0].to_bits(), 1);
}

#[test]
fn b_underflow_and_overflow_are_checked_and_preserve_the_learner() {
    let small = 2.0f64.powi(-200);
    let learner = one_weight(small);
    let weights_before = learner.weights().clone();
    let bias_before = learner.intercept();
    let diagnostics_before = learner.diagnostics().clone();

    assert_eq!(
        learner.to_serving_f32(UnderflowPolicy::Reject),
        Err(ServingConversionError::Underflow {
            field: ServingField::Weight { index: 0 },
        })
    );
    assert_eq!(learner.weights(), &weights_before);
    assert_eq!(learner.intercept().to_bits(), bias_before.to_bits());
    assert_eq!(learner.diagnostics(), &diagnostics_before);

    let serving = learner
        .to_serving_f32(UnderflowPolicy::AllowLoss)
        .expect("underflow loss is allowed");
    assert_eq!(serving.weights().as_slice()[0].to_bits(), 0);

    let large = 2.0 * f64::from(f32::MAX);
    let learner = one_weight(large);
    assert_eq!(
        learner.to_serving_f32(UnderflowPolicy::Reject),
        Err(ServingConversionError::Overflow {
            field: ServingField::Weight { index: 0 },
        })
    );

    let near_max = f64::from(f32::MAX) * (1.0 + 2.0f64.powi(-26));
    let learner = one_weight(near_max);
    let serving = learner
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("finite source rounding to f32::MAX is accepted");
    assert_eq!(
        serving.weights().as_slice()[0].to_bits(),
        f32::MAX.to_bits()
    );
}

#[test]
fn b_signed_zero_and_serving_inference_are_preserved() {
    let result = one_weight(-0.0);
    let serving = result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("signed zero conversion");
    assert_eq!(serving.weights().as_slice()[0].to_bits(), 0x8000_0000);

    let result = fit_a(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[2.0])],
        &[1.0, 3.0, 5.0],
        &[0, 1, 2],
        &FitConfig64::default(),
    );
    let serving = result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("serving conversion");
    let observations = [obs32(&[3.0]), obs32(&[4.0]), obs32(&[-1.0])];
    let group = serving.predict_batch(&observations).unwrap();
    let checked_group = serving.predict_batch_checked(&observations).unwrap();
    for (left, right) in group.iter().zip(&checked_group) {
        assert_eq!(left.to_bits(), right.to_bits());
    }

    let checked = seqvex::models::linear::ols::CheckedLinearRegression::new(&serving);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let streamed: Vec<f32> = observations
        .iter()
        .map(|observation| *executor.process_one(observation).unwrap())
        .collect();
    for (single, streamed) in group.iter().zip(&streamed) {
        assert_eq!(single.to_bits(), streamed.to_bits());
    }

    let overflow_group = [obs32(&[3.0]), obs32(&[f32::MAX]), obs32(&[4.0])];
    assert_eq!(
        serving.predict_batch_checked(&overflow_group),
        Err(RegressionError::NonFiniteOutput {
            observation: Some(1),
        })
    );

    let active = seqvex::models::linear::ols::LinearRegression::new(row32(&[2.0]), 1.0).unwrap();
    let active_before = active.predict_checked(&row32(&[3.0])).unwrap();
    let learner_before = result.weights().clone();
    let failure = serving.predict_checked(&row32(&[f32::INFINITY]));
    assert!(matches!(
        failure,
        Err(seqvex::models::linear::ols::RegressionError::NonFiniteInput)
    ));
    assert_eq!(result.weights(), &learner_before);
    assert_eq!(
        active.predict_checked(&row32(&[3.0])).unwrap().to_bits(),
        active_before.to_bits()
    );
}

#[test]
fn b_conversion_failure_keeps_the_learner_usable() {
    let learner = one_weight(2.0f64.powi(-200));
    let failure = learner.to_serving_f32(UnderflowPolicy::Reject);
    assert!(failure.is_err());
    let predictor = learner.predictor();
    assert_eq!(predictor.bias(), learner.intercept());
    assert_eq!(predictor.weights(), learner.weights());
}

#[test]
fn c_recovers_f32_models_and_matches_ordered_rss() {
    let exact = fit_c(
        &[row32(&[0.0]), row32(&[1.0]), row32(&[2.0])],
        &[1.0, 3.0, 5.0],
        &[0, 1, 2],
        &FitConfig32::default(),
    );
    assert!((exact.weights().as_slice()[0] - 2.0).abs() <= 1.0e-5);
    assert!((exact.intercept() - 1.0).abs() <= 1.0e-5);

    let weights = exact.weights().as_slice();
    let bias = exact.intercept();
    let features = [row32(&[0.0]), row32(&[1.0]), row32(&[2.0])];
    let targets = [1.0f32, 3.0, 5.0];
    let mut rss = 0.0f32;
    for (&index, target) in [0usize, 1, 2].iter().zip(targets) {
        let prediction = weights
            .iter()
            .zip(features[index].as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + bias;
        let residual = prediction - target;
        rss += residual * residual;
    }
    assert_eq!(exact.diagnostics().rss_f32.to_bits(), rss.to_bits());

    let noisy = fit_c(
        &[row32(&[0.0]), row32(&[1.0]), row32(&[2.0])],
        &[1.0, 2.0, 2.0],
        &[0, 1, 2],
        &FitConfig32::default(),
    );
    assert!((noisy.weights().as_slice()[0] - 0.5).abs() <= 1.0e-5);
    assert!((noisy.intercept() - 7.0 / 6.0).abs() <= 1.0e-5);
}

#[test]
fn c_recovers_multifeature_with_ordered_rss() {
    let features = [
        row32(&[0.0, 0.0]),
        row32(&[1.0, 0.0]),
        row32(&[0.0, 1.0]),
        row32(&[1.0, 1.0]),
        row32(&[2.0, 1.0]),
        row32(&[1.0, 2.0]),
    ];
    let targets = [1.0f32, 3.0, -2.0, 0.0, 2.0, -3.0];
    let result = fit_c(
        &features,
        &targets,
        &[0, 1, 2, 3, 4, 5],
        &FitConfig32::default(),
    );
    let weights = result.weights().as_slice();
    let bias = result.intercept();
    assert!((weights[0] - 2.0).abs() <= 1.0e-5);
    assert!((weights[1] + 3.0).abs() <= 1.0e-5);
    assert!((bias - 1.0).abs() <= 1.0e-5);

    let mut rss = 0.0f32;
    for (row, &target) in features.iter().zip(targets.iter()) {
        let expected = 1.0 + 2.0 * row.as_slice()[0] - 3.0 * row.as_slice()[1];
        let prediction = weights
            .iter()
            .zip(row.as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + bias;
        assert!((prediction - expected).abs() <= 1.0e-5);
        let residual = prediction - target;
        rss += residual * residual;
    }
    assert_eq!(result.diagnostics().rss_f32.to_bits(), rss.to_bits());
}

#[test]
fn c_native_f32_rank_and_inference() {
    let features = [row32(&[0.0]), row32(&[2.0])];
    let result = fit_c(&features, &[1.0, 5.0], &[0, 1], &FitConfig32::default());
    assert_eq!(result.diagnostics().rank, 2);
    let model = result.into_predictor();
    assert!((model.predict_checked(&features[0]).unwrap() - 1.0).abs() < 1.0e-5);
    assert!((model.predict_checked(&features[1]).unwrap() - 5.0).abs() < 1.0e-5);

    let near_dependence = fit_f32(
        &[row32(&[16777216.0]), row32(&[16777218.0])],
        &[0.0, 1.0],
        &[0, 1],
        &FitConfig32::default(),
    );
    assert!(matches!(
        near_dependence,
        Err(FitError32::RankDeficient { .. })
    ));

    let duplicate = fit_f32(
        &[row32(&[1.0, 1.0]), row32(&[1.0, 1.0]), row32(&[1.0, 1.0])],
        &[1.0, 2.0, 3.0],
        &[0, 1, 2],
        &FitConfig32::default(),
    );
    assert!(matches!(duplicate, Err(FitError32::RankDeficient { .. })));

    let observations = [obs32(&[3.0]), obs32(&[4.0]), obs32(&[-1.0])];
    let group = model.predict_batch(&observations).unwrap();
    let checked = seqvex::models::linear::ols::CheckedLinearRegression::new(&model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let streamed: Vec<f32> = observations
        .iter()
        .map(|observation| *executor.process_one(observation).unwrap())
        .collect();
    for (single, streamed) in group.iter().zip(&streamed) {
        assert_eq!(single.to_bits(), streamed.to_bits());
    }

    let overflow_group = [obs32(&[3.0]), obs32(&[f32::MAX]), obs32(&[4.0])];
    assert_eq!(
        model.predict_batch_checked(&overflow_group),
        Err(RegressionError::NonFiniteOutput {
            observation: Some(1),
        })
    );

    let rss_overflow = fit_f32(
        &[row32(&[0.0]), row32(&[1.0]), row32(&[2.0])],
        &[1.0e20, -1.0e20, 1.0e20],
        &[0, 1, 2],
        &FitConfig32::default(),
    );
    assert_eq!(rss_overflow, Err(FitError32::NonFiniteRss));
}

#[test]
fn profiles_coexist_without_shared_mutation() {
    let a_features = [row64(&[0.0]), row64(&[1.0]), row64(&[2.0])];
    let a_targets = [1.0, 3.0, 5.0];
    let a = fit_a(&a_features, &a_targets, &[0, 1, 2], &FitConfig64::default());
    let a_model = a.predictor();
    let b_model = a
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion");

    let c_features = [row32(&[0.0]), row32(&[1.0]), row32(&[2.0])];
    let c = fit_c(
        &c_features,
        &[1.0, 3.0, 5.0],
        &[0, 1, 2],
        &FitConfig32::default(),
    );
    let c_model = c.into_predictor();

    let a_before = a_model.predict_checked(&a_features[2]).unwrap();
    let b_before = b_model.predict_checked(&c_features[2]).unwrap();
    let c_before = c_model.predict_checked(&c_features[2]).unwrap();

    let _other_a = fit_a(
        &[row64(&[0.0]), row64(&[1.0])],
        &[2.0, 4.0],
        &[0, 1],
        &FitConfig64::default(),
    );
    let _other_c = fit_c(
        &[row32(&[0.0]), row32(&[1.0])],
        &[4.0, 8.0],
        &[0, 1],
        &FitConfig32::default(),
    );

    assert_eq!(
        a_model.predict_checked(&a_features[2]).unwrap().to_bits(),
        a_before.to_bits()
    );
    assert_eq!(
        b_model.predict_checked(&c_features[2]).unwrap().to_bits(),
        b_before.to_bits()
    );
    assert_eq!(
        c_model.predict_checked(&c_features[2]).unwrap().to_bits(),
        c_before.to_bits()
    );
}
