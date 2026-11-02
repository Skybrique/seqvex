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
fn a_direct_validation_failures_are_isolated() {
    let valid_features = [row64(&[0.0]), row64(&[1.0]), row64(&[2.0])];
    let valid_targets = [1.0, 3.0, 5.0];

    assert_eq!(
        fit_f64(
            &valid_features[..2],
            &valid_targets[..1],
            &[0],
            &FitConfig64::default(),
        ),
        Err(FitError64::RowTargetMismatch {
            features: 2,
            targets: 1,
        })
    );
    assert_eq!(
        fit_f64(
            &valid_features,
            &valid_targets,
            &[],
            &FitConfig64::default(),
        ),
        Err(FitError64::TooFewRows)
    );
    assert_eq!(
        fit_f64(
            &valid_features[..2],
            &valid_targets[..2],
            &[2],
            &FitConfig64::default(),
        ),
        Err(FitError64::IndexOutOfRange { index: 2, rows: 2 })
    );
    assert_eq!(
        fit_f64(
            &[row64(&[0.0]), row64(&[1.0, 2.0])],
            &[1.0, 3.0],
            &[0, 1],
            &FitConfig64::default(),
        ),
        Err(FitError64::RaggedFeatures {
            row: 1,
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        fit_f64(
            &valid_features[..2],
            &valid_targets[..2],
            &[0, 0],
            &FitConfig64::default(),
        ),
        Err(FitError64::DuplicateIndex { index: 0 })
    );
    assert_eq!(
        fit_f64(
            &valid_features[..2],
            &valid_targets[..2],
            &[0, 1],
            &FitConfig64 {
                rank_threshold: -1.0,
                ..FitConfig64::default()
            },
        ),
        Err(FitError64::InvalidRankThreshold { value: -1.0 })
    );
    assert_eq!(
        fit_f64(
            &[row64(&[0.0]), row64(&[f64::NAN])],
            &[1.0, 3.0],
            &[0, 1],
            &FitConfig64::default(),
        ),
        Err(FitError64::NonFiniteTrainingValue { row: 1, column: 0 })
    );

    assert_eq!(
        fit_f64(&[row64(&[0.0])], &[1.0], &[0], &FitConfig64::default(),),
        Err(FitError64::RankDeficient {
            rank: 1,
            required: 2,
        }),
        "one row cannot satisfy the two intercept-augmented fitting columns"
    );
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
    assert!(
        (f64::from(noisy.diagnostics().rss_f32) - 1.0 / 6.0).abs() <= 1.0e-6,
        "C noisy native training RSS must match the independent 1/6 reference"
    );
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
fn c_direct_validation_failures_are_isolated() {
    let valid_features = [row32(&[0.0]), row32(&[1.0]), row32(&[2.0])];
    let valid_targets = [1.0, 3.0, 5.0];

    assert_eq!(
        fit_f32(
            &valid_features[..2],
            &valid_targets[..1],
            &[0],
            &FitConfig32::default(),
        ),
        Err(FitError32::RowTargetMismatch {
            features: 2,
            targets: 1,
        })
    );
    assert_eq!(
        fit_f32(
            &valid_features,
            &valid_targets,
            &[],
            &FitConfig32::default(),
        ),
        Err(FitError32::TooFewRows)
    );
    assert_eq!(
        fit_f32(
            &valid_features[..2],
            &valid_targets[..2],
            &[2],
            &FitConfig32::default(),
        ),
        Err(FitError32::IndexOutOfRange { index: 2, rows: 2 })
    );
    assert_eq!(
        fit_f32(
            &[row32(&[0.0]), row32(&[1.0, 2.0])],
            &[1.0, 3.0],
            &[0, 1],
            &FitConfig32::default(),
        ),
        Err(FitError32::RaggedFeatures {
            row: 1,
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        fit_f32(
            &valid_features[..2],
            &valid_targets[..2],
            &[0, 0],
            &FitConfig32::default(),
        ),
        Err(FitError32::DuplicateIndex { index: 0 })
    );
    assert_eq!(
        fit_f32(
            &valid_features[..2],
            &valid_targets[..2],
            &[0, 1],
            &FitConfig32 {
                rank_threshold: -1.0,
                ..FitConfig32::default()
            },
        ),
        Err(FitError32::InvalidRankThreshold { value: -1.0 })
    );
    assert_eq!(
        fit_f32(
            &[row32(&[0.0]), row32(&[f32::NAN])],
            &[1.0, 3.0],
            &[0, 1],
            &FitConfig32::default(),
        ),
        Err(FitError32::NonFiniteTrainingValue { row: 1, column: 0 })
    );

    assert_eq!(
        fit_f32(&[row32(&[0.0])], &[1.0], &[0], &FitConfig32::default(),),
        Err(FitError32::RankDeficient {
            rank: 1,
            required: 2,
        }),
        "one row cannot satisfy the two intercept-augmented fitting columns"
    );
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

#[test]
fn a_and_c_ignore_unselected_nonfinite_values_and_use_one_feature() {
    let features64 = [row64(&[0.0]), row64(&[1.0]), row64(&[f64::NAN])];
    let targets64 = [1.0, 3.0, f64::NAN];
    let a = fit_a(&features64, &targets64, &[0, 1], &FitConfig64::default());
    let mut changed_features64 = features64.clone();
    changed_features64[2] = row64(&[f64::INFINITY]);
    let changed_targets64 = [1.0, 3.0, f64::NEG_INFINITY];
    let a_changed = fit_a(
        &changed_features64,
        &changed_targets64,
        &[0, 1],
        &FitConfig64::default(),
    );
    assert_eq!(a, a_changed);
    assert_eq!(a.weights().len(), 1);
    assert_eq!(a.diagnostics().features, 1);
    assert_eq!(a.diagnostics().rank, 2);
    assert_eq!(a.diagnostics().required_rank, 2);

    let selected_change = fit_a(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[f64::NAN])],
        &[1.0, 4.0, f64::NAN],
        &[0, 1],
        &FitConfig64::default(),
    );
    assert_ne!(a.weights(), selected_change.weights());

    let features32 = [row32(&[0.0]), row32(&[1.0]), row32(&[f32::NAN])];
    let targets32 = [1.0, 3.0, f32::NAN];
    let c = fit_c(&features32, &targets32, &[0, 1], &FitConfig32::default());
    let changed_features32 = [row32(&[0.0]), row32(&[1.0]), row32(&[f32::INFINITY])];
    let changed_targets32 = [1.0, 3.0, f32::NEG_INFINITY];
    let c_changed = fit_c(
        &changed_features32,
        &changed_targets32,
        &[0, 1],
        &FitConfig32::default(),
    );
    assert_eq!(c, c_changed);
    assert_eq!(c.weights().len(), 1);
    assert_eq!(c.diagnostics().features, 1);
    assert_eq!(c.diagnostics().rank, 2);
    assert_eq!(c.diagnostics().required_rank, 2);

    let selected_change = fit_c(
        &[row32(&[0.0]), row32(&[1.0]), row32(&[f32::NAN])],
        &[1.0, 4.0, f32::NAN],
        &[0, 1],
        &FitConfig32::default(),
    );
    assert_ne!(c.weights(), selected_change.weights());
}

#[test]
fn a_and_c_enforce_requested_workspace_boundary() {
    let features64 = [row64(&[0.0]), row64(&[1.0]), row64(&[2.0])];
    let targets64 = [1.0, 3.0, 5.0];
    let baseline_a = fit_a(&features64, &targets64, &[0, 1, 2], &FitConfig64::default());
    let requested_a = baseline_a.diagnostics().requested_workspace_bytes;
    let exact_a = fit_f64(
        &features64,
        &targets64,
        &[0, 1, 2],
        &FitConfig64 {
            max_workspace_bytes: requested_a,
            ..FitConfig64::default()
        },
    )
    .expect("A fits at its requested workspace boundary");
    assert_eq!(exact_a.diagnostics().requested_workspace_bytes, requested_a);
    assert_eq!(
        fit_f64(
            &features64,
            &targets64,
            &[0, 1, 2],
            &FitConfig64 {
                max_workspace_bytes: requested_a - 1,
                ..FitConfig64::default()
            },
        ),
        Err(FitError64::Capacity {
            requested: requested_a,
            limit: requested_a - 1,
        })
    );

    let features32 = [row32(&[0.0]), row32(&[1.0]), row32(&[2.0])];
    let targets32 = [1.0, 3.0, 5.0];
    let baseline_c = fit_c(&features32, &targets32, &[0, 1, 2], &FitConfig32::default());
    let requested_c = baseline_c.diagnostics().requested_workspace_bytes;
    let exact_c = fit_f32(
        &features32,
        &targets32,
        &[0, 1, 2],
        &FitConfig32 {
            max_workspace_bytes: requested_c,
            ..FitConfig32::default()
        },
    )
    .expect("C fits at its requested workspace boundary");
    assert_eq!(exact_c.diagnostics().requested_workspace_bytes, requested_c);
    assert_eq!(
        fit_f32(
            &features32,
            &targets32,
            &[0, 1, 2],
            &FitConfig32 {
                max_workspace_bytes: requested_c - 1,
                ..FitConfig32::default()
            },
        ),
        Err(FitError32::Capacity {
            requested: requested_c,
            limit: requested_c - 1,
        })
    );
}

#[test]
fn native_profiles_preserve_index_before_rank_configuration_precedence() {
    let features64 = [row64(&[0.0]), row64(&[1.0])];
    let targets64 = [0.0, 1.0];
    let config64 = FitConfig64 {
        rank_threshold: f64::NAN,
        max_workspace_bytes: 0,
        ..FitConfig64::default()
    };
    assert_eq!(
        fit_f64(&features64, &targets64, &[2], &config64),
        Err(FitError64::IndexOutOfRange { index: 2, rows: 2 })
    );

    let features32 = [row32(&[0.0]), row32(&[1.0])];
    let targets32 = [0.0, 1.0];
    let config32 = FitConfig32 {
        rank_threshold: f32::NAN,
        max_workspace_bytes: 0,
        ..FitConfig32::default()
    };
    assert_eq!(
        fit_f32(&features32, &targets32, &[2], &config32),
        Err(FitError32::IndexOutOfRange { index: 2, rows: 2 })
    );
}

#[test]
fn c_rank_threshold_has_controls_on_each_side() {
    let below = fit_f32(
        &[row32(&[0.0]), row32(&[1.0e-8])],
        &[0.0, 1.0e-8],
        &[0, 1],
        &FitConfig32::default(),
    );
    assert!(matches!(below, Err(FitError32::RankDeficient { .. })));

    let above = fit_f32(
        &[row32(&[0.0]), row32(&[1.0e-5])],
        &[0.0, 1.0e-5],
        &[0, 1],
        &FitConfig32::default(),
    )
    .expect("C above-threshold fixture is full rank");
    assert_eq!(above.diagnostics().features, 1);
    assert_eq!(above.diagnostics().rank, 2);
    assert_eq!(above.diagnostics().required_rank, 2);
    assert!(above.diagnostics().rank_threshold.is_finite());
}

#[test]
fn b_uses_independent_rounding_reference_and_survives_source_drop_and_refit() {
    let features = [row64(&[0.0]), row64(&[1.0]), row64(&[2.0])];
    let targets = [1.0, 3.0, 5.0];
    let a = fit_a(&features, &targets, &[0, 1, 2], &FitConfig64::default());
    let weights_before: Vec<u64> = a
        .weights()
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let bias_before = a.intercept().to_bits();
    let diagnostics_before = a.diagnostics().clone();
    let rounded_weights: Vec<f32> = a
        .weights()
        .as_slice()
        .iter()
        .map(|&value| value as f32)
        .collect();
    let rounded_bias = a.intercept() as f32;
    let b = a
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion");
    assert_eq!(b.weights().len(), rounded_weights.len());
    for (index, (&actual, &expected)) in b
        .weights()
        .as_slice()
        .iter()
        .zip(&rounded_weights)
        .enumerate()
    {
        assert_eq!(actual.to_bits(), expected.to_bits(), "B weight {index}");
    }
    assert_eq!(b.bias().to_bits(), rounded_bias.to_bits());
    assert_eq!(
        a.weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        weights_before
    );
    assert_eq!(a.intercept().to_bits(), bias_before);
    assert_eq!(a.diagnostics(), &diagnostics_before);

    let reference =
        seqvex::models::linear::ols::LinearRegression::new(row32(&rounded_weights), rounded_bias)
            .unwrap();
    for value in [0.0f32, 1.0, 2.5, 7.1] {
        let row = row32(&[value]);
        let expected = {
            let mut sum = -0.0f32;
            for (&weight, &feature) in reference.weights().as_slice().iter().zip(row.as_slice()) {
                sum += weight * feature;
            }
            sum + reference.bias()
        };
        assert_eq!(
            b.predict_checked(&row).unwrap().to_bits(),
            expected.to_bits()
        );
    }

    let selected_before = b.predict_checked(&row32(&[1.0])).unwrap();
    let shifted_targets = [101.0, 103.0, 105.0];
    let _shifted = fit_f64(
        &features,
        &shifted_targets,
        &[0, 1, 2],
        &FitConfig64::default(),
    )
    .expect("separate shifted refit");
    assert_eq!(
        b.predict_checked(&row32(&[1.0])).unwrap().to_bits(),
        selected_before.to_bits()
    );
}

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

fn br64(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite fixture")
}

fn br32(value: f32) -> BigRational {
    BigRational::from_float(f64::from(value)).expect("finite fixture")
}

fn rat_int(value: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(value))
}

#[allow(clippy::needless_range_loop)]
fn exact_rank(matrix: &[Vec<BigRational>]) -> usize {
    let rows = matrix.len();
    if rows == 0 {
        return 0;
    }
    let cols = matrix[0].len();
    let mut a = matrix.to_vec();
    let mut rank = 0usize;
    for col in 0..cols {
        let Some(pivot_row) = (rank..rows).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };
        a.swap(rank, pivot_row);
        let pivot = a[rank][col].clone();
        for r in 0..rows {
            if r != rank && !a[r][col].is_zero() {
                let factor = &a[r][col] / &pivot;
                for c in col..cols {
                    let sub = &factor * &a[rank][c];
                    a[r][c] = &a[r][c] - &sub;
                }
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

fn exact_rank64(xs: &[[f64; 3]], intercept: bool) -> usize {
    let matrix = xs
        .iter()
        .map(|row| {
            let mut columns = Vec::with_capacity(if intercept { 4 } else { 3 });
            if intercept {
                columns.push(rat_int(1));
            }
            columns.extend(row.iter().map(|&value| br64(value)));
            columns
        })
        .collect::<Vec<_>>();
    exact_rank(&matrix)
}

fn exact_rank32(xs: &[[f32; 3]], intercept: bool) -> usize {
    let matrix = xs
        .iter()
        .map(|row| {
            let mut columns = Vec::with_capacity(if intercept { 4 } else { 3 });
            if intercept {
                columns.push(rat_int(1));
            }
            columns.extend(row.iter().map(|&value| br32(value)));
            columns
        })
        .collect::<Vec<_>>();
    exact_rank(&matrix)
}

fn rows64(xs: &[[f64; 3]]) -> Vec<Vector64> {
    xs.iter().map(|row| row64(row)).collect()
}

fn rows32(xs: &[[f32; 3]]) -> Vec<Vector> {
    xs.iter().map(|row| row32(row)).collect()
}

fn fit_a_result(
    xs: &[[f64; 3]],
    ys: &[f64],
    intercept: bool,
) -> Result<seqvex::models::linear::ols::FitResult64, FitError64> {
    let indices: Vec<usize> = (0..xs.len()).collect();
    fit_f64(
        &rows64(xs),
        ys,
        &indices,
        &FitConfig64 {
            intercept,
            rank_threshold: 1.0,
            ..FitConfig64::default()
        },
    )
}

fn fit_c_result(
    xs: &[[f32; 3]],
    ys: &[f32],
    intercept: bool,
) -> Result<seqvex::models::linear::ols::FitResult32, FitError32> {
    let indices: Vec<usize> = (0..xs.len()).collect();
    fit_f32(
        &rows32(xs),
        ys,
        &indices,
        &FitConfig32 {
            intercept,
            rank_threshold: 1.0,
            ..FitConfig32::default()
        },
    )
}

fn assert_a_rank_deficient(xs: &[[f64; 3]], ys: &[f64], intercept: bool, required: usize) {
    match fit_a_result(xs, ys, intercept) {
        Err(FitError64::RankDeficient {
            rank,
            required: got,
        }) => {
            assert_eq!(got, required);
            assert!(rank < required);
        }
        other => panic!("A expected RankDeficient required={required}, got {other:?}"),
    }
}

fn assert_c_rank_deficient(xs: &[[f32; 3]], ys: &[f32], intercept: bool, required: usize) {
    match fit_c_result(xs, ys, intercept) {
        Err(FitError32::RankDeficient {
            rank,
            required: got,
        }) => {
            assert_eq!(got, required);
            assert!(rank < required);
        }
        other => panic!("C expected RankDeficient required={required}, got {other:?}"),
    }
}

fn assert_a_weights(
    result: &seqvex::models::linear::ols::FitResult64,
    expected: &[f64],
    bias: f64,
) {
    assert_eq!(result.weights().len(), expected.len());
    for (index, (&actual, &wanted)) in result.weights().as_slice().iter().zip(expected).enumerate()
    {
        assert!(
            (actual - wanted).abs() <= 1.0e-12,
            "A weight {index}: {actual} vs {wanted}"
        );
    }
    assert!((result.intercept() - bias).abs() <= 1.0e-12);
}

fn assert_c_weights(
    result: &seqvex::models::linear::ols::FitResult32,
    expected: &[f32],
    bias: f32,
) {
    assert_eq!(result.weights().len(), expected.len());
    for (index, (&actual, &wanted)) in result.weights().as_slice().iter().zip(expected).enumerate()
    {
        assert!(
            (f64::from(actual) - f64::from(wanted)).abs() <= 1.0e-5,
            "C weight {index}: {actual} vs {wanted}"
        );
    }
    assert!((f64::from(result.intercept()) - f64::from(bias)).abs() <= 1.0e-5);
}

#[test]
fn faer_265_public_controls_use_explicit_policy_and_independent_rank() {
    let original = [
        [
            f64::from_bits(0x4341c37937e08000),
            f64::from_bits(0xc30c6bf526340000),
            0.0,
        ],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let original32 = original.map(|row| row.map(|value| value as f32));
    let zero3 = [0.0f64; 3];
    let zero3f = [0.0f32; 3];
    assert_eq!(exact_rank64(&original, false), 3);
    assert_eq!(exact_rank32(&original32, false), 3);
    assert_a_rank_deficient(&original, &zero3, false, 3);
    assert_c_rank_deficient(&original32, &zero3f, false, 3);
    assert_a_rank_deficient(&original, &zero3, true, 4);
    assert_c_rank_deficient(&original32, &zero3f, true, 4);

    let augmented = [original[0], original[1], original[2], [0.0, 0.0, 0.0]];
    let augmented32 = augmented.map(|row| row.map(|value| value as f32));
    let zero4 = [0.0f64; 4];
    let zero4f = [0.0f32; 4];
    assert_eq!(exact_rank64(&augmented, true), 4);
    assert_eq!(exact_rank32(&augmented32, true), 4);
    assert_a_rank_deficient(&augmented, &zero4, true, 4);
    assert_c_rank_deficient(&augmented32, &zero4f, true, 4);

    let rank_one = [
        [1.0e6, 1.0e6, 1.0e6],
        [1.0e3, 1.0e3, 1.0e3],
        [1.0, 1.0, 1.0],
    ];
    let rank_one32 = rank_one.map(|row| row.map(|value| value as f32));
    assert_eq!(exact_rank64(&rank_one, false), 1);
    assert_eq!(exact_rank32(&rank_one32, false), 1);
    assert_a_rank_deficient(&rank_one, &zero3, false, 3);
    assert_c_rank_deficient(&rank_one32, &zero3f, false, 3);
    assert_a_rank_deficient(&rank_one, &zero3, true, 4);
    assert_c_rank_deficient(&rank_one32, &zero3f, true, 4);
}

#[test]
fn moderate_controls_solve_and_restore_feature_order() {
    let x = [[2.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let y = [0.0, 2.0, 3.0];
    assert_eq!(exact_rank64(&x, false), 3);
    assert_eq!(exact_rank32(&x.map(|row| row.map(|v| v as f32)), false), 3);
    let a = fit_a_result(&x, &y, false).expect("moderate A fit");
    assert_eq!(a.diagnostics().rank, 3);
    assert_a_weights(&a, &[1.0, 2.0, 3.0], 0.0);
    assert!(a.diagnostics().rss_f64.abs() <= 2.0 * 3.0 * 1.0e-22);
    let a_model = a.into_predictor();
    for (row, target) in rows64(&x).iter().zip(y) {
        assert!((a_model.predict_checked(row).unwrap() - target).abs() <= 1.0e-11);
    }

    let x32 = x.map(|row| row.map(|value| value as f32));
    let y32 = y.map(|value| value as f32);
    let c = fit_c_result(&x32, &y32, false).expect("moderate C fit");
    assert_eq!(c.diagnostics().rank, 3);
    assert_c_weights(&c, &[1.0, 2.0, 3.0], 0.0);
    assert!(c.diagnostics().rss_f32.abs() <= 2.0 * 3.0 * 1.0e-8);
    let c_model = c.into_predictor();
    for (row, target) in rows32(&x32).iter().zip(y32) {
        assert!((c_model.predict_checked(row).unwrap() - target).abs() <= 1.0e-4);
    }

    let x_aug = [x[0], x[1], x[2], [0.0, 0.0, 0.0]];
    let y_aug = [4.0, 6.0, 7.0, 4.0];
    assert_eq!(exact_rank64(&x_aug, true), 4);
    let a_aug = fit_a_result(&x_aug, &y_aug, true).expect("moderate intercept A fit");
    assert_eq!(a_aug.diagnostics().required_rank, 4);
    assert_a_weights(&a_aug, &[1.0, 2.0, 3.0], 4.0);
    assert!(a_aug.diagnostics().rss_f64.abs() <= 2.0 * 4.0 * 1.0e-22);

    let x_aug32 = x_aug.map(|row| row.map(|value| value as f32));
    let y_aug32 = y_aug.map(|value| value as f32);
    let c_aug = fit_c_result(&x_aug32, &y_aug32, true).expect("moderate intercept C fit");
    assert_eq!(c_aug.diagnostics().required_rank, 4);
    assert_c_weights(&c_aug, &[1.0, 2.0, 3.0], 4.0);
    assert!(c_aug.diagnostics().rss_f32.abs() <= 2.0 * 4.0 * 1.0e-8);

    let permuted = x.map(|row| [row[2], row[0], row[1]]);
    let permuted32 = x_aug32.map(|row| [row[2], row[0], row[1]]);
    assert_eq!(exact_rank64(&permuted, false), 3);
    let permuted_a = fit_a_result(&permuted, &y, false).expect("permuted A fit");
    assert_a_weights(&permuted_a, &[3.0, 1.0, 2.0], 0.0);
    let permuted_a_model = permuted_a.into_predictor();
    for (row, target) in rows64(&permuted).iter().zip(y) {
        assert!((permuted_a_model.predict_checked(row).unwrap() - target).abs() <= 1.0e-11);
    }
    let permuted_c = fit_c_result(&permuted32, &y_aug32, true).expect("permuted C fit");
    assert_c_weights(&permuted_c, &[3.0, 1.0, 2.0], 4.0);
}
