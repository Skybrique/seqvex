use super::support::{fit_a, obs64, row64};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::models::linear::ols::{
    CheckedLinearRegression64, FitConfig64, FitError64, RegressionError, fit_f64,
};

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
