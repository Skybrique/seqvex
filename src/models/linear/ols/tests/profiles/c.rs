use super::support::{fit_c, obs32, row32};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::models::linear::ols::{FitConfig32, FitError32, RegressionError, fit_f32};

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
