use super::support::{fit_a, obs32, one_weight, row32, row64};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::models::linear::ols::{
    FitConfig64, RegressionError, ServingConversionError, ServingField, UnderflowPolicy, fit_f64,
};

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
