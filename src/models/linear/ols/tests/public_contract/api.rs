use super::support::{assert_bits_eq, model, observation};
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::ols::{LinearRegression, RegressionError};

#[test]
fn regression_error_reports_its_known_failure_without_a_source() {
    fn assert_error_traits<
        T: std::error::Error + Clone + Copy + PartialEq + Eq + std::fmt::Debug,
    >() {
    }
    assert_error_traits::<RegressionError>();

    let cases = [
        (RegressionError::ZeroDimension, "zero dimension"),
        (
            RegressionError::NonFiniteParameter,
            "parameters must be finite",
        ),
        (
            RegressionError::NonFiniteInput,
            "input features must be finite",
        ),
    ];
    for (error, expected) in cases {
        assert!(error.to_string().contains(expected), "{error}");
        assert!(std::error::Error::source(&error).is_none());
    }

    let mismatch = RegressionError::DimensionMismatch {
        expected: 17,
        actual: 3,
    };
    assert_eq!(
        mismatch.to_string(),
        "dimension mismatch: expected 17, got 3"
    );
    assert!(std::error::Error::source(&mismatch).is_none());
    assert_eq!(
        RegressionError::NonFiniteOutput { observation: None }.to_string(),
        "prediction produced a non-finite output"
    );
    assert_eq!(
        RegressionError::NonFiniteOutput {
            observation: Some(3)
        }
        .to_string(),
        "group prediction produced a non-finite output at observation 3"
    );
    assert!(
        std::error::Error::source(&RegressionError::NonFiniteOutput {
            observation: Some(0)
        })
        .is_none()
    );
}

#[test]
fn reference_prediction_matches_hand_computed_value() {
    // 1*3 + 2*4 + 0.5 = 11.5, exactly representable in f32.
    let model = model(&[1.0, 2.0], 0.5);
    assert_eq!(model.predict(&Vector::from_slice(&[3.0, 4.0])), Ok(11.5));
}

#[test]
fn zero_weights_are_rejected() {
    assert_eq!(
        LinearRegression::new(Vector::zeros(0), 0.0),
        Err(RegressionError::ZeroDimension)
    );
}

#[test]
fn non_finite_parameter_is_rejected_on_construction() {
    assert_eq!(
        LinearRegression::new(Vector::from_slice(&[1.0, f32::NAN]), 0.0),
        Err(RegressionError::NonFiniteParameter)
    );
    assert_eq!(
        LinearRegression::new(Vector::from_slice(&[1.0]), f32::INFINITY),
        Err(RegressionError::NonFiniteParameter)
    );
}

#[test]
fn dimension_mismatch_is_reported() {
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn non_finite_input_is_rejected() {
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0, f32::NAN])),
        Err(RegressionError::NonFiniteInput)
    );
}

#[test]
fn streaming_prediction_matches_reference() {
    let reference = model(&[0.25, -0.5, 1.5], 0.125);
    let streaming_model = model(&[0.25, -0.5, 1.5], 0.125);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    for features in [
        [1.0_f32, 2.0, 3.0],
        [-1.0, 0.5, 0.25],
        [0.0, 0.0, 0.0],
        [10.0, -3.0, 7.5],
    ] {
        let observation = observation(&features);
        let expected = reference.predict(observation.value()).unwrap();
        let produced = *executor.process_one(&observation).unwrap();
        assert_bits_eq(expected, produced);
    }
}

#[test]
fn streaming_state_is_the_latest_prediction() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[3.0])).unwrap();
    assert_bits_eq(*executor.state(), 7.0);
    executor.process_one(&observation(&[4.0])).unwrap();
    assert_bits_eq(*executor.state(), 9.0);
}

#[test]
fn failed_streaming_prediction_preserves_last_prediction() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[3.0])).unwrap();
    let error = executor.process_one(&observation(&[1.0, 2.0])).unwrap_err();
    assert_eq!(
        error,
        RegressionError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
    );
    assert_bits_eq(*executor.state(), 7.0);
}

#[test]
fn reset_starts_a_new_streaming_sequence() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[3.0])).unwrap();
    assert_bits_eq(*executor.state(), 7.0);
    executor.reset(0.0);
    assert_bits_eq(*executor.state(), 0.0);
}

#[test]
fn state_model_update_matches_reference_prediction() {
    let model = model(&[1.5, -2.0], 0.25);
    let observation = observation(&[4.0, 5.0]);
    let updated = model.update(&0.0, &observation).unwrap();
    assert_bits_eq(updated, model.predict(observation.value()).unwrap());
}

#[test]
fn micro_batch_matches_repeated_single_observation_prediction() {
    let model = model(&[0.5, 1.25, -2.0], -0.75);
    let batch = vec![
        observation(&[1.0, 2.0, 3.0]),
        observation(&[-1.0, 0.5, 0.25]),
        observation(&[0.0, 0.0, 0.0]),
    ];

    let micro_batched = model.predict_batch(&batch).unwrap();
    for (index, observation) in batch.iter().enumerate() {
        assert_bits_eq(
            micro_batched[index],
            model.predict(observation.value()).unwrap(),
        );
    }
}

#[test]
fn micro_batch_matches_streaming_prediction_bitwise() {
    let batch = vec![
        observation(&[1.0, 2.0]),
        observation(&[-3.0, 0.5]),
        observation(&[7.0, -0.25]),
        observation(&[0.125, 8.0]),
    ];

    let model = model(&[1.5, -0.5], 2.0);
    let micro_batched = model.predict_batch(&batch).unwrap();

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    for (index, observation) in batch.iter().enumerate() {
        let streamed = *executor.process_one(observation).unwrap();
        assert_bits_eq(micro_batched[index], streamed);
    }
}

#[test]
fn micro_batch_preserves_observation_order() {
    let model = model(&[2.0, 3.0], 0.0);
    let first = observation(&[1.0, 0.0]);
    let second = observation(&[0.0, 1.0]);

    let in_order = model
        .predict_batch(&[first.clone(), second.clone()])
        .unwrap();
    let reversed = model.predict_batch(&[second, first]).unwrap();

    assert_eq!(in_order, vec![2.0, 3.0]);
    assert_eq!(reversed, vec![3.0, 2.0]);
}

#[test]
fn empty_micro_batch_returns_no_predictions() {
    let model = model(&[1.0], 0.0);
    assert_eq!(model.predict_batch(&[]), Ok(Vec::new()));
}

#[test]
fn micro_batch_propagates_element_failure() {
    let model = model(&[1.0, 1.0], 0.0);
    let batch = vec![
        observation(&[1.0, 1.0]),
        observation(&[1.0]),
        observation(&[1.0, 1.0]),
    ];
    assert_eq!(
        model.predict_batch(&batch),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}
