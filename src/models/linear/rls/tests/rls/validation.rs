//! RLS validation tests: dimensions, parameters, finiteness, denominators,
//! public error contracts, admission, checked prediction and failure atomicity.

use super::support::*;
use super::*;

#[test]
fn zero_dimension_is_rejected() {
    assert_eq!(Rls::new(0, 0.99, 1.0), Err(RlsError::ZeroDimension));
}

#[test]
fn dimension_mismatch_is_reported_for_features_weights_and_covariance() {
    let model = Rls::new(2, 0.99, 1.0).unwrap();
    let good = model.initial_state();

    assert_eq!(
        model.update(&good, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_w = state(&[0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
    assert_eq!(
        model
            .update(&short_w, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_rows = state(&[0.0, 0.0], &[&[1.0, 0.0]]);
    assert_eq!(
        model
            .update(&short_rows, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_cols = state(&[0.0, 0.0], &[&[1.0], &[0.0]]);
    assert_eq!(
        model
            .update(&short_cols, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    assert_eq!(
        model
            .predict(&good, &Vector::from_slice(&[1.0]))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn invalid_forgetting_factor_is_rejected() {
    for lambda in [0.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Rls::new(1, lambda, 1.0),
            Err(RlsError::InvalidForgettingFactor),
            "lambda = {lambda}"
        );
    }
    assert!(Rls::new(1, 1.0, 1.0).is_ok());
}

#[test]
fn invalid_initial_covariance_is_rejected() {
    for delta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Rls::new(1, 0.5, delta),
            Err(RlsError::InvalidInitialCovariance),
            "delta = {delta}"
        );
    }
}

#[test]
fn non_finite_observation_is_rejected() {
    let model = Rls::new(2, 1.0, 1.0).unwrap();
    let state = model.initial_state();
    assert_eq!(
        model
            .update(&state, &sample(&[f32::NAN, 0.0], 0.0))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
    assert_eq!(
        model
            .update(&state, &sample(&[1.0, 2.0], f32::INFINITY))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
    assert_eq!(
        model
            .predict(&state, &Vector::from_slice(&[f32::INFINITY, 0.0]))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
}

#[test]
fn non_positive_denominator_is_rejected() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    // P = [-1] makes d = λ + xᵀPx = 1 - 1 = 0.
    let corrupted = state(&[0.0], &[&[-1.0]]);
    assert_eq!(
        model.update(&corrupted, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::InvalidDenominator
    );

    // A non-finite P makes d non-finite.
    let nan_p = state(&[0.0], &[&[f32::NAN]]);
    assert_eq!(
        model.update(&nan_p, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::InvalidDenominator
    );
}

#[test]
fn non_finite_candidate_is_rejected() {
    // A tiny δ keeps d finite while w = f32::MAX overflows the prediction error.
    let model = Rls::new(1, 1.0, 1.0e-30).unwrap();
    let explosive = state(&[f32::MAX], &[&[1.0e-30]]);
    let before = explosive.clone();
    assert_eq!(
        model
            .update(&explosive, &sample(&[1.0], -f32::MAX))
            .unwrap_err(),
        RlsError::NonFiniteCandidate
    );
    assert_state_bitwise_eq(&explosive, &before);
}

#[test]
fn rls_error_display_and_batch_cause_contract() {
    fn assert_leaf_error<T: std::error::Error + Clone + Copy + PartialEq + Eq + std::fmt::Debug>() {
    }
    fn assert_batch_error<T: std::error::Error + Clone + PartialEq + std::fmt::Debug>() {}
    assert_leaf_error::<RlsError>();
    assert_batch_error::<RlsBatchError>();

    let mismatch = RlsError::DimensionMismatch {
        expected: 17,
        actual: 3,
    };
    assert_eq!(
        mismatch.to_string(),
        "dimension mismatch: expected 17, got 3"
    );
    assert!(std::error::Error::source(&mismatch).is_none());

    for (error, designated) in [
        (RlsError::ZeroDimension, "dimension must be nonzero"),
        (
            RlsError::InvalidForgettingFactor,
            "forgetting factor must be finite and in (0, 1]",
        ),
        (
            RlsError::InvalidInitialCovariance,
            "initial covariance scale must be finite and positive",
        ),
        (
            RlsError::NonFiniteInput,
            "observation features and target must be finite",
        ),
        (
            RlsError::InvalidDenominator,
            "denominator must be finite and positive",
        ),
        (
            RlsError::NonFiniteOutput,
            "RLS prediction produced a non-finite output",
        ),
    ] {
        let message = error.to_string();
        assert!(message.contains(designated), "{message:?}");
        assert!(std::error::Error::source(&error).is_none());
    }

    assert_eq!(
        RlsError64::NonFiniteOutput.to_string(),
        "RLS prediction produced a non-finite output"
    );
    assert!(std::error::Error::source(&RlsError64::NonFiniteOutput).is_none());

    let candidate = RlsError::NonFiniteCandidate;
    assert_eq!(
        candidate.to_string(),
        "RLS candidate state is not finite. The candidate update was not committed; the supplied state is unchanged."
    );
    assert!(std::error::Error::source(&candidate).is_none());

    let supplied = state(&[12345.0], &[&[67890.0]]);
    let oversized = RlsBatchError::BatchTooLarge {
        max: 2,
        actual: 17,
        state: supplied,
    };
    let message = oversized.to_string();
    assert!(message.contains("17"));
    assert!(message.contains("2"));
    assert!(message.contains("no observation was processed"));
    assert!(message.contains("returned unchanged"));
    assert!(!message.contains("12345"));
    assert!(!message.contains("67890"));
    assert!(std::error::Error::source(&oversized).is_none());

    let cause = RlsError::InvalidForgettingFactor;
    let transition = RlsBatchError::Transition {
        failed_index: 2,
        state: Rls::new(1, 0.9, 1.0).unwrap().initial_state(),
        error: cause,
    };
    let message = transition.to_string();
    assert!(message.contains("observation 2"));
    assert!(message.contains("2 observations were committed"));
    assert!(message.contains("failed candidate update was not committed"));
    assert!(message.contains("no whole-batch rollback"));
    assert!(!message.contains(&cause.to_string()));
    let stored = match &transition {
        RlsBatchError::Transition { error, .. } => error,
        RlsBatchError::BatchTooLarge { .. } => unreachable!(),
    };
    let source = std::error::Error::source(&transition).unwrap();
    let source = source.downcast_ref::<RlsError>().unwrap();
    assert!(std::ptr::eq(source, stored));
}

#[test]
fn failed_update_leaves_committed_state_bitwise_unchanged() {
    let model = Rls::new(2, 0.99, 10.0).unwrap();
    let mut committed = model
        .update(&model.initial_state(), &sample(&[1.0, 2.0], 1.0))
        .unwrap();
    let snapshot = committed.clone();

    assert!(model.update(&committed, &sample(&[1.0], 0.0)).is_err());
    assert_state_bitwise_eq(&committed, &snapshot);

    assert!(
        model
            .update(&committed, &sample(&[f32::NAN, 0.0], 0.0))
            .is_err()
    );
    assert_state_bitwise_eq(&committed, &snapshot);

    // The next valid observation continues from the untouched committed state.
    committed = model
        .update(&committed, &sample(&[0.5, -0.5], 1.0))
        .unwrap();
    let expected = model.update(&snapshot, &sample(&[0.5, -0.5], 1.0)).unwrap();
    assert_state_bitwise_eq(&committed, &expected);
}

#[test]
fn executor_failure_preserves_state_and_the_stream_continues() {
    let streaming_model = Rls::new(2, 0.99, 10.0).unwrap();
    let reference = streaming_model.clone();
    let initial = streaming_model.initial_state();
    let mut executor = StreamingExecutor::new(&streaming_model, initial);

    executor.process_one(&sample(&[1.0, 2.0], 1.0)).unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor.process_one(&sample(&[1.0], 0.0)).unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    assert_state_bitwise_eq(executor.state(), &committed);

    executor.process_one(&sample(&[0.5, -0.5], 1.0)).unwrap();
    let expected = reference
        .update(&committed, &sample(&[0.5, -0.5], 1.0))
        .unwrap();
    assert_state_bitwise_eq(executor.state(), &expected);
}

#[test]
fn checked_prediction_guards_non_finite_output_and_preserves_finite_parity() {
    // Frozen deterministic public-update fixture: D = 1, lambda = 1, delta = 1.
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    let state = model
        .update(&model.initial_state(), &sample(&[1.0_f32], 1.0e30_f32))
        .expect("frozen public update must succeed");
    // The committed weights and covariance are finite.
    assert!(state.w.as_slice().iter().all(|value| value.is_finite()));
    assert!(state.p.as_slice().iter().all(|value| value.is_finite()));

    let overflow = Vector::from_slice(&[1.0e9_f32]);
    // Ordinary prediction rejects the non-finite output.
    let snapshot = state.clone();
    assert_eq!(
        model.predict(&state, &overflow),
        Err(RlsError::NonFiniteOutput)
    );
    assert_eq!(
        model.predict_checked(&state, &overflow),
        Err(RlsError::NonFiniteOutput)
    );
    // Prediction is non-mutating: the supplied state is unchanged.
    assert_eq!(state.w.as_slice(), snapshot.w.as_slice());
    assert_eq!(state.p.as_slice(), snapshot.p.as_slice());

    // Finite outputs keep bit-for-bit parity with bare prediction.
    let finite = Vector::from_slice(&[1.0_f32]);
    let bare_finite = model.predict(&state, &finite).unwrap();
    let checked_finite = model.predict_checked(&state, &finite).unwrap();
    assert!(checked_finite.is_finite());
    assert_eq!(bare_finite.to_bits(), checked_finite.to_bits());

    // Inherited validation precedence: dimension before output checking.
    assert_eq!(
        model.predict_checked(&state, &Vector::from_slice(&[1.0_f32, 2.0_f32])),
        Err(RlsError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
    // Non-finite features are rejected before any output check.
    assert_eq!(
        model.predict_checked(&state, &Vector::from_slice(&[f32::NAN])),
        Err(RlsError::NonFiniteInput)
    );
}

fn reference_from_state64(state: &RlsState64) -> RefState64 {
    RefState64 {
        w: state.weights().as_slice().to_vec(),
        p: (0..state.covariance().rows())
            .map(|row| {
                (0..state.covariance().cols())
                    .map(|column| state.covariance().get(row, column).unwrap())
                    .collect()
            })
            .collect(),
    }
}

#[test]
fn admission_precedence_and_non_certification_are_explicit() {
    let zero =
        RlsState64::admit(Vector64::zeros(0), Matrix64::from_fn(0, 5, |_, _| 1.0)).unwrap_err();
    assert_eq!(zero, RlsStateAdmissionError::ZeroDimension);

    let not_square = RlsState64::admit(
        Vector64::from_slice(&[1.0, 2.0]),
        Matrix64::from_fn(2, 3, |row, column| (row + column) as f64),
    )
    .unwrap_err();
    assert_eq!(
        not_square,
        RlsStateAdmissionError::CovarianceNotSquare { rows: 2, cols: 3 }
    );

    let wrong_length = RlsState64::admit(
        Vector64::from_slice(&[f64::NAN]),
        Matrix64::from_fn(2, 2, |row, column| if row == column { 1.0 } else { 0.0 }),
    )
    .unwrap_err();
    assert_eq!(
        wrong_length,
        RlsStateAdmissionError::WeightDimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let bad_weight = RlsState64::admit(
        Vector64::from_slice(&[1.0, f64::NAN, f64::INFINITY]),
        Matrix64::from_fn(3, 3, |row, column| if row == column { 1.0 } else { 0.0 }),
    )
    .unwrap_err();
    assert_eq!(
        bad_weight,
        RlsStateAdmissionError::NonFiniteWeight { index: 1 }
    );

    let bad_covariance = RlsState64::admit(
        Vector64::from_slice(&[1.0, 2.0]),
        Matrix64::from_rows(&[&[1.0, f64::NAN], &[0.0, 1.0]]).unwrap(),
    )
    .unwrap_err();
    assert_eq!(
        bad_covariance,
        RlsStateAdmissionError::NonFiniteCovariance { row: 0, column: 1 }
    );

    let signed_zero_asymmetry = RlsState64::admit(
        Vector64::from_slice(&[1.0, 2.0]),
        Matrix64::from_rows(&[&[1.0, 0.0], &[-0.0, 1.0]]).unwrap(),
    )
    .unwrap_err();
    assert_eq!(
        signed_zero_asymmetry,
        RlsStateAdmissionError::AsymmetricCovariance { row: 0, column: 1 }
    );

    let indefinite = RlsState64::admit(
        Vector64::from_slice(&[0.0, 0.0]),
        Matrix64::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]).unwrap(),
    )
    .unwrap();
    assert_eq!(indefinite.covariance().get(0, 0), Some(-1.0));

    let c_indefinite = RlsState::admit(
        Vector::from_slice(&[0.0, 0.0]),
        Matrix::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]).unwrap(),
    )
    .unwrap();
    assert_eq!(c_indefinite.p.get(0, 0), Some(-1.0));

    let mut mutated = c_indefinite;
    mutated.w = Vector::from_slice(&[1.0, 2.0]);
    let c_model = Rls::new(2, 1.0, 1.0).unwrap();
    assert_eq!(
        c_model
            .predict(&mutated, &Vector::from_slice(&[1.0, 1.0]))
            .unwrap(),
        3.0
    );

    let bypassed = RlsState {
        w: Vector::from_slice(&[0.0, 0.0]),
        p: Matrix::from_rows(&[&[1.0, 0.0], &[1.0, 1.0]]).unwrap(),
    };
    assert!(
        c_model.update(&bypassed, &sample(&[1.0, 1.0], 0.0)).is_ok(),
        "legacy update must not acquire a new admission check"
    );
}

fn admitted64(weights: &[f64], diagonal: &[f64]) -> RlsState64 {
    RlsState64::admit(Vector64::from_slice(weights), diagonal64(diagonal)).unwrap()
}

#[test]
fn failure_preservation_candidate_overflow_and_continuation_are_exact() {
    let (model, learned) = learned_state64();
    let learned_snapshot = learned.clone();

    let shape_error = model.update(&learned, &sample64(&[1.0, 2.0, 3.0], 1.0));
    assert_eq!(
        shape_error,
        Err(RlsError64::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
    assert_state64_bitwise_eq(&learned, &learned_snapshot);

    assert_eq!(
        model.update(&learned, &sample64(&[f64::NAN, 0.0], 1.0)),
        Err(RlsError64::NonFiniteInput)
    );
    assert_state64_bitwise_eq(&learned, &learned_snapshot);
    assert_eq!(
        model.update(&learned, &sample64(&[1.0, 0.0], f64::NAN)),
        Err(RlsError64::NonFiniteInput)
    );
    assert_state64_bitwise_eq(&learned, &learned_snapshot);

    let indefinite = RlsState64::admit(
        learned.weights().clone(),
        Matrix64::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]).unwrap(),
    )
    .unwrap();
    let indefinite_snapshot = indefinite.clone();
    assert_eq!(
        model.update(&indefinite, &sample64(&[1.0, 0.0], 1.0)),
        Err(RlsError64::InvalidDenominator)
    );
    assert_state64_bitwise_eq(&indefinite, &indefinite_snapshot);
    let unexcited = model
        .update(&indefinite, &sample64(&[0.0, 1.0], 1.0))
        .expect("the unexcited direction passes the denominator guard");
    assert_state64_bitwise_eq(&indefinite, &indefinite_snapshot);
    assert!(unexcited.covariance().get(0, 0).unwrap() < 0.0);

    let huge_covariance = admitted64(learned.weights().as_slice(), &[f64::MAX, 1.0]);
    let huge_snapshot = huge_covariance.clone();
    assert_eq!(
        model.update(&huge_covariance, &sample64(&[2.0, 0.0], 0.0)),
        Err(RlsError64::InvalidDenominator)
    );
    assert_state64_bitwise_eq(&huge_covariance, &huge_snapshot);

    let overflow_state = admitted64(&[f64::MAX, 0.0], &[1.0, 1.0]);
    let overflow_snapshot = overflow_state.clone();
    assert_eq!(
        model.update(&overflow_state, &sample64(&[2.0, 0.0], -f64::MAX)),
        Err(RlsError64::NonFiniteCandidate)
    );
    assert_state64_bitwise_eq(&overflow_state, &overflow_snapshot);

    let continuation = model
        .update(&learned, &sample64(&[0.5, -0.25], 2.0))
        .unwrap();
    let reference = ref_step64(
        &reference_from_state64(&learned),
        &[0.5, -0.25],
        2.0,
        model.lambda(),
    );
    assert_state64_bitwise_eq(&continuation, &to_state64(&reference));
}

#[test]
fn conversion_boundaries_are_checked_without_overflow_or_underflow_repair() {
    let model = model64(1, 1.0, 1.0);
    let exact_cases: [(f64, f32); 7] = [
        (1.0, 1.0),
        (-2.0, -2.0),
        (1.0 + 2f64.powi(-24), 1.0),
        (1.0 + 3.0 * 2f64.powi(-24), 1.0 + 2f32.powi(-22)),
        (f64::from(f32::MAX) * (1.0 + 2f64.powi(-26)), f32::MAX),
        (-0.0, -0.0),
        (f64::from(f32::from_bits(1)), f32::from_bits(1)),
    ];
    for (source, expected) in exact_cases {
        let state = state64_with_weights(&model, &[source]);
        let before = state.clone();
        let snapshot = model.to_serving_f32(&state).unwrap();
        assert_eq!(
            snapshot.weights().get(0).unwrap().to_bits(),
            expected.to_bits()
        );
        assert_state64_bitwise_eq(&state, &before);
    }

    let overflow = state64_with_weights(&model, &[2.0 * f64::from(f32::MAX)]);
    let overflow_before = overflow.clone();
    assert_eq!(
        model.to_serving_f32(&overflow).unwrap_err(),
        RlsServingConversionError::Overflow { index: 0 }
    );
    assert_state64_bitwise_eq(&overflow, &overflow_before);

    let underflow = state64_with_weights(&model, &[2f64.powi(-200)]);
    let underflow_before = underflow.clone();
    let active = model.to_serving_f32(&model.initial_state()).unwrap();
    let active_bits = snapshot_bits(&active);
    assert_eq!(
        model.to_serving_f32(&underflow).unwrap_err(),
        RlsServingConversionError::Underflow { index: 0 }
    );
    assert_state64_bitwise_eq(&underflow, &underflow_before);
    assert_eq!(snapshot_bits(&active), active_bits);

    let wide_model = model64(2, 1.0, 1.0);
    let other_dimension = model64(3, 1.0, 1.0).initial_state();
    assert_eq!(
        wide_model.to_serving_f32(&other_dimension).unwrap_err(),
        RlsServingConversionError::StateDimensionMismatch {
            expected: 2,
            actual: 3,
        }
    );
}

fn collect_a(model: &Rls64, state: &RlsState64, rows: &[Vector64]) -> Result<Vec<f64>, RlsError64> {
    let mut outputs = Vec::with_capacity(rows.len());
    for row in rows {
        outputs.push(model.predict_checked(state, row)?);
    }
    Ok(outputs)
}

fn collect_f32(
    rows: &[Vector],
    mut predict: impl FnMut(&Vector) -> Result<f32, RlsError>,
) -> Result<Vec<f32>, RlsError> {
    let mut outputs = Vec::with_capacity(rows.len());
    for row in rows {
        outputs.push(predict(row)?);
    }
    Ok(outputs)
}

fn accept_candidate(
    active: &mut RlsServing32,
    candidate: RlsServing32,
    rows: &[Vector],
) -> Result<bool, RlsError> {
    if rows.is_empty() {
        return Ok(false);
    }
    let _ = collect_f32(rows, |row| candidate.predict_checked(row))?;
    *active = candidate;
    Ok(true)
}

#[test]
fn prediction_precedence_collectors_and_handover_preserve_versions() {
    let (model, learned) = learned_state64();
    let finite = Vector64::from_slice(&[0.5, -0.25]);
    let bare = model.predict(&learned, &finite).unwrap();
    let checked = model.predict_checked(&learned, &finite).unwrap();
    assert_eq!(bare.to_bits(), checked.to_bits());

    let latent = Vector64::from_slice(&[f64::NAN, 0.0, 0.0]);
    assert_eq!(
        model.predict_checked(&learned, &latent),
        Err(RlsError64::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
    assert_eq!(
        model.predict_checked(&learned, &Vector64::from_slice(&[f64::NAN, 0.0])),
        Err(RlsError64::NonFiniteInput)
    );
    let huge_source = state64_with_weights(&model, &[f64::MAX, 0.0]);
    let huge_features = Vector64::from_slice(&[2.0, 0.0]);
    let huge_before = huge_source.clone();
    assert_eq!(
        model.predict(&huge_source, &huge_features),
        Err(RlsError64::NonFiniteOutput)
    );
    assert_eq!(
        model.predict_checked(&huge_source, &huge_features),
        Err(RlsError64::NonFiniteOutput)
    );
    assert_state64_bitwise_eq(&huge_source, &huge_before);

    let active = model.to_serving_f32(&learned).unwrap();
    let snapshot_before = active.clone();
    let serving_finite = Vector::from_slice(&[0.5, -0.25]);
    let serving_bare = active.predict(&serving_finite).unwrap();
    let serving_checked = active.predict_checked(&serving_finite).unwrap();
    assert_eq!(serving_bare.to_bits(), serving_checked.to_bits());
    assert_eq!(
        active.predict_checked(&Vector::from_slice(&[f32::NAN, 0.0, 0.0])),
        Err(RlsError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
    assert_eq!(
        active.predict_checked(&Vector::from_slice(&[f32::NAN, 0.0])),
        Err(RlsError::NonFiniteInput)
    );
    assert_eq!(
        active.weights().as_slice(),
        snapshot_before.weights().as_slice()
    );

    let huge_serving = model
        .to_serving_f32(&state64_with_weights(&model, &[f32::MAX as f64, 0.0]))
        .unwrap();
    let huge_serving_features = Vector::from_slice(&[2.0, 0.0]);
    assert_eq!(
        huge_serving.predict(&huge_serving_features),
        Err(RlsError::NonFiniteOutput)
    );
    assert_eq!(
        huge_serving.predict_checked(&huge_serving_features),
        Err(RlsError::NonFiniteOutput)
    );

    let c_model = Rls::new(2, 1.0, 1.0).unwrap();
    let c_state = RlsState {
        w: Vector::from_slice(&[f32::MAX, 0.0]),
        p: Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]).unwrap(),
    };
    let c_features = Vector::from_slice(&[2.0, 0.0]);
    assert_eq!(
        c_model.predict(&c_state, &c_features),
        Err(RlsError::NonFiniteOutput)
    );
    assert_eq!(
        c_model.predict_checked(&c_state, &c_features),
        Err(RlsError::NonFiniteOutput)
    );

    let empty_a: [Vector64; 0] = [];
    assert!(collect_a(&model, &learned, &empty_a).unwrap().is_empty());
    let empty_c: [Vector; 0] = [];
    assert!(
        collect_f32(&empty_c, |row| active.predict_checked(row))
            .unwrap()
            .is_empty()
    );

    let failure_rows = [
        Vector::from_slice(&[2.0, 0.0]),
        Vector::from_slice(&[1.0, 0.0]),
        Vector::from_slice(&[1.0, 0.0]),
    ];
    for start in 0..failure_rows.len() {
        let mut rows = failure_rows.clone();
        rows.rotate_left(start);
        assert_eq!(
            collect_f32(&rows, |row| huge_serving.predict_checked(row)),
            Err(RlsError::NonFiniteOutput)
        );
        let rows64: Vec<Vector64> = rows
            .iter()
            .map(|row| {
                Vector64::from_slice(
                    &row.as_slice()
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            collect_a(&model, &huge_source, &rows64),
            Err(RlsError64::NonFiniteOutput)
        );
        assert_eq!(
            collect_f32(&rows, |row| c_model.predict_checked(&c_state, row)),
            Err(RlsError::NonFiniteOutput)
        );
    }

    let mut active = model.to_serving_f32(&learned).unwrap();
    let learned_before = learned.clone();
    let wrong_model = model64(3, 1.0, 1.0);
    let wrong_state = wrong_model.initial_state();
    assert_eq!(
        model.to_serving_f32(&wrong_state),
        Err(RlsServingConversionError::StateDimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
    assert_state64_bitwise_eq(&learned, &learned_before);
    assert_eq!(
        snapshot_bits(&active),
        snapshot_bits(&model.to_serving_f32(&learned).unwrap())
    );

    let continued = model
        .update(&learned, &sample64(&[0.75, -0.5], 1.5))
        .unwrap();
    let candidate = model.to_serving_f32(&continued).unwrap();
    assert_eq!(
        snapshot_bits(&active),
        snapshot_bits(&model.to_serving_f32(&learned).unwrap()),
        "conversion must not install a candidate"
    );
    let acceptance_rows = [
        Vector::from_slice(&[0.5, -0.25]),
        Vector::from_slice(&[1.0, 0.25]),
    ];
    assert!(!accept_candidate(&mut active, candidate.clone(), &[]).unwrap());
    assert_eq!(
        snapshot_bits(&active),
        snapshot_bits(&model.to_serving_f32(&learned).unwrap())
    );
    assert!(accept_candidate(&mut active, candidate.clone(), &acceptance_rows).unwrap());
    assert_eq!(snapshot_bits(&active), snapshot_bits(&candidate));

    let failing_source = state64_with_weights(&model, &[f32::MAX as f64, 0.0]);
    let failing_source_before = failing_source.clone();
    let failing_candidate = model.to_serving_f32(&failing_source).unwrap();
    let active_before_failure = snapshot_bits(&active);
    assert_eq!(
        accept_candidate(
            &mut active,
            failing_candidate,
            &[Vector::from_slice(&[2.0, 0.0])],
        ),
        Err(RlsError::NonFiniteOutput)
    );
    assert_eq!(snapshot_bits(&active), active_before_failure);
    assert_state64_bitwise_eq(&failing_source, &failing_source_before);

    let next = model
        .update(&learned, &sample64(&[0.25, 0.75], -0.5))
        .unwrap();
    let reference = ref_step64(
        &reference_from_state64(&learned),
        &[0.25, 0.75],
        -0.5,
        model.lambda(),
    );
    assert_state64_bitwise_eq(&next, &to_state64(&reference));
}
