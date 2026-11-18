use super::support::{features, independent_acceptance_rss};
use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::{
    AcceptanceCheckError, AcceptanceCheckReport, FitConfig, RegressionError, SelectedField,
    StoredDiagnostic, fit,
};

#[test]
fn acceptance_reports_independently_accumulated_selected_rss() {
    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 2.0, 2.0];
    let indices = [0, 1, 2];
    let candidate = fit(&x, &y, &indices, &FitConfig::default()).expect("fit");
    let weights_before = candidate.weights().clone();
    let bias_before = candidate.intercept();
    let diagnostics_before = candidate.diagnostics().clone();
    let expected = independent_acceptance_rss(&candidate, &x, &y, &indices);

    let report = candidate
        .check_acceptance(&x, &y, &indices)
        .expect("candidate admitted");
    assert_eq!(report.observations_checked, 3);
    assert_eq!(
        report.rss_of_checked_predictions_f64.to_bits(),
        expected.to_bits()
    );
    assert!(report.rss_of_checked_predictions_f64.is_finite());
    assert_eq!(candidate.weights(), &weights_before);
    assert_eq!(candidate.intercept(), bias_before);
    assert_eq!(candidate.diagnostics(), &diagnostics_before);
}

#[test]
fn acceptance_counts_duplicate_indices_and_repeats_their_rss_term() {
    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 2.0, 2.0];
    let candidate = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    let indices = [0, 0];
    let expected = independent_acceptance_rss(&candidate, &x, &y, &indices);

    let report = candidate
        .check_acceptance(&x, &y, &indices)
        .expect("repeated selections are allowed");
    assert_eq!(report.observations_checked, 2);
    assert_eq!(
        report.rss_of_checked_predictions_f64.to_bits(),
        expected.to_bits()
    );
    assert_eq!(
        report.rss_of_checked_predictions_f64.to_bits(),
        (2.0 * independent_acceptance_rss(&candidate, &x, &y, &[0])).to_bits()
    );
}

#[test]
fn acceptance_selection_errors_report_positions_and_original_indices() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 4.0];
    let candidate = fit(&x, &y, &[0, 1], &FitConfig::default()).expect("fit");

    assert_eq!(
        candidate.check_acceptance(&x, &y[..1], &[]),
        Err(AcceptanceCheckError::FeatureTargetLengthMismatch {
            features: 2,
            targets: 1,
        })
    );
    assert_eq!(
        candidate.check_acceptance(&x, &y, &[]),
        Err(AcceptanceCheckError::EmptySelection)
    );
    assert_eq!(
        candidate.check_acceptance(&x, &y, &[0, 8]),
        Err(AcceptanceCheckError::IndexOutOfRange {
            position: 1,
            index: 8,
            rows: 2,
        })
    );

    let ragged = [Vector::from_slice(&[1.0]), Vector::from_slice(&[2.0, 3.0])];
    assert_eq!(
        candidate.check_acceptance(&ragged, &[1.0, 2.0], &[1, 2]),
        Err(AcceptanceCheckError::IndexOutOfRange {
            position: 1,
            index: 2,
            rows: 2,
        })
    );
}

#[test]
fn acceptance_inspects_only_selected_rows_and_uses_declared_precedence() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 4.0];
    let candidate = fit(&x, &y, &[0, 1], &FitConfig::default()).expect("fit");

    let unselected_bad = [
        Vector::from_slice(&[1.0]),
        Vector::from_slice(&[f32::NAN, 3.0]),
    ];
    assert_eq!(
        candidate
            .check_acceptance(&unselected_bad, &y, &[0])
            .unwrap()
            .observations_checked,
        1
    );
    assert_eq!(
        candidate.check_acceptance(&unselected_bad, &y, &[1]),
        Err(AcceptanceCheckError::RowDimensionMismatch {
            position: 0,
            index: 1,
            expected: 1,
            actual: 2,
        })
    );

    let selected_nan = [Vector::from_slice(&[f32::NAN]), Vector::from_slice(&[2.0])];
    assert_eq!(
        candidate.check_acceptance(&selected_nan, &[f32::NAN, 4.0], &[0]),
        Err(AcceptanceCheckError::NonFiniteSelectedValue {
            position: 0,
            index: 0,
            field: SelectedField::Feature { column: 0 },
        })
    );
    let non_finite_target = [1.0, f32::NAN];
    assert_eq!(
        candidate.check_acceptance(&x, &non_finite_target, &[1]),
        Err(AcceptanceCheckError::NonFiniteSelectedValue {
            position: 0,
            index: 1,
            field: SelectedField::Target,
        })
    );

    let two_feature_x = features(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
    let two_feature_y = [1.0f32, 2.0, 3.0];
    let two_feature_candidate = fit(
        &two_feature_x,
        &two_feature_y,
        &[0, 1, 2],
        &FitConfig::default(),
    )
    .expect("full-rank two-feature fit");
    let two_non_finite_features = [Vector::from_slice(&[f32::NAN, f32::INFINITY])];
    assert_eq!(
        two_feature_candidate.check_acceptance(&two_non_finite_features, &[0.0], &[0]),
        Err(AcceptanceCheckError::NonFiniteSelectedValue {
            position: 0,
            index: 0,
            field: SelectedField::Feature { column: 0 },
        })
    );

    let infinite_rss_x = features(&[&[1.0], &[2.0]]);
    let no_intercept = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let candidate =
        fit(&infinite_rss_x, &[1.0, 2.0], &[0, 1], &no_intercept).expect("fit succeeds");
    let selected_nan = [Vector::from_slice(&[f32::NAN])];
    assert!(matches!(
        candidate.check_acceptance(&selected_nan, &[0.0], &[0]),
        Err(AcceptanceCheckError::NonFiniteSelectedValue {
            field: SelectedField::Feature { column: 0 },
            ..
        })
    ));
}

#[test]
fn acceptance_wraps_checked_prediction_failure_with_selection_context() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 4.0];
    let candidate = fit(&x, &y, &[0, 1], &FitConfig::default()).expect("fit");
    let acceptance_x = features(&[&[1.0], &[2.0], &[f32::MAX]]);
    let acceptance_y = [1.0f32, 4.0, 0.0];

    assert_eq!(
        candidate.check_acceptance(&acceptance_x, &acceptance_y, &[2]),
        Err(AcceptanceCheckError::Prediction {
            position: 0,
            index: 2,
            error: RegressionError::NonFiniteOutput { observation: None },
        })
    );
}

#[test]
fn acceptance_errors_have_context_and_only_prediction_has_a_cause() {
    fn assert_error_traits<
        T: std::error::Error + std::fmt::Debug + Clone + Copy + PartialEq + Eq,
    >() {
    }
    fn assert_copy_partial_eq<T: std::fmt::Debug + Clone + Copy + PartialEq>() {}

    assert_error_traits::<AcceptanceCheckError>();
    assert_copy_partial_eq::<AcceptanceCheckReport>();
    assert_copy_partial_eq::<SelectedField>();
    assert_copy_partial_eq::<StoredDiagnostic>();

    let errors = [
        AcceptanceCheckError::NonFiniteWeight { index: 4 },
        AcceptanceCheckError::NonFiniteBias,
        AcceptanceCheckError::FeatureTargetLengthMismatch {
            features: 3,
            targets: 2,
        },
        AcceptanceCheckError::EmptySelection,
        AcceptanceCheckError::IndexOutOfRange {
            position: 2,
            index: 8,
            rows: 4,
        },
        AcceptanceCheckError::RowDimensionMismatch {
            position: 1,
            index: 5,
            expected: 2,
            actual: 3,
        },
        AcceptanceCheckError::NonFiniteSelectedValue {
            position: 1,
            index: 5,
            field: SelectedField::Feature { column: 2 },
        },
        AcceptanceCheckError::NonFiniteSelectedValue {
            position: 1,
            index: 5,
            field: SelectedField::Target,
        },
        AcceptanceCheckError::NonFiniteStoredDiagnostic {
            diagnostic: StoredDiagnostic::RssF64Reference,
        },
    ];
    for error in errors {
        assert!(!error.to_string().is_empty());
        assert!(std::error::Error::source(&error).is_none());
    }
    assert!(errors[0].to_string().contains("index 4"));
    assert!(errors[1].to_string().contains("bias"));
    assert!(errors[2].to_string().contains("feature row count 3"));
    assert!(errors[2].to_string().contains("target count 2"));
    assert!(errors[3].to_string().contains("empty"));
    assert!(errors[4].to_string().contains("position 2"));
    assert!(errors[4].to_string().contains("index 8"));
    assert!(errors[4].to_string().contains("4 rows"));
    assert!(errors[5].to_string().contains("position 1"));
    assert!(errors[5].to_string().contains("row 5"));
    assert!(errors[5].to_string().contains("width 3"));
    assert!(errors[5].to_string().contains("expected 2"));
    assert!(errors[6].to_string().contains("position 1"));
    assert!(errors[6].to_string().contains("row 5"));
    assert!(errors[6].to_string().contains("column 2"));
    assert!(errors[7].to_string().contains("position 1"));
    assert!(errors[7].to_string().contains("row 5"));
    assert!(errors[7].to_string().contains("target"));
    assert!(errors[8].to_string().contains("rss_f64_reference"));
    assert!(
        AcceptanceCheckError::NonFiniteStoredDiagnostic {
            diagnostic: StoredDiagnostic::RssF32Predictor,
        }
        .to_string()
        .contains("rss_f32_predictor")
    );

    let inner = RegressionError::NonFiniteOutput { observation: None };
    let error = AcceptanceCheckError::Prediction {
        position: 3,
        index: 7,
        error: inner,
    };
    let display = error.to_string();
    assert!(display.contains("position 3"));
    assert!(display.contains("row 7"));
    assert!(!display.contains(&inner.to_string()));
    let source = std::error::Error::source(&error).expect("prediction cause");
    assert!(source.is::<RegressionError>());
    assert_eq!(source.to_string(), inner.to_string());
}
