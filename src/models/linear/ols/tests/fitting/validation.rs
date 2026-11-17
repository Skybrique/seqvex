use super::support::features;
use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::{ConversionCause, FitConfig, FitError, UnderflowPolicy, fit};

#[test]
fn rejects_row_target_mismatch() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32];
    match fit(&x, &y, &[0], &FitConfig::default()) {
        Err(FitError::RowTargetMismatch { features, targets }) => {
            assert_eq!((features, targets), (2, 1));
        }
        other => panic!("expected RowTargetMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_empty_training_selection() {
    let x = features(&[&[1.0]]);
    let y = [1.0f32];
    assert_eq!(
        fit(&x, &y, &[], &FitConfig::default()),
        Err(FitError::TooFewRows)
    );
}

#[test]
fn rejects_out_of_range_index() {
    let x = features(&[&[1.0]]);
    let y = [1.0f32];
    assert_eq!(
        fit(&x, &y, &[0, 5], &FitConfig::default()),
        Err(FitError::IndexOutOfRange { index: 5, rows: 1 })
    );
}

#[test]
fn rejects_zero_features() {
    let x = vec![Vector::zeros(0)];
    let y = [1.0f32];
    assert_eq!(
        fit(&x, &y, &[0], &FitConfig::default()),
        Err(FitError::ZeroFeatures)
    );
}

#[test]
fn rejects_ragged_features() {
    let x = features(&[&[1.0, 2.0], &[3.0]]);
    let y = [1.0f32, 2.0];
    match fit(&x, &y, &[0, 1], &FitConfig::default()) {
        Err(FitError::RaggedFeatures {
            row,
            expected,
            actual,
        }) => {
            assert_eq!((row, expected, actual), (1, 2, 1));
        }
        other => panic!("expected RaggedFeatures, got {other:?}"),
    }
}

#[test]
fn rejects_duplicate_index() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 2.0];
    assert_eq!(
        fit(&x, &y, &[0, 0], &FitConfig::default()),
        Err(FitError::DuplicateIndex { index: 0 })
    );
}

#[test]
fn rejects_non_finite_training_feature_and_target() {
    let x = features(&[&[f32::INFINITY], &[2.0]]);
    let y = [1.0f32, 2.0];
    match fit(&x, &y, &[0, 1], &FitConfig::default()) {
        Err(FitError::NonFiniteTrainingValue { row, column }) => {
            assert_eq!((row, column), (0, 0));
        }
        other => panic!("expected NonFiniteTrainingValue, got {other:?}"),
    }

    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, f32::NAN];
    match fit(&x, &y, &[0, 1], &FitConfig::default()) {
        Err(FitError::NonFiniteTrainingValue { row, column }) => {
            assert_eq!((row, column), (1, 1));
        }
        other => panic!("expected NonFiniteTrainingValue, got {other:?}"),
    }
}

#[test]
fn rejects_duplicate_columns_rank_deficient() {
    // Second column is exactly twice the first: rank 1 < required 2.
    let x = features(&[&[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0]]);
    let y = [1.0f32, 2.0, 3.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1, 2], &config) {
        Err(FitError::RankDeficient { rank, required }) => {
            assert_eq!((rank, required), (1, 2));
        }
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn rejects_constant_column_with_intercept() {
    // A constant feature duplicates the intercept column: rank 1 < required 2.
    let x = features(&[&[1.0], &[1.0], &[1.0]]);
    let y = [1.0f32, 2.0, 3.0];
    match fit(&x, &y, &[0, 1, 2], &FitConfig::default()) {
        Err(FitError::RankDeficient { rank, required }) => {
            assert_eq!((rank, required), (1, 2));
        }
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn rejects_over_budget_workspace() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 2.0];
    let config = FitConfig {
        max_workspace_bytes: 1,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1], &config) {
        Err(FitError::Capacity { requested, limit }) => {
            assert!(requested > limit);
            assert_eq!(limit, 1);
        }
        other => panic!("expected Capacity, got {other:?}"),
    }
}

#[test]
fn conversion_underflow_rejects_by_default_and_allows_opt_in() {
    // x = 2^100, y = 2^-100, no intercept -> exact coefficient 2^-200, whose
    // f32 conversion is zero. Default policy rejects; opt-in deploys zero.
    let x = features(&[&[2.0f32.powi(0) * (2.0f64.powi(100) as f32)]]);
    let y = [(2.0f64.powi(-100)) as f32];
    let rejected = fit(
        &x,
        &y,
        &[0],
        &FitConfig {
            intercept: false,
            ..FitConfig::default()
        },
    );
    assert_eq!(
        rejected,
        Err(FitError::ConversionRejected(
            ConversionCause::CoefficientUnderflow
        ))
    );

    let allowed = fit(
        &x,
        &y,
        &[0],
        &FitConfig {
            intercept: false,
            on_underflow: UnderflowPolicy::AllowLoss,
            ..FitConfig::default()
        },
    )
    .expect("opt-in underflow");
    assert_eq!(allowed.weights().as_slice(), &[0.0f32]);
}

#[test]
fn invalid_rank_threshold_is_a_configuration_error() {
    // No-intercept X=[[1,0],[0,0]], y=[1,0]: the exact-zero second pivot must
    // not be counted via a negative/NaN threshold, so these are config errors.
    let x = features(&[&[1.0, 0.0], &[0.0, 0.0]]);
    let y = [1.0f32, 0.0];
    let base = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    for &value in &[-1.0f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let config = FitConfig {
            rank_threshold: value,
            ..base.clone()
        };
        match fit(&x, &y, &[0, 1], &config) {
            Err(FitError::InvalidRankThreshold { value: got }) => {
                assert_eq!(got.is_nan(), value.is_nan(), "value {value}");
            }
            other => panic!("expected InvalidRankThreshold for {value}, got {other:?}"),
        }
    }
    // The singular system is never reached through a valid configuration.
    assert!(matches!(
        fit(
            &x,
            &y,
            &[0, 1],
            &FitConfig {
                intercept: false,
                rank_threshold: 1.0,
                ..FitConfig::default()
            }
        ),
        Err(FitError::RankDeficient { .. })
    ));
}

#[test]
fn zero_rank_threshold_still_rejects_exact_zero_pivots() {
    // Zero is a valid multiplier: exact zero pivots are still not counted.
    let x = features(&[&[1.0, 0.0], &[0.0, 0.0]]);
    let y = [1.0f32, 0.0];
    let config = FitConfig {
        intercept: false,
        rank_threshold: 0.0,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1], &config) {
        Err(FitError::RankDeficient { rank, required }) => assert_eq!((rank, required), (1, 2)),
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn default_rank_threshold_accepts_full_rank() {
    let x = features(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
    let y = [1.0f32, 2.0, 3.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1, 2], &config).expect("full rank");
    assert_eq!(result.diagnostics().rank, 2);
}

#[test]
fn underdetermined_input_is_rank_deficient() {
    // One selected row, intercept + one feature => p = 2 > N.
    let x = features(&[&[1.0]]);
    let y = [2.0f32];
    match fit(&x, &y, &[0], &FitConfig::default()) {
        Err(FitError::RankDeficient { rank, required }) => assert_eq!((rank, required), (1, 2)),
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn near_dependent_below_threshold_is_rank_deficient() {
    // Exactly full rank, but the second feature is scaled ~1e-20 below the
    // first, so its relative pivot falls below the default threshold.
    let x = features(&[&[1.0, 1.0e-20], &[1.0, 2.0e-20], &[1.0, 3.0e-20]]);
    let y = [1.0f32, 2.0, 3.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1, 2], &config) {
        Err(FitError::RankDeficient { rank, required }) => assert_eq!((rank, required), (1, 2)),
        other => panic!("expected RankDeficient, got {other:?}"),
    }
}

#[test]
fn scaled_full_rank_control_is_supported() {
    // Same shape with a representable scale separation: full rank is accepted.
    let x = features(&[&[1.0, 1.0e-2], &[1.0, 2.0e-2], &[1.0, 3.0e-2]]);
    let y = [1.0f32, 2.0, 3.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1, 2], &config).expect("full rank");
    assert_eq!(result.diagnostics().rank, 2);
}

#[test]
fn coefficient_conversion_overflow_is_rejected() {
    // No intercept, x=[0.5,0.5], y=[MAX,MAX] => w = 2*MAX, finite in f64 but
    // overflowing f32.
    let x = features(&[&[0.5], &[0.5]]);
    let y = [f32::MAX, f32::MAX];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    assert_eq!(
        fit(&x, &y, &[0, 1], &config),
        Err(FitError::ConversionRejected(ConversionCause::Overflow))
    );
}

#[test]
fn bias_conversion_underflow_is_rejected_by_default_but_allowed_on_opt_in() {
    // Concrete f32-representable fixture (recorded after a bounded search):
    // subnormal targets 1 and 2 ulps with x=[0.25, 0.5]. The fitted intercept is
    // a nonzero f64 value too small to represent in f32, while the slope is a
    // representable subnormal.
    let x = features(&[&[0.25], &[0.5]]);
    let y = [f32::from_bits(1), f32::from_bits(2)];

    // Default policy rejects the lost stored bias.
    assert_eq!(
        fit(&x, &y, &[0, 1], &FitConfig::default()),
        Err(FitError::ConversionRejected(ConversionCause::BiasUnderflow))
    );

    // Policy-dependent behavior: with AllowLoss the fit is accepted, the stored
    // bias is lost to numerical zero, and the weight remains finite and nonzero.
    // This is evidence of a lost stored bias under the opt-in policy; it does
    // not independently prove the pre-conversion f64 intercept was nonzero.
    let allow = FitConfig {
        on_underflow: UnderflowPolicy::AllowLoss,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1], &allow).expect("AllowLoss accepts the lost bias");
    assert_eq!(result.intercept(), 0.0, "stored intercept must be zero");
    let weight = result.weights().as_slice()[0];
    assert!(
        weight.is_finite() && weight != 0.0,
        "stored weight must be finite and nonzero, got {weight}"
    );
}

#[test]
fn representable_subnormal_coefficient_is_retained() {
    let x = features(&[&[1.0e20]]);
    let y = [1.0e-20f32];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0], &config).expect("fit");
    let weight = result.weights().as_slice()[0];
    assert!(weight > 0.0 && weight.is_subnormal(), "weight {weight}");
}

#[test]
fn invalid_rank_threshold_precedes_budget_error() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [1.0f32, 2.0];
    let config = FitConfig {
        rank_threshold: f64::NAN,
        max_workspace_bytes: 0,
        ..FitConfig::default()
    };
    assert!(matches!(
        fit(&x, &y, &[0, 1], &config),
        Err(FitError::InvalidRankThreshold { .. })
    ));
}

#[test]
fn index_error_precedes_invalid_rank_threshold() {
    let x = features(&[&[1.0]]);
    let y = [1.0f32];
    let config = FitConfig {
        rank_threshold: -1.0,
        ..FitConfig::default()
    };
    assert_eq!(
        fit(&x, &y, &[5], &config),
        Err(FitError::IndexOutOfRange { index: 5, rows: 1 })
    );
}

#[test]
fn deployed_overflow_diagnostic_is_rejected_by_legacy_fit() {
    // Finite coefficient, but the deployed slope-1.2*MAX row-2 prediction
    // overflows f32. Legacy fit now rejects the non-finite diagnostic instead of
    // returning a successful result with an invalid stored RSS.
    let x = features(&[&[1.0], &[2.0]]);
    let y = [f32::MAX, f32::MAX];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    assert_eq!(fit(&x, &y, &[0, 1], &config), Err(FitError::NonFiniteRss));
}

#[test]
fn legacy_fit_rejects_infinite_deployed_rss() {
    let x = features(&[&[1.0], &[2.0]]);
    let y = [f32::MAX, f32::MAX];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    assert_eq!(fit(&x, &y, &[0, 1], &config), Err(FitError::NonFiniteRss));
}
