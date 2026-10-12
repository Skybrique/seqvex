//! Fitter tests for the first working OLS workflow.
//!
//! Independent references are closed-form / algebraically known values; the
//! fitter's own faer QR path is never reused as the oracle.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    ConversionCause, FitConfig, FitError, RegressionError, UnderflowPolicy, fit,
};

fn features(rows: &[&[f32]]) -> Vec<Vector> {
    rows.iter().map(|row| Vector::from_slice(row)).collect()
}

fn approx(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual} (tol {tol})"
    );
}

#[test]
fn exact_line_with_intercept() {
    // y = 2x + 1 through (0,1), (1,3), (2,5).
    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 3.0, 5.0];
    let result = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    approx(result.weights().as_slice()[0], 2.0, 1e-6);
    approx(result.intercept(), 1.0, 1e-6);
    assert!(result.diagnostics().rss_f32_predictor < 1e-10);
    let model = result.into_predictor();
    approx(
        model.predict(&Vector::from_slice(&[3.0])).unwrap(),
        7.0,
        1e-4,
    );
}

#[test]
fn noisy_line_matches_closed_form_reference() {
    // x = [0,1,2], y = [1,2,2]; closed form: b = 7/6, w = 1/2.
    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 2.0, 2.0];
    let result = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    approx(result.weights().as_slice()[0], 0.5, 1e-6);
    approx(result.intercept(), 7.0 / 6.0, 1e-6);
}

#[test]
fn multiple_features_exact_recovery() {
    // Construct y = A w* with intercept; recover w* in original order.
    let x = features(&[&[1.0, 2.0], &[2.0, 3.0], &[3.0, 5.0], &[4.0, 7.0]]);
    let w_star = [0.5f64, 2.0, -1.0];
    let y: Vec<f32> = [
        [1.0f64, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 5.0],
        [1.0, 4.0, 7.0],
    ]
    .iter()
    .map(|row| row.iter().zip(&w_star).map(|(a, b)| a * b).sum::<f64>() as f32)
    .collect();
    let result = fit(&x, &y, &[0, 1, 2, 3], &FitConfig::default()).expect("fit");
    approx(result.intercept(), w_star[0] as f32, 1e-5);
    approx(result.weights().as_slice()[0], w_star[1] as f32, 1e-5);
    approx(result.weights().as_slice()[1], w_star[2] as f32, 1e-5);
}

#[test]
fn fit_without_intercept() {
    let x = features(&[&[1.0], &[2.0], &[3.0]]);
    let y = [3.0f32, 6.0, 9.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1, 2], &config).expect("fit");
    approx(result.weights().as_slice()[0], 3.0, 1e-6);
    approx(result.intercept(), 0.0, 0.0);
}

#[test]
fn coefficients_are_in_original_column_order_after_pivoting() {
    // Feature columns of very different scale force column pivoting; the
    // recovered coefficients must remain in the original feature order.
    let x = features(&[
        &[1.0, 1.0e-4],
        &[2.0, 3.0e-4],
        &[3.0, 2.0e-4],
        &[4.0, 5.0e-4],
    ]);
    let w_star = [0.5f64, 2.0, 3.0e4]; // intercept, feature1, feature2
    let y: Vec<f32> = [
        [1.0f64, 1.0, 1.0e-4],
        [1.0, 2.0, 3.0e-4],
        [1.0, 3.0, 2.0e-4],
        [1.0, 4.0, 5.0e-4],
    ]
    .iter()
    .map(|row| row.iter().zip(&w_star).map(|(a, b)| a * b).sum::<f64>() as f32)
    .collect();
    let result = fit(&x, &y, &[0, 1, 2, 3], &FitConfig::default()).expect("fit");
    approx(result.intercept(), w_star[0] as f32, 1e-4);
    approx(result.weights().as_slice()[0], w_star[1] as f32, 1e-3);
    let got = result.weights().as_slice()[1];
    assert!(
        (got - w_star[2] as f32).abs() / (w_star[2] as f32) < 1e-4,
        "feature2 coefficient {got} vs {}",
        w_star[2]
    );
}

#[test]
fn training_only_access_ignores_unselected_rows() {
    // The middle row has a non-finite target but is not selected.
    let x = features(&[&[0.0], &[2.0], &[5.0]]);
    let y = [0.0f32, f32::NAN, 5.0];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 2], &config).expect("held-out target not inspected");
    approx(result.weights().as_slice()[0], 1.0, 1e-6);
    assert_eq!(result.diagnostics().rows, 2);
}

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
fn diagnostics_distinguish_f64_reference_and_f32_predictor() {
    let x = features(&[&[1.0], &[2.0], &[3.0]]);
    let y = [2.0f32, 4.0, 6.0];
    let result = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    let diag = result.diagnostics();
    assert_eq!(diag.rows, 3);
    assert_eq!(diag.features, 1);
    assert_eq!(diag.rank, diag.required_rank);
    assert!(diag.rank_threshold >= 0.0);
    assert!(diag.requested_workspace_bytes > 0);
    assert!(diag.rss_f64_reference.is_finite());
    assert!(diag.rss_f32_predictor.is_finite());
}

#[test]
fn predictor_streaming_behavior_is_preserved() {
    let x = features(&[&[1.0], &[2.0], &[3.0]]);
    let y = [2.0f32, 4.0, 6.0];
    let model = fit(&x, &y, &[0, 1, 2], &FitConfig::default())
        .expect("fit")
        .into_predictor();
    let mut executor = StreamingExecutor::new(&model, 0.0f32);
    let streamed: Vec<f32> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|value| {
            *executor
                .process_one(&Observation::new(Vector::from_slice(&[*value])))
                .unwrap()
        })
        .collect();
    let grouped = model
        .predict_batch(&[
            Observation::new(Vector::from_slice(&[1.0])),
            Observation::new(Vector::from_slice(&[2.0])),
            Observation::new(Vector::from_slice(&[3.0])),
        ])
        .unwrap();
    assert_eq!(streamed, grouped);
}

// ---- production-fitter boundaries (invalid config, rank, conversion, budget) ----

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
fn nonzero_residual_diagnostics_match_independent_expected() {
    // No intercept, x=[0,1], y=[0.5,0.5]: closed form w=0.5, residuals
    // [0.5, 0], so RSS = 0.25 exactly (both f64 and f32).
    let x = features(&[&[0.0], &[1.0]]);
    let y = [0.5f32, 0.5];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1], &config).expect("fit");
    approx(result.weights().as_slice()[0], 0.5, 1e-7);
    let diagnostics = result.diagnostics();
    assert!((diagnostics.rss_f64_reference - 0.25).abs() < 1e-12);
    assert!((diagnostics.rss_f32_predictor - 0.25).abs() < 1e-6);
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
fn workspace_budget_matches_independently_enumerated_total() {
    use faer::dyn_stack::StackReq;
    use faer::linalg::qr::col_pivoting::factor::{qr_in_place_scratch, recommended_block_size};
    use faer::linalg::qr::col_pivoting::solve::solve_lstsq_in_place_scratch;
    use faer::{Par, Spec};

    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 3.0, 5.0];
    let base = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    let requested = base.diagnostics().requested_workspace_bytes;

    // Independently enumerate every owned backing buffer for the fixture's
    // actual dimensions (N=3, d=1, p=2, intercept on). The faer block size and
    // combined scratch requirement come from the pinned public APIs, not from
    // Seqvex's budget helper or `FitDiagnostics`.
    let rows = 3usize;
    let feature_count = 1usize;
    let p = feature_count + 1; // intercept column
    let size = rows.min(p);
    let block = recommended_block_size::<f64>(rows, p);
    let scratch = StackReq::any_of(&[
        qr_in_place_scratch::<usize, f64>(rows, p, block, Par::Seq, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f64>(rows, p, block, 1, Par::Seq),
    ])
    .unaligned_bytes_required();

    let f64_bytes = std::mem::size_of::<f64>();
    let f32_bytes = std::mem::size_of::<f32>();
    let usize_bytes = std::mem::size_of::<usize>();
    let a = rows * p * f64_bytes;
    let q = block * size * f64_bytes;
    let rhs = rows * f64_bytes;
    let perm = 2 * p * usize_bytes;
    let solution = p * f64_bytes;
    let converted = p * f32_bytes;
    let retained = p * f32_bytes;
    let diagnostics_bytes = p * f32_bytes;
    let uniq = x.len().div_ceil(8);

    let mut full = 0usize;
    for term in [
        a,
        q,
        rhs,
        perm,
        solution,
        converted,
        retained,
        diagnostics_bytes,
        uniq,
        scratch,
    ] {
        full = full.checked_add(term).expect("no overflow");
    }
    assert_eq!(
        requested, full,
        "requested {requested} != independently enumerated {full}"
    );

    // The pre-correction formula omitted the f64 solution plus the retained and
    // transient diagnostics coefficient vectors, keeping only a single 4p
    // converted buffer: exactly 16p bytes short. The equality above would fail
    // under that accounting, which would report `full - 16p`.
    let pre_correction = a + q + rhs + perm + converted + uniq + scratch;
    assert_eq!(full.checked_sub(pre_correction).expect("ordered"), 16 * p);

    // Exact budget accepts; a budget one byte below the complete total rejects.
    let exact = FitConfig {
        max_workspace_bytes: full,
        ..FitConfig::default()
    };
    assert!(fit(&x, &y, &[0, 1, 2], &exact).is_ok());
    let below = FitConfig {
        max_workspace_bytes: full - 1,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1, 2], &below) {
        Err(FitError::Capacity {
            requested: got,
            limit,
        }) => {
            assert_eq!(got, full);
            assert_eq!(limit, full - 1);
        }
        other => panic!("expected Capacity, got {other:?}"),
    }
}

#[test]
fn deployed_overflow_diagnostic_boundary_is_preserved() {
    // Finite coefficient, but the deployed slope-1.2*MAX row-2 prediction
    // overflows f32: fitting succeeds, the f32 RSS is infinite, and checked
    // serving rejects the output. This is a documented boundary, not a new
    // fitting-accuracy gate.
    let x = features(&[&[1.0], &[2.0]]);
    let y = [f32::MAX, f32::MAX];
    let config = FitConfig {
        intercept: false,
        ..FitConfig::default()
    };
    let result = fit(&x, &y, &[0, 1], &config).expect("finite coefficient fit succeeds");
    let diagnostics = result.diagnostics();
    assert!(diagnostics.rss_f64_reference.is_finite());
    assert!(diagnostics.rss_f32_predictor.is_infinite());
    let model = result.into_predictor();
    let row = Vector::from_slice(&[2.0]);
    assert!(model.predict(&row).unwrap().is_infinite());
    assert_eq!(
        model.predict_checked(&row),
        Err(RegressionError::NonFiniteOutput { observation: None })
    );
}
