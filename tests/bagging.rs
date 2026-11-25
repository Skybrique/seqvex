#![allow(clippy::single_range_in_vec_init)]

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::ensemble::bagging::{
    BaggingError, BaggingLimits, BaggingSpec, DecisionTreeBag, DecisionTreeBag64,
    DecisionTreeBagError, OlsBag, OlsBag64, OlsBagError,
};
use seqvex::models::linear::ols::{FitConfig32, FitConfig64, FitError64, UnderflowPolicy};
use seqvex::models::trees::decision_tree::{
    TreeFitConfig, TreeServingConversionError, TreeServingField, TreeUnderflowPolicy,
};
use seqvex::preprocessing::sampling::{OutcomeInterval, SampleUnit, SamplingMode};

#[allow(clippy::single_range_in_vec_init)]
fn units(count: usize) -> Vec<SampleUnit> {
    (0..count)
        .map(|index| SampleUnit {
            rows: vec![index..index + 1],
            outcomes: vec![OutcomeInterval {
                start: index as i64,
                end: index as i64 + 1,
            }],
            sequence: 0,
        })
        .collect()
}

fn disable(members: usize) -> BaggingSpec {
    BaggingSpec::new(members, SamplingMode::Disabled, 0, 7, 0)
}

fn tree_data() -> (Vec<Vector64>, Vec<f64>, Vec<Vector>, Vec<f32>) {
    let mut f64_rows = Vec::new();
    let mut f32_rows = Vec::new();
    let mut y64 = Vec::new();
    let mut y32 = Vec::new();
    for index in 0..6usize {
        let x = index as f64;
        let target = if x <= 2.0 { 1.0 } else { 3.0 };
        f64_rows.push(Vector64::from_slice(&[x]));
        f32_rows.push(Vector::from_slice(&[x as f32]));
        y64.push(target);
        y32.push(target as f32);
    }
    (f64_rows, y64, f32_rows, y32)
}

fn ols_data() -> (Vec<Vector64>, Vec<f64>, Vec<Vector>, Vec<f32>) {
    let mut f64_rows = Vec::new();
    let mut f32_rows = Vec::new();
    let mut y64 = Vec::new();
    let mut y32 = Vec::new();
    for index in 0..6usize {
        let x = index as f64;
        f64_rows.push(Vector64::from_slice(&[x]));
        f32_rows.push(Vector::from_slice(&[x as f32]));
        y64.push(1.0 + 2.0 * x);
        y32.push(1.0 + 2.0 * index as f32);
    }
    (f64_rows, y64, f32_rows, y32)
}

#[test]
fn tree_profiles_fit_predict_and_batch_equivalence() {
    let (features64, targets64, features32, targets32) = tree_data();
    let sample_units = units(features64.len());
    let config = TreeFitConfig::default();

    let a = DecisionTreeBag64::fit(&features64, &targets64, &sample_units, &disable(2), &config)
        .unwrap();
    assert_eq!(a.members().len(), 2);
    assert!(a.retained_bytes() > 0);
    let scalar = a.predict(&features64[0]).unwrap();
    assert!(scalar.is_finite());
    let batch = a
        .predict_batch(
            &features64
                .iter()
                .map(|row| Observation::new(row.clone()))
                .collect::<Vec<_>>(),
            8,
        )
        .unwrap();
    assert_eq!(batch[0], scalar);

    let b = DecisionTreeBag::fit_from_f64(
        &features64,
        &targets64,
        &sample_units,
        &disable(2),
        &config,
        TreeUnderflowPolicy::Reject,
    )
    .unwrap();
    assert!(b.predict(&features32[0]).unwrap().is_finite());

    let c = DecisionTreeBag::fit_f32(&features32, &targets32, &sample_units, &disable(2), &config)
        .unwrap();
    assert!(c.predict(&features32[0]).unwrap().is_finite());
}

#[test]
fn ols_profiles_fit_predict() {
    let (features64, targets64, features32, targets32) = ols_data();
    let sample_units = units(features64.len());

    let a = OlsBag64::fit(
        &features64,
        &targets64,
        &sample_units,
        &disable(2),
        &FitConfig64::default(),
    )
    .unwrap();
    assert!(a.predict(&features64[0]).unwrap().is_finite());

    let b = OlsBag::fit_from_f64(
        &features64,
        &targets64,
        &sample_units,
        &disable(2),
        &FitConfig64::default(),
        UnderflowPolicy::Reject,
    )
    .unwrap();
    assert!(b.predict(&features32[0]).unwrap().is_finite());

    let c = OlsBag::fit_f32(
        &features32,
        &targets32,
        &sample_units,
        &disable(2),
        &FitConfig32::default(),
    )
    .unwrap();
    assert!(c.predict(&features32[0]).unwrap().is_finite());
}

#[test]
fn retained_limit_equality_and_over_limit_preserve_active_bag() {
    let (features64, targets64, _, _) = tree_data();
    let sample_units = units(features64.len());
    let config = TreeFitConfig::default();
    let active =
        DecisionTreeBag64::fit(&features64, &targets64, &sample_units, &disable(2), &config)
            .unwrap();
    let active_prediction = active.predict(&features64[3]).unwrap();
    let retained = active.retained_bytes();
    let expected = active
        .members()
        .iter()
        .map(|member| std::mem::size_of::<usize>() + std::mem::size_of_val(member.nodes()))
        .sum::<usize>();
    assert_eq!(retained, expected);

    let exact = BaggingSpec::new(2, SamplingMode::Disabled, 0, 7, 0).with_limits(BaggingLimits {
        max_retained_bytes: retained,
        ..BaggingLimits::default()
    });
    assert!(
        DecisionTreeBag64::fit(&features64, &targets64, &sample_units, &exact, &config).is_ok()
    );

    let over = BaggingSpec::new(2, SamplingMode::Disabled, 0, 7, 0).with_limits(BaggingLimits {
        max_retained_bytes: retained - 1,
        ..BaggingLimits::default()
    });
    match DecisionTreeBag64::fit(&features64, &targets64, &sample_units, &over, &config) {
        Err(DecisionTreeBagError::Bagging(BaggingError::RetainedBytesExceeded {
            requested,
            limit,
        })) => {
            assert_eq!(requested, expected);
            assert_eq!(limit, expected - 1);
        }
        other => panic!("unexpected retained-limit result: {other:?}"),
    }
    assert_eq!(active.predict(&features64[3]).unwrap(), active_prediction);
}

#[test]
fn failed_without_replacement_ols_candidate_is_indexed_and_preserves_active() {
    let (features64, targets64, _, _) = ols_data();
    let sample_units = units(features64.len());
    let active = OlsBag64::fit(
        &features64,
        &targets64,
        &sample_units,
        &disable(1),
        &FitConfig64::default(),
    )
    .unwrap();
    let prediction = active.predict(&features64[0]).unwrap();

    let bad_features = vec![Vector64::from_slice(&[1.0, 2.0])];
    let bad_targets = vec![1.0];
    let bad_units = units(1);
    let bad_spec = BaggingSpec::new(
        1,
        SamplingMode::Method(seqvex::preprocessing::sampling::SampleMethod::WithoutReplacement),
        1,
        7,
        0,
    );
    match OlsBag64::fit(
        &bad_features,
        &bad_targets,
        &bad_units,
        &bad_spec,
        &FitConfig64::default(),
    ) {
        Err(OlsBagError::MemberFit64 {
            member: 0,
            source:
                FitError64::RankDeficient {
                    rank: 1,
                    required: 3,
                },
        }) => {}
        other => panic!("unexpected short-history OLS result: {other:?}"),
    }
    assert_eq!(active.predict(&features64[0]).unwrap(), prediction);
}

#[test]
fn profile_b_conversion_failure_is_indexed_and_preserves_active() {
    let (features64, targets64, _, _) = tree_data();
    let sample_units = units(features64.len());
    let active = DecisionTreeBag::fit_from_f64(
        &features64,
        &targets64,
        &sample_units,
        &disable(1),
        &TreeFitConfig::default(),
        TreeUnderflowPolicy::Reject,
    )
    .unwrap();
    let prediction = active.predict(&Vector::from_slice(&[0.0])).unwrap();

    let overflow_features = vec![Vector64::from_slice(&[0.0])];
    let overflow_targets = vec![1.0e40];
    let overflow_units = units(1);
    let config = TreeFitConfig {
        max_depth: 0,
        ..TreeFitConfig::default()
    };
    match DecisionTreeBag::fit_from_f64(
        &overflow_features,
        &overflow_targets,
        &overflow_units,
        &disable(1),
        &config,
        TreeUnderflowPolicy::Reject,
    ) {
        Err(DecisionTreeBagError::MemberConversion {
            member: 0,
            source:
                TreeServingConversionError::Overflow {
                    node: 0,
                    field: TreeServingField::Leaf,
                },
        }) => {}
        other => panic!("unexpected Profile B conversion result: {other:?}"),
    }
    assert_eq!(
        active.predict(&Vector::from_slice(&[0.0])).unwrap(),
        prediction
    );
}

#[test]
fn real_ols_predictor_error_is_indexed() {
    use seqvex::models::ensemble::bagging::{MemberPredictCause, OlsBagError, PredictionError};
    use seqvex::models::linear::ols::RegressionError;

    let features = vec![
        Vector64::from_slice(&[0.0]),
        Vector64::from_slice(&[1.0]),
        Vector64::from_slice(&[2.0]),
        Vector64::from_slice(&[3.0]),
    ];
    let targets = vec![0.0, 1e200, 2.0, 3.0];
    let sample_units = units(4);
    let mut found = false;
    for base in 0..256u64 {
        let spec = BaggingSpec::new(
            2,
            SamplingMode::Method(seqvex::preprocessing::sampling::SampleMethod::WithoutReplacement),
            2,
            base,
            0,
        );
        let Ok(bag) = OlsBag64::fit(
            &features,
            &targets,
            &sample_units,
            &spec,
            &FitConfig64::default(),
        ) else {
            continue;
        };
        let batch = vec![
            Observation::new(Vector64::from_slice(&[1.0])),
            Observation::new(Vector64::from_slice(&[1.0])),
            Observation::new(Vector64::from_slice(&[1.0])),
            Observation::new(Vector64::from_slice(&[1e200])),
        ];
        if let Err(OlsBagError::Predict(PredictionError::MemberPredict {
            row_index: 3,
            member_index: 1,
            cause: MemberPredictCause::Predictor(RegressionError::NonFiniteOutput { .. }),
            evaluated_rows: 3,
        })) = bag.predict_batch(&batch, 8)
        {
            found = true;
            break;
        }
    }
    assert!(
        found,
        "expected a real OLS non-finite predictor error at member 1"
    );
}

#[test]
fn batch_preflight_empty_and_zero_limit() {
    let (features64, targets64, _, _) = tree_data();
    let sample_units = units(features64.len());
    let bag = DecisionTreeBag64::fit(
        &features64,
        &targets64,
        &sample_units,
        &disable(1),
        &TreeFitConfig::default(),
    )
    .unwrap();
    assert!(bag.predict_batch(&[], 0).unwrap().is_empty());
    let row = Observation::new(features64[0].clone());
    assert!(bag.predict_batch(&[row], 0).is_err());
}

#[test]
fn root_only_tree_repeat_weight_fixture() {
    use seqvex::models::trees::decision_tree::fit_tree_f64;

    let features = vec![Vector64::from_slice(&[0.0]); 4];
    let targets = vec![0.0, 0.0, 10.0, 10.0];
    let config = TreeFitConfig {
        max_depth: 0,
        ..TreeFitConfig::default()
    };
    let repeated = fit_tree_f64(&features, &targets, &[0, 0, 2], &config)
        .unwrap()
        .into_predictor();
    assert_eq!(repeated.predict(&features[0]).unwrap(), 10.0 / 3.0);

    let deduplicated = fit_tree_f64(&features, &targets, &[0, 2], &config)
        .unwrap()
        .into_predictor();
    assert_eq!(deduplicated.predict(&features[0]).unwrap(), 5.0);
}

#[test]
fn profile_b_tree_route_change_through_public_bagger() {
    let threshold = 1.0_f64 + 2.0_f64.powi(-25);
    let query = 1.0_f64 + 2.0_f64.powi(-24);
    let features = vec![
        Vector64::from_slice(&[threshold]),
        Vector64::from_slice(&[2.0]),
    ];
    let targets = vec![0.0, 10.0];
    let sample_units = units(2);
    let spec = disable(1);

    let a = DecisionTreeBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    assert_eq!(a.predict(&Vector64::from_slice(&[query])).unwrap(), 10.0);

    let b = DecisionTreeBag::fit_from_f64(
        &features,
        &targets,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
        TreeUnderflowPolicy::Reject,
    )
    .unwrap();
    assert_eq!(
        b.predict(&Vector::from_slice(&[query as f32])).unwrap(),
        0.0
    );
}

#[test]
fn tree_a_b_sequential_sampling_through_public_baggers() {
    let (features64, targets64, _, _) = tree_data();
    let sample_units = units(features64.len());
    let spec = BaggingSpec::new(
        2,
        SamplingMode::Method(seqvex::preprocessing::sampling::SampleMethod::SequentialBootstrap),
        6,
        0x51,
        0,
    );
    let a = DecisionTreeBag64::fit(
        &features64,
        &targets64,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    assert_eq!(a.members().len(), 2);
    assert!(a.predict(&features64[0]).unwrap().is_finite());

    let b = DecisionTreeBag::fit_from_f64(
        &features64,
        &targets64,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
        TreeUnderflowPolicy::Reject,
    )
    .unwrap();
    assert_eq!(b.members().len(), 2);
    assert!(b.predict(&Vector::from_slice(&[0.0])).unwrap().is_finite());
}

#[test]
fn tree_c_sequential_is_rejected_pending_103() {
    let (_, _, features32, targets32) = tree_data();
    let sample_units = units(features32.len());
    let spec = BaggingSpec::new(
        1,
        SamplingMode::Method(seqvex::preprocessing::sampling::SampleMethod::SequentialBootstrap),
        6,
        0x51,
        0,
    );
    assert!(
        DecisionTreeBag::fit_f32(
            &features32,
            &targets32,
            &sample_units,
            &spec,
            &TreeFitConfig::default(),
        )
        .is_err()
    );
}

#[test]
fn fragmented_unit_uses_full_expanded_row_count_before_fit() {
    use seqvex::models::ensemble::bagging::BaggingError;

    let features = vec![
        Vector64::from_slice(&[0.0]),
        Vector64::from_slice(&[1.0]),
        Vector64::from_slice(&[2.0]),
        Vector64::from_slice(&[3.0]),
    ];
    let targets = vec![0.0, 1.0, 0.0, 1.0];
    let sample_units = vec![SampleUnit {
        rows: vec![0..1, 2..3],
        outcomes: Vec::new(),
        sequence: 0,
    }];
    let spec = BaggingSpec::new(
        1,
        SamplingMode::Method(seqvex::preprocessing::sampling::SampleMethod::WithReplacement),
        2,
        7,
        0,
    )
    .with_limits(BaggingLimits {
        max_expanded_rows_per_member: 2,
        ..BaggingLimits::default()
    });
    let config = TreeFitConfig {
        max_split_evaluations: 0,
        ..TreeFitConfig::default()
    };
    match DecisionTreeBag64::fit(&features, &targets, &sample_units, &spec, &config) {
        Err(DecisionTreeBagError::Bagging(BaggingError::ExpandedRowsExceeded {
            rows: 4,
            limit: 2,
        })) => {}
        other => panic!("unexpected fragmented-unit result: {other:?}"),
    }
}

#[test]
fn integer_draw_does_not_consume_sequential_work_budget() {
    let features = vec![Vector64::from_slice(&[0.0])];
    let targets = vec![1.0];
    let sample_units = vec![SampleUnit {
        rows: vec![0..1],
        outcomes: vec![seqvex::preprocessing::sampling::OutcomeInterval { start: 0, end: 1 }],
        sequence: 0,
    }];
    let spec = BaggingSpec::new(
        1,
        SamplingMode::Method(seqvex::preprocessing::sampling::SampleMethod::WithReplacement),
        1,
        9,
        0,
    )
    .with_limits(BaggingLimits {
        max_total_sequential_work: 1,
        ..BaggingLimits::default()
    });
    assert!(
        DecisionTreeBag64::fit(
            &features,
            &targets,
            &sample_units,
            &spec,
            &TreeFitConfig::default(),
        )
        .is_ok()
    );
}
