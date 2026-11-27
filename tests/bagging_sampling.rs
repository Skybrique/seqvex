use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::models::ensemble::bagging::{
    BAGGING_MEMBER_DOMAIN, BaggingError, BaggingSpec, DecisionTreeBag, DecisionTreeBag64,
    DecisionTreeBagError,
};
use seqvex::models::trees::decision_tree::{TreeFitConfig, fit_tree_f32};
use seqvex::preprocessing::sampling::{
    OutcomeInterval, SampleMethod, SampleUnit, SamplingMode, SamplingSpec, SequentialLimits,
    sample_training, sample_training_sequential_f32,
};

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

#[test]
fn p9_seed_vectors_are_literal() {
    let base0 = BaggingSpec::new(1, SamplingMode::Disabled, 0, 0, 0);
    assert_eq!(base0.member_seed(0), 0x1060_0000_0000_0001);
    assert_eq!(base0.member_seed(1), 0x1060_0000_0000_0000);

    let fold1 = BaggingSpec::new(1, SamplingMode::Disabled, 0, 0, 1);
    assert_eq!(fold1.member_seed(0), 0x1060_0001_0000_0001);

    let mixed = BaggingSpec::new(1, SamplingMode::Disabled, 0, 0x0123_4567_89AB_CDEF, 2);
    assert_eq!(mixed.member_seed(3), 0x1143_4565_89AB_CDED);
    assert_ne!(mixed.member_seed(0), mixed.member_seed(1));
    assert_eq!(BAGGING_MEMBER_DOMAIN, 0x1060_0000_0000_0001);
}

#[test]
fn cutoff_equality_passes_and_later_end_is_typed() {
    let features = vec![Vector64::from_slice(&[0.0]), Vector64::from_slice(&[1.0])];
    let targets = vec![0.0, 1.0];
    let sample_units = units(2);
    let config = TreeFitConfig::default();

    let exact = BaggingSpec::new(1, SamplingMode::Disabled, 0, 1, 0).with_support_cutoff(Some(2));
    assert!(DecisionTreeBag64::fit(&features, &targets, &sample_units, &exact, &config).is_ok());

    let too_late =
        BaggingSpec::new(1, SamplingMode::Disabled, 0, 1, 0).with_support_cutoff(Some(1));
    match DecisionTreeBag64::fit(&features, &targets, &sample_units, &too_late, &config) {
        Err(DecisionTreeBagError::Bagging(BaggingError::SupportBeyondCutoff {
            end: 2,
            cutoff: 1,
            ..
        })) => {}
        other => panic!("unexpected cutoff result: {other:?}"),
    }
}

#[test]
fn fragmented_repeat_emits_rows_in_draw_order() {
    let unit = SampleUnit {
        rows: vec![0..1, 2..3],
        outcomes: Vec::new(),
        sequence: 0,
    };
    let spec = SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 2, 7);
    let sample = sample_training(&[unit], spec).unwrap();
    assert_eq!(sample.row_indices(), &[0, 2, 0, 2]);
    assert_eq!(sample.unit_counts(), &[2]);
}

#[test]
fn p9_without_replacement_subsets_are_fixed() {
    let sample_units = units(4);
    let member0 = sample_training(
        &sample_units,
        SamplingSpec::new(
            SamplingMode::Method(SampleMethod::WithoutReplacement),
            2,
            0x1060_0000_0000_0001,
        ),
    )
    .unwrap();
    assert_eq!(member0.row_indices(), &[0, 2]);
    assert_eq!(member0.unit_counts(), &[1, 0, 1, 0]);

    let member2 = sample_training(
        &sample_units,
        SamplingSpec::new(
            SamplingMode::Method(SampleMethod::WithoutReplacement),
            2,
            0x1060_0000_0000_0003,
        ),
    )
    .unwrap();
    assert_eq!(member2.row_indices(), &[1, 2]);
    assert_eq!(member2.unit_counts(), &[0, 1, 1, 0]);
}

#[test]
fn tree_c_sequential_member_matches_direct_strict_sample() {
    let features: Vec<Vector> = (0..4)
        .map(|row| Vector::from_slice(&[row as f32]))
        .collect();
    let targets = vec![0.0_f32, 10.0, 3.0, 7.0];
    let sample_units = units(4);
    let config = TreeFitConfig::default();
    let spec = BaggingSpec::new(3, SamplingMode::Default, 5, 0, 0);

    let bag = DecisionTreeBag::fit_f32(&features, &targets, &sample_units, &spec, &config).unwrap();
    assert_eq!(bag.members().len(), 3);

    for member in 0..spec.members() {
        let seed = spec.member_seed(member);
        let direct = sample_training_sequential_f32(
            &sample_units,
            SamplingSpec::new(spec.mode(), spec.sample_size(), seed),
            &SequentialLimits::default(),
        )
        .unwrap();
        let reference = fit_tree_f32(&features, &targets, direct.row_indices(), &config)
            .unwrap()
            .into_predictor();
        assert_eq!(bag.members()[member], reference, "member {member}");

        if member == 0 {
            assert_eq!(seed, 0x1060_0000_0000_0001);
            assert_eq!(direct.row_indices(), &[0, 0, 2, 2, 3]);
            assert_eq!(direct.unit_counts(), &[2, 0, 2, 1]);
        }
    }

    let prediction = bag.predict(&features[2]).unwrap();
    let refit =
        DecisionTreeBag::fit_f32(&features, &targets, &sample_units, &spec, &config).unwrap();
    assert_eq!(
        prediction.to_bits(),
        refit.predict(&features[2]).unwrap().to_bits()
    );
}

#[allow(clippy::single_range_in_vec_init)]
fn tree_c_integer_fixture() -> (Vec<Vector>, Vec<f32>, Vec<SampleUnit>) {
    let features: Vec<Vector> = (0..5)
        .map(|row| Vector::from_slice(&[row as f32]))
        .collect();
    let targets = vec![0.0_f32, 12.0, 16.0, 24.0, 48.0];
    let sample_units = (0..5)
        .map(|index| SampleUnit {
            rows: vec![index..index + 1],
            outcomes: Vec::new(),
            sequence: 0,
        })
        .collect();
    (features, targets, sample_units)
}

fn assert_tree_c_integer_member(
    method: SampleMethod,
    expected_rows: &[usize],
    expected_unit_counts: &[usize],
    expected_prediction: f32,
) {
    let (features, targets, sample_units) = tree_c_integer_fixture();
    let config = TreeFitConfig {
        max_depth: 0,
        ..TreeFitConfig::default()
    };
    let spec = BaggingSpec::new(1, SamplingMode::Method(method), 4, 0x1060_0000_5EED_5EEC, 0);
    assert_eq!(spec.member_seed(0), 0x0000_0000_5EED_5EED);

    let direct = sample_training(
        &sample_units,
        SamplingSpec::new(SamplingMode::Method(method), 4, 0x0000_0000_5EED_5EED),
    )
    .unwrap();
    assert_eq!(direct.row_indices(), expected_rows);
    assert_eq!(direct.unit_counts(), expected_unit_counts);

    let reference = fit_tree_f32(&features, &targets, direct.row_indices(), &config)
        .unwrap()
        .into_predictor();
    let bag = DecisionTreeBag::fit_f32(&features, &targets, &sample_units, &spec, &config).unwrap();
    assert_eq!(bag.members().len(), 1);
    assert_eq!(bag.members()[0], reference);

    let query = Vector::from_slice(&[5.0]);
    let prediction = bag.predict(&query).unwrap();
    assert_eq!(prediction.to_bits(), expected_prediction.to_bits());
    assert_eq!(
        reference.predict(&query).unwrap().to_bits(),
        expected_prediction.to_bits()
    );

    let identity = fit_tree_f32(&features, &targets, &[0, 1, 2, 3, 4], &config)
        .unwrap()
        .into_predictor();
    assert_ne!(
        identity.predict(&query).unwrap().to_bits(),
        expected_prediction.to_bits(),
        "identity fallback must not reproduce the sampled mean"
    );
}

#[test]
fn tree_c_with_replacement_matches_direct_integer_sample() {
    assert_tree_c_integer_member(
        SampleMethod::WithReplacement,
        &[0, 0, 1, 3],
        &[2, 1, 0, 1, 0],
        9.0,
    );
}

#[test]
fn tree_c_without_replacement_matches_direct_integer_sample() {
    assert_tree_c_integer_member(
        SampleMethod::WithoutReplacement,
        &[0, 1, 2, 3],
        &[1, 1, 1, 1, 0],
        13.0,
    );
}
