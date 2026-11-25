use seqvex::foundation::numerical::Vector64;
use seqvex::models::ensemble::bagging::{
    BAGGING_MEMBER_DOMAIN, BaggingError, BaggingSpec, DecisionTreeBag64, DecisionTreeBagError,
};
use seqvex::models::trees::decision_tree::TreeFitConfig;
use seqvex::preprocessing::sampling::{
    OutcomeInterval, SampleMethod, SampleUnit, SamplingMode, SamplingSpec, sample_training,
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
#[ignore = "#103 strict-f32 sequential sampler pending delivery and validation"]
fn strict_f32_sequential_ticket_trace_pending() {}
