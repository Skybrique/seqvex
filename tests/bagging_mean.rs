use seqvex::foundation::numerical::Vector64;
use seqvex::foundation::observation::Observation;
use seqvex::models::ensemble::bagging::{BaggingSpec, DecisionTreeBag64, OlsBag64};
use seqvex::models::linear::ols::FitConfig64;
use seqvex::models::trees::decision_tree::{DecisionTree64, DecisionTreeNode64, TreeFitConfig};
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

fn e0_04_spec() -> BaggingSpec {
    BaggingSpec::new(
        2,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        2,
        0x1060_0000_0000_0002,
        0,
    )
}

fn assert_e0_04_replay(sample_units: &[SampleUnit], spec: &BaggingSpec) {
    assert_eq!(spec.member_seed(0), 3);
    assert_eq!(spec.member_seed(1), 2);
    for (member, expected_rows, expected_counts) in [
        (0usize, vec![0usize, 2], vec![1usize, 0, 1, 0]),
        (1usize, vec![0usize, 1], vec![1usize, 1, 0, 0]),
    ] {
        let sample = sample_training(
            sample_units,
            SamplingSpec::new(spec.mode(), spec.sample_size(), spec.member_seed(member)),
        )
        .unwrap();
        assert_eq!(sample.row_indices(), expected_rows.as_slice());
        assert_eq!(sample.unit_counts(), expected_counts.as_slice());
    }
}

fn assert_root_leaf(member: &DecisionTree64, expected: f64) {
    assert_eq!(member.nodes().len(), 1);
    match member.nodes() {
        [DecisionTreeNode64::Leaf { value }] => assert_eq!(value.to_bits(), expected.to_bits()),
        other => panic!("expected a single root leaf, found {other:?}"),
    }
}

fn assert_profile_a_aggregate(predictions: [f64; 2], scalar: f64, batch: &[f64]) {
    assert_eq!(predictions[0].to_bits(), 2.0_f64.to_bits());
    assert_eq!(predictions[1].to_bits(), 6.0_f64.to_bits());

    let oracle = (predictions[0] + predictions[1]) / 2.0;
    assert_eq!(oracle.to_bits(), 4.0_f64.to_bits());
    assert_eq!(scalar.to_bits(), oracle.to_bits());

    assert_eq!(batch.len(), 3);
    assert!(
        batch
            .iter()
            .all(|value| value.to_bits() == oracle.to_bits())
    );
}

#[test]
fn tree_profile_a_exact_fitted_member_mean() {
    let features = vec![
        Vector64::from_slice(&[0.0]),
        Vector64::from_slice(&[1.0]),
        Vector64::from_slice(&[2.0]),
        Vector64::from_slice(&[3.0]),
    ];
    let targets = vec![2.0, 10.0, 2.0, 0.0];
    let sample_units = units(4);
    let spec = e0_04_spec();
    assert_e0_04_replay(&sample_units, &spec);

    let bag = DecisionTreeBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
    )
    .unwrap();

    assert_eq!(bag.members().len(), 2);
    assert_eq!(bag.feature_dimension(), 1);
    assert_root_leaf(&bag.members()[0], 2.0);
    assert_root_leaf(&bag.members()[1], 6.0);

    let query = Vector64::from_slice(&[1.0]);
    let predictions = [
        bag.members()[0].predict(&query).unwrap(),
        bag.members()[1].predict(&query).unwrap(),
    ];
    let scalar = bag.predict(&query).unwrap();
    let batch = vec![Observation::new(query); 3];
    let outputs = bag.predict_batch(&batch, 3).unwrap();
    assert_profile_a_aggregate(predictions, scalar, &outputs);
}

#[test]
fn ols_profile_a_exact_fitted_member_mean() {
    let features = vec![
        Vector64::from_slice(&[0.0]),
        Vector64::from_slice(&[1.0]),
        Vector64::from_slice(&[1.0]),
        Vector64::from_slice(&[0.0]),
    ];
    let targets = vec![0.0, 6.0, 2.0, 0.0];
    let sample_units = units(4);
    let spec = e0_04_spec();
    assert_e0_04_replay(&sample_units, &spec);

    let bag = OlsBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &FitConfig64 {
            intercept: false,
            ..FitConfig64::default()
        },
    )
    .unwrap();

    assert_eq!(bag.members().len(), 2);
    assert_eq!(bag.feature_dimension(), 1);
    assert_eq!(bag.members()[0].weights().as_slice(), &[2.0][..]);
    assert_eq!(bag.members()[1].weights().as_slice(), &[6.0][..]);
    assert_eq!(bag.members()[0].bias(), 0.0);
    assert_eq!(bag.members()[1].bias(), 0.0);

    let query = Vector64::from_slice(&[1.0]);
    let predictions = [
        bag.members()[0].predict(&query).unwrap(),
        bag.members()[1].predict(&query).unwrap(),
    ];
    let scalar = bag.predict(&query).unwrap();
    let batch = vec![Observation::new(query); 3];
    let outputs = bag.predict_batch(&batch, 3).unwrap();
    assert_profile_a_aggregate(predictions, scalar, &outputs);
}

#[test]
fn fitted_tree_mean_matches_independent_member_mean() {
    let features: Vec<Vector64> = (0..8)
        .map(|index| Vector64::from_slice(&[index as f64]))
        .collect();
    let targets: Vec<f64> = (0..8).map(|index| (index as f64) * 2.0).collect();
    let sample_units = units(8);
    let spec = BaggingSpec::new(
        3,
        SamplingMode::Method(SampleMethod::WithReplacement),
        8,
        0x1234,
        0,
    );
    let bag = DecisionTreeBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &TreeFitConfig::default(),
    )
    .unwrap();
    let row = Vector64::from_slice(&[3.5]);
    let independent = bag
        .members()
        .iter()
        .map(|member| member.predict(&row).unwrap())
        .sum::<f64>()
        / bag.members().len() as f64;
    assert_eq!(bag.predict(&row).unwrap(), independent);

    let batch = vec![Observation::new(row.clone()); 3];
    let batch_out = bag.predict_batch(&batch, 3).unwrap();
    for value in batch_out {
        assert_eq!(value, bag.predict(&row).unwrap());
    }
}

#[test]
fn fitted_ols_mean_matches_independent_member_mean() {
    let features: Vec<Vector64> = (0..8)
        .map(|index| Vector64::from_slice(&[index as f64]))
        .collect();
    let targets: Vec<f64> = (0..8).map(|index| 1.0 + 3.0 * index as f64).collect();
    let sample_units = units(8);
    let spec = BaggingSpec::new(
        2,
        SamplingMode::Method(SampleMethod::WithoutReplacement),
        6,
        0x99,
        0,
    );
    let bag = OlsBag64::fit(
        &features,
        &targets,
        &sample_units,
        &spec,
        &FitConfig64::default(),
    )
    .unwrap();
    let row = Vector64::from_slice(&[4.5]);
    let independent = bag
        .members()
        .iter()
        .map(|member| member.predict(&row).unwrap())
        .sum::<f64>()
        / bag.members().len() as f64;
    assert_eq!(bag.predict(&row).unwrap(), independent);
}
