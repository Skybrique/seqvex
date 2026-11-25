use seqvex::foundation::numerical::Vector64;
use seqvex::foundation::observation::Observation;
use seqvex::models::ensemble::bagging::{BaggingSpec, DecisionTreeBag64, OlsBag64};
use seqvex::models::linear::ols::FitConfig64;
use seqvex::models::trees::decision_tree::TreeFitConfig;
use seqvex::preprocessing::sampling::{OutcomeInterval, SampleMethod, SampleUnit, SamplingMode};

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
