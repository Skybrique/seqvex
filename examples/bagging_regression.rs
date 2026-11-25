//! Runnable Profile B regression-bagging workflow on an ordered split.

use seqvex::foundation::numerical::Vector64;
use seqvex::foundation::observation::Observation;
use seqvex::models::ensemble::bagging::{BaggingSpec, DecisionTreeBag};
use seqvex::models::trees::decision_tree::{TreeFitConfig, TreeUnderflowPolicy};
use seqvex::preprocessing::sampling::{SampleMethod, SampleUnit, SamplingMode};

#[allow(clippy::single_range_in_vec_init)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features: Vec<Vector64> = (0..24)
        .map(|index| Vector64::from_slice(&[index as f64, (index % 5) as f64]))
        .collect();
    let targets: Vec<f64> = (0..24)
        .map(|index| if index < 12 { 1.0 } else { 3.0 })
        .collect();
    let train_units: Vec<SampleUnit> = (0..18)
        .map(|index| SampleUnit {
            rows: vec![index..index + 1],
            outcomes: Vec::new(),
            sequence: 0,
        })
        .collect();
    let spec = BaggingSpec::new(
        4,
        SamplingMode::Method(SampleMethod::WithReplacement),
        18,
        0x1060,
        0,
    );

    let candidate = DecisionTreeBag::fit_from_f64(
        &features,
        &targets,
        &train_units,
        &spec,
        &TreeFitConfig::default(),
        TreeUnderflowPolicy::Reject,
    )?;
    let mut active = None;
    active.replace(candidate);

    let holdout: Vec<Observation<seqvex::foundation::numerical::Vector>> = (18..24)
        .map(|row| {
            Observation::new(seqvex::foundation::numerical::Vector::from_slice(&[
                row as f32,
                (row % 5) as f32,
            ]))
        })
        .collect();
    let bag = active.expect("published candidate");
    let predictions = bag.predict_batch(&holdout, holdout.len())?;
    let abs = predictions
        .iter()
        .zip(18..24)
        .map(|(prediction, row)| (*prediction as f64 - targets[row]).abs())
        .sum::<f64>();
    println!(
        "members={} retained_bytes={} holdout_rows={} test_mae={:.6}",
        bag.members().len(),
        bag.retained_bytes(),
        holdout.len(),
        abs / holdout.len() as f64
    );
    Ok(())
}
