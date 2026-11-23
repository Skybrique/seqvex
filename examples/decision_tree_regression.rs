//! Runnable standalone Decision Tree regression example.
//!
//! The caller owns the rows and the selected-row list. Profile A fits and serves
//! in f64, Profile B narrows to the existing f32 serving type, and Profile C fits
//! with strict f32 arithmetic. Installation is an explicit caller replacement;
//! the example retains the previous active model until validation succeeds.

use std::process::ExitCode;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::models::trees::decision_tree::{
    DecisionTree, TreeFitConfig, TreeUnderflowPolicy, fit_tree_f32, fit_tree_f64,
};

fn run() -> Result<(), String> {
    let features32 = vec![
        Vector::from_slice(&[0.0]),
        Vector::from_slice(&[1.0]),
        Vector::from_slice(&[2.0]),
        Vector::from_slice(&[3.0]),
    ];
    let features64 = vec![
        Vector64::from_slice(&[0.0]),
        Vector64::from_slice(&[1.0]),
        Vector64::from_slice(&[2.0]),
        Vector64::from_slice(&[3.0]),
    ];
    let targets32 = [0.0_f32, 0.0, 10.0, 10.0];
    let targets64 = [0.0_f64, 0.0, 10.0, 10.0];
    let selected = [0_usize, 1, 2, 3];
    let config = TreeFitConfig::default();

    let a = fit_tree_f64(&features64, &targets64, &selected, &config)
        .map_err(|error| format!("A fit failed: {error}"))?;
    let b = a
        .to_serving_f32(TreeUnderflowPolicy::Reject)
        .map_err(|error| format!("B conversion failed: {error}"))?;
    let c = fit_tree_f32(&features32, &targets32, &selected, &config)
        .map_err(|error| format!("C fit failed: {error}"))?;

    let mut active = DecisionTree::new(
        1,
        vec![seqvex::models::trees::decision_tree::DecisionTreeNode::Leaf { value: -1.0 }],
    )
    .map_err(|error| format!("old model construction failed: {error:?}"))?;
    let old_prediction = active
        .predict(&Vector::from_slice(&[1.5]))
        .map_err(|error| format!("old prediction failed: {error:?}"))?;

    let b_prediction = b
        .predict(&Vector::from_slice(&[1.5]))
        .map_err(|error| format!("B prediction failed: {error:?}"))?;
    if b_prediction != 10.0 {
        return Err(format!("B expected 10.0, got {b_prediction}"));
    }
    active = b;

    let q = Vector64::from_slice(&[1.5]);
    let a_prediction = a
        .predictor()
        .predict(&q)
        .map_err(|error| format!("A prediction failed: {error:?}"))?;
    let c_prediction = c
        .predictor()
        .predict(&Vector::from_slice(&[1.5]))
        .map_err(|error| format!("C prediction failed: {error:?}"))?;
    let installed = active
        .predict(&Vector::from_slice(&[1.5]))
        .map_err(|error| format!("installed prediction failed: {error:?}"))?;
    if a_prediction != 10.0 || c_prediction != 10.0 || installed != 10.0 {
        return Err(format!(
            "expected A/C/installed 10.0, got {a_prediction}/{c_prediction}/{installed}"
        ));
    }

    println!(
        "A leaves={} C leaves={} A={a_prediction} B={b_prediction} C={c_prediction} installed={installed} old={old_prediction}",
        a.diagnostics().leaves,
        c.diagnostics().leaves
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("decision_tree_regression: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn example_workflow_succeeds() {
        super::run().expect("decision tree regression example");
    }
}
