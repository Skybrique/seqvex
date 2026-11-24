//! Profile A: native f64 fit.

use crate::foundation::numerical::Vector64;

use super::{
    DecisionTreeNode64, TreeFitConfig, TreeFitDiagnostics64, TreeFitError, TreeFitResult64,
    fit_core,
};

/// Fits a regression tree with native f64 arithmetic.
pub fn fit_tree_f64(
    features: &[Vector64],
    targets: &[f64],
    selected: &[usize],
    config: &TreeFitConfig,
) -> Result<TreeFitResult64, TreeFitError> {
    let plan =
        fit_core::<f64, Vector64, DecisionTreeNode64>(features, targets, selected, config, None)?;
    let nodes = plan.nodes;
    let diagnostics = TreeFitDiagnostics64 {
        rows: plan.diagnostics.rows,
        features: plan.diagnostics.features,
        nodes: nodes.len(),
        leaves: plan.diagnostics.leaves,
        depth: plan.diagnostics.depth,
        training_sse: plan.diagnostics.training_sse,
        requested_workspace_bytes: plan.diagnostics.requested_workspace_bytes,
        split_evaluations: plan.diagnostics.split_evaluations,
        computed_non_positive_gain_leaves: plan.diagnostics.computed_non_positive_gain_leaves,
        computed_positive_gain_below_confidence_leaves: plan
            .diagnostics
            .computed_positive_gain_below_confidence_leaves,
        computed_positive_gain_candidates_below_confidence: plan
            .diagnostics
            .computed_positive_gain_candidates_below_confidence,
    };
    Ok(TreeFitResult64 {
        feature_dimension: plan.feature_dimension,
        nodes,
        diagnostics,
    })
}
