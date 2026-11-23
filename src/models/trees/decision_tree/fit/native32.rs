//! Profile C: strict f32 fit.

use crate::foundation::numerical::Vector;

use super::{
    DecisionTreeNode, TreeFitConfig, TreeFitDiagnostics32, TreeFitError, TreeFitResult32, fit_core,
};

/// Fits a regression tree with strict f32 arithmetic.
pub fn fit_tree_f32(
    features: &[Vector],
    targets: &[f32],
    selected: &[usize],
    config: &TreeFitConfig,
) -> Result<TreeFitResult32, TreeFitError> {
    let plan =
        fit_core::<f32, Vector, DecisionTreeNode>(features, targets, selected, config, None)?;
    let nodes = plan.nodes;
    let diagnostics = TreeFitDiagnostics32 {
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
    Ok(TreeFitResult32 {
        feature_dimension: plan.feature_dimension,
        nodes,
        diagnostics,
    })
}
