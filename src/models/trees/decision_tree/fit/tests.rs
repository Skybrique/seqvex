use super::*;
use crate::foundation::numerical::{Vector, Vector64};
use crate::models::trees::decision_tree::{DecisionTreeNode, DecisionTreeNode64};

fn rows32(values: &[&[f32]]) -> Vec<Vector> {
    values.iter().map(|row| Vector::from_slice(row)).collect()
}

fn rows64(values: &[&[f64]]) -> Vec<Vector64> {
    values.iter().map(|row| Vector64::from_slice(row)).collect()
}

fn fit32(
    features: &[Vector],
    targets: &[f32],
    selected: &[usize],
    config: TreeFitConfig,
) -> (FitPlan<f32, DecisionTreeNode>, Vec<CandidateTrace<f32>>) {
    let mut trace = Vec::new();
    let plan = fit_core::<f32, Vector, DecisionTreeNode>(
        features,
        targets,
        selected,
        &config,
        Some(&mut trace),
    )
    .expect("fit should succeed");
    (plan, trace)
}

fn fit64(
    features: &[Vector64],
    targets: &[f64],
    selected: &[usize],
    config: TreeFitConfig,
) -> (FitPlan<f64, DecisionTreeNode64>, Vec<CandidateTrace<f64>>) {
    let mut trace = Vec::new();
    let plan = fit_core::<f64, Vector64, DecisionTreeNode64>(
        features,
        targets,
        selected,
        &config,
        Some(&mut trace),
    )
    .expect("fit should succeed");
    (plan, trace)
}

fn selected_all(n: usize) -> Vec<usize> {
    (0..n).collect()
}

trait TestNode<T: Scalar> {
    fn split_parts(&self) -> Option<(usize, T, usize, usize)>;
    fn leaf_value(&self) -> Option<T>;
}

impl TestNode<f32> for DecisionTreeNode {
    fn split_parts(&self) -> Option<(usize, f32, usize, usize)> {
        match self {
            DecisionTreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => Some((*feature_index, *threshold, *left, *right)),
            DecisionTreeNode::Leaf { .. } => None,
        }
    }

    fn leaf_value(&self) -> Option<f32> {
        match self {
            DecisionTreeNode::Leaf { value } => Some(*value),
            DecisionTreeNode::Split { .. } => None,
        }
    }
}

impl TestNode<f64> for DecisionTreeNode64 {
    fn split_parts(&self) -> Option<(usize, f64, usize, usize)> {
        match self {
            DecisionTreeNode64::Split {
                feature_index,
                threshold,
                left,
                right,
            } => Some((*feature_index, *threshold, *left, *right)),
            DecisionTreeNode64::Leaf { .. } => None,
        }
    }

    fn leaf_value(&self) -> Option<f64> {
        match self {
            DecisionTreeNode64::Leaf { value } => Some(*value),
            DecisionTreeNode64::Split { .. } => None,
        }
    }
}

fn split<T: Scalar, N: TestNode<T>>(node: &N) -> (usize, T, usize, usize) {
    node.split_parts().expect("expected split, got leaf")
}

fn leaf<T: Scalar, N: TestNode<T>>(node: &N) -> T {
    node.leaf_value().expect("expected leaf, got split")
}

#[test]
fn threshold_is_lower_observed_value_and_equality_goes_left() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let targets = [0.0, 0.0, 10.0, 10.0];
    let (plan, trace) = fit32(
        &features,
        &targets,
        &selected_all(4),
        TreeFitConfig::default(),
    );
    let (feature, threshold, left, right) = split(&plan.nodes[0]);
    assert_eq!(feature, 0);
    assert_eq!(threshold, 1.0);
    assert_eq!(leaf(&plan.nodes[left]), 0.0);
    assert_eq!(leaf(&plan.nodes[right]), 10.0);
    assert_eq!(plan.diagnostics.leaves, 2);
    assert_eq!(plan.diagnostics.depth, 1);
    let chosen = trace
        .iter()
        .find(|candidate| candidate.selected)
        .expect("a candidate is selected");
    assert_eq!(chosen.feature, 0);
    assert_eq!(chosen.threshold, 1.0);
    assert!(chosen.lo.unwrap() > 0.0);
}

#[test]
fn forced_root_leaf_uses_centered_mean() {
    let features = rows32(&[&[0.0], &[1.0]]);
    let targets = [10.0, 12.0];
    let config = TreeFitConfig {
        max_depth: 0,
        ..TreeFitConfig::default()
    };
    let (plan, _) = fit32(&features, &targets, &selected_all(2), config);
    assert_eq!(plan.nodes.len(), 1);
    assert_eq!(leaf(&plan.nodes[0]), 11.0);
    assert_eq!(plan.diagnostics.leaves, 1);
    assert_eq!(plan.diagnostics.depth, 0);
}

#[test]
fn zero_gain_is_a_successful_leaf() {
    let features = rows32(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
    let targets = [0.0, 2.0, 0.0, 2.0];
    let (plan, trace) = fit32(
        &features,
        &targets,
        &selected_all(4),
        TreeFitConfig::default(),
    );
    assert_eq!(plan.nodes.len(), 1);
    assert_eq!(leaf(&plan.nodes[0]), 1.0);
    assert_eq!(plan.diagnostics.computed_non_positive_gain_leaves, 1);
    let candidate = trace.iter().find(|c| c.valid).expect("one valid candidate");
    assert_eq!(candidate.gain.unwrap(), 0.0);
    assert!(!candidate.selected);
}

#[test]
fn evaluate_all_weak_first_candidate() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0]]);
    let targets = [0.0, 1.0, -1.0];
    let config = TreeFitConfig {
        max_depth: 1,
        ..TreeFitConfig::default()
    };
    let (plan, trace) = fit32(&features, &targets, &selected_all(3), config);
    let (feature, threshold, left, right) = split(&plan.nodes[0]);
    assert_eq!(feature, 0);
    assert_eq!(threshold, 1.0);
    assert_eq!(leaf(&plan.nodes[left]), 0.5);
    assert_eq!(leaf(&plan.nodes[right]), -1.0);
    let first = trace
        .iter()
        .find(|c| c.valid && c.threshold == 0.0)
        .expect("first candidate");
    assert_eq!(first.gain.unwrap(), 0.0);
    assert!(trace.iter().any(|c| c.selected && c.threshold == 1.0));
}

#[test]
fn near_floor_leaf_is_permitted() {
    let features = rows32(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
    let targets = [0.0, 2.0, 0.00479_f32, 2.00479_f32];
    let (plan, trace) = fit32(
        &features,
        &targets,
        &selected_all(4),
        TreeFitConfig::default(),
    );
    assert_eq!(plan.nodes.len(), 1);
    assert_eq!(leaf(&plan.nodes[0]), 1.002_395_f32);
    assert_eq!(
        plan.diagnostics
            .computed_positive_gain_below_confidence_leaves,
        1
    );
    let candidate = trace.iter().find(|c| c.valid).expect("one valid candidate");
    assert!(candidate.gain.unwrap() > 0.0);
    assert!(candidate.lo.unwrap() <= 0.0);
}

#[test]
fn cancellation_child_sse_is_correct_before_commit() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0]]);
    let targets = [0.0, 10_000.0, 10_001.0];
    let (plan, trace) = fit32(
        &features,
        &targets,
        &selected_all(3),
        TreeFitConfig::default(),
    );
    let root_candidate = trace
        .iter()
        .find(|c| c.valid && c.feature == 0 && c.threshold == 0.0)
        .expect("root candidate");
    assert_eq!(root_candidate.right_sse, Some(0.5));
    let (feature, threshold, _, right) = split(&plan.nodes[0]);
    assert_eq!(feature, 0);
    assert_eq!(threshold, 0.0);
    assert!(plan.nodes[right].split_parts().is_some());
}

#[test]
fn large_offset_singletons_retain_targets() {
    let features32 = rows32(&[&[0.0], &[1.0]]);
    let targets32 = [16_777_218.0_f32, 16_777_216.0_f32];
    let (plan32, _) = fit32(
        &features32,
        &targets32,
        &selected_all(2),
        TreeFitConfig::default(),
    );
    assert_eq!(plan32.diagnostics.training_sse, 2.0);
    let (_, _, left, right) = split(&plan32.nodes[0]);
    assert_eq!(leaf(&plan32.nodes[left]), targets32[0]);
    assert_eq!(leaf(&plan32.nodes[right]), targets32[1]);

    let features64 = rows64(&[&[0.0], &[1.0]]);
    let targets64 = [9_007_199_254_740_994.0_f64, 9_007_199_254_740_992.0_f64];
    let (plan64, _) = fit64(
        &features64,
        &targets64,
        &selected_all(2),
        TreeFitConfig::default(),
    );
    assert_eq!(plan64.diagnostics.training_sse, 2.0);
    let (_, _, left, right) = split(&plan64.nodes[0]);
    assert_eq!(leaf(&plan64.nodes[left]), targets64[0]);
    assert_eq!(leaf(&plan64.nodes[right]), targets64[1]);
}

#[test]
fn compensated_accumulation_stays_within_bound() {
    let mut targets = vec![0.0_f32, 4096.0];
    targets.extend(std::iter::repeat_n(1.0_f32, 1534));
    let features: Vec<Vector> = (0..targets.len())
        .map(|index| Vector::from_slice(&[index as f32]))
        .collect();
    let config = TreeFitConfig {
        max_leaves: 2,
        ..TreeFitConfig::default()
    };
    let (plan, trace) = fit32(&features, &targets, &selected_all(targets.len()), config);
    let candidate = trace.iter().find(|c| c.valid).expect("a candidate");
    let bound = candidate.b_parent;
    let reference = {
        let n = targets.len() as f64;
        let mean = targets.iter().map(|v| *v as f64).sum::<f64>() / n;
        targets
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
    };
    let error = (plan.diagnostics.training_sse as f64 - reference).abs();
    assert!(
        error <= bound as f64,
        "error {error} exceeded compensated bound {bound}"
    );
}

#[test]
fn pure_node_is_a_leaf() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0]]);
    let targets = [4.0, 4.0, 4.0];
    let (plan, trace) = fit32(
        &features,
        &targets,
        &selected_all(3),
        TreeFitConfig::default(),
    );
    assert_eq!(plan.nodes.len(), 1);
    assert_eq!(leaf(&plan.nodes[0]), 4.0);
    assert!(trace.is_empty());
}

#[test]
fn tie_break_prefers_lower_feature_index() {
    let features = rows32(&[&[0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], &[1.0, 1.0]]);
    let targets = [0.0, 0.0, 10.0, 10.0];
    let (plan, _) = fit32(
        &features,
        &targets,
        &selected_all(4),
        TreeFitConfig::default(),
    );
    let (feature, threshold, _, _) = split(&plan.nodes[0]);
    assert_eq!(feature, 0);
    assert_eq!(threshold, 0.0);
}

#[test]
fn invalid_candidates_are_traced_and_rejected() {
    let features = rows32(&[&[0.0], &[1.0], &[1.0], &[1.0]]);
    let targets = [0.0, 10.0, 10.0, 10.0];
    let config = TreeFitConfig {
        min_samples_leaf: 2,
        ..TreeFitConfig::default()
    };
    let (plan, trace) = fit32(&features, &targets, &selected_all(4), config);
    assert_eq!(plan.nodes.len(), 1);
    assert_eq!(leaf(&plan.nodes[0]), 7.5);
    assert!(trace.iter().any(|candidate| !candidate.valid));
    assert!(trace.iter().all(|candidate| !candidate.selected));
}

#[test]
fn work_limit_counts_rejected_boundaries() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0]]);
    let targets = [0.0, 10.0, 20.0];
    let config = TreeFitConfig {
        max_split_evaluations: 1,
        ..TreeFitConfig::default()
    };
    let error = fit_core::<f32, Vector, DecisionTreeNode>(
        &features,
        &targets,
        &selected_all(3),
        &config,
        None,
    )
    .expect_err("second boundary must exceed the limit");
    assert_eq!(
        error,
        TreeFitError::WorkBudgetExceeded {
            performed: 2,
            limit: 1
        }
    );
}

#[test]
fn validation_and_error_precedence() {
    let features = rows32(&[&[0.0], &[1.0]]);
    let targets = [0.0, 1.0];
    assert!(matches!(
        fit_core::<f32, Vector, DecisionTreeNode>(
            &features,
            &targets[..1],
            &[0],
            &TreeFitConfig::default(),
            None
        ),
        Err(TreeFitError::RowTargetMismatch {
            features: 2,
            targets: 1
        })
    ));
    assert!(matches!(
        fit_core::<f32, Vector, DecisionTreeNode>(
            &features,
            &targets,
            &[],
            &TreeFitConfig::default(),
            None
        ),
        Err(TreeFitError::TooFewRows)
    ));
    assert!(matches!(
        fit_core::<f32, Vector, DecisionTreeNode>(
            &features,
            &targets,
            &[2],
            &TreeFitConfig::default(),
            None
        ),
        Err(TreeFitError::IndexOutOfRange {
            position: 0,
            row: 2,
            len: 2
        })
    ));
    assert!(matches!(
        fit_core::<f32, Vector, DecisionTreeNode>(
            &features,
            &targets,
            &[0],
            &TreeFitConfig {
                min_samples_leaf: 0,
                ..TreeFitConfig::default()
            },
            None
        ),
        Err(TreeFitError::InvalidConfig)
    ));
    assert!(matches!(
        fit_core::<f32, Vector, DecisionTreeNode>(
            &features,
            &[0.0, f32::NAN],
            &[0, 1],
            &TreeFitConfig::default(),
            None
        ),
        Err(TreeFitError::NonFiniteTrainingValue { row: 1 })
    ));
}

#[test]
fn conversion_rejects_non_finite_source_and_keeps_source() {
    use crate::models::trees::decision_tree::{
        DecisionTreeNode64, TreeServingConversionError, TreeServingField, TreeUnderflowPolicy,
    };
    let features = rows64(&[&[0.0]]);
    let mut result = crate::models::trees::decision_tree::fit_tree_f64(
        &features,
        &[1.0],
        &[0],
        &TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
    )
    .expect("source fit");
    result.nodes[0] = DecisionTreeNode64::Leaf { value: f64::NAN };
    assert_eq!(
        result.to_serving_f32(TreeUnderflowPolicy::Reject),
        Err(TreeServingConversionError::SourceNonFinite {
            node: 0,
            field: TreeServingField::Leaf,
        })
    );
    match &result.nodes[0] {
        DecisionTreeNode64::Leaf { value } => assert!(value.is_nan()),
        DecisionTreeNode64::Split { .. } => panic!("expected leaf"),
    }
}

#[test]
fn workspace_cap_is_exact_and_typed() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let targets = [0.0, 0.0, 10.0, 10.0];
    let first = fit_core::<f32, Vector, DecisionTreeNode>(
        &features,
        &targets,
        &selected_all(4),
        &TreeFitConfig::default(),
        None,
    )
    .expect("baseline fit");
    let requested = first.diagnostics.requested_workspace_bytes;
    assert!(requested > 0);

    fit_core::<f32, Vector, DecisionTreeNode>(
        &features,
        &targets,
        &selected_all(4),
        &TreeFitConfig {
            max_workspace_bytes: requested,
            ..TreeFitConfig::default()
        },
        None,
    )
    .expect("exact workspace cap must succeed");

    assert!(matches!(
        fit_core::<f32, Vector, DecisionTreeNode>(
            &features,
            &targets,
            &selected_all(4),
            &TreeFitConfig {
                max_workspace_bytes: requested - 1,
                ..TreeFitConfig::default()
            },
            None
        ),
        Err(TreeFitError::Capacity {
            requested: actual,
            limit,
        }) if actual == requested && limit == requested - 1
    ));
}

// ---------------------------------------------------------------------------
// Independent split oracle: reference arithmetic, full valid-boundary enumeration.
// ---------------------------------------------------------------------------

trait RefScalar: Scalar {
    fn ref64(self) -> f64;
}
impl RefScalar for f32 {
    fn ref64(self) -> f64 {
        self as f64
    }
}
impl RefScalar for f64 {
    fn ref64(self) -> f64 {
        self
    }
}

#[derive(Debug, Clone)]
struct RefCandidate<T: Scalar> {
    feature: usize,
    threshold: T,
    parent_sse: f64,
    left_sse: f64,
    right_sse: f64,
    gain: f64,
}

fn round_f32(value: f64) -> f64 {
    (value as f32) as f64
}

fn rounded_margin_rejected(
    chosen_gain: Option<f64>,
    alternatives: &[(f64, f64)],
    round_profile: fn(f64) -> f64,
) -> bool {
    match chosen_gain {
        Some(chosen) => alternatives
            .iter()
            .any(|&(gain, bound)| round_profile(gain - 2.0 * bound) > round_profile(chosen)),
        None => alternatives
            .iter()
            .any(|&(gain, bound)| round_profile(gain - 2.0 * bound) > 0.0),
    }
}

fn sse64(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sse = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
    (mean, sse)
}

fn integer_value(value: f64) -> Option<i128> {
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    let integer = value as i128;
    (integer as f64 == value).then_some(integer)
}

fn integer_sum(values: &[f64]) -> Option<i128> {
    let mut sum = 0_i128;
    for value in values {
        sum = sum.checked_add(integer_value(*value)?)?;
    }
    Some(sum)
}

/// Exact integer SSE using `sum((y_i - y_j)^2 for i < j) / n`.
///
/// Returns `None` when a value is not exactly integer or the checked
/// numerator overflows. Non-integer fixtures fall back to [`sse64`].
fn integer_sse(values: &[f64]) -> Option<f64> {
    let mut integers = Vec::with_capacity(values.len());
    for value in values {
        integers.push(integer_value(*value)?);
    }
    let n = integers.len();
    if n < 2 {
        return Some(0.0);
    }
    let mut numerator = 0_i128;
    for i in 0..n {
        for j in (i + 1)..n {
            let difference = integers[i].checked_sub(integers[j])?;
            numerator = numerator.checked_add(difference.checked_mul(difference)?)?;
        }
    }
    Some(numerator as f64 / n as f64)
}

fn reference_sse(values: &[f64]) -> f64 {
    integer_sse(values).unwrap_or_else(|| sse64(values).1)
}

/// Reference gain before production rounding.
///
/// Integer-valued fixtures use exact integer arithmetic for
/// `(n_R*S_L - n_L*S_R)^2 / ((n_L+n_R)*n_L*n_R)`. Non-integer fixtures fall
/// back to the direct f64 form so a gain is never recovered by subtracting
/// three rounded SSEs.
fn reference_gain(left: &[f64], right: &[f64]) -> f64 {
    let n_l = left.len() as i128;
    let n_r = right.len() as i128;
    if let (Some(s_l), Some(s_r)) = (integer_sum(left), integer_sum(right)) {
        let denominator = (n_l + n_r)
            .checked_mul(n_l)
            .and_then(|value| value.checked_mul(n_r));
        let numerator = n_r
            .checked_mul(s_l)
            .and_then(|a| n_l.checked_mul(s_r).and_then(|b| a.checked_sub(b)));
        if let (Some(denominator), Some(numerator)) = (denominator, numerator) {
            if let Some(squared) = numerator.checked_mul(numerator) {
                return squared as f64 / denominator as f64;
            }
        }
    }

    let s_l = left.iter().sum::<f64>();
    let s_r = right.iter().sum::<f64>();
    let numerator = n_r as f64 * s_l - n_l as f64 * s_r;
    numerator * numerator / ((n_l + n_r) as f64 * n_l as f64 * n_r as f64)
}

fn reference_candidates<T: Scalar + RefScalar, R: FeatureRow<T>>(
    features: &[R],
    targets: &[T],
    rows: &[usize],
    min_samples_leaf: usize,
) -> Vec<RefCandidate<T>> {
    let mut source = rows.to_vec();
    source.sort_unstable();
    let parent_sse = reference_sse(
        &source
            .iter()
            .map(|&row| targets[row].ref64())
            .collect::<Vec<_>>(),
    );
    let mut output = Vec::new();
    let width = features[source[0]].width();
    for feature in 0..width {
        let mut ordered: Vec<(T, usize)> = source
            .iter()
            .map(|&row| (features[row].feature(feature), row))
            .collect();
        ordered.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        for k in 0..ordered.len() - 1 {
            if ordered[k].0 >= ordered[k + 1].0 {
                continue;
            }
            let left_n = k + 1;
            let right_n = ordered.len() - left_n;
            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }
            let threshold = ordered[k].0;
            let left_values = ordered[..left_n]
                .iter()
                .map(|(_, row)| targets[*row].ref64())
                .collect::<Vec<_>>();
            let right_values = ordered[left_n..]
                .iter()
                .map(|(_, row)| targets[*row].ref64())
                .collect::<Vec<_>>();
            let left = reference_sse(&left_values);
            let right = reference_sse(&right_values);
            output.push(RefCandidate {
                feature,
                threshold,
                parent_sse,
                left_sse: left,
                right_sse: right,
                gain: reference_gain(&left_values, &right_values),
            });
        }
    }
    output
}

fn is_confidence_stop(stop: LeafStop) -> bool {
    matches!(stop, LeafStop::NonPositive | LeafStop::BelowConfidence)
}

fn count_missed_reference<T: Scalar + RefScalar, R: FeatureRow<T>>(
    features: &[R],
    targets: &[T],
    audit: &[LeafAudit<T>],
) -> (usize, usize) {
    let mut candidates = 0usize;
    let mut leaves = 0usize;
    for leaf in audit {
        if !is_confidence_stop(leaf.stop) {
            continue;
        }
        let positive = reference_candidates(features, targets, &leaf.rows, 1)
            .iter()
            .filter(|candidate| candidate.gain > 0.0)
            .count();
        candidates += positive;
        if positive > 0 {
            leaves += 1;
        }
    }
    (candidates, leaves)
}

fn production_best<T: Scalar>(trace: &[CandidateTrace<T>], node: usize) -> Option<(usize, T, T)> {
    trace
        .iter()
        .filter(|candidate| candidate.node == node && candidate.valid)
        .filter_map(|candidate| {
            let lo = candidate.lo?;
            (lo > T::zero()).then_some((candidate.feature, candidate.threshold, lo))
        })
        .fold(None, |best, current| match best {
            None => Some(current),
            Some(previous) => {
                let better = current.2 > previous.2
                    || (current.2 == previous.2
                        && (current.0 < previous.0
                            || (current.0 == previous.0 && current.1 < previous.1)));
                Some(if better { current } else { previous })
            }
        })
}

fn assert_reference_inside_production<T: Scalar + RefScalar>(
    reference: &RefCandidate<T>,
    production: &CandidateTrace<T>,
) {
    assert!(
        (reference.parent_sse - production.parent_sse.ref64()).abs() <= production.b_parent.ref64(),
        "parent reference outside production interval"
    );
    assert!(
        (reference.left_sse - production.left_sse.unwrap().ref64()).abs()
            <= production.b_left.unwrap().ref64(),
        "left reference outside production interval"
    );
    assert!(
        (reference.right_sse - production.right_sse.unwrap().ref64()).abs()
            <= production.b_right.unwrap().ref64(),
        "right reference outside production interval"
    );
    assert!(
        (reference.gain - production.gain.unwrap().ref64()).abs()
            <= production.b_gain.unwrap().ref64(),
        "gain reference outside production interval"
    );
}

#[test]
fn independent_oracle_enumerates_valid_boundaries_and_choice() {
    let features = rows32(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let targets = [0.0, 0.0, 10.0, 10.0];
    let (plan, trace) = fit32(
        &features,
        &targets,
        &selected_all(4),
        TreeFitConfig::default(),
    );
    let node_rows = trace.first().expect("root trace").node_rows.clone();
    let reference = reference_candidates(&features, &targets, &node_rows, 1);
    assert_eq!(
        reference.len(),
        trace.iter().filter(|candidate| candidate.valid).count()
    );
    for candidate in &reference {
        let production = trace
            .iter()
            .find(|entry| {
                entry.valid
                    && entry.feature == candidate.feature
                    && entry.threshold == candidate.threshold
            })
            .expect("production candidate");
        assert_reference_inside_production(candidate, production);
    }
    let best = production_best(&trace, 0).expect("confident root split");
    assert_eq!(best.0, 0);
    assert_eq!(best.1, 1.0);
    let selected = trace
        .iter()
        .find(|candidate| candidate.selected)
        .expect("selected candidate");
    assert_eq!((selected.feature, selected.threshold), (0, 1.0));
    let (feature, threshold, _, _) = split(&plan.nodes[0]);
    assert_eq!((feature, threshold), (0, 1.0));

    let chosen_reference = reference
        .iter()
        .find(|candidate| candidate.feature == best.0 && candidate.threshold == best.1)
        .expect("chosen reference");
    let alternatives: Vec<(f64, f64)> = reference
        .iter()
        .filter(|candidate| candidate.feature != best.0 || candidate.threshold != best.1)
        .map(|candidate| {
            let production = trace
                .iter()
                .find(|entry| {
                    entry.valid
                        && entry.feature == candidate.feature
                        && entry.threshold == candidate.threshold
                })
                .expect("production alternative");
            (candidate.gain, production.b_gain.unwrap().ref64())
        })
        .collect();
    assert!(!rounded_margin_rejected(
        Some(chosen_reference.gain),
        &alternatives,
        round_f32,
    ));
}

#[test]
fn computed_zero_reference_positive_is_leaf_and_counts_differ() {
    let features = rows32(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
    let targets = [0.0_f32, 2.0, 0.00001, 2.00001];
    let mut trace = Vec::new();
    let mut audit = Vec::new();
    let plan = fit_core_with_audit::<f32, Vector, DecisionTreeNode>(
        &features,
        &targets,
        &selected_all(4),
        &TreeFitConfig::default(),
        Some(&mut trace),
        Some(&mut audit),
    )
    .expect("computed-zero fit");
    assert_eq!(plan.nodes.len(), 1);
    let production_positive = trace
        .iter()
        .filter(|candidate| candidate.valid && candidate.gain.unwrap() > 0.0)
        .count();
    assert_eq!(production_positive, 0);
    assert_eq!(
        plan.diagnostics
            .computed_positive_gain_candidates_below_confidence,
        0
    );
    assert_eq!(audit.len(), 1);
    assert!(is_confidence_stop(audit[0].stop));
    let (oracle_candidates, oracle_leaves) = count_missed_reference(&features, &targets, &audit);
    assert_eq!((oracle_candidates, oracle_leaves), (1, 1));
    println!(
        "computed-zero production_positive={production_positive} oracle_missed_candidates={oracle_candidates} oracle_affected_leaves={oracle_leaves} confidence_stop={}",
        is_confidence_stop(audit[0].stop)
    );
    assert!(trace.iter().all(|candidate| !candidate.selected));

    let reference = reference_candidates(&features, &targets, &audit[0].rows, 1);
    let production = trace
        .iter()
        .find(|entry| entry.valid)
        .expect("production candidate");
    assert_reference_inside_production(&reference[0], production);
    let alternatives: Vec<(f64, f64)> = reference
        .iter()
        .map(|candidate| (candidate.gain, production.b_gain.unwrap().ref64()))
        .collect();
    assert!(!rounded_margin_rejected(None, &alternatives, round_f32));
}

#[test]
fn dyadic_rounding_tie_accepts_w() {
    let b_w = 2.0_f64.powi(-18);
    let g_w = 1.0_f64 + 2.0_f64.powi(-18);
    let g_w_ref = 1.0_f64 + 2.0_f64.powi(-26);
    let b_i = 2.0_f64.powi(-18) + 2.0_f64.powi(-24);
    let g_i = 1.0_f64 + 2.0_f64.powi(-18) + 2.0_f64.powi(-23);
    let g_i_ref = g_i + b_i - 2.0_f64.powi(-26);

    assert!((g_w_ref - g_w).abs() <= b_w);
    assert!((g_i_ref - g_i).abs() <= b_i);
    assert!(g_i_ref - 2.0 * b_i > g_w_ref);
    let left = (g_i_ref - 2.0 * b_i) as f32;
    let right = g_w_ref as f32;
    assert_eq!(left, 1.0);
    assert_eq!(right, 1.0);
    assert!(left <= right);

    let dyadic_alternatives = [(g_i_ref, b_i)];
    let accepted = !rounded_margin_rejected(Some(g_w_ref), &dyadic_alternatives, round_f32);
    assert!(accepted);
    let separated_alternatives = [(g_i_ref + 1.0, b_i)];
    let separated_rejected =
        rounded_margin_rejected(Some(g_w_ref), &separated_alternatives, round_f32);
    assert!(separated_rejected);
    println!(
        "dyadic rounded_w={} rounded_i={} accepted_w={accepted} separated_rejected={separated_rejected}",
        round_f32(g_w_ref),
        round_f32(g_i_ref - 2.0 * b_i),
    );
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn workload_2k_vectors() -> (Vec<Vector>, Vec<f32>, Vec<Vector64>, Vec<f64>) {
    let mut f32_rows = Vec::with_capacity(2048);
    let mut f32_targets = Vec::with_capacity(2048);
    let mut f64_rows = Vec::with_capacity(2048);
    let mut f64_targets = Vec::with_capacity(2048);
    for index in 0..2048usize {
        let row: Vec<f32> = (0..8)
            .map(|feature| (((index as u64 * 131 + feature as u64 * 977) % 1009) as f32) / 1008.0)
            .collect();
        let noise = (((splitmix64(index as u64) % 2001) as f32 / 1000.0) - 1.0) * 0.05;
        let target = 10.0 * f32::from(row[0] > 0.5)
            + 5.0 * f32::from(row[1] > 0.25)
            + 2.0 * f32::from(row[2] > 0.75)
            + 1.0 * f32::from(row[3] > 0.5)
            + noise;
        f64_rows.push(Vector64::from_slice(
            &row.iter().map(|value| *value as f64).collect::<Vec<_>>(),
        ));
        f64_targets.push(target as f64);
        f32_rows.push(Vector::from_slice(&row));
        f32_targets.push(target);
    }
    (f32_rows, f32_targets, f64_rows, f64_targets)
}

#[test]
#[ignore = "pinned 2K independent reference-count replay; run explicitly"]
fn pinned_2k_reference_counts() {
    let selected: Vec<usize> = (0..1536).collect();
    let (f32_rows, f32_targets, f64_rows, f64_targets) = workload_2k_vectors();

    let mut audit32 = Vec::new();
    let plan32 = fit_core_with_audit::<f32, Vector, DecisionTreeNode>(
        &f32_rows,
        &f32_targets,
        &selected,
        &TreeFitConfig::default(),
        None,
        Some(&mut audit32),
    )
    .expect("2K C fit");
    let (reference_candidates32, reference_leaves32) =
        count_missed_reference(&f32_rows, &f32_targets, &audit32);

    let mut audit64 = Vec::new();
    let plan64 = fit_core_with_audit::<f64, Vector64, DecisionTreeNode64>(
        &f64_rows,
        &f64_targets,
        &selected,
        &TreeFitConfig::default(),
        None,
        Some(&mut audit64),
    )
    .expect("2K A fit");
    let (reference_candidates64, reference_leaves64) =
        count_missed_reference(&f64_rows, &f64_targets, &audit64);

    println!(
        "2K production A positive_candidates={} affected_leaves={} | oracle missed_reference_candidates={} affected_leaves={}",
        plan64
            .diagnostics
            .computed_positive_gain_candidates_below_confidence,
        plan64
            .diagnostics
            .computed_positive_gain_below_confidence_leaves,
        reference_candidates64,
        reference_leaves64,
    );
    println!(
        "2K production C positive_candidates={} affected_leaves={} | oracle missed_reference_candidates={} affected_leaves={}",
        plan32
            .diagnostics
            .computed_positive_gain_candidates_below_confidence,
        plan32
            .diagnostics
            .computed_positive_gain_below_confidence_leaves,
        reference_candidates32,
        reference_leaves32,
    );

    assert_eq!(reference_candidates64, 0);
    assert_eq!(reference_leaves64, 0);
    assert_eq!(reference_candidates32, 24);
    assert_eq!(reference_leaves32, 3);
    assert_eq!(
        plan32
            .diagnostics
            .computed_positive_gain_candidates_below_confidence,
        24
    );
    assert_eq!(
        plan32
            .diagnostics
            .computed_positive_gain_below_confidence_leaves,
        3
    );
}

#[test]
fn profile_a_reference_gain_preserves_integer_gain() {
    let features = rows64(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
    let targets = [0.0, (1u64 << 30) as f64, 1.0, ((1u64 << 30) + 1) as f64];
    let mut trace = Vec::new();
    let mut audit = Vec::new();
    let plan = fit_core_with_audit::<f64, Vector64, DecisionTreeNode64>(
        &features,
        &targets,
        &selected_all(4),
        &TreeFitConfig::default(),
        Some(&mut trace),
        Some(&mut audit),
    )
    .expect("Profile A large-gain fit");

    let production_positive = trace
        .iter()
        .filter(|candidate| candidate.valid && candidate.gain.unwrap() > 0.0)
        .count();
    assert_eq!(production_positive, 0);
    assert_eq!(
        plan.diagnostics
            .computed_positive_gain_candidates_below_confidence,
        0
    );
    assert_eq!(audit.len(), 1);
    assert!(is_confidence_stop(audit[0].stop));
    assert_eq!(count_missed_reference(&features, &targets, &audit), (1, 1));

    let reference = reference_candidates(&features, &targets, &audit[0].rows, 1);
    assert_eq!(reference.len(), 1);
    assert_eq!(reference[0].gain, 1.0);
    let production = trace
        .iter()
        .find(|entry| entry.valid)
        .expect("production candidate");
    assert_reference_inside_production(&reference[0], production);
    let alternatives = [(reference[0].gain, production.b_gain.unwrap())];
    let accepts_leaf = !rounded_margin_rejected(None, &alternatives, |value| value);
    assert!(accepts_leaf);
    println!(
        "profile-a-large-gain production_positive={production_positive} oracle_missed_candidates=1 oracle_affected_leaves=1 exact_gain={} verifier_accepts_leaf={accepts_leaf}",
        reference[0].gain
    );
}

#[test]
fn shared_oracle_preserves_large_offset_integer_sse() {
    let features = rows64(&[&[0.0], &[1.0]]);
    let targets = [((1u64 << 53) + 2) as f64, (1u64 << 53) as f64];

    // The old reference helper loses the exact difference through f64 rounding.
    assert_eq!(sse64(&targets).1, 4.0);
    assert_eq!(reference_sse(&targets), 2.0);

    let (plan, trace) = fit64(
        &features,
        &targets,
        &selected_all(2),
        TreeFitConfig::default(),
    );
    assert_eq!(plan.diagnostics.training_sse, 2.0);
    let node_rows = trace.first().expect("root trace").node_rows.clone();
    let reference = reference_candidates(&features, &targets, &node_rows, 1);
    assert_eq!(reference.len(), 1);
    assert_eq!(reference[0].parent_sse, 2.0);
    assert_eq!(reference[0].left_sse, 0.0);
    assert_eq!(reference[0].right_sse, 0.0);
    assert_eq!(reference[0].gain, 2.0);
    let production = trace
        .iter()
        .find(|candidate| candidate.valid)
        .expect("production candidate");
    assert_reference_inside_production(&reference[0], production);
    println!(
        "large-offset shared oracle parent_sse={} left_sse={} right_sse={} gain={} old_sse64={}",
        reference[0].parent_sse,
        reference[0].left_sse,
        reference[0].right_sse,
        reference[0].gain,
        sse64(&targets).1,
    );
}
