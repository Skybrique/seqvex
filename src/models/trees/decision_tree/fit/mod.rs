//! Selected-row regression fitting for the decision tree.
//!
//! The public entry points are [`fit_tree_f64`] and [`fit_tree_f32`]. The
//! implementation is generic over a private scalar trait so Profile C stays in
//! `f32` throughout; there is no hidden `f64` promotion.

use std::cmp::Ordering;
use std::fmt;
use std::mem::size_of;

use crate::foundation::numerical::{Vector, Vector64};

use super::{
    DecisionTree, DecisionTree64, DecisionTreeError, DecisionTreeNode, DecisionTreeNode64,
};

mod native32;
mod native64;

pub use native32::fit_tree_f32;
pub use native64::fit_tree_f64;

#[cfg(test)]
mod tests;

/// Default requested-payload workspace for the pinned layout.
pub const DEFAULT_WORKSPACE_BYTES: usize = 8 * 1024 * 1024;
/// Default candidate-boundary limit.
pub const DEFAULT_SPLIT_EVALUATIONS: u64 = 10_000_000;

/// Fit configuration shared by Profiles A and C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeFitConfig {
    /// Minimum rows on each side of a split.
    pub min_samples_leaf: usize,
    /// Maximum root-to-leaf depth. `0` produces a single leaf.
    pub max_depth: usize,
    /// Maximum number of leaves.
    pub max_leaves: usize,
    /// Requested-payload workspace cap in bytes.
    pub max_workspace_bytes: usize,
    /// Maximum candidate boundaries examined.
    pub max_split_evaluations: u64,
}

impl Default for TreeFitConfig {
    fn default() -> Self {
        Self {
            min_samples_leaf: 1,
            max_depth: 16,
            max_leaves: 1024,
            max_workspace_bytes: DEFAULT_WORKSPACE_BYTES,
            max_split_evaluations: DEFAULT_SPLIT_EVALUATIONS,
        }
    }
}

/// Fit failure classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeFitError {
    /// Whole-array feature/target lengths differ.
    RowTargetMismatch {
        /// Number of feature rows.
        features: usize,
        /// Number of targets.
        targets: usize,
    },
    /// The selected row list is empty.
    TooFewRows,
    /// A selected row index is out of range.
    IndexOutOfRange {
        /// Position in the supplied selection order.
        position: usize,
        /// Offending row index.
        row: usize,
        /// Number of available rows.
        len: usize,
    },
    /// The first selected row has zero width.
    ZeroFeatures,
    /// A selected row has a different width.
    RaggedFeatures {
        /// Offending source row.
        row: usize,
        /// Expected width.
        expected: usize,
        /// Actual width.
        actual: usize,
    },
    /// A configuration field is invalid.
    InvalidConfig,
    /// Checked size arithmetic overflowed.
    SizeOverflow,
    /// The requested workspace or node capacity exceeds its limit.
    Capacity {
        /// Requested bytes or nodes.
        requested: usize,
        /// Configured limit.
        limit: usize,
    },
    /// A fallible allocation failed.
    AllocationFailed,
    /// A selected feature or target is not finite.
    NonFiniteTrainingValue {
        /// Source row.
        row: usize,
    },
    /// The candidate-boundary limit was exceeded.
    WorkBudgetExceeded {
        /// First boundary not evaluated.
        performed: u64,
        /// Configured limit.
        limit: u64,
    },
    /// A numerical score was unreliable or non-finite.
    NumericalFailure {
        /// Node index.
        node: usize,
    },
    /// A fit-produced topology failed the existing tree error contract.
    Topology(DecisionTreeError),
}

impl fmt::Display for TreeFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TreeFitError {}

/// Underflow policy for checked f64-to-f32 serving conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TreeUnderflowPolicy {
    /// Reject a nonzero value that narrows to zero.
    #[default]
    Reject,
    /// Deploy signed zero.
    AllowLoss,
}

/// Field that failed checked serving conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeServingField {
    /// Split threshold.
    Threshold,
    /// Leaf value.
    Leaf,
}

/// Checked f64-to-f32 serving conversion failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeServingConversionError {
    /// The source value was not finite.
    SourceNonFinite {
        /// Source node index.
        node: usize,
        /// Failing field.
        field: TreeServingField,
    },
    /// The source value overflowed `f32`.
    Overflow {
        /// Source node index.
        node: usize,
        /// Failing field.
        field: TreeServingField,
    },
    /// The source value narrowed to zero under `Reject`.
    Underflow {
        /// Source node index.
        node: usize,
        /// Failing field.
        field: TreeServingField,
    },
}

impl fmt::Display for TreeServingConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TreeServingConversionError {}

/// Diagnostics for a fitted Profile A tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeFitDiagnostics64 {
    /// Expanded selected rows.
    pub rows: usize,
    /// Feature dimension.
    pub features: usize,
    /// Node count.
    pub nodes: usize,
    /// Leaf count.
    pub leaves: usize,
    /// Maximum depth reached.
    pub depth: usize,
    /// Root training SSE in f64.
    pub training_sse: f64,
    /// Conservative requested-payload bytes.
    pub requested_workspace_bytes: usize,
    /// Candidate boundaries examined.
    pub split_evaluations: u64,
    /// Leaves stopped with only computed non-positive nominal gains.
    pub computed_non_positive_gain_leaves: usize,
    /// Leaves stopped with a computed positive gain below confidence.
    pub computed_positive_gain_below_confidence_leaves: usize,
    /// Valid candidates with a computed positive gain at those leaves.
    pub computed_positive_gain_candidates_below_confidence: usize,
}

/// Diagnostics for a fitted Profile C tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeFitDiagnostics32 {
    /// Expanded selected rows.
    pub rows: usize,
    /// Feature dimension.
    pub features: usize,
    /// Node count.
    pub nodes: usize,
    /// Leaf count.
    pub leaves: usize,
    /// Maximum depth reached.
    pub depth: usize,
    /// Root training SSE in f32.
    pub training_sse: f32,
    /// Conservative requested-payload bytes.
    pub requested_workspace_bytes: usize,
    /// Candidate boundaries examined.
    pub split_evaluations: u64,
    /// Leaves stopped with only computed non-positive nominal gains.
    pub computed_non_positive_gain_leaves: usize,
    /// Leaves stopped with a computed positive gain below confidence.
    pub computed_positive_gain_below_confidence_leaves: usize,
    /// Valid candidates with a computed positive gain at those leaves.
    pub computed_positive_gain_candidates_below_confidence: usize,
}

/// Fitted Profile A result.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeFitResult64 {
    feature_dimension: usize,
    nodes: Vec<DecisionTreeNode64>,
    diagnostics: TreeFitDiagnostics64,
}

impl TreeFitResult64 {
    /// Fitted nodes in pre-order.
    pub fn nodes(&self) -> &[DecisionTreeNode64] {
        &self.nodes
    }

    /// Feature dimension.
    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    /// Fit diagnostics.
    pub fn diagnostics(&self) -> &TreeFitDiagnostics64 {
        &self.diagnostics
    }

    /// Builds an independent f64 serving model.
    pub fn predictor(&self) -> DecisionTree64 {
        DecisionTree64::from_validated_parts(self.feature_dimension, self.nodes.clone())
    }

    /// Moves the fitted nodes into an f64 serving model.
    pub fn into_predictor(self) -> DecisionTree64 {
        DecisionTree64::from_validated_parts(self.feature_dimension, self.nodes)
    }

    /// Narrows the fitted tree to the existing f32 serving type.
    pub fn to_serving_f32(
        &self,
        policy: TreeUnderflowPolicy,
    ) -> Result<DecisionTree, TreeServingConversionError> {
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            nodes.push(match node {
                DecisionTreeNode64::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => DecisionTreeNode::Split {
                    feature_index: *feature_index,
                    threshold: narrow(*threshold, policy, index, TreeServingField::Threshold)?,
                    left: *left,
                    right: *right,
                },
                DecisionTreeNode64::Leaf { value } => DecisionTreeNode::Leaf {
                    value: narrow(*value, policy, index, TreeServingField::Leaf)?,
                },
            });
        }
        Ok(DecisionTree::from_validated_parts(
            self.feature_dimension,
            nodes,
        ))
    }
}

/// Fitted Profile C result.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeFitResult32 {
    feature_dimension: usize,
    nodes: Vec<DecisionTreeNode>,
    diagnostics: TreeFitDiagnostics32,
}

impl TreeFitResult32 {
    /// Fitted nodes in pre-order.
    pub fn nodes(&self) -> &[DecisionTreeNode] {
        &self.nodes
    }

    /// Feature dimension.
    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    /// Fit diagnostics.
    pub fn diagnostics(&self) -> &TreeFitDiagnostics32 {
        &self.diagnostics
    }

    /// Builds an independent f32 serving model.
    pub fn predictor(&self) -> DecisionTree {
        DecisionTree::from_validated_parts(self.feature_dimension, self.nodes.clone())
    }

    /// Moves the fitted nodes into the existing f32 serving model.
    pub fn into_predictor(self) -> DecisionTree {
        DecisionTree::from_validated_parts(self.feature_dimension, self.nodes)
    }
}

fn narrow(
    value: f64,
    policy: TreeUnderflowPolicy,
    node: usize,
    field: TreeServingField,
) -> Result<f32, TreeServingConversionError> {
    if !value.is_finite() {
        return Err(TreeServingConversionError::SourceNonFinite { node, field });
    }
    if value.abs() > f32::MAX as f64 {
        return Err(TreeServingConversionError::Overflow { node, field });
    }
    let narrowed = value as f32;
    if narrowed.is_infinite() {
        return Err(TreeServingConversionError::Overflow { node, field });
    }
    if narrowed == 0.0 && value != 0.0 {
        return match policy {
            TreeUnderflowPolicy::Reject => {
                Err(TreeServingConversionError::Underflow { node, field })
            }
            TreeUnderflowPolicy::AllowLoss => Ok(if value.is_sign_negative() { -0.0 } else { 0.0 }),
        };
    }
    Ok(narrowed)
}

// ---------------------------------------------------------------------------
// Private scalar/feature machinery
// ---------------------------------------------------------------------------

trait Scalar: Copy + PartialOrd + PartialEq + fmt::Debug + Default + Send + Sync + 'static {
    const UNIT_ROUNDOFF: Self;
    fn zero() -> Self;
    fn from_usize(value: usize) -> Self;
    fn is_finite(self) -> bool;
    fn bits(self) -> u64;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn abs(self) -> Self;
    fn neg(self) -> Self;
}

macro_rules! impl_scalar {
    ($ty:ty, $u:expr) => {
        impl Scalar for $ty {
            const UNIT_ROUNDOFF: Self = $u;
            fn zero() -> Self {
                0.0
            }
            fn from_usize(value: usize) -> Self {
                value as $ty
            }
            fn is_finite(self) -> bool {
                <$ty>::is_finite(self)
            }
            fn bits(self) -> u64 {
                self.to_bits() as u64
            }
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            fn div(self, rhs: Self) -> Self {
                self / rhs
            }
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            fn neg(self) -> Self {
                -self
            }
        }
    };
}

impl_scalar!(f32, f32::from_bits(0x3380_0000)); // 2^-24
impl_scalar!(f64, f64::from_bits(0x3ca0_0000_0000_0000)); // 2^-53

trait FeatureRow<T: Scalar> {
    fn width(&self) -> usize;
    fn feature(&self, index: usize) -> T;
}

impl FeatureRow<f32> for Vector {
    fn width(&self) -> usize {
        self.len()
    }
    fn feature(&self, index: usize) -> f32 {
        self.as_slice()[index]
    }
}

impl FeatureRow<f64> for Vector64 {
    fn width(&self) -> usize {
        self.len()
    }
    fn feature(&self, index: usize) -> f64 {
        self.as_slice()[index]
    }
}

#[derive(Clone, Copy)]
struct CompSum<T: Scalar> {
    sum: T,
    comp: T,
}

impl<T: Scalar> CompSum<T> {
    fn new() -> Self {
        Self {
            sum: T::zero(),
            comp: T::zero(),
        }
    }

    fn add(&mut self, value: T) {
        let total = self.sum.add(value);
        if self.sum.abs() >= value.abs() {
            self.comp = self.comp.add(self.sum.sub(total).add(value));
        } else {
            self.comp = self.comp.add(value.sub(total).add(self.sum));
        }
        self.sum = total;
    }

    fn corrected(&self) -> T {
        self.sum.add(self.comp)
    }
}

trait ServingNode<T: Scalar>: Clone + fmt::Debug {
    fn split_node(feature_index: usize, threshold: T, left: usize, right: usize) -> Self;
    fn leaf_node(value: T) -> Self;
    fn set_child(&mut self, side: u8, child: usize) -> bool;
    fn is_leaf(&self) -> bool;
}

impl ServingNode<f32> for DecisionTreeNode {
    fn split_node(feature_index: usize, threshold: f32, left: usize, right: usize) -> Self {
        DecisionTreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        }
    }

    fn leaf_node(value: f32) -> Self {
        DecisionTreeNode::Leaf { value }
    }

    fn set_child(&mut self, side: u8, child: usize) -> bool {
        match self {
            DecisionTreeNode::Split { left, right, .. } => {
                if side == SIDE_LEFT {
                    *left = child;
                } else {
                    *right = child;
                }
                true
            }
            DecisionTreeNode::Leaf { .. } => false,
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, DecisionTreeNode::Leaf { .. })
    }
}

impl ServingNode<f64> for DecisionTreeNode64 {
    fn split_node(feature_index: usize, threshold: f64, left: usize, right: usize) -> Self {
        DecisionTreeNode64::Split {
            feature_index,
            threshold,
            left,
            right,
        }
    }

    fn leaf_node(value: f64) -> Self {
        DecisionTreeNode64::Leaf { value }
    }

    fn set_child(&mut self, side: u8, child: usize) -> bool {
        match self {
            DecisionTreeNode64::Split { left, right, .. } => {
                if side == SIDE_LEFT {
                    *left = child;
                } else {
                    *right = child;
                }
                true
            }
            DecisionTreeNode64::Leaf { .. } => false,
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, DecisionTreeNode64::Leaf { .. })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PendingBuild {
    rows_start: usize,
    rows_len: usize,
    parent_index: usize,
    depth: u32,
    parent_side: u8,
}

const NO_PARENT: usize = usize::MAX;
const SIDE_LEFT: u8 = 0;
const SIDE_RIGHT: u8 = 1;

#[derive(Debug)]
struct Diagnostics<T: Scalar> {
    rows: usize,
    features: usize,
    leaves: usize,
    depth: usize,
    training_sse: T,
    requested_workspace_bytes: usize,
    split_evaluations: u64,
    computed_non_positive_gain_leaves: usize,
    computed_positive_gain_below_confidence_leaves: usize,
    computed_positive_gain_candidates_below_confidence: usize,
}

#[derive(Debug)]
struct FitPlan<T: Scalar, N: ServingNode<T>> {
    feature_dimension: usize,
    nodes: Vec<N>,
    diagnostics: Diagnostics<T>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CandidateTrace<T: Scalar> {
    node: usize,
    feature: usize,
    threshold: T,
    valid: bool,
    node_rows: Vec<usize>,
    parent_sse: T,
    b_parent: T,
    left_sse: Option<T>,
    right_sse: Option<T>,
    total: Option<T>,
    gain: Option<T>,
    b_left: Option<T>,
    b_right: Option<T>,
    b_gain: Option<T>,
    lo: Option<T>,
    hi: Option<T>,
    selected: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LeafAudit<T: Scalar> {
    node: usize,
    rows: Vec<usize>,
    production_positive: usize,
    stop: LeafStop,
    _marker: std::marker::PhantomData<T>,
}

struct Best<T: Scalar> {
    feature: usize,
    threshold: T,
    lo: T,
    trace_offset: usize,
}

struct NodeDecision<T: Scalar> {
    split: Option<(usize, T)>,
    leaf_value: T,
    parent_sse: T,
    stop: LeafStop,
    positive_candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafStop {
    Limit,
    Pure,
    NonPositive,
    BelowConfidence,
}

fn fit_core<T, R, N>(
    features: &[R],
    targets: &[T],
    selected: &[usize],
    config: &TreeFitConfig,
    trace: Option<&mut Vec<CandidateTrace<T>>>,
) -> Result<FitPlan<T, N>, TreeFitError>
where
    T: Scalar,
    R: FeatureRow<T>,
    N: ServingNode<T>,
{
    fit_core_with_audit(features, targets, selected, config, trace, None)
}

#[allow(clippy::too_many_arguments)]
fn fit_core_with_audit<T, R, N>(
    features: &[R],
    targets: &[T],
    selected: &[usize],
    config: &TreeFitConfig,
    mut trace: Option<&mut Vec<CandidateTrace<T>>>,
    mut leaf_audit: Option<&mut Vec<LeafAudit<T>>>,
) -> Result<FitPlan<T, N>, TreeFitError>
where
    T: Scalar,
    R: FeatureRow<T>,
    N: ServingNode<T>,
{
    if features.len() != targets.len() {
        return Err(TreeFitError::RowTargetMismatch {
            features: features.len(),
            targets: targets.len(),
        });
    }
    if selected.is_empty() {
        return Err(TreeFitError::TooFewRows);
    }
    for (position, &row) in selected.iter().enumerate() {
        if row >= features.len() {
            return Err(TreeFitError::IndexOutOfRange {
                position,
                row,
                len: features.len(),
            });
        }
    }
    let feature_dimension = features[selected[0]].width();
    if feature_dimension == 0 {
        return Err(TreeFitError::ZeroFeatures);
    }
    for &row in selected {
        let actual = features[row].width();
        if actual != feature_dimension {
            return Err(TreeFitError::RaggedFeatures {
                row,
                expected: feature_dimension,
                actual,
            });
        }
    }
    if config.min_samples_leaf == 0 || config.max_leaves == 0 || config.max_split_evaluations == 0 {
        return Err(TreeFitError::InvalidConfig);
    }

    let n = selected.len();
    let max_nodes = config
        .max_leaves
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(TreeFitError::SizeOverflow)?;
    let pending_capacity = config.max_depth.saturating_add(1).min(max_nodes);

    let s = n
        .checked_mul(size_of::<usize>())
        .ok_or(TreeFitError::SizeOverflow)?;
    let p = s;
    let g = n
        .checked_mul(size_of::<(T, usize)>())
        .ok_or(TreeFitError::SizeOverflow)?;
    let m = n
        .checked_mul(4)
        .and_then(|value| value.checked_add(2))
        .and_then(|value| value.checked_mul(size_of::<T>()))
        .ok_or(TreeFitError::SizeOverflow)?;
    let r = 4usize
        .checked_mul(size_of::<T>())
        .ok_or(TreeFitError::SizeOverflow)?;
    let o = max_nodes
        .checked_mul(size_of::<N>())
        .ok_or(TreeFitError::SizeOverflow)?;
    let q = pending_capacity
        .checked_mul(size_of::<PendingBuild>())
        .ok_or(TreeFitError::SizeOverflow)?;
    let requested = [s, p, g, m, r, o, q]
        .into_iter()
        .try_fold(0usize, |acc, value| acc.checked_add(value))
        .ok_or(TreeFitError::SizeOverflow)?;
    if requested > config.max_workspace_bytes {
        return Err(TreeFitError::Capacity {
            requested,
            limit: config.max_workspace_bytes,
        });
    }

    let mut rows = Vec::new();
    rows.try_reserve_exact(n)
        .map_err(|_| TreeFitError::AllocationFailed)?;
    rows.extend_from_slice(selected);
    rows.sort_unstable();

    let mut partition = Vec::new();
    partition
        .try_reserve_exact(n)
        .map_err(|_| TreeFitError::AllocationFailed)?;

    let mut gather = Vec::new();
    gather
        .try_reserve_exact(n)
        .map_err(|_| TreeFitError::AllocationFailed)?;

    let mut pre_d = Vec::new();
    pre_d
        .try_reserve_exact(n)
        .map_err(|_| TreeFitError::AllocationFailed)?;
    let mut pre_q = Vec::new();
    pre_q
        .try_reserve_exact(n)
        .map_err(|_| TreeFitError::AllocationFailed)?;
    let mut suf_d = Vec::new();
    suf_d
        .try_reserve_exact(n + 1)
        .map_err(|_| TreeFitError::AllocationFailed)?;
    let mut suf_q = Vec::new();
    suf_q
        .try_reserve_exact(n + 1)
        .map_err(|_| TreeFitError::AllocationFailed)?;

    let mut nodes: Vec<N> = Vec::new();
    nodes
        .try_reserve_exact(max_nodes)
        .map_err(|_| TreeFitError::AllocationFailed)?;
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(pending_capacity)
        .map_err(|_| TreeFitError::AllocationFailed)?;
    for &row in selected {
        if !targets[row].is_finite()
            || (0..feature_dimension).any(|index| !features[row].feature(index).is_finite())
        {
            return Err(TreeFitError::NonFiniteTrainingValue { row });
        }
    }

    pending.push(PendingBuild {
        rows_start: 0,
        rows_len: n,
        parent_index: NO_PARENT,
        depth: 0,
        parent_side: SIDE_LEFT,
    });

    let mut leaves = 1usize;
    let mut work = 0u64;
    let mut max_depth = 0usize;
    let mut training_sse = T::zero();
    let mut non_positive_leaves = 0usize;
    let mut below_confidence_leaves = 0usize;
    let mut below_confidence_candidates = 0usize;
    let mut root_recorded = false;

    while let Some(task) = pending.pop() {
        let node_index = nodes.len();
        if node_index >= max_nodes {
            return Err(TreeFitError::Capacity {
                requested: node_index + 1,
                limit: max_nodes,
            });
        }
        let start = task.rows_start;
        let end = start + task.rows_len;
        let may_split = (task.depth as usize) < config.max_depth && leaves < config.max_leaves;
        let decision = evaluate_node(
            features,
            targets,
            &rows[start..end],
            feature_dimension,
            config,
            may_split,
            node_index,
            &mut work,
            &mut gather,
            &mut pre_d,
            &mut pre_q,
            &mut suf_d,
            &mut suf_q,
            &mut trace,
        )?;
        if !root_recorded {
            training_sse = decision.parent_sse;
            root_recorded = true;
        }
        max_depth = max_depth.max(task.depth as usize);

        let appended = match decision.split {
            Some((feature_index, threshold)) if may_split => {
                if pending.len() + 2 > pending_capacity {
                    return Err(TreeFitError::Capacity {
                        requested: pending.len() + 2,
                        limit: pending_capacity,
                    });
                }
                let mid = partition_rows(
                    features,
                    &mut rows,
                    &mut partition,
                    start,
                    end,
                    feature_index,
                    threshold,
                );
                pending.push(PendingBuild {
                    rows_start: mid,
                    rows_len: end - mid,
                    parent_index: node_index,
                    depth: task.depth + 1,
                    parent_side: SIDE_RIGHT,
                });
                pending.push(PendingBuild {
                    rows_start: start,
                    rows_len: mid - start,
                    parent_index: node_index,
                    depth: task.depth + 1,
                    parent_side: SIDE_LEFT,
                });
                N::split_node(feature_index, threshold, NO_PARENT, NO_PARENT)
            }
            _ => {
                match decision.stop {
                    LeafStop::NonPositive => non_positive_leaves += 1,
                    LeafStop::BelowConfidence => {
                        below_confidence_leaves += 1;
                        below_confidence_candidates += decision.positive_candidates;
                    }
                    LeafStop::Pure | LeafStop::Limit => {}
                }
                if let Some(audit) = leaf_audit.as_deref_mut() {
                    audit.push(LeafAudit {
                        node: node_index,
                        rows: rows[start..end].to_vec(),
                        production_positive: decision.positive_candidates,
                        stop: decision.stop,
                        _marker: std::marker::PhantomData,
                    });
                }
                N::leaf_node(decision.leaf_value)
            }
        };

        if decision.split.is_some() && may_split {
            leaves += 1;
        }
        nodes.push(appended);
        if task.parent_index != NO_PARENT
            && !nodes[task.parent_index].set_child(task.parent_side, node_index)
        {
            return Err(TreeFitError::Topology(
                DecisionTreeError::MalformedTraversal,
            ));
        }
    }

    let leaves = nodes.iter().filter(|node| node.is_leaf()).count();
    Ok(FitPlan {
        feature_dimension,
        nodes,
        diagnostics: Diagnostics {
            rows: n,
            features: feature_dimension,
            leaves,
            depth: max_depth,
            training_sse,
            requested_workspace_bytes: requested,
            split_evaluations: work,
            computed_non_positive_gain_leaves: non_positive_leaves,
            computed_positive_gain_below_confidence_leaves: below_confidence_leaves,
            computed_positive_gain_candidates_below_confidence: below_confidence_candidates,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_node<T: Scalar, R: FeatureRow<T>>(
    features: &[R],
    targets: &[T],
    rows: &[usize],
    feature_dimension: usize,
    config: &TreeFitConfig,
    may_split: bool,
    node_index: usize,
    work: &mut u64,
    gather: &mut Vec<(T, usize)>,
    pre_d: &mut Vec<T>,
    pre_q: &mut Vec<T>,
    suf_d: &mut Vec<T>,
    suf_q: &mut Vec<T>,
    trace: &mut Option<&mut Vec<CandidateTrace<T>>>,
) -> Result<NodeDecision<T>, TreeFitError> {
    let n = rows.len();
    let n_t = T::from_usize(n);
    let c_parent = targets[rows[0]];
    let mut parent_p = CompSum::new();
    let mut parent_q = CompSum::new();
    for &row in rows {
        let d = targets[row].sub(c_parent);
        parent_p.add(d);
        parent_q.add(d.mul(d));
    }
    let p = parent_p.corrected();
    let q = parent_q.corrected();
    let parent_sse = q.sub(p.mul(p).div(n_t));
    let b_parent = T::from_usize(16)
        .mul(T::UNIT_ROUNDOFF)
        .mul(q.add(p.mul(p).div(n_t)));
    let leaf_value = c_parent.add(p.div(n_t));
    if !parent_sse.is_finite() || !b_parent.is_finite() || !leaf_value.is_finite() {
        return Err(TreeFitError::NumericalFailure { node: node_index });
    }

    let pure = rows
        .iter()
        .all(|&row| targets[row].bits() == c_parent.bits());
    if pure {
        return Ok(NodeDecision {
            split: None,
            leaf_value,
            parent_sse,
            stop: LeafStop::Pure,
            positive_candidates: 0,
        });
    }
    if !may_split || n < 2 {
        return Ok(NodeDecision {
            split: None,
            leaf_value,
            parent_sse,
            stop: LeafStop::Limit,
            positive_candidates: 0,
        });
    }
    if parent_sse <= b_parent {
        return Err(TreeFitError::NumericalFailure { node: node_index });
    }

    let trace_start = trace.as_ref().map_or(0, |values| values.len());
    let mut best: Option<Best<T>> = None;
    let mut positive_candidates = 0usize;

    for feature in 0..feature_dimension {
        gather.clear();
        gather.extend(
            rows.iter()
                .map(|&row| (features[row].feature(feature), row)),
        );
        gather.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let c_l = targets[gather[0].1];
        let c_r = targets[gather[n - 1].1];

        pre_d.clear();
        pre_q.clear();
        let mut left_d = CompSum::new();
        let mut left_q = CompSum::new();
        for &(_, row) in gather.iter() {
            let d = targets[row].sub(c_l);
            left_d.add(d);
            left_q.add(d.mul(d));
            pre_d.push(left_d.corrected());
            pre_q.push(left_q.corrected());
        }

        suf_d.clear();
        suf_q.clear();
        suf_d.resize(n + 1, T::zero());
        suf_q.resize(n + 1, T::zero());
        let mut right_d = CompSum::new();
        let mut right_q = CompSum::new();
        for index in (0..n).rev() {
            let d = targets[gather[index].1].sub(c_r);
            right_d.add(d);
            right_q.add(d.mul(d));
            suf_d[index] = right_d.corrected();
            suf_q[index] = right_q.corrected();
        }

        for k in 0..n - 1 {
            if *work == config.max_split_evaluations {
                return Err(TreeFitError::WorkBudgetExceeded {
                    performed: *work + 1,
                    limit: config.max_split_evaluations,
                });
            }
            *work += 1;

            let n_l = k + 1;
            let n_r = n - n_l;
            let valid = gather[k].0 < gather[k + 1].0
                && n_l >= config.min_samples_leaf
                && n_r >= config.min_samples_leaf;
            let threshold = gather[k].0;
            if !valid {
                if let Some(sink) = trace.as_deref_mut() {
                    sink.push(CandidateTrace {
                        node: node_index,
                        feature,
                        threshold,
                        valid: false,
                        node_rows: rows.to_vec(),
                        parent_sse,
                        b_parent,
                        left_sse: None,
                        right_sse: None,
                        total: None,
                        gain: None,
                        b_left: None,
                        b_right: None,
                        b_gain: None,
                        lo: None,
                        hi: None,
                        selected: false,
                    });
                }
                continue;
            }

            let n_l_t = T::from_usize(n_l);
            let n_r_t = T::from_usize(n_r);
            let left_d_k = pre_d[k];
            let left_q_k = pre_q[k];
            let right_d_k = suf_d[k + 1];
            let right_q_k = suf_q[k + 1];
            let left_sse = left_q_k.sub(left_d_k.mul(left_d_k).div(n_l_t));
            let right_sse = right_q_k.sub(right_d_k.mul(right_d_k).div(n_r_t));
            let total = left_sse.add(right_sse);
            let b_candidate = T::from_usize(16).mul(T::UNIT_ROUNDOFF).mul(
                left_q_k
                    .add(left_d_k.mul(left_d_k).div(n_l_t))
                    .add(right_q_k)
                    .add(right_d_k.mul(right_d_k).div(n_r_t)),
            );
            let gain = parent_sse.sub(total);
            let b_gain = b_parent.add(b_candidate);
            let lo = gain.sub(b_gain);
            let hi = gain.add(b_gain);
            if !left_sse.is_finite()
                || !right_sse.is_finite()
                || !gain.is_finite()
                || !b_gain.is_finite()
            {
                return Err(TreeFitError::NumericalFailure { node: node_index });
            }
            let b_left = T::from_usize(16)
                .mul(T::UNIT_ROUNDOFF)
                .mul(left_q_k.add(left_d_k.mul(left_d_k).div(n_l_t)));
            let b_right = T::from_usize(16)
                .mul(T::UNIT_ROUNDOFF)
                .mul(right_q_k.add(right_d_k.mul(right_d_k).div(n_r_t)));
            if left_sse < b_left.neg() || right_sse < b_right.neg() {
                return Err(TreeFitError::NumericalFailure { node: node_index });
            }
            if gain > T::zero() {
                positive_candidates += 1;
            }

            let offset = trace
                .as_ref()
                .map_or(0, |values| values.len() - trace_start);
            if let Some(sink) = trace.as_deref_mut() {
                sink.push(CandidateTrace {
                    node: node_index,
                    feature,
                    threshold,
                    valid: true,
                    node_rows: rows.to_vec(),
                    parent_sse,
                    b_parent,
                    left_sse: Some(left_sse),
                    right_sse: Some(right_sse),
                    total: Some(total),
                    gain: Some(gain),
                    b_left: Some(b_left),
                    b_right: Some(b_right),
                    b_gain: Some(b_gain),
                    lo: Some(lo),
                    hi: Some(hi),
                    selected: false,
                });
            }

            if lo > T::zero()
                && best
                    .as_ref()
                    .is_none_or(|current| better(feature, threshold, lo, current))
            {
                best = Some(Best {
                    feature,
                    threshold,
                    lo,
                    trace_offset: offset,
                });
            }
        }
    }

    if let Some(best) = best {
        if let Some(sink) = trace.as_deref_mut() {
            sink[trace_start + best.trace_offset].selected = true;
        }
        Ok(NodeDecision {
            split: Some((best.feature, best.threshold)),
            leaf_value,
            parent_sse,
            stop: LeafStop::Limit,
            positive_candidates,
        })
    } else {
        Ok(NodeDecision {
            split: None,
            leaf_value,
            parent_sse,
            stop: if positive_candidates > 0 {
                LeafStop::BelowConfidence
            } else {
                LeafStop::NonPositive
            },
            positive_candidates,
        })
    }
}

fn better<T: Scalar>(feature: usize, threshold: T, lo: T, current: &Best<T>) -> bool {
    lo > current.lo
        || (lo == current.lo
            && (feature < current.feature
                || (feature == current.feature && threshold < current.threshold)))
}

fn partition_rows<T: Scalar, R: FeatureRow<T>>(
    features: &[R],
    rows: &mut [usize],
    scratch: &mut Vec<usize>,
    start: usize,
    end: usize,
    feature_index: usize,
    threshold: T,
) -> usize {
    let mut left_count = 0usize;
    for &row in &rows[start..end] {
        if features[row].feature(feature_index) <= threshold {
            left_count += 1;
        }
    }
    let mid = start + left_count;
    scratch.clear();
    scratch.resize(end - start, 0usize);
    let mut left = 0usize;
    let mut right = left_count;
    for &row in &rows[start..end] {
        if features[row].feature(feature_index) <= threshold {
            scratch[left] = row;
            left += 1;
        } else {
            scratch[right] = row;
            right += 1;
        }
    }
    rows[start..end].copy_from_slice(scratch);
    mid
}
