//! Decision tree prediction: read-only traversal of a trained tree.
//!
//! This is the Decision Tree model in the trees family. It implements **inference only** on a
//! tree the caller supplies; training/fitting is deliberately out of scope (see
//! `docs/ML_VERTICAL_SLICES.md` and `docs/KILO_CONNECTOME_SPRINT_REVISED.md`
//! §17). The first task variant is regression: each leaf stores a finite `f32`.
//! Classification is deferred until it is explicitly approved.
//!
//! # Representation
//!
//! The tree is a model-local [`Vec`] of [`DecisionTreeNode`] values with child
//! references stored as indices (root = index 0). This is **not** the deferred
//! global arena/storage/allocator decision of `ARCHITECTURE.md` §13: it is one
//! private `Vec` inside one model, the same pattern [`Vector`]/[`Matrix`]
//! already use, and it adds no cross-cutting contract. The layout is
//! **provisional** and may change once evidence justifies it.
//!
//! # Ownership
//!
//! Nodes are immutable once constructed. Prediction needs no mutable model state
//! and no reusable scratch, so this model implements [`StateModel`] with a
//! `&self` contract and is freely shareable. Traversal is iterative with a step
//! bound, so a malformed graph cannot recurse without bound.
//!
//! # State semantics
//!
//! Prediction is stateless: each output depends only on its own observation and
//! the immutable tree. The [`StateModel::State`] carried by the framework is the
//! **most recent prediction**, so the ordered streaming semantics of the
//! execution layer apply unchanged, even though successive predictions do not
//! influence one another. This is the same convention [`LinearRegression`] uses.
//!
//! # Split convention
//!
//! An observation goes **left** when `feature <= threshold`, and **right**
//! otherwise. The threshold is finite and the feature is validated finite before
//! traversal, so the comparison is never made against `NaN`.
//!
//! # Micro-batch
//!
//! Because observations are independent under prediction, a bounded micro-batch
//! is admissible: [`DecisionTree::predict_batch`] returns one prediction per
//! observation in input order, bitwise-equal to repeated single-observation
//! prediction. It is a concrete, model-local capability, not a generic
//! micro-batch executor.
//!
//! # Numerical envelope
//!
//! Feature vectors use the existing `f32` substrate ([`Vector`]). Construction
//! rejects non-finite thresholds ([`DecisionTreeError::NonFiniteThreshold`]),
//! non-finite leaves ([`DecisionTreeError::NonFiniteLeaf`]), an out-of-range
//! feature index, out-of-range children, cycles, shared children, and
//! unreachable nodes. Because a prediction returns a validated leaf value
//! directly, it performs no arithmetic and is always finite; prediction
//! validates the feature dimension first, then feature finiteness
//! ([`DecisionTreeError::NonFiniteInput`]), and does not re-validate the output.
//!
//! [`LinearRegression`]: crate::models::linear::ols::LinearRegression

use crate::foundation::numerical::Vector;
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

/// A node of a trained regression tree.
///
/// Public because the trained tree is supplied by the caller; the representation
/// is provisional and model-local, not a framework-wide storage type.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionTreeNode {
    /// An internal split: go `left` when `feature[feature_index] <= threshold`.
    Split {
        /// Index of the feature compared at this node.
        feature_index: usize,
        /// Finite comparison threshold.
        threshold: f32,
        /// Child taken when `feature <= threshold`.
        left: usize,
        /// Child taken when `feature > threshold`.
        right: usize,
    },
    /// A leaf holding the finite regression output.
    Leaf {
        /// The prediction returned when traversal reaches this node.
        value: f32,
    },
}

/// Failure classes reported by decision-tree construction or prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTreeError {
    /// The node list is empty (no root).
    EmptyTree,
    /// The configured feature dimension is zero.
    ZeroDimension,
    /// A feature index is not below the feature dimension.
    FeatureIndexOutOfRange {
        /// The offending node.
        node: usize,
        /// The configured feature index.
        feature_index: usize,
        /// The model feature dimension.
        dimension: usize,
    },
    /// A child index is not below the node count.
    ChildOutOfRange {
        /// The offending node.
        node: usize,
        /// The configured child index.
        child: usize,
        /// The number of nodes.
        len: usize,
    },
    /// A split threshold is not finite.
    NonFiniteThreshold {
        /// The offending node.
        node: usize,
    },
    /// A leaf value is not finite.
    NonFiniteLeaf {
        /// The offending node.
        node: usize,
    },
    /// A node is its own ancestor; the graph is not a tree.
    Cycle {
        /// A node on the cycle.
        node: usize,
    },
    /// Two parents reference the same node; the graph is not a tree.
    SharedNode {
        /// The node reached more than once.
        node: usize,
    },
    /// A node cannot be reached from the root.
    UnreachableNode {
        /// The first unreachable node.
        node: usize,
    },
    /// The feature vector length differs from the configured feature dimension.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// An observation contains a non-finite feature.
    NonFiniteInput,
    /// Internal invariant: a validated tree failed to terminate at a leaf.
    ///
    /// Unreachable through the validated constructor; retained so the bounded
    /// traversal loop has a total return type instead of a panic.
    MalformedTraversal,
}

#[derive(Clone, Copy)]
enum Visit {
    Unvisited,
    InProgress,
    Done,
}

/// An immutable trained regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree {
    feature_dimension: usize,
    nodes: Vec<DecisionTreeNode>,
}

impl DecisionTree {
    /// Builds a model from a validated node list, with the root at index 0.
    ///
    /// Returns a classified [`DecisionTreeError`] when the graph is not a
    /// usable tree: empty nodes, zero dimension, an out-of-range feature index
    /// or child, a non-finite threshold or leaf, a cycle, a shared child, or a
    /// node unreachable from the root.
    pub fn new(
        feature_dimension: usize,
        nodes: Vec<DecisionTreeNode>,
    ) -> Result<Self, DecisionTreeError> {
        if nodes.is_empty() {
            return Err(DecisionTreeError::EmptyTree);
        }
        if feature_dimension == 0 {
            return Err(DecisionTreeError::ZeroDimension);
        }
        for (node, value) in nodes.iter().enumerate() {
            match value {
                DecisionTreeNode::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    if !threshold.is_finite() {
                        return Err(DecisionTreeError::NonFiniteThreshold { node });
                    }
                    if *feature_index >= feature_dimension {
                        return Err(DecisionTreeError::FeatureIndexOutOfRange {
                            node,
                            feature_index: *feature_index,
                            dimension: feature_dimension,
                        });
                    }
                    if *left >= nodes.len() {
                        return Err(DecisionTreeError::ChildOutOfRange {
                            node,
                            child: *left,
                            len: nodes.len(),
                        });
                    }
                    if *right >= nodes.len() {
                        return Err(DecisionTreeError::ChildOutOfRange {
                            node,
                            child: *right,
                            len: nodes.len(),
                        });
                    }
                }
                DecisionTreeNode::Leaf { value } => {
                    if !value.is_finite() {
                        return Err(DecisionTreeError::NonFiniteLeaf { node });
                    }
                }
            }
        }
        validate_structure(&nodes)?;
        Ok(Self {
            feature_dimension,
            nodes,
        })
    }

    /// The number of features each observation must contain.
    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    /// The node list, with the root at index 0.
    pub fn nodes(&self) -> &[DecisionTreeNode] {
        &self.nodes
    }

    /// Reference prediction: iterate from the root to a leaf.
    ///
    /// The feature dimension is validated first, then feature finiteness,
    /// matching the validation order used by the existing models. The output is
    /// a validated finite leaf value.
    pub fn predict(&self, features: &Vector) -> Result<f32, DecisionTreeError> {
        if features.len() != self.feature_dimension {
            return Err(DecisionTreeError::DimensionMismatch {
                expected: self.feature_dimension,
                actual: features.len(),
            });
        }
        if !features.as_slice().iter().all(|value| value.is_finite()) {
            return Err(DecisionTreeError::NonFiniteInput);
        }

        let slice = features.as_slice();
        let mut node = 0_usize;
        // The constructor guarantees an acyclic tree, so a path reaches a leaf in
        // at most `nodes.len()` steps. The bound keeps a malformed graph from
        // looping, without recursion or unbounded work.
        for _ in 0..=self.nodes.len() {
            match &self.nodes[node] {
                DecisionTreeNode::Leaf { value } => return Ok(*value),
                DecisionTreeNode::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    node = if slice[*feature_index] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
        Err(DecisionTreeError::MalformedTraversal)
    }

    /// Bounded micro-batch prediction over independent observations.
    ///
    /// The batch is exactly the supplied slice (the caller bounds it) and the
    /// outputs are returned in input order. Prediction is order-independent, so
    /// aggregating observations cannot change any individual result; the tests
    /// assert bitwise equality with repeated single-observation prediction.
    ///
    /// ponytail: a straightforward sequential map, not a vectorized kernel.
    /// There is no measured Decision Tree bottleneck justifying SIMD, so none is
    /// added.
    pub fn predict_batch(
        &self,
        batch: &[Observation<Vector>],
    ) -> Result<Vec<f32>, DecisionTreeError> {
        batch
            .iter()
            .map(|observation| self.predict(observation.value()))
            .collect()
    }
}

impl StateModel for DecisionTree {
    /// The most recent prediction; see the module documentation.
    type State = f32;
    type Observation = Observation<Vector>;
    type Error = DecisionTreeError;

    fn update(
        &self,
        _state: &f32,
        observation: &Observation<Vector>,
    ) -> Result<f32, DecisionTreeError> {
        self.predict(observation.value())
    }
}

/// Rejects a graph that is not a single-parent, acyclic tree rooted at index 0.
///
/// An in-progress back-edge is a cycle and a second arrival at a finished node
/// is a shared child; either breaks the traversal invariant. Nodes left
/// unvisited after the walk are unreachable from the root.
fn validate_structure(nodes: &[DecisionTreeNode]) -> Result<(), DecisionTreeError> {
    let mut status = vec![Visit::Unvisited; nodes.len()];
    let mut stack = vec![(0_usize, false)];
    let mut visited = 0_usize;

    while let Some((node, exiting)) = stack.pop() {
        if exiting {
            status[node] = Visit::Done;
            continue;
        }
        match status[node] {
            Visit::InProgress => return Err(DecisionTreeError::Cycle { node }),
            Visit::Done => return Err(DecisionTreeError::SharedNode { node }),
            Visit::Unvisited => {}
        }
        status[node] = Visit::InProgress;
        visited += 1;
        stack.push((node, true));
        if let DecisionTreeNode::Split { left, right, .. } = nodes[node] {
            stack.push((left, false));
            stack.push((right, false));
        }
    }

    if visited == nodes.len() {
        Ok(())
    } else {
        let node = status
            .iter()
            .position(|state| matches!(state, Visit::Unvisited))
            .expect("a node was left unvisited");
        Err(DecisionTreeError::UnreachableNode { node })
    }
}
