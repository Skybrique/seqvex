//! Native `f64` serving for a fitted regression tree.

use crate::foundation::numerical::Vector64;
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

use super::DecisionTreeError;

/// A node of a fitted `f64` regression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionTreeNode64 {
    /// An internal split: go `left` when `feature[feature_index] <= threshold`.
    Split {
        /// Index of the feature compared at this node.
        feature_index: usize,
        /// Finite comparison threshold.
        threshold: f64,
        /// Child taken when `feature <= threshold`.
        left: usize,
        /// Child taken when `feature > threshold`.
        right: usize,
    },
    /// A leaf holding the finite regression output.
    Leaf {
        /// The prediction returned when traversal reaches this node.
        value: f64,
    },
}

/// An immutable fitted `f64` regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree64 {
    feature_dimension: usize,
    nodes: Vec<DecisionTreeNode64>,
}

impl DecisionTree64 {
    /// Builds a serving tree from fit-produced, already-validated parts.
    ///
    /// This constructor is crate-private on purpose: external supplied-node
    /// `f64` construction is a separate scope decision. The fitter guarantees
    /// non-empty nodes, a non-zero dimension, finite values, and valid child
    /// indices before calling this.
    pub(crate) fn from_validated_parts(
        feature_dimension: usize,
        nodes: Vec<DecisionTreeNode64>,
    ) -> Self {
        debug_assert!(feature_dimension > 0);
        debug_assert!(!nodes.is_empty());
        debug_assert!(nodes.iter().enumerate().all(|(index, node)| match node {
            DecisionTreeNode64::Leaf { value } => value.is_finite(),
            DecisionTreeNode64::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                threshold.is_finite()
                    && *feature_index < feature_dimension
                    && *left < nodes.len()
                    && *right < nodes.len()
                    && *left != index
                    && *right != index
            }
        }));
        Self {
            feature_dimension,
            nodes,
        }
    }

    /// The number of features each observation must contain.
    pub fn feature_dimension(&self) -> usize {
        self.feature_dimension
    }

    /// The fitted node list, with the root at index 0 in pre-order.
    pub fn nodes(&self) -> &[DecisionTreeNode64] {
        &self.nodes
    }

    /// Reference prediction: iterate from the root to a leaf.
    pub fn predict(&self, features: &Vector64) -> Result<f64, DecisionTreeError> {
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
        for _ in 0..=self.nodes.len() {
            match &self.nodes[node] {
                DecisionTreeNode64::Leaf { value } => return Ok(*value),
                DecisionTreeNode64::Split {
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
    pub fn predict_batch(
        &self,
        batch: &[Observation<Vector64>],
    ) -> Result<Vec<f64>, DecisionTreeError> {
        batch
            .iter()
            .map(|observation| self.predict(observation.value()))
            .collect()
    }
}

impl StateModel for DecisionTree64 {
    type State = f64;
    type Observation = Observation<Vector64>;
    type Error = DecisionTreeError;

    fn update(
        &self,
        _state: &f64,
        observation: &Observation<Vector64>,
    ) -> Result<f64, DecisionTreeError> {
        self.predict(observation.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::numerical::Vector64;

    fn tree() -> DecisionTree64 {
        DecisionTree64::from_validated_parts(
            1,
            vec![
                DecisionTreeNode64::Split {
                    feature_index: 0,
                    threshold: 1.0,
                    left: 1,
                    right: 2,
                },
                DecisionTreeNode64::Leaf { value: 0.0 },
                DecisionTreeNode64::Leaf { value: 10.0 },
            ],
        )
    }

    #[test]
    fn private_construction_and_traversal() {
        let tree = tree();
        assert_eq!(tree.feature_dimension(), 1);
        assert_eq!(tree.nodes().len(), 3);
        assert_eq!(tree.predict(&Vector64::from_slice(&[1.0])).unwrap(), 0.0);
        assert_eq!(tree.predict(&Vector64::from_slice(&[1.5])).unwrap(), 10.0);
    }

    #[test]
    fn prediction_validation_and_batch() {
        let tree = tree();
        assert_eq!(
            tree.predict(&Vector64::from_slice(&[1.0, 2.0])),
            Err(DecisionTreeError::DimensionMismatch {
                expected: 1,
                actual: 2,
            })
        );
        assert_eq!(
            tree.predict(&Vector64::from_slice(&[f64::NAN])),
            Err(DecisionTreeError::NonFiniteInput)
        );
        let batch = [
            Observation::new(Vector64::from_slice(&[0.0])),
            Observation::new(Vector64::from_slice(&[2.0])),
        ];
        assert_eq!(tree.predict_batch(&batch).unwrap(), vec![0.0, 10.0]);
    }

    #[test]
    fn state_model_matches_prediction() {
        use crate::foundation::state::StateModel;
        let tree = tree();
        let state = 0.0;
        assert_eq!(
            tree.update(&state, &Observation::new(Vector64::from_slice(&[2.0])))
                .unwrap(),
            10.0
        );
    }
}
