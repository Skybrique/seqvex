//! K-Nearest Neighbors (KNN) prediction: read-only inference over stored
//! reference observations and targets.
//!
//! This is the KNN model in the neighbors family. It exists to probe
//! observation-backed model state, distance computation,
//! and nearest-neighbor search, not to justify new storage infrastructure.
//!
//! # Mathematical contract
//!
//! With `N` stored references `(r_i, y_i)`, `r_i ∈ R^d`, `y_i ∈ R`, query `x`,
//! and `k` neighbors, the reference uses **squared Euclidean** distance
//!
//! ```text
//! D(x, r_i) = Σ_j (x_j − r_ij)^2
//! ```
//!
//! Neighbors are the first `k` references under the total order
//! `(D(x, r_i), i)` ascending; prediction is the **unweighted mean** of their
//! targets in that order. `sqrt` is strictly increasing on `[0, ∞)`, so ordering
//! by `D` equals ordering by Euclidean distance; squared distance is used because
//! only neighbor *rank* matters for an unweighted mean. A future distance-
//! weighted rule would have to weight by `sqrt(D)`, since `D` and `sqrt(D)` are
//! not interchangeable as magnitudes. Classification, metric configuration, and
//! weighting are deliberately out of scope.
//!
//! # Ownership
//!
//! [`Knn`] owns its immutable `Vec<Reference>` (features and targets),
//! `dimension`, and `k` after construction. Inference takes `&self`, never
//! mutates the model, and needs no reusable scratch or workspace: the `O(N)`
//! candidate buffer is a transient local allocation inside [`Knn::predict`].
//! This is the ownership hypothesis's "immutable parameters shared, scratch
//! local" case, not the GRU's model-owned-workspace topology.
//!
//! # State semantics
//!
//! A query depends only on itself and the immutable references, so prediction is
//! independent across queries and `State` is the **most recent prediction**,
//! matching [`LinearRegression`](crate::models::linear::ols::LinearRegression) and
//! [`DecisionTree`](crate::models::trees::decision_tree::DecisionTree). No temporal state is
//! introduced merely because Seqvex is streaming-first.
//!
//! # Micro-batch
//!
//! Because queries are independent, [`Knn::predict_batch`] is a model-local,
//! order-preserving capability over a caller-bounded slice, not a generic
//! micro-batch executor. The first element failure fails the whole call
//! [`KnnError`] with no partial output, matching the other prediction-only models.
//!
//! # Numerics
//!
//! The `f32` substrate ([`Vector`]) is reused; no `f64` distance accumulation is
//! introduced. Finite references are required at construction
//! ([`KnnError::NonFiniteReference`]); a query is validated for dimension then
//! finiteness ([`KnnError::NonFiniteInput`]); the output is not validated, in
//! line with the other prediction-only slices. Squared distances are non-negative
//! and can overflow to `+inf` for large finite inputs (multiple `+inf` distances
//! then tie and are ordered by reference index); distances can underflow to
//! `0.0` for very small values, which likewise tie deterministically. These are
//! expected IEEE-754 `f32` behaviours, and the caller is responsible for keeping
//! magnitudes representable.
//!
//! [`LinearRegression`]: crate::models::linear::ols::LinearRegression
//! [`DecisionTree`]: crate::models::trees::decision_tree::DecisionTree

use crate::foundation::numerical::Vector;
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

/// One stored reference observation and its target.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    /// Stored feature vector (`dimension` values).
    pub features: Vector,
    /// Stored target value.
    pub target: f32,
}

impl Reference {
    /// Pairs a feature vector with a target.
    pub fn new(features: Vector, target: f32) -> Self {
        Self { features, target }
    }
}

/// Failure classes reported by KNN construction or prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnError {
    /// The configured feature dimension is zero.
    ZeroDimension,
    /// No reference observations were supplied.
    NoReferences,
    /// `k` is zero or exceeds the number of references.
    InvalidK {
        /// The configured neighbor count.
        k: usize,
        /// The number of references.
        references: usize,
    },
    /// A reference feature vector length differs from the configured dimension.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// A reference feature or target is not finite.
    NonFiniteReference {
        /// The offending reference index.
        index: usize,
    },
    /// A query feature is not finite.
    NonFiniteInput,
}

/// An immutable KNN model: stored references plus `k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Knn {
    dimension: usize,
    k: usize,
    references: Vec<Reference>,
}

impl Knn {
    /// Builds a model from a dimension, `k`, and the reference set.
    ///
    /// Validation order is deterministic: zero dimension, empty references,
    /// invalid `k` (`0` or greater than the reference count), then each
    /// reference's dimension and finiteness.
    pub fn new(dimension: usize, k: usize, references: Vec<Reference>) -> Result<Self, KnnError> {
        if dimension == 0 {
            return Err(KnnError::ZeroDimension);
        }
        if references.is_empty() {
            return Err(KnnError::NoReferences);
        }
        if k == 0 || k > references.len() {
            return Err(KnnError::InvalidK {
                k,
                references: references.len(),
            });
        }
        for (index, reference) in references.iter().enumerate() {
            if reference.features.len() != dimension {
                return Err(KnnError::DimensionMismatch {
                    expected: dimension,
                    actual: reference.features.len(),
                });
            }
            if !reference.target.is_finite()
                || !reference
                    .features
                    .as_slice()
                    .iter()
                    .all(|value| value.is_finite())
            {
                return Err(KnnError::NonFiniteReference { index });
            }
        }
        Ok(Self {
            dimension,
            k,
            references,
        })
    }

    /// The number of features each query and reference must contain.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The number of neighbors used for prediction.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The stored reference observations and targets.
    pub fn references(&self) -> &[Reference] {
        &self.references
    }

    /// Reference prediction: mean target of the `k` nearest references.
    ///
    /// The query dimension is validated first, then query finiteness, matching
    /// the other classic slices. Neighbors are selected by
    /// `(squared_distance, reference_index)` ascending and their targets are
    /// averaged in that order, so equal distances and duplicate references are
    /// resolved deterministically. The output is a validated-leaf-style value
    /// and is not re-checked.
    pub fn predict(&self, features: &Vector) -> Result<f32, KnnError> {
        if features.len() != self.dimension {
            return Err(KnnError::DimensionMismatch {
                expected: self.dimension,
                actual: features.len(),
            });
        }
        let query = features.as_slice();
        if !query.iter().all(|value| value.is_finite()) {
            return Err(KnnError::NonFiniteInput);
        }

        let mut candidates: Vec<(f32, usize)> = self
            .references
            .iter()
            .enumerate()
            .map(|(index, reference)| {
                (
                    squared_distance(query, reference.features.as_slice()),
                    index,
                )
            })
            .collect();
        candidates.sort_unstable_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let mut total = 0.0_f32;
        for (_, index) in &candidates[..self.k] {
            total += self.references[*index].target;
        }
        Ok(total / self.k as f32)
    }

    /// Bounded micro-batch prediction over independent queries.
    ///
    /// The batch is exactly the supplied caller-bounded slice and outputs are
    /// returned in input order; the first element failure returns [`KnnError`]
    /// for the whole call with no partial output vector.
    pub fn predict_batch(&self, batch: &[Observation<Vector>]) -> Result<Vec<f32>, KnnError> {
        batch
            .iter()
            .map(|observation| self.predict(observation.value()))
            .collect()
    }
}

impl StateModel for Knn {
    /// The most recent prediction; see the module documentation.
    type State = f32;
    type Observation = Observation<Vector>;
    type Error = KnnError;

    fn update(&self, _state: &f32, observation: &Observation<Vector>) -> Result<f32, KnnError> {
        self.predict(observation.value())
    }
}

/// Squared Euclidean distance over two equal-length slices.
fn squared_distance(query: &[f32], reference: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (query_value, reference_value) in query.iter().zip(reference) {
        let delta = query_value - reference_value;
        sum += delta * delta;
    }
    sum
}
