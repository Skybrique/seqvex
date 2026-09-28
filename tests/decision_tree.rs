use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::classic::decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode};

fn split(feature_index: usize, threshold: f32, left: usize, right: usize) -> DecisionTreeNode {
    DecisionTreeNode::Split {
        feature_index,
        threshold,
        left,
        right,
    }
}

fn leaf(value: f32) -> DecisionTreeNode {
    DecisionTreeNode::Leaf { value }
}

fn tree(dimension: usize, nodes: Vec<DecisionTreeNode>) -> DecisionTree {
    DecisionTree::new(dimension, nodes).unwrap()
}

fn vector(values: &[f32]) -> Vector {
    Vector::from_slice(values)
}

fn observation(values: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(values))
}

fn assert_bits_eq(left: f32, right: f32) {
    assert_eq!(
        left.to_bits(),
        right.to_bits(),
        "expected bitwise equality: {left} vs {right}"
    );
}

/// Root split on feature 0 at 0.0: left leaf 1.0, right leaf 2.0.
fn two_leaf_tree() -> DecisionTree {
    tree(1, vec![split(0, 0.0, 1, 2), leaf(1.0), leaf(2.0)])
}

#[test]
fn single_leaf_tree_returns_leaf_value() {
    let model = tree(1, vec![leaf(5.0)]);
    assert_bits_eq(model.predict(&vector(&[3.0])).unwrap(), 5.0);
    assert_bits_eq(model.predict(&vector(&[-7.0])).unwrap(), 5.0);
}

#[test]
fn split_routes_left_on_threshold_boundary() {
    let model = two_leaf_tree();
    // feature <= threshold goes left.
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 1.0);
    assert_bits_eq(model.predict(&vector(&[-0.5])).unwrap(), 1.0);
    assert_bits_eq(model.predict(&vector(&[0.5])).unwrap(), 2.0);
}

#[test]
fn deeper_tree_uses_selected_feature_and_hand_computed_value() {
    // 0: feature 0 <= 0.0 ? node 1 : node 2
    // 1: feature 1 <= 0.0 ? leaf 1.0 : leaf 2.0
    // 2: leaf 10.0
    let model = tree(
        2,
        vec![
            split(0, 0.0, 1, 2),
            split(1, 0.0, 3, 4),
            leaf(10.0),
            leaf(1.0),
            leaf(2.0),
        ],
    );

    assert_bits_eq(model.predict(&vector(&[0.5, 0.5])).unwrap(), 10.0);
    assert_bits_eq(model.predict(&vector(&[0.5, -0.5])).unwrap(), 10.0);
    assert_bits_eq(model.predict(&vector(&[-0.5, 0.5])).unwrap(), 2.0);
    assert_bits_eq(model.predict(&vector(&[-0.5, -0.5])).unwrap(), 1.0);
}

#[test]
fn empty_tree_is_rejected() {
    assert_eq!(
        DecisionTree::new(1, Vec::new()),
        Err(DecisionTreeError::EmptyTree)
    );
}

#[test]
fn zero_dimension_is_rejected() {
    assert_eq!(
        DecisionTree::new(0, vec![leaf(1.0)]),
        Err(DecisionTreeError::ZeroDimension)
    );
}

#[test]
fn feature_index_out_of_range_is_rejected() {
    assert_eq!(
        DecisionTree::new(1, vec![split(1, 0.0, 1, 1), leaf(1.0)]),
        Err(DecisionTreeError::FeatureIndexOutOfRange {
            node: 0,
            feature_index: 1,
            dimension: 1,
        })
    );
}

#[test]
fn child_out_of_range_is_rejected() {
    assert_eq!(
        DecisionTree::new(1, vec![split(0, 0.0, 9, 0)]),
        Err(DecisionTreeError::ChildOutOfRange {
            node: 0,
            child: 9,
            len: 1,
        })
    );
}

#[test]
fn non_finite_threshold_is_rejected() {
    assert_eq!(
        DecisionTree::new(1, vec![split(0, f32::NAN, 1, 1), leaf(1.0)]),
        Err(DecisionTreeError::NonFiniteThreshold { node: 0 })
    );
}

#[test]
fn non_finite_leaf_is_rejected() {
    assert_eq!(
        DecisionTree::new(1, vec![leaf(f32::INFINITY)]),
        Err(DecisionTreeError::NonFiniteLeaf { node: 0 })
    );
}

#[test]
fn cycle_is_rejected() {
    // 0 -> 1 and 0 -> 0 (self-loop).
    assert_eq!(
        DecisionTree::new(1, vec![split(0, 0.0, 1, 0), leaf(1.0)]),
        Err(DecisionTreeError::Cycle { node: 0 })
    );
}

#[test]
fn shared_child_is_rejected() {
    // Both branches of node 0 point at node 1.
    assert_eq!(
        DecisionTree::new(1, vec![split(0, 0.0, 1, 1), leaf(1.0)]),
        Err(DecisionTreeError::SharedNode { node: 1 })
    );
}

#[test]
fn unreachable_node_is_rejected() {
    assert_eq!(
        DecisionTree::new(1, vec![leaf(1.0), leaf(2.0)]),
        Err(DecisionTreeError::UnreachableNode { node: 1 })
    );
}

#[test]
fn dimension_mismatch_is_reported_for_short_and_long_inputs() {
    let model = tree(
        2,
        vec![
            split(0, 0.0, 1, 2),
            split(1, 0.0, 3, 4),
            leaf(9.0),
            leaf(1.0),
            leaf(2.0),
        ],
    );
    assert_eq!(
        model.predict(&vector(&[1.0])),
        Err(DecisionTreeError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        model.predict(&vector(&[1.0, 2.0, 3.0])),
        Err(DecisionTreeError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
    assert_eq!(
        model.predict(&vector(&[])),
        Err(DecisionTreeError::DimensionMismatch {
            expected: 2,
            actual: 0,
        })
    );
}

#[test]
fn non_finite_input_is_rejected_for_all_kinds() {
    let model = two_leaf_tree();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            model.predict(&vector(&[value])),
            Err(DecisionTreeError::NonFiniteInput),
            "value = {value}"
        );
    }
}

#[test]
fn validation_order_dimension_before_finiteness() {
    let model = two_leaf_tree();
    assert_eq!(
        model.predict(&vector(&[f32::NAN, 0.0])),
        Err(DecisionTreeError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn repeated_prediction_is_bitwise_stable() {
    let model = two_leaf_tree();
    let first = model.predict(&vector(&[0.25])).unwrap();
    for _ in 0..1_000 {
        assert_bits_eq(model.predict(&vector(&[0.25])).unwrap(), first);
    }
}

#[test]
fn prediction_is_independent_of_previous_observations() {
    let model = two_leaf_tree();
    let baseline = model.predict(&vector(&[0.25])).unwrap();
    for interposed in [-100.0_f32, 0.0, 100.0, f32::NAN] {
        let _ = model.predict(&vector(&[interposed]));
        assert_bits_eq(model.predict(&vector(&[0.25])).unwrap(), baseline);
    }
}

#[test]
fn state_model_update_matches_reference_and_ignores_incoming_state() {
    let model = two_leaf_tree();
    let observation = observation(&[-1.0]);
    let expected = model.predict(observation.value()).unwrap();
    for state in [0.0_f32, 1.0e30, -3.5, f32::NAN] {
        assert_bits_eq(model.update(&state, &observation).unwrap(), expected);
    }
}

#[test]
fn streaming_matches_reference_bitwise() {
    let reference = tree(
        2,
        vec![
            split(0, 0.0, 1, 2),
            split(1, 0.0, 3, 4),
            leaf(10.0),
            leaf(1.0),
            leaf(2.0),
        ],
    );
    let streaming_model = reference.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    for features in [[1.0_f32, 2.0], [-1.0, 0.5], [0.0, 0.0], [10.0, -3.0]] {
        let observation = observation(&features);
        let expected = reference.predict(observation.value()).unwrap();
        let produced = *executor.process_one(&observation).unwrap();
        assert_bits_eq(expected, produced);
    }
}

#[test]
fn micro_batch_matches_repeated_single_observation_prediction() {
    let model = two_leaf_tree();
    let batch = vec![
        observation(&[1.0]),
        observation(&[-1.0]),
        observation(&[0.0]),
    ];

    let batched = model.predict_batch(&batch).unwrap();
    for (index, observation) in batch.iter().enumerate() {
        assert_bits_eq(batched[index], model.predict(observation.value()).unwrap());
    }
}

#[test]
fn micro_batch_matches_streaming_prediction_bitwise() {
    let model = two_leaf_tree();
    let batch = vec![
        observation(&[1.0]),
        observation(&[-3.0]),
        observation(&[0.5]),
        observation(&[-0.25]),
    ];
    let batched = model.predict_batch(&batch).unwrap();

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    for (index, observation) in batch.iter().enumerate() {
        let streamed = *executor.process_one(observation).unwrap();
        assert_bits_eq(batched[index], streamed);
    }
}

#[test]
fn micro_batch_preserves_observation_order() {
    let model = two_leaf_tree();
    let first = observation(&[1.0]);
    let second = observation(&[-1.0]);

    let in_order = model
        .predict_batch(&[first.clone(), second.clone()])
        .unwrap();
    let reversed = model.predict_batch(&[second, first]).unwrap();

    assert_eq!(in_order, vec![2.0, 1.0]);
    assert_eq!(reversed, vec![1.0, 2.0]);
}

#[test]
fn empty_micro_batch_returns_no_predictions() {
    let model = two_leaf_tree();
    assert_eq!(model.predict_batch(&[]), Ok(Vec::new()));
}

#[test]
fn micro_batch_propagates_element_failure() {
    let model = two_leaf_tree();
    let batch = vec![
        observation(&[1.0]),
        observation(&[1.0, 2.0]),
        observation(&[1.0]),
    ];
    assert_eq!(
        model.predict_batch(&batch),
        Err(DecisionTreeError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn failed_streaming_prediction_preserves_last_prediction() {
    let streaming_model = two_leaf_tree();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[-1.0])).unwrap();
    let error = executor.process_one(&observation(&[1.0, 2.0])).unwrap_err();
    assert_eq!(
        error,
        DecisionTreeError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
    );
    assert_bits_eq(*executor.state(), 1.0);
}

#[test]
fn stream_is_a_finite_repeatable_series() {
    let model = tree(
        2,
        vec![
            split(0, 0.0, 1, 2),
            split(1, 0.0, 3, 4),
            leaf(10.0),
            leaf(1.0),
            leaf(2.0),
        ],
    );
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            let value = index as f32 * 0.05 - 6.0;
            vec![value, 0.5 - value]
        })
        .collect();

    let first: Vec<f32> = series
        .iter()
        .map(|features| model.predict(&vector(features)).unwrap())
        .collect();
    let second: Vec<f32> = series
        .iter()
        .map(|features| model.predict(&vector(features)).unwrap())
        .collect();
    for (index, (a, b)) in first.iter().zip(&second).enumerate() {
        assert!(a.is_finite(), "non-finite output at step {index}");
        assert_bits_eq(*a, *b);
    }
}

// ============================================================================
// Durable mathematical evidence for Issue #24 (Decision Tree).
//
// These tests assert explicitly hand-derived expected values and one independent
// closed-form region function. They are deliberately NOT derived from
// `DecisionTree::predict` and do NOT re-implement the production traversal, so a
// shared traversal mistake cannot make expected and actual agree. They complement
// (and do not replace) the internal-consistency, structural-validation, and
// semantic-equivalence tests above.
// ============================================================================

/// `(below, equal, above)` for a finite nonzero `threshold`, using adjacent
/// representable `f32` values, with the strict ordering asserted.
///
/// This exists because decimal literals such as `2.7500001` round back to
/// `2.75` in `f32`; tests must not assume a literal is distinct.
fn representable_boundary(threshold: f32) -> (f32, f32, f32) {
    assert!(
        threshold.is_finite() && threshold != 0.0,
        "boundary helper requires a finite, nonzero threshold, got {threshold}"
    );
    let (below, above) = if threshold > 0.0 {
        (
            f32::from_bits(threshold.to_bits() - 1),
            f32::from_bits(threshold.to_bits() + 1),
        )
    } else {
        (
            f32::from_bits(threshold.to_bits() + 1),
            f32::from_bits(threshold.to_bits() - 1),
        )
    };
    assert!(below < threshold, "{below} must be < {threshold}");
    assert!(above > threshold, "{above} must be > {threshold}");
    (below, threshold, above)
}

#[test]
fn representable_boundary_is_strictly_ordered() {
    for threshold in [2.75_f32, -2.5, 0.5, 1.5, 0.75, 7.25] {
        let (below, equal, above) = representable_boundary(threshold);
        assert!(below < equal && equal < above, "threshold {threshold}");
        assert_eq!(equal.to_bits(), threshold.to_bits());
    }
}

#[test]
fn nonzero_fractional_threshold_sides_and_equality() {
    // Independent mathematical evidence: f(x) = -1.0 for x <= 2.75, 9.0 otherwise.
    let model = tree(1, vec![split(0, 2.75, 1, 2), leaf(-1.0), leaf(9.0)]);
    let (below, equal, above) = representable_boundary(2.75);
    assert_bits_eq(model.predict(&vector(&[below])).unwrap(), -1.0);
    assert_bits_eq(model.predict(&vector(&[equal])).unwrap(), -1.0);
    assert_bits_eq(model.predict(&vector(&[above])).unwrap(), 9.0);
    assert_bits_eq(model.predict(&vector(&[2.0])).unwrap(), -1.0);
    assert_bits_eq(model.predict(&vector(&[3.0])).unwrap(), 9.0);
}

#[test]
fn negative_threshold_sides_and_equality() {
    // Independent mathematical evidence: f(x) = 4.0 for x <= -2.5, 8.0 otherwise.
    let model = tree(1, vec![split(0, -2.5, 1, 2), leaf(4.0), leaf(8.0)]);
    let (below, equal, above) = representable_boundary(-2.5);
    assert_bits_eq(model.predict(&vector(&[below])).unwrap(), 4.0);
    assert_bits_eq(model.predict(&vector(&[equal])).unwrap(), 4.0);
    assert_bits_eq(model.predict(&vector(&[above])).unwrap(), 8.0);
    assert_bits_eq(model.predict(&vector(&[-3.0])).unwrap(), 4.0);
    assert_bits_eq(model.predict(&vector(&[-2.0])).unwrap(), 8.0);
}

#[test]
fn fractional_threshold_sides_and_equality() {
    // Independent mathematical evidence: f(x) = 3.0 for x <= 0.5, 6.0 otherwise.
    let model = tree(1, vec![split(0, 0.5, 1, 2), leaf(3.0), leaf(6.0)]);
    let (below, equal, above) = representable_boundary(0.5);
    assert_bits_eq(model.predict(&vector(&[below])).unwrap(), 3.0);
    assert_bits_eq(model.predict(&vector(&[equal])).unwrap(), 3.0);
    assert_bits_eq(model.predict(&vector(&[above])).unwrap(), 6.0);
    assert_bits_eq(model.predict(&vector(&[0.25])).unwrap(), 3.0);
    assert_bits_eq(model.predict(&vector(&[0.75])).unwrap(), 6.0);
}

/// Three distinct thresholds and three distinct feature indices:
///   0: x0 <= 10.0 ? 1 : 2
///   1: x1 <= -2.5 ? 3 : 4
///   2: x2 <= 7.25 ? 5 : 6
///   3: 100.0   4: 200.0   5: 300.0   6: 400.0
fn node_specific_tree() -> DecisionTree {
    tree(
        3,
        vec![
            split(0, 10.0, 1, 2),
            split(1, -2.5, 3, 4),
            split(2, 7.25, 5, 6),
            leaf(100.0),
            leaf(200.0),
            leaf(300.0),
            leaf(400.0),
        ],
    )
}

#[test]
fn each_node_uses_its_own_threshold_and_feature() {
    // Expected leaves derived by hand from `node_specific_tree`:
    //   [10, 0, 0]   -> root equality -> node1 -> x1=0 > -2.5  -> 200
    //   [0, -2.5, 0] -> node1 equality on feature 1            -> 100
    //   [0, -5, 0]   -> node1 left                             -> 100
    //   [0, 0, 0]    -> node1 right                            -> 200
    //   [20, 0, 0]   -> root right -> node2 -> x2=0 <= 7.25    -> 300
    //   [20, 0, 7.25]-> node2 equality                         -> 300
    //   [20, 0, 8]   -> node2 right                            -> 400
    let model = node_specific_tree();
    assert_bits_eq(model.predict(&vector(&[10.0, 0.0, 0.0])).unwrap(), 200.0);
    assert_bits_eq(model.predict(&vector(&[0.0, -2.5, 0.0])).unwrap(), 100.0);
    assert_bits_eq(model.predict(&vector(&[0.0, -5.0, 0.0])).unwrap(), 100.0);
    assert_bits_eq(model.predict(&vector(&[0.0, 0.0, 0.0])).unwrap(), 200.0);
    assert_bits_eq(model.predict(&vector(&[20.0, 0.0, 0.0])).unwrap(), 300.0);
    assert_bits_eq(model.predict(&vector(&[20.0, 0.0, 7.25])).unwrap(), 300.0);
    assert_bits_eq(model.predict(&vector(&[20.0, 0.0, 8.0])).unwrap(), 400.0);
}

#[test]
fn irrelevant_feature_invariance_and_relevant_feature_sensitivity() {
    let model = node_specific_tree();
    // Path (0, -5, *): x2 is never consulted, so it cannot change the result.
    for x2 in [-1.0e6_f32, 0.0, 1.0e6] {
        assert_bits_eq(model.predict(&vector(&[0.0, -5.0, x2])).unwrap(), 100.0);
    }
    // Path (20, *, 0): x1 is never consulted.
    for x1 in [-1.0e6_f32, 0.0, 1.0e6] {
        assert_bits_eq(model.predict(&vector(&[20.0, x1, 0.0])).unwrap(), 300.0);
    }
    // Guards against a trivially constant model: a relevant feature does matter.
    assert_bits_eq(model.predict(&vector(&[0.0, -5.0, 0.0])).unwrap(), 100.0);
    assert_bits_eq(model.predict(&vector(&[0.0, 5.0, 0.0])).unwrap(), 200.0);
}

#[test]
fn four_region_closed_form_grid_matches_explicit_regions() {
    // The expected function is encoded as explicit regions, independently of any
    // traversal:
    //   x0 <= 0.5 and x1 <= -0.25 -> -1.5
    //   x0 <= 0.5 and x1 >  -0.25 -> 0.125
    //   x0 >  0.5 and x2 <= 2.75  -> 3.5
    //   x0 >  0.5 and x2 >  2.75  -> 7.0
    fn region(x: &[f32; 3]) -> f32 {
        if x[0] <= 0.5 {
            if x[1] <= -0.25 { -1.5 } else { 0.125 }
        } else if x[2] <= 2.75 {
            3.5
        } else {
            7.0
        }
    }
    let model = tree(
        3,
        vec![
            split(0, 0.5, 1, 2),
            split(1, -0.25, 3, 4),
            split(2, 2.75, 5, 6),
            leaf(-1.5),
            leaf(0.125),
            leaf(3.5),
            leaf(7.0),
        ],
    );
    let f0s = [-1.0_f32, 0.0, 0.5, 0.5000001, 1.0];
    let f1s = [-1.0_f32, -0.25, -0.2499999, 0.0];
    let f2s = [-1.0_f32, 2.0, 2.75, 3.0];
    for &f0 in &f0s {
        for &f1 in &f1s {
            for &f2 in &f2s {
                let x = [f0, f1, f2];
                assert_bits_eq(model.predict(&vector(&x)).unwrap(), region(&x));
            }
        }
    }
}

#[test]
fn multi_level_chain_uses_negative_threshold_boundary() {
    // 0: x0 <= 1.5  ? 1 : 2
    // 1: x1 <= 0.75 ? 3 : 4
    // 2: leaf 2.0
    // 3: x0 <= -2.5 ? 5 : 6
    // 4: 4.0   5: 5.0   6: 6.0
    let model = tree(
        2,
        vec![
            split(0, 1.5, 1, 2),
            split(1, 0.75, 3, 4),
            leaf(2.0),
            split(0, -2.5, 5, 6),
            leaf(4.0),
            leaf(5.0),
            leaf(6.0),
        ],
    );
    let (below_root, equal_root, above_root) = representable_boundary(1.5);
    // Equality at the root goes left; x0 is then > -2.5 at node 3.
    assert_bits_eq(model.predict(&vector(&[equal_root, 0.75])).unwrap(), 6.0);
    assert_bits_eq(model.predict(&vector(&[below_root, 0.75])).unwrap(), 6.0);
    // Above the root -> right leaf.
    assert_bits_eq(model.predict(&vector(&[above_root, 0.75])).unwrap(), 2.0);

    let (below_neg, equal_neg, above_neg) = representable_boundary(-2.5);
    assert_bits_eq(model.predict(&vector(&[equal_neg, 0.0])).unwrap(), 5.0);
    assert_bits_eq(model.predict(&vector(&[below_neg, 0.0])).unwrap(), 5.0);
    assert_bits_eq(model.predict(&vector(&[above_neg, 0.0])).unwrap(), 6.0);
}
