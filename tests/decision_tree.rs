use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::trees::decision_tree::{
    DecisionTree, DecisionTreeError, DecisionTreeNode, TreeFitConfig, TreeFitError,
    TreeServingConversionError, TreeUnderflowPolicy, fit_tree_f32, fit_tree_f64,
};
use seqvex::preprocessing::split::{
    TimeCoverage, TimeFoldOutcome, TimeRow, TimeWindowSpec, expanding_time,
};

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
fn dimension_mismatch_display_reports_feature_counts() {
    let error = DecisionTreeError::DimensionMismatch {
        expected: 3,
        actual: 1,
    };
    assert_eq!(error.to_string(), "expected 3 features, got 1");
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

// ---------------------------------------------------------------------------
// Fitted regression tree: public fit/serving/replacement contracts.
// ---------------------------------------------------------------------------

fn vector64(values: &[f64]) -> Vector64 {
    Vector64::from_slice(values)
}

fn features32(values: &[&[f32]]) -> Vec<Vector> {
    values.iter().map(|row| Vector::from_slice(row)).collect()
}

fn features64(values: &[&[f64]]) -> Vec<Vector64> {
    values.iter().map(|row| Vector64::from_slice(row)).collect()
}

fn assert_bits_eq64(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits(), "expected bitwise equality");
}

#[test]
fn profile_a_fits_and_serves_native_f64() {
    let features = features64(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let targets = [0.0, 0.0, 10.0, 10.0];
    let fit = fit_tree_f64(
        &features,
        &targets,
        &[0, 1, 2, 3],
        &TreeFitConfig::default(),
    )
    .expect("f64 fit");
    assert_eq!(fit.diagnostics().leaves, 2);
    assert_eq!(fit.diagnostics().depth, 1);
    let model = fit.into_predictor();
    assert_eq!(model.feature_dimension(), 1);
    assert_bits_eq64(model.predict(&vector64(&[0.5])).unwrap(), 0.0);
    assert_bits_eq64(model.predict(&vector64(&[1.0])).unwrap(), 0.0);
    assert_bits_eq64(model.predict(&vector64(&[1.5])).unwrap(), 10.0);
    let batch = [
        Observation::new(vector64(&[0.0])),
        Observation::new(vector64(&[3.0])),
    ];
    assert_eq!(model.predict_batch(&batch).unwrap(), vec![0.0, 10.0]);
    assert_eq!(
        model.predict(&vector64(&[1.0, 2.0])),
        Err(DecisionTreeError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn profile_c_fits_and_forced_root_leaf_uses_mean() {
    let features = features32(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let targets = [0.0, 0.0, 10.0, 10.0];
    let fit = fit_tree_f32(
        &features,
        &targets,
        &[0, 1, 2, 3],
        &TreeFitConfig::default(),
    )
    .expect("f32 fit");
    let model = fit.into_predictor();
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 0.0);
    assert_bits_eq(model.predict(&vector(&[1.0])).unwrap(), 0.0);
    assert_bits_eq(model.predict(&vector(&[1.5])).unwrap(), 10.0);

    let forced = fit_tree_f32(
        &features,
        &targets,
        &[0, 1, 2, 3],
        &TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
    )
    .expect("forced leaf fit");
    assert_eq!(forced.diagnostics().leaves, 1);
    assert_bits_eq(
        forced.into_predictor().predict(&vector(&[99.0])).unwrap(),
        5.0,
    );
}

#[test]
fn selected_rows_preserve_repeats_and_source_order() {
    let features = features32(&[&[0.0], &[1.0], &[2.0]]);
    let targets = [0.0, 10.0, 20.0];
    let forward = fit_tree_f32(
        &features,
        &targets,
        &[0, 2, 2, 0],
        &TreeFitConfig::default(),
    )
    .expect("repeated fit");
    assert_eq!(forward.diagnostics().rows, 4);
    let reversed = fit_tree_f32(
        &features,
        &targets,
        &[2, 0, 0, 2],
        &TreeFitConfig::default(),
    )
    .expect("permuted fit");
    assert_eq!(forward.nodes(), reversed.nodes());
}

#[test]
fn fit_validation_and_typed_limits() {
    let features = features32(&[&[0.0], &[1.0], &[2.0]]);
    let targets = [0.0, 10.0, 20.0];
    assert!(matches!(
        fit_tree_f32(&features, &targets[..2], &[0], &TreeFitConfig::default()),
        Err(TreeFitError::RowTargetMismatch { .. })
    ));
    assert!(matches!(
        fit_tree_f32(&features, &targets, &[], &TreeFitConfig::default()),
        Err(TreeFitError::TooFewRows)
    ));
    assert!(matches!(
        fit_tree_f32(&features, &targets, &[3], &TreeFitConfig::default()),
        Err(TreeFitError::IndexOutOfRange { .. })
    ));
    assert!(matches!(
        fit_tree_f32(
            &features,
            &targets,
            &[0],
            &TreeFitConfig {
                min_samples_leaf: 0,
                ..TreeFitConfig::default()
            }
        ),
        Err(TreeFitError::InvalidConfig)
    ));
    assert!(matches!(
        fit_tree_f32(
            &features,
            &[0.0, f32::NAN, 20.0],
            &[0, 1, 2],
            &TreeFitConfig::default()
        ),
        Err(TreeFitError::NonFiniteTrainingValue { row: 1 })
    ));
    assert!(matches!(
        fit_tree_f32(
            &features,
            &targets,
            &[0, 1, 2],
            &TreeFitConfig {
                max_workspace_bytes: 1,
                ..TreeFitConfig::default()
            }
        ),
        Err(TreeFitError::Capacity { .. })
    ));
    assert!(matches!(
        fit_tree_f32(
            &features,
            &targets,
            &[0, 1, 2],
            &TreeFitConfig {
                max_split_evaluations: 1,
                ..TreeFitConfig::default()
            }
        ),
        Err(TreeFitError::WorkBudgetExceeded { .. })
    ));
}

#[test]
fn profile_b_route_change_through_public_prediction() {
    let threshold = 1.0_f64 + 2.0_f64.powi(-25);
    let query = 1.0_f64 + 2.0_f64.powi(-24);
    let features = [
        Vector64::from_slice(&[threshold]),
        Vector64::from_slice(&[2.0]),
    ];
    let fit = fit_tree_f64(&features, &[0.0, 10.0], &[0, 1], &TreeFitConfig::default())
        .expect("route-change fit");
    let a = fit.predictor();
    assert_eq!(a.predict(&vector64(&[query])).unwrap(), 10.0);

    let b = fit
        .to_serving_f32(TreeUnderflowPolicy::Reject)
        .expect("checked conversion");
    assert_eq!(b.predict(&vector(&[query as f32])).unwrap(), 0.0);
    assert_eq!(
        b.predict(&vector(&[f32::from_bits(0x3F80_0001)])).unwrap(),
        10.0
    );
    assert_eq!(
        b.predict(&vector(&[f32::from_bits(0x3F7F_FFFF)])).unwrap(),
        0.0
    );
}

#[test]
fn profile_b_failures_preserve_source_and_active_model() {
    let old = two_leaf_tree();
    let old_left = old.predict(&vector(&[-1.0])).unwrap();
    let old_right = old.predict(&vector(&[1.0])).unwrap();

    let overflow = fit_tree_f64(
        &features64(&[&[0.0]]),
        &[1.0e40],
        &[0],
        &TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
    )
    .expect("overflow source fit");
    assert_eq!(
        overflow.to_serving_f32(TreeUnderflowPolicy::Reject),
        Err(TreeServingConversionError::Overflow {
            node: 0,
            field: seqvex::models::trees::decision_tree::TreeServingField::Leaf,
        })
    );
    assert_eq!(
        overflow.predictor().predict(&vector64(&[0.0])).unwrap(),
        1.0e40
    );
    assert_eq!(old.predict(&vector(&[-1.0])).unwrap(), old_left);
    assert_eq!(old.predict(&vector(&[1.0])).unwrap(), old_right);

    let threshold_overflow = fit_tree_f64(
        &features64(&[&[1.0e40], &[2.0e40]]),
        &[0.0, 10.0],
        &[0, 1],
        &TreeFitConfig::default(),
    )
    .expect("threshold-overflow source fit");
    assert_eq!(
        threshold_overflow.to_serving_f32(TreeUnderflowPolicy::Reject),
        Err(TreeServingConversionError::Overflow {
            node: 0,
            field: seqvex::models::trees::decision_tree::TreeServingField::Threshold,
        })
    );

    let underflow = fit_tree_f64(
        &features64(&[&[0.0]]),
        &[1.0e-50],
        &[0],
        &TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
    )
    .expect("underflow source fit");
    assert_eq!(
        underflow.to_serving_f32(TreeUnderflowPolicy::Reject),
        Err(TreeServingConversionError::Underflow {
            node: 0,
            field: seqvex::models::trees::decision_tree::TreeServingField::Leaf,
        })
    );
    let allowed = underflow
        .to_serving_f32(TreeUnderflowPolicy::AllowLoss)
        .expect("allow-loss conversion");
    assert_bits_eq(allowed.predict(&vector(&[0.0])).unwrap(), 0.0);
    assert_eq!(
        underflow.predictor().predict(&vector64(&[0.0])).unwrap(),
        1.0e-50
    );

    let replacement = fit_tree_f64(
        &features64(&[&[0.0]]),
        &[42.0],
        &[0],
        &TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
    )
    .expect("replacement source fit");
    let candidate = replacement
        .to_serving_f32(TreeUnderflowPolicy::Reject)
        .expect("replacement conversion");
    assert_eq!(old.predict(&vector(&[-1.0])).unwrap(), old_left);
    assert_eq!(candidate.predict(&vector(&[0.0])).unwrap(), 42.0);
    assert_eq!(
        replacement.predictor().predict(&vector64(&[0.0])).unwrap(),
        42.0
    );
}

#[test]
fn non_finite_leaf_statistics_are_rejected_for_a_and_c() {
    let features32 = features32(&[&[0.0], &[1.0]]);
    let targets32 = [f32::MAX, -f32::MAX];
    for config in [
        TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
        TreeFitConfig {
            max_leaves: 1,
            ..TreeFitConfig::default()
        },
    ] {
        assert_eq!(
            fit_tree_f32(&features32, &targets32, &[0, 1], &config),
            Err(TreeFitError::NumericalFailure { node: 0 })
        );
    }

    let features64 = features64(&[&[0.0], &[1.0]]);
    let targets64 = [f64::MAX, -f64::MAX];
    for config in [
        TreeFitConfig {
            max_depth: 0,
            ..TreeFitConfig::default()
        },
        TreeFitConfig {
            max_leaves: 1,
            ..TreeFitConfig::default()
        },
    ] {
        assert_eq!(
            fit_tree_f64(&features64, &targets64, &[0, 1], &config),
            Err(TreeFitError::NumericalFailure { node: 0 })
        );
    }
}

fn walk24_fixture() -> (Vec<Vector>, Vec<f32>) {
    let mut features = Vec::with_capacity(24);
    let mut targets = Vec::with_capacity(24);
    for index in 0..24usize {
        let x0 = (index % 6) as f32;
        let x1 = (index / 6) as f32;
        features.push(Vector::from_slice(&[x0, x1]));
        let base = if x0 <= 2.0 {
            if x1 <= 1.0 { 10.0 } else { 30.0 }
        } else if x1 <= 1.0 {
            20.0
        } else {
            40.0
        };
        let delta = match index {
            3 | 10 | 15 | 20 => 1.0,
            5 | 12 | 17 | 22 => -1.0,
            _ => 0.0,
        };
        targets.push(base + delta);
    }
    (features, targets)
}

#[test]
fn walk24_fixed_fold_eligibility_and_held_out_prediction() {
    let rows: Vec<TimeRow> = (0..24usize)
        .map(|index| TimeRow {
            decision_at: index as i64,
            features_available_at: -1,
            label_available_at: if index == 7 { None } else { Some(index as i64) },
        })
        .collect();
    let folds: Vec<_> = expanding_time(
        24,
        &rows,
        TimeCoverage::new(0, 24, 0),
        TimeWindowSpec::new(16, 4, 4, 0, 2),
    )
    .expect("WALK24 schedule")
    .collect();
    assert_eq!(folds.len(), 2);
    let fold = &folds[0];
    assert_eq!(fold.outcome(), TimeFoldOutcome::Usable);
    assert_eq!(fold.test_range(), 16..20);
    assert_eq!(fold.train_ranges(), &[0..7, 8..16]);

    let (features, targets) = walk24_fixture();
    let selected: Vec<usize> = fold
        .train_ranges()
        .iter()
        .flat_map(|range| range.clone())
        .collect();
    assert_eq!(selected.len(), 15);
    assert!(!selected.contains(&7));

    let fit32 = fit_tree_f32(&features, &targets, &selected, &TreeFitConfig::default())
        .expect("WALK24 f32 fit");
    let tree32 = fit32.predictor();
    let fit64 = fit_tree_f64(
        &features
            .iter()
            .map(|row| {
                Vector64::from_slice(&row.iter().map(|value| *value as f64).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>(),
        &targets
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
        &selected,
        &TreeFitConfig::default(),
    )
    .expect("WALK24 f64 fit");
    let tree64 = fit64.predictor();
    let tree_b = fit64
        .to_serving_f32(TreeUnderflowPolicy::Reject)
        .expect("WALK24 B conversion");

    let baseline =
        selected.iter().map(|&row| targets[row] as f64).sum::<f64>() / selected.len() as f64;
    let mut abs32 = 0.0;
    let mut sq32 = 0.0;
    let mut abs64 = 0.0;
    let mut sq64 = 0.0;
    let mut abs_b = 0.0;
    let mut sq_b = 0.0;
    for row in fold.test_range() {
        let actual = targets[row] as f64;
        let p32 = tree32.predict(&features[row]).unwrap() as f64;
        let p64 = tree64
            .predict(&Vector64::from_slice(&[
                features[row].as_slice()[0] as f64,
                features[row].as_slice()[1] as f64,
            ]))
            .unwrap();
        let pb = tree_b.predict(&features[row]).unwrap() as f64;
        abs32 += (p32 - actual).abs();
        sq32 += (p32 - actual) * (p32 - actual);
        abs64 += (p64 - actual).abs();
        sq64 += (p64 - actual) * (p64 - actual);
        abs_b += (pb - actual).abs();
        sq_b += (pb - actual) * (pb - actual);
    }
    let n = fold.test_range().len() as f64;
    println!(
        "WALK24 baseline={baseline:.6} C mae={:.6} rmse={:.6} A mae={:.6} rmse={:.6} B mae={:.6} rmse={:.6}",
        abs32 / n,
        (sq32 / n).sqrt(),
        abs64 / n,
        (sq64 / n).sqrt(),
        abs_b / n,
        (sq_b / n).sqrt(),
    );
    assert!((abs32 / n).is_finite() && (sq32 / n).sqrt().is_finite());
    assert!((abs64 / n).is_finite() && (sq64 / n).sqrt().is_finite());
    assert!((abs_b / n).is_finite() && (sq_b / n).sqrt().is_finite());
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn workload_2k() -> (Vec<Vector>, Vec<f32>, Vec<Vector64>, Vec<f64>) {
    let mut features32 = Vec::with_capacity(2048);
    let mut targets32 = Vec::with_capacity(2048);
    let mut features64 = Vec::with_capacity(2048);
    let mut targets64 = Vec::with_capacity(2048);
    for index in 0..2048usize {
        let mut row = [0.0_f32; 8];
        for (feature, value) in row.iter_mut().enumerate() {
            *value = (((index as u64 * 131 + feature as u64 * 977) % 1009) as f32) / 1008.0;
        }
        let noise = (((splitmix64(index as u64) % 2001) as f32 / 1000.0) - 1.0) * 0.05;
        let target = 10.0 * f32::from(row[0] > 0.5)
            + 5.0 * f32::from(row[1] > 0.25)
            + 2.0 * f32::from(row[2] > 0.75)
            + 1.0 * f32::from(row[3] > 0.5)
            + noise;
        features32.push(Vector::from_slice(&row));
        targets32.push(target);
        features64.push(Vector64::from_slice(
            &row.iter().map(|value| *value as f64).collect::<Vec<_>>(),
        ));
        targets64.push(target as f64);
    }
    (features32, targets32, features64, targets64)
}

#[test]
#[ignore = "pinned 2K design replay; run explicitly"]
fn dt_reg_workload_2k_replay() {
    let selected: Vec<usize> = (0..1536).collect();
    let (features32, targets32, features64, targets64) = workload_2k();
    let a = fit_tree_f64(
        &features64,
        &targets64,
        &selected,
        &TreeFitConfig::default(),
    )
    .expect("2K A fit");
    let c = fit_tree_f32(
        &features32,
        &targets32,
        &selected,
        &TreeFitConfig::default(),
    )
    .expect("2K C fit");
    let a_model = a.predictor();
    let c_model = c.predictor();
    assert_eq!(a.diagnostics().leaves, 521);
    assert_eq!(a.diagnostics().split_evaluations, 167_008);
    assert_eq!(
        a.diagnostics()
            .computed_positive_gain_candidates_below_confidence,
        0
    );
    assert_eq!(c.diagnostics().leaves, 518);
    assert_eq!(c.diagnostics().split_evaluations, 166_976);
    assert_eq!(
        c.diagnostics()
            .computed_positive_gain_below_confidence_leaves,
        3
    );
    assert_eq!(
        c.diagnostics()
            .computed_positive_gain_candidates_below_confidence,
        24
    );
    assert!(a.diagnostics().requested_workspace_bytes <= 180_776);
    assert!(c.diagnostics().requested_workspace_bytes <= 156_176);

    let baseline = selected
        .iter()
        .map(|&row| targets32[row] as f64)
        .sum::<f64>()
        / selected.len() as f64;
    let mut abs_a = 0.0;
    let mut sq_a = 0.0;
    let mut abs_c = 0.0;
    let mut sq_c = 0.0;
    for row in 1536..2048 {
        let actual = targets32[row] as f64;
        let pa = a_model.predict(&features64[row]).unwrap();
        let pc = c_model.predict(&features32[row]).unwrap() as f64;
        abs_a += (pa - actual).abs();
        sq_a += (pa - actual) * (pa - actual);
        abs_c += (pc - actual).abs();
        sq_c += (pc - actual) * (pc - actual);
    }
    let n = 512.0;
    println!(
        "2K A leaves={} boundaries={} mae={:.6} rmse={:.6}; C leaves={} boundaries={} mae={:.6} rmse={:.6}; baseline={baseline:.6}; C production_positive_candidates={} production_affected_leaves={}",
        a.diagnostics().leaves,
        a.diagnostics().split_evaluations,
        abs_a / n,
        (sq_a / n).sqrt(),
        c.diagnostics().leaves,
        c.diagnostics().split_evaluations,
        abs_c / n,
        (sq_c / n).sqrt(),
        c.diagnostics()
            .computed_positive_gain_candidates_below_confidence,
        c.diagnostics()
            .computed_positive_gain_below_confidence_leaves,
    );
    assert!((abs_a / n).is_finite() && (sq_a / n).sqrt().is_finite());
    assert!((abs_c / n).is_finite() && (sq_c / n).sqrt().is_finite());
}
