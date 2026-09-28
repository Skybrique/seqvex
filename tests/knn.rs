use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::classic::knn::{Knn, KnnError, Reference};

fn reference(features: &[f32], target: f32) -> Reference {
    Reference::new(Vector::from_slice(features), target)
}

fn model(dimension: usize, k: usize, references: Vec<Reference>) -> Knn {
    Knn::new(dimension, k, references).unwrap()
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

/// Independent mathematical oracle, derived from the definition and computed in
/// `f64`: squared distances, ordering by `(distance, index)`, first `k`, mean.
/// It does not call `Knn::predict` or any production helper.
fn oracle(references: &[Reference], query: &[f32], k: usize) -> f32 {
    let mut distances: Vec<(f64, usize)> = references
        .iter()
        .enumerate()
        .map(|(index, reference)| {
            let distance: f64 = reference
                .features
                .as_slice()
                .iter()
                .zip(query)
                .map(|(reference_value, query_value)| {
                    let delta = f64::from(*query_value) - f64::from(*reference_value);
                    delta * delta
                })
                .sum();
            (distance, index)
        })
        .collect();
    distances.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap()
            .then_with(|| left.1.cmp(&right.1))
    });
    let total: f64 = distances[..k]
        .iter()
        .map(|(_, index)| f64::from(references[*index].target))
        .sum();
    (total / k as f64) as f32
}

fn assert_close(actual: f32, expected: f32) {
    let tolerance = 1.0e-5 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected} ± {tolerance}, got {actual}"
    );
}

/// Three one-dimensional references at 1.0, 2.0, 3.0 with targets 1.0, 2.0, 3.0.
fn line_references() -> Vec<Reference> {
    vec![
        reference(&[1.0], 1.0),
        reference(&[2.0], 2.0),
        reference(&[3.0], 3.0),
    ]
}

#[test]
fn single_reference_returns_its_target() {
    let model = model(1, 1, vec![reference(&[4.0], 42.0)]);
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 42.0);
    assert_bits_eq(model.predict(&vector(&[99.0])).unwrap(), 42.0);
}

#[test]
fn k_equals_n_returns_mean_of_all_targets() {
    // Query 0.0: distances 1, 4, 9 -> all three neighbors; mean (1+2+3)/3 = 2.
    let model = model(1, 3, line_references());
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 2.0);
}

#[test]
fn k_one_returns_the_nearest_target() {
    // Query 0.0: nearest is 1.0 (target 1.0).
    let model = model(1, 1, line_references());
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 1.0);
    // Query 2.6: nearest is 3.0 (target 3.0).
    assert_bits_eq(model.predict(&vector(&[2.6])).unwrap(), 3.0);
}

#[test]
fn hand_computed_two_dimensional_prediction() {
    // References: [0,0]->1, [1,1]->5, [3,3]->9. Query [0.9, 0.9].
    // D([0,0]) = 0.81+0.81 = 1.62; D([1,1]) = 0.01+0.01 = 0.02;
    // D([3,3]) = 2.1^2 + 2.1^2 = 8.82.
    let model = model(
        2,
        2,
        vec![
            reference(&[0.0, 0.0], 1.0),
            reference(&[1.0, 1.0], 5.0),
            reference(&[3.0, 3.0], 9.0),
        ],
    );
    // k = 1 -> nearest is [1,1] -> 5.
    let k1 = Knn::new(
        2,
        1,
        vec![
            reference(&[0.0, 0.0], 1.0),
            reference(&[1.0, 1.0], 5.0),
            reference(&[3.0, 3.0], 9.0),
        ],
    )
    .unwrap();
    assert_bits_eq(k1.predict(&vector(&[0.9, 0.9])).unwrap(), 5.0);
    // k = 2 -> nearest are [1,1] (5) and [0,0] (1) -> mean 3.
    assert_bits_eq(model.predict(&vector(&[0.9, 0.9])).unwrap(), 3.0);
}

#[test]
fn equal_distance_tie_breaks_by_reference_index() {
    // Query 0.0: +1.0 and -1.0 both at distance 1.0; index 0 wins for k = 1.
    let model = model(
        1,
        1,
        vec![reference(&[1.0], 100.0), reference(&[-1.0], 200.0)],
    );
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 100.0);
}

#[test]
fn more_than_k_share_a_boundary_distance() {
    // Query 0.0: all three at distance 1.0; k = 2 selects indices 0 and 1.
    let model = model(
        1,
        2,
        vec![
            reference(&[1.0], 1.0),
            reference(&[-1.0], 2.0),
            reference(&[1.0], 4.0),
        ],
    );
    assert_bits_eq(model.predict(&vector(&[0.0])).unwrap(), 1.5);
}

#[test]
fn duplicate_references_are_not_collapsed() {
    // Identical features and conflicting targets: both count as neighbors.
    let model = model(1, 2, vec![reference(&[2.0], 10.0), reference(&[2.0], 20.0)]);
    assert_bits_eq(model.predict(&vector(&[2.0])).unwrap(), 15.0);
}

#[test]
fn zero_distance_neighbors_are_selected() {
    let model = model(1, 1, vec![reference(&[1.0], 7.0), reference(&[9.0], 99.0)]);
    assert_bits_eq(model.predict(&vector(&[1.0])).unwrap(), 7.0);
}

#[test]
fn overflowing_distances_tie_by_reference_index() {
    // Both squared distances overflow to +inf, so the index tiebreak decides.
    let model = model(
        1,
        1,
        vec![reference(&[2.0e38], 1.0), reference(&[-1.0e38], 2.0)],
    );
    assert_bits_eq(model.predict(&vector(&[1.0e38])).unwrap(), 1.0);
}

#[test]
fn empty_references_are_rejected() {
    assert_eq!(Knn::new(1, 1, Vec::new()), Err(KnnError::NoReferences));
}

#[test]
fn zero_k_is_rejected() {
    assert_eq!(
        Knn::new(1, 0, line_references()),
        Err(KnnError::InvalidK {
            k: 0,
            references: 3
        })
    );
}

#[test]
fn k_greater_than_reference_count_is_rejected() {
    assert_eq!(
        Knn::new(1, 4, line_references()),
        Err(KnnError::InvalidK {
            k: 4,
            references: 3
        })
    );
}

#[test]
fn zero_dimension_is_rejected() {
    assert_eq!(
        Knn::new(0, 1, vec![reference(&[], 1.0)]),
        Err(KnnError::ZeroDimension)
    );
}

#[test]
fn reference_dimension_mismatch_is_rejected() {
    assert_eq!(
        Knn::new(1, 1, vec![reference(&[1.0, 2.0], 1.0)]),
        Err(KnnError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn non_finite_reference_feature_is_rejected() {
    assert_eq!(
        Knn::new(1, 1, vec![reference(&[f32::NAN], 1.0)]),
        Err(KnnError::NonFiniteReference { index: 0 })
    );
}

#[test]
fn non_finite_reference_target_is_rejected() {
    assert_eq!(
        Knn::new(
            1,
            1,
            vec![reference(&[1.0], 1.0), reference(&[2.0], f32::INFINITY)],
        ),
        Err(KnnError::NonFiniteReference { index: 1 })
    );
}

#[test]
fn query_dimension_mismatch_is_reported() {
    let model = model(2, 1, vec![reference(&[1.0, 1.0], 1.0)]);
    assert_eq!(
        model.predict(&vector(&[1.0])),
        Err(KnnError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        model.predict(&vector(&[1.0, 1.0, 1.0])),
        Err(KnnError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn non_finite_query_is_rejected() {
    let model = model(1, 1, line_references());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            model.predict(&vector(&[value])),
            Err(KnnError::NonFiniteInput),
            "value = {value}"
        );
    }
}

#[test]
fn validation_order_dimension_before_finiteness() {
    let model = model(1, 1, line_references());
    assert_eq!(
        model.predict(&vector(&[f32::NAN, 0.0])),
        Err(KnnError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn repeated_prediction_is_bitwise_stable() {
    let model = model(1, 2, line_references());
    let first = model.predict(&vector(&[1.5])).unwrap();
    for _ in 0..1_000 {
        assert_bits_eq(model.predict(&vector(&[1.5])).unwrap(), first);
    }
}

#[test]
fn state_model_update_matches_predict_and_ignores_incoming_state() {
    let model = model(1, 2, line_references());
    let observation = observation(&[0.4]);
    let expected = model.predict(observation.value()).unwrap();
    for state in [0.0_f32, 1.0e30, -3.5, f32::NAN] {
        assert_bits_eq(model.update(&state, &observation).unwrap(), expected);
    }
}

#[test]
fn streaming_matches_reference_bitwise() {
    let reference_model = model(1, 2, line_references());
    let streaming_model = model(1, 2, line_references());
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    for value in [0.0_f32, 1.6, 2.4, -3.0, 10.0] {
        let observation = observation(&[value]);
        let expected = reference_model.predict(observation.value()).unwrap();
        assert_bits_eq(*executor.process_one(&observation).unwrap(), expected);
    }
}

#[test]
fn micro_batch_matches_repeated_single_prediction() {
    let model = model(1, 2, line_references());
    let batch = vec![
        observation(&[0.0]),
        observation(&[1.6]),
        observation(&[8.0]),
    ];
    let batched = model.predict_batch(&batch).unwrap();
    for (index, observation) in batch.iter().enumerate() {
        assert_bits_eq(batched[index], model.predict(observation.value()).unwrap());
    }
}

#[test]
fn micro_batch_matches_streaming_bitwise() {
    let reference_model = model(1, 2, line_references());
    let batch = vec![
        observation(&[2.9]),
        observation(&[-1.0]),
        observation(&[1.5]),
        observation(&[0.1]),
    ];
    let batched = reference_model.predict_batch(&batch).unwrap();

    let streaming_model = model(1, 2, line_references());
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    for (index, observation) in batch.iter().enumerate() {
        assert_bits_eq(batched[index], *executor.process_one(observation).unwrap());
    }
}

#[test]
fn micro_batch_preserves_observation_order() {
    let model = model(1, 1, line_references());
    let first = observation(&[0.0]);
    let second = observation(&[10.0]);

    let in_order = model
        .predict_batch(&[first.clone(), second.clone()])
        .unwrap();
    let reversed = model.predict_batch(&[second, first]).unwrap();

    assert_bits_eq(in_order[0], 1.0);
    assert_bits_eq(in_order[1], 3.0);
    assert_bits_eq(reversed[0], 3.0);
    assert_bits_eq(reversed[1], 1.0);
}

#[test]
fn empty_micro_batch_returns_no_predictions() {
    let model = model(1, 1, line_references());
    assert_eq!(model.predict_batch(&[]), Ok(Vec::new()));
}

#[test]
fn micro_batch_element_failure_returns_error_without_partial_output() {
    let model = model(1, 1, line_references());
    let batch = vec![
        observation(&[0.0]),
        observation(&[0.0, 1.0]),
        observation(&[0.0]),
    ];
    assert_eq!(
        model.predict_batch(&batch),
        Err(KnnError::DimensionMismatch {
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn failed_streaming_prediction_preserves_last_prediction() {
    let streaming_model = model(1, 2, line_references());
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[0.0])).unwrap();
    assert_eq!(
        executor.process_one(&observation(&[0.0, 1.0])).unwrap_err(),
        KnnError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
    );
    // The last committed prediction is preserved, and a valid query continues.
    assert_bits_eq(*executor.state(), 1.5);
    assert_bits_eq(*executor.process_one(&observation(&[2.0])).unwrap(), 1.5);
}

#[test]
fn reordered_queries_give_the_same_per_query_results() {
    // Non-IID execution semantics: query order changes only the output sequence.
    let model = model(1, 2, line_references());
    let queries = [0.0_f32, 10.0, 1.5, -4.0];

    let forward: Vec<f32> = queries
        .iter()
        .map(|value| model.predict(&vector(&[*value])).unwrap())
        .collect();
    let reversed: Vec<f32> = queries
        .iter()
        .rev()
        .map(|value| model.predict(&vector(&[*value])).unwrap())
        .collect();

    for index in 0..queries.len() {
        assert_bits_eq(forward[index], reversed[queries.len() - 1 - index]);
    }
}

#[test]
fn clustered_queries_do_not_contaminate_each_other() {
    // Two query clusters with a shift between them; each prediction must match
    // its isolated single-query result (the model has no history).
    let model = model(1, 1, line_references());
    let baseline = 1.0_f32;
    let cluster_a = [-100.0_f32, -99.0, -101.0];
    for value in cluster_a {
        assert_bits_eq(model.predict(&vector(&[value])).unwrap(), baseline);
    }
    let cluster_b = [100.0_f32, 101.0, 99.0];
    for value in cluster_b {
        assert_bits_eq(model.predict(&vector(&[value])).unwrap(), 3.0);
    }
    // Re-checking the first cluster is unchanged after the second.
    for value in cluster_a {
        assert_bits_eq(model.predict(&vector(&[value])).unwrap(), baseline);
    }
}

#[test]
fn independent_oracle_grid_matches_production() {
    // Deterministic references over a fixed grid; compare against the
    // independent f64 oracle rather than the production path.
    let references: Vec<Reference> = (0..24)
        .map(|index| {
            let a = (index as f32 * 0.37).sin();
            let b = (index as f32 * 0.11).cos();
            reference(&[a, b], (index as f32 * 0.5) - 4.0)
        })
        .collect();

    for k in [1_usize, 5, 24] {
        let model = Knn::new(2, k, references.clone()).unwrap();
        for step in 0..40 {
            let x = (step as f32 * 0.21) - 4.0;
            let y = (step as f32 * 0.09).cos();
            let query = [x, y];
            let produced = model.predict(&vector(&query)).unwrap();
            assert_close(produced, oracle(&references, &query, k));
        }
    }
}
