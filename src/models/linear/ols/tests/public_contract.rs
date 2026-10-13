use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::classic::linear_regression::{LinearRegression, RegressionError};

fn model(weights: &[f32], bias: f32) -> LinearRegression {
    LinearRegression::new(Vector::from_slice(weights), bias).unwrap()
}

fn observation(features: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(features))
}

fn assert_bits_eq(left: f32, right: f32) {
    assert_eq!(
        left.to_bits(),
        right.to_bits(),
        "expected bitwise equality: {left} vs {right}"
    );
}

#[test]
fn reference_prediction_matches_hand_computed_value() {
    // 1*3 + 2*4 + 0.5 = 11.5, exactly representable in f32.
    let model = model(&[1.0, 2.0], 0.5);
    assert_eq!(model.predict(&Vector::from_slice(&[3.0, 4.0])), Ok(11.5));
}

#[test]
fn zero_weights_are_rejected() {
    assert_eq!(
        LinearRegression::new(Vector::zeros(0), 0.0),
        Err(RegressionError::ZeroDimension)
    );
}

#[test]
fn non_finite_parameter_is_rejected_on_construction() {
    assert_eq!(
        LinearRegression::new(Vector::from_slice(&[1.0, f32::NAN]), 0.0),
        Err(RegressionError::NonFiniteParameter)
    );
    assert_eq!(
        LinearRegression::new(Vector::from_slice(&[1.0]), f32::INFINITY),
        Err(RegressionError::NonFiniteParameter)
    );
}

#[test]
fn dimension_mismatch_is_reported() {
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn non_finite_input_is_rejected() {
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0, f32::NAN])),
        Err(RegressionError::NonFiniteInput)
    );
}

#[test]
fn streaming_prediction_matches_reference() {
    let reference = model(&[0.25, -0.5, 1.5], 0.125);
    let streaming_model = model(&[0.25, -0.5, 1.5], 0.125);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    for features in [
        [1.0_f32, 2.0, 3.0],
        [-1.0, 0.5, 0.25],
        [0.0, 0.0, 0.0],
        [10.0, -3.0, 7.5],
    ] {
        let observation = observation(&features);
        let expected = reference.predict(observation.value()).unwrap();
        let produced = *executor.process_one(&observation).unwrap();
        assert_bits_eq(expected, produced);
    }
}

#[test]
fn streaming_state_is_the_latest_prediction() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[3.0])).unwrap();
    assert_bits_eq(*executor.state(), 7.0);
    executor.process_one(&observation(&[4.0])).unwrap();
    assert_bits_eq(*executor.state(), 9.0);
}

#[test]
fn failed_streaming_prediction_preserves_last_prediction() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[3.0])).unwrap();
    let error = executor.process_one(&observation(&[1.0, 2.0])).unwrap_err();
    assert_eq!(
        error,
        RegressionError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
    );
    assert_bits_eq(*executor.state(), 7.0);
}

#[test]
fn reset_starts_a_new_streaming_sequence() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);

    executor.process_one(&observation(&[3.0])).unwrap();
    assert_bits_eq(*executor.state(), 7.0);
    executor.reset(0.0);
    assert_bits_eq(*executor.state(), 0.0);
}

#[test]
fn state_model_update_matches_reference_prediction() {
    let model = model(&[1.5, -2.0], 0.25);
    let observation = observation(&[4.0, 5.0]);
    let updated = model.update(&0.0, &observation).unwrap();
    assert_bits_eq(updated, model.predict(observation.value()).unwrap());
}

#[test]
fn micro_batch_matches_repeated_single_observation_prediction() {
    let model = model(&[0.5, 1.25, -2.0], -0.75);
    let batch = vec![
        observation(&[1.0, 2.0, 3.0]),
        observation(&[-1.0, 0.5, 0.25]),
        observation(&[0.0, 0.0, 0.0]),
    ];

    let micro_batched = model.predict_batch(&batch).unwrap();
    for (index, observation) in batch.iter().enumerate() {
        assert_bits_eq(
            micro_batched[index],
            model.predict(observation.value()).unwrap(),
        );
    }
}

#[test]
fn micro_batch_matches_streaming_prediction_bitwise() {
    let batch = vec![
        observation(&[1.0, 2.0]),
        observation(&[-3.0, 0.5]),
        observation(&[7.0, -0.25]),
        observation(&[0.125, 8.0]),
    ];

    let model = model(&[1.5, -0.5], 2.0);
    let micro_batched = model.predict_batch(&batch).unwrap();

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    for (index, observation) in batch.iter().enumerate() {
        let streamed = *executor.process_one(observation).unwrap();
        assert_bits_eq(micro_batched[index], streamed);
    }
}

#[test]
fn micro_batch_preserves_observation_order() {
    let model = model(&[2.0, 3.0], 0.0);
    let first = observation(&[1.0, 0.0]);
    let second = observation(&[0.0, 1.0]);

    let in_order = model
        .predict_batch(&[first.clone(), second.clone()])
        .unwrap();
    let reversed = model.predict_batch(&[second, first]).unwrap();

    assert_eq!(in_order, vec![2.0, 3.0]);
    assert_eq!(reversed, vec![3.0, 2.0]);
}

#[test]
fn empty_micro_batch_returns_no_predictions() {
    let model = model(&[1.0], 0.0);
    assert_eq!(model.predict_batch(&[]), Ok(Vec::new()));
}

#[test]
fn micro_batch_propagates_element_failure() {
    let model = model(&[1.0, 1.0], 0.0);
    let batch = vec![
        observation(&[1.0, 1.0]),
        observation(&[1.0]),
        observation(&[1.0, 1.0]),
    ];
    assert_eq!(
        model.predict_batch(&batch),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

// ============================================================================
// Linear Regression correctness / sequential-semantics / numerical audit (#23)
//
// Test-only. This block adds an independent test-side oracle and evidence for
// the audit-matrix properties. No production code is touched: LR has a single
// implementation (`predict`), and `predict_batch` / `StateModel::update`
// delegate to it.
// ============================================================================

/// Independent scalar oracle in `f32`, accumulated in weight-index order.
///
/// Deliberately a plain index loop, not `Vector::dot`: it is a different code
/// path and a different formulation, so agreement catches wiring/omission
/// errors and pins the documented reduction order.
// The explicit scalar index loop is the independent formulation, not an
// iterator rewrite of the production path.
#[allow(clippy::needless_range_loop)]
fn scalar_predict_f32(weights: &[f32], bias: f32, features: &[f32]) -> f32 {
    let mut accumulator = 0.0_f32;
    for index in 0..weights.len() {
        accumulator += weights[index] * features[index];
    }
    accumulator + bias
}

/// Independent `f64` oracle used as a numerical diagnostic, not as the
/// definition of correctness.
#[allow(clippy::needless_range_loop)]
fn scalar_predict_f64(weights: &[f32], bias: f32, features: &[f32]) -> f64 {
    let mut accumulator = 0.0_f64;
    for index in 0..weights.len() {
        accumulator += f64::from(weights[index]) * f64::from(features[index]);
    }
    accumulator + f64::from(bias)
}

/// `Σ |w_i| · |x_i|`, the scale of the `f64` forward-error bound.
fn sum_abs_products(weights: &[f32], features: &[f32]) -> f64 {
    weights
        .iter()
        .zip(features)
        .map(|(&weight, &feature)| f64::from(weight.abs()) * f64::from(feature.abs()))
        .sum()
}

/// Pre-registered absolute tolerance: `8 · (D+1) · f32::EPSILON · scale`.
///
/// `f32::EPSILON = 2u` is 2× the standard unit roundoff; the factor 8 covers the
/// `f64` oracle, the differing reduction precision, and the bias add. Effective
/// envelope ≈ `16·(D+1)·u·scale`.
fn tolerance(weights: &[f32], bias: f32, features: &[f32]) -> f64 {
    let scale = sum_abs_products(weights, features) + f64::from(bias.abs());
    if scale == 0.0 {
        return 0.0;
    }
    8.0 * (weights.len() as f64 + 1.0) * f64::from(f32::EPSILON) * scale
}

fn assert_matches_f64_oracle(model: &LinearRegression, features: &[f32]) {
    let weights = model.weights().as_slice();
    let produced = f64::from(model.predict(&Vector::from_slice(features)).unwrap());
    let oracle = scalar_predict_f64(weights, model.bias(), features);
    let error = (produced - oracle).abs();
    let tolerance = tolerance(weights, model.bias(), features);
    assert!(
        error <= tolerance,
        "f32 {produced} vs f64 oracle {oracle}: |error| {error} > tolerance {tolerance}"
    );
}

fn snapshot_params(model: &LinearRegression) -> (Vec<f32>, f32) {
    (model.weights().as_slice().to_vec(), model.bias())
}

fn assert_params_unchanged(model: &LinearRegression, snapshot: &(Vec<f32>, f32)) {
    assert_eq!(
        model.weights().as_slice(),
        snapshot.0.as_slice(),
        "weights changed"
    );
    assert_bits_eq(model.bias(), snapshot.1);
}

/// Fixed-seed linear congruential generator (mirrors the style of the GRU/RLS
/// deterministic fixtures); deterministic pseudorandom test inputs only.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
    }
}

/// Deterministic bijection on `0..len` (`gcd(7, len) = 1` for `len = 64`).
fn permutation(len: usize) -> Vec<usize> {
    (0..len).map(|index| (index * 7 + 13) % len).collect()
}

fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0_usize; order.len()];
    for (position, &source) in order.iter().enumerate() {
        inverse[source] = position;
    }
    inverse
}

fn deterministic_features(index: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|coordinate| ((index + 1) as f32 * 0.37 + (coordinate + 1) as f32 * 0.11).sin())
        .collect()
}

fn assert_series_matches_oracle(model: &LinearRegression, series: &[Vec<f32>]) {
    for features in series {
        let produced = model.predict(&Vector::from_slice(features)).unwrap();
        let scalar = scalar_predict_f32(model.weights().as_slice(), model.bias(), features);
        assert_bits_eq(produced, scalar);
        assert_matches_f64_oracle(model, features);
    }
}

// --- E.1 mathematical correctness: hand-computable exact cases --------------

#[test]
fn oracle_zero_vector_is_bias_bitwise() {
    let zero_weights = model(&[0.0, 0.0], 0.0);
    let prediction = zero_weights
        .predict(&Vector::from_slice(&[0.0, 0.0]))
        .unwrap();
    assert_bits_eq(prediction, 0.0);

    let with_bias = model(&[1.5, -2.0, 0.25], 0.5);
    let prediction = with_bias
        .predict(&Vector::from_slice(&[0.0, 0.0, 0.0]))
        .unwrap();
    assert_bits_eq(prediction, 0.5);
}

#[test]
fn oracle_one_and_two_dimensional_exact() {
    let one_dimensional = model(&[3.0], 1.0);
    assert_bits_eq(
        one_dimensional
            .predict(&Vector::from_slice(&[2.0]))
            .unwrap(),
        7.0,
    );

    let two_dimensional = model(&[1.0, 2.0], 0.5);
    assert_bits_eq(
        two_dimensional
            .predict(&Vector::from_slice(&[3.0, 4.0]))
            .unwrap(),
        11.5,
    );
}

#[test]
fn oracle_sign_combinations_exact() {
    let negative_coefficients = model(&[-2.0, 3.0], -0.25);
    assert_bits_eq(
        negative_coefficients
            .predict(&Vector::from_slice(&[4.0, -2.0]))
            .unwrap(),
        -14.25,
    );

    let mixed_sign_features = model(&[1.5, -1.5], 2.0);
    assert_bits_eq(
        mixed_sign_features
            .predict(&Vector::from_slice(&[-4.0, 4.0]))
            .unwrap(),
        -10.0,
    );

    let zero_bias = model(&[1.0, 1.0], 0.0);
    assert_bits_eq(
        zero_bias
            .predict(&Vector::from_slice(&[0.5, 0.25]))
            .unwrap(),
        0.75,
    );
}

#[test]
fn oracle_integer_representable_f32_exact() {
    let cases: [(Vec<f32>, f32, Vec<f32>, f32); 5] = [
        (vec![0.25, 4.0], 0.5, vec![8.0, 2.0], 10.5),
        (vec![0.5, 0.5], 0.0, vec![2.0, 4.0], 3.0),
        (vec![8.0, 1024.0], 1.0, vec![0.125, 0.0625], 66.0),
        (vec![16_777_216.0], 0.0, vec![1.0], 16_777_216.0),
        (vec![-0.25, 0.5, 1024.0], 0.5, vec![4.0, -4.0, 0.5], 509.5),
    ];
    for (weights, bias, features, expected) in &cases {
        let model = model(weights, *bias);
        let prediction = model.predict(&Vector::from_slice(features)).unwrap();
        assert_bits_eq(prediction, *expected);
        assert_bits_eq(scalar_predict_f32(weights, *bias, features), *expected);
    }
}

#[test]
fn oracle_exact_cancellation_bitwise() {
    let model = model(&[1.0, 1.0], 1.0);
    let features = [16_777_216.0_f32, -16_777_216.0_f32];
    assert_bits_eq(model.predict(&Vector::from_slice(&features)).unwrap(), 1.0);
}

#[test]
fn reduction_order_is_weight_index_order() {
    // At |1e8| the f32 ulp is 8, so 1e8 + 1.0 rounds back to 1e8. A left fold in
    // weight order therefore gives 0.0, while a reordered sum (pairing the two
    // 1e8-magnitude terms first) would give 1.0. Pinning production to the
    // scalar index-order loop documents the reduction-order contract the
    // micro-batch policy relies on.
    let weights = [1.0e8_f32, 1.0, -1.0e8];
    let features = [1.0_f32, 1.0, 1.0];
    let model = model(&weights, 0.0);
    let produced = model.predict(&Vector::from_slice(&features)).unwrap();
    assert_bits_eq(produced, 0.0);
    assert_bits_eq(produced, scalar_predict_f32(&weights, 0.0, &features));

    let reordered = [1.0e8_f32, -1.0e8, 1.0];
    assert_bits_eq(scalar_predict_f32(&reordered, 0.0, &features), 1.0);
}

// --- E.2 dimension correctness ----------------------------------------------

#[test]
fn predict_rejects_short_and_long_inputs() {
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        model.predict(&Vector::from_slice(&[1.0, 2.0, 3.0])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn predict_rejects_zero_length_input() {
    let model = model(&[1.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[])),
        Err(RegressionError::DimensionMismatch {
            expected: 1,
            actual: 0,
        })
    );
}

#[test]
fn validation_order_dimension_before_finiteness() {
    // Wrong dimension and non-finite simultaneously: the documented order
    // reports the dimension mismatch first.
    let model = model(&[1.0, 2.0], 0.0);
    assert_eq!(
        model.predict(&Vector::from_slice(&[f32::NAN])),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn non_finite_input_rejected_all_kinds() {
    let model = model(&[1.0, 2.0], 0.0);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            model.predict(&Vector::from_slice(&[1.0, value])),
            Err(RegressionError::NonFiniteInput),
            "value = {value}"
        );
    }
}

#[test]
fn matching_dimension_succeeds() {
    for dimension in [1_usize, 2, 4, 8] {
        let weights: Vec<f32> = (0..dimension).map(|index| index as f32 * 0.25).collect();
        let features: Vec<f32> = (0..dimension).map(|index| index as f32 * -0.5).collect();
        let model = model(&weights, 0.125);
        let prediction = model.predict(&Vector::from_slice(&features)).unwrap();
        assert!(prediction.is_finite());
        assert_matches_f64_oracle(&model, &features);
    }
}

// --- E.3 statelessness ------------------------------------------------------

#[test]
fn predict_does_not_mutate_parameters() {
    let model = model(&[0.5, -1.25, 3.0], -0.75);
    let snapshot = snapshot_params(&model);
    let probes = [
        vec![1.0_f32, 2.0, 3.0],
        vec![-1.0, 0.0, 0.5],
        vec![f32::NAN, 0.0, 0.0],
        vec![1.0],
    ];
    for features in probes {
        let _ = model.predict(&Vector::from_slice(&features));
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn repeated_identical_prediction_is_bitwise_stable() {
    let model = model(&[0.25, -0.5, 1.5], 0.125);
    let features = [1.0_f32, -2.0, 0.75];
    let first = model.predict(&Vector::from_slice(&features)).unwrap();
    for _ in 0..1_000 {
        assert_bits_eq(
            model.predict(&Vector::from_slice(&features)).unwrap(),
            first,
        );
    }
}

#[test]
fn prediction_is_independent_of_previous_observations() {
    let model = model(&[0.5, 1.25, -2.0], -0.75);
    let target = [1.0_f32, 2.0, 3.0];
    let baseline = model.predict(&Vector::from_slice(&target)).unwrap();

    for interposed in [
        [0.0_f32, 0.0, 0.0],
        [1.0e30, -1.0e30, 1.0e30],
        [-3.0, 4.0, -5.0],
    ] {
        model.predict(&Vector::from_slice(&interposed)).unwrap();
        assert_bits_eq(
            model.predict(&Vector::from_slice(&target)).unwrap(),
            baseline,
        );
    }
    // A failed call in between leaves the result unchanged too.
    assert!(model.predict(&Vector::from_slice(&[1.0])).is_err());
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );
}

#[test]
fn state_model_update_ignores_incoming_state() {
    // Supporting evidence of state independence: the carried state is not an
    // input to the computation. This is NOT the primary causality test; see
    // prefix_and_suffix_do_not_affect_prediction.
    let model = model(&[1.5, -2.0], 0.25);
    let observation = observation(&[4.0, 5.0]);
    let expected = model.predict(observation.value()).unwrap();
    for state in [0.0_f32, 1.0e30, -3.5, f32::NAN] {
        assert_bits_eq(model.update(&state, &observation).unwrap(), expected);
    }
}

// --- E.4 ordering / permutation ---------------------------------------------

#[test]
fn permutation_does_not_change_per_observation_prediction() {
    let dimension = 4;
    let count = 64;
    let weights = [0.5_f32, -1.25, 2.0, 0.125];
    let model = model(&weights, -0.375);
    let features: Vec<Vec<f32>> = (0..count)
        .map(|index| deterministic_features(index, dimension))
        .collect();

    let in_order: Vec<f32> = features
        .iter()
        .map(|feature| model.predict(&Vector::from_slice(feature)).unwrap())
        .collect();

    let order = permutation(count);
    let permuted: Vec<f32> = order
        .iter()
        .map(|&index| {
            model
                .predict(&Vector::from_slice(&features[index]))
                .unwrap()
        })
        .collect();
    let inverse = inverse_permutation(&order);
    for (index, expected) in in_order.iter().enumerate() {
        // Compare each observation's own prediction across evaluation orders;
        // this is not a position-by-position comparison of the output sequence.
        assert_bits_eq(permuted[inverse[index]], *expected);
    }
}

#[test]
fn predict_batch_permutation_aligns_outputs() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let batch: Vec<Observation<Vector>> = (0..8)
        .map(|index| observation(&deterministic_features(index, 3)))
        .collect();
    let forward = model.predict_batch(&batch).unwrap();

    let order = permutation(8);
    let permuted_batch: Vec<Observation<Vector>> =
        order.iter().map(|&index| batch[index].clone()).collect();
    let permuted = model.predict_batch(&permuted_batch).unwrap();
    for (position, &source) in order.iter().enumerate() {
        assert_bits_eq(permuted[position], forward[source]);
    }
}

// --- E.5 causality / prefix / suffix / context ------------------------------

#[test]
fn prefix_and_suffix_do_not_affect_prediction() {
    // Primary causality evidence: predict(x_t) is unchanged by any prefix,
    // suffix, position, or sequence context.
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let target = deterministic_features(5, 3);
    let baseline = model.predict(&Vector::from_slice(&target)).unwrap();

    let prefix: Vec<Vec<f32>> = (0..17)
        .map(|index| deterministic_features(index, 3))
        .collect();
    for features in &prefix {
        model.predict(&Vector::from_slice(features)).unwrap();
    }
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );

    // A different-distribution prefix (large magnitudes) must not matter either.
    for features in &prefix {
        let scaled: Vec<f32> = features.iter().map(|value| value * 1.0e6).collect();
        model.predict(&Vector::from_slice(&scaled)).unwrap();
    }
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );

    // Inside a longer sequence, with an arbitrary suffix after x_t.
    let mut sequence = Vec::new();
    sequence.extend_from_slice(&prefix[..3]);
    sequence.push(target.clone());
    sequence.extend((0..9).map(|index| deterministic_features(index + 100, 3)));
    for features in &sequence {
        model.predict(&Vector::from_slice(features)).unwrap();
    }
    assert_bits_eq(
        model.predict(&Vector::from_slice(&target)).unwrap(),
        baseline,
    );
    assert_bits_eq(
        scalar_predict_f32(model.weights().as_slice(), model.bias(), &target),
        baseline,
    );
}

#[test]
fn suffix_does_not_affect_committed_stream_state_for_x_t() {
    let prefix = [
        observation(&[0.1, 0.2, 0.3]),
        observation(&[-0.4, 0.5, -0.6]),
    ];
    let target = observation(&[1.0, -2.0, 0.5]);
    let suffix = [
        observation(&[9.0, -8.0, 7.0]),
        observation(&[0.0, 0.0, 0.0]),
    ];
    let reference = model(&[0.5, -1.25, 2.0], -0.375);

    let running_model = reference.clone();
    let mut running = StreamingExecutor::new(&running_model, 0.0);
    for observation in prefix.iter().chain(std::iter::once(&target)) {
        running.process_one(observation).unwrap();
    }
    let committed_for_target = *running.state();
    for observation in &suffix {
        running.process_one(observation).unwrap();
    }

    // The value committed for x_t (snapshotted before any suffix) equals the
    // direct prediction for x_t; the suffix cannot retroactively influence it.
    assert_bits_eq(
        committed_for_target,
        reference.predict(target.value()).unwrap(),
    );
    assert_bits_eq(
        *running.state(),
        reference.predict(suffix.last().unwrap().value()).unwrap(),
    );
}

#[test]
fn sequence_number_context_does_not_change_prediction() {
    use seqvex::foundation::observation::SequenceNumber;

    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let features = [1.0_f32, -2.0, 0.5];
    let plain = model.predict(&Vector::from_slice(&features)).unwrap();
    let with_sequence =
        Observation::new(Vector::from_slice(&features)).with_sequence(SequenceNumber::new(7));
    assert_bits_eq(model.predict(with_sequence.value()).unwrap(), plain);

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    assert_bits_eq(*executor.process_one(&with_sequence).unwrap(), plain);
}

// --- E.6 batch / chunk equivalence ------------------------------------------

#[test]
fn single_observation_batch_equals_predict() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let observation = observation(&[1.0, -2.0, 0.5]);
    let batch = model
        .predict_batch(std::slice::from_ref(&observation))
        .unwrap();
    assert_bits_eq(batch[0], model.predict(observation.value()).unwrap());
}

#[test]
fn arbitrary_chunking_preserves_each_prediction_bitwise() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let observations: Vec<Observation<Vector>> = (0..12)
        .map(|index| observation(&deterministic_features(index, 3)))
        .collect();
    let full = model.predict_batch(&observations).unwrap();

    let chunk_plans: [Vec<usize>; 3] = [
        vec![1, 2, 9],
        vec![6, 6],
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    for plan in &chunk_plans {
        let mut chunked = Vec::new();
        let mut start = 0;
        for &size in plan {
            chunked.extend(
                model
                    .predict_batch(&observations[start..start + size])
                    .unwrap(),
            );
            start += size;
        }
        assert_eq!(chunked.len(), full.len());
        for (index, (chunked, whole)) in chunked.iter().zip(&full).enumerate() {
            // Per-observation comparison: the same observation in a different
            // caller-defined chunk must yield the same prediction.
            assert_bits_eq(*chunked, *whole);
            assert_bits_eq(
                *chunked,
                model.predict(observations[index].value()).unwrap(),
            );
        }
    }
}

#[test]
fn batch_handles_repeated_zero_and_mixed_sign() {
    let model = model(&[0.5, -1.25, 2.0], -0.75);
    let observations = [
        observation(&[1.0, -2.0, 0.5]),
        observation(&[1.0, -2.0, 0.5]),
        observation(&[0.0, 0.0, 0.0]),
        observation(&[-1.0, 2.0, -0.5]),
        observation(&[0.0, 0.0, 0.0]),
    ];
    let batch = model.predict_batch(&observations).unwrap();
    for (index, observation) in observations.iter().enumerate() {
        assert_bits_eq(batch[index], model.predict(observation.value()).unwrap());
    }
}

#[test]
fn batch_element_failure_preserves_model_and_recovers() {
    let model = model(&[1.0, 1.0], -0.5);
    let snapshot = snapshot_params(&model);
    let failing = [
        observation(&[1.0, 1.0]),
        observation(&[1.0]),
        observation(&[1.0, 1.0]),
    ];
    assert_eq!(
        model.predict_batch(&failing),
        Err(RegressionError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_params_unchanged(&model, &snapshot);

    let valid = [observation(&[1.0, 1.0]), observation(&[2.0, -2.0])];
    let outputs = model.predict_batch(&valid).unwrap();
    for (index, observation) in valid.iter().enumerate() {
        assert_bits_eq(outputs[index], model.predict(observation.value()).unwrap());
    }
}

#[test]
fn larger_batch_equivalence() {
    let model = model(&[0.5, -1.25, 2.0, 0.125], -0.375);
    let observations: Vec<Observation<Vector>> = (0..1_000)
        .map(|index| observation(&deterministic_features(index, 4)))
        .collect();
    let batch = model.predict_batch(&observations).unwrap();
    for (index, observation) in observations.iter().enumerate() {
        assert_bits_eq(batch[index], model.predict(observation.value()).unwrap());
    }
}

// --- E.7 non-IID prediction behavior ----------------------------------------

#[test]
fn non_iid_stationary_iid_like() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let mut lcg = Lcg::new(0x5eed_0001);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|_| (0..3).map(|_| lcg.next_f32()).collect())
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_strongly_ordered() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            let value = index as f32 * 0.25;
            vec![value, value * value * 0.01, -value]
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_clustered_regimes() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            if index % 2 == 0 {
                vec![0.5, 0.5, 0.5]
            } else {
                vec![-1.5, 2.5, -3.5]
            }
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_abrupt_regime_shift() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            if index < 128 {
                vec![0.1, -0.2, 0.3]
            } else {
                vec![10.0, -20.0, 30.0]
            }
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_gradual_drift() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            let time = index as f32 * 0.05;
            vec![time, 1.0 - 0.5 * time, -0.25 + 0.1 * time]
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_autocorrelated_ar1() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let mut lcg = Lcg::new(0xa0c0_0001);
    let mut previous = [0.0_f32; 3];
    let mut series = Vec::with_capacity(256);
    for _ in 0..256 {
        let mut current = [0.0_f32; 3];
        for (slot, &value) in current.iter_mut().zip(&previous) {
            *slot = 0.8 * value + 0.1 * lcg.next_f32();
        }
        previous = current;
        series.push(current.to_vec());
    }
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_repeated_observations() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series = vec![vec![1.0_f32, -2.0, 0.5]; 256];
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_heterogeneous_feature_magnitudes() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series = vec![vec![1.0e-3_f32, 1.0, 1.0e3]; 64];
    assert_series_matches_oracle(&model, &series);
}

// --- E.8 long-run determinism -----------------------------------------------

#[test]
fn long_run_repeated_identical_is_stable() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let features = [1.0_f32, -2.0, 0.5];
    let first = model.predict(&Vector::from_slice(&features)).unwrap();
    for step in 0..10_000 {
        let prediction = model.predict(&Vector::from_slice(&features)).unwrap();
        assert!(prediction.is_finite(), "non-finite at step {step}");
        assert_bits_eq(prediction, first);
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn long_run_alternating_and_cyclic_are_stable() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);

    let alternating = [vec![1.0_f32, -2.0, 0.5], vec![-0.5, 0.25, -0.75]];
    let alternating_baseline: Vec<f32> = alternating
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    for step in 0..10_000 {
        let index = step % alternating.len();
        assert_bits_eq(
            model
                .predict(&Vector::from_slice(&alternating[index]))
                .unwrap(),
            alternating_baseline[index],
        );
    }

    let cycle: Vec<Vec<f32>> = (0..7)
        .map(|index| deterministic_features(index, 3))
        .collect();
    let cycle_baseline: Vec<f32> = cycle
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    for step in 0..10_000 {
        let index = step % cycle.len();
        assert_bits_eq(
            model.predict(&Vector::from_slice(&cycle[index])).unwrap(),
            cycle_baseline[index],
        );
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn long_run_pseudorandom_matches_oracle_and_replay() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let mut lcg = Lcg::new(0xfeed_0001);
    let series: Vec<Vec<f32>> = (0..10_000)
        .map(|_| vec![lcg.next_f32(), lcg.next_f32(), lcg.next_f32()])
        .collect();

    let first_pass: Vec<f32> = series
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    let second_pass: Vec<f32> = series
        .iter()
        .map(|features| model.predict(&Vector::from_slice(features)).unwrap())
        .collect();
    for (first, second) in first_pass.iter().zip(&second_pass) {
        assert_bits_eq(*first, *second);
    }
    for (index, features) in series.iter().enumerate() {
        assert!(first_pass[index].is_finite());
        assert_bits_eq(
            first_pass[index],
            scalar_predict_f32(model.weights().as_slice(), model.bias(), features),
        );
        assert_matches_f64_oracle(&model, features);
    }
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn long_run_streaming_state_is_last_prediction_only() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let mut lcg = Lcg::new(0xfeed_0002);
    let series: Vec<Vec<f32>> = (0..10_000)
        .map(|_| vec![lcg.next_f32(), lcg.next_f32(), lcg.next_f32()])
        .collect();

    let streaming_model = model.clone();
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    let mut last = 0.0_f32;
    for features in &series {
        last = *executor.process_one(&observation(features)).unwrap();
        assert_bits_eq(last, model.predict(&Vector::from_slice(features)).unwrap());
    }
    assert_bits_eq(*executor.state(), last);

    // Reset and replay: the identical trajectory shows no hidden accumulation.
    executor.reset(0.0);
    for features in &series {
        assert_bits_eq(
            *executor.process_one(&observation(features)).unwrap(),
            model.predict(&Vector::from_slice(features)).unwrap(),
        );
    }
    assert_params_unchanged(&model, &snapshot);
}

// --- E.9 / E.10 numerical behavior and f32/f64 diagnostic -------------------

fn assert_numerical_case(weights: &[f32], bias: f32, features: &[f32]) {
    let model = model(weights, bias);
    let produced = model.predict(&Vector::from_slice(features)).unwrap();
    assert!(produced.is_finite(), "unexpected non-finite output");
    assert_bits_eq(produced, scalar_predict_f32(weights, bias, features));
    let oracle = scalar_predict_f64(weights, bias, features);
    let error = (f64::from(produced) - oracle).abs();
    let tolerance = tolerance(weights, bias, features);
    assert!(
        error <= tolerance,
        "|error| {error} > tolerance {tolerance} (f32 {produced}, f64 {oracle})"
    );
}

#[test]
fn numerical_small_magnitudes() {
    for scale in [1.0e-6_f32, 1.0e-4, 1.0e-2] {
        let weights = [scale, -2.0 * scale, 0.5 * scale];
        let features = [scale, scale, -scale];
        let bias = 0.25 * scale;
        assert_numerical_case(&weights, bias, &features);
    }
}

#[test]
fn numerical_normal_magnitudes() {
    for scale in [0.1_f32, 1.0, 10.0, 100.0] {
        let weights = [scale, -0.5 * scale, 0.25 * scale];
        let features = [scale, scale, scale];
        let bias = 0.125 * scale;
        assert_numerical_case(&weights, bias, &features);
    }
}

#[test]
fn numerical_large_finite_magnitudes() {
    for scale in [1.0e3_f32, 1.0e4, 1.0e5, 1.0e10] {
        let weights = [scale, -0.5 * scale, 0.25 * scale];
        let features = [scale, scale, scale];
        let bias = 0.125 * scale;
        assert_numerical_case(&weights, bias, &features);
    }
}

#[test]
fn numerical_cancellation_is_expected_rounding() {
    // At |1e8| the f32 ulp is 8, so the residual 1.0 is added to a 1e8 term
    // first and absorbed, and the subsequent exact cancellation leaves 0.0.
    // Reordering the same terms gives 1.0. This is expected f32 rounding, not a
    // defect.
    let model = model(&[1.0, 1.0, 1.0], 0.0);
    let absorbing = [1.0e8_f32, 1.0, -1.0e8];
    assert_bits_eq(model.predict(&Vector::from_slice(&absorbing)).unwrap(), 0.0);

    let reordered = [1.0e8_f32, -1.0e8, 1.0];
    assert_bits_eq(model.predict(&Vector::from_slice(&reordered)).unwrap(), 1.0);
}

#[test]
fn numerical_dynamic_range_reports_boundary() {
    let weights = [0.5_f32, -1.25, 2.0];
    let features = [1.0e-3_f32, 1.0, 1.0e3];
    assert_numerical_case(&weights, -0.375, &features);
}

#[test]
fn finite_input_non_finite_output_is_observed() {
    // OBSERVED BEHAVIOUR (characterization, not a defect assertion): validated
    // finite parameters and finite inputs can overflow f32 and produce a
    // non-finite output. `predict` currently has no output-finiteness guard.
    let product_overflow = model(&[1.0e20], 0.0);
    let result = product_overflow
        .predict(&Vector::from_slice(&[1.0e20]))
        .unwrap();
    assert!(
        result.is_infinite() && result > 0.0,
        "expected +inf, got {result}"
    );

    let summation_overflow = model(&[2.0e38, 2.0e38], 0.0);
    let result = summation_overflow
        .predict(&Vector::from_slice(&[1.0, 1.0]))
        .unwrap();
    assert!(
        result.is_infinite() && result > 0.0,
        "expected +inf, got {result}"
    );

    let nan_from_infinities = model(&[1.0e20, -1.0e20], 0.0);
    let result = nan_from_infinities
        .predict(&Vector::from_slice(&[1.0e20, 1.0e20]))
        .unwrap();
    assert!(result.is_nan(), "expected NaN, got {result}");

    // Boundary bracket: 1e38 is finite, 1e40 is not.
    let below = model(&[1.0e19], 0.0);
    assert!(
        below
            .predict(&Vector::from_slice(&[1.0e19]))
            .unwrap()
            .is_finite()
    );
    let above = model(&[1.0e20], 0.0);
    assert!(
        !above
            .predict(&Vector::from_slice(&[1.0e20]))
            .unwrap()
            .is_finite()
    );
}

#[test]
fn f32_vs_f64_error_envelope() {
    // Diagnostic table for the audit report. The only assertion is the
    // pre-registered tolerance; the printed values are the measured envelope.
    let cases: [(&str, f32, [f32; 3], [f32; 3]); 6] = [
        ("small-1e-6", 1.0e-6, [1.0, -2.0, 0.5], [1.0, 1.0, -1.0]),
        ("small-1e-2", 1.0e-2, [1.0, -2.0, 0.5], [1.0, 1.0, -1.0]),
        ("normal-1", 1.0, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
        ("normal-100", 100.0, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
        ("large-1e5", 1.0e5, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
        ("large-1e10", 1.0e10, [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]),
    ];
    for (label, scale, weight_factors, feature_factors) in cases {
        let weights: Vec<f32> = weight_factors
            .iter()
            .map(|&factor| factor * scale)
            .collect();
        let features: Vec<f32> = feature_factors
            .iter()
            .map(|&factor| factor * scale)
            .collect();
        let bias = 0.125 * scale;
        let model = model(&weights, bias);
        let produced = model.predict(&Vector::from_slice(&features)).unwrap();
        let oracle = scalar_predict_f64(&weights, bias, &features);
        let absolute = (f64::from(produced) - oracle).abs();
        let relative = absolute / oracle.abs().max(1.0e-30);
        let tolerance = tolerance(&weights, bias, &features);
        println!(
            "f32/f64 {label}: scale={scale:e} D={} abs_err={absolute:e} rel_err={relative:e} tol={tolerance:e}",
            weights.len()
        );
        assert!(produced.is_finite(), "{label}: non-finite output");
        assert!(absolute <= tolerance, "{label}: {absolute} > {tolerance}");
    }
}

// --- E.11 dimension scaling -------------------------------------------------

#[test]
fn dimension_scaling_oracle_remains_valid() {
    for dimension in [1_usize, 2, 4, 8, 32, 128, 256] {
        let weights: Vec<f32> = (0..dimension)
            .map(|index| (index as f32 * 0.37).sin())
            .collect();
        let features: Vec<f32> = (0..dimension)
            .map(|index| (index as f32 * 0.31 + 0.5).cos())
            .collect();
        let model = model(&weights, 0.125);
        let produced = model.predict(&Vector::from_slice(&features)).unwrap();
        assert!(produced.is_finite(), "non-finite at D={dimension}");
        assert_bits_eq(produced, scalar_predict_f32(&weights, 0.125, &features));
        let oracle = scalar_predict_f64(&weights, 0.125, &features);
        let error = (f64::from(produced) - oracle).abs();
        let tolerance = tolerance(&weights, 0.125, &features);
        println!("dimension D={dimension}: abs_err={error:e} tol={tolerance:e}");
        assert!(
            error <= tolerance,
            "D={dimension}: |error| {error} > tolerance {tolerance}"
        );
    }
}

#[test]
#[ignore = "release-only diagnostic: D=256 at 1e10 scale (finite-envelope characterization)"]
fn dimension_scaling_extreme_ignored() {
    let dimension = 256;
    let weights: Vec<f32> = (0..dimension)
        .map(|index| if index % 2 == 0 { 1.0e10 } else { -1.0e10 })
        .collect();
    let features: Vec<f32> = (0..dimension)
        .map(|index| (index as f32 * 0.37).sin() * 1.0e10)
        .collect();
    let model = model(&weights, 0.0);
    let produced = model.predict(&Vector::from_slice(&features)).unwrap();
    // Products are ≈ ±1e20 and partial sums stay well below f32::MAX.
    assert!(produced.is_finite(), "unexpected non-finite: {produced}");
}

// --- E.12 failure behavior --------------------------------------------------

#[test]
fn failure_does_not_corrupt_subsequent_prediction() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let snapshot = snapshot_params(&model);
    let reference = model.clone();
    let valid = [1.0_f32, -2.0, 0.5];

    assert!(model.predict(&Vector::from_slice(&[1.0])).is_err());
    assert!(
        model
            .predict(&Vector::from_slice(&[f32::NAN, 0.0, 0.0]))
            .is_err()
    );
    assert_bits_eq(
        model.predict(&Vector::from_slice(&valid)).unwrap(),
        reference.predict(&Vector::from_slice(&valid)).unwrap(),
    );
    assert_params_unchanged(&model, &snapshot);
}

#[test]
fn failed_streaming_prediction_then_valid_continues() {
    let streaming_model = model(&[2.0], 1.0);
    let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
    executor.process_one(&observation(&[3.0])).unwrap();
    assert!(executor.process_one(&observation(&[1.0, 2.0])).is_err());
    assert_bits_eq(*executor.process_one(&observation(&[4.0])).unwrap(), 9.0);
    assert_bits_eq(*executor.state(), 9.0);
}

#[test]
fn failed_batch_then_valid_batch_succeeds() {
    let model = model(&[1.0, 1.0], -0.5);
    let snapshot = snapshot_params(&model);
    assert!(model.predict_batch(&[observation(&[1.0])]).is_err());
    let valid = model.predict_batch(&[observation(&[1.0, 1.0])]).unwrap();
    assert_bits_eq(
        valid[0],
        model.predict(&Vector::from_slice(&[1.0, 1.0])).unwrap(),
    );
    assert_params_unchanged(&model, &snapshot);
}

// --- E.13 reference vs production classification ----------------------------

#[test]
fn single_implementation_delegation() {
    // LR has one implementation (`predict`); `StateModel::update` and
    // `predict_batch` delegate to it. LR has no dual reference/production path.
    let model = model(&[1.5, -2.0, 0.25], 0.125);
    let observation = observation(&[4.0, 5.0, -6.0]);
    let reference = model.predict(observation.value()).unwrap();
    for state in [0.0_f32, 7.5, -3.0] {
        assert_bits_eq(model.update(&state, &observation).unwrap(), reference);
    }
    let batch = model
        .predict_batch(std::slice::from_ref(&observation))
        .unwrap();
    assert_bits_eq(batch[0], reference);
}
