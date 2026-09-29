//! GRU behavior tests.
//!
//! The reference values are computed by an independent scalar implementation
//! (plain `Vec` loops and standard-library `exp`/`tanh`) written directly from
//! the documented gate convention in `src/models/recurrent/gru.rs`. It does not
//! call the library's kernels, so agreement is evidence the implementation is
//! correct rather than evidence the two paths share a bug.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Matrix, RandomGenerator, Vector};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{StateModel, process_batch, process_one, process_stream};
use seqvex::models::recurrent::gru::{Gru, GruError, GruExecutor, GruParameters};

// --- independent scalar reference -------------------------------------------

#[derive(Clone)]
struct RefParams {
    w_z: Vec<Vec<f32>>,
    u_z: Vec<Vec<f32>>,
    b_z: Vec<f32>,
    w_r: Vec<Vec<f32>>,
    u_r: Vec<Vec<f32>>,
    b_r: Vec<f32>,
    w_h: Vec<Vec<f32>>,
    u_h: Vec<Vec<f32>>,
    b_h: Vec<f32>,
}

fn ref_matvec(matrix: &[Vec<f32>], input: &[f32]) -> Vec<f32> {
    matrix
        .iter()
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}

fn ref_add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn ref_mul(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a * b).collect()
}

fn ref_map(values: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    values.iter().map(|&value| f(value)).collect()
}

fn ref_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn ref_tanh(value: f32) -> f32 {
    value.tanh()
}

fn ref_step(params: &RefParams, previous: &[f32], input: &[f32]) -> Vec<f32> {
    let update_gate = ref_map(
        &ref_add(
            &ref_add(
                &ref_matvec(&params.w_z, input),
                &ref_matvec(&params.u_z, previous),
            ),
            &params.b_z,
        ),
        ref_sigmoid,
    );
    let reset_gate = ref_map(
        &ref_add(
            &ref_add(
                &ref_matvec(&params.w_r, input),
                &ref_matvec(&params.u_r, previous),
            ),
            &params.b_r,
        ),
        ref_sigmoid,
    );
    let candidate = ref_map(
        &ref_add(
            &ref_add(
                &ref_matvec(&params.w_h, input),
                &ref_matvec(&params.u_h, &ref_mul(&reset_gate, previous)),
            ),
            &params.b_h,
        ),
        ref_tanh,
    );
    ref_add(
        &ref_mul(&ref_map(&update_gate, |value| 1.0 - value), previous),
        &ref_mul(&update_gate, &candidate),
    )
}

/// A different exact identity for the logistic sigmoid:
/// `0.5 * (1 + tanh(0.5 * x)) == 1 / (1 + exp(-x))`.
fn ref_sigmoid_alt(value: f32) -> f32 {
    0.5 * (1.0 + (0.5 * value).tanh())
}

/// The same four documented equations as `ref_step`, but with explicit scalar
/// index loops and the alternative sigmoid identity above. The gate convention
/// is unchanged; only the sigmoid formulation and the accumulation order differ,
/// so agreement with production is within tolerance rather than bitwise.
// The explicit scalar index loops are deliberate: they are the independent
// formulation, not an iterator rewrite of the production path.
#[allow(clippy::needless_range_loop)]
fn ref_step_alt(params: &RefParams, previous: &[f32], input: &[f32]) -> Vec<f32> {
    let hidden = previous.len();
    let input_dim = input.len();

    let mut update_gate = vec![0.0_f32; hidden];
    for unit in 0..hidden {
        let mut activation = params.b_z[unit];
        for feature in 0..input_dim {
            activation += params.w_z[unit][feature] * input[feature];
        }
        for state in 0..hidden {
            activation += params.u_z[unit][state] * previous[state];
        }
        update_gate[unit] = ref_sigmoid_alt(activation);
    }

    let mut reset_gate = vec![0.0_f32; hidden];
    for unit in 0..hidden {
        let mut activation = params.b_r[unit];
        for feature in 0..input_dim {
            activation += params.w_r[unit][feature] * input[feature];
        }
        for state in 0..hidden {
            activation += params.u_r[unit][state] * previous[state];
        }
        reset_gate[unit] = ref_sigmoid_alt(activation);
    }

    let mut candidate = vec![0.0_f32; hidden];
    for unit in 0..hidden {
        let mut activation = params.b_h[unit];
        for feature in 0..input_dim {
            activation += params.w_h[unit][feature] * input[feature];
        }
        for state in 0..hidden {
            activation += params.u_h[unit][state] * reset_gate[state] * previous[state];
        }
        candidate[unit] = activation.tanh();
    }

    let mut next = vec![0.0_f32; hidden];
    for unit in 0..hidden {
        next[unit] =
            (1.0 - update_gate[unit]) * previous[unit] + update_gate[unit] * candidate[unit];
    }
    next
}

/// Fixed-seed linear congruential generator, mirroring the style of
/// `GruParameters::deterministic`; deterministic pseudorandom parameters and
/// inputs only.
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

fn random_matrix(lcg: &mut Lcg, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| lcg.next_f32()).collect())
        .collect()
}

fn random_vector(lcg: &mut Lcg, len: usize) -> Vec<f32> {
    (0..len).map(|_| lcg.next_f32()).collect()
}

fn random_params(input_dim: usize, hidden_dim: usize, seed: u64) -> RefParams {
    let mut lcg = Lcg::new(seed);
    RefParams {
        w_z: random_matrix(&mut lcg, hidden_dim, input_dim),
        u_z: random_matrix(&mut lcg, hidden_dim, hidden_dim),
        b_z: random_vector(&mut lcg, hidden_dim),
        w_r: random_matrix(&mut lcg, hidden_dim, input_dim),
        u_r: random_matrix(&mut lcg, hidden_dim, hidden_dim),
        b_r: random_vector(&mut lcg, hidden_dim),
        w_h: random_matrix(&mut lcg, hidden_dim, input_dim),
        u_h: random_matrix(&mut lcg, hidden_dim, hidden_dim),
        b_h: random_vector(&mut lcg, hidden_dim),
    }
}

fn matrix(rows: &[Vec<f32>]) -> Matrix {
    let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    Matrix::from_rows(&slices).unwrap()
}

fn to_parameters(params: &RefParams) -> GruParameters {
    GruParameters {
        w_z: matrix(&params.w_z),
        u_z: matrix(&params.u_z),
        b_z: Vector::from_slice(&params.b_z),
        w_r: matrix(&params.w_r),
        u_r: matrix(&params.u_r),
        b_r: Vector::from_slice(&params.b_r),
        w_h: matrix(&params.w_h),
        u_h: matrix(&params.u_h),
        b_h: Vector::from_slice(&params.b_h),
    }
}

fn sample_params() -> RefParams {
    RefParams {
        w_z: vec![vec![0.5, -0.25], vec![0.1, 0.3]],
        u_z: vec![vec![-0.2, 0.4], vec![0.05, -0.15]],
        b_z: vec![0.1, -0.05],
        w_r: vec![vec![0.15, 0.2], vec![-0.3, 0.05]],
        u_r: vec![vec![0.25, -0.1], vec![0.2, 0.35]],
        b_r: vec![0.0, 0.05],
        w_h: vec![vec![0.4, -0.35], vec![0.2, 0.1]],
        u_h: vec![vec![0.3, 0.15], vec![-0.25, 0.45]],
        b_h: vec![-0.1, 0.2],
    }
}

fn observation(values: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(values))
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_bitwise_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
    }
}

// --- construction -----------------------------------------------------------

#[test]
fn construction_rejects_zero_dimensions() {
    assert_eq!(
        Gru::new(0, 1, GruParameters::deterministic(1, 1)).unwrap_err(),
        GruError::ZeroDimension
    );
    assert_eq!(
        Gru::new(1, 0, GruParameters::deterministic(1, 1)).unwrap_err(),
        GruError::ZeroDimension
    );
}

#[test]
fn construction_rejects_wrong_parameter_dimensions() {
    assert_eq!(
        Gru::new(3, 2, GruParameters::deterministic(2, 2)).unwrap_err(),
        GruError::DimensionMismatch {
            expected: 3,
            actual: 2,
        }
    );
}

#[test]
fn construction_rejects_non_finite_parameters() {
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[f32::NAN]]).unwrap(),
        ..GruParameters::deterministic(1, 1)
    };
    assert_eq!(
        Gru::new(1, 1, parameters).unwrap_err(),
        GruError::NonFiniteParameter
    );
}

// --- one step and reference values ------------------------------------------

#[test]
fn single_step_matches_hand_computed_value() {
    // z = r = sigmoid(0) = 0.5; the reset gate zeroes h_0 in the candidate, so
    // h̃ = tanh(1); h_1 = (1 - 0.5) * 0 + 0.5 * tanh(1).
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        u_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_z: Vector::zeros(1),
        w_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        u_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_r: Vector::zeros(1),
        w_h: Matrix::from_rows(&[&[1.0]]).unwrap(),
        u_h: Matrix::from_rows(&[&[1.0]]).unwrap(),
        b_h: Vector::zeros(1),
    };
    let gru = Gru::new(1, 1, parameters).unwrap();
    let hidden = gru.update(&Vector::zeros(1), &observation(&[1.0])).unwrap();
    assert_close(hidden.as_slice(), &[0.5 * 1.0_f32.tanh()], 1e-6);
}

#[test]
fn repeated_steps_match_independent_reference() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut state = Vector::zeros(2);
    let mut hidden = vec![0.0, 0.0];
    for input in [
        vec![0.5, -0.5],
        vec![1.0, 0.25],
        vec![-0.75, 0.1],
        vec![0.0, 0.9],
    ] {
        state = gru.update(&state, &observation(&input)).unwrap();
        hidden = ref_step(&reference, &hidden, &input);
        assert_close(state.as_slice(), &hidden, 1e-5);
    }
}

#[test]
fn step_uses_previous_committed_state() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let after_first = gru
        .update(&Vector::zeros(2), &observation(&[0.5, -0.5]))
        .unwrap();
    let state = gru
        .update(&after_first, &observation(&[1.0, 0.25]))
        .unwrap();
    let expected = ref_step(&reference, after_first.as_slice(), &[1.0, 0.25]);
    assert_close(state.as_slice(), &expected, 1e-5);
}

#[test]
fn state_model_update_matches_independent_reference() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let input = [0.3, -0.2];
    let produced = gru.update(&Vector::zeros(2), &observation(&input)).unwrap();
    let expected = ref_step(&reference, &[0.0, 0.0], &input);
    assert_close(produced.as_slice(), &expected, 1e-5);
}

#[test]
fn sequence_context_does_not_change_the_computation() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let plain = gru
        .update(&Vector::zeros(2), &observation(&[0.5, -0.5]))
        .unwrap();
    let ordered = gru
        .update(
            &Vector::zeros(2),
            &observation(&[0.5, -0.5])
                .with_sequence(seqvex::foundation::observation::SequenceNumber::new(7)),
        )
        .unwrap();
    assert_eq!(plain, ordered);
}

// --- independent closed-form and alternative-formulation oracle ---------------

#[test]
fn exact_closed_form_decays_hidden_state_bitwise() {
    // With W_h = U_h = b_h = 0 the candidate is tanh(0) = 0, and with
    // W_z = U_z = b_z = 0 the update gate is sigmoid(0) = 0.5 exactly. Then
    // h_t = 0.5 * h_{t-1}, and multiplication by 0.5 is exact in f32, so
    // h_n = 0.5^n * h_0 bitwise. h_0 is supplied through StateModel::update,
    // which takes an explicit previous state; the model is not mutated.
    let input_dim = 2;
    let hidden_dim = 2;
    let zero_matrix = |rows: usize, cols: usize| vec![vec![0.0_f32; cols]; rows];
    let params = RefParams {
        w_z: zero_matrix(hidden_dim, input_dim),
        u_z: zero_matrix(hidden_dim, hidden_dim),
        b_z: vec![0.0; hidden_dim],
        w_r: zero_matrix(hidden_dim, input_dim),
        u_r: zero_matrix(hidden_dim, hidden_dim),
        b_r: vec![0.0; hidden_dim],
        w_h: zero_matrix(hidden_dim, input_dim),
        u_h: zero_matrix(hidden_dim, hidden_dim),
        b_h: vec![0.0; hidden_dim],
    };
    let gru = Gru::new(input_dim, hidden_dim, to_parameters(&params)).unwrap();
    let observation = observation(&[0.0, 0.0]);

    for steps in [1_usize, 3, 8] {
        let mut hidden = Vector::from_slice(&[1.0, 2.0]);
        for _ in 0..steps {
            hidden = gru.update(&hidden, &observation).unwrap();
        }
        let mut expected = vec![1.0_f32, 2.0];
        for _ in 0..steps {
            for value in &mut expected {
                *value *= 0.5;
            }
        }
        assert_bitwise_eq(hidden.as_slice(), &expected);
    }
}

#[test]
fn alt_reference_matches_production_over_randomized_sequences() {
    // Same documented equations, independent sigmoid identity
    // (0.5*(1+tanh(0.5x))) and independent operation structure, so agreement is
    // within 1e-5 absolute rather than bitwise.
    let cases = [(1_usize, 1_usize), (2, 2), (3, 5), (5, 3)];
    for (case, (input_dim, hidden_dim)) in cases.into_iter().enumerate() {
        let params = random_params(input_dim, hidden_dim, 0xa11c_e000 + case as u64);
        let gru = Gru::new(input_dim, hidden_dim, to_parameters(&params)).unwrap();
        let mut lcg = Lcg::new(0xbeef_5eed + case as u64);
        let mut hidden = vec![0.0_f32; hidden_dim];
        let mut state = Vector::zeros(hidden_dim);

        for _ in 0..80 {
            let input = random_vector(&mut lcg, input_dim);
            let observation = observation(&input);
            let expected = ref_step_alt(&params, &hidden, &input);

            state = gru.update(&state, &observation).unwrap();
            assert_close(state.as_slice(), &expected, 1e-5);

            hidden = expected;
        }
    }
}

// --- long sequences, reset ---------------------------------------------------

#[test]
fn long_sequence_matches_independent_reference() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut state = Vector::zeros(2);
    let mut hidden = vec![0.0, 0.0];
    for step in 0..500 {
        let input = vec![(step as f32 * 0.37).sin(), (step as f32 * 0.11).cos()];
        state = gru.update(&state, &observation(&input)).unwrap();
        hidden = ref_step(&reference, &hidden, &input);
    }
    assert_close(state.as_slice(), &hidden, 1e-4);
}

#[test]
fn reset_starts_a_new_sequence() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let reset = Vector::zeros(2);
    assert_close(reset.as_slice(), &[0.0, 0.0], 0.0);

    let input = [0.2, -0.7];
    // After a reset (zero state), the next observation equals a fresh model's
    // first observation.
    let after_reset = gru.update(&reset, &observation(&input)).unwrap();
    let fresh = gru.update(&Vector::zeros(2), &observation(&input)).unwrap();
    assert_close(after_reset.as_slice(), fresh.as_slice(), 1e-6);

    let continued_first = gru.update(&reset, &observation(&[0.5, -0.5])).unwrap();
    let continued = gru.update(&continued_first, &observation(&input)).unwrap();
    assert_ne!(after_reset.as_slice(), continued.as_slice());
}

// --- failure behavior --------------------------------------------------------

#[test]
fn non_finite_input_is_rejected_and_state_is_preserved() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let committed = gru
        .update(&Vector::zeros(2), &observation(&[0.5, -0.5]))
        .unwrap();
    assert_eq!(
        gru.update(&committed, &observation(&[f32::NAN, 0.0]))
            .unwrap_err(),
        GruError::NonFiniteInput
    );
    // The failed update returned no state; the committed state still matches the
    // independent oracle for the last successful transition.
    let expected = ref_step(&reference, &[0.0, 0.0], &[0.5, -0.5]);
    assert_close(committed.as_slice(), &expected, 1e-5);
}

#[test]
fn wrong_input_length_is_rejected_and_state_is_preserved() {
    let reference = sample_params();
    let gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let committed = gru
        .update(&Vector::zeros(2), &observation(&[0.5, -0.5]))
        .unwrap();
    assert_eq!(
        gru.update(&committed, &observation(&[1.0])).unwrap_err(),
        GruError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    let expected = ref_step(&reference, &[0.0, 0.0], &[0.5, -0.5]);
    assert_close(committed.as_slice(), &expected, 1e-5);
}

#[test]
fn non_finite_candidate_is_rejected_and_state_is_preserved() {
    // inf + (-inf) in the update gate produces NaN from finite inputs and
    // finite parameters, exercising the candidate validation boundary.
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[f32::MAX, f32::MAX]]).unwrap(),
        u_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_z: Vector::zeros(1),
        w_r: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_r: Vector::zeros(1),
        w_h: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_h: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_h: Vector::zeros(1),
    };
    let gru = Gru::new(2, 1, parameters).unwrap();
    let zero = Vector::zeros(1);
    assert_eq!(
        gru.update(&zero, &observation(&[f32::MAX, -f32::MAX]))
            .unwrap_err(),
        GruError::NonFiniteCandidate
    );
    assert_close(zero.as_slice(), &[0.0], 0.0);
}

// --- determinism -------------------------------------------------------------

#[test]
fn deterministic_parameters_are_reproducible() {
    assert_eq!(
        GruParameters::deterministic(3, 2),
        GruParameters::deterministic(3, 2)
    );

    let left = Gru::new(2, 2, GruParameters::deterministic(2, 2)).unwrap();
    let right = Gru::new(2, 2, GruParameters::deterministic(2, 2)).unwrap();
    let observation = observation(&[0.4, 0.6]);
    let left_state = left.update(&Vector::zeros(2), &observation).unwrap();
    let right_state = right.update(&Vector::zeros(2), &observation).unwrap();
    assert_eq!(left_state, right_state);
}

// --- production path equivalence --------------------------------------------

#[test]
fn production_step_matches_reference_over_long_sequence() {
    let reference_model = Gru::new(8, 16, GruParameters::deterministic(8, 16)).unwrap();
    let production_model = Gru::new(8, 16, GruParameters::deterministic(8, 16)).unwrap();
    let mut reference = StreamingExecutor::new(&reference_model, Vector::zeros(16));
    let mut production = GruExecutor::new(&production_model, Vector::zeros(16));

    for step in 0..500 {
        let input = Vector::from_fn(8, |i| ((step as f32) * 0.13 + (i as f32) * 0.07).sin());
        let observation = Observation::new(input);
        reference.process_one(&observation).unwrap();
        production.process_one_optimized(&observation).unwrap();
        assert_bitwise_eq(production.state().as_slice(), reference.state().as_slice());

        if step == 250 {
            reference.reset(Vector::zeros(16));
            production.reset(Vector::zeros(16));
            assert_bitwise_eq(production.state().as_slice(), reference.state().as_slice());
        }
    }
}

#[test]
fn production_matches_update_bit_for_bit() {
    let reference = sample_params();
    let model = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut update_state = Vector::zeros(2);
    let mut executor = GruExecutor::new(&model, Vector::zeros(2));

    for step in 0..64 {
        let input = vec![(step as f32 * 0.23).sin(), (step as f32 * 0.07).cos()];
        let observation = observation(&input);
        update_state = model.update(&update_state, &observation).unwrap();
        executor.process_one_optimized(&observation).unwrap();
        assert_bitwise_eq(executor.state().as_slice(), update_state.as_slice());
    }
}

#[test]
fn production_step_rejects_non_finite_input_and_preserves_state() {
    let reference = sample_params();
    let model = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut executor = GruExecutor::new(&model, Vector::zeros(2));
    executor
        .process_one_optimized(&observation(&[0.5, -0.5]))
        .unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor
            .process_one_optimized(&observation(&[f32::NAN, 0.0]))
            .unwrap_err(),
        GruError::NonFiniteInput
    );
    assert_bitwise_eq(executor.state().as_slice(), committed.as_slice());
}

#[test]
fn production_step_rejects_wrong_input_length_and_preserves_state() {
    let reference = sample_params();
    let model = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut executor = GruExecutor::new(&model, Vector::zeros(2));
    executor
        .process_one_optimized(&observation(&[0.5, -0.5]))
        .unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor
            .process_one_optimized(&observation(&[1.0]))
            .unwrap_err(),
        GruError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    assert_bitwise_eq(executor.state().as_slice(), committed.as_slice());
}

#[test]
fn production_step_rejects_non_finite_candidate_and_preserves_state() {
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[f32::MAX, f32::MAX]]).unwrap(),
        u_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_z: Vector::zeros(1),
        w_r: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_r: Vector::zeros(1),
        w_h: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_h: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_h: Vector::zeros(1),
    };
    let model = Gru::new(2, 1, parameters).unwrap();
    let mut executor = GruExecutor::new(&model, Vector::zeros(1));

    assert_eq!(
        executor
            .process_one_optimized(&observation(&[f32::MAX, -f32::MAX]))
            .unwrap_err(),
        GruError::NonFiniteCandidate
    );
    assert_bitwise_eq(executor.state().as_slice(), &[0.0]);
}

// --- sequential / causal / non-IID audit (#27) ------------------------------
//
// The GRU recurrence is order-dependent by construction. These checks close the
// sequential-audit dimensions the cross-algorithm gate (#27) records as
// remaining for GRU: prefix/causality, chunked continuation, permutation
// expectation, controlled non-IID regimes, and long-run numerical behaviour.
// All expectations come from the independent scalar `ref_step` oracle, never
// from the production recurrence itself.

fn states_differ(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .any(|(a, b)| a.to_bits() != b.to_bits())
}

/// The GRU recurrence is a convex blend of the previous state and `tanh`
/// (magnitude < 1), so from a zero start the hidden state stays within `[-1, 1]`
/// up to `f32` rounding. Divergence smoke test only; not a statistical bound.
fn assert_unit_bound(values: &[f32]) {
    for &value in values {
        assert!(
            value.is_finite() && value.abs() <= 1.0 + 1e-6,
            "hidden entry outside the unit bound: {value}"
        );
    }
}

fn scalar_fold(params: &RefParams, inputs: &[Vec<f32>]) -> Vec<f32> {
    let mut hidden = vec![0.0_f32; params.b_z.len()];
    for input in inputs {
        hidden = ref_step(params, &hidden, input);
    }
    hidden
}

#[test]
fn prefix_state_depends_only_on_observations_through_t() {
    // State at time t must be identical whether or not later observations exist:
    // the implementation consumes observations in order and cannot see the
    // future. A prefix model stopped after 32 steps must match the full model at
    // that point, and the independent scalar oracle on the same prefix.
    let params = random_params(2, 3, 0xabc_1234);
    let inputs: Vec<Vec<f32>> = (0..64)
        .map(|step| vec![(step as f32 * 0.21).sin(), (step as f32 * 0.13).cos()])
        .collect();

    let full_model = Gru::new(2, 3, to_parameters(&params)).unwrap();
    let prefix_model = Gru::new(2, 3, to_parameters(&params)).unwrap();

    let mut full_state = Vector::zeros(3);
    let mut prefix_state = Vector::zeros(3);
    for (step, input) in inputs.iter().enumerate() {
        let observation = observation(input);
        full_state = process_one(&full_model, &full_state, &observation).unwrap();
        if step < 32 {
            prefix_state = process_one(&prefix_model, &prefix_state, &observation).unwrap();
        }
        if step == 31 {
            assert_bitwise_eq(prefix_state.as_slice(), full_state.as_slice());
        }
    }

    assert_close(
        prefix_state.as_slice(),
        &scalar_fold(&params, &inputs[..32]),
        1e-5,
    );
    assert!(
        states_differ(prefix_state.as_slice(), full_state.as_slice()),
        "the full run must have advanced past the prefix"
    );
}

#[test]
fn chunked_execution_matches_continuous_stream() {
    // Feeding the stream in chunks while preserving state across chunk
    // boundaries must equal one continuous ordered fold, both through the
    // foundation `process_batch` and through `StreamingExecutor::process_stream`.
    let params = random_params(3, 4, 0xc0ff_ee01);
    let gru = Gru::new(3, 4, to_parameters(&params)).unwrap();
    let inputs: Vec<Vec<f32>> = (0..200)
        .map(|step| {
            vec![
                (step as f32 * 0.17).sin(),
                (step as f32 * 0.07).cos(),
                ((step * step) as f32 * 0.001).sin(),
            ]
        })
        .collect();
    let observations: Vec<Observation<Vector>> =
        inputs.iter().map(|input| observation(input)).collect();

    let continuous = process_batch(&gru, Vector::zeros(4), observations.iter().cloned()).unwrap();

    let mut chunked = Vector::zeros(4);
    for chunk in observations.chunks(7) {
        chunked = process_batch(&gru, chunked, chunk.iter().cloned()).unwrap();
    }
    assert_bitwise_eq(chunked.as_slice(), continuous.as_slice());

    let executor_model = Gru::new(3, 4, to_parameters(&params)).unwrap();
    let mut executor = StreamingExecutor::new(&executor_model, Vector::zeros(4));
    let streamed = executor
        .process_stream(observations.iter().cloned(), |_| {})
        .clone();
    assert_bitwise_eq(streamed.as_slice(), continuous.as_slice());

    assert_close(continuous.as_slice(), &scalar_fold(&params, &inputs), 1e-5);
}

#[test]
fn sequence_order_is_semantically_significant() {
    // #27 requires the correct permutation property rather than a universal
    // rule: a recurrent model is order-dependent, so reordering the same
    // observations defines a different trajectory. Verify that expectation
    // directly against the independent oracle.
    let params = random_params(2, 2, 0x0dd_ba11);
    let inputs: Vec<Vec<f32>> = (0..12)
        .map(|step| vec![(step as f32 * 0.5).sin(), (step as f32 * 0.9).cos()])
        .collect();
    let forward = scalar_fold(&params, &inputs);
    let mut reversed_inputs = inputs.clone();
    reversed_inputs.reverse();
    let reversed = scalar_fold(&params, &reversed_inputs);
    assert!(
        states_differ(&forward, &reversed),
        "GRU recurrence must be order-sensitive"
    );

    let gru = Gru::new(2, 2, to_parameters(&params)).unwrap();
    let mut state = Vector::zeros(2);
    for input in &inputs {
        state = gru.update(&state, &observation(input)).unwrap();
        assert!(state.as_slice().iter().all(|value| value.is_finite()));
    }
    assert_close(state.as_slice(), &forward, 1e-5);
}

#[test]
fn regime_shift_sequence_matches_independent_reference() {
    // Piecewise-stationary inputs (one regime, then a different one) are a
    // controlled non-IID sequence. The transition itself must remain causal and
    // numerically bounded, and match the independent oracle throughout.
    let params = random_params(3, 4, 0x5e90_0001);
    let mut inputs = Vec::with_capacity(200);
    for step in 0..200 {
        if step < 100 {
            inputs.push(vec![
                0.4 + (step as f32 * 0.1).sin(),
                -0.2,
                0.1 * (step as f32 * 0.05).cos(),
            ]);
        } else {
            inputs.push(vec![(step as f32 * 0.7).sin(), 0.6, -0.5]);
        }
    }

    let gru = Gru::new(3, 4, to_parameters(&params)).unwrap();
    let mut state = Vector::zeros(4);
    let mut expected = vec![0.0_f32; 4];
    for input in &inputs {
        state = gru.update(&state, &observation(input)).unwrap();
        expected = ref_step(&params, &expected, input);
        assert!(state.as_slice().iter().all(|value| value.is_finite()));
        assert_unit_bound(state.as_slice());
    }
    assert_close(state.as_slice(), &expected, 1e-5);
}

#[test]
fn autocorrelated_sequence_matches_independent_reference() {
    // Temporally dependent (AR(1)) inputs exercise the recurrence under
    // autocorrelation rather than IID draws.
    let params = random_params(2, 3, 0xa0c0_0001);
    let mut lcg = Lcg::new(0x1234_5678);
    let mut previous = vec![0.0_f32; 2];
    let gru = Gru::new(2, 3, to_parameters(&params)).unwrap();
    let mut state = Vector::zeros(3);
    let mut expected = vec![0.0_f32; 3];
    for _ in 0..400 {
        let mut input = vec![0.0_f32; 2];
        for (slot, previous_value) in input.iter_mut().zip(previous.iter()) {
            *slot = 0.8 * *previous_value + 0.1 * lcg.next_f32();
        }
        previous.clone_from(&input);
        state = gru.update(&state, &observation(&input)).unwrap();
        expected = ref_step(&params, &expected, &input);
        assert_unit_bound(state.as_slice());
    }
    assert_close(state.as_slice(), &expected, 1e-5);
}

#[test]
fn repeated_observation_does_not_drift_from_reference() {
    // A repeated identical observation is the degenerate low-diversity case; the
    // state must still track the independent oracle over a long run.
    let params = random_params(2, 2, 0xdead_1000);
    let gru = Gru::new(2, 2, to_parameters(&params)).unwrap();
    let input = vec![0.37_f32, -0.62];
    let observation = observation(&input);
    let mut state = Vector::zeros(2);
    let mut expected = vec![0.0_f32; 2];
    for step in 0..2_000 {
        state = gru.update(&state, &observation).unwrap();
        expected = ref_step(&params, &expected, &input);
        assert!(state.as_slice().iter().all(|value| value.is_finite()));
        assert_unit_bound(state.as_slice());
        if step % 250 == 0 || step == 1_999 {
            assert_close(state.as_slice(), &expected, 1e-5);
        }
    }
    assert_close(state.as_slice(), &expected, 1e-5);
}

#[test]
fn long_run_matches_reference_and_stays_bounded() {
    // Long-horizon numerical audit: 10_000 deterministic observations, with the
    // reference and production paths bitwise identical at every step and both
    // tracking the independent scalar oracle. The hidden state is a convex blend
    // of the previous state and `tanh` (magnitude < 1), so from a zero start
    // |h| <= 1 is a mathematical invariant, not a fitted tolerance; it is
    // checked as a divergence smoke test.
    let params = random_params(3, 5, 0x106e_0001);
    let reference_model = Gru::new(3, 5, to_parameters(&params)).unwrap();
    let production_model = Gru::new(3, 5, to_parameters(&params)).unwrap();
    let mut reference = StreamingExecutor::new(&reference_model, Vector::zeros(5));
    let mut production = GruExecutor::new(&production_model, Vector::zeros(5));
    let mut scalar = vec![0.0_f32; 5];
    let mut lcg = Lcg::new(0xfeed_0001);
    let mut max_abs = 0.0_f32;

    for _ in 0..10_000 {
        let input = vec![lcg.next_f32(), lcg.next_f32(), lcg.next_f32()];
        let observation = observation(&input);
        reference.process_one(&observation).unwrap();
        production.process_one_optimized(&observation).unwrap();
        scalar = ref_step(&params, &scalar, &input);

        assert_bitwise_eq(production.state().as_slice(), reference.state().as_slice());
        assert!(
            reference
                .state()
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        );
        max_abs = max_abs.max(
            reference
                .state()
                .as_slice()
                .iter()
                .fold(0.0_f32, |max, value| max.max(value.abs())),
        );
    }

    assert_close(reference.state().as_slice(), &scalar, 1e-4);
    assert!(
        max_abs <= 1.0 + 1e-6,
        "hidden state exceeded the unit bound: {max_abs}"
    );
}

#[test]
fn stream_failure_preserves_state_and_continues_from_last_valid() {
    // A failed update must not advance the committed state; the stream continues
    // from the last valid state, so skipping the bad observation is equivalent to
    // never having supplied it.
    let params = random_params(2, 2, 0xfa11_0001);
    let gru = Gru::new(2, 2, to_parameters(&params)).unwrap();
    let good = observation(&[0.3, -0.4]);
    let observations = [good.clone(), observation(&[f32::NAN, 0.1]), good.clone()];
    let mut failures = 0_usize;
    let final_state = process_stream(&gru, Vector::zeros(2), observations.iter().cloned(), |_| {
        failures += 1
    })
    .clone();

    assert_eq!(failures, 1);
    let mut expected = vec![0.0_f32; 2];
    expected = ref_step(&params, &expected, &[0.3, -0.4]);
    expected = ref_step(&params, &expected, &[0.3, -0.4]);
    assert_close(final_state.as_slice(), &expected, 1e-5);
}

// --- production parameter initialization -------------------------------------
//
// KAT values below are regression guards for a fixed dependency version; they do
// not prove PRNG statistical quality (`docs/RANDOMNESS.md`).

const INIT_SEED: u64 = 0x20_2026;
const INIT_W_Z_BITS: [u32; 6] = [
    1040303696, 3183172272, 3193933668, 3213583158, 1062216716, 3196331252,
];
const INIT_U_Z_BITS: [u32; 9] = [
    3181920064, 3206166352, 3197753888, 1056856344, 1056296408, 3204752864, 3199735400, 1050635104,
    1058511716,
];
const INIT_W_R_BITS: [u32; 6] = [
    1059577298, 3210317944, 3205156764, 1032894032, 3212080696, 1051457936,
];
const INIT_U_R_BITS: [u32; 9] = [
    1062129092, 3195957568, 3207081800, 3204079688, 3180773952, 1042619376, 3198690040, 3199229072,
    1057853656,
];
const INIT_W_H_BITS: [u32; 6] = [
    3205365034, 1055788580, 3158734208, 3200476934, 3188849436, 1065420982,
];
const INIT_U_H_BITS: [u32; 9] = [
    1055376800, 1049922032, 1058680456, 1059606408, 1058086072, 1060379132, 1029657344, 3206315656,
    3204765208,
];

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|bits| f32::from_bits(*bits)).collect()
}

fn ref_params_from(parameters: &GruParameters) -> RefParams {
    let rows = |matrix: &Matrix| -> Vec<Vec<f32>> {
        (0..matrix.rows())
            .map(|row| {
                (0..matrix.cols())
                    .map(|col| matrix.get(row, col).unwrap())
                    .collect()
            })
            .collect()
    };
    RefParams {
        w_z: rows(&parameters.w_z),
        u_z: rows(&parameters.u_z),
        b_z: parameters.b_z.as_slice().to_vec(),
        w_r: rows(&parameters.w_r),
        u_r: rows(&parameters.u_r),
        b_r: parameters.b_r.as_slice().to_vec(),
        w_h: rows(&parameters.w_h),
        u_h: rows(&parameters.u_h),
        b_h: parameters.b_h.as_slice().to_vec(),
    }
}

#[test]
fn initializer_is_reproducible_for_a_fixed_seed() {
    let mut left = RandomGenerator::from_seed(0xabcd);
    let mut right = RandomGenerator::from_seed(0xabcd);
    let left_parameters = GruParameters::init(3, 2, &mut left).unwrap();
    let right_parameters = GruParameters::init(3, 2, &mut right).unwrap();
    assert_eq!(left_parameters, right_parameters);
}

#[test]
fn initializer_matches_frozen_parameter_vector() {
    let mut rng = RandomGenerator::from_seed(INIT_SEED);
    let parameters = GruParameters::init(2, 3, &mut rng).unwrap();
    assert_bitwise_eq(parameters.w_z.as_slice(), &from_bits(&INIT_W_Z_BITS));
    assert_bitwise_eq(parameters.u_z.as_slice(), &from_bits(&INIT_U_Z_BITS));
    assert_bitwise_eq(parameters.w_r.as_slice(), &from_bits(&INIT_W_R_BITS));
    assert_bitwise_eq(parameters.u_r.as_slice(), &from_bits(&INIT_U_R_BITS));
    assert_bitwise_eq(parameters.w_h.as_slice(), &from_bits(&INIT_W_H_BITS));
    assert_bitwise_eq(parameters.u_h.as_slice(), &from_bits(&INIT_U_H_BITS));
}

#[test]
fn initializer_produces_shaped_finite_parameters_and_zero_biases() {
    let mut rng = RandomGenerator::from_seed(0x5ca1e);
    let parameters = GruParameters::init(2, 3, &mut rng).unwrap();
    for matrix in [
        &parameters.w_z,
        &parameters.u_z,
        &parameters.w_r,
        &parameters.u_r,
        &parameters.w_h,
        &parameters.u_h,
    ] {
        assert!(matrix.as_slice().iter().all(|value| value.is_finite()));
    }
    assert_eq!((parameters.w_z.rows(), parameters.w_z.cols()), (3, 2));
    assert_eq!((parameters.u_z.rows(), parameters.u_z.cols()), (3, 3));
    assert_eq!(parameters.b_z.as_slice(), &[0.0, 0.0, 0.0]);
    assert_eq!(parameters.b_r.as_slice(), &[0.0, 0.0, 0.0]);
    assert_eq!(parameters.b_h.as_slice(), &[0.0, 0.0, 0.0]);
    assert!(Gru::new(2, 3, parameters).is_ok());
}

#[test]
fn initializer_rejects_zero_dimensions() {
    let mut rng = RandomGenerator::from_seed(1);
    assert_eq!(
        GruParameters::init(0, 3, &mut rng).unwrap_err(),
        GruError::ZeroDimension
    );
    assert_eq!(
        GruParameters::init(3, 0, &mut rng).unwrap_err(),
        GruError::ZeroDimension
    );
}

#[test]
fn initializer_scale_matches_glorot_variance_and_shrinks_with_dimension() {
    let mut previous_input_half_width = f32::INFINITY;
    for (input_dim, hidden_dim) in [(2_usize, 3_usize), (8, 16), (32, 64)] {
        let input_half_width = (6.0_f64 / (input_dim as f64 + hidden_dim as f64)).sqrt() as f32;
        let recurrent_half_width = (3.0_f32 / hidden_dim as f32).sqrt();
        assert!(input_half_width < previous_input_half_width);
        previous_input_half_width = input_half_width;

        let mut rng =
            RandomGenerator::from_seed(0x5ca1e + input_dim as u64 * 31 + hidden_dim as u64);
        let parameters = GruParameters::init(input_dim, hidden_dim, &mut rng).unwrap();
        assert!(
            parameters
                .w_z
                .as_slice()
                .iter()
                .all(|value| value.abs() <= input_half_width)
        );
        assert!(
            parameters
                .u_z
                .as_slice()
                .iter()
                .all(|value| value.abs() <= recurrent_half_width)
        );

        let mut samples = parameters.w_z.as_slice().to_vec();
        samples.extend_from_slice(parameters.u_z.as_slice());
        let count = samples.len() as f32;
        let mean = samples.iter().sum::<f32>() / count;
        let variance = samples
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / count;
        let input_variance = input_half_width * input_half_width / 3.0;
        let recurrent_variance = recurrent_half_width * recurrent_half_width / 3.0;
        let expected_variance = (input_variance * (hidden_dim * input_dim) as f32
            + recurrent_variance * (hidden_dim * hidden_dim) as f32)
            / count;
        let standard_error = (expected_variance / count).sqrt();
        assert!(
            mean.abs() <= 4.0 * standard_error + 1e-6,
            "dim {input_dim}x{hidden_dim} mean {mean}"
        );
        assert!(
            (0.5..=2.0).contains(&(variance / expected_variance)),
            "dim {input_dim}x{hidden_dim} variance {variance}, expected ~{expected_variance}"
        );
    }
}

#[test]
fn initialized_model_matches_independent_reference() {
    let mut rng = RandomGenerator::from_seed(0x1234_5678);
    let parameters = GruParameters::init(2, 2, &mut rng).unwrap();
    let reference = ref_params_from(&parameters);
    let gru = Gru::new(2, 2, parameters).unwrap();
    let input = [0.4, -0.6];
    let produced = gru.update(&Vector::zeros(2), &observation(&input)).unwrap();
    let expected = ref_step(&reference, &[0.0, 0.0], &input);
    assert_close(produced.as_slice(), &expected, 1e-6);
}
