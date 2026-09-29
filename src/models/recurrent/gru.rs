//! A minimal stateful GRU (gated recurrent unit) for single-observation,
//! sequential inference.
//!
//! # Gate convention
//!
//! This module implements exactly one convention
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §14.3):
//!
//! ```text
//! z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)
//! r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)
//! h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
//! h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
//! ```
//!
//! Do not mix it with a different reference.
//!
//! # Atomicity
//!
//! [`Gru`] implements the foundation [`StateModel`], so a step computes a
//! candidate and commits it only on success. A failed step leaves the
//! previously committed hidden state untouched.
//!
//! # State ownership
//!
//! `Gru` holds only immutable configuration and parameters. The authoritative
//! hidden state `h` is per execution, not model-owned: a reference caller holds
//! it in a `Vector` advanced through [`StateModel::update`], and the optimized
//! [`GruExecutor`] holds the single authoritative `Vector` for its execution.
//! There is no second, model-owned hidden state.
//!
//! # Reference and production paths
//!
//! The [`StateModel::update`] implementation (via the private `compute`) is the
//! **reference** path: value-returning, allocating one `Vector` per operation,
//! and the semantic oracle every optimized path must be tested against.
//!
//! [`GruExecutor::process_one_optimized`] is the **optimized** path. It borrows
//! the model immutably, owns a private reusable `GruWorkspace`, allocates
//! nothing in steady state, and is bit-identical to the reference path in the
//! tested range. The workspace is per-execution scratch, never model-owned and
//! never shared between live executions; it does not generalize to other models.

use crate::foundation::numerical::{
    DimensionMismatch, Matrix, RandomGenerator, Vector, sigmoid, tanh,
};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

/// Failure classes reported by a GRU transition or construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GruError {
    /// A configured dimension is zero.
    ZeroDimension,
    /// A parameter or observation has the wrong length.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// A model parameter is not finite.
    NonFiniteParameter,
    /// An observation is not finite.
    NonFiniteInput,
    /// The computed candidate state is not finite.
    NonFiniteCandidate,
}

impl From<DimensionMismatch> for GruError {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// All parameters required by the gate convention above.
#[derive(Debug, Clone, PartialEq)]
pub struct GruParameters {
    /// Update-gate input weights.
    pub w_z: Matrix,
    /// Update-gate recurrent weights.
    pub u_z: Matrix,
    /// Update-gate bias.
    pub b_z: Vector,
    /// Reset-gate input weights.
    pub w_r: Matrix,
    /// Reset-gate recurrent weights.
    pub u_r: Matrix,
    /// Reset-gate bias.
    pub b_r: Vector,
    /// Candidate input weights.
    pub w_h: Matrix,
    /// Candidate recurrent weights.
    pub u_h: Matrix,
    /// Candidate bias.
    pub b_h: Vector,
}

impl GruParameters {
    /// Deterministic pseudo-random parameters for examples, tests, and benches.
    ///
    /// ponytail: this is a fixed LCG, not a trained or statistically sound
    /// initialization. It exists so a GRU can be constructed without adding a
    /// randomness dependency; trained parameters arrive from outside the core.
    /// however once we finalize and validate the algorithm; consider
    /// including hull-dobell therom we may need to take this apart
    /// Bit-Truncation Efficiency included wrapping_* will overflow automatically
    /// discard low-order bits and extracting only most significant bits
    pub fn deterministic(input_dim: usize, hidden_dim: usize) -> Self {
        let mut state = 0x5eed_5eed_5eed_5eed_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
        };
        Self {
            w_z: Matrix::from_fn(hidden_dim, input_dim, |_, _| next()),
            u_z: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| next()),
            b_z: Vector::from_fn(hidden_dim, |_| next()),
            w_r: Matrix::from_fn(hidden_dim, input_dim, |_, _| next()),
            u_r: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| next()),
            b_r: Vector::from_fn(hidden_dim, |_| next()),
            w_h: Matrix::from_fn(hidden_dim, input_dim, |_, _| next()),
            u_h: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| next()),
            b_h: Vector::from_fn(hidden_dim, |_| next()),
        }
    }

    /// Builds parameters by drawing bounded uniform values for the weight
    /// matrices and zeroing the biases.
    ///
    /// Input matrices `W_*` use Glorot/Xavier uniform scaling
    /// `a = sqrt(6 / (input_dim + hidden_dim))`; recurrent matrices `U_*` use
    /// `a = sqrt(3 / hidden_dim)`. Biases are zero and consume no draws. The
    /// generated parameters are an untrained starting point: they are not
    /// predictive, and they are not competition-ready.
    pub fn init(
        input_dim: usize,
        hidden_dim: usize,
        rng: &mut RandomGenerator,
    ) -> Result<Self, GruError> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(GruError::ZeroDimension);
        }
        let input_half_width = glorot_half_width(input_dim as f64 + hidden_dim as f64);
        let recurrent_half_width = glorot_half_width(2.0 * hidden_dim as f64);
        let input_range = rng
            .uniform_range(-input_half_width, input_half_width)
            .expect("positive dimensions yield a finite non-empty range");
        let recurrent_range = rng
            .uniform_range(-recurrent_half_width, recurrent_half_width)
            .expect("positive dimensions yield a finite non-empty range");

        Ok(Self {
            w_z: Matrix::from_fn(hidden_dim, input_dim, |_, _| rng.draw_f32(&input_range)),
            u_z: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| {
                rng.draw_f32(&recurrent_range)
            }),
            b_z: Vector::zeros(hidden_dim),
            w_r: Matrix::from_fn(hidden_dim, input_dim, |_, _| rng.draw_f32(&input_range)),
            u_r: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| {
                rng.draw_f32(&recurrent_range)
            }),
            b_r: Vector::zeros(hidden_dim),
            w_h: Matrix::from_fn(hidden_dim, input_dim, |_, _| rng.draw_f32(&input_range)),
            u_h: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| {
                rng.draw_f32(&recurrent_range)
            }),
            b_h: Vector::zeros(hidden_dim),
        })
    }
}

fn glorot_half_width(fan_sum: f64) -> f32 {
    (6.0 / fan_sum).sqrt() as f32
}

/// Reusable scratch storage for the allocation-free optimized path.
///
/// Owned by a [`GruExecutor`], allocated once per execution and reused every
/// observation, so the optimized step allocates nothing in steady state. It
/// holds only intermediate values, never the committed hidden state, and is
/// never shared between live executions.
#[derive(Debug)]
struct GruWorkspace {
    /// Holds `z_t`, then `(1 − z_t) ⊙ h`, then the committed candidate `h_t`.
    acc: Vec<f32>,
    /// Holds `r_t`, then `r_t ⊙ h`, then the candidate `h̃_t`.
    gate: Vec<f32>,
    /// Mat-vec partials and the `z_t ⊙ h̃_t` blend term.
    scratch: Vec<f32>,
}

impl GruWorkspace {
    fn new(hidden_dim: usize) -> Self {
        Self {
            acc: vec![0.0; hidden_dim],
            gate: vec![0.0; hidden_dim],
            scratch: vec![0.0; hidden_dim],
        }
    }
}

/// A GRU with immutable parameters and configuration.
///
/// The recurrent hidden state is not stored here; it is per-execution state
/// (see the module documentation).
#[derive(Debug)]
pub struct Gru {
    input_dim: usize,
    hidden_dim: usize,
    parameters: GruParameters,
}

impl Gru {
    /// Builds a GRU, validating dimensions and parameter finiteness.
    ///
    /// The initial hidden state is supplied by the caller, through
    /// [`StateModel::update`] or [`GruExecutor::new`].
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        parameters: GruParameters,
    ) -> Result<Self, GruError> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(GruError::ZeroDimension);
        }
        validate_matrix(&parameters.w_z, hidden_dim, input_dim)?;
        validate_matrix(&parameters.u_z, hidden_dim, hidden_dim)?;
        validate_vector(&parameters.b_z, hidden_dim)?;
        validate_matrix(&parameters.w_r, hidden_dim, input_dim)?;
        validate_matrix(&parameters.u_r, hidden_dim, hidden_dim)?;
        validate_vector(&parameters.b_r, hidden_dim)?;
        validate_matrix(&parameters.w_h, hidden_dim, input_dim)?;
        validate_matrix(&parameters.u_h, hidden_dim, hidden_dim)?;
        validate_vector(&parameters.b_h, hidden_dim)?;

        Ok(Self {
            input_dim,
            hidden_dim,
            parameters,
        })
    }

    /// The configured input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// The configured hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn compute(
        parameters: &GruParameters,
        previous: &Vector,
        input: &Vector,
    ) -> Result<Vector, GruError> {
        if input.len() != parameters.w_z.cols() {
            return Err(GruError::DimensionMismatch {
                expected: parameters.w_z.cols(),
                actual: input.len(),
            });
        }
        if previous.len() != parameters.u_z.rows() {
            return Err(GruError::DimensionMismatch {
                expected: parameters.u_z.rows(),
                actual: previous.len(),
            });
        }
        if !input.as_slice().iter().all(|value| value.is_finite()) {
            return Err(GruError::NonFiniteInput);
        }

        let update_gate = parameters
            .w_z
            .mul_vector(input)?
            .add(&parameters.u_z.mul_vector(previous)?)?
            .add(&parameters.b_z)?
            .map(sigmoid);
        let reset_gate = parameters
            .w_r
            .mul_vector(input)?
            .add(&parameters.u_r.mul_vector(previous)?)?
            .add(&parameters.b_r)?
            .map(sigmoid);
        let candidate = parameters
            .w_h
            .mul_vector(input)?
            .add(&parameters.u_h.mul_vector(&reset_gate.multiply(previous)?)?)?
            .add(&parameters.b_h)?
            .map(tanh);

        let next = update_gate
            .complement()
            .multiply(previous)?
            .add(&update_gate.multiply(&candidate)?)?;

        if !next.as_slice().iter().all(|value| value.is_finite()) {
            return Err(GruError::NonFiniteCandidate);
        }
        Ok(next)
    }
}

impl StateModel for Gru {
    type State = Vector;
    type Observation = Observation<Vector>;
    type Error = GruError;

    fn update(
        &self,
        state: &Vector,
        observation: &Observation<Vector>,
    ) -> Result<Vector, GruError> {
        Self::compute(&self.parameters, state, observation.value())
    }
}

/// Algorithm-local optimized execution context for a [`Gru`].
///
/// This is execution machinery, not a generic execution framework: it borrows
/// the model immutably, owns exactly one authoritative hidden `State`, and owns
/// a private reusable `GruWorkspace`. It is the only entry point to the
/// allocation-free optimized path.
pub struct GruExecutor<'m> {
    model: &'m Gru,
    state: Vector,
    workspace: GruWorkspace,
}

impl<'m> GruExecutor<'m> {
    /// Creates an executor over `model`, starting from `initial`.
    ///
    /// The initial state length is validated by the first
    /// [`GruExecutor::process_one_optimized`] call, mirroring the generic
    /// executor's per-step validation.
    pub fn new(model: &'m Gru, initial: Vector) -> Self {
        Self {
            model,
            state: initial,
            workspace: GruWorkspace::new(model.hidden_dim),
        }
    }

    /// The current committed hidden state.
    pub fn state(&self) -> &Vector {
        &self.state
    }

    /// Replaces the committed hidden state, starting a new sequence.
    ///
    /// The reusable workspace is retained: its buffers are fully overwritten on
    /// the next step, so they need no clearing.
    pub fn reset(&mut self, initial: Vector) {
        self.state = initial;
    }

    /// Advances the committed hidden state using the reusable workspace.
    ///
    /// Allocation-free in steady state and equivalent to the reference
    /// [`StateModel::update`] path; the test suite verifies bitwise agreement.
    /// On failure the committed state is unchanged and the workspace remains
    /// usable.
    pub fn process_one_optimized(
        &mut self,
        observation: &Observation<Vector>,
    ) -> Result<&Vector, GruError> {
        compute_in_place(
            &self.model.parameters,
            &mut self.workspace,
            &mut self.state,
            observation,
        )?;
        Ok(&self.state)
    }
}

fn validate_matrix(matrix: &Matrix, rows: usize, cols: usize) -> Result<(), GruError> {
    if matrix.rows() != rows {
        return Err(GruError::DimensionMismatch {
            expected: rows,
            actual: matrix.rows(),
        });
    }
    if matrix.cols() != cols {
        return Err(GruError::DimensionMismatch {
            expected: cols,
            actual: matrix.cols(),
        });
    }
    if !matrix.as_slice().iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteParameter);
    }
    Ok(())
}

fn validate_vector(vector: &Vector, len: usize) -> Result<(), GruError> {
    if vector.len() != len {
        return Err(GruError::DimensionMismatch {
            expected: len,
            actual: vector.len(),
        });
    }
    if !vector.as_slice().iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteParameter);
    }
    Ok(())
}

/// Computes one GRU step into `state` using only reusable workspace storage.
///
/// The operation order matches [`Gru::compute`] exactly, so the result is
/// bit-identical to the reference path. `state` is read as the previous hidden
/// state and overwritten only after the candidate is validated.
fn compute_in_place(
    parameters: &GruParameters,
    workspace: &mut GruWorkspace,
    state: &mut Vector,
    observation: &Observation<Vector>,
) -> Result<(), GruError> {
    let input = observation.value();
    if input.len() != parameters.w_z.cols() {
        return Err(GruError::DimensionMismatch {
            expected: parameters.w_z.cols(),
            actual: input.len(),
        });
    }
    if state.len() != parameters.u_z.rows() {
        return Err(GruError::DimensionMismatch {
            expected: parameters.u_z.rows(),
            actual: state.len(),
        });
    }
    if !input.as_slice().iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteInput);
    }

    {
        let h = state.as_slice();
        let x = input.as_slice();

        // acc = sigmoid(W_z x + U_z h + b_z)
        matvec_into(&parameters.w_z, x, &mut workspace.acc);
        matvec_into(&parameters.u_z, h, &mut workspace.scratch);
        add_assign(&mut workspace.acc, &workspace.scratch);
        add_assign(&mut workspace.acc, parameters.b_z.as_slice());
        sigmoid_assign(&mut workspace.acc);

        // gate = sigmoid(W_r x + U_r h + b_r)
        matvec_into(&parameters.w_r, x, &mut workspace.gate);
        matvec_into(&parameters.u_r, h, &mut workspace.scratch);
        add_assign(&mut workspace.gate, &workspace.scratch);
        add_assign(&mut workspace.gate, parameters.b_r.as_slice());
        sigmoid_assign(&mut workspace.gate);

        // gate = tanh(W_h x + U_h (gate ⊙ h) + b_h)
        mul_assign(&mut workspace.gate, h);
        matvec_into(&parameters.u_h, &workspace.gate, &mut workspace.scratch);
        matvec_into(&parameters.w_h, x, &mut workspace.gate);
        add_assign(&mut workspace.gate, &workspace.scratch);
        add_assign(&mut workspace.gate, parameters.b_h.as_slice());
        tanh_assign(&mut workspace.gate);

        // scratch = z_t ⊙ h̃_t
        for (scratch, (z, candidate)) in workspace
            .scratch
            .iter_mut()
            .zip(workspace.acc.iter().zip(workspace.gate.iter()))
        {
            *scratch = *z * *candidate;
        }
        // acc = (1 − z_t) ⊙ h + z_t ⊙ h̃_t
        for value in workspace.acc.iter_mut() {
            *value = 1.0 - *value;
        }
        mul_assign(&mut workspace.acc, h);
        add_assign(&mut workspace.acc, &workspace.scratch);
    }

    if !workspace.acc.iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteCandidate);
    }
    state.as_mut_slice().copy_from_slice(&workspace.acc);
    Ok(())
}

/// Writes `matrix · input` into `out` with the same per-row accumulation order
/// as [`Matrix::mul_vector`].
fn matvec_into(matrix: &Matrix, input: &[f32], out: &mut [f32]) {
    let cols = matrix.cols();
    let data = matrix.as_slice();
    for (row, out_row) in out.iter_mut().enumerate() {
        let start = row * cols;
        let mut sum = 0.0_f32;
        for (weight, value) in data[start..start + cols].iter().zip(input) {
            sum += weight * value;
        }
        *out_row = sum;
    }
}

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
}

fn mul_assign(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst *= *src;
    }
}

fn sigmoid_assign(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = sigmoid(*value);
    }
}

fn tanh_assign(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = tanh(*value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::numerical::Vector;
    use crate::foundation::observation::Observation;

    /// `reset` replaces the committed State and must reuse the executor's
    /// existing private workspace buffers rather than reallocating them.
    ///
    /// The workspace is private and not observable through the public API, so
    /// this private test checks buffer identity directly: the pointers and
    /// capacities of all three scratch buffers must be unchanged after a
    /// successful step, a reset, and a further successful step.
    #[test]
    fn reset_retains_workspace_buffers() {
        let model = Gru::new(2, 2, GruParameters::deterministic(2, 2)).unwrap();
        let mut executor = GruExecutor::new(&model, Vector::zeros(2));

        let identity = |executor: &GruExecutor<'_>| {
            let workspace = &executor.workspace;
            (
                workspace.acc.as_ptr(),
                workspace.gate.as_ptr(),
                workspace.scratch.as_ptr(),
                workspace.acc.capacity(),
                workspace.gate.capacity(),
                workspace.scratch.capacity(),
            )
        };

        let before = identity(&executor);
        executor
            .process_one_optimized(&Observation::new(Vector::from_slice(&[0.1, 0.2])))
            .unwrap();
        executor.reset(Vector::zeros(2));
        executor
            .process_one_optimized(&Observation::new(Vector::from_slice(&[0.3, -0.4])))
            .unwrap();

        assert_eq!(
            before,
            identity(&executor),
            "reset must reuse the existing workspace buffers"
        );
    }
}
