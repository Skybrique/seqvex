//! Per-step GRU benchmarks comparing the two agreed execution contexts: the
//! reference `StreamingExecutor` (generic, via the GRU's `StateModel`) and the
//! algorithm-local optimized `GruExecutor` (`process_one_optimized`).
//!
//! Run with `cargo bench --bench gru`.
//!
//! Uses the shared dependency-free harness in `benches/common` (counting
//! allocator, `Instant`, median/IQR). Both contexts borrow the same immutable
//! `&Gru`, start from the same zero hidden state, and process the same
//! observations, so the measured comparison is executor-to-executor
//! per-observation work. Construction/initialization costs, including a full
//! parameter clone for model construction, are reported separately and are not
//! mixed into the per-observation numbers.
//!
//! The `GruParameters::deterministic` fixture is a test/benchmark fixture only:
//! it is not production initialization, and nothing here claims that production
//! randomness or initialization is solved (see issue #20).

use std::hint::black_box;
use std::mem::size_of;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::recurrent::gru::{Gru, GruExecutor, GruParameters};

#[allow(dead_code)]
mod common;

/// Workspace scratch buffers on the optimized path (three `H`-sized buffers).
const WORKSPACE_BUFFERS: usize = 3;

/// Prefix length used to confirm both execution contexts advance an equal
/// hidden state before the timed comparison.
const EQUIVALENCE_PREFIX: usize = 16;

fn parameter_bytes(parameters: &GruParameters) -> usize {
    let matrices = [
        &parameters.w_z,
        &parameters.u_z,
        &parameters.w_r,
        &parameters.u_r,
        &parameters.w_h,
        &parameters.u_h,
    ];
    let biases = [&parameters.b_z, &parameters.b_r, &parameters.b_h];
    let elements = matrices.iter().map(|m| m.as_slice().len()).sum::<usize>()
        + biases.iter().map(|b| b.len()).sum::<usize>();
    elements * size_of::<f32>()
}

/// Confirms both execution contexts start from the same zero hidden state and
/// stay bitwise identical over a short deterministic prefix of the same
/// observation. This is a fairness check for the benchmark, not a correctness
/// proof; the test suite owns reference/optimized equivalence.
fn verify_paths_agree(model: &Gru, observation: &Observation<Vector>, hidden_dim: usize) {
    let mut reference = StreamingExecutor::new(model, Vector::zeros(hidden_dim));
    let mut optimized = GruExecutor::new(model, Vector::zeros(hidden_dim));
    for _ in 0..EQUIVALENCE_PREFIX {
        let reference_state = reference.process_one(observation).unwrap();
        let optimized_state = optimized.process_one_optimized(observation).unwrap();
        assert!(
            reference_state
                .as_slice()
                .iter()
                .zip(optimized_state.as_slice())
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "reference StreamingExecutor and optimized GruExecutor diverged"
        );
    }
}

fn main() {
    println!(
        "GRU per-step benchmark ({} runs/measurement)\n",
        common::RUNS
    );
    println!("{}\n", common::env_summary());
    println!(
        "paths: \"ref StreamingExecutor\" = StreamingExecutor::process_one (StateModel); \
         \"opt GruExecutor\" = GruExecutor::process_one_optimized\n"
    );

    for (input_dim, hidden_dim, steps) in [
        (8_usize, 16_usize, 100_000_u32),
        (32, 64, 50_000),
        (128, 256, 5_000),
        (256, 512, 2_000),
    ] {
        let steps = common::timed_steps(steps);
        println!("input={input_dim} hidden={hidden_dim} steps={steps}");
        let observation = Observation::new(Vector::from_fn(input_dim, |i| (i as f32 * 0.31).sin()));

        let parameters = GruParameters::deterministic(input_dim, hidden_dim);
        let model = Gru::new(input_dim, hidden_dim, parameters.clone()).unwrap();

        // Fairness check: both contexts start at zero and advance identically.
        verify_paths_agree(&model, &observation, hidden_dim);

        // Reference execution context: generic StreamingExecutor over the GRU's
        // `StateModel` implementation.
        let mut reference = StreamingExecutor::new(&model, Vector::zeros(hidden_dim));
        common::measure("ref StreamingExecutor", steps, 1, || {
            black_box(reference.process_one(black_box(&observation)).unwrap());
        });

        // Optimized execution context: algorithm-local GruExecutor owning its
        // private per-execution workspace.
        let mut optimized = GruExecutor::new(&model, Vector::zeros(hidden_dim));
        common::measure("opt GruExecutor", steps, 1, || {
            black_box(
                optimized
                    .process_one_optimized(black_box(&observation))
                    .unwrap(),
            );
        });

        // Construction/initialization, reported separately from per-observation
        // work and scoped precisely:
        // - `params-clone`: one full `GruParameters` clone (allocates the
        //   parameter tensors).
        // - `params-clone+model-init`: that clone plus `Gru::new`, which
        //   validates the parameters and moves them (allocating nothing itself).
        // - `streaming-init`: `StreamingExecutor::new` plus the initial-state
        //   `Vector::zeros`.
        // - `gru-executor-init`: `GruExecutor::new` plus its private workspace.
        // Parameter generation (`GruParameters::deterministic`) happens once
        // before this block and is not part of any measurement here.
        common::measure_startup("params-clone", common::startup_reps(), || {
            black_box(parameters.clone());
        });
        common::measure_startup("params-clone+model-init", common::startup_reps(), || {
            black_box(Gru::new(input_dim, hidden_dim, parameters.clone()).unwrap());
        });
        common::measure_startup("streaming-init", common::startup_reps(), || {
            black_box(StreamingExecutor::new(&model, Vector::zeros(hidden_dim)));
        });
        common::measure_startup("gru-executor-init", common::startup_reps(), || {
            black_box(GruExecutor::new(&model, Vector::zeros(hidden_dim)));
        });

        println!(
            "    hidden footprint {} bytes; workspace scratch {} bytes; parameters {} bytes\n",
            hidden_dim * size_of::<f32>(),
            WORKSPACE_BUFFERS * hidden_dim * size_of::<f32>(),
            parameter_bytes(&parameters),
        );
    }
}
