//! Per-step GRU benchmarks: latency, throughput, and allocation behavior for the
//! reference path and the algorithm-local optimized path.
//!
//! Run with `cargo bench --bench gru`.
//!
//! Uses the shared dependency-free harness in `benches/common` (counting
//! allocator, `Instant`, median/IQR). The former model-owned-workspace path no
//! longer exists: the optimized path is now [`GruExecutor`], which owns a private
//! per-execution workspace. Its historical `20 -> 0` steady-state allocation
//! result is not re-measured here; this benchmark confirms that executor-owned
//! scratch preserves zero steady-state allocations and reports reference vs
//! optimized latency. It asserts no performance target.

use std::hint::black_box;
use std::mem::size_of;

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::recurrent::gru::{Gru, GruExecutor, GruParameters};

#[allow(dead_code)]
mod common;

/// Workspace scratch buffers on the optimized path (three `H`-sized buffers).
const WORKSPACE_BUFFERS: usize = 3;

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

fn main() {
    println!(
        "GRU per-step benchmark ({} runs/measurement)\n",
        common::RUNS
    );
    println!("{}\n", common::env_summary());

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

        // Reference: value-returning `StateModel::update`, allocating per step.
        let mut reference_state = Vector::zeros(hidden_dim);
        common::measure("reference", steps, 1, || {
            reference_state = model
                .update(black_box(&reference_state), black_box(&observation))
                .unwrap();
        });

        // Optimized: executor-owned State plus a private per-execution workspace.
        let mut executor = GruExecutor::new(&model, Vector::zeros(hidden_dim));
        common::measure("optimized", steps, 1, || {
            black_box(
                executor
                    .process_one_optimized(black_box(&observation))
                    .unwrap(),
            );
        });

        // Construction cost: model parameters, then the executor and its
        // workspace allocation, reported separately from steady state.
        common::measure_startup("model-init", common::startup_reps(), || {
            black_box(Gru::new(input_dim, hidden_dim, parameters.clone()).unwrap());
        });
        common::measure_startup("executor-init", common::startup_reps(), || {
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
