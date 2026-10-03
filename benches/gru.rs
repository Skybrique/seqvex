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
//! This file also prepares a bounded reference-batch comparison (#34, unit B):
//! grouped `StreamingExecutor` steps, `Gru::process_batch_reference`, and the
//! foundation `process_batch` ordered fold. Correctness prechecks run outside
//! timing; each path is measured once per batch size with `work_per_call = B`.
//! The foundation fold is a labelled control whose timed body includes the
//! observation clone it requires. `(256,512)` is out of scope for the bounded
//! comparison. Release evidence is pending.
//!
//! The `GruParameters::deterministic` fixture is a test/benchmark fixture only:
//! it is not production initialization, and nothing here claims that production
//! randomness or initialization is solved (see issue #20).

use std::hint::black_box;
use std::mem::size_of;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_batch;
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

/// Largest batch size in the bounded reference comparison.
const MAX_BATCH_SIZE: usize = 128;

/// Batch sizes compared for bounded reference batching (#34, unit B).
const BOUNDED_BATCH_SIZES: [usize; 4] = [1, 8, 32, 128];

/// Provisional per-window observation budget by dimension in release. These are
/// workload limits, not performance evidence. Each value divides exactly by
/// every batch size in [`BOUNDED_BATCH_SIZES`], so `calls = budget / B` keeps the
/// observation count equal across paths and batch sizes within a dimension.
fn bounded_window_budget(input_dim: usize, hidden_dim: usize) -> u32 {
    match (input_dim, hidden_dim) {
        (8, 16) => 8_192,
        (32, 64) => 2_048,
        (128, 256) => 256,
        _ => 256,
    }
}

/// Consecutive-call depth of the pre-timing bitwise equivalence check.
fn precheck_calls() -> usize {
    if cfg!(debug_assertions) { 2 } else { 4 }
}

/// Batch sizes used by the bounded comparison. The two largest are skipped in
/// debug so that `cargo test --all-targets` — which executes this binary — stays
/// cheap; the bounded block multiplies `steps` by `B`, and the largest
/// dimension's reference transition is expensive.
fn bounded_batch_sizes() -> &'static [usize] {
    if cfg!(debug_assertions) {
        &BOUNDED_BATCH_SIZES[..2]
    } else {
        &BOUNDED_BATCH_SIZES
    }
}

/// Calls per measurement for the bounded block. Debug is deliberately tiny and
/// is not evidence. In release, `calls = window_budget / B`, so every `B`
/// processes the same number of observations within a dimension.
fn bounded_steps(batch_size: usize, window_budget: u32) -> u32 {
    if cfg!(debug_assertions) {
        10
    } else {
        (window_budget / batch_size as u32).max(1)
    }
}

/// Deterministic ordered observations shared by every path for one dimension
/// pair. The prefix `[..B]` is used for batch size `B`, so smaller batches are
/// nested inside larger ones and all paths see identical inputs.
fn bounded_observations(input_dim: usize, seed: u64) -> Vec<Observation<Vector>> {
    let mut lcg = common::Lcg::new(seed);
    (0..MAX_BATCH_SIZE)
        .map(|_| Observation::new(Vector::from_fn(input_dim, |_| lcg.next_f32())))
        .collect()
}

fn bitwise_same(left: &Vector, right: &Vector) -> bool {
    left.as_slice().len() == right.as_slice().len()
        && left
            .as_slice()
            .iter()
            .zip(right.as_slice())
            .all(|(l, r)| l.to_bits() == r.to_bits())
}

/// Confirms, outside any timing region, that the grouped reference executor,
/// the bounded reference batch, and the foundation ordered fold stay bitwise
/// identical across several consecutive calls over the same `[..batch_size]`
/// prefix. Continuity (state carried between calls) is part of what is checked.
fn verify_bounded_equivalence(
    model: &Gru,
    observations: &[Observation<Vector>],
    batch_size: usize,
    hidden_dim: usize,
    calls: usize,
) {
    let batch = &observations[..batch_size];
    let mut grouped = StreamingExecutor::new(model, Vector::zeros(hidden_dim));
    let mut carried_batch = Some(Vector::zeros(hidden_dim));
    let mut carried_fold = Some(Vector::zeros(hidden_dim));

    for _ in 0..calls {
        for observation in batch {
            grouped.process_one(observation).unwrap();
        }

        let state = carried_batch.take().expect("bounded batch state available");
        carried_batch = Some(
            model
                .process_batch_reference(state, batch, batch_size)
                .expect("validated benchmark fixture"),
        );

        let state = carried_fold.take().expect("fold state available");
        carried_fold = Some(
            process_batch(model, state, batch.iter().cloned())
                .expect("validated benchmark fixture"),
        );

        assert!(
            bitwise_same(grouped.state(), carried_batch.as_ref().unwrap()),
            "grouped StreamingExecutor and bounded reference batch diverged"
        );
        assert!(
            bitwise_same(grouped.state(), carried_fold.as_ref().unwrap()),
            "grouped StreamingExecutor and foundation process_batch diverged"
        );
    }
}

/// Prints the derived amortized per-batch duration for one path. This reverses
/// the helper's per-observation normalization of the same measurement window;
/// it is not an individual-call latency percentile and not per-event latency.
fn report_amortized_batch(label: &str, stats: common::Stats, batch_size: usize) {
    let ns_per_batch = stats.median * batch_size as f64;
    let batches_per_second = if ns_per_batch > 0.0 {
        1e9 / ns_per_batch
    } else {
        0.0
    };
    println!(
        "    {label:<24} derived amortized {ns_per_batch:>10.1} ns/batch  \
         {batches_per_second:>10.0} batches/s  (window derived; not per-call latency)"
    );
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

        // --- bounded reference batching (#34, unit B) -------------------------
        //
        // One measurement pass per path/batch-size/dimension with
        // `work_per_call = B`; `ns/obs` comes from the shared harness and the
        // amortized `ns/batch` is derived by reversing the same normalization.
        // `p95`/`IQR` are variability across the repeated measurement windows,
        // not per-call latency percentiles. `foundation process_batch` is a
        // labelled control: it consumes owned observations, so its timed body
        // includes the `iter().cloned()` copy and is not a batching-speedup
        // claim. `(256,512)` is excluded in both profiles as the optional
        // dimension; the existing per-step benchmarks above still cover it.
        //
        // Continuity: each path builds a fresh zero State before measurement and
        // then carries it through the harness warm-up and across the successive
        // timed windows (that is how `measure` works); no window resets it. For a
        // given dimension and `B`, all three paths run the same number of calls,
        // so they receive equivalent processing histories. Release uses the
        // per-dimension window budgets from `bounded_window_budget` as workload
        // limits (not evidence); the timed totals exclude the harness warm-up and
        // the pre-timing correctness prechecks, which are accounted separately.
        // Debug keeps a reduced batch-size set and a tiny step count so the
        // all-targets test run stays cheap.
        if hidden_dim > 256 {
            continue;
        }
        let observations =
            bounded_observations(input_dim, 0x34ba_0000 + (input_dim * hidden_dim) as u64);
        let window_budget = bounded_window_budget(input_dim, hidden_dim);
        let calls = precheck_calls();
        for &batch_size in bounded_batch_sizes() {
            verify_bounded_equivalence(&model, &observations, batch_size, hidden_dim, calls);
            let batch_steps = bounded_steps(batch_size, window_budget);
            let batch = &observations[..batch_size];
            println!(
                "bounded batch B={batch_size} dims={input_dim}x{hidden_dim} \
                 steps={batch_steps} precheck_calls={calls}"
            );

            let mut grouped = StreamingExecutor::new(&model, Vector::zeros(hidden_dim));
            let grouped_stats =
                common::measure("bounded grouped exec", batch_steps, batch_size, || {
                    for observation in batch {
                        black_box(grouped.process_one(black_box(observation)).unwrap());
                    }
                });

            let mut carried = Some(Vector::zeros(hidden_dim));
            let batch_stats = common::measure("bounded ref batch", batch_steps, batch_size, || {
                let state = carried.take().expect("benchmark state available");
                let next = model
                    .process_batch_reference(state, batch, batch_size)
                    .expect("validated benchmark fixture");
                carried = Some(black_box(next));
            });

            let mut carried = Some(Vector::zeros(hidden_dim));
            let fold_stats =
                common::measure("foundation process_batch", batch_steps, batch_size, || {
                    let state = carried.take().expect("benchmark state available");
                    let next = process_batch(&model, state, batch.iter().cloned())
                        .expect("validated benchmark fixture");
                    carried = Some(black_box(next));
                });

            report_amortized_batch("bounded grouped exec", grouped_stats, batch_size);
            report_amortized_batch("bounded ref batch", batch_stats, batch_size);
            report_amortized_batch("foundation process_batch", fold_stats, batch_size);
        }
    }
}
