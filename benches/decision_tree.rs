//! Decision-tree prediction benchmarks: reference / streaming / model-local
//! micro-batch, with allocations and bytes.
//!
//! Run with `cargo bench --bench decision_tree`.
//!
//! Decision-tree inference is expected to be `O(depth)` and allocation-free
//! (one construction allocation). These baselines are descriptive; no
//! optimization is implied and none is implemented.

use std::hint::black_box;
use std::mem::size_of;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::classic::decision_tree::{DecisionTree, DecisionTreeNode};

#[allow(dead_code)]
mod common;

/// A balanced tree of the given depth over `features` features; leaves hold
/// deterministic finite values. Depth `d` has `2^(d+1) - 1` nodes.
fn balanced_tree(features: usize, depth: usize) -> DecisionTree {
    fn build(
        nodes: &mut Vec<DecisionTreeNode>,
        depth: usize,
        level: usize,
        features: usize,
    ) -> usize {
        let index = nodes.len();
        if depth == 0 {
            nodes.push(DecisionTreeNode::Leaf {
                value: (index as f32 * 0.25).sin(),
            });
            return index;
        }
        nodes.push(DecisionTreeNode::Split {
            feature_index: level % features,
            threshold: 0.0,
            left: 0,
            right: 0,
        });
        let left = build(nodes, depth - 1, level + 1, features);
        let right = build(nodes, depth - 1, level + 1, features);
        nodes[index] = DecisionTreeNode::Split {
            feature_index: level % features,
            threshold: 0.0,
            left,
            right,
        };
        index
    }

    let mut nodes = Vec::new();
    build(&mut nodes, depth, 0, features);
    DecisionTree::new(features, nodes).unwrap()
}

fn observation(features: usize) -> Observation<Vector> {
    Observation::new(Vector::from_fn(features, |index| {
        (index as f32 * 0.31).sin()
    }))
}

fn control_steps(features: usize) -> u32 {
    common::timed_steps(match features {
        8 | 32 => 200_000,
        128 => 50_000,
        _ => 20_000,
    })
}

/// Batch sizes swept within each fixed model for batch-size sensitivity (#24).
const SWEEP_BATCH_SIZES: [usize; 4] = [1, 8, 32, 128];

/// Observation budget per `(configuration, batch size)` window in release.
const SWEEP_BUDGET: u32 = 128_000;

/// Timed calls per batch size under the debug test profile.
const DEBUG_SWEEP_STEPS: u32 = 10;

/// Batch sizes used by the sensitivity sweep; the debug subset stays small.
fn sweep_batch_sizes() -> &'static [usize] {
    if cfg!(debug_assertions) {
        &SWEEP_BATCH_SIZES[..2]
    } else {
        &SWEEP_BATCH_SIZES
    }
}

/// Timed calls for one sweep window. Debug sets the count directly (10 calls)
/// and does not route through `common::timed_steps`; release uses `BUDGET / B`.
fn sweep_steps(batch_size: usize) -> u32 {
    if cfg!(debug_assertions) {
        DEBUG_SWEEP_STEPS
    } else {
        (SWEEP_BUDGET / batch_size as u32).max(1)
    }
}

/// The sweep exercises the smaller trees in the debug test profile.
fn sweep_enabled(features: usize) -> bool {
    !cfg!(debug_assertions) || features <= 128
}

/// Correctness precheck outside timing: `predict_batch` must be bitwise-equal to
/// repeated single-observation prediction. The returned output `Vec`'s capacity
/// is captured here, not inside the timed closure.
fn verify_sweep_batch(
    model: &DecisionTree,
    fixture: &[Observation<Vector>],
    batch_size: usize,
) -> usize {
    let batch = &fixture[..batch_size];
    let batched = model.predict_batch(batch).unwrap();
    for (index, observation) in batch.iter().enumerate() {
        let single = model.predict(observation.value()).unwrap();
        assert_eq!(
            batched[index].to_bits(),
            single.to_bits(),
            "predict_batch diverged from predict at B={batch_size}, index={index}"
        );
    }
    batched.capacity()
}

fn main() {
    println!(
        "Decision-tree prediction benchmark ({} runs/measurement)",
        common::RUNS
    );
    println!("{}", common::env_summary());

    for (features, depth, micro_batch) in [
        (8_usize, 6_usize, 32_usize),
        (32, 8, 32),
        (128, 10, 16),
        (256, 10, 8),
    ] {
        let steps = control_steps(features);
        println!(
            "features={features} depth={depth} nodes={} micro_batch={micro_batch} steps={steps}",
            (1_usize << (depth + 1)) - 1
        );

        let model = balanced_tree(features, depth);
        let observation = observation(features);
        let batch = vec![observation.clone(); micro_batch];

        common::measure_startup("init", common::startup_reps(), || {
            black_box(balanced_tree(features, depth));
        });

        common::measure("reference", steps, 1, || {
            black_box(model.predict(black_box(observation.value())).unwrap());
        });

        let streaming_model = balanced_tree(features, depth);
        let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
        common::measure("streaming", steps, 1, || {
            black_box(executor.process_one(black_box(&observation)).unwrap());
        });

        common::measure("micro-batch", steps, micro_batch, || {
            black_box(model.predict_batch(black_box(&batch)).unwrap());
        });

        if sweep_enabled(features) {
            // Fixed repeated-input fixture: one deterministic observation at every
            // position, so traversal workload is held constant across B. This
            // measures repeated-input behavior, not input-distribution performance.
            let fixture_observation = observation.clone();
            let fixture_length = SWEEP_BATCH_SIZES[SWEEP_BATCH_SIZES.len() - 1];
            let fixture = vec![fixture_observation; fixture_length];
            println!(
                "  batch-size sweep: stats bytes/obs is allocator-counted allocation/reallocation \
                 traffic (not live/peak memory); ns and bytes per batch below are derived"
            );
            println!(
                "  p95/IQR describe variability across normalized measurement windows, not \
                 individual-call or event latency. Debug measurements are not performance \
                 evidence; under RUNS=1, p95 equals the median and IQR is zero"
            );
            for &batch_size in sweep_batch_sizes() {
                let capacity = verify_sweep_batch(&model, &fixture, batch_size);
                let steps = sweep_steps(batch_size);
                let stats = common::measure("sweep micro-batch", steps, batch_size, || {
                    black_box(
                        model
                            .predict_batch(black_box(&fixture[..batch_size]))
                            .unwrap(),
                    );
                });
                let ns_per_batch = stats.median * batch_size as f64;
                let bytes_per_batch = stats.bytes * batch_size as f64;
                println!(
                    "    B={batch_size:>3} capacity={capacity:>4} logical_payload_bytes={:>4} \
                     derived_ns/batch={ns_per_batch:>12.1} derived_bytes/batch={bytes_per_batch:>12.1}",
                    batch_size * size_of::<f32>()
                );
            }
        }
    }
}
