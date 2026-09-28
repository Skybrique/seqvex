//! Decision-tree prediction benchmarks: reference / streaming / model-local
//! micro-batch, with allocations and bytes.
//!
//! Run with `cargo bench --bench decision_tree`.
//!
//! Decision-tree inference is expected to be `O(depth)` and allocation-free
//! (one construction allocation). These baselines are descriptive; no
//! optimization is implied and none is implemented.

use std::hint::black_box;

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
    }
}
