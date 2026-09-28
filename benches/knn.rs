//! KNN prediction benchmarks: reference / streaming / model-local micro-batch,
//! plus a distance-scan-only control, with allocations and bytes.
//!
//! Run with `cargo bench --bench knn`.
//!
//! One factor is varied at a time across reference count `N`, dimension `d`,
//! neighbor count `k`, and bounded batch size `B`, so the distance scan and the
//! neighbor selection can be read separately. These baselines are descriptive;
//! no optimization is implied and none is implemented.

use std::hint::black_box;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::classic::knn::{Knn, Reference};

#[allow(dead_code)]
mod common;

/// `usize::MAX` marks the "k = N" configuration.
const K_ALL: usize = usize::MAX;

/// One factor at a time, sized so each measurement stays within the shared
/// harness's fixed warm-up budget (`1000` calls): large `N` uses `d = 8`, and
/// the batch-size sweep uses a small reference set so large `B` stays cheap.
const CONFIGS: &[(usize, usize, usize, usize)] = &[
    // N sweep (d = 8, k = 4, B = 1)
    (1_024, 8, 4, 1),
    (16_384, 8, 4, 1),
    (65_536, 8, 4, 1),
    (262_144, 8, 4, 1),
    // d sweep (N = 8_192, k = 4, B = 1)
    (8_192, 8, 4, 1),
    (8_192, 32, 4, 1),
    (8_192, 128, 4, 1),
    (8_192, 256, 4, 1),
    // k sweep (N = 8_192, d = 32, B = 1)
    (8_192, 32, 1, 1),
    (8_192, 32, 8, 1),
    (8_192, 32, 32, 1),
    (8_192, 32, K_ALL, 1),
    // B sweep (N = 2_048, d = 8, k = 4)
    (2_048, 8, 4, 1),
    (2_048, 8, 4, 8),
    (2_048, 8, 4, 32),
    (2_048, 8, 4, 128),
];

/// Debug builds shrink the reference set so `cargo test --all-targets` stays
/// fast; decision numbers are release-only.
fn scaled(n: usize) -> usize {
    if cfg!(debug_assertions) {
        (n / 64).max(32)
    } else {
        n
    }
}

/// Keeps each measurement to a bounded number of scans regardless of `N·d`.
fn steps_for(n: usize, d: usize) -> u32 {
    let target = 20_000_000_u64;
    let raw = (target / (n as u64 * d as u64)).clamp(50, 20_000) as u32;
    common::timed_steps(raw)
}

fn build_references(n: usize, d: usize) -> Vec<Reference> {
    let mut lcg = common::Lcg::new(0x5eed_2500_0000_0001);
    let mut references = Vec::with_capacity(n);
    for _ in 0..n {
        let features = Vector::from_fn(d, |_| 2.0 * lcg.next_f32());
        let target = features.as_slice().iter().sum::<f32>() / d as f32;
        references.push(Reference::new(features, target));
    }
    references
}

fn query(d: usize) -> Observation<Vector> {
    Observation::new(Vector::from_fn(d, |index| (index as f32 * 0.13).cos()))
}

fn batch(d: usize, size: usize) -> Vec<Observation<Vector>> {
    (0..size)
        .map(|step| {
            Observation::new(Vector::from_fn(d, |index| {
                ((step + index) as f32 * 0.07).sin()
            }))
        })
        .collect()
}

/// Distance-scan-only control: sums every squared distance without selecting or
/// sorting, so the reference path's selection cost can be read as the difference.
fn scan(references: &[Reference], query: &Vector) -> f32 {
    let values = query.as_slice();
    let mut total = 0.0_f32;
    for reference in references {
        let mut sum = 0.0_f32;
        for (query_value, reference_value) in values.iter().zip(reference.features.as_slice()) {
            let delta = query_value - reference_value;
            sum += delta * delta;
        }
        total += sum;
    }
    total
}

fn main() {
    println!(
        "KNN prediction benchmark ({} runs/measurement)",
        common::RUNS
    );
    println!("{}", common::env_summary());

    let init_reps = if cfg!(debug_assertions) { 1 } else { 5 };

    for &(n, d, k, b) in CONFIGS {
        let n = scaled(n);
        let k = if k == K_ALL { n } else { k.min(n) };
        let steps = steps_for(n, d);
        println!("N={n} d={d} k={k} B={b} steps={steps}");

        let model = Knn::new(d, k, build_references(n, d)).unwrap();
        let query = query(d);
        let batch = batch(d, b);

        common::measure_startup("init", init_reps, || {
            black_box(Knn::new(d, k, build_references(n, d)).unwrap());
        });

        common::measure("reference", steps, 1, || {
            black_box(model.predict(black_box(query.value())).unwrap());
        });

        common::measure("scan-only", steps, 1, || {
            black_box(scan(
                black_box(model.references()),
                black_box(query.value()),
            ));
        });

        let streaming_model = Knn::new(d, k, build_references(n, d)).unwrap();
        let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
        common::measure("streaming", steps, 1, || {
            black_box(executor.process_one(black_box(&query)).unwrap());
        });

        common::measure("micro-batch", steps, b, || {
            black_box(model.predict_batch(black_box(&batch)).unwrap());
        });
    }
}
