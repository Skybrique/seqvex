//! Linear-regression prediction benchmarks: control (fixed observation) and
//! decision-grade sequential workloads A/B/C, with allocations and bytes.
//!
//! Run with `cargo bench --bench linear_regression`.
//!
//! LR is the profiling control: effectively stateless prediction expected to be
//! `O(D)` and allocation-free. The decision-grade workload matrix is executed in
//! the optimized profile only; the debug test profile runs the existing control
//! and `verify_workloads()` so `cargo test --all-targets` stays usable, and its
//! numbers are never used for decisions.

use std::hint::black_box;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_one;
use seqvex::models::classic::LinearRegression;

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

/// Existing fixed-observation control model (unchanged formula).
fn model(features: usize) -> LinearRegression {
    LinearRegression::new(
        Vector::from_fn(features, |index| (index as f32 * 0.37).sin()),
        0.125,
    )
    .unwrap()
}

/// Workload model whose weights are the deterministic `w*`, so the sequential
/// workload's predictions are meaningful.
fn workload_model(features: usize) -> LinearRegression {
    LinearRegression::new(Vector::from_slice(&common::fixed_weights(features)), 0.125).unwrap()
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
        "Linear-regression prediction benchmark ({} runs/measurement)",
        common::RUNS
    );
    common::verify_workloads();
    println!("{}", common::env_summary());

    // Control: existing fixed observation, reference / streaming / micro-batch.
    println!("\n== control (fixed observation) ==");
    for (features, micro_batch) in [(8_usize, 32_usize), (32, 32), (128, 16), (256, 8)] {
        let steps = control_steps(features);
        println!("features={features} micro_batch={micro_batch} steps={steps}");

        let observation = Observation::new(Vector::from_fn(features, |i| (i as f32 * 0.31).sin()));
        let batch = vec![observation.clone(); micro_batch];

        let reference = model(features);
        common::measure("reference", steps, 1, || {
            black_box(reference.predict(black_box(observation.value())).unwrap());
        });

        let streaming_model = model(features);
        let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
        common::measure("streaming", steps, 1, || {
            black_box(executor.process_one(black_box(&observation)).unwrap());
        });

        let micro_batch_model = model(features);
        common::measure("micro-batch", steps, micro_batch, || {
            black_box(micro_batch_model.predict_batch(black_box(&batch)).unwrap());
        });
    }

    if cfg!(debug_assertions) {
        println!("\ndecision-grade workload matrix skipped in the debug profile");
        return;
    }

    // Decision-grade workloads A/B/C over the required dimensions.
    let dimensions = [8_usize, 32, 128, 256];
    for workload in ["A", "B", "C"] {
        println!("\n== workload {workload} ==");
        let mut scaling: Vec<(f64, f64)> = Vec::new();
        for &features in &dimensions {
            let steps = control_steps(features);
            let xs = match workload {
                "A" => common::persistent_excitation_features(features, 2048),
                "B" => common::structured_features(
                    features,
                    2048,
                    common::AR_COEFFICIENT,
                    common::STRUCTURED_SEED,
                ),
                _ => common::persistent_excitation_features(features, steps as usize),
            };
            let observations: Vec<Observation<Vector>> = xs
                .iter()
                .map(|x| Observation::new(Vector::from_slice(x)))
                .collect();
            println!(
                "LR workload={workload} features={features} steps={steps} stream={}",
                observations.len()
            );

            common::measure_startup("init", common::startup_reps(), || {
                black_box(workload_model(features));
            });

            let direct = workload_model(features);
            let mut index = 0_usize;
            let stats = common::measure("reference", steps, 1, || {
                let observation = &observations[index % observations.len()];
                index += 1;
                black_box(direct.predict(black_box(observation.value())).unwrap());
            });

            let streaming_model = workload_model(features);
            let mut executor = StreamingExecutor::new(&streaming_model, 0.0);
            let mut stream_index = 0_usize;
            common::measure("streaming", steps, 1, || {
                let observation = &observations[stream_index % observations.len()];
                stream_index += 1;
                black_box(executor.process_one(black_box(observation)).unwrap());
            });

            if workload == "A" {
                // Model sharing through the existing foundation `&M` path: one
                // model with two independent states versus two models.
                let shared = workload_model(features);
                let mut state_a = 0.0_f32;
                let mut state_b = 0.0_f32;
                let mut shared_index = 0_usize;
                let shared_stats = common::measure("sharing-1model", steps, 1, || {
                    let observation = &observations[shared_index % observations.len()];
                    shared_index += 1;
                    state_a = process_one(&shared, &state_a, black_box(observation)).unwrap();
                    state_b = process_one(&shared, &state_b, black_box(observation)).unwrap();
                });
                let model_a = workload_model(features);
                let model_b = workload_model(features);
                let mut separate_a = 0.0_f32;
                let mut separate_b = 0.0_f32;
                let mut separate_index = 0_usize;
                let separate_stats = common::measure("sharing-2models", steps, 1, || {
                    let observation = &observations[separate_index % observations.len()];
                    separate_index += 1;
                    separate_a =
                        process_one(&model_a, &separate_a, black_box(observation)).unwrap();
                    separate_b =
                        process_one(&model_b, &separate_b, black_box(observation)).unwrap();
                });
                println!(
                    "    sharing delta={:.1} ns/obs -> {}",
                    (shared_stats.median - separate_stats.median).abs(),
                    common::materiality(
                        shared_stats.median,
                        shared_stats.iqr,
                        separate_stats.median,
                        separate_stats.iqr,
                    )
                );
            }

            if workload != "B" {
                scaling.push((features as f64, stats.median));
            }
        }
        if scaling.len() >= 2 {
            println!(
                "LR workload={workload} scaling exponent={:.3} (expected ~1.0)",
                common::log_log_slope(&scaling)
            );
        }
    }
}
