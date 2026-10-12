//! OLS fitter / checked-serving / artifact resource evidence (benchmark-only).
//!
//! Measures the current implementation on deterministic synthetic workloads.
//! This is evidence, not an optimization or a performance guarantee, and it is
//! Seqvex-only (no cross-library comparison).
//!
//! Run with:
//! `cargo +1.85.1 bench --locked --bench ols_fit`
//!
//! Memory is measured with the shared counting allocator's live/peak counters
//! (`benches/common`). Logical requested workspace (the fitter's budget) is
//! reported separately from allocator traffic, retained bytes, and peak bytes.

use std::hint::black_box;
use std::time::Instant;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    ArtifactExpectation, ArtifactLimits, CheckedLinearRegression, Expect, FitConfig, FitResult,
    PublicationId, TrainedModel, TrainingMeta, fit,
};

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

const FEATURE_SEED: u64 = 0x0051_0ee1_0000_0001;
const RESIDUAL_SCALE: f32 = 1.0e-3;

fn transaction_meta() -> TrainingMeta {
    TrainingMeta {
        schema_id: [7u8; 16],
        transform_link: None,
        cutoff: None,
        fit_revision: [1, 0, 0, 0, 0, 0, 0, 0],
    }
}

fn limits() -> ArtifactLimits {
    ArtifactLimits {
        max_features: 1024,
        max_bytes: 1 << 20,
    }
}

/// Deterministic dense features on `[-1, 1)` (fixed-seed LCG draws).
fn build_features(n: usize, d: usize) -> Vec<Vector> {
    let mut lcg = common::Lcg::new(FEATURE_SEED);
    (0..n)
        .map(|_| {
            let row: Vec<f32> = (0..d).map(|_| 2.0 * lcg.next_f32()).collect();
            Vector::from_slice(&row)
        })
        .collect()
}

/// `y_i = x_i · w* + 1e-3·sin(0.01·i)` (bounded deterministic residual).
fn build_targets(features: &[Vector], weights: &[f32]) -> Vec<f32> {
    features
        .iter()
        .enumerate()
        .map(|(index, x)| {
            let dot: f32 = x.as_slice().iter().zip(weights).map(|(a, b)| a * b).sum();
            dot + RESIDUAL_SCALE * ((index as f32) * 0.01).sin()
        })
        .collect()
}

struct Meas {
    median_ns: f64,
    min_ns: f64,
    max_ns: f64,
    allocs_per_op: f64,
    bytes_per_op: f64,
}

/// Two warm-up samples, then `samples` timed samples of `reps` operations each.
fn measure_ops(reps: usize, samples: usize, mut op: impl FnMut()) -> Meas {
    for _ in 0..2 {
        op();
    }
    // Sample storage is reserved *before* the snapshot so only `op()` allocations
    // are counted; `values.push` stays allocation-free and sorting/formatting run
    // after the closing snapshot.
    let mut values = Vec::with_capacity(samples);
    let (alloc_before, bytes_before) = common::allocation_totals();
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..reps {
            op();
        }
        values.push(start.elapsed().as_nanos() as f64 / reps as f64);
    }
    let (alloc_after, bytes_after) = common::allocation_totals();
    let operations = (reps * samples) as f64;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Meas {
        median_ns: values[values.len() / 2],
        min_ns: values[0],
        max_ns: values[values.len() - 1],
        allocs_per_op: (alloc_after - alloc_before) as f64 / operations,
        bytes_per_op: (bytes_after - bytes_before) as f64 / operations,
    }
}

#[derive(Debug, Clone, Copy)]
struct FitMemory {
    rank: usize,
    required_rank: usize,
    requested_bytes: usize,
    allocs_per_fit: f64,
    alloc_bytes_per_fit: f64,
    peak_add_bytes: i64,
    retained_bytes: i64,
    drop_delta_bytes: i64,
}

fn fit_once(features: &[Vector], targets: &[f32], train: &[usize]) -> FitResult {
    fit(features, targets, train, &FitConfig::default()).expect("fit succeeds")
}

fn main() {
    println!("OLS resource/performance evidence (synthetic workloads; Seqvex only)");
    println!("{}", common::env_summary());
    println!("warm-ups: 2 samples; measured samples: 10 samples (fit) / 20 (serving)");
    println!("fit workload: y = x·w* + 1e-3·sin(0.01·i); intercept enabled; all rows selected");

    let mut baseline = Vec::new();
    println!(
        "\n== fit timing & memory ==\n{:<11} {:>6} {:>9} {:>11} {:>11} {:>11} {:>9} {:>12} {:>12} {:>11} {:>12}",
        "N x d",
        "rank",
        "req_B",
        "median_us",
        "min_us",
        "max_us",
        "allocs",
        "alloc_B",
        "peak_add_B",
        "retain_B",
        "drop_B"
    );
    for &n in &[1024usize, 16384] {
        for &d in &[8usize, 32, 128] {
            let weights = common::fixed_weights(d);
            let features = build_features(n, d);
            let targets = build_targets(&features, &weights);
            let train: Vec<usize> = (0..n).collect();

            // Correctness before timing: the fit must succeed at full rank.
            let check = fit_once(&features, &targets, &train);
            let diagnostics = check.diagnostics();
            assert_eq!(diagnostics.rank, diagnostics.required_rank, "required rank");
            let requested_bytes = diagnostics.requested_workspace_bytes;
            drop(check);

            // Timing and allocator traffic (fixtures are created outside the window).
            let timing = measure_ops(1, 10, || {
                let result = fit_once(&features, &targets, &train);
                black_box(result);
            });

            // Memory: one fit observed with the live/peak counters.
            let live_before = common::live_bytes();
            common::reset_peak();
            let result = fit_once(&features, &targets, &train);
            let peak_during = common::peak_bytes();
            let live_with_result = common::live_bytes();
            let rank = result.diagnostics().rank;
            let required_rank = result.diagnostics().required_rank;
            drop(result);
            let live_after = common::live_bytes();

            let memory = FitMemory {
                rank,
                required_rank,
                requested_bytes,
                allocs_per_fit: timing.allocs_per_op,
                alloc_bytes_per_fit: timing.bytes_per_op,
                peak_add_bytes: peak_during as i64 - live_before as i64,
                retained_bytes: live_with_result as i64 - live_before as i64,
                drop_delta_bytes: live_after as i64 - live_before as i64,
            };
            println!(
                "{:<11} {:>6} {:>9} {:>11.1} {:>11.1} {:>11.1} {:>9.1} {:>12.0} {:>12} {:>11} {:>12}",
                format!("{n} x {d}"),
                format!("{}/{}", memory.rank, memory.required_rank),
                memory.requested_bytes,
                timing.median_ns / 1e3,
                timing.min_ns / 1e3,
                timing.max_ns / 1e3,
                memory.allocs_per_fit,
                memory.alloc_bytes_per_fit,
                memory.peak_add_bytes,
                memory.retained_bytes,
                memory.drop_delta_bytes,
            );
            baseline.push((n, d, memory, timing));
        }
    }

    // ---- checked serving ----
    println!(
        "\n== checked serving ==\n{:<9} {:>10} {:>8} {:>11} {:>11} {:>13} {:>8} {:>9} {:>9}",
        "path", "d", "amount", "median_ns", "min_ns", "ns/obs", "allocs", "bytes/op", "out_incl"
    );
    for &d in &[8usize, 128] {
        let n = 1024usize;
        let weights = common::fixed_weights(d);
        let features = build_features(n, d);
        let targets = build_targets(&features, &weights);
        let train: Vec<usize> = (0..n).collect();
        let model = fit_once(&features, &targets, &train).into_predictor();

        // Prebuilt observation and batches (outside measurement windows).
        let observation = Observation::new(features[0].clone());
        let single = measure_ops(1000, 20, || {
            let value = model
                .predict_checked(black_box(observation.value()))
                .expect("finite checked output");
            black_box(value);
        });
        assert!(
            model
                .predict_checked(observation.value())
                .unwrap()
                .is_finite()
        );
        println!(
            "{:<9} {:>10} {:>8} {:>11.1} {:>11.1} {:>13.1} {:>8.2} {:>9.2} {:>9}",
            "single",
            d,
            1,
            single.median_ns,
            single.min_ns,
            single.median_ns,
            single.allocs_per_op,
            single.bytes_per_op,
            "no"
        );

        let checked = CheckedLinearRegression::new(&model);
        let mut executor = StreamingExecutor::new(&checked, 0.0f32);
        let stream = measure_ops(1000, 20, || {
            let value = *executor
                .process_one(black_box(&observation))
                .expect("finite checked stream");
            black_box(value);
        });
        println!(
            "{:<9} {:>10} {:>8} {:>11.1} {:>11.1} {:>13.1} {:>8.2} {:>9.2} {:>9}",
            "stream",
            d,
            1,
            stream.median_ns,
            stream.min_ns,
            stream.median_ns,
            stream.allocs_per_op,
            stream.bytes_per_op,
            "no"
        );

        for &batch in &[1usize, 8, 32, 128] {
            let group: Vec<Observation<Vector>> = (0..batch)
                .map(|i| Observation::new(features[i % n].clone()))
                .collect();
            let reps = if batch >= 32 { 10 } else { 100 };
            let measured = measure_ops(reps, 20, || {
                let values = model
                    .predict_batch_checked(black_box(&group))
                    .expect("finite checked group");
                black_box(values);
            });
            println!(
                "{:<9} {:>10} {:>8} {:>11.1} {:>11.1} {:>13.1} {:>8.2} {:>9.2} {:>9}",
                "group",
                d,
                batch,
                measured.median_ns,
                measured.min_ns,
                measured.median_ns / batch as f64,
                measured.allocs_per_op,
                measured.bytes_per_op,
                "yes"
            );
        }

        // Correctness: grouped and streaming agree bitwise with single prediction.
        let reference: Vec<u32> = (0..8)
            .map(|i| model.predict_checked(&features[i]).unwrap().to_bits())
            .collect();
        let batch: Vec<Observation<Vector>> = (0..8)
            .map(|i| Observation::new(features[i].clone()))
            .collect();
        let grouped = model.predict_batch_checked(&batch).unwrap();
        for (i, value) in grouped.iter().enumerate() {
            assert_eq!(value.to_bits(), reference[i], "group vs single");
        }
    }

    // ---- artifact & refit ownership ----
    println!("\n== artifact & refit ownership (d=128, N=1024) ==");
    let d = 128usize;
    let n = 1024usize;
    let weights = common::fixed_weights(d);
    let features = build_features(n, d);
    let targets = build_targets(&features, &weights);
    let train: Vec<usize> = (0..n).collect();
    let meta = transaction_meta();

    let conversion = measure_ops(1, 10, || {
        let trained = fit_once(&features, &targets, &train)
            .into_trained(&meta)
            .expect("into_trained");
        black_box(trained);
    });
    let trained = fit_once(&features, &targets, &train)
        .into_trained(&meta)
        .expect("into_trained");
    println!(
        "fit + TrainedModel construction: median {:.1} us (min {:.1}, max {:.1}), {:.2} allocs/op, {:.1} bytes/op (construction adds no allocations beyond the fit; source-confirmed)",
        conversion.median_ns / 1e3,
        conversion.min_ns / 1e3,
        conversion.max_ns / 1e3,
        conversion.allocs_per_op,
        conversion.bytes_per_op
    );

    let publication: PublicationId = *b"ols-evidence-001";
    // Export buffer is pre-allocated outside the window (caller-owned; excluded).
    let mut buffer: Vec<u8> = Vec::with_capacity(1 << 12);
    let export = measure_ops(100, 20, || {
        buffer.clear();
        trained
            .export_to(black_box(&mut buffer), limits(), Some(publication))
            .expect("export");
        black_box(&buffer);
    });
    println!(
        "export: median {:.1} ns, {:.2} allocs/op, {:.1} bytes/op (caller output-buffer pre-allocated, excluded)",
        export.median_ns, export.allocs_per_op, export.bytes_per_op
    );

    // Caller-owned artifact bytes prepared outside the import window.
    let mut artifact: Vec<u8> = Vec::new();
    trained
        .export_to(&mut artifact, limits(), None)
        .expect("export for import");
    let expectation = ArtifactExpectation {
        feature_count: Expect::Required(d as u32),
        intercept: Expect::Required(true),
        schema_id: Expect::Required(meta.schema_id),
        transform_link: Expect::Absent,
        cutoff: Expect::Absent,
    };
    let live_before_import = common::live_bytes();
    let import = measure_ops(20, 20, || {
        let mut reader: &[u8] = black_box(&artifact);
        let loaded =
            TrainedModel::import_from(&mut reader, &expectation, limits()).expect("import");
        black_box(loaded);
    });
    let loaded = {
        let mut reader: &[u8] = &artifact;
        TrainedModel::import_from(&mut reader, &expectation, limits()).expect("import")
    };
    let live_with_import = common::live_bytes();
    println!(
        "import: median {:.1} ns, {:.2} allocs/op, {:.1} bytes/op; caller artifact={} bytes; imported retained={} bytes",
        import.median_ns,
        import.allocs_per_op,
        import.bytes_per_op,
        artifact.len(),
        live_with_import as i64 - live_before_import as i64
    );

    // Round-trip preserves stored parameter bits and prediction bits.
    let before_bits: Vec<u32> = trained
        .predictor()
        .weights()
        .as_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let after_bits: Vec<u32> = loaded
        .model
        .predictor()
        .weights()
        .as_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(before_bits, after_bits, "round-trip parameter bits");
    assert_eq!(
        trained.predictor().bias().to_bits(),
        loaded.model.predictor().bias().to_bits(),
        "round-trip intercept bits"
    );
    assert_eq!(
        trained.model_id(),
        loaded.model.model_id(),
        "round-trip identity"
    );
    for row in features.iter().take(4) {
        assert_eq!(
            trained.predictor().predict(row).unwrap().to_bits(),
            loaded.model.predictor().predict(row).unwrap().to_bits(),
            "round-trip prediction bits"
        );
    }
    drop(loaded);

    // Refit overlap: active model alive, candidate constructed separately.
    let active_live = common::live_bytes();
    common::reset_peak();
    let candidate = fit_once(&features, &targets, &train)
        .into_trained(&meta)
        .expect("candidate");
    let candidate_peak_add = common::peak_bytes() as i64 - active_live as i64;
    let candidate_retained = common::live_bytes() as i64 - active_live as i64;
    black_box(&candidate);
    drop(candidate);
    let candidate_dropped = common::live_bytes() as i64 - active_live as i64;
    println!(
        "refit overlap: additional peak {candidate_peak_add} bytes, candidate retained {candidate_retained} bytes, after drop {candidate_dropped} bytes"
    );
    // The active model is unchanged after constructing and dropping the candidate.
    let active_bits: Vec<u32> = trained
        .predictor()
        .weights()
        .as_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(active_bits, before_bits, "active model unchanged");

    println!("\nDone. Measurements are synthetic and device-local; not production guarantees.");
    black_box(&baseline);
}
