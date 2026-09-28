//! RLS benchmarks: control (fixed observation) and decision-grade sequential
//! workloads A/B/C, with direct, streaming, bounded-fold, allocation-attribution,
//! and model-sharing measurements.
//!
//! Run with `cargo bench --bench rls`.
//!
//! The persistent-excitation workload (A, `lambda < 1`) is the primary
//! decision-grade workload; the existing fixed-observation `lambda = 1` block is
//! retained only as a labelled control. The decision-grade matrix runs in the
//! optimized profile only; the debug test profile runs the control and
//! `verify_workloads()`.

use std::hint::black_box;
use std::mem::size_of;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Matrix, Vector};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{StateModel, process_batch, process_one};
use seqvex::models::online::rls::{Rls, RlsSample, RlsState};

#[allow(dead_code)]
mod common;

/// Forgetting factor for the fixed-observation control only.
const LAMBDA_CONTROL: f32 = 1.0;
/// Initial covariance scale used by every path.
const DELTA: f32 = 1.0e4;
/// Bounded fold size; the fold is a local reference experiment, not a speed claim.
const BATCH: usize = 8;

fn model(features: usize, lambda: f32) -> Rls {
    Rls::new(features, lambda, DELTA).unwrap()
}

fn control_observation(features: usize) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_fn(features, |index| (index as f32 * 0.31).sin()),
        target: 0.25,
    })
}

fn workload_observations(xs: &[Vec<f32>], ys: &[f32]) -> Vec<Observation<RlsSample>> {
    xs.iter()
        .zip(ys)
        .map(|(x, &target)| {
            Observation::new(RlsSample {
                features: Vector::from_slice(x),
                target,
            })
        })
        .collect()
}

fn control_steps(features: usize) -> u32 {
    common::timed_steps(match features {
        8 => 200_000,
        32 => 50_000,
        128 => 5_000,
        _ => 2_000,
    })
}

fn workload_steps(features: usize) -> u32 {
    common::timed_steps(match features {
        8 => 200_000,
        32 => 50_000,
        64 => 20_000,
        128 => 5_000,
        _ => 2_000,
    })
}

// ===== benchmark-only analytical/control calculations =========================
// These mirror the production `Rls::update` P' entry formula so the evidence run
// can separate the D^2 arithmetic from the D^2 allocation. They are used only by
// this benchmark and are NOT production primitives (`src/` is unchanged).

/// benchmark-only analytical/control calculation: full `D^2` P' arithmetic.
fn p_full_into(
    out: &mut [f32],
    p: &[f32],
    v: &[f32],
    dimension: usize,
    lambda: f32,
    denominator: f32,
) {
    for row in 0..dimension {
        for col in 0..dimension {
            out[row * dimension + col] =
                (p[row * dimension + col] - v[row] * v[col] / denominator) / lambda;
        }
    }
}

/// benchmark-only analytical/control calculation: upper-triangle P' arithmetic,
/// mirrored. It performs ~half the multiplications/divisions and produces
/// identical entries because `P` is symmetric and `v[r] * v[c] == v[c] * v[r]`
/// exactly in `f32`.
fn p_symmetric_into(
    out: &mut [f32],
    p: &[f32],
    v: &[f32],
    dimension: usize,
    lambda: f32,
    denominator: f32,
) {
    for row in 0..dimension {
        for col in row..dimension {
            let value = (p[row * dimension + col] - v[row] * v[col] / denominator) / lambda;
            out[row * dimension + col] = value;
            out[col * dimension + row] = value;
        }
    }
}

/// benchmark-only analytical/control calculation: a full RLS update substituting
/// the symmetric P'. The extra `Matrix::from_fn` copy is included and is a
/// conservative bias against the symmetric variant.
fn rls_update_symmetric_p(model: &Rls, state: &RlsState, sample: &RlsSample) -> RlsState {
    let features = &sample.features;
    let v = state.p.mul_vector(features).unwrap();
    let denominator = model.lambda() + features.dot(&v).unwrap();
    if !denominator.is_finite() || denominator <= 0.0 {
        // Mirror the production guard by leaving the state unchanged (rare path).
        return state.clone();
    }
    let error = sample.target - state.w.dot(features).unwrap();
    let gain = v.map(|value| value / denominator);
    let next_w = state.w.add(&gain.scale(error)).unwrap();

    let dimension = model.dimension();
    let mut buffer = vec![0.0_f32; dimension * dimension];
    p_symmetric_into(
        &mut buffer,
        state.p.as_slice(),
        v.as_slice(),
        dimension,
        model.lambda(),
        denominator,
    );
    let next_p = Matrix::from_fn(dimension, dimension, |row, col| {
        buffer[row * dimension + col]
    });
    // Mirror the production candidate validation so the end-to-end comparison
    // isolates the P' strategy rather than the presence of the O(D^2) scan.
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return state.clone();
    }
    RlsState {
        w: next_w,
        p: next_p,
    }
}

/// Evidence-phase step counts for the RLS decomposition (release only).
fn evidence_steps(features: usize) -> u32 {
    match features {
        32 => 50_000,
        64 => 20_000,
        128 => 5_000,
        256 => 2_000,
        _ => 1_000,
    }
}

/// Category B evidence: RLS P' allocation vs computation decomposition.
fn run_evidence() {
    if cfg!(debug_assertions) {
        println!("RLS evidence decomposition skipped in the debug profile");
        return;
    }
    const RUNS: usize = 30;
    println!("\n== evidence: RLS P' allocation vs computation ({RUNS} runs) ==");
    for features in [32_usize, 64, 128, 256, 512] {
        let steps = evidence_steps(features);
        let model = model(features, common::LAMBDA_PE);
        let xs = common::persistent_excitation_features(features, 64);
        let weights = common::fixed_weights(features);
        let samples: Vec<Observation<RlsSample>> = xs
            .iter()
            .map(|x| {
                let target = x.iter().zip(&weights).map(|(a, b)| a * b).sum();
                Observation::new(RlsSample {
                    features: Vector::from_slice(x),
                    target,
                })
            })
            .collect();

        // Warm to a representative committed P. `λ < 1` can trip the production
        // denominator guard (recorded numerical data, not a failure), so this
        // warm-up is guard-tolerant like the main workload loop; if it cannot
        // advance, `P = δI` from the initial state is used.
        let mut state = model.initial_state();
        let mut warm_ok = 0_usize;
        for sample in samples.iter() {
            if warm_ok >= 16 {
                break;
            }
            match model.update(&state, sample) {
                Ok(next) => {
                    state = next;
                    warm_ok += 1;
                }
                Err(_) => state = model.initial_state(),
            }
        }
        let p_fixed = state.p.clone();
        let sample_features = Vector::from_slice(&xs[0]);
        let v_fixed = p_fixed.mul_vector(&sample_features).unwrap();
        let denominator_fixed = model.lambda() + sample_features.dot(&v_fixed).unwrap();

        println!(
            "RLS evidence features={features} steps={steps} warm_ok={warm_ok} denom={denominator_fixed:.3e}"
        );
        let mut index = 0_usize;
        let mut guards = 0_u64;
        let full = common::measure_runs("rls-full", steps, 1, RUNS, || {
            let observation = &samples[index % samples.len()];
            index += 1;
            match model.update(&state, observation) {
                Ok(next) => state = next,
                Err(_) => {
                    guards += 1;
                    state = model.initial_state();
                }
            }
        });
        if guards > 0 {
            println!(
                "    rls-full guards={guards} (restarted from initial state, as the main loop does)"
            );
        }

        let mut out_full = vec![0.0_f32; features * features];
        common::measure_runs("p-full-compute", steps, 1, RUNS, || {
            p_full_into(
                &mut out_full,
                p_fixed.as_slice(),
                v_fixed.as_slice(),
                features,
                common::LAMBDA_PE,
                denominator_fixed,
            );
            black_box(&out_full);
        });

        let mut out_symmetric = vec![0.0_f32; features * features];
        common::measure_runs("p-symmetric-compute", steps, 1, RUNS, || {
            p_symmetric_into(
                &mut out_symmetric,
                p_fixed.as_slice(),
                v_fixed.as_slice(),
                features,
                common::LAMBDA_PE,
                denominator_fixed,
            );
            black_box(&out_symmetric);
        });

        common::measure_runs("alloc-control", steps, 1, RUNS, || {
            let buffer = Vec::<f32>::with_capacity(features * features);
            black_box(&buffer);
        });

        let mut symmetric_state = model.initial_state();
        let mut symmetric_index = 0_usize;
        let symmetric = common::measure_runs("rls-symmetric-p", steps, 1, RUNS, || {
            let observation = &samples[symmetric_index % samples.len()];
            symmetric_index += 1;
            symmetric_state = rls_update_symmetric_p(&model, &symmetric_state, observation.value());
        });

        println!(
            "    D={features} full={:.1} ns/obs  symmetric-p={:.1} ns/obs  delta={:.1} ns -> {}",
            full.median,
            symmetric.median,
            (full.median - symmetric.median).abs(),
            common::materiality(full.median, full.iqr, symmetric.median, symmetric.iqr),
        );
    }
}

fn main() {
    println!(
        "RLS benchmark ({} runs/measurement, delta={DELTA}, control lambda={LAMBDA_CONTROL})",
        common::RUNS
    );
    common::verify_workloads();
    println!("{}", common::env_summary());

    // Evidence-phase convenience: `SEQVEX_EVIDENCE_ONLY=1` runs only the
    // decomposition section. The default run still executes everything and also
    // calls `run_evidence()` at the end.
    if std::env::var_os("SEQVEX_EVIDENCE_ONLY").is_some() {
        run_evidence();
        return;
    }

    // Control: fixed observation, lambda = 1 (existing benchmark).
    println!("\n== control (fixed observation, lambda=1) ==");
    for features in [8_usize, 32, 128, 256] {
        let steps = control_steps(features);
        println!("features={features} bounded_batch={BATCH} steps={steps}");

        let observation = control_observation(features);
        let batch = vec![observation.clone(); BATCH];

        let reference = model(features, LAMBDA_CONTROL);
        let mut reference_state = reference.initial_state();
        common::measure("reference", steps, 1, || {
            reference_state = reference
                .update(black_box(&reference_state), black_box(&observation))
                .unwrap();
        });

        let streaming_model = model(features, LAMBDA_CONTROL);
        let initial = streaming_model.initial_state();
        let mut executor = StreamingExecutor::new(&streaming_model, initial);
        common::measure("streaming", steps, 1, || {
            black_box(executor.process_one(black_box(&observation)).unwrap());
        });

        let fold_model = model(features, LAMBDA_CONTROL);
        let mut fold_state = Some(fold_model.initial_state());
        common::measure("bounded-fold", steps, BATCH, || {
            fold_state = Some(
                process_batch(
                    &fold_model,
                    fold_state.take().unwrap(),
                    batch.iter().cloned(),
                )
                .unwrap(),
            );
        });

        let committed = (features + features * features) * size_of::<f32>();
        let transient = (features + features + features * features) * size_of::<f32>();
        println!(
            "    committed state {committed} bytes (w {features} + P {}); candidate + intermediates ~{transient} bytes",
            features * features,
        );
    }

    if cfg!(debug_assertions) {
        println!("\ndecision-grade workload matrix skipped in the debug profile");
        return;
    }

    // Decision-grade workloads A/B/C over the required dimensions.
    let dimensions = [8_usize, 32, 64, 128, 256];
    for workload in ["A", "B", "C"] {
        println!("\n== workload {workload} (lambda={}) ==", common::LAMBDA_PE);
        let mut scaling: Vec<(f64, f64)> = Vec::new();
        for &features in &dimensions {
            let steps = workload_steps(features);
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
            let weights = common::fixed_weights(features);
            let ys = if workload == "B" {
                let mut shifted = weights.clone();
                shifted[0] += 4.0;
                common::regime_targets(&xs, &weights, &shifted, xs.len() / 2)
            } else {
                common::linear_targets(&xs, &weights)
            };
            let observations = workload_observations(&xs, &ys);
            println!(
                "RLS workload={workload} features={features} steps={steps} stream={}",
                observations.len()
            );

            common::measure_startup("init", common::startup_reps(), || {
                let model = model(features, common::LAMBDA_PE);
                black_box(model.initial_state());
            });

            let reference = model(features, common::LAMBDA_PE);
            let mut state = reference.initial_state();
            let mut index = 0_usize;
            let mut reference_failures = 0_u64;
            let stats = common::measure("reference", steps, 1, || {
                let observation = &observations[index % observations.len()];
                index += 1;
                match reference.update(black_box(&state), black_box(observation)) {
                    Ok(next) => state = next,
                    Err(_) => {
                        // A guard is expected numerical data (R5 contract), not a
                        // benchmark abort: count it and restart from the initial
                        // state so the cost measurement completes.
                        reference_failures += 1;
                        state = reference.initial_state();
                    }
                }
            });
            if reference_failures > 0 {
                println!(
                    "    reference guards={reference_failures} (restarted from initial state)"
                );
            }

            let streaming_model = model(features, common::LAMBDA_PE);
            let initial = streaming_model.initial_state();
            let reset_state = model(features, common::LAMBDA_PE).initial_state();
            let mut executor = StreamingExecutor::new(&streaming_model, initial);
            let mut stream_index = 0_usize;
            let mut streaming_failures = 0_u64;
            common::measure("streaming", steps, 1, || {
                let observation = &observations[stream_index % observations.len()];
                stream_index += 1;
                if executor.process_one(black_box(observation)).is_err() {
                    streaming_failures += 1;
                    executor.reset(reset_state.clone());
                }
            });
            if streaming_failures > 0 {
                println!(
                    "    streaming guards={streaming_failures} (restarted from initial state)"
                );
            }

            if workload == "A" {
                let fold_model = model(features, common::LAMBDA_PE);
                // The cross-algorithm reference fold consumes owned observations,
                // so this clones BATCH observations of the moving persistent-
                // excitation window per call; the clone cost is included and
                // reported rather than hidden (as in the control).
                let mut fold_state = Some(fold_model.initial_state());
                let mut fold_index = 0_usize;
                let mut fold_failures = 0_u64;
                common::measure("bounded-fold", steps, BATCH, || {
                    let total = observations.len();
                    let mut window = Vec::with_capacity(BATCH);
                    for offset in 0..BATCH {
                        window.push(observations[(fold_index + offset) % total].clone());
                    }
                    fold_index += BATCH;
                    match process_batch(&fold_model, fold_state.take().unwrap(), window) {
                        Ok(next) => fold_state = Some(next),
                        Err(_) => {
                            fold_failures += 1;
                            fold_state = Some(fold_model.initial_state());
                        }
                    }
                });
                if fold_failures > 0 {
                    println!(
                        "    bounded-fold guards={fold_failures} (restarted from initial state)"
                    );
                }

                // Measurement-only allocation attribution: time an equivalent
                // `D x D` buffer allocation; never integrated into RLS.
                common::measure("alloc-control", steps, 1, || {
                    black_box(Vec::<f32>::with_capacity(features * features));
                });

                // Model sharing through the existing foundation `&M` path.
                let shared = model(features, common::LAMBDA_PE);
                let shared_reset = shared.initial_state();
                let mut shared_a = shared.initial_state();
                let mut shared_b = shared.initial_state();
                let mut shared_index = 0_usize;
                let mut shared_failures = 0_u64;
                let shared_stats = common::measure("sharing-1model", steps, 1, || {
                    let observation = &observations[shared_index % observations.len()];
                    shared_index += 1;
                    match process_one(&shared, &shared_a, black_box(observation)) {
                        Ok(next) => shared_a = next,
                        Err(_) => {
                            shared_failures += 1;
                            shared_a = shared_reset.clone();
                        }
                    }
                    match process_one(&shared, &shared_b, black_box(observation)) {
                        Ok(next) => shared_b = next,
                        Err(_) => {
                            shared_failures += 1;
                            shared_b = shared_reset.clone();
                        }
                    }
                });
                let model_a = model(features, common::LAMBDA_PE);
                let model_b = model(features, common::LAMBDA_PE);
                let separate_reset_a = model_a.initial_state();
                let separate_reset_b = model_b.initial_state();
                let mut separate_a = model_a.initial_state();
                let mut separate_b = model_b.initial_state();
                let mut separate_index = 0_usize;
                let mut separate_failures = 0_u64;
                let separate_stats = common::measure("sharing-2models", steps, 1, || {
                    let observation = &observations[separate_index % observations.len()];
                    separate_index += 1;
                    match process_one(&model_a, &separate_a, black_box(observation)) {
                        Ok(next) => separate_a = next,
                        Err(_) => {
                            separate_failures += 1;
                            separate_a = separate_reset_a.clone();
                        }
                    }
                    match process_one(&model_b, &separate_b, black_box(observation)) {
                        Ok(next) => separate_b = next,
                        Err(_) => {
                            separate_failures += 1;
                            separate_b = separate_reset_b.clone();
                        }
                    }
                });
                if shared_failures + separate_failures > 0 {
                    println!(
                        "    sharing guards={} (1model) {} (2models)",
                        shared_failures, separate_failures
                    );
                }
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
                "RLS workload={workload} scaling exponent={:.3} (expected ~2.0)",
                common::log_log_slope(&scaling)
            );
        }
    }

    run_evidence();
}
