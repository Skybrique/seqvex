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

// ===== bounded reference batch-size sensitivity (#26) =========================
//
// Compares grouping through `StreamingExecutor::process_one` against the
// RLS-local bounded reference API over an identical ordered stream. Observation
// histories are equalized across batch sizes by fixing the outer window
// (`work_per_call = OUTER`) and the per-run outer-call count: every path and
// every `B` sees the same warm-up and timed observation counts. The fixture is a
// deterministic cyclic replay; its limitations are noted where it is built.

/// Batch sizes swept for the bounded reference API (#26), release.
const SWEEP_BATCH_SIZES: [usize; 4] = [1, 8, 32, 128];
/// Outer observation window per harness invocation, release.
const SWEEP_OUTER_RELEASE: usize = 128;
/// Outer observation window under the debug test profile.
const SWEEP_OUTER_DEBUG: usize = 8;
/// Deterministic cyclic fixture length; `OUTER | LEN` and every swept `B | OUTER`.
const SWEEP_FIXTURE_LEN: usize = 2048;
/// Short bitwise precheck outer windows per batch size, release.
const SWEEP_PRECHECK_OUTER_CALLS: usize = 4;
/// Fixed timed outer calls under the debug test profile.
const SWEEP_DEBUG_STEPS: u32 = 4;
/// Planned default dimension matrix; `D = 128` is deferred.
const SWEEP_DIMENSIONS: [usize; 2] = [8, 32];

fn sweep_outer() -> usize {
    if cfg!(debug_assertions) {
        SWEEP_OUTER_DEBUG
    } else {
        SWEEP_OUTER_RELEASE
    }
}

fn sweep_batch_sizes() -> &'static [usize] {
    if cfg!(debug_assertions) {
        &SWEEP_BATCH_SIZES[..2]
    } else {
        &SWEEP_BATCH_SIZES
    }
}

fn sweep_precheck_outer_calls() -> usize {
    if cfg!(debug_assertions) {
        2
    } else {
        SWEEP_PRECHECK_OUTER_CALLS
    }
}

/// Outer-window harness invocations per timed run. Debug uses a fixed tiny count
/// and is validation only, not evidence.
fn sweep_steps(features: usize) -> u32 {
    if cfg!(debug_assertions) {
        SWEEP_DEBUG_STEPS
    } else {
        match features {
            8 => 32,
            32 => 16,
            _ => 16,
        }
    }
}

/// One deterministic dense persistent-excitation fixture per dimension, replayed
/// cyclically. Replay makes the driving stream periodic with period
/// `SWEEP_FIXTURE_LEN`; `lambda < 1` down-weights older contributions
/// geometrically but never to exactly zero, so the effective memory is a scale,
/// not a hard cutoff, and periodicity is not literally invisible. This is a
/// controlled grouping-cost experiment, not a claim of statistical applicability.
fn sweep_fixture(features: usize) -> Vec<Observation<RlsSample>> {
    let xs = common::persistent_excitation_features(features, SWEEP_FIXTURE_LEN);
    let weights = common::fixed_weights(features);
    let ys = common::linear_targets(&xs, &weights);
    workload_observations(&xs, &ys)
}

/// Bitwise comparison of **both** coupled state components (`w` and `p`).
fn bitwise_same_state(left: &RlsState, right: &RlsState) -> bool {
    left.w.as_slice().len() == right.w.as_slice().len()
        && left.p.as_slice().len() == right.p.as_slice().len()
        && left
            .w
            .as_slice()
            .iter()
            .zip(right.w.as_slice())
            .all(|(l, r)| l.to_bits() == r.to_bits())
        && left
            .p
            .as_slice()
            .iter()
            .zip(right.p.as_slice())
            .all(|(l, r)| l.to_bits() == r.to_bits())
}

/// Final state after `outer_calls` contiguous cyclic outer windows, carried
/// across calls. `foundation` selects the cloning `process_batch` control.
fn sweep_state_after_outer_calls(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    outer: usize,
    batch: usize,
    outer_calls: usize,
    foundation: bool,
) -> RlsState {
    let mut state = model.initial_state();
    let mut index = 0_usize;
    for _ in 0..outer_calls {
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        for chunk in window.chunks(batch) {
            state = if foundation {
                process_batch(model, state, chunk.iter().cloned()).unwrap()
            } else {
                model.process_batch_reference(state, chunk, batch).unwrap()
            };
        }
        index += outer;
    }
    state
}

/// Short bitwise equivalence prechecks, outside every timing region: cross-path
/// and cross-`B` agreement over equal outer windows, differing-chunk continuity,
/// and wrap-crossing continuity.
fn verify_sweep_equivalence(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    outer: usize,
    batch_sizes: &[usize],
    outer_calls: usize,
) {
    // Grouped reference over the same outer windows.
    let mut executor = StreamingExecutor::new(model, model.initial_state());
    let mut index = 0_usize;
    for _ in 0..outer_calls {
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        for observation in window {
            executor.process_one(observation).unwrap();
        }
        index += outer;
    }
    let grouped = executor.state().clone();

    let mut first_bounded: Option<RlsState> = None;
    for &batch in batch_sizes {
        let batched =
            sweep_state_after_outer_calls(model, observations, outer, batch, outer_calls, false);
        let folded =
            sweep_state_after_outer_calls(model, observations, outer, batch, outer_calls, true);
        assert!(
            bitwise_same_state(&grouped, &batched),
            "short precheck: grouped vs bounded diverged at B={batch}"
        );
        assert!(
            bitwise_same_state(&grouped, &folded),
            "short precheck: grouped vs foundation fold diverged at B={batch}"
        );
        match &first_bounded {
            None => first_bounded = Some(batched),
            Some(reference) => assert!(
                bitwise_same_state(reference, &batched),
                "short precheck: cross-B continuity failed at B={batch}"
            ),
        }
    }

    // Differing-chunk continuity over a fixed ordered segment.
    let segment = &observations[..32];
    let single = model
        .process_batch_reference(model.initial_state(), segment, segment.len())
        .unwrap();
    let mut chunked = model.initial_state();
    let mut start = 0_usize;
    for &size in &[2_usize, 3, 5, 7, 15] {
        let end = start + size;
        chunked = model
            .process_batch_reference(chunked, &segment[start..end], size)
            .unwrap();
        start = end;
    }
    assert!(
        bitwise_same_state(&single, &chunked),
        "short precheck: differing-chunk continuity failed"
    );

    // Wrap-crossing continuity: two outer windows starting at the fixture end.
    let mut wrap_grouped = StreamingExecutor::new(model, model.initial_state());
    let mut wrap_batched = model.initial_state();
    let mut index = observations.len() - outer;
    for _ in 0..2 {
        let offset = index % observations.len();
        let window = &observations[offset..offset + outer];
        for observation in window {
            wrap_grouped.process_one(observation).unwrap();
        }
        wrap_batched = model
            .process_batch_reference(wrap_batched, window, outer)
            .unwrap();
        index += outer;
    }
    assert!(
        bitwise_same_state(wrap_grouped.state(), &wrap_batched),
        "short precheck: wrap-crossing continuity failed"
    );
}

/// Full-horizon numerical validation (release only), run before any measurement.
/// Traverses the complete intended trajectory once per dimension, including wrap
/// boundaries, and aborts on the first failed update or non-finite state. No
/// reset, skip, retry, or silent reduction.
fn verify_release_horizon(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    outer: usize,
    steps: u32,
) {
    let warm_up_calls = (steps as usize / 10).max(1000);
    let total_outer_calls = warm_up_calls + common::RUNS * steps as usize;
    let mut state = model.initial_state();
    let mut index = 0_usize;
    let mut wrap_crossings = 0_usize;
    for call in 0..total_outer_calls {
        let offset = index % observations.len();
        if offset == 0 && call > 0 {
            wrap_crossings += 1;
        }
        let window = &observations[offset..offset + outer];
        state = model
            .process_batch_reference(state, window, outer)
            .expect("release full-horizon validation: update must succeed");
        for value in state.w.as_slice().iter().chain(state.p.as_slice()) {
            assert!(
                value.is_finite(),
                "release full-horizon validation: non-finite state at outer call {call}"
            );
        }
        index += outer;
    }
    println!(
        "  full-horizon release validation PASS: D={} obs={} wrap_crossings={wrap_crossings}",
        model.dimension(),
        total_outer_calls * outer,
    );
}

fn report_sweep_row(
    path: &str,
    features: usize,
    batch: usize,
    steps: u32,
    outer: usize,
    api_calls_per_window: usize,
    stats: common::Stats,
) {
    let api_calls_per_run = steps as usize * api_calls_per_window;
    let timed_obs = steps as usize * outer;
    let ns_per_batch = stats.median * batch as f64;
    let obs_per_second = if stats.median > 0.0 {
        1e9 / stats.median
    } else {
        0.0
    };
    println!(
        "    D={features} lambda={} B={batch:>3} path={path:<15} \
         outer_invocations/run={steps} api_calls/window={api_calls_per_window} \
         api_calls/run={api_calls_per_run} timed_obs={timed_obs} \
         ns/obs={:.1} p95={:.1} iqr={:.1} obs/s={obs_per_second:.0} \
         allocs/obs={:.3} bytes/obs={:.1} derived_ns/batch={ns_per_batch:.1}",
        common::LAMBDA_PE,
        stats.median,
        stats.p95,
        stats.iqr,
        stats.allocations,
        stats.bytes,
    );
}

fn run_batch_sweep() {
    let outer = sweep_outer();
    let batch_sizes = sweep_batch_sizes();
    let outer_calls = sweep_precheck_outer_calls();

    println!(
        "\n== bounded reference batch-size sensitivity (lambda={}, outer={outer}) ==",
        common::LAMBDA_PE
    );
    println!(
        "  paths: grouped=StreamingExecutor::process_one; bounded=Rls::process_batch_reference; \
         foundation=process_batch (labelled cloning control, release only)"
    );
    println!(
        "  work_per_call=OUTER={outer}; observation histories are equalized across paths and B. \
         Fixture={} cyclic deterministic persistent-excitation observations. \
         ns/batch is window-derived amortized, p95/IQR are window variability, \
         bytes/obs is allocator traffic (not live/peak memory), and committed \
         payload is (D + D^2) x 4 bytes.",
        SWEEP_FIXTURE_LEN
    );
    if cfg!(debug_assertions) {
        println!(
            "  debug profile: validation only (B={{1,8}}, outer={outer}, {} timed outer calls); \
             full-horizon release validation and release measurements are deferred",
            SWEEP_DEBUG_STEPS
        );
    }

    // Short prechecks in every profile.
    for &features in &SWEEP_DIMENSIONS {
        let model = model(features, common::LAMBDA_PE);
        let observations = sweep_fixture(features);
        verify_sweep_equivalence(&model, &observations, outer, batch_sizes, outer_calls);
    }

    // Full-horizon numerical validation, release only, before measurement.
    if !cfg!(debug_assertions) {
        for &features in &SWEEP_DIMENSIONS {
            let model = model(features, common::LAMBDA_PE);
            let observations = sweep_fixture(features);
            verify_release_horizon(&model, &observations, outer, sweep_steps(features));
        }
    }

    for &features in &SWEEP_DIMENSIONS {
        let model = model(features, common::LAMBDA_PE);
        let observations = sweep_fixture(features);
        let steps = sweep_steps(features);
        let timed_obs = steps as usize * outer;
        println!(
            "  -- D={features} steps(outer invocations/run)={steps} outer={outer} timed_obs={timed_obs} --"
        );

        // Grouped path: OUTER process_one calls per outer window.
        let mut grouped_executor = StreamingExecutor::new(&model, model.initial_state());
        let mut grouped_index = 0_usize;
        common::measure("sweep grouped", steps, outer, || {
            let offset = grouped_index % observations.len();
            let window = &observations[offset..offset + outer];
            for observation in window {
                black_box(
                    grouped_executor
                        .process_one(black_box(observation))
                        .unwrap(),
                );
            }
            grouped_index += outer;
        });
        println!(
            "    D={features} lambda={} B=-   path=grouped          \
             outer_invocations/run={steps} api_calls/window={outer} api_calls/run={timed_obs}",
            common::LAMBDA_PE,
        );

        // Bounded reference path: OUTER/B calls per outer window.
        for &batch in batch_sizes {
            let mut carried = Some(model.initial_state());
            let mut index = 0_usize;
            let api_calls_per_window = outer / batch;
            let stats = common::measure("sweep bounded-ref", steps, outer, || {
                let offset = index % observations.len();
                let window = &observations[offset..offset + outer];
                let mut state = carried.take().expect("bounded state available");
                for chunk in window.chunks(batch) {
                    state = model
                        .process_batch_reference(state, black_box(chunk), batch)
                        .unwrap();
                }
                carried = Some(state);
                index += outer;
            });
            report_sweep_row(
                "bounded-ref",
                features,
                batch,
                steps,
                outer,
                api_calls_per_window,
                stats,
            );
        }

        // Foundation cloning control, release only. The observation clone is
        // inside its timing; it is a labelled accounting item, not a speedup.
        if !cfg!(debug_assertions) {
            for &batch in batch_sizes {
                let mut carried = Some(model.initial_state());
                let mut index = 0_usize;
                let api_calls_per_window = outer / batch;
                let stats = common::measure("sweep foundation-fold", steps, outer, || {
                    let offset = index % observations.len();
                    let window = &observations[offset..offset + outer];
                    let mut state = carried.take().expect("fold state available");
                    for chunk in window.chunks(batch) {
                        state = process_batch(&model, state, chunk.iter().cloned()).unwrap();
                    }
                    carried = Some(state);
                    index += outer;
                });
                report_sweep_row(
                    "foundation-fold",
                    features,
                    batch,
                    steps,
                    outer,
                    api_calls_per_window,
                    stats,
                );
            }
        }
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

    run_batch_sweep();

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
