//! SGD prediction/update/resource baseline (benchmark-only).
//!
//! Measures the unchanged public SGD implementation on deterministic synthetic
//! workloads. This is evidence, not an optimization or a performance guarantee.
//!
//! Release decision run (only after execution authorization):
//! `cargo bench --locked --bench sgd`
//!
//! Debug smoke (targeted `harness = false` execution):
//! `cargo test --locked --bench sgd --all-features`
//!
//! The debug path performs a tiny smoke and returns before the release matrix.
//! Run-average percentiles are not individual-operation tail latency.
//! Allocator live/requested bytes are not operating-system resident memory.
//! The `4D`/`12D`/`8D` payload figures are source-derived hypotheses.

use std::hint::black_box;
use std::mem::size_of;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{StateModel, process_one};
use seqvex::models::linear::sgd::{Sgd, SgdSample, SgdState};

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

const DIMENSIONS: [usize; 5] = [2, 8, 32, 128, 512];
const STEPS: [u32; 5] = [400_000, 200_000, 100_000, 25_000, 5_000];
const FIXTURE_LEN: usize = 2048;
const WARM_START: usize = 512;
const ETA: f32 = 0.05;
const SEED: u64 = 0x5eed_55d0_0000_0001;
const PREDICTION_PASSES: usize = 2;

/// f32 feature generation with frozen conversion/grouping points:
/// `raw = Lcg::next_f32()`; `inv = (1.0f64 / sqrt(D as f64)) as f32`;
/// `x_j = (2.0f32 * raw) * inv`.
fn scaled_features(d: usize, n: usize) -> Vec<Vec<f32>> {
    let inv = (1.0f64 / (d as f64).sqrt()) as f32;
    let mut lcg = common::Lcg::new(SEED);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    let raw = lcg.next_f32();
                    (2.0f32 * raw) * inv
                })
                .collect()
        })
        .collect()
}

fn observations_for(d: usize) -> Vec<Observation<SgdSample>> {
    let xs = scaled_features(d, FIXTURE_LEN);
    let weights = common::fixed_weights(d);
    let ys = common::linear_targets(&xs, &weights);
    xs.iter()
        .zip(&ys)
        .map(|(x, y)| {
            Observation::new(SgdSample {
                features: Vector::from_slice(x),
                target: *y,
            })
        })
        .collect()
}

fn harness_warmup(steps: u32) -> u32 {
    (steps / 10).max(if cfg!(debug_assertions) { 50 } else { 1_000 })
}

fn warmed_state(model: &Sgd, observations: &[Observation<SgdSample>]) -> SgdState {
    let mut state = model.initial_state();
    for (index, observation) in observations.iter().take(WARM_START).enumerate() {
        state = model.update(&state, observation).unwrap_or_else(|error| {
            panic!("warm-start update call={index} index={index} failed: {error:?}")
        });
    }
    state
}

fn verify_fixture(d: usize) {
    let first = scaled_features(d, FIXTURE_LEN);
    let second = scaled_features(d, FIXTURE_LEN);
    assert_eq!(
        first, second,
        "D={d}: fixture generation must be deterministic"
    );
    assert!(
        first.iter().flatten().all(|value| value.is_finite()),
        "D={d}: fixture features must be finite"
    );
    let targets = common::linear_targets(&first, &common::fixed_weights(d));
    assert!(
        targets.iter().all(|value| value.is_finite()),
        "D={d}: fixture targets must be finite"
    );
}

fn assert_state_bits(
    d: usize,
    stage: &str,
    call: usize,
    index: usize,
    direct: &SgdState,
    foundation: &SgdState,
    executor: &SgdState,
) {
    let direct = direct.w.as_slice();
    let foundation = foundation.w.as_slice();
    let executor = executor.w.as_slice();
    assert_eq!(direct.len(), d);
    for (j, value) in direct.iter().enumerate() {
        assert!(
            value.is_finite(),
            "D={d} stage={stage} call={call} index={index}: direct w[{j}] is non-finite"
        );
        assert_eq!(
            value.to_bits(),
            foundation[j].to_bits(),
            "D={d} stage={stage} call={call} index={index}: foundation w[{j}] differs"
        );
        assert_eq!(
            value.to_bits(),
            executor[j].to_bits(),
            "D={d} stage={stage} call={call} index={index}: executor w[{j}] differs"
        );
    }
}

/// Full `H + RUNS*steps` lockstep trajectory: direct, foundation `process_one`,
/// and `StreamingExecutor`, compared bitwise after harness warm-up and each
/// timed-run boundary. Any failure aborts; no reset/skip/restart.
fn precheck_updates(
    d: usize,
    steps: u32,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) {
    let warmup = harness_warmup(steps) as usize;
    let total = warmup + common::RUNS * steps as usize;
    let mut direct = warm.clone();
    let mut foundation = warm.clone();
    let mut executor = StreamingExecutor::new(model, warm.clone());

    for call in 0..total {
        let index = (WARM_START + call) % FIXTURE_LEN;
        let observation = &observations[index];
        direct = model.update(&direct, observation).unwrap_or_else(|error| {
            panic!("D={d} precheck direct call={call} index={index}: update failed: {error:?}")
        });
        foundation = process_one(model, &foundation, observation).unwrap_or_else(|error| {
            panic!("D={d} precheck foundation call={call} index={index}: update failed: {error:?}")
        });
        executor.process_one(observation).unwrap_or_else(|error| {
            panic!("D={d} precheck executor call={call} index={index}: update failed: {error:?}")
        });

        let completed = call + 1;
        let boundary = completed == warmup
            || (completed > warmup && (completed - warmup) % steps as usize == 0);
        if boundary {
            assert_state_bits(
                d,
                "post-warm-up/timed-run boundary",
                call,
                index,
                &direct,
                &foundation,
                executor.state(),
            );
        }
    }
    println!("precheck updates D={d}: {total} calls lockstep; H={warmup}; boundaries verified");
}

/// Prediction reference table per observation index, built outside timing.
/// Both passes compare every output with the table using exact `to_bits`
/// equality; therefore pass 2 also matches pass 1 at every index. The printed
/// fingerprint is supplementary only.
fn precheck_predictions(
    d: usize,
    steps: u32,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) {
    let reference: Vec<u32> = (0..FIXTURE_LEN)
        .map(|index| {
            model
                .predict(warm, &observations[index].value().features)
                .unwrap_or_else(|error| {
                    panic!("D={d} prediction reference index={index} failed: {error:?}")
                })
                .to_bits()
        })
        .collect();

    let warmup = harness_warmup(steps) as usize;
    let total = warmup + common::RUNS * steps as usize;
    for pass in 0..PREDICTION_PASSES {
        let mut fingerprint = 0u64;
        for call in 0..total {
            let index = (WARM_START + call) % FIXTURE_LEN;
            let bits = model
                .predict(warm, &observations[index].value().features)
                .unwrap_or_else(|error| {
                    panic!(
                        "D={d} prediction pass={pass} call={call} index={index} failed: {error:?}"
                    )
                })
                .to_bits();
            assert_eq!(
                bits, reference[index],
                "D={d} prediction pass={pass} call={call} index={index}: output bits mismatch"
            );
            fingerprint = fingerprint.rotate_left(5) ^ u64::from(bits);
        }
        println!(
            "precheck predictions D={d}: pass={pass} calls={total}; fingerprint={fingerprint:016x} (supplementary)"
        );
    }
}

fn predict_stats(
    d: usize,
    steps: u32,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) -> common::Stats {
    let mut index = WARM_START;
    common::measure("predict", steps, 1, || {
        let i = index % FIXTURE_LEN;
        index += 1;
        let observation = black_box(&observations[i]);
        let value = model
            .predict(black_box(warm), &observation.value().features)
            .unwrap_or_else(|error| panic!("D={d} predict index={i} failed: {error:?}"));
        black_box(value);
    })
}

fn update_direct_stats(
    d: usize,
    steps: u32,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) -> common::Stats {
    let mut state = warm.clone();
    let mut index = WARM_START;
    common::measure("update-direct", steps, 1, || {
        let i = index % FIXTURE_LEN;
        index += 1;
        let observation = black_box(&observations[i]);
        state = model
            .update(black_box(&state), observation)
            .unwrap_or_else(|error| panic!("D={d} update-direct index={i} failed: {error:?}"));
        black_box(&state);
    })
}

fn update_foundation_stats(
    d: usize,
    steps: u32,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) -> common::Stats {
    let mut state = warm.clone();
    let mut index = WARM_START;
    common::measure("update-process_one", steps, 1, || {
        let i = index % FIXTURE_LEN;
        index += 1;
        let observation = black_box(&observations[i]);
        state = process_one(model, black_box(&state), observation)
            .unwrap_or_else(|error| panic!("D={d} update-process_one index={i} failed: {error:?}"));
        black_box(&state);
    })
}

fn update_executor_stats(
    d: usize,
    steps: u32,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) -> common::Stats {
    let mut executor = StreamingExecutor::new(model, warm.clone());
    let mut index = WARM_START;
    common::measure("update-executor", steps, 1, || {
        let i = index % FIXTURE_LEN;
        index += 1;
        let observation = black_box(&observations[i]);
        let state = executor
            .process_one(observation)
            .unwrap_or_else(|error| panic!("D={d} update-executor index={i} failed: {error:?}"));
        black_box(state);
    })
}

fn next_cyclic_index(index: &mut usize) -> usize {
    let current = *index % FIXTURE_LEN;
    *index += 1;
    current
}

fn verify_cyclic_index() {
    let mut index = WARM_START;
    let expected: Vec<usize> = (WARM_START..FIXTURE_LEN).chain(0..WARM_START).collect();
    for _cycle in 0..2 {
        for &expected_index in &expected {
            assert_eq!(
                next_cyclic_index(&mut index),
                expected_index,
                "cyclic index sequence mismatch"
            );
        }
    }
    assert_eq!(index, WARM_START + 2 * FIXTURE_LEN);
    assert_eq!(next_cyclic_index(&mut index), WARM_START);
    println!("cyclic index check: ok");
}

/// Descriptive `dot`-only control. It follows the same cyclic observation-index
/// sequence as prediction (`WARM_START`, then advancing per call), using fixed
/// warmed weights and prebuilt features. It is not a production path, is not
/// used for timing subtraction, and establishes no allocator cost share.
fn dot_control_stats(
    d: usize,
    steps: u32,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) -> common::Stats {
    let weights = warm.w.clone();
    let mut index = WARM_START;
    common::measure("dot-control", steps, 1, || {
        let i = next_cyclic_index(&mut index);
        let observation = black_box(&observations[i]);
        let value = weights
            .dot(&observation.value().features)
            .unwrap_or_else(|error| panic!("D={d} dot-control index={i} failed: {error:?}"));
        black_box(value);
    })
}

/// Descriptive two-allocation control with explicit overlapping lifetimes,
/// observable use and explicit drops. It is not a cost-share bound.
fn alloc_control_stats(d: usize, steps: u32) -> common::Stats {
    common::measure("alloc-control", steps, 1, || {
        let delta = Vec::<f32>::with_capacity(d);
        let candidate = Vec::<f32>::with_capacity(d);
        black_box(delta.as_ptr());
        black_box(candidate.as_ptr());
        drop(delta);
        drop(candidate);
    })
}

/// Allocation-count verification window: replicate `H` harness warm-up calls,
/// then count two consecutive `steps` blocks and require identical per-op
/// counts/bytes. Setup, warm-up, printing and unrelated allocations are outside
/// the windows.
fn verify_allocation_window<F: FnMut(usize)>(
    d: usize,
    label: &str,
    steps: u32,
    warmup: u32,
    mut call: F,
) {
    let warmup = warmup as usize;
    let steps = steps as usize;
    for k in 0..warmup {
        call(k);
    }
    let (alloc_0, bytes_0) = common::allocation_totals();
    for k in 0..steps {
        call(warmup + k);
    }
    let (alloc_1, bytes_1) = common::allocation_totals();
    for k in 0..steps {
        call(warmup + steps + k);
    }
    let (alloc_2, bytes_2) = common::allocation_totals();

    let operations = steps as f64;
    let allocs_1 = (alloc_1 - alloc_0) as f64 / operations;
    let bytes_per_op_1 = (bytes_1 - bytes_0) as f64 / operations;
    let allocs_2 = (alloc_2 - alloc_1) as f64 / operations;
    let bytes_per_op_2 = (bytes_2 - bytes_1) as f64 / operations;
    println!(
        "alloc-window D={d} {label}: block1 {allocs_1:.3} allocs/op {bytes_per_op_1:.1} B/op; \
         block2 {allocs_2:.3} allocs/op {bytes_per_op_2:.1} B/op (requested heap; not RSS)"
    );
    assert_eq!(
        allocs_1, allocs_2,
        "D={d} {label}: allocation counts differ between windows"
    );
    assert_eq!(
        bytes_per_op_1, bytes_per_op_2,
        "D={d} {label}: allocated bytes differ between windows"
    );
}

/// Separate retained-state probe: the state is alive at both snapshots.
fn retained_probe(d: usize, model: &Sgd) {
    let payload = d * size_of::<f32>();
    println!(
        "storage D={d}: weight_payload={payload} B; \
         vector_header={} B; state={} B; model={} B; executor={} B",
        size_of::<Vector>(),
        size_of::<SgdState>(),
        size_of::<Sgd>(),
        size_of::<StreamingExecutor<'static, Sgd>>()
    );
    println!(
        "storage D={d}: logical caller-owned={} B; executor-owned={} B (embedded headers counted once)",
        size_of::<Sgd>() + size_of::<SgdState>() + payload,
        size_of::<Sgd>() + size_of::<StreamingExecutor<'static, Sgd>>() + payload
    );

    let live_before = common::live_bytes();
    common::reset_peak();
    let state = model.initial_state();
    let peak = common::peak_bytes();
    let live_with_state = common::live_bytes();
    println!(
        "retained D={d}: allocator live delta={} B; additional peak={} B (state alive at snapshots)",
        live_with_state as i64 - live_before as i64,
        peak as i64 - live_before as i64
    );
    drop(state);
    println!(
        "retained D={d}: after drop={} B",
        common::live_bytes() as i64 - live_before as i64
    );
}

/// One-update resource probe: commit/replace before the post-replacement
/// snapshots. The delta drops inside `update`; the old committed state drops at
/// the assignment. The peak counter retains the transient overlap.
fn update_peak_probe(
    d: usize,
    model: &Sgd,
    warm: &SgdState,
    observations: &[Observation<SgdSample>],
) {
    let mut state = warm.clone();
    let index = WARM_START % FIXTURE_LEN;
    let live_before = common::live_bytes();
    common::reset_peak();
    state = model
        .update(&state, &observations[index])
        .unwrap_or_else(|error| panic!("D={d} peak probe update failed: {error:?}"));
    let peak = common::peak_bytes();
    let live_after = common::live_bytes();
    println!(
        "peak update D={d}: baseline={} B; additional_peak={} B; live_after={} B \
         (allocator live; not RSS; 4D/12D/8D remain hypotheses)",
        live_before,
        peak as i64 - live_before as i64,
        live_after as i64 - live_before as i64
    );
    drop(state);
}

fn run_dimension(d: usize, steps: u32) -> (f64, f64) {
    verify_fixture(d);
    let observations = observations_for(d);
    let model = Sgd::new(d, ETA).expect("valid SGD configuration");
    let warm = warmed_state(&model, &observations);
    let warmup = harness_warmup(steps);
    let measured_calls = warmup as usize + common::RUNS * steps as usize;
    println!(
        "\n== D={d} steps={steps} H={warmup} measured_calls={measured_calls} runs={} ==",
        common::RUNS
    );

    precheck_predictions(d, steps, &model, &warm, &observations);
    precheck_updates(d, steps, &model, &warm, &observations);
    retained_probe(d, &model);
    update_peak_probe(d, &model, &warm, &observations);

    println!(
        "initialization: repeated warmed initialization/drop cost \
         (aggregate average; counter sampling included; not cold-start, \
         first-allocation latency, or a distribution)"
    );
    common::measure_startup("sgd-new", common::startup_reps(), || {
        black_box(Sgd::new(d, ETA).expect("valid SGD configuration"));
    });
    common::measure_startup("initial-state", common::startup_reps(), || {
        black_box(model.initial_state());
    });

    let predict = predict_stats(d, steps, &model, &warm, &observations);
    let direct = update_direct_stats(d, steps, &model, &warm, &observations);
    let foundation = update_foundation_stats(d, steps, &model, &warm, &observations);
    let executor = update_executor_stats(d, steps, &model, &warm, &observations);
    let dot = dot_control_stats(d, steps, &warm, &observations);
    let alloc = alloc_control_stats(d, steps);

    println!(
        "wrapper materiality D={d}: direct-vs-process_one={}; direct-vs-executor={}",
        common::materiality(direct.median, direct.iqr, foundation.median, foundation.iqr),
        common::materiality(direct.median, direct.iqr, executor.median, executor.iqr)
    );
    println!(
        "controls D={d}: dot median={:.1} ns/op; alloc median={:.1} ns/op (descriptive; no subtraction)",
        dot.median, alloc.median
    );

    // Allocation-count windows: replicate the first two timed-run windows.
    let mut predict_index = WARM_START;
    verify_allocation_window(d, "predict", steps, warmup, |_| {
        let i = predict_index % FIXTURE_LEN;
        predict_index += 1;
        let value = model
            .predict(&warm, &observations[i].value().features)
            .unwrap_or_else(|error| panic!("D={d} count predict index={i}: {error:?}"));
        black_box(value);
    });

    let mut direct_state = warm.clone();
    let mut direct_index = WARM_START;
    verify_allocation_window(d, "update-direct", steps, warmup, |_| {
        let i = direct_index % FIXTURE_LEN;
        direct_index += 1;
        direct_state = model
            .update(&direct_state, &observations[i])
            .unwrap_or_else(|error| panic!("D={d} count update-direct index={i}: {error:?}"));
        black_box(&direct_state);
    });

    let mut foundation_state = warm.clone();
    let mut foundation_index = WARM_START;
    verify_allocation_window(d, "update-process_one", steps, warmup, |_| {
        let i = foundation_index % FIXTURE_LEN;
        foundation_index += 1;
        foundation_state = process_one(&model, &foundation_state, &observations[i])
            .unwrap_or_else(|error| panic!("D={d} count update-process_one index={i}: {error:?}"));
        black_box(&foundation_state);
    });

    let mut count_executor = StreamingExecutor::new(&model, warm.clone());
    let mut executor_index = WARM_START;
    verify_allocation_window(d, "update-executor", steps, warmup, |_| {
        let i = executor_index % FIXTURE_LEN;
        executor_index += 1;
        let state = count_executor
            .process_one(&observations[i])
            .unwrap_or_else(|error| panic!("D={d} count update-executor index={i}: {error:?}"));
        black_box(state);
    });

    verify_allocation_window(d, "alloc-control", steps, 0, |_| {
        let delta = Vec::<f32>::with_capacity(d);
        let candidate = Vec::<f32>::with_capacity(d);
        black_box(delta.as_ptr());
        black_box(candidate.as_ptr());
        drop(delta);
        drop(candidate);
    });

    black_box((predict, direct, foundation, executor, dot, alloc));
    (predict.median, direct.median)
}

fn smoke() {
    println!("sgd bench: debug smoke only (debug_assertions=true); release matrix skipped");
    verify_cyclic_index();
    let d = 8usize;
    let observations = observations_for(d);
    let model = Sgd::new(d, ETA).expect("valid SGD configuration");
    let mut warm = model.initial_state();
    for observation in observations.iter().take(4) {
        warm = model.update(&warm, observation).expect("smoke warm update");
    }

    let mut direct = warm.clone();
    let mut foundation = warm.clone();
    let mut executor = StreamingExecutor::new(&model, warm.clone());
    for k in 0..8 {
        let index = (WARM_START + k) % FIXTURE_LEN;
        let observation = &observations[index];
        direct = model
            .update(&direct, observation)
            .expect("smoke direct update");
        foundation =
            process_one(&model, &foundation, observation).expect("smoke foundation update");
        executor
            .process_one(observation)
            .expect("smoke executor update");
    }
    assert_state_bits(
        d,
        "smoke final",
        8,
        (WARM_START + 7) % FIXTURE_LEN,
        &direct,
        &foundation,
        executor.state(),
    );
    let prediction = model
        .predict(&warm, &observations[0].value().features)
        .expect("smoke prediction");
    assert!(prediction.is_finite(), "smoke prediction must be finite");
    let initial = model.initial_state();
    assert!(initial.w.as_slice().iter().all(|value| *value == 0.0));
    drop(initial);
    println!("sgd bench: smoke complete");
}

fn main() {
    println!("SGD prediction/update/resource baseline (benchmark-only; Seqvex only)");
    println!(
        "seed=0x{SEED:016x} fixture_len={FIXTURE_LEN} warm_start={WARM_START} eta={ETA} runs={}",
        common::RUNS
    );
    println!("{}", common::env_summary());
    println!(
        "profile: debug_assertions={} (smoke expects true; release matrix expects false)",
        cfg!(debug_assertions)
    );
    println!(
        "RUSTFLAGS={:?}",
        std::env::var("RUSTFLAGS").unwrap_or_else(|_| "(unset)".to_owned())
    );
    println!(
        "sample unit: run-average per-op latency; update runs are continued-state samples; \
         p95 is over run averages, not individual-op tail; init is repeated warmed init/drop"
    );
    common::verify_workloads();

    if cfg!(debug_assertions) {
        smoke();
        return;
    }

    verify_cyclic_index();

    let mut predict_scaling: Vec<(f64, f64)> = Vec::new();
    let mut update_scaling: Vec<(f64, f64)> = Vec::new();
    for (&d, &steps) in DIMENSIONS.iter().zip(STEPS.iter()) {
        let (predict_median, update_median) = run_dimension(d, steps);
        predict_scaling.push((d as f64, predict_median));
        update_scaling.push((d as f64, update_median));
    }
    println!(
        "\nscaling: predict exponent~{:.3}; update-direct exponent~{:.3}",
        common::log_log_slope(&predict_scaling),
        common::log_log_slope(&update_scaling)
    );
    println!("\nDone. Measurements are synthetic and device-local; not production guarantees.");
}
