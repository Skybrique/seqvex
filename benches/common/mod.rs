//! Shared benchmark infrastructure for the decision-grade profiling benches.
//!
//! Benchmark-only: deterministic workloads (A persistent excitation, B
//! structured temporal, C long steady state), the project's dependency-free
//! measurement harness (counting allocator, `Instant`, `black_box`, median/IQR),
//! and small analysis helpers. No production code, no abstraction, no new
//! dependency, and no algorithm change.
//!
//! This is a subdirectory module, not a Cargo bench target, so `Cargo.toml` is
//! unchanged. The debug test profile (`cargo test --all-targets`) executes
//! `harness = false` benches; decision numbers are only ever taken from the
//! optimized profile.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        let live =
            LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed) + layout.size() as u64;
        PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        let old_size = layout.size() as u64;
        let new_size_u64 = new_size as u64;
        if new_size_u64 >= old_size {
            let delta = new_size_u64 - old_size;
            let live = LIVE_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
            PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        } else {
            LIVE_BYTES.fetch_sub(old_size - new_size_u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Currently live bytes across the whole process.
pub fn live_bytes() -> u64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// High-water live bytes since the last [`reset_peak`] (or process start).
pub fn peak_bytes() -> u64 {
    PEAK_BYTES.load(Ordering::Relaxed)
}

/// Resets the peak high-water mark to the current live bytes, so the next
/// measurement window reports its own additional peak.
pub fn reset_peak() {
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Cumulative `(allocation count, allocated bytes)` since process start.
pub fn allocation_totals() -> (u64, u64) {
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

/// Repeated timing runs per path/configuration (release only).
pub const RUNS: usize = if cfg!(debug_assertions) { 1 } else { 20 };

/// Pre-registered forgetting factor for the persistent-excitation workload.
///
/// `0.999` keeps the windowed weighted design full-rank across the profiling
/// dimension matrix (effective memory about `1000`). At the largest dimensions
/// the margin is modest and the `f32` reference may still hit its numerical
/// guard; the RLS profiler treats a guard as measured numerical data (it counts
/// and reports it) rather than aborting. This is an experiment-design choice,
/// not a production or criterion change.
pub const LAMBDA_PE: f32 = 0.999;
/// AR(1) coefficient for the structured temporal workload.
pub const AR_COEFFICIENT: f32 = 0.9;
/// Fixed LCG seed for the structured temporal workload.
pub const STRUCTURED_SEED: u64 = 0x5eed_1234_abcd_0001;

/// Shortens the measured step count under the debug test profile. Debug numbers
/// are never used for decisions.
pub fn timed_steps(steps: u32) -> u32 {
    if cfg!(debug_assertions) { 200 } else { steps }
}

/// Fixed-seed linear congruential generator (mirrors the
/// `GruParameters::deterministic` generator); deterministic inputs only, no
/// `rand`.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
    }
}

/// Deterministic target weights `w*`, fixed across runs.
pub fn fixed_weights(d: usize) -> Vec<f32> {
    (0..d).map(|j| (j as f32 * 0.9).sin() + 0.3).collect()
}

/// Fixed LCG seed for the persistent-excitation workload.
pub const EXCITATION_SEED: u64 = 0x5eed_0a11_ce00_0001;

/// Workload A: deterministic dense persistent-excitation features (fixed-seed
/// LCG draws on `[-1, 1)`).
///
/// A dense i.i.d.-style design has a well-conditioned Gram at every profiling
/// dimension, which keeps the `f32` reference free of numerical guards at
/// `D = 256`; a sinusoidal design of the same dimension is not reliably
/// conditioned in `f32`. Determinism comes from the fixed seed and draw order,
/// so the stream is reproducible bit-for-bit and prefix-consistent. This is an
/// experiment-design choice, not a production or criterion change.
pub fn persistent_excitation_features(d: usize, n: usize) -> Vec<Vec<f32>> {
    let mut lcg = Lcg::new(EXCITATION_SEED);
    (0..n)
        .map(|_| (0..d).map(|_| 2.0 * lcg.next_f32()).collect())
        .collect()
}

/// Linear targets `y_t = x_t · w` in input order.
pub fn linear_targets(xs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(weights)
                .map(|(value, weight)| value * weight)
                .sum()
        })
        .collect()
}

/// Workload B: AR(1) structured temporal features with fixed-seed LCG noise.
pub fn structured_features(d: usize, n: usize, coefficient: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut lcg = Lcg::new(seed);
    let mut previous = vec![0.0_f32; d];
    let mut xs = Vec::with_capacity(n);
    for t in 0..n {
        let mut x = vec![0.0_f32; d];
        for (j, value) in x.iter_mut().enumerate() {
            let noise = 0.1 * lcg.next_f32();
            *value = if t == 0 {
                noise
            } else {
                coefficient * previous[j] + noise
            };
        }
        previous.clone_from(&x);
        xs.push(x);
    }
    xs
}

/// Deterministic targets switching from `w_a` to `w_b` at `split`.
pub fn regime_targets(xs: &[Vec<f32>], w_a: &[f32], w_b: &[f32], split: usize) -> Vec<f32> {
    xs.iter()
        .enumerate()
        .map(|(index, x)| {
            let weights = if index < split { w_a } else { w_b };
            x.iter()
                .zip(weights)
                .map(|(value, weight)| value * weight)
                .sum()
        })
        .collect()
}

/// Robust percentile of a sorted, non-empty sample.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

/// Per-observation measurement result.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub median: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    pub iqr: f64,
    pub allocations: f64,
    pub bytes: f64,
}

/// Runs `body` for a warm-up plus `RUNS` timed runs and reports per-observation
/// latency, throughput, and allocations. `work_per_call` is the number of
/// observations each `body` call processes.
pub fn measure(label: &str, steps: u32, work_per_call: usize, body: impl FnMut()) -> Stats {
    measure_runs(label, steps, work_per_call, RUNS, body)
}

/// As [`measure`], but with an explicit run count (evidence phase only).
///
/// The measurement methodology is otherwise identical; the default entry point
/// still uses [`RUNS`]. Used by the Category B evidence decomposition to reach
/// 30+ samples; it does not change the materiality rule.
pub fn measure_runs(
    label: &str,
    steps: u32,
    work_per_call: usize,
    runs: usize,
    mut body: impl FnMut(),
) -> Stats {
    let work = work_per_call as f64;
    let observations = f64::from(steps) * work;

    for _ in 0..(steps / 10).max(if cfg!(debug_assertions) { 50 } else { 1000 }) {
        body();
    }

    let mut samples = Vec::with_capacity(runs);
    let mut allocations = 0.0;
    let mut bytes = 0.0;
    for _ in 0..runs {
        let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
        let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let start = Instant::now();
        for _ in 0..steps {
            body();
        }
        let elapsed = start.elapsed();
        allocations =
            (ALLOCATIONS.load(Ordering::Relaxed) - allocations_before) as f64 / observations;
        bytes = (ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before) as f64 / observations;
        samples.push(elapsed.as_nanos() as f64 / observations);
    }

    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let stats = Stats {
        median: percentile(&samples, 0.5),
        p95: percentile(&samples, 0.95),
        min: samples[0],
        max: samples[samples.len() - 1],
        iqr: percentile(&samples, 0.75) - percentile(&samples, 0.25),
        allocations,
        bytes,
    };
    let Stats {
        median,
        p95,
        min,
        max,
        iqr,
        allocations,
        bytes,
    } = stats;
    println!(
        "{label:<20} {median:>9.1} ns/obs median (p95 {p95:.1}, min {min:.1}, max {max:.1}, iqr {iqr:.1})  \
         {throughput:.0} obs/s  {allocations:.3} allocs/obs  {bytes:.1} bytes/obs",
        throughput = 1e9 / median,
    );
    stats
}

/// Times construction (model + workspace allocation), reported separately from
/// steady state. Never mixed into per-observation numbers.
pub fn measure_startup(label: &str, reps: u32, mut body: impl FnMut()) {
    for _ in 0..reps.min(10) {
        body();
    }
    let mut allocations = 0.0;
    let mut bytes = 0.0;
    let start = Instant::now();
    for _ in 0..reps {
        let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
        let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
        body();
        allocations += (ALLOCATIONS.load(Ordering::Relaxed) - allocations_before) as f64;
        bytes += (ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before) as f64;
    }
    let reps_f = f64::from(reps);
    let nanoseconds = start.elapsed().as_nanos() as f64 / reps_f;
    let allocations = allocations / reps_f;
    let bytes = bytes / reps_f;
    println!(
        "{label:<20} {nanoseconds:>9.1} ns/init  {allocations:.2} allocs/init  {bytes:.0} bytes/init"
    );
}

/// Repetitions for startup (construction) timing.
pub fn startup_reps() -> u32 {
    if cfg!(debug_assertions) { 1 } else { 100 }
}

/// Least-squares slope of `ln(y)` versus `ln(x)` over positive points.
pub fn log_log_slope(points: &[(f64, f64)]) -> f64 {
    let mut count = 0.0_f64;
    let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) = (0.0_f64, 0.0, 0.0, 0.0);
    for &(x, y) in points {
        if x > 0.0 && y > 0.0 {
            let (log_x, log_y) = (x.ln(), y.ln());
            count += 1.0;
            sum_x += log_x;
            sum_y += log_y;
            sum_xx += log_x * log_x;
            sum_xy += log_x * log_y;
        }
    }
    let denominator = count * sum_xx - sum_x * sum_x;
    if denominator == 0.0 {
        0.0
    } else {
        (count * sum_xy - sum_x * sum_y) / denominator
    }
}

/// Approved materiality rule: `2 x IQR` noise proxy **and** `5%` relative delta.
pub fn materiality(median_a: f64, iqr_a: f64, median_b: f64, iqr_b: f64) -> &'static str {
    let difference = (median_a - median_b).abs();
    let noise = 2.0 * iqr_a.max(iqr_b);
    let relative = 0.05 * median_a.max(median_b);
    if difference > noise && difference > relative {
        "clearly measurable"
    } else if difference > noise || difference > relative {
        "borderline/noisy"
    } else {
        "not materially different"
    }
}

/// Environment summary printed at the top of every benchmark.
pub fn env_summary() -> String {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents
                .lines()
                .find(|line| line.starts_with("model name"))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "cpu: (unavailable)".to_owned());
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let parallelism = std::thread::available_parallelism().map_or(0, |value| value.get());
    format!("env os={os} arch={arch} profile={profile} parallelism={parallelism} {cpu}")
}

/// Cheap determinism/finiteness checks for the workload generators. Called from
/// each benchmark `main`; runs under `cargo test --all-targets` in debug.
pub fn verify_workloads() {
    let short = persistent_excitation_features(4, 64);
    let short_again = persistent_excitation_features(4, 64);
    assert_eq!(short, short_again, "workload A must be deterministic");
    assert!(short.iter().flatten().all(|value| value.is_finite()));

    let long = persistent_excitation_features(4, 128);
    assert_eq!(&long[..64], &short[..], "workload C prefix must equal A");

    let structured = structured_features(4, 40, AR_COEFFICIENT, STRUCTURED_SEED);
    let structured_again = structured_features(4, 40, AR_COEFFICIENT, STRUCTURED_SEED);
    assert_eq!(
        structured, structured_again,
        "workload B must be deterministic"
    );
    assert!(structured.iter().flatten().all(|value| value.is_finite()));

    let w_a = fixed_weights(4);
    let mut w_b = fixed_weights(4);
    w_b[0] += 4.0;
    let split = 20;
    let targets = regime_targets(&structured, &w_a, &w_b, split);
    let before: f32 = structured[split - 1]
        .iter()
        .zip(&w_a)
        .map(|(value, weight)| value * weight)
        .sum();
    let after: f32 = structured[split]
        .iter()
        .zip(&w_b)
        .map(|(value, weight)| value * weight)
        .sum();
    assert_eq!(
        targets[split - 1],
        before,
        "regime switch must begin at split"
    );
    assert_eq!(
        targets[split], after,
        "regime switch must apply after split"
    );
}
