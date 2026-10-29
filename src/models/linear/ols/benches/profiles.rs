//! OLS profile-specific performance/resource evidence (benchmark-only).
//!
//! Debug builds run bounded correctness guards only. Release execution emits a
//! fixed 41-row matrix over native-f64 A, strict-f32 C, explicit B conversion,
//! A/B/C checked serving, and matched legacy hybrid fitting. Window summaries
//! are not individual-request tails. No production code or optimization is
//! changed by this target.

use std::hint::black_box;
use std::time::Instant;

use sha2::{Digest, Sha256};

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, FitConfig, FitConfig32, FitConfig64,
    FitResult, FitResult32, FitResult64, LinearRegression, LinearRegression64, UnderflowPolicy,
    fit, fit_f32, fit_f64,
};

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

const PROBE_COUNT: usize = 128;
const WARMUP_WINDOWS: usize = 2;
const MEASURED_WINDOWS: usize = 20;
const FIXTURE_LABEL: &str = "dense_q32_residual_v1";
const FIXTURE_MAGIC: &[u8] = b"SEQVEX-OLS-PROFILE-M1\0";
const CELLS: [(usize, usize); 3] = [(1024, 8), (16384, 8), (1024, 128)];

struct Fixture {
    n: usize,
    d: usize,
    all_features: Vec<Vec<f32>>,
    all_targets: Vec<f32>,
    features32: Vec<Vector>,
    targets32: Vec<f32>,
    features64: Vec<Vector64>,
    targets64: Vec<f64>,
    probes32: Vec<Vector>,
    probes64: Vec<Vector64>,
    selection: Vec<usize>,
}

fn widen(row: &[f32]) -> Vector64 {
    let values: Vec<f64> = row.iter().map(|&value| f64::from(value)).collect();
    Vector64::from_slice(&values)
}

fn build_fixture(n: usize, d: usize) -> Fixture {
    let all_features = common::persistent_excitation_features(d, n + PROBE_COUNT);
    let weights = common::fixed_weights(d);
    let mut all_targets = common::linear_targets(&all_features, &weights);
    for (index, target) in all_targets.iter_mut().enumerate() {
        *target += 1.0e-3 * ((index as f32) * 0.01).sin();
    }

    let features32: Vec<Vector> = all_features[..n]
        .iter()
        .map(|row| Vector::from_slice(row))
        .collect();
    let targets32 = all_targets[..n].to_vec();
    let features64: Vec<Vector64> = all_features[..n].iter().map(|row| widen(row)).collect();
    let targets64 = all_targets[..n]
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    let probes32: Vec<Vector> = all_features[n..]
        .iter()
        .map(|row| Vector::from_slice(row))
        .collect();
    let probes64: Vec<Vector64> = all_features[n..].iter().map(|row| widen(row)).collect();

    Fixture {
        n,
        d,
        all_features,
        all_targets,
        features32,
        targets32,
        features64,
        targets64,
        probes32,
        probes64,
        selection: (0..n).collect(),
    }
}

fn fixture_key(fixture: &Fixture) -> String {
    format!("(N={},d={})", fixture.n, fixture.d)
}

fn fixture_digest(fixture: &Fixture) -> String {
    let mut hasher = Sha256::new();
    hasher.update(FIXTURE_MAGIC);
    hasher.update(1_u16.to_le_bytes());
    hasher.update((fixture.n as u32).to_le_bytes());
    hasher.update((fixture.d as u32).to_le_bytes());
    hasher.update((PROBE_COUNT as u32).to_le_bytes());
    hasher.update(((fixture.n + PROBE_COUNT) as u32).to_le_bytes());
    hasher.update(common::EXCITATION_SEED.to_le_bytes());
    hasher.update((fixture.selection.len() as u32).to_le_bytes());
    for &index in &fixture.selection {
        hasher.update((index as u32).to_le_bytes());
    }
    hasher.update((PROBE_COUNT as u32).to_le_bytes());
    for index in fixture.n..(fixture.n + PROBE_COUNT) {
        hasher.update((index as u32).to_le_bytes());
    }
    for index in 0..(fixture.n + PROBE_COUNT) {
        hasher.update((index as u32).to_le_bytes());
        for &value in &fixture.all_features[index] {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(fixture.all_targets[index].to_bits().to_le_bytes());
        for &value in &fixture.all_features[index] {
            hasher.update(f64::from(value).to_bits().to_le_bytes());
        }
        hasher.update(
            f64::from(fixture.all_targets[index])
                .to_bits()
                .to_le_bytes(),
        );
    }
    format!("{:x}", hasher.finalize())
}

fn fit_a(fixture: &Fixture) -> FitResult64 {
    fit_f64(
        &fixture.features64,
        &fixture.targets64,
        &fixture.selection,
        &FitConfig64::default(),
    )
    .expect("A fit succeeds")
}

fn fit_c(fixture: &Fixture) -> FitResult32 {
    fit_f32(
        &fixture.features32,
        &fixture.targets32,
        &fixture.selection,
        &FitConfig32::default(),
    )
    .expect("C fit succeeds")
}

fn fit_legacy(fixture: &Fixture) -> FitResult {
    fit(
        &fixture.features32,
        &fixture.targets32,
        &fixture.selection,
        &FitConfig::default(),
    )
    .expect("legacy fit succeeds")
}

fn check_fit_a(result: &FitResult64, fixture: &Fixture) {
    let diagnostics = result.diagnostics();
    assert_eq!(diagnostics.rows, fixture.n);
    assert_eq!(diagnostics.features, fixture.d);
    assert!(diagnostics.intercept);
    assert_eq!(diagnostics.rank, diagnostics.required_rank);
    assert!(diagnostics.rank_threshold.is_finite());
    assert!(diagnostics.rss_f64.is_finite());
    assert!(
        result
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.intercept().is_finite());
    let mut rss = 0.0f64;
    for (row, &target) in fixture.features64.iter().zip(&fixture.targets64) {
        let prediction = result
            .weights()
            .as_slice()
            .iter()
            .zip(row.as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f64>()
            + result.intercept();
        let residual = prediction - target;
        rss += residual * residual;
    }
    assert_eq!(rss.to_bits(), diagnostics.rss_f64.to_bits());
    let predictor = result.predictor();
    for probe in &fixture.probes64 {
        assert!(
            predictor
                .predict_checked(probe)
                .expect("A measured prediction")
                .is_finite()
        );
    }
}

fn check_fit_c(result: &FitResult32, fixture: &Fixture) {
    let diagnostics = result.diagnostics();
    assert_eq!(diagnostics.rows, fixture.n);
    assert_eq!(diagnostics.features, fixture.d);
    assert!(diagnostics.intercept);
    assert_eq!(diagnostics.rank, diagnostics.required_rank);
    assert!(diagnostics.rank_threshold.is_finite());
    assert!(diagnostics.rss_f32.is_finite());
    assert!(
        result
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.intercept().is_finite());
    let mut rss = 0.0f32;
    for (row, &target) in fixture.features32.iter().zip(&fixture.targets32) {
        let prediction = result
            .weights()
            .as_slice()
            .iter()
            .zip(row.as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + result.intercept();
        let residual = prediction - target;
        rss += residual * residual;
    }
    assert_eq!(rss.to_bits(), diagnostics.rss_f32.to_bits());
    let predictor = result.predictor();
    for probe in &fixture.probes32 {
        assert!(
            predictor
                .predict_checked(probe)
                .expect("C measured prediction")
                .is_finite()
        );
    }
}

fn check_fit_legacy(result: &FitResult, fixture: &Fixture) {
    let diagnostics = result.diagnostics();
    assert_eq!(diagnostics.rows, fixture.n);
    assert_eq!(diagnostics.features, fixture.d);
    assert!(diagnostics.intercept);
    assert_eq!(diagnostics.rank, diagnostics.required_rank);
    assert!(diagnostics.rank_threshold.is_finite());
    assert!(diagnostics.rss_f64_reference.is_finite());
    assert!(diagnostics.rss_f32_predictor.is_finite());
    assert!(
        result
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.intercept().is_finite());
    let predictor = result.predictor();
    let mut rss = 0.0f64;
    for (row, &target) in fixture.features32.iter().zip(&fixture.targets32) {
        let prediction = predictor.predict(row).expect("legacy RSS prediction");
        let residual = f64::from(prediction) - f64::from(target);
        rss += residual * residual;
    }
    assert_eq!(rss.to_bits(), diagnostics.rss_f32_predictor.to_bits());
    for probe in &fixture.probes32 {
        assert!(
            predictor
                .predict_checked(probe)
                .expect("legacy measured prediction")
                .is_finite()
        );
    }
}

fn abs32(actual: f32, expected: f32, tolerance: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, got {actual}"
    );
}

fn abs64(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, got {actual}"
    );
}

fn check_serving32(model: &LinearRegression, probes: &[Vector]) {
    let observations: Vec<Observation<Vector>> =
        probes.iter().cloned().map(Observation::new).collect();
    let singles: Vec<u32> = probes
        .iter()
        .map(|probe| {
            model
                .predict_checked(probe)
                .expect("single succeeds")
                .to_bits()
        })
        .collect();
    let checked = CheckedLinearRegression::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let streamed: Vec<u32> = observations
        .iter()
        .map(|observation| (*executor.process_one(observation).expect("stream succeeds")).to_bits())
        .collect();
    assert_eq!(streamed, singles, "A/B/C f32 stream vs single");
    assert_eq!(
        executor.state().to_bits(),
        *singles.last().expect("nonempty probes"),
        "f32 final executor state"
    );
    for &batch in &[1usize, 32, 128] {
        let group: Vec<Observation<Vector>> = (0..batch)
            .map(|index| observations[index % observations.len()].clone())
            .collect();
        let grouped: Vec<u32> = model
            .predict_batch_checked(&group)
            .expect("group succeeds")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected: Vec<u32> = (0..batch)
            .map(|index| singles[index % singles.len()])
            .collect();
        assert_eq!(grouped, expected, "f32 group {batch} vs single");
    }
}

fn check_serving64(model: &LinearRegression64, probes: &[Vector64]) {
    let observations: Vec<Observation<Vector64>> =
        probes.iter().cloned().map(Observation::new).collect();
    let singles: Vec<u64> = probes
        .iter()
        .map(|probe| {
            model
                .predict_checked(probe)
                .expect("single succeeds")
                .to_bits()
        })
        .collect();
    let checked = CheckedLinearRegression64::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f64);
    let streamed: Vec<u64> = observations
        .iter()
        .map(|observation| (*executor.process_one(observation).expect("stream succeeds")).to_bits())
        .collect();
    assert_eq!(streamed, singles, "A stream vs single");
    assert_eq!(
        executor.state().to_bits(),
        *singles.last().expect("nonempty probes"),
        "A final executor state"
    );
    for &batch in &[1usize, 32, 128] {
        let group: Vec<Observation<Vector64>> = (0..batch)
            .map(|index| observations[index % observations.len()].clone())
            .collect();
        let grouped: Vec<u64> = model
            .predict_batch_checked(&group)
            .expect("group succeeds")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected: Vec<u64> = (0..batch)
            .map(|index| singles[index % singles.len()])
            .collect();
        assert_eq!(grouped, expected, "A group {batch} vs single");
    }
}

fn verify_profiles() {
    let features32 = vec![
        Vector::from_slice(&[0.0, 0.0]),
        Vector::from_slice(&[1.0, 0.0]),
        Vector::from_slice(&[0.0, 1.0]),
        Vector::from_slice(&[1.0, 1.0]),
        Vector::from_slice(&[2.0, 1.0]),
        Vector::from_slice(&[1.0, 2.0]),
    ];
    let features64: Vec<Vector64> = features32.iter().map(|row| widen(row.as_slice())).collect();
    let targets32 = [1.0f32, 3.0, -2.0, 0.0, 2.0, -3.0];
    let targets64: Vec<f64> = targets32.iter().map(|&value| f64::from(value)).collect();

    let a_result = fit_f64(
        &features64,
        &targets64,
        &[0, 1, 2, 3, 4, 5],
        &FitConfig64::default(),
    )
    .expect("analytical A fit");
    abs64(a_result.weights().as_slice()[0], 2.0, 1.0e-12, "A w0");
    abs64(a_result.weights().as_slice()[1], -3.0, 1.0e-12, "A w1");
    abs64(a_result.intercept(), 1.0, 1.0e-12, "A bias");
    abs64(a_result.diagnostics().rss_f64, 0.0, 1.0e-12, "A rss");

    let c_result = fit_f32(
        &features32,
        &targets32,
        &[0, 1, 2, 3, 4, 5],
        &FitConfig32::default(),
    )
    .expect("analytical C fit");
    abs32(c_result.weights().as_slice()[0], 2.0, 1.0e-5, "C w0");
    abs32(c_result.weights().as_slice()[1], -3.0, 1.0e-5, "C w1");
    abs32(c_result.intercept(), 1.0, 1.0e-5, "C bias");
    let mut c_rss = 0.0f32;
    for (index, &target) in targets32.iter().enumerate() {
        let prediction = c_result
            .weights()
            .as_slice()
            .iter()
            .zip(features32[index].as_slice())
            .map(|(&weight, &feature)| weight * feature)
            .sum::<f32>()
            + c_result.intercept();
        let residual = prediction - target;
        c_rss += residual * residual;
    }
    assert_eq!(c_result.diagnostics().rss_f32.to_bits(), c_rss.to_bits());
    let c_model = c_result.into_predictor();
    for (index, probe) in features32.iter().enumerate() {
        abs32(
            c_model
                .predict_checked(probe)
                .expect("C analytical checked prediction"),
            targets32[index],
            1.0e-5,
            "C analytical prediction",
        );
    }

    let source_weights: Vec<u64> = a_result
        .weights()
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let source_bias = a_result.intercept().to_bits();
    let b_model = a_result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion succeeds");
    let expected_weights: Vec<f32> = a_result
        .weights()
        .as_slice()
        .iter()
        .map(|&value| value as f32)
        .collect();
    for (actual, expected) in b_model.weights().as_slice().iter().zip(&expected_weights) {
        assert_eq!(actual.to_bits(), expected.to_bits(), "B weight bits");
    }
    assert_eq!(
        b_model.bias().to_bits(),
        (a_result.intercept() as f32).to_bits(),
        "B bias bits"
    );
    let expected_b = LinearRegression::new(
        Vector::from_slice(&expected_weights),
        a_result.intercept() as f32,
    )
    .expect("expected B model");
    for probe in &features32 {
        assert_eq!(
            b_model
                .predict_checked(probe)
                .expect("B prediction")
                .to_bits(),
            expected_b
                .predict_checked(probe)
                .expect("expected B prediction")
                .to_bits(),
            "B prediction bits"
        );
    }
    assert_eq!(
        source_weights,
        a_result
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "B conversion source weights"
    );
    assert_eq!(source_bias, a_result.intercept().to_bits());

    let a_model = a_result.into_predictor();
    check_serving64(&a_model, &features64);
    check_serving32(&b_model, &features32);
    check_serving32(&c_model, &features32);
}

#[derive(Clone, Copy)]
struct Work {
    calls: usize,
    observations: usize,
}

struct Window {
    elapsed_ns: u64,
    allocs: u64,
    bytes: u64,
}

struct Measured {
    windows: Vec<Window>,
    work: Work,
}

fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

fn measure_windows(work: Work, mut op: impl FnMut()) -> Measured {
    for _ in 0..WARMUP_WINDOWS {
        for _ in 0..work.calls {
            op();
        }
    }
    let mut windows = Vec::with_capacity(MEASURED_WINDOWS);
    for _ in 0..MEASURED_WINDOWS {
        let (alloc0, bytes0) = common::allocation_totals();
        let start = Instant::now();
        for _ in 0..work.calls {
            op();
        }
        let elapsed = start.elapsed().as_nanos() as u64;
        let (alloc1, bytes1) = common::allocation_totals();
        windows.push(Window {
            elapsed_ns: elapsed,
            allocs: alloc1 - alloc0,
            bytes: bytes1 - bytes0,
        });
    }
    Measured { windows, work }
}

struct MemoryProbe {
    peak_add: i64,
    retained: i64,
    after_drop: i64,
    allocs: u64,
    bytes: u64,
}

fn probe_memory<T>(mut op: impl FnMut() -> T) -> MemoryProbe {
    let live0 = common::live_bytes();
    common::reset_peak();
    let (alloc0, bytes0) = common::allocation_totals();
    let output = op();
    let peak = common::peak_bytes();
    let live1 = common::live_bytes();
    let (alloc1, bytes1) = common::allocation_totals();
    drop(output);
    let live2 = common::live_bytes();
    MemoryProbe {
        peak_add: peak as i64 - live0 as i64,
        retained: live1 as i64 - live0 as i64,
        after_drop: live2 as i64 - live0 as i64,
        allocs: alloc1 - alloc0,
        bytes: bytes1 - bytes0,
    }
}

fn row_identity(profile: &str, operation: &str, fixture: &str, batch: Option<usize>) -> String {
    match batch {
        Some(batch) => {
            format!("profile={profile} operation={operation} fixture={fixture} batch={batch}")
        }
        None => format!("profile={profile} operation={operation} fixture={fixture} batch=none"),
    }
}

fn report(label: &str, measured: &Measured, memory: &MemoryProbe, row_ids: &mut Vec<String>) {
    row_ids.push(label.to_string());
    let raw_elapsed_ns: Vec<u64> = measured
        .windows
        .iter()
        .map(|window| window.elapsed_ns)
        .collect();
    let raw_allocs: Vec<u64> = measured
        .windows
        .iter()
        .map(|window| window.allocs)
        .collect();
    let raw_bytes: Vec<u64> = measured.windows.iter().map(|window| window.bytes).collect();
    let mut call_values: Vec<f64> = raw_elapsed_ns
        .iter()
        .map(|elapsed| *elapsed as f64 / measured.work.calls as f64)
        .collect();
    let obs_values: Vec<f64> = if measured.work.observations == 0 {
        Vec::new()
    } else {
        raw_elapsed_ns
            .iter()
            .map(|elapsed| *elapsed as f64 / measured.work.observations as f64)
            .collect()
    };
    call_values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let obs_summary = if obs_values.is_empty() {
        "n/a".to_string()
    } else {
        let mut values = obs_values.clone();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        format!(
            "median={:.1} min={:.1} max={:.1} iqr={:.1}",
            quantile(&values, 0.5),
            values[0],
            values[values.len() - 1],
            quantile(&values, 0.75) - quantile(&values, 0.25)
        )
    };
    let allocs: u64 = measured.windows.iter().map(|window| window.allocs).sum();
    let bytes: u64 = measured.windows.iter().map(|window| window.bytes).sum();
    let total_calls = (measured.work.calls * MEASURED_WINDOWS) as f64;
    println!(
        "{label}: calls/window={} observations/window={} raw_elapsed_ns={raw_elapsed_ns:?} \
         ns/call median={:.1} min={:.1} max={:.1} iqr={:.1} ns/observation {obs_summary} \
         allocs/call={:.4} bytes/call={:.1} raw_allocs={raw_allocs:?} raw_bytes={raw_bytes:?} \
         peak_add_B={} retained_B={} after_drop_B={} mem_allocs={} mem_bytes={}",
        measured.work.calls,
        measured.work.observations,
        quantile(&call_values, 0.5),
        call_values[0],
        call_values[call_values.len() - 1],
        quantile(&call_values, 0.75) - quantile(&call_values, 0.25),
        allocs as f64 / total_calls,
        bytes as f64 / total_calls,
        memory.peak_add,
        memory.retained,
        memory.after_drop,
        memory.allocs,
        memory.bytes,
    );
}

fn report_fit(
    label: &str,
    rank_record: &str,
    requested: usize,
    native_rss: &str,
    measured: &Measured,
    memory: &MemoryProbe,
    row_ids: &mut Vec<String>,
) {
    println!(
        "{label}: metadata_source=untimed_guard_fit_same_fixture_config {rank_record} requested_workspace_B={requested} native_{native_rss}"
    );
    report(label, measured, memory, row_ids);
}

fn run_fit_rows(fixture: &Fixture, row_ids: &mut Vec<String>) {
    let key = fixture_key(fixture);
    let work = Work {
        calls: 1,
        observations: fixture.n,
    };

    let legacy = fit_legacy(fixture);
    check_fit_legacy(&legacy, fixture);
    let legacy_diagnostics = legacy.diagnostics();
    let legacy_rss = format!(
        "rss_f64_reference={:.17e} rss_f32_predictor={:.17e}",
        legacy_diagnostics.rss_f64_reference, legacy_diagnostics.rss_f32_predictor
    );
    let legacy_rank = format!(
        "rank={} required_rank={} rank_threshold={:.17e}",
        legacy_diagnostics.rank,
        legacy_diagnostics.required_rank,
        legacy_diagnostics.rank_threshold
    );
    let requested = legacy_diagnostics.requested_workspace_bytes;
    drop(legacy);
    let config = FitConfig::default();
    let mut legacy_timed = || {
        let features = black_box(&fixture.features32);
        let targets = black_box(&fixture.targets32);
        let selection = black_box(&fixture.selection);
        drop(black_box(
            fit(features, targets, selection, black_box(&config)).expect("legacy fit succeeds"),
        ));
    };
    let mut legacy_memory = || {
        fit(
            black_box(&fixture.features32),
            black_box(&fixture.targets32),
            black_box(&fixture.selection),
            black_box(&config),
        )
        .expect("legacy fit succeeds")
    };
    let legacy_measured = measure_windows(work, &mut legacy_timed);
    let legacy_probe = probe_memory(&mut legacy_memory);
    let legacy_id = row_identity("legacy_hybrid", "fit", &key, None);
    report_fit(
        &legacy_id,
        &legacy_rank,
        requested,
        &legacy_rss,
        &legacy_measured,
        &legacy_probe,
        row_ids,
    );

    let a = fit_a(fixture);
    check_fit_a(&a, fixture);
    let a_rss = format!("rss_f64={:.17e}", a.diagnostics().rss_f64);
    let a_rank = format!(
        "rank={} required_rank={} rank_threshold={:.17e}",
        a.diagnostics().rank,
        a.diagnostics().required_rank,
        a.diagnostics().rank_threshold
    );
    let requested = a.diagnostics().requested_workspace_bytes;
    drop(a);
    let config = FitConfig64::default();
    let mut a_timed = || {
        let features = black_box(&fixture.features64);
        let targets = black_box(&fixture.targets64);
        let selection = black_box(&fixture.selection);
        drop(black_box(
            fit_f64(features, targets, selection, black_box(&config)).expect("A fit succeeds"),
        ));
    };
    let mut a_memory = || {
        fit_f64(
            black_box(&fixture.features64),
            black_box(&fixture.targets64),
            black_box(&fixture.selection),
            black_box(&config),
        )
        .expect("A fit succeeds")
    };
    let a_measured = measure_windows(work, &mut a_timed);
    let a_probe = probe_memory(&mut a_memory);
    let a_id = row_identity("A", "fit", &key, None);
    report_fit(
        &a_id,
        &a_rank,
        requested,
        &a_rss,
        &a_measured,
        &a_probe,
        row_ids,
    );

    let c = fit_c(fixture);
    check_fit_c(&c, fixture);
    let c_rss = format!("rss_f32={:.9e}", c.diagnostics().rss_f32);
    let c_rank = format!(
        "rank={} required_rank={} rank_threshold={:.9e}",
        c.diagnostics().rank,
        c.diagnostics().required_rank,
        c.diagnostics().rank_threshold
    );
    let requested = c.diagnostics().requested_workspace_bytes;
    drop(c);
    let config = FitConfig32::default();
    let mut c_timed = || {
        let features = black_box(&fixture.features32);
        let targets = black_box(&fixture.targets32);
        let selection = black_box(&fixture.selection);
        drop(black_box(
            fit_f32(features, targets, selection, black_box(&config)).expect("C fit succeeds"),
        ));
    };
    let mut c_memory = || {
        fit_f32(
            black_box(&fixture.features32),
            black_box(&fixture.targets32),
            black_box(&fixture.selection),
            black_box(&config),
        )
        .expect("C fit succeeds")
    };
    let c_measured = measure_windows(work, &mut c_timed);
    let c_probe = probe_memory(&mut c_memory);
    let c_id = row_identity("C", "fit", &key, None);
    report_fit(
        &c_id,
        &c_rank,
        requested,
        &c_rss,
        &c_measured,
        &c_probe,
        row_ids,
    );
}

fn build_group_cycle<T: Clone>(observations: &[T], batch: usize) -> Vec<Vec<T>> {
    (0..(PROBE_COUNT / batch))
        .map(|group| {
            (0..batch)
                .map(|index| observations[group * batch + index].clone())
                .collect()
        })
        .collect()
}

fn run_single32(
    model: &LinearRegression,
    probes: &[Vector],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let mut index = 0;
    let mut timed = || {
        let probe = &probes[index % probes.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_checked(black_box(probe))
                .expect("single succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_checked(black_box(&probes[0]))
            .expect("single memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

fn run_single64(
    model: &LinearRegression64,
    probes: &[Vector64],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let mut index = 0;
    let mut timed = || {
        let probe = &probes[index % probes.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_checked(black_box(probe))
                .expect("single succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_checked(black_box(&probes[0]))
            .expect("single memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

fn run_stream32(
    model: &LinearRegression,
    observations: &[Observation<Vector>],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let checked = CheckedLinearRegression::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let memory_checked = CheckedLinearRegression::new(model);
    let mut memory_executor = StreamingExecutor::new(&memory_checked, 0.0f32);
    let mut index = 0;
    let mut timed = || {
        let observation = &observations[index % observations.len()];
        index += 1;
        black_box(
            *executor
                .process_one(black_box(observation))
                .expect("stream succeeds"),
        );
    };
    let mut memory = || {
        *memory_executor
            .process_one(black_box(&observations[0]))
            .expect("stream memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

fn run_stream64(
    model: &LinearRegression64,
    observations: &[Observation<Vector64>],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let checked = CheckedLinearRegression64::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f64);
    let memory_checked = CheckedLinearRegression64::new(model);
    let mut memory_executor = StreamingExecutor::new(&memory_checked, 0.0f64);
    let mut index = 0;
    let mut timed = || {
        let observation = &observations[index % observations.len()];
        index += 1;
        black_box(
            *executor
                .process_one(black_box(observation))
                .expect("stream succeeds"),
        );
    };
    let mut memory = || {
        *memory_executor
            .process_one(black_box(&observations[0]))
            .expect("stream memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

fn run_groups32(
    model: &LinearRegression,
    observations: &[Observation<Vector>],
    batch: usize,
    fixture: &str,
    profile: &str,
    row_ids: &mut Vec<String>,
) {
    let groups = build_group_cycle(observations, batch);
    let mut index = 0;
    let mut timed = || {
        let group = &groups[index % groups.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_batch_checked(black_box(group))
                .expect("group succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_batch_checked(black_box(&groups[0]))
            .expect("group memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096 / batch,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(
        &row_identity(profile, "checked_group", fixture, Some(batch)),
        &measured,
        &memory,
        row_ids,
    );
}

fn run_groups64(
    model: &LinearRegression64,
    observations: &[Observation<Vector64>],
    batch: usize,
    fixture: &str,
    profile: &str,
    row_ids: &mut Vec<String>,
) {
    let groups = build_group_cycle(observations, batch);
    let mut index = 0;
    let mut timed = || {
        let group = &groups[index % groups.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_batch_checked(black_box(group))
                .expect("group succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_batch_checked(black_box(&groups[0]))
            .expect("group memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096 / batch,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(
        &row_identity(profile, "checked_group", fixture, Some(batch)),
        &measured,
        &memory,
        row_ids,
    );
}

fn source_bits64(result: &FitResult64) -> (Vec<u64>, u64) {
    (
        result
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        result.intercept().to_bits(),
    )
}

fn model_bits64(model: &LinearRegression64) -> (Vec<u64>, u64) {
    (
        model
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        model.bias().to_bits(),
    )
}

fn model_bits32(model: &LinearRegression) -> (Vec<u32>, u32) {
    (
        model
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        model.bias().to_bits(),
    )
}

fn check_b_conversion(source: &FitResult64, model: &LinearRegression, probes: &[Vector]) {
    let source_weights: Vec<f32> = source
        .weights()
        .as_slice()
        .iter()
        .map(|&value| value as f32)
        .collect();
    assert_eq!(
        model.weights().as_slice().len(),
        source_weights.len(),
        "measured B weight length"
    );
    for (actual, expected) in model.weights().as_slice().iter().zip(&source_weights) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "measured B weight bits"
        );
    }
    assert_eq!(
        model.bias().to_bits(),
        (source.intercept() as f32).to_bits(),
        "measured B bias bits"
    );
    let reference = LinearRegression::new(
        Vector::from_slice(&source_weights),
        source.intercept() as f32,
    )
    .expect("measured B reference");
    for probe in probes {
        assert_eq!(
            model
                .predict_checked(probe)
                .expect("measured B prediction")
                .to_bits(),
            reference
                .predict_checked(probe)
                .expect("measured B reference prediction")
                .to_bits(),
            "measured B prediction bits"
        );
    }
}

fn retained_model_heap<T>(mut op: impl FnMut() -> T) -> (T, i64) {
    let live0 = common::live_bytes();
    let output = op();
    (output, common::live_bytes() as i64 - live0 as i64)
}

fn report_common_diagnostics(profile: &str, fixture: &Fixture, predictions: &[f64]) {
    let targets = &fixture.all_targets[fixture.n..];
    assert_eq!(predictions.len(), targets.len());
    let mut rss = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&prediction, &target) in predictions.iter().zip(targets) {
        let target = f64::from(target);
        let residual = prediction - target;
        rss += residual * residual;
        max_abs = max_abs.max(residual.abs());
    }
    let rmse = (rss / predictions.len() as f64).sqrt();
    println!(
        "common_f64_diagnostics profile={profile} fixture={} probes={} rss_f64={:.17e} rmse_f64={:.17e} max_abs_f64={:.17e}",
        fixture_key(fixture),
        predictions.len(),
        rss,
        rmse,
        max_abs
    );
}

fn report_cross_profile_differences(fixture: &Fixture, profiles: &[(&str, &[f64])]) {
    for left in 0..profiles.len() {
        for right in (left + 1)..profiles.len() {
            let (left_name, left_values) = profiles[left];
            let (right_name, right_values) = profiles[right];
            assert_eq!(left_values.len(), right_values.len());
            let mut squared = 0.0f64;
            let mut max_abs = 0.0f64;
            for (&left_value, &right_value) in left_values.iter().zip(right_values) {
                let difference = left_value - right_value;
                squared += difference * difference;
                max_abs = max_abs.max(difference.abs());
            }
            let rmse = (squared / left_values.len() as f64).sqrt();
            println!(
                "cross_profile_prediction_difference fixture={} pair={left_name},{right_name} probes={} rmse_f64={:.17e} max_abs_f64={:.17e}",
                fixture_key(fixture),
                left_values.len(),
                rmse,
                max_abs
            );
        }
    }
}

fn run_conversion_rows(fixture: &Fixture, row_ids: &mut Vec<String>) -> (FitResult64, i64) {
    let key = fixture_key(fixture);
    let (a_result, a_heap) = retained_model_heap(|| fit_a(fixture));
    check_fit_a(&a_result, fixture);
    let source_before = source_bits64(&a_result);
    let b_check = a_result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion check succeeds");
    check_b_conversion(&a_result, &b_check, &fixture.probes32);
    drop(b_check);
    assert_eq!(source_bits64(&a_result), source_before);

    let mut conversion_timed = || {
        let source = black_box(&a_result);
        drop(black_box(
            source
                .to_serving_f32(black_box(UnderflowPolicy::Reject))
                .expect("B conversion succeeds"),
        ));
    };
    let mut conversion_memory = || {
        black_box(&a_result)
            .to_serving_f32(black_box(UnderflowPolicy::Reject))
            .expect("B conversion memory succeeds")
    };
    let conversion = measure_windows(
        Work {
            calls: 256,
            observations: 0,
        },
        &mut conversion_timed,
    );
    let conversion_probe = probe_memory(&mut conversion_memory);
    assert_eq!(source_bits64(&a_result), source_before);
    report(
        &row_identity("B", "conversion", &key, None),
        &conversion,
        &conversion_probe,
        row_ids,
    );
    (a_result, a_heap)
}

fn run_serving_rows(
    fixture: &Fixture,
    a_result: FitResult64,
    a_heap: i64,
    row_ids: &mut Vec<String>,
) {
    let key = fixture_key(fixture);
    let source_before = source_bits64(&a_result);
    let (b_model, b_heap) = retained_model_heap(|| {
        a_result
            .to_serving_f32(UnderflowPolicy::Reject)
            .expect("B serving conversion succeeds")
    });
    check_b_conversion(&a_result, &b_model, &fixture.probes32);
    assert_eq!(source_bits64(&a_result), source_before);
    let a_model = a_result.into_predictor();
    let (c_result, c_heap) = retained_model_heap(|| fit_c(fixture));
    check_fit_c(&c_result, fixture);
    let c_model = c_result.into_predictor();
    let legacy_result = fit_legacy(fixture);
    check_fit_legacy(&legacy_result, fixture);
    let legacy_model = legacy_result.into_predictor();

    println!(
        "retained_serving_model_heap profile=A fixture={key} heap_add_B={a_heap} object_size_B={} logical_payload_B={}",
        std::mem::size_of_val(&a_model),
        std::mem::size_of_val(a_model.weights().as_slice())
    );
    println!(
        "retained_serving_model_heap profile=B fixture={key} heap_add_B={b_heap} object_size_B={} logical_payload_B={}",
        std::mem::size_of_val(&b_model),
        std::mem::size_of_val(b_model.weights().as_slice())
    );
    println!(
        "retained_serving_model_heap profile=C fixture={key} heap_add_B={c_heap} object_size_B={} logical_payload_B={}",
        std::mem::size_of_val(&c_model),
        std::mem::size_of_val(c_model.weights().as_slice())
    );

    let a_predictions: Vec<f64> = fixture
        .probes64
        .iter()
        .map(|probe| a_model.predict_checked(probe).expect("A diagnostic"))
        .collect();
    let b_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(b_model.predict_checked(probe).expect("B diagnostic")))
        .collect();
    let c_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(c_model.predict_checked(probe).expect("C diagnostic")))
        .collect();
    let legacy_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| {
            f64::from(
                legacy_model
                    .predict_checked(probe)
                    .expect("legacy diagnostic"),
            )
        })
        .collect();
    report_common_diagnostics("A", fixture, &a_predictions);
    report_common_diagnostics("B", fixture, &b_predictions);
    report_common_diagnostics("C", fixture, &c_predictions);
    report_common_diagnostics("legacy_hybrid", fixture, &legacy_predictions);
    report_cross_profile_differences(
        fixture,
        &[
            ("A", &a_predictions),
            ("B", &b_predictions),
            ("C", &c_predictions),
            ("legacy_hybrid", &legacy_predictions),
        ],
    );

    let aa = fixture
        .probes32
        .iter()
        .cloned()
        .map(Observation::new)
        .collect::<Vec<_>>();
    let aa64 = fixture
        .probes64
        .iter()
        .cloned()
        .map(Observation::new)
        .collect::<Vec<_>>();

    check_serving64(&a_model, &fixture.probes64);
    check_serving32(&b_model, &fixture.probes32);
    check_serving32(&c_model, &fixture.probes32);
    let a_before = model_bits64(&a_model);
    let b_before = model_bits32(&b_model);
    let c_before = model_bits32(&c_model);

    run_single64(
        &a_model,
        &fixture.probes64,
        &row_identity("A", "checked_single", &key, None),
        row_ids,
    );
    run_single32(
        &b_model,
        &fixture.probes32,
        &row_identity("B", "checked_single", &key, None),
        row_ids,
    );
    run_single32(
        &c_model,
        &fixture.probes32,
        &row_identity("C", "checked_single", &key, None),
        row_ids,
    );
    run_stream64(
        &a_model,
        &aa64,
        &row_identity("A", "checked_stream", &key, None),
        row_ids,
    );
    run_stream32(
        &b_model,
        &aa,
        &row_identity("B", "checked_stream", &key, None),
        row_ids,
    );
    run_stream32(
        &c_model,
        &aa,
        &row_identity("C", "checked_stream", &key, None),
        row_ids,
    );
    for batch in [1usize, 32, 128] {
        run_groups64(&a_model, &aa64, batch, &key, "A", row_ids);
        run_groups32(&b_model, &aa, batch, &key, "B", row_ids);
        run_groups32(&c_model, &aa, batch, &key, "C", row_ids);
    }
    assert_eq!(model_bits64(&a_model), a_before, "A model preserved");
    assert_eq!(model_bits32(&b_model), b_before, "B model preserved");
    assert_eq!(model_bits32(&c_model), c_before, "C model preserved");
}

fn report_anchor_diagnostics(fixture: &Fixture) {
    let a_result = fit_a(fixture);
    check_fit_a(&a_result, fixture);
    let source_before = source_bits64(&a_result);
    let b_model = a_result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("anchor B serving conversion");
    check_b_conversion(&a_result, &b_model, &fixture.probes32);
    assert_eq!(source_bits64(&a_result), source_before);
    let a_model = a_result.into_predictor();
    let c_result = fit_c(fixture);
    check_fit_c(&c_result, fixture);
    let c_model = c_result.into_predictor();
    let legacy_result = fit_legacy(fixture);
    check_fit_legacy(&legacy_result, fixture);
    let legacy_model = legacy_result.into_predictor();

    let a_predictions: Vec<f64> = fixture
        .probes64
        .iter()
        .map(|probe| a_model.predict_checked(probe).expect("anchor A diagnostic"))
        .collect();
    let b_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(b_model.predict_checked(probe).expect("anchor B diagnostic")))
        .collect();
    let c_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(c_model.predict_checked(probe).expect("anchor C diagnostic")))
        .collect();
    let legacy_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| {
            f64::from(
                legacy_model
                    .predict_checked(probe)
                    .expect("anchor legacy diagnostic"),
            )
        })
        .collect();
    report_common_diagnostics("A", fixture, &a_predictions);
    report_common_diagnostics("B", fixture, &b_predictions);
    report_common_diagnostics("C", fixture, &c_predictions);
    report_common_diagnostics("legacy_hybrid", fixture, &legacy_predictions);
    report_cross_profile_differences(
        fixture,
        &[
            ("A", &a_predictions),
            ("B", &b_predictions),
            ("C", &c_predictions),
            ("legacy_hybrid", &legacy_predictions),
        ],
    );
}

fn run_release_matrix() {
    println!("OLS profile-specific resource/performance evidence (synthetic workloads)");
    println!("{}", common::env_summary());
    println!(
        "fixture={FIXTURE_LABEL} seed=0x{:016x} representation=f32_quantized_to_exact_f64 warmup_windows={WARMUP_WINDOWS} measured_windows={MEASURED_WINDOWS}",
        common::EXCITATION_SEED
    );
    verify_profiles();

    let fixtures = CELLS
        .iter()
        .map(|&(n, d)| build_fixture(n, d))
        .collect::<Vec<_>>();
    for fixture in &fixtures {
        println!(
            "fixture={} key={} selection={}..{} probes={}..{} total_rows={} digest={}",
            FIXTURE_LABEL,
            fixture_key(fixture),
            fixture.selection[0],
            fixture.selection[fixture.selection.len() - 1] + 1,
            fixture.n,
            fixture.n + PROBE_COUNT,
            fixture.n + PROBE_COUNT,
            fixture_digest(fixture)
        );
    }
    println!(
        "size_of model A={} B/C={} checked64={} checked32={}",
        std::mem::size_of::<LinearRegression64>(),
        std::mem::size_of::<LinearRegression>(),
        std::mem::size_of::<CheckedLinearRegression64<'static>>(),
        std::mem::size_of::<CheckedLinearRegression<'static>>(),
    );

    let mut row_ids = Vec::new();
    for fixture in &fixtures {
        run_fit_rows(fixture, &mut row_ids);
    }
    let (a8, a8_heap) = run_conversion_rows(&fixtures[0], &mut row_ids);
    let (a128, a128_heap) = run_conversion_rows(&fixtures[2], &mut row_ids);
    run_serving_rows(&fixtures[0], a8, a8_heap, &mut row_ids);
    run_serving_rows(&fixtures[2], a128, a128_heap, &mut row_ids);
    report_anchor_diagnostics(&fixtures[1]);

    assert_eq!(row_ids.len(), 41, "exactly 41 timing/resource rows");
    let mut unique = row_ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 41, "unique timing/resource row identities");
    println!("profile measurement matrix complete rows={}", row_ids.len());
}

fn main() {
    if cfg!(debug_assertions) {
        println!("ols_profiles: bounded correctness guards only; no performance evidence");
        verify_profiles();
        println!("ols_profiles: correctness guards passed; no measurement windows run");
        return;
    }
    run_release_matrix();
}
