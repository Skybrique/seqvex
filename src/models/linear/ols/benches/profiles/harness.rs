use std::time::Instant;

use crate::common;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, LinearRegression, LinearRegression64,
};

use super::fixture::{
    CELLS, FIXTURE_LABEL, PROBE_COUNT, build_fixture, fixture_digest, fixture_key,
};
use super::guards::verify_profiles;
use super::rows::{report_anchor_diagnostics, run_conversion_rows, run_fit_rows, run_serving_rows};

const WARMUP_WINDOWS: usize = 2;
pub(super) const MEASURED_WINDOWS: usize = 20;

#[derive(Clone, Copy)]
pub(super) struct Work {
    pub(super) calls: usize,
    pub(super) observations: usize,
}

pub(super) struct Window {
    pub(super) elapsed_ns: u64,
    pub(super) allocs: u64,
    pub(super) bytes: u64,
}

pub(super) struct Measured {
    pub(super) windows: Vec<Window>,
    pub(super) work: Work,
}

pub(super) fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

pub(super) fn measure_windows(work: Work, mut op: impl FnMut()) -> Measured {
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

pub(super) struct MemoryProbe {
    pub(super) peak_add: i64,
    pub(super) retained: i64,
    pub(super) after_drop: i64,
    pub(super) allocs: u64,
    pub(super) bytes: u64,
}

pub(super) fn probe_memory<T>(mut op: impl FnMut() -> T) -> MemoryProbe {
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

pub(super) fn retained_model_heap<T>(mut op: impl FnMut() -> T) -> (T, i64) {
    let live0 = common::live_bytes();
    let output = op();
    (output, common::live_bytes() as i64 - live0 as i64)
}

pub(super) fn run_release_matrix() {
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
