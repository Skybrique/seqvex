//! Shared comparison-evidence mechanics for benchmark consumers.
//!
//! Benchmark/test support only. This module owns capture ordering, raw-window
//! bookkeeping, exact registration verification, and evidence-only memory
//! snapshots. Fixtures, algorithms and correctness oracles remain local.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::time::Instant;

use crate::common;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capture {
    One,
    Two,
    Three,
}

impl Capture {
    pub fn number(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

fn capture_from_value(value: &str) -> Capture {
    match value {
        "1" => Capture::One,
        "2" => Capture::Two,
        "3" => Capture::Three,
        other => panic!("SEQVEX_CAPTURE must be 1, 2, or 3, got {other:?}"),
    }
}

pub fn capture_from_env() -> Capture {
    std::env::var("SEQVEX_CAPTURE")
        .map(|value| capture_from_value(&value))
        .unwrap_or(Capture::One)
}

pub fn baseline_first<T: Copy>(capture: Capture, candidates: &[T]) -> Vec<T> {
    let Some((&baseline, rest)) = candidates.split_first() else {
        return Vec::new();
    };
    let mut rest = rest.to_vec();
    if capture == Capture::Two {
        rest.reverse();
    }
    let mut ordered = Vec::with_capacity(candidates.len());
    ordered.push(baseline);
    ordered.extend(rest);
    ordered
}

/// Consume the black-boxed runtime input inside the timed body.
pub fn timed_inputs<T, R>(inputs: T, body: impl FnOnce(T) -> R) -> R {
    body(black_box(inputs))
}

#[derive(Debug, Clone, Copy)]
pub struct RawWindow {
    pub window: u32,
    pub elapsed_ns: u128,
    pub requests: u64,
    pub requested_bytes: u64,
    pub units: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct RowSummary {
    pub median: f64,
    pub iqr: f64,
    pub requests: f64,
    pub requested_bytes: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimingMetadata {
    pub calls: u32,
    pub units: u64,
}

#[derive(Debug, Clone)]
pub struct RowEvidence {
    pub id: String,
    pub raw: Vec<RawWindow>,
    pub summary: RowSummary,
}

/// Run two warm-up windows and twenty measured windows. Raw records are
/// retained in acquisition order; summaries use the existing median/IQR rule.
pub fn run_raw_row(
    id: &str,
    calls: u32,
    units_per_window: u64,
    unit_name: &str,
    mut body: impl FnMut(),
) -> RowEvidence {
    assert!(units_per_window > 0, "{id}: zero denominator");
    for _ in 0..2 {
        for _ in 0..calls {
            body();
        }
    }
    let mut raw = Vec::with_capacity(20);
    for window in 0..20 {
        let before = common::allocation_totals();
        let start = Instant::now();
        for _ in 0..calls {
            body();
        }
        let elapsed_ns = start.elapsed().as_nanos();
        let after = common::allocation_totals();
        raw.push(RawWindow {
            window,
            elapsed_ns,
            requests: after.0 - before.0,
            requested_bytes: after.1 - before.1,
            units: units_per_window,
        });
    }
    for record in &raw {
        println!(
            "{id} window={} elapsed_ns={} requests={} requested_bytes={} units={}",
            record.window, record.elapsed_ns, record.requests, record.requested_bytes, record.units
        );
    }
    let mut elapsed: Vec<f64> = raw
        .iter()
        .map(|record| record.elapsed_ns as f64 / units_per_window as f64)
        .collect();
    let mut requests: Vec<f64> = raw
        .iter()
        .map(|record| record.requests as f64 / units_per_window as f64)
        .collect();
    let mut requested: Vec<f64> = raw
        .iter()
        .map(|record| record.requested_bytes as f64 / units_per_window as f64)
        .collect();
    elapsed.sort_by(|a, b| a.partial_cmp(b).expect("finite elapsed"));
    requests.sort_by(|a, b| a.partial_cmp(b).expect("finite requests"));
    requested.sort_by(|a, b| a.partial_cmp(b).expect("finite bytes"));
    let summary = RowSummary {
        median: elapsed[10],
        iqr: elapsed[15] - elapsed[5],
        requests: requests[10],
        requested_bytes: requested[10],
    };
    println!(
        "{id} median={:.1}ns/{unit_name} iqr={:.1} requests/{unit_name}={:.3} bytes/{unit_name}={:.1}",
        summary.median, summary.iqr, summary.requests, summary.requested_bytes
    );
    RowEvidence {
        id: id.to_owned(),
        raw,
        summary,
    }
}

pub fn verify_row_evidence(
    expected_ids: &[&str],
    evidence: &[RowEvidence],
    expected_windows: usize,
    label: &str,
) {
    let expected: BTreeSet<&str> = expected_ids.iter().copied().collect();
    assert_eq!(
        expected.len(),
        expected_ids.len(),
        "{label}: duplicate expected row identity"
    );
    let mut actual = BTreeSet::new();
    for row in evidence {
        assert!(
            actual.insert(row.id.as_str()),
            "{label}: duplicate actual row identity {}",
            row.id
        );
        assert_eq!(
            row.raw.len(),
            expected_windows,
            "{label}: {} has {} windows, expected {expected_windows}",
            row.id,
            row.raw.len()
        );
        let first = row.raw.first().expect("non-empty verified windows");
        assert!(first.units > 0, "{label}: {} has zero units", row.id);
        assert!(
            row.raw.iter().all(|window| window.units == first.units),
            "{label}: {} has inconsistent units",
            row.id
        );
        assert!(
            row.raw
                .iter()
                .enumerate()
                .all(|(index, window)| window.window == index as u32),
            "{label}: {} raw windows are not in acquisition order",
            row.id
        );
        let mut elapsed: Vec<f64> = row
            .raw
            .iter()
            .map(|window| window.elapsed_ns as f64 / first.units as f64)
            .collect();
        let mut requests: Vec<f64> = row
            .raw
            .iter()
            .map(|window| window.requests as f64 / first.units as f64)
            .collect();
        let mut bytes: Vec<f64> = row
            .raw
            .iter()
            .map(|window| window.requested_bytes as f64 / first.units as f64)
            .collect();
        elapsed.sort_by(|a, b| a.partial_cmp(b).expect("finite elapsed"));
        requests.sort_by(|a, b| a.partial_cmp(b).expect("finite requests"));
        bytes.sort_by(|a, b| a.partial_cmp(b).expect("finite bytes"));
        assert_eq!(
            row.summary.median.to_bits(),
            elapsed[10].to_bits(),
            "{label}: {} median does not match raw windows",
            row.id
        );
        assert_eq!(
            row.summary.iqr.to_bits(),
            (elapsed[15] - elapsed[5]).to_bits(),
            "{label}: {} IQR does not match raw windows",
            row.id
        );
        assert_eq!(
            row.summary.requests.to_bits(),
            requests[10].to_bits(),
            "{label}: {} request summary does not match raw windows",
            row.id
        );
        assert_eq!(
            row.summary.requested_bytes.to_bits(),
            bytes[10].to_bits(),
            "{label}: {} byte summary does not match raw windows",
            row.id
        );
    }
    verify_id_sets(&expected, &actual, label);
}

pub fn verify_registration_set(expected: &[&str], actual: &[&str], label: &str) {
    let expected_set: BTreeSet<&str> = expected.iter().copied().collect();
    let actual_set: BTreeSet<&str> = actual.iter().copied().collect();
    assert_eq!(
        expected_set.len(),
        expected.len(),
        "{label}: duplicate expected registration"
    );
    assert_eq!(
        actual_set.len(),
        actual.len(),
        "{label}: duplicate actual registration"
    );
    verify_id_sets(&expected_set, &actual_set, label);
}

fn verify_id_sets(expected: &BTreeSet<&str>, actual: &BTreeSet<&str>, label: &str) {
    let missing: Vec<_> = expected.difference(actual).copied().collect();
    let unexpected: Vec<_> = actual.difference(expected).copied().collect();
    assert!(
        missing.is_empty(),
        "{label}: missing registrations: {missing:?}"
    );
    assert!(
        unexpected.is_empty(),
        "{label}: unexpected registrations: {unexpected:?}"
    );
}

pub fn verify_timing_metadata(
    expected: &BTreeMap<String, TimingMetadata>,
    actual: &BTreeMap<String, TimingMetadata>,
    label: &str,
) {
    let expected_ids: Vec<&str> = expected.keys().map(String::as_str).collect();
    let actual_ids: Vec<&str> = actual.keys().map(String::as_str).collect();
    verify_registration_set(&expected_ids, &actual_ids, label);
    for (id, expected_metadata) in expected {
        let observed = actual
            .get(id)
            .unwrap_or_else(|| panic!("{label}: missing registration {id}"));
        assert_eq!(
            observed.calls, expected_metadata.calls,
            "{label}: {id}: calls expected={} observed={}",
            expected_metadata.calls, observed.calls
        );
        assert_eq!(
            observed.units, expected_metadata.units,
            "{label}: {id}: units expected={} observed={}",
            expected_metadata.units, observed.units
        );
    }
}

pub fn verify_row_evidence_with_metadata(
    expected: &BTreeMap<String, TimingMetadata>,
    evidence: &[RowEvidence],
    expected_windows: usize,
    label: &str,
) {
    let expected_ids: Vec<&str> = expected.keys().map(String::as_str).collect();
    verify_row_evidence(&expected_ids, evidence, expected_windows, label);
    for row in evidence {
        let expected_metadata = expected
            .get(&row.id)
            .unwrap_or_else(|| panic!("{label}: missing expected metadata for {}", row.id));
        let observed_units = row.raw.first().expect("non-empty verified windows").units;
        assert_eq!(
            observed_units, expected_metadata.units,
            "{label}: {}: units expected={} observed={}",
            row.id, expected_metadata.units, observed_units
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryEvidence {
    pub before: u64,
    pub live: u64,
    pub peak: u64,
    pub after: u64,
    pub retained_bytes: u64,
    pub peak_above_baseline: u64,
}

pub fn checked_delta(value: u64, baseline: u64, label: &str) -> u64 {
    value
        .checked_sub(baseline)
        .unwrap_or_else(|| panic!("{label} below baseline: value={value}, baseline={baseline}"))
}

/// Evidence-only probe: construct once, retain through snapshots, destroy once,
/// verify exact baseline restoration, and return no result object.
pub fn memory_probe<T>(body: impl FnOnce() -> T) -> MemoryEvidence {
    let before = common::live_bytes();
    common::reset_peak();
    let value = body();
    let live = common::live_bytes();
    let peak = common::peak_bytes();
    black_box(&value);
    drop(value);
    let after = common::live_bytes();
    assert_eq!(after, before, "after-drop baseline mismatch");
    MemoryEvidence {
        before,
        live,
        peak,
        after,
        retained_bytes: checked_delta(live, before, "retained memory"),
        peak_above_baseline: checked_delta(peak, before, "peak memory"),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        String::from("<non-string panic>")
    }
}

fn expect_panic(contains: &str, body: impl FnOnce() + std::panic::UnwindSafe) {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(body);
    std::panic::set_hook(previous_hook);
    let message = panic_message(result.expect_err("expected panic"));
    assert!(
        message.contains(contains),
        "expected diagnostic {contains:?}, got {message:?}"
    );
}

/// Synthetic helper checks. The actual consumer registration sets are checked
/// separately by each benchmark before its debug early return.
pub fn comparison_self_check() {
    fn synthetic_row(id: &str, windows: usize, units: u64, summary: RowSummary) -> RowEvidence {
        let raw = (0..windows)
            .map(|index| RawWindow {
                window: index as u32,
                elapsed_ns: index as u128,
                requests: index as u64,
                requested_bytes: index as u64,
                units,
            })
            .collect();
        RowEvidence {
            id: id.to_owned(),
            raw,
            summary,
        }
    }

    let valid_summary = RowSummary {
        median: 10.0,
        iqr: 10.0,
        requests: 10.0,
        requested_bytes: 10.0,
    };
    let valid = synthetic_row("self/row", 20, 1, valid_summary);

    let one = baseline_first(Capture::One, &[1, 2, 3]);
    let two = baseline_first(Capture::Two, &[1, 2, 3]);
    let three = baseline_first(Capture::Three, &[1, 2, 3]);
    assert_eq!(one, vec![1, 2, 3]);
    assert_eq!(two, vec![1, 3, 2]);
    assert_eq!(three, vec![1, 2, 3]);

    verify_row_evidence(
        &["self/row"],
        std::slice::from_ref(&valid),
        20,
        "self-check",
    );

    let expected_metadata =
        BTreeMap::from([("self/row".to_owned(), TimingMetadata { calls: 1, units: 1 })]);
    let actual_metadata = expected_metadata.clone();
    verify_timing_metadata(&expected_metadata, &actual_metadata, "self-check");

    let mut wrong_calls = actual_metadata.clone();
    wrong_calls
        .get_mut("self/row")
        .expect("self timing metadata")
        .calls = 2;
    expect_panic("calls expected=1 observed=2", || {
        verify_timing_metadata(&expected_metadata, &wrong_calls, "self-check");
    });

    let mut wrong_units = actual_metadata.clone();
    wrong_units
        .get_mut("self/row")
        .expect("self timing metadata")
        .units = 8;
    expect_panic("units expected=1 observed=8", || {
        verify_timing_metadata(&expected_metadata, &wrong_units, "self-check");
    });

    let units_eight = synthetic_row(
        "self/row",
        20,
        8,
        RowSummary {
            median: 1.25,
            iqr: 1.25,
            requests: 1.25,
            requested_bytes: 1.25,
        },
    );
    expect_panic("units expected=1 observed=8", || {
        verify_row_evidence_with_metadata(
            &expected_metadata,
            std::slice::from_ref(&units_eight),
            20,
            "self-check",
        );
    });

    expect_panic("missing registrations", || {
        verify_registration_set(&["a", "b"], &["a"], "self-check");
    });
    expect_panic("duplicate actual registration", || {
        verify_registration_set(&["a", "b"], &["a", "a", "b"], "self-check");
    });
    expect_panic("unexpected registrations", || {
        verify_registration_set(&["a"], &["a", "b"], "self-check");
    });
    expect_panic("missing registrations", || {
        verify_row_evidence(
            &["self/row", "self/missing"],
            std::slice::from_ref(&valid),
            20,
            "self-check",
        );
    });
    expect_panic("duplicate actual row", || {
        verify_row_evidence(
            &["self/row"],
            &[valid.clone(), valid.clone()],
            20,
            "self-check",
        );
    });
    expect_panic("unexpected registrations", || {
        verify_row_evidence(
            &["self/row"],
            &[
                valid.clone(),
                synthetic_row("self/unexpected", 20, 1, valid_summary),
            ],
            20,
            "self-check",
        );
    });
    expect_panic("duplicate expected row", || {
        verify_row_evidence(
            &["self/row", "self/row"],
            std::slice::from_ref(&valid),
            20,
            "self-check",
        );
    });
    expect_panic("windows", || {
        verify_row_evidence(
            &["self/row"],
            &[synthetic_row("self/row", 19, 1, valid_summary)],
            20,
            "self-check",
        );
    });
    expect_panic("zero units", || {
        verify_row_evidence(
            &["self/row"],
            &[synthetic_row("self/row", 20, 0, valid_summary)],
            20,
            "self-check",
        );
    });
    let mut inconsistent_units = valid.clone();
    inconsistent_units.raw[1].units = 2;
    expect_panic("inconsistent units", || {
        verify_row_evidence(
            &["self/row"],
            std::slice::from_ref(&inconsistent_units),
            20,
            "self-check",
        );
    });
    let inconsistent_summary = RowSummary {
        median: 0.0,
        ..valid_summary
    };
    expect_panic("median does not match", || {
        verify_row_evidence(
            &["self/row"],
            &[synthetic_row("self/row", 20, 1, inconsistent_summary)],
            20,
            "self-check",
        );
    });
    expect_panic("below baseline", || {
        checked_delta(9, 10, "self-check");
    });
    expect_panic("zero denominator", || {
        run_raw_row("self/zero", 1, 0, "unit", || {});
    });
    expect_panic("SEQVEX_CAPTURE", || {
        capture_from_value("4");
    });

    struct DropCount<'a>(&'a std::cell::Cell<usize>);
    impl Drop for DropCount<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let builds = std::cell::Cell::new(0);
    let drops = std::cell::Cell::new(0);
    let scratch = Vec::<u8>::new();
    let evidence = memory_probe(|| {
        builds.set(builds.get() + 1);
        DropCount(&drops)
    });
    drop(scratch);
    assert_eq!(builds.get(), 1);
    assert_eq!(drops.get(), 1);
    assert_eq!(evidence.after, evidence.before);
    assert_eq!(evidence.retained_bytes, 0);

    println!("comparison helper self-check: PASS");
}

/// Debug-only isolated fault injection. Each branch intentionally fails with
/// its named diagnostic; callers exit immediately and do not run measurements.
pub fn run_failure_case_if_requested() -> bool {
    let Ok(case) = std::env::var("SEQVEX_COMPARISON_FAILURE_CASE") else {
        return false;
    };
    match case.as_str() {
        "external-retention" => {
            let mut retention_slot: Option<Vec<u8>> = None;
            memory_probe(|| {
                let mut retained = Vec::with_capacity(64);
                retained.extend_from_slice(&[1, 2, 3, 4]);
                retention_slot = Some(retained);
                0u8
            });
            panic!("COMPARISON_FAILURE_CASE_UNEXPECTED_SUCCESS: {case}");
        }
        "negative-delta" => {
            checked_delta(9, 10, "negative-delta");
            panic!("COMPARISON_FAILURE_CASE_UNEXPECTED_SUCCESS: {case}");
        }
        "builder-panic" => {
            memory_probe(|| {
                panic!("comparison builder sentinel");
            });
            panic!("COMPARISON_FAILURE_CASE_UNEXPECTED_SUCCESS: {case}");
        }
        other => panic!("unknown failure case: {other}"),
    }
}
