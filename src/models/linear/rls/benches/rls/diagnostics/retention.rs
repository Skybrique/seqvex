//! Retained-record emission and catalog retention diagnostics.

use std::collections::BTreeSet;

use crate::diagnostics::failure::diag_check;
use crate::evidence::{RawWindow, emit_retained_records};
use crate::measurement::{RowIdentity, catalog_report};

fn retention_fixture(
    row: &str,
    category: &'static str,
    unit: &'static str,
    status: &'static str,
    window: Option<usize>,
    peak: Option<u64>,
) -> RawWindow {
    let update = unit == "update";
    let units = u64::from(update);
    RawWindow {
        row_id: row.to_owned(),
        category,
        precision: "f32",
        dimension: 1,
        unit,
        window_index: window,
        observation_start: None,
        observation_end: None,
        scheduled_calls: units,
        attempted_calls: units,
        work_per_call: units,
        scheduled_units: units,
        attempted_units: units,
        successful_units: units,
        failed_units: 0,
        unattempted_units: 0,
        completed_updates: units,
        validation_failure: false,
        postcondition_failure: false,
        first_divergence: None,
        stream_a_attempted: None,
        stream_a_committed: None,
        stream_b_attempted: None,
        stream_b_committed: None,
        timed: false,
        elapsed_ns: None,
        allocation_requests: None,
        requested_bytes: None,
        peak_live_requested_bytes: peak,
        status,
        failure_record_id: None,
        emitted: false,
    }
}

pub(super) fn run_retention_diagnostics() -> bool {
    let mut records = vec![
        retention_fixture(
            "F-Retention-Measured",
            "S",
            "update",
            "valid",
            Some(0),
            None,
        ),
        retention_fixture("F-Retention-Warmup", "S", "update", "warmup", Some(0), None),
        retention_fixture(
            "F-Retention-Memory",
            "memory",
            "memory_probe",
            "valid",
            None,
            Some(64),
        ),
    ];
    let first_emission = emit_retained_records(&mut records);
    let second_emission = emit_retained_records(&mut records);
    let emitted_once =
        first_emission == 3 && second_emission == 0 && records.iter().all(|record| record.emitted);
    let output_ok = diag_check(
        "F-RetentionOutput",
        emitted_once,
        "expected the shared output path to serialize measured-schema, warm-up and memory records exactly once",
    );

    let identity = |row: &str| -> RowIdentity { (row.to_owned(), "f32", 1, 0) };
    let expected: BTreeSet<RowIdentity> = ["F-Retention-A", "F-Retention-B", "F-Retention-C"]
        .into_iter()
        .map(identity)
        .collect();
    let report = |rows: &[&str]| {
        let raws: Vec<RawWindow> = rows
            .iter()
            .map(|row| retention_fixture(row, "S", "update", "valid", Some(0), None))
            .collect();
        catalog_report(&expected, &raws)
    };

    let (missing, extra, duplicates) = report(&["F-Retention-A", "F-Retention-B"]);
    let missing_ok =
        missing == vec![identity("F-Retention-C")] && extra.is_empty() && duplicates.is_empty();
    let (missing, extra, duplicates) = report(&[
        "F-Retention-A",
        "F-Retention-B",
        "F-Retention-C",
        "F-Retention-D",
    ]);
    let extra_ok =
        missing.is_empty() && extra == vec![identity("F-Retention-D")] && duplicates.is_empty();
    let (missing, extra, duplicates) = report(&[
        "F-Retention-A",
        "F-Retention-A",
        "F-Retention-B",
        "F-Retention-C",
    ]);
    let duplicate_ok = missing.is_empty()
        && extra.is_empty()
        && duplicates == vec![(identity("F-Retention-A"), 2)];
    let (missing, extra, duplicates) = report(&["F-Retention-A", "F-Retention-B", "F-Retention-C"]);
    let exact_ok = missing.is_empty() && extra.is_empty() && duplicates.is_empty();
    let catalog_ok = diag_check(
        "F-RetentionCatalog",
        missing_ok && extra_ok && duplicate_ok && exact_ok,
        "expected missing, unexpected, duplicate and exact identity sets to be classified correctly",
    );

    output_ok && catalog_ok
}
