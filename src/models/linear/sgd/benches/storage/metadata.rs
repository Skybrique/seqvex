//! Registration metadata, probe catalogs and evidence verification.

use std::collections::{BTreeMap, BTreeSet};

use super::candidates::Candidate;
use super::common::comparison::{
    RowEvidence, TimingMetadata, verify_registration_set, verify_row_evidence_with_metadata,
    verify_timing_metadata,
};
use super::{DETACHED_DIMS, DIMS};

const PRECISIONS: [&str; 2] = ["f32", "f64"];
const CANDIDATE_NAMES: [&str; 5] = ["baseline", "direct", "array", "reuse", "tinyvec"];

pub(super) fn detached_id(precision: &str, candidate: Candidate, d: usize) -> String {
    format!("storage/detached/{precision}/{}/dim={d}", candidate.name())
}

pub(super) fn sequence_id(
    sequence: usize,
    precision: &str,
    candidate: Candidate,
    d: usize,
) -> String {
    format!(
        "storage/seq{sequence}/{precision}/{}/dim={d}",
        candidate.name()
    )
}

const EXPECTED_DETACHED_CALLS: u32 = 4096;
const EXPECTED_DETACHED_UNITS: u64 = 4096;
const EXPECTED_SEQUENCE_METADATA: [(usize, u32, u64); 2] = [(32, 32, 1024), (256, 8, 2048)];

fn insert_timing(
    catalog: &mut BTreeMap<String, TimingMetadata>,
    id: String,
    metadata: TimingMetadata,
) {
    assert!(
        catalog.insert(id.clone(), metadata).is_none(),
        "duplicate SGD timing registration {id}"
    );
}

pub(super) fn timing_catalog() -> BTreeMap<String, TimingMetadata> {
    let mut catalog = BTreeMap::new();
    for precision in PRECISIONS {
        for d in DIMS {
            if DETACHED_DIMS.contains(&d) {
                for candidate in Candidate::ALL {
                    if candidate.applicable(d) {
                        insert_timing(
                            &mut catalog,
                            detached_id(precision, candidate, d),
                            TimingMetadata {
                                // Actual-side detached contract; intentionally independent
                                // of the expected declarations below.
                                calls: 4096,
                                units: 4096,
                            },
                        );
                    }
                }
            }
            for sequence in [32, 256] {
                let calls = if sequence == 32 { 32 } else { 8 };
                let units = u64::from(calls) * sequence as u64;
                for candidate in Candidate::ALL {
                    if candidate.applicable(d) {
                        insert_timing(
                            &mut catalog,
                            sequence_id(sequence, precision, candidate, d),
                            TimingMetadata { calls, units },
                        );
                    }
                }
            }
        }
    }
    catalog
}

pub(super) fn probe_catalog() -> BTreeSet<String> {
    let mut catalog = BTreeSet::new();
    for precision in PRECISIONS {
        for d in DIMS {
            for candidate in Candidate::ALL {
                if candidate.applicable(d) {
                    assert!(catalog.insert(sequence_id(256, precision, candidate, d)));
                }
            }
        }
    }
    catalog
}

fn expected_timing_metadata() -> BTreeMap<String, TimingMetadata> {
    let mut expected = BTreeMap::new();
    let mut insert = |id: String, calls: u32, units: u64| {
        assert!(
            expected
                .insert(id.clone(), TimingMetadata { calls, units })
                .is_none(),
            "duplicate expected SGD timing registration {id}"
        );
    };
    for precision in PRECISIONS {
        for d in DIMS {
            if DETACHED_DIMS.contains(&d) {
                for candidate in CANDIDATE_NAMES {
                    if candidate != "array" || d <= 8 {
                        insert(
                            format!("storage/detached/{precision}/{candidate}/dim={d}"),
                            EXPECTED_DETACHED_CALLS,
                            EXPECTED_DETACHED_UNITS,
                        );
                    }
                }
            }
            for (sequence, calls, units) in EXPECTED_SEQUENCE_METADATA {
                for candidate in CANDIDATE_NAMES {
                    if candidate != "array" || d <= 8 {
                        insert(
                            format!("storage/seq{sequence}/{precision}/{candidate}/dim={d}"),
                            calls,
                            units,
                        );
                    }
                }
            }
        }
    }
    assert_eq!(expected.len(), 156);
    expected
}

fn expected_probe_ids() -> Vec<String> {
    let mut ids = Vec::with_capacity(64);
    for precision in PRECISIONS {
        for d in DIMS {
            for candidate in CANDIDATE_NAMES {
                if candidate != "array" || d <= 8 {
                    ids.push(format!("storage/seq256/{precision}/{candidate}/dim={d}"));
                }
            }
        }
    }
    ids
}

fn expect_metadata_panic(expected: &str, body: impl FnOnce()) {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
    std::panic::set_hook(previous_hook);
    let message = result.expect_err("expected metadata panic");
    let message = message
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            message
                .downcast_ref::<&str>()
                .map(|value| (*value).to_owned())
        })
        .unwrap_or_default();
    assert!(
        message.contains(expected),
        "expected metadata diagnostic {expected:?}, got {message:?}"
    );
}

pub(super) fn verify_detached_metadata_adversarial() {
    let expected = expected_timing_metadata();
    let actual = timing_catalog();
    let detached: Vec<String> = actual
        .keys()
        .filter(|id| id.starts_with("storage/detached/"))
        .cloned()
        .collect();
    assert_eq!(detached.len(), 28, "detached registration coverage");
    for id in detached {
        let expected_meta = expected
            .get(&id)
            .unwrap_or_else(|| panic!("missing detached expectation {id}"));
        assert_eq!(expected_meta.calls, 4096, "detached calls contract {id}");
        assert_eq!(expected_meta.units, 4096, "detached units contract {id}");
        assert_eq!(
            actual.get(&id),
            Some(expected_meta),
            "detached actual/expected metadata mismatch {id}"
        );

        let mut wrong_calls = actual.clone();
        wrong_calls
            .get_mut(&id)
            .expect("detached calls entry")
            .calls = 4097;
        expect_metadata_panic("calls expected=4096 observed=4097", || {
            verify_timing_metadata(&expected, &wrong_calls, "detached calls adversarial");
        });

        let mut wrong_units = actual.clone();
        wrong_units
            .get_mut(&id)
            .expect("detached units entry")
            .units = 4097;
        expect_metadata_panic("units expected=4096 observed=4097", || {
            verify_timing_metadata(&expected, &wrong_units, "detached units adversarial");
        });
    }
    println!(
        "detached metadata adversarial PASS: 28 registrations, calls/units mutations rejected"
    );
}

pub(super) fn verify_registrations() {
    let expected_timing = expected_timing_metadata();
    let timing = timing_catalog();
    verify_timing_metadata(&expected_timing, &timing, "SGD timing registrations");
    let expected_probes = expected_probe_ids();
    let probes = probe_catalog();
    let actual_probes: Vec<&str> = probes.iter().map(String::as_str).collect();
    let expected_refs: Vec<&str> = expected_probes.iter().map(String::as_str).collect();
    assert_eq!(expected_probes.len(), 64);
    verify_registration_set(&expected_refs, &actual_probes, "SGD probe registrations");
    println!("registration PASS: SGD 156 timing rows, 64 memory probes");
}

pub(super) fn verify_actual_evidence(evidence: &[RowEvidence], probe_ids: &[String]) {
    let expected_timing = expected_timing_metadata();
    verify_row_evidence_with_metadata(&expected_timing, evidence, 20, "SGD actual timing rows");
    let expected_probes = expected_probe_ids();
    let expected_refs: Vec<&str> = expected_probes.iter().map(String::as_str).collect();
    let actual_probes: Vec<&str> = probe_ids.iter().map(String::as_str).collect();
    verify_registration_set(&expected_refs, &actual_probes, "SGD actual probe execution");
}
