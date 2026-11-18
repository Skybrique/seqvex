use std::collections::{BTreeMap, BTreeSet};

use crate::common::comparison::{
    RowEvidence, TimingMetadata, verify_registration_set, verify_row_evidence_with_metadata,
    verify_timing_metadata,
};

use super::candidates::Candidate;
use super::fixtures::{FIXTURE_NAMES, fixture_calls, fixture_rows};

pub(super) const CANDIDATE_NAMES: [&str; 4] = ["baseline", "string", "bytes", "serde"];
pub(super) const WORKFLOW_NAMES: [&str; 3] = ["a", "c", "ab"];
pub(super) const MODEL_IDS: [&str; 6] = [
    "model/afit/csv-can-8",
    "model/cfit/csv-can-8",
    "model/bconvert/csv-can-8",
    "model/aserve/csv-can-8",
    "model/bserve/csv-can-8",
    "model/cserve/csv-can-8",
];
pub(super) fn parse_id(candidate: Candidate, fixture: &str) -> String {
    format!("ingest/parse/{}/{fixture}", candidate.name())
}

pub(super) fn complete_id(candidate: Candidate, fixture: &str) -> String {
    format!("ingest/complete/{}/{fixture}", candidate.name())
}

pub(super) fn materialize_id(fixture: &str) -> String {
    format!("ingest/materialize/baseline/{fixture}")
}

pub(super) fn workflow_id(candidate: Candidate, workflow: &str) -> String {
    format!("workflow/{workflow}/{}/csv-can-8", candidate.name())
}

pub(super) const EXPECTED_INGEST_METADATA: [(&str, u32, u64); 3] = [
    ("csv-can-8", 64, 512),
    ("csv-rep-32", 16, 512),
    ("csv-rep-256", 4, 1024),
];
pub(super) const EXPECTED_MODEL_METADATA: [(&str, u32, u64); 6] = [
    ("model/afit/csv-can-8", 16, 16),
    ("model/cfit/csv-can-8", 16, 16),
    ("model/bconvert/csv-can-8", 16, 16),
    ("model/aserve/csv-can-8", 16, 48),
    ("model/bserve/csv-can-8", 16, 48),
    ("model/cserve/csv-can-8", 16, 48),
];

fn insert_timing(
    catalog: &mut BTreeMap<String, TimingMetadata>,
    id: String,
    metadata: TimingMetadata,
) {
    assert!(
        catalog.insert(id.clone(), metadata).is_none(),
        "duplicate OLS timing registration {id}"
    );
}

pub(super) fn timing_catalog() -> BTreeMap<String, TimingMetadata> {
    let mut catalog = BTreeMap::new();
    for candidate in Candidate::ALL {
        for fixture in FIXTURE_NAMES {
            let rows = fixture_rows(fixture);
            let calls = fixture_calls(rows);
            let units = u64::from(calls) * rows as u64;
            let metadata = TimingMetadata { calls, units };
            insert_timing(&mut catalog, parse_id(candidate, fixture), metadata);
            insert_timing(&mut catalog, complete_id(candidate, fixture), metadata);
        }
    }
    for fixture in FIXTURE_NAMES {
        let rows = fixture_rows(fixture);
        let calls = fixture_calls(rows);
        let units = u64::from(calls) * rows as u64;
        insert_timing(
            &mut catalog,
            materialize_id(fixture),
            TimingMetadata { calls, units },
        );
    }
    for id in MODEL_IDS {
        let units = if id.ends_with("serve/csv-can-8") {
            48
        } else {
            16
        };
        insert_timing(
            &mut catalog,
            id.to_owned(),
            TimingMetadata { calls: 16, units },
        );
    }
    for candidate in Candidate::ALL {
        for workflow in WORKFLOW_NAMES {
            insert_timing(
                &mut catalog,
                workflow_id(candidate, workflow),
                TimingMetadata { calls: 8, units: 1 },
            );
        }
    }
    catalog
}

pub(super) fn probe_catalog() -> BTreeSet<String> {
    let mut catalog = BTreeSet::new();
    for candidate in Candidate::ALL {
        for fixture in FIXTURE_NAMES {
            assert!(catalog.insert(complete_id(candidate, fixture)));
        }
    }
    for fixture in FIXTURE_NAMES {
        assert!(catalog.insert(materialize_id(fixture)));
    }
    for candidate in Candidate::ALL {
        for workflow in WORKFLOW_NAMES {
            assert!(catalog.insert(workflow_id(candidate, workflow)));
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
            "duplicate expected OLS timing registration {id}"
        );
    };
    for candidate in CANDIDATE_NAMES {
        for (fixture, calls, units) in EXPECTED_INGEST_METADATA {
            insert(format!("ingest/parse/{candidate}/{fixture}"), calls, units);
            insert(
                format!("ingest/complete/{candidate}/{fixture}"),
                calls,
                units,
            );
        }
    }
    for (fixture, calls, units) in EXPECTED_INGEST_METADATA {
        insert(
            format!("ingest/materialize/baseline/{fixture}"),
            calls,
            units,
        );
    }
    for (id, calls, units) in EXPECTED_MODEL_METADATA {
        insert(id.to_owned(), calls, units);
    }
    for candidate in CANDIDATE_NAMES {
        for workflow in WORKFLOW_NAMES {
            insert(format!("workflow/{workflow}/{candidate}/csv-can-8"), 8, 1);
        }
    }
    assert_eq!(expected.len(), 45);
    expected
}

fn expected_probe_ids() -> Vec<String> {
    let mut ids = Vec::with_capacity(27);
    for candidate in CANDIDATE_NAMES {
        for fixture in FIXTURE_NAMES {
            ids.push(format!("ingest/complete/{candidate}/{fixture}"));
        }
    }
    for fixture in FIXTURE_NAMES {
        ids.push(format!("ingest/materialize/baseline/{fixture}"));
    }
    for candidate in CANDIDATE_NAMES {
        for workflow in WORKFLOW_NAMES {
            ids.push(format!("workflow/{workflow}/{candidate}/csv-can-8"));
        }
    }
    ids
}

pub(super) fn verify_registrations() {
    let expected_timing = expected_timing_metadata();
    let timing = timing_catalog();
    verify_timing_metadata(&expected_timing, &timing, "OLS timing registrations");

    let expected_probes = expected_probe_ids();
    let probes = probe_catalog();
    let actual_probes: Vec<&str> = probes.iter().map(String::as_str).collect();
    let expected_probe_refs: Vec<&str> = expected_probes.iter().map(String::as_str).collect();
    assert_eq!(expected_probes.len(), 27);
    verify_registration_set(
        &expected_probe_refs,
        &actual_probes,
        "OLS probe registrations",
    );
    println!("registration PASS: OLS 45 timing rows, 27 memory probes");
}

pub(super) fn verify_actual_evidence(evidence: &[RowEvidence], probe_ids: &[String]) {
    let expected_timing = expected_timing_metadata();
    verify_row_evidence_with_metadata(&expected_timing, evidence, 20, "OLS actual timing rows");
    let expected_probes = expected_probe_ids();
    let expected_probe_refs: Vec<&str> = expected_probes.iter().map(String::as_str).collect();
    let actual_probe_refs: Vec<&str> = probe_ids.iter().map(String::as_str).collect();
    verify_registration_set(
        &expected_probe_refs,
        &actual_probe_refs,
        "OLS actual probe execution",
    );
}
