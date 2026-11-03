//! Bounded CSV ingestion comparison for the reviewed component-design plan.
//!
//! Debug execution runs equivalence guards only. Release execution runs the
//! pre-registered rows; see the implementation-plan row registry.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;

use csv::{ByteRecord, ReaderBuilder, StringRecord};
use seqvex::models::linear::ols::{FitConfig32, FitConfig64, UnderflowPolicy, fit_f32, fit_f64};
use sha2::{Digest, Sha256};

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

use common::comparison::{
    RowEvidence, TimingMetadata, baseline_first, capture_from_env,
    memory_probe as shared_memory_probe, run_raw_row, timed_inputs, verify_registration_set,
    verify_row_evidence_with_metadata, verify_timing_metadata,
};

#[allow(dead_code, unused_imports)]
#[path = "../examples/fit_profiles.rs"]
mod baseline;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Baseline,
    String,
    Bytes,
    Serde,
}

impl Candidate {
    const ALL: [Self; 4] = [Self::Baseline, Self::String, Self::Bytes, Self::Serde];

    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::String => "string",
            Self::Bytes => "bytes",
            Self::Serde => "serde",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TypedRow {
    feature64: f64,
    target64: f64,
    feature32: f32,
    target32: f32,
}

struct Fixture {
    name: &'static str,
    bytes: Vec<u8>,
    rows: usize,
    frozen_sha256: &'static str,
}

fn ordered_candidates() -> Vec<Candidate> {
    baseline_first(capture_from_env(), &Candidate::ALL)
}

const FIXTURE_NAMES: [&str; 3] = ["csv-can-8", "csv-rep-32", "csv-rep-256"];
const CANDIDATE_NAMES: [&str; 4] = ["baseline", "string", "bytes", "serde"];
const WORKFLOW_NAMES: [&str; 3] = ["a", "c", "ab"];
const MODEL_IDS: [&str; 6] = [
    "model/afit/csv-can-8",
    "model/cfit/csv-can-8",
    "model/bconvert/csv-can-8",
    "model/aserve/csv-can-8",
    "model/bserve/csv-can-8",
    "model/cserve/csv-can-8",
];
fn parse_id(candidate: Candidate, fixture: &str) -> String {
    format!("ingest/parse/{}/{fixture}", candidate.name())
}

fn complete_id(candidate: Candidate, fixture: &str) -> String {
    format!("ingest/complete/{}/{fixture}", candidate.name())
}

fn materialize_id(fixture: &str) -> String {
    format!("ingest/materialize/baseline/{fixture}")
}

fn workflow_id(candidate: Candidate, workflow: &str) -> String {
    format!("workflow/{workflow}/{}/csv-can-8", candidate.name())
}

const EXPECTED_INGEST_METADATA: [(&str, u32, u64); 3] = [
    ("csv-can-8", 64, 512),
    ("csv-rep-32", 16, 512),
    ("csv-rep-256", 4, 1024),
];
const EXPECTED_MODEL_METADATA: [(&str, u32, u64); 6] = [
    ("model/afit/csv-can-8", 16, 16),
    ("model/cfit/csv-can-8", 16, 16),
    ("model/bconvert/csv-can-8", 16, 16),
    ("model/aserve/csv-can-8", 16, 48),
    ("model/bserve/csv-can-8", 16, 48),
    ("model/cserve/csv-can-8", 16, 48),
];

fn fixture_rows(fixture: &str) -> usize {
    match fixture {
        "csv-can-8" => 8,
        "csv-rep-32" => 32,
        "csv-rep-256" => 256,
        _ => panic!("unknown fixture {fixture}"),
    }
}

fn fixture_calls(rows: usize) -> u32 {
    match rows {
        8 => 64,
        32 => 16,
        256 => 4,
        _ => panic!("unknown fixture row count {rows}"),
    }
}

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

fn timing_catalog() -> BTreeMap<String, TimingMetadata> {
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

fn probe_catalog() -> BTreeSet<String> {
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

fn verify_registrations() {
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

fn repeat_fixture(rows: usize) -> Vec<u8> {
    let text = std::str::from_utf8(baseline::CANONICAL_CSV).expect("canonical CSV is UTF-8");
    let mut lines = text.lines();
    let header = lines.next().expect("canonical header");
    let data: Vec<&str> = lines.collect();
    assert_eq!(data.len(), 8);
    let mut out = String::with_capacity(header.len() + 1 + rows * 9);
    out.push_str(header);
    out.push('\n');
    for index in 0..rows {
        out.push_str(data[index % data.len()]);
        out.push('\n');
    }
    out.into_bytes()
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "csv-can-8",
            bytes: baseline::CANONICAL_CSV.to_vec(),
            rows: 8,
            frozen_sha256: "d1931cca4a6c08473bf640cb09f77a711138345d741ea9d74c89eb86aa2587b6",
        },
        Fixture {
            name: "csv-rep-32",
            bytes: repeat_fixture(32),
            rows: 32,
            frozen_sha256: "ad6dab9237f4c516cee254dc97e3d274ebaf0d9b67072eed8686efc4aa855bd9",
        },
        Fixture {
            name: "csv-rep-256",
            bytes: repeat_fixture(256),
            rows: 256,
            frozen_sha256: "42e9a68a3bf841081d883f7a5428da469fc83de7e4a40c3f85103b20d0ec60dc",
        },
    ]
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to String");
    }
    out
}

fn header_ok(record: &StringRecord) -> bool {
    record.len() == 2
        && record.get(0) == Some(baseline::FEATURE_HEADER)
        && record.get(1) == Some(baseline::TARGET_HEADER)
}

fn header_error() -> String {
    format!(
        "CSV headers must be exactly [{}, {}] in that order",
        baseline::FEATURE_HEADER,
        baseline::TARGET_HEADER
    )
}

fn parse_rows_baseline(bytes: &[u8]) -> Result<Vec<TypedRow>, String> {
    if bytes.len() > baseline::MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {}",
            bytes.len(),
            baseline::MAX_INPUT_BYTES
        ));
    }
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes);
    let mut records = reader.records();
    let headers = records
        .next()
        .ok_or_else(|| "CSV is empty; expected a header row".to_owned())?
        .map_err(|error| format!("CSV header: {error}"))?;
    if !header_ok(&headers) {
        return Err(header_error());
    }
    let mut rows = Vec::new();
    for (row, record) in records.enumerate() {
        if rows.len() == baseline::MAX_DATA_ROWS {
            return Err(format!(
                "CSV exceeds the {}-record example limit",
                baseline::MAX_DATA_ROWS
            ));
        }
        let record = record.map_err(|error| format!("CSV data row {row}: {error}"))?;
        rows.push(parse_record(&record, row)?);
    }
    if rows.len() < baseline::MIN_DATA_ROWS {
        return Err(format!(
            "CSV has {} data records; at least {} are required",
            rows.len(),
            baseline::MIN_DATA_ROWS
        ));
    }
    Ok(rows)
}

fn parse_record(record: &StringRecord, row: usize) -> Result<TypedRow, String> {
    let feature = record
        .get(0)
        .ok_or_else(|| format!("CSV data row {row}: missing {}", baseline::FEATURE_HEADER))?;
    let target = record
        .get(1)
        .ok_or_else(|| format!("CSV data row {row}: missing {}", baseline::TARGET_HEADER))?;
    let feature64 = baseline::parse_f64(feature, row, baseline::FEATURE_HEADER)?;
    let target64 = baseline::parse_f64(target, row, baseline::TARGET_HEADER)?;
    let feature32 = baseline::parse_f32(feature, row, baseline::FEATURE_HEADER)?;
    let target32 = baseline::parse_f32(target, row, baseline::TARGET_HEADER)?;
    Ok(TypedRow {
        feature64,
        target64,
        feature32,
        target32,
    })
}

fn exact_csv_error(bytes: &[u8], row: Option<usize>) -> String {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes);
    let mut record = StringRecord::new();
    let header = reader
        .read_record(&mut record)
        .map_err(|error| format!("CSV header: {error}"));
    if let Err(error) = header {
        return error;
    }
    let Some(row) = row else {
        return "CSV header: invalid UTF-8".to_owned();
    };
    for index in 0..=row {
        match reader.read_record(&mut record) {
            Ok(true) => {}
            Ok(false) => return format!("CSV data row {row}: unexpected end of input"),
            Err(error) => return format!("CSV data row {index}: {error}"),
        }
    }
    format!("CSV data row {row}: invalid UTF-8")
}

fn parse_rows(candidate: Candidate, bytes: &[u8]) -> Result<Vec<TypedRow>, String> {
    if bytes.len() > baseline::MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {}",
            bytes.len(),
            baseline::MAX_INPUT_BYTES
        ));
    }
    if candidate == Candidate::Baseline {
        return parse_rows_baseline(bytes);
    }
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes);
    let mut string_record = StringRecord::new();
    let mut byte_record = ByteRecord::new();
    let mut rows = Vec::new();

    let have_header = match candidate {
        Candidate::Bytes => reader
            .read_byte_record(&mut byte_record)
            .map_err(|error| format!("CSV header: {error}"))?,
        _ => reader
            .read_record(&mut string_record)
            .map_err(|error| format!("CSV header: {error}"))?,
    };
    if !have_header {
        return Err("CSV is empty; expected a header row".to_owned());
    }
    if candidate == Candidate::Bytes {
        string_record = StringRecord::from_byte_record(byte_record.clone())
            .map_err(|_| exact_csv_error(bytes, None))?;
    }
    if !header_ok(&string_record) {
        return Err(header_error());
    }

    loop {
        let row = rows.len();
        let read = match candidate {
            Candidate::Bytes => reader
                .read_byte_record(&mut byte_record)
                .map_err(|error| format!("CSV data row {row}: {error}")),
            _ => reader
                .read_record(&mut string_record)
                .map_err(|error| format!("CSV data row {row}: {error}")),
        };
        if rows.len() == baseline::MAX_DATA_ROWS {
            return match read {
                Ok(false) => Ok(rows),
                Ok(true) | Err(_) => Err(format!(
                    "CSV exceeds the {}-record example limit",
                    baseline::MAX_DATA_ROWS
                )),
            };
        }
        let present = read?;
        if !present {
            break;
        }
        if candidate == Candidate::Bytes {
            string_record = StringRecord::from_byte_record(byte_record.clone())
                .map_err(|_| exact_csv_error(bytes, Some(row)))?;
        }
        let typed = if candidate == Candidate::Serde {
            let (feature, target): (&str, &str) = string_record
                .deserialize(None)
                .map_err(|error| format!("CSV data row {row}: {error}"))?;
            let feature64 = baseline::parse_f64(feature, row, baseline::FEATURE_HEADER)?;
            let target64 = baseline::parse_f64(target, row, baseline::TARGET_HEADER)?;
            let feature32 = baseline::parse_f32(feature, row, baseline::FEATURE_HEADER)?;
            let target32 = baseline::parse_f32(target, row, baseline::TARGET_HEADER)?;
            TypedRow {
                feature64,
                target64,
                feature32,
                target32,
            }
        } else {
            parse_record(&string_record, row)?
        };
        rows.push(typed);
    }
    if rows.len() < baseline::MIN_DATA_ROWS {
        return Err(format!(
            "CSV has {} data records; at least {} are required",
            rows.len(),
            baseline::MIN_DATA_ROWS
        ));
    }
    Ok(rows)
}

fn materialize(rows: &[TypedRow], bytes: Vec<u8>, source: String) -> baseline::Dataset {
    let canonical = bytes.as_slice() == baseline::CANONICAL_CSV;
    baseline::Dataset {
        bytes,
        source,
        canonical,
        features64: rows
            .iter()
            .map(|row| seqvex::foundation::numerical::Vector64::from_slice(&[row.feature64]))
            .collect(),
        targets64: rows.iter().map(|row| row.target64).collect(),
        features32: rows
            .iter()
            .map(|row| seqvex::foundation::numerical::Vector::from_slice(&[row.feature32]))
            .collect(),
        targets32: rows.iter().map(|row| row.target32).collect(),
    }
}

fn parse_candidate(candidate: Candidate, bytes: &[u8]) -> Result<baseline::Dataset, String> {
    if bytes.len() > baseline::MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {}",
            bytes.len(),
            baseline::MAX_INPUT_BYTES
        ));
    }
    if candidate == Candidate::Baseline {
        return baseline::parse_dataset(bytes.to_vec(), source_for(bytes));
    }
    let rows = parse_rows(candidate, bytes)?;
    Ok(materialize(&rows, bytes.to_vec(), source_for(bytes)))
}

fn source_for(bytes: &[u8]) -> String {
    if bytes == baseline::CANONICAL_CSV {
        "embedded canonical fixture".to_owned()
    } else {
        "generated repeat fixture".to_owned()
    }
}

fn datasets_equal(left: &baseline::Dataset, right: &baseline::Dataset) -> bool {
    left.bytes == right.bytes
        && left.source == right.source
        && left.canonical == right.canonical
        && left.targets64.len() == right.targets64.len()
        && left
            .targets64
            .iter()
            .zip(&right.targets64)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left
            .targets32
            .iter()
            .zip(&right.targets32)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left.features64.len() == right.features64.len()
        && left
            .features64
            .iter()
            .zip(&right.features64)
            .all(|(a, b)| a.as_slice()[0].to_bits() == b.as_slice()[0].to_bits())
        && left
            .features32
            .iter()
            .zip(&right.features32)
            .all(|(a, b)| a.as_slice()[0].to_bits() == b.as_slice()[0].to_bits())
}

fn typed_rows_from_dataset(data: &baseline::Dataset) -> Vec<TypedRow> {
    data.features64
        .iter()
        .zip(&data.targets64)
        .zip(data.features32.iter().zip(&data.targets32))
        .map(|((feature64, target64), (feature32, target32))| TypedRow {
            feature64: feature64.as_slice()[0],
            target64: *target64,
            feature32: feature32.as_slice()[0],
            target32: *target32,
        })
        .collect()
}

fn typed_rows_equal(left: &[TypedRow], right: &[TypedRow]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.feature64.to_bits() == b.feature64.to_bits()
                && a.target64.to_bits() == b.target64.to_bits()
                && a.feature32.to_bits() == b.feature32.to_bits()
                && a.target32.to_bits() == b.target32.to_bits()
        })
}

fn error_of(result: Result<baseline::Dataset, String>) -> String {
    match result {
        Ok(_) => panic!("expected an ingestion error"),
        Err(error) => error,
    }
}

fn guard_fixtures() {
    for fixture in fixtures() {
        let expected = baseline::parse_dataset(fixture.bytes.clone(), source_for(&fixture.bytes))
            .unwrap_or_else(|error| panic!("{} baseline: {error}", fixture.name));
        assert_eq!(fixture.bytes.len(), expected.bytes.len());
        let text = std::str::from_utf8(&fixture.bytes).expect("UTF-8 fixture");
        assert_eq!(text.lines().count(), fixture.rows + 1);
        assert_eq!(sha256_hex(&fixture.bytes), fixture.frozen_sha256);
        let authoritative_typed = typed_rows_from_dataset(&expected);
        let baseline_typed = parse_rows_baseline(&fixture.bytes).unwrap();
        assert!(
            typed_rows_equal(&authoritative_typed, &baseline_typed),
            "{} baseline typed-row mismatch",
            fixture.name
        );
        for candidate in Candidate::ALL {
            let typed = parse_rows(candidate, &fixture.bytes).unwrap();
            assert!(
                typed_rows_equal(&authoritative_typed, &typed),
                "{} {} typed-row mismatch",
                fixture.name,
                candidate.name()
            );
            let actual = parse_candidate(candidate, &fixture.bytes)
                .unwrap_or_else(|error| panic!("{} {}: {error}", fixture.name, candidate.name()));
            assert!(
                datasets_equal(&expected, &actual),
                "{} {} output mismatch",
                fixture.name,
                candidate.name()
            );
        }
    }
    let invalid = b"DisplacementMeters,TimeOffsetSeconds\n\xff,1\n";
    let expected = error_of(baseline::parse_dataset(
        invalid.to_vec(),
        source_for(invalid),
    ));
    for candidate in Candidate::ALL {
        assert_eq!(
            error_of(parse_candidate(candidate, invalid)),
            expected,
            "{} invalid-UTF-8 error mismatch",
            candidate.name()
        );
    }
    let underflow = b"DisplacementMeters,TimeOffsetSeconds\n1e-310,1\n1,2\n2,3\n3,4\n4,5\n5,6\n";
    let expected = error_of(baseline::parse_dataset(
        underflow.to_vec(),
        source_for(underflow),
    ));
    for candidate in Candidate::ALL {
        assert_eq!(
            error_of(parse_candidate(candidate, underflow)),
            expected,
            "{} underflow error mismatch",
            candidate.name()
        );
    }

    let full = repeat_fixture(256);
    for candidate in Candidate::ALL {
        parse_candidate(candidate, &full).unwrap_or_else(|error| {
            panic!(
                "{} must accept exactly 256 records: {error}",
                candidate.name()
            )
        });
    }
    let too_many = repeat_fixture(257);
    let cap_error = error_of(baseline::parse_dataset(
        too_many.clone(),
        source_for(&too_many),
    ));
    let mut malformed = repeat_fixture(256);
    malformed.extend_from_slice(b"not-a-number,1\n");
    let mut invalid_utf8 = repeat_fixture(256);
    invalid_utf8.extend_from_slice(b"\xff,1\n");
    let mut wrong_fields = repeat_fixture(256);
    wrong_fields.extend_from_slice(b"1,2,3\n");
    for candidate in Candidate::ALL {
        assert_eq!(
            error_of(parse_candidate(candidate, &too_many)),
            cap_error,
            "{} 257th valid record cap mismatch",
            candidate.name()
        );
        assert_eq!(
            error_of(parse_candidate(candidate, &malformed)),
            cap_error,
            "{} 257th malformed record cap mismatch",
            candidate.name()
        );
        assert_eq!(
            error_of(parse_candidate(candidate, &invalid_utf8)),
            cap_error,
            "{} 257th invalid-UTF-8 cap mismatch",
            candidate.name()
        );
        assert_eq!(
            error_of(parse_candidate(candidate, &wrong_fields)),
            cap_error,
            "{} 257th wrong-field cap mismatch",
            candidate.name()
        );
    }
}

fn black_box_dataset(data: &baseline::Dataset) {
    black_box(&data.bytes);
    black_box(&data.features64);
    black_box(&data.targets64);
    black_box(&data.features32);
    black_box(&data.targets32);
}

fn run_row(timing: &BTreeMap<String, TimingMetadata>, id: &str, body: impl FnMut()) -> RowEvidence {
    let registration = timing
        .get(id)
        .unwrap_or_else(|| panic!("unregistered OLS timing row {id}"));
    run_raw_row(id, registration.calls, registration.units, "unit", body)
}

fn memory_probe<T>(id: &str, body: impl FnOnce() -> T) {
    let evidence = shared_memory_probe(body);
    println!(
        "{id}/mem before={} live={} peak={} after={}",
        evidence.before, evidence.live, evidence.peak, evidence.after
    );
}

fn run_ingestion_rows(
    timing: &BTreeMap<String, TimingMetadata>,
    probes: &BTreeSet<String>,
    evidence: &mut Vec<RowEvidence>,
    probe_ids: &mut Vec<String>,
) {
    let fixtures = fixtures();
    for candidate in ordered_candidates() {
        for fixture in &fixtures {
            let id = parse_id(candidate, fixture.name);
            evidence.push(run_row(timing, &id, || {
                drop(black_box(timed_inputs(
                    (candidate, fixture.bytes.as_slice()),
                    |(candidate, input)| parse_rows(candidate, input).unwrap(),
                )));
            }));
            let id = complete_id(candidate, fixture.name);
            evidence.push(run_row(timing, &id, || {
                timed_inputs(
                    (candidate, fixture.bytes.as_slice()),
                    |(candidate, input)| {
                        let data = parse_candidate(candidate, input).unwrap();
                        black_box_dataset(&data);
                    },
                );
            }));
        }
    }
    for fixture in &fixtures {
        let rows = parse_rows_baseline(&fixture.bytes).unwrap();
        let id = materialize_id(fixture.name);
        evidence.push(run_row(timing, &id, || {
            timed_inputs(
                (rows.as_slice(), fixture.bytes.as_slice()),
                |(rows, bytes)| {
                    let data = materialize(rows, bytes.to_vec(), source_for(bytes));
                    black_box_dataset(&data);
                },
            );
        }));
    }
    for candidate in ordered_candidates() {
        for fixture in &fixtures {
            let id = complete_id(candidate, fixture.name);
            assert!(probes.contains(&id), "unregistered OLS probe {id}");
            memory_probe(&id, || parse_candidate(candidate, &fixture.bytes).unwrap());
            probe_ids.push(id);
        }
    }
    for fixture in &fixtures {
        let rows = parse_rows_baseline(&fixture.bytes).unwrap();
        let id = materialize_id(fixture.name);
        assert!(probes.contains(&id), "unregistered OLS probe {id}");
        memory_probe(&id, || {
            materialize(&rows, fixture.bytes.clone(), source_for(&fixture.bytes))
        });
        probe_ids.push(id);
    }
    for candidate in ordered_candidates() {
        for workflow in WORKFLOW_NAMES {
            let id = workflow_id(candidate, workflow);
            assert!(probes.contains(&id), "unregistered OLS probe {id}");
            workflow_memory_probe(candidate, workflow);
            probe_ids.push(id);
        }
    }
}

fn verify_actual_evidence(evidence: &[RowEvidence], probe_ids: &[String]) {
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

fn narrow_rows(
    features: &[seqvex::foundation::numerical::Vector64],
) -> Vec<seqvex::foundation::numerical::Vector> {
    features
        .iter()
        .enumerate()
        .map(|(row, feature)| {
            let value = feature.as_slice()[0];
            assert!(value.is_finite(), "B serving input row {row}: non-finite");
            let rounded = value as f32;
            assert!(rounded.is_finite(), "B serving input row {row}: overflow");
            assert!(
                !(value != 0.0 && rounded == 0.0),
                "B serving input row {row}: underflow"
            );
            seqvex::foundation::numerical::Vector::from_slice(&[rounded])
        })
        .collect()
}

fn run_model_rows(timing: &BTreeMap<String, TimingMetadata>, evidence: &mut Vec<RowEvidence>) {
    let data = parse_candidate(Candidate::Baseline, baseline::CANONICAL_CSV).unwrap();
    let train: Vec<usize> = (0..baseline::TRAIN_ROWS).collect();
    let held: Vec<usize> = (baseline::TRAIN_ROWS..data.features64.len()).collect();
    let fit_config64 = FitConfig64::default();
    let fit_config32 = FitConfig32::default();
    let a_fit = fit_f64(&data.features64, &data.targets64, &train, &fit_config64).unwrap();
    let b_model = a_fit.to_serving_f32(UnderflowPolicy::Reject).unwrap();
    let a_model = fit_f64(&data.features64, &data.targets64, &train, &fit_config64)
        .unwrap()
        .into_predictor();
    let c_model = fit_f32(&data.features32, &data.targets32, &train, &fit_config32)
        .unwrap()
        .into_predictor();
    let b_inputs = narrow_rows(&data.features64);

    let id = "model/afit/csv-can-8";
    evidence.push(run_row(timing, id, || {
        drop(black_box(timed_inputs(
            (&data, &train, &fit_config64),
            |(data, train, config)| {
                fit_f64(&data.features64, &data.targets64, train, config).unwrap()
            },
        )));
    }));
    let id = "model/cfit/csv-can-8";
    evidence.push(run_row(timing, id, || {
        drop(black_box(timed_inputs(
            (&data, &train, &fit_config32),
            |(data, train, config)| {
                fit_f32(&data.features32, &data.targets32, train, config).unwrap()
            },
        )));
    }));
    let id = "model/bconvert/csv-can-8";
    evidence.push(run_row(timing, id, || {
        drop(black_box(timed_inputs(
            (&a_fit, UnderflowPolicy::Reject),
            |(a_fit, policy)| a_fit.to_serving_f32(policy).unwrap(),
        )));
    }));
    let id = "model/aserve/csv-can-8";
    evidence.push(run_row(timing, id, || {
        timed_inputs((&a_model, &data, &held), |(model, data, held)| {
            for &index in held {
                black_box(model.predict_checked(&data.features64[index]).unwrap());
            }
        });
    }));
    let id = "model/bserve/csv-can-8";
    evidence.push(run_row(timing, id, || {
        timed_inputs((&b_model, &b_inputs, &held), |(model, inputs, held)| {
            for &index in held {
                black_box(model.predict_checked(&inputs[index]).unwrap());
            }
        });
    }));
    let id = "model/cserve/csv-can-8";
    evidence.push(run_row(timing, id, || {
        timed_inputs((&c_model, &data, &held), |(model, data, held)| {
            for &index in held {
                black_box(model.predict_checked(&data.features32[index]).unwrap());
            }
        });
    }));
}

struct WorkflowRetained {
    data: baseline::Dataset,
    model: Box<dyn std::any::Any>,
    inputs: Option<Vec<seqvex::foundation::numerical::Vector>>,
}

fn workflow_once(candidate: Candidate, workflow: &str, bytes: &[u8]) -> WorkflowRetained {
    let data = parse_candidate(candidate, bytes).unwrap();
    let train: Vec<usize> = (0..baseline::TRAIN_ROWS).collect();
    match workflow {
        "a" => {
            let held: Vec<usize> = (baseline::TRAIN_ROWS..data.features64.len()).collect();
            let model = fit_f64(
                &data.features64,
                &data.targets64,
                &train,
                &FitConfig64::default(),
            )
            .unwrap()
            .into_predictor();
            for index in &held {
                black_box(model.predict_checked(&data.features64[*index]).unwrap());
            }
            WorkflowRetained {
                data,
                model: Box::new(model),
                inputs: None,
            }
        }
        "c" => {
            let held: Vec<usize> = (baseline::TRAIN_ROWS..data.features32.len()).collect();
            let model = fit_f32(
                &data.features32,
                &data.targets32,
                &train,
                &FitConfig32::default(),
            )
            .unwrap()
            .into_predictor();
            for index in &held {
                black_box(model.predict_checked(&data.features32[*index]).unwrap());
            }
            WorkflowRetained {
                data,
                model: Box::new(model),
                inputs: None,
            }
        }
        "ab" => {
            let held: Vec<usize> = (baseline::TRAIN_ROWS..data.features64.len()).collect();
            let a = fit_f64(
                &data.features64,
                &data.targets64,
                &train,
                &FitConfig64::default(),
            )
            .unwrap();
            let b = a.to_serving_f32(UnderflowPolicy::Reject).unwrap();
            let inputs = narrow_rows(&data.features64);
            for index in &held {
                black_box(b.predict_checked(&inputs[*index]).unwrap());
            }
            WorkflowRetained {
                data,
                model: Box::new(b),
                inputs: Some(inputs),
            }
        }
        _ => panic!("unknown workflow"),
    }
}

fn above_baseline(value: u64, baseline: u64, label: &str) -> u64 {
    value
        .checked_sub(baseline)
        .unwrap_or_else(|| panic!("{label} below baseline: value={value}, baseline={baseline}"))
}

fn workflow_memory_probe(candidate: Candidate, workflow: &str) {
    let before = common::live_bytes();
    common::reset_peak();
    let retained = workflow_once(candidate, workflow, baseline::CANONICAL_CSV);
    let aggregate_live = common::live_bytes();
    let aggregate_peak = common::peak_bytes();
    black_box(&retained);
    let aggregate_retained = above_baseline(aggregate_live, before, "aggregate retained");
    let aggregate_peak_above_baseline = above_baseline(aggregate_peak, before, "aggregate peak");

    let WorkflowRetained {
        data,
        model,
        inputs,
    } = retained;
    drop(data);
    drop(inputs);
    let predictor_live = common::live_bytes();
    let predictor_retained = above_baseline(predictor_live, before, "predictor retained");

    common::reset_peak();
    let predictor_retained_peak = common::peak_bytes();
    let predictor_retained_peak_above_baseline =
        above_baseline(predictor_retained_peak, before, "predictor retained peak");

    drop(model);
    let after = common::live_bytes();
    assert_eq!(
        after,
        before,
        "workflow/{workflow}/{} after-drop baseline mismatch",
        candidate.name()
    );

    println!(
        "workflow/{workflow}/{}/aggregate/mem before={before} aggregate_live={aggregate_live} aggregate_peak={aggregate_peak} aggregate_retained={aggregate_retained} aggregate_peak_above_baseline={aggregate_peak_above_baseline}",
        candidate.name()
    );
    println!(
        "workflow/{workflow}/{}/predictor-retained/mem before={before} predictor_live={predictor_live} predictor_retained={predictor_retained} predictor_retained_peak={predictor_retained_peak} predictor_retained_peak_above_baseline={predictor_retained_peak_above_baseline}",
        candidate.name()
    );
    println!(
        "workflow/{workflow}/{}/drop/mem before={before} after={after}",
        candidate.name()
    );
}

fn run_workflow_rows(timing: &BTreeMap<String, TimingMetadata>, evidence: &mut Vec<RowEvidence>) {
    for candidate in ordered_candidates() {
        for workflow in WORKFLOW_NAMES {
            let id = workflow_id(candidate, workflow);
            evidence.push(run_row(timing, &id, || {
                drop(black_box(timed_inputs(
                    (candidate, workflow, baseline::CANONICAL_CSV),
                    |(candidate, workflow, bytes)| workflow_once(candidate, workflow, bytes),
                )));
            }));
        }
    }
}

fn main() {
    println!(
        "CSV ingestion comparison (benchmark-only); capture={}",
        capture_from_env().number()
    );
    println!("registered: 45 timing rows, 27 memory probes");
    println!("{}", common::env_summary());
    if cfg!(debug_assertions) {
        if common::comparison::run_failure_case_if_requested() {
            return;
        }
        verify_registrations();
        common::comparison::comparison_self_check();
        guard_fixtures();
        println!("debug: ingestion guards complete; release rows skipped");
        return;
    }
    verify_registrations();
    guard_fixtures();
    let timing = timing_catalog();
    let probes = probe_catalog();
    let mut evidence = Vec::with_capacity(45);
    let mut probe_ids = Vec::with_capacity(27);
    run_ingestion_rows(&timing, &probes, &mut evidence, &mut probe_ids);
    run_model_rows(&timing, &mut evidence);
    run_workflow_rows(&timing, &mut evidence);
    verify_actual_evidence(&evidence, &probe_ids);
}
