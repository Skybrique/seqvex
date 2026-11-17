use sha2::{Digest, Sha256};

use super::baseline::{
    datasets_equal, error_of, parse_candidate, source_for, typed_rows_equal,
    typed_rows_from_dataset,
};
use super::candidates::{Candidate, parse_rows, parse_rows_baseline};
use super::fit_profiles_support;

pub(super) struct Fixture {
    pub(super) name: &'static str,
    pub(super) bytes: Vec<u8>,
    pub(super) rows: usize,
    pub(super) frozen_sha256: &'static str,
}

pub(super) const FIXTURE_NAMES: [&str; 3] = ["csv-can-8", "csv-rep-32", "csv-rep-256"];

pub(super) fn fixture_rows(fixture: &str) -> usize {
    match fixture {
        "csv-can-8" => 8,
        "csv-rep-32" => 32,
        "csv-rep-256" => 256,
        _ => panic!("unknown fixture {fixture}"),
    }
}

pub(super) fn fixture_calls(rows: usize) -> u32 {
    match rows {
        8 => 64,
        32 => 16,
        256 => 4,
        _ => panic!("unknown fixture row count {rows}"),
    }
}

fn repeat_fixture(rows: usize) -> Vec<u8> {
    let text =
        std::str::from_utf8(fit_profiles_support::CANONICAL_CSV).expect("canonical CSV is UTF-8");
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

pub(super) fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "csv-can-8",
            bytes: fit_profiles_support::CANONICAL_CSV.to_vec(),
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

pub(super) fn guard_fixtures() {
    for fixture in fixtures() {
        let expected =
            fit_profiles_support::parse_dataset(fixture.bytes.clone(), source_for(&fixture.bytes))
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
    let expected = error_of(fit_profiles_support::parse_dataset(
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
    let expected = error_of(fit_profiles_support::parse_dataset(
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
    let cap_error = error_of(fit_profiles_support::parse_dataset(
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
