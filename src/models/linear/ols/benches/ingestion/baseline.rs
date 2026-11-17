use std::hint::black_box;

use super::candidates::{Candidate, TypedRow, parse_rows};
use super::fit_profiles_support;

pub(super) fn materialize(
    rows: &[TypedRow],
    bytes: Vec<u8>,
    source: String,
) -> fit_profiles_support::Dataset {
    let canonical = bytes.as_slice() == fit_profiles_support::CANONICAL_CSV;
    fit_profiles_support::Dataset {
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

pub(super) fn parse_candidate(
    candidate: Candidate,
    bytes: &[u8],
) -> Result<fit_profiles_support::Dataset, String> {
    if bytes.len() > fit_profiles_support::MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {}",
            bytes.len(),
            fit_profiles_support::MAX_INPUT_BYTES
        ));
    }
    if candidate == Candidate::Baseline {
        return fit_profiles_support::parse_dataset(bytes.to_vec(), source_for(bytes));
    }
    let rows = parse_rows(candidate, bytes)?;
    Ok(materialize(&rows, bytes.to_vec(), source_for(bytes)))
}

pub(super) fn source_for(bytes: &[u8]) -> String {
    if bytes == fit_profiles_support::CANONICAL_CSV {
        "embedded canonical fixture".to_owned()
    } else {
        "generated repeat fixture".to_owned()
    }
}

pub(super) fn datasets_equal(
    left: &fit_profiles_support::Dataset,
    right: &fit_profiles_support::Dataset,
) -> bool {
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

pub(super) fn typed_rows_from_dataset(data: &fit_profiles_support::Dataset) -> Vec<TypedRow> {
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

pub(super) fn typed_rows_equal(left: &[TypedRow], right: &[TypedRow]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.feature64.to_bits() == b.feature64.to_bits()
                && a.target64.to_bits() == b.target64.to_bits()
                && a.feature32.to_bits() == b.feature32.to_bits()
                && a.target32.to_bits() == b.target32.to_bits()
        })
}

pub(super) fn error_of(result: Result<fit_profiles_support::Dataset, String>) -> String {
    match result {
        Ok(_) => panic!("expected an ingestion error"),
        Err(error) => error,
    }
}

pub(super) fn black_box_dataset(data: &fit_profiles_support::Dataset) {
    black_box(&data.bytes);
    black_box(&data.features64);
    black_box(&data.targets64);
    black_box(&data.features32);
    black_box(&data.targets32);
}
