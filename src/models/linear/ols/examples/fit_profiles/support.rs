use std::fs::File;
use std::io::{Read, Take};
use std::path::PathBuf;

use seqvex::foundation::numerical::{Vector, Vector64};

pub(crate) const CANONICAL_CSV: &[u8] =
    include_bytes!("../../../../../../examples/data/displacement_time_offset.csv");
pub(crate) const MAX_INPUT_BYTES: usize = 64 * 1024;
pub(crate) const MAX_DATA_ROWS: usize = 256;
pub(crate) const MIN_DATA_ROWS: usize = 6;
pub(crate) const TRAIN_ROWS: usize = 5;
pub(super) const MAX_GROUP_SIZE: usize = 32;
pub(crate) const FEATURE_HEADER: &str = "DisplacementMeters";
pub(crate) const TARGET_HEADER: &str = "TimeOffsetSeconds";

pub(crate) struct Dataset {
    pub(crate) bytes: Vec<u8>,
    pub(crate) source: String,
    pub(crate) canonical: bool,
    pub(crate) features64: Vec<Vector64>,
    pub(crate) targets64: Vec<f64>,
    pub(crate) features32: Vec<Vector>,
    pub(crate) targets32: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(super) struct Metrics {
    pub(super) rss: f64,
    pub(super) mae: f64,
    pub(super) rmse: f64,
}

pub(super) fn read_input(args: &[std::ffi::OsString]) -> Result<(Vec<u8>, String), String> {
    match args {
        [] => Ok((
            CANONICAL_CSV.to_vec(),
            "embedded canonical fixture".to_owned(),
        )),
        [path] => {
            let path = PathBuf::from(path);
            let file = File::open(&path)
                .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
            let mut limited: Take<File> = file.take((MAX_INPUT_BYTES + 1) as u64);
            let mut bytes = Vec::new();
            limited
                .read_to_end(&mut bytes)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if bytes.len() > MAX_INPUT_BYTES {
                return Err(format!(
                    "{} exceeds the {MAX_INPUT_BYTES}-byte example input limit",
                    path.display()
                ));
            }
            Ok((bytes, path.to_string_lossy().into_owned()))
        }
        _ => Err("usage: fit_profiles [<two-column-csv-path>]".to_owned()),
    }
}

pub(crate) fn parse_dataset(bytes: Vec<u8>, source: String) -> Result<Dataset, String> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {MAX_INPUT_BYTES}",
            bytes.len()
        ));
    }
    let canonical = bytes.as_slice() == CANONICAL_CSV;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes.as_slice());
    let mut records = reader.records();
    let headers = records
        .next()
        .ok_or_else(|| "CSV is empty; expected a header row".to_owned())?
        .map_err(|error| format!("CSV header: {error}"))?;
    if headers.len() != 2
        || headers.get(0) != Some(FEATURE_HEADER)
        || headers.get(1) != Some(TARGET_HEADER)
    {
        return Err(format!(
            "CSV headers must be exactly [{FEATURE_HEADER}, {TARGET_HEADER}] in that order"
        ));
    }

    let mut features64 = Vec::new();
    let mut targets64 = Vec::new();
    let mut features32 = Vec::new();
    let mut targets32 = Vec::new();
    for (row, record) in records.enumerate() {
        if features64.len() == MAX_DATA_ROWS {
            return Err(format!(
                "CSV exceeds the {MAX_DATA_ROWS}-record example limit"
            ));
        }
        let record = record.map_err(|error| format!("CSV data row {row}: {error}"))?;
        let feature = record
            .get(0)
            .ok_or_else(|| format!("CSV data row {row}: missing {FEATURE_HEADER}"))?;
        let target = record
            .get(1)
            .ok_or_else(|| format!("CSV data row {row}: missing {TARGET_HEADER}"))?;
        let feature64 = parse_f64(feature, row, FEATURE_HEADER)?;
        let target64 = parse_f64(target, row, TARGET_HEADER)?;
        let feature32 = parse_f32(feature, row, FEATURE_HEADER)?;
        let target32 = parse_f32(target, row, TARGET_HEADER)?;
        features64.push(Vector64::from_slice(&[feature64]));
        targets64.push(target64);
        features32.push(Vector::from_slice(&[feature32]));
        targets32.push(target32);
    }
    if features64.len() < MIN_DATA_ROWS {
        return Err(format!(
            "CSV has {} data records; at least {MIN_DATA_ROWS} are required",
            features64.len()
        ));
    }

    Ok(Dataset {
        bytes,
        source,
        canonical,
        features64,
        targets64,
        features32,
        targets32,
    })
}

pub(crate) fn mantissa_has_nonzero_digit(token: &str) -> bool {
    token
        .split(['e', 'E'])
        .next()
        .unwrap_or(token)
        .bytes()
        .any(|byte| (b'1'..=b'9').contains(&byte))
}

pub(crate) fn parse_f64(token: &str, row: usize, column: &str) -> Result<f64, String> {
    let token = token.trim();
    let value = token.parse::<f64>().map_err(|error| {
        format!("data row {row}, {column}, f64: invalid number `{token}`: {error}")
    })?;
    if !value.is_finite() {
        return Err(format!(
            "data row {row}, {column}, f64: value must be finite"
        ));
    }
    if value == 0.0 && mantissa_has_nonzero_digit(token) {
        return Err(format!(
            "data row {row}, {column}, f64: nonzero value underflowed to zero"
        ));
    }
    Ok(value)
}

pub(crate) fn parse_f32(token: &str, row: usize, column: &str) -> Result<f32, String> {
    let token = token.trim();
    let value = token.parse::<f32>().map_err(|error| {
        format!("data row {row}, {column}, f32: invalid number `{token}`: {error}")
    })?;
    if !value.is_finite() {
        return Err(format!(
            "data row {row}, {column}, f32: value must be finite"
        ));
    }
    if value == 0.0 && mantissa_has_nonzero_digit(token) {
        return Err(format!(
            "data row {row}, {column}, f32: nonzero value underflowed to zero"
        ));
    }
    Ok(value)
}

pub(super) fn narrow_b_input(value: f64, row: usize) -> Result<f32, String> {
    if !value.is_finite() {
        return Err(format!("B serving input row {row}: source is non-finite"));
    }
    let rounded = value as f32;
    if !rounded.is_finite() {
        return Err(format!("B serving input row {row}: f64-to-f32 overflow"));
    }
    if value != 0.0 && rounded == 0.0 {
        return Err(format!(
            "B serving input row {row}: nonzero value underflowed to zero"
        ));
    }
    Ok(rounded)
}

pub(super) fn narrow_b_inputs(features: &[Vector64]) -> Result<Vec<Vector>, String> {
    features
        .iter()
        .enumerate()
        .map(|(row, feature)| {
            Ok(Vector::from_slice(&[narrow_b_input(
                feature.as_slice()[0],
                row,
            )?]))
        })
        .collect()
}

pub(super) fn metrics(predictions: &[f64], targets: &[f64]) -> Metrics {
    assert_eq!(predictions.len(), targets.len());
    assert!(!targets.is_empty());
    let mut rss = 0.0;
    let mut absolute = 0.0;
    for (&prediction, &target) in predictions.iter().zip(targets) {
        assert!(prediction.is_finite() && target.is_finite());
        let residual = prediction - target;
        rss += residual * residual;
        absolute += residual.abs();
    }
    let mae = absolute / targets.len() as f64;
    Metrics {
        rss,
        mae,
        rmse: (rss / targets.len() as f64).sqrt(),
    }
}

pub(super) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}
