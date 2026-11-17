use csv::{ByteRecord, ReaderBuilder, StringRecord};

use super::fit_profiles_support;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Candidate {
    Baseline,
    String,
    Bytes,
    Serde,
}

impl Candidate {
    pub(super) const ALL: [Self; 4] = [Self::Baseline, Self::String, Self::Bytes, Self::Serde];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::String => "string",
            Self::Bytes => "bytes",
            Self::Serde => "serde",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TypedRow {
    pub(super) feature64: f64,
    pub(super) target64: f64,
    pub(super) feature32: f32,
    pub(super) target32: f32,
}

fn header_ok(record: &StringRecord) -> bool {
    record.len() == 2
        && record.get(0) == Some(fit_profiles_support::FEATURE_HEADER)
        && record.get(1) == Some(fit_profiles_support::TARGET_HEADER)
}

fn header_error() -> String {
    format!(
        "CSV headers must be exactly [{}, {}] in that order",
        fit_profiles_support::FEATURE_HEADER,
        fit_profiles_support::TARGET_HEADER
    )
}

pub(super) fn parse_rows_baseline(bytes: &[u8]) -> Result<Vec<TypedRow>, String> {
    if bytes.len() > fit_profiles_support::MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {}",
            bytes.len(),
            fit_profiles_support::MAX_INPUT_BYTES
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
        if rows.len() == fit_profiles_support::MAX_DATA_ROWS {
            return Err(format!(
                "CSV exceeds the {}-record example limit",
                fit_profiles_support::MAX_DATA_ROWS
            ));
        }
        let record = record.map_err(|error| format!("CSV data row {row}: {error}"))?;
        rows.push(parse_record(&record, row)?);
    }
    if rows.len() < fit_profiles_support::MIN_DATA_ROWS {
        return Err(format!(
            "CSV has {} data records; at least {} are required",
            rows.len(),
            fit_profiles_support::MIN_DATA_ROWS
        ));
    }
    Ok(rows)
}

fn parse_record(record: &StringRecord, row: usize) -> Result<TypedRow, String> {
    let feature = record.get(0).ok_or_else(|| {
        format!(
            "CSV data row {row}: missing {}",
            fit_profiles_support::FEATURE_HEADER
        )
    })?;
    let target = record.get(1).ok_or_else(|| {
        format!(
            "CSV data row {row}: missing {}",
            fit_profiles_support::TARGET_HEADER
        )
    })?;
    let feature64 =
        fit_profiles_support::parse_f64(feature, row, fit_profiles_support::FEATURE_HEADER)?;
    let target64 =
        fit_profiles_support::parse_f64(target, row, fit_profiles_support::TARGET_HEADER)?;
    let feature32 =
        fit_profiles_support::parse_f32(feature, row, fit_profiles_support::FEATURE_HEADER)?;
    let target32 =
        fit_profiles_support::parse_f32(target, row, fit_profiles_support::TARGET_HEADER)?;
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

pub(super) fn parse_rows(candidate: Candidate, bytes: &[u8]) -> Result<Vec<TypedRow>, String> {
    if bytes.len() > fit_profiles_support::MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; example limit is {}",
            bytes.len(),
            fit_profiles_support::MAX_INPUT_BYTES
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
        if rows.len() == fit_profiles_support::MAX_DATA_ROWS {
            return match read {
                Ok(false) => Ok(rows),
                Ok(true) | Err(_) => Err(format!(
                    "CSV exceeds the {}-record example limit",
                    fit_profiles_support::MAX_DATA_ROWS
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
            let feature64 = fit_profiles_support::parse_f64(
                feature,
                row,
                fit_profiles_support::FEATURE_HEADER,
            )?;
            let target64 =
                fit_profiles_support::parse_f64(target, row, fit_profiles_support::TARGET_HEADER)?;
            let feature32 = fit_profiles_support::parse_f32(
                feature,
                row,
                fit_profiles_support::FEATURE_HEADER,
            )?;
            let target32 =
                fit_profiles_support::parse_f32(target, row, fit_profiles_support::TARGET_HEADER)?;
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
    if rows.len() < fit_profiles_support::MIN_DATA_ROWS {
        return Err(format!(
            "CSV has {} data records; at least {} are required",
            rows.len(),
            fit_profiles_support::MIN_DATA_ROWS
        ));
    }
    Ok(rows)
}
