//! Shared example-only support for the two real-data Trump CSV examples.
//!
//! This file is not a Cargo target. Each example includes it with `#[path]`.
//! It owns only preparation shared by both examples: the seven-column schema,
//! parsing, the frozen prefix standardizer, prediction/metric records, the
//! target baselines, formatting, and a same-byte hashing reader. Learner
//! protocols stay in the example-specific modules.

use std::error::Error;
use std::fs::File;
use std::io::{self, Read};

use sha2::{Digest, Sha256};

use seqvex::foundation::numerical::Vector;

/// Date column (metadata only; never a model input).
pub(crate) const DATE_COLUMN: &str = "ordinal_date";
/// Target column.
pub(crate) const TARGET_COLUMN: &str = "five_thirty_eight";
/// Poll feature columns, in fixed model order.
pub(crate) const FEATURE_COLUMNS: [&str; 5] =
    ["gallup", "ipsos", "morning_consult", "rasmussen", "you_gov"];
/// Model dimension: five poll features plus a generated intercept.
pub(crate) const DIMENSION: usize = 6;
/// Learning-prefix length in data rows.
pub(crate) const PREFIX: usize = 400;

/// One parsed, validated observation in original units.
#[derive(Debug)]
pub(crate) struct Row {
    pub(crate) ordinal_date: i64,
    pub(crate) features: [f32; 5],
    pub(crate) target: f32,
}

/// Frozen prefix-only standardization of the five poll features.
pub(crate) struct Standardizer {
    pub(crate) mean: [f64; 5],
    pub(crate) scale: [f64; 5],
}

impl Standardizer {
    /// Computes the frozen transform from the first [`PREFIX`] rows only.
    pub(crate) fn from_prefix(rows: &[Row]) -> Result<Self, Box<dyn Error>> {
        let mut mean = [0.0f64; 5];
        for row in &rows[..PREFIX] {
            for (j, m) in mean.iter_mut().enumerate() {
                *m += f64::from(row.features[j]);
            }
        }
        for m in &mut mean {
            *m /= PREFIX as f64;
        }

        let mut variance = [0.0f64; 5];
        for row in &rows[..PREFIX] {
            for (j, m) in mean.iter().enumerate() {
                let delta = f64::from(row.features[j]) - m;
                variance[j] += delta * delta;
            }
        }

        let mut scale = [0.0f64; 5];
        for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
            scale[j] = (variance[j] / PREFIX as f64).sqrt();
            if !mean[j].is_finite() {
                return Err(format!(
                    "preprocessing: column {name}: non-finite prefix mean {}",
                    mean[j]
                )
                .into());
            }
            if !scale[j].is_finite() || scale[j] == 0.0 {
                return Err(format!(
                    "preprocessing: column {name}: non-finite or zero prefix scale {} \
                     (no epsilon or 1.0 substitution)",
                    scale[j]
                )
                .into());
            }
        }
        Ok(Self { mean, scale })
    }

    /// Standardizes one row with the frozen transform, re-checking finiteness
    /// after `f64` -> `f32` conversion. The intercept stays exactly `1.0`.
    pub(crate) fn apply(&self, row: &Row, row_index: usize) -> Result<Vector, Box<dyn Error>> {
        let mut values = [0.0f32; DIMENSION];
        for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
            let standardized = (f64::from(row.features[j]) - self.mean[j]) / self.scale[j];
            if !standardized.is_finite() {
                return Err(format!(
                    "preprocessing: row {row_index} (date {}): column {name}: \
                     non-finite standardized value {standardized}",
                    row.ordinal_date
                )
                .into());
            }
            let value = standardized as f32;
            if !value.is_finite() {
                return Err(format!(
                    "preprocessing: row {row_index} (date {}): column {name}: \
                     value {standardized} is non-finite after f32 conversion",
                    row.ordinal_date
                )
                .into());
            }
            values[j] = value;
        }
        values[5] = 1.0;
        Ok(Vector::from_slice(&values))
    }
}

/// Parsed metrics over a set of scored rows.
pub(crate) struct Metrics {
    pub(crate) mae: f64,
    pub(crate) rmse: f64,
}

/// One scored prediction recorded before its target was revealed to the model.
pub(crate) struct Prediction {
    pub(crate) ordinal_date: i64,
    pub(crate) actual: f32,
    pub(crate) prediction: f32,
    pub(crate) error: f64,
}

/// A reader that hashes every byte it returns from the wrapped reader.
pub(crate) struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> HashingReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    /// Consumes the reader and returns the lowercase hexadecimal digest of every
    /// byte read through it.
    pub(crate) fn finalize_hex(self) -> String {
        hex(self.hasher.finalize().as_slice())
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        Ok(read)
    }
}

/// Lowercase hexadecimal encoding of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// Reads and validates a dataset through a hashing reader.
///
/// The digest covers every byte of the successfully parsed file, including any
/// bytes consumed only by the post-parse drain to EOF. Parse errors are returned
/// before a digest exists; no second file read is performed.
pub(crate) fn read_dataset_with_hash(path: &str) -> Result<(Vec<Row>, String), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = HashingReader::new(file);
    let rows = parse_reader(&mut reader)?;
    io::copy(&mut reader, &mut io::sink())?;
    Ok((rows, reader.finalize_hex()))
}

/// Parses and validates the seven-column schema from any reader.
///
/// Exactly the seven required headers are accepted, each once; numeric fields
/// must be finite; `ordinal_date` must be strictly increasing; at least
/// `PREFIX + 1` rows are required. Values are never reordered, dropped, imputed,
/// or clamped.
pub(crate) fn parse_reader<R: Read>(reader: R) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().from_reader(reader);
    let headers = reader.headers()?.clone();

    let mut date_idx: Option<usize> = None;
    let mut target_idx: Option<usize> = None;
    let mut feature_idx: [Option<usize>; 5] = [None; 5];
    for (index, header) in headers.iter().enumerate() {
        let name = header.trim();
        if name == DATE_COLUMN {
            if date_idx.replace(index).is_some() {
                return Err(format!("duplicate required header {DATE_COLUMN}").into());
            }
        } else if name == TARGET_COLUMN {
            if target_idx.replace(index).is_some() {
                return Err(format!("duplicate required header {TARGET_COLUMN}").into());
            }
        } else if let Some(j) = FEATURE_COLUMNS.iter().position(|column| *column == name) {
            if feature_idx[j].replace(index).is_some() {
                return Err(format!("duplicate required header {name}").into());
            }
        } else {
            return Err(format!("unexpected header {name}").into());
        }
    }

    let date_idx = date_idx.ok_or_else(|| -> Box<dyn Error> {
        format!("missing required header {DATE_COLUMN}").into()
    })?;
    let target_idx = target_idx.ok_or_else(|| -> Box<dyn Error> {
        format!("missing required header {TARGET_COLUMN}").into()
    })?;
    let mut feature_cols = [0usize; 5];
    for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
        feature_cols[j] = feature_idx[j].ok_or_else(|| -> Box<dyn Error> {
            format!("missing required header {name}").into()
        })?;
    }

    let mut rows: Vec<Row> = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        let record =
            record.map_err(|error| format!("row {row_index}: malformed CSV record: {error}"))?;

        let raw_date = record.get(date_idx).ok_or_else(|| -> Box<dyn Error> {
            format!("row {row_index}: missing field for {DATE_COLUMN}").into()
        })?;
        let ordinal_date: i64 = raw_date.trim().parse().map_err(|error| {
            format!(
                "row {row_index}: column {DATE_COLUMN}: invalid integer {}: {error}",
                raw_date.trim()
            )
        })?;

        let mut features = [0.0f32; 5];
        for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
            let raw = record
                .get(feature_cols[j])
                .ok_or_else(|| -> Box<dyn Error> {
                    format!("row {row_index}: missing field for {name}").into()
                })?;
            let value: f32 = raw.trim().parse().map_err(|error| {
                format!(
                    "row {row_index}: column {name}: invalid f32 {}: {error}",
                    raw.trim()
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "row {row_index}: column {name}: non-finite value {}",
                    raw.trim()
                )
                .into());
            }
            features[j] = value;
        }

        let raw_target = record.get(target_idx).ok_or_else(|| -> Box<dyn Error> {
            format!("row {row_index}: missing field for {TARGET_COLUMN}").into()
        })?;
        let target: f32 = raw_target.trim().parse().map_err(|error| {
            format!(
                "row {row_index}: column {TARGET_COLUMN}: invalid f32 {}: {error}",
                raw_target.trim()
            )
        })?;
        if !target.is_finite() {
            return Err(format!(
                "row {row_index}: column {TARGET_COLUMN}: non-finite value {}",
                raw_target.trim()
            )
            .into());
        }

        rows.push(Row {
            ordinal_date,
            features,
            target,
        });
    }

    if rows.len() < PREFIX + 1 {
        return Err(format!(
            "dataset has {} data rows; need at least {} for the {PREFIX}-row learning prefix",
            rows.len(),
            PREFIX + 1
        )
        .into());
    }
    for i in 1..rows.len() {
        if rows[i].ordinal_date <= rows[i - 1].ordinal_date {
            return Err(format!(
                "chronology violation at row {i}: ordinal_date {} is not strictly greater than previous {}",
                rows[i].ordinal_date,
                rows[i - 1].ordinal_date
            )
            .into());
        }
    }
    Ok(rows)
}

/// Metrics for the three model-independent baselines over the scoring rows.
/// Historical baselines use only targets strictly preceding the current row.
pub(crate) fn baseline_metrics(
    rows: &[Row],
) -> Result<[(&'static str, Metrics); 3], Box<dyn Error>> {
    let mut running_sum: f64 = rows[..PREFIX].iter().map(|row| f64::from(row.target)).sum();
    let mut running_count = PREFIX as f64;
    let mut expanding = (0.0f64, 0.0f64);
    let mut previous = (0.0f64, 0.0f64);
    let mut raw_feature_mean = (0.0f64, 0.0f64);

    let mut previous_target = f64::from(rows[PREFIX - 1].target);
    for row in rows.iter().skip(PREFIX) {
        let actual = f64::from(row.target);
        let expanding_prediction = running_sum / running_count;
        let previous_prediction = previous_target;
        let raw_prediction = row
            .features
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / 5.0;

        for (accumulator, prediction) in [
            (&mut expanding, expanding_prediction),
            (&mut previous, previous_prediction),
            (&mut raw_feature_mean, raw_prediction),
        ] {
            let error = actual - prediction;
            accumulator.0 += error.abs();
            accumulator.1 += error * error;
        }

        running_sum += actual;
        running_count += 1.0;
        previous_target = actual;
    }

    let count = (rows.len() - PREFIX) as f64;
    let finish = |(sum_abs, sum_sq): (f64, f64)| Metrics {
        mae: sum_abs / count,
        rmse: (sum_sq / count).sqrt(),
    };
    let result = [
        ("expanding target mean", finish(expanding)),
        ("previous target", finish(previous)),
        ("raw five-feature mean", finish(raw_feature_mean)),
    ];
    for (name, metrics) in &result {
        if !metrics.mae.is_finite() || !metrics.rmse.is_finite() {
            return Err(format!("baseline {name}: non-finite metrics").into());
        }
    }
    Ok(result)
}

/// Formats weights with six decimal places, preserving the established output.
pub(crate) fn format_weights(weights: &[f32]) -> String {
    weights
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use std::io::{self, Read};
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};

    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "seqvex_trump_support_{name}_{}.csv",
            std::process::id()
        ));
        path
    }

    fn write_temp(name: &str, content: &str) -> PathBuf {
        let path = temp_path(name);
        std::fs::write(&path, content).expect("write temp csv");
        path
    }

    fn csv_rows(count: usize, trailing_blank_line: bool) -> String {
        let mut content = String::from(
            "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov\n",
        );
        for i in 0..count {
            content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                736_389 + i as i64,
                1.0 + i as f64 * 0.5,
                10.0 + i as f64,
                20.0,
                30.0,
                40.0,
                50.0
            ));
        }
        if trailing_blank_line {
            content.push('\n');
        }
        content
    }

    struct Chunked {
        data: Vec<u8>,
        offset: usize,
        chunk: usize,
    }

    impl Read for Chunked {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.offset >= self.data.len() {
                return Ok(0);
            }
            let count = self
                .chunk
                .min(buffer.len())
                .min(self.data.len() - self.offset);
            buffer[..count].copy_from_slice(&self.data[self.offset..self.offset + count]);
            self.offset += count;
            Ok(count)
        }
    }

    struct BytesThenError {
        data: Vec<u8>,
        offset: usize,
        error: Option<io::Error>,
    }

    impl Read for BytesThenError {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.offset < self.data.len() {
                let count = (self.data.len() - self.offset).min(buffer.len());
                buffer[..count].copy_from_slice(&self.data[self.offset..self.offset + count]);
                self.offset += count;
                Ok(count)
            } else if let Some(error) = self.error.take() {
                Err(error)
            } else {
                Ok(0)
            }
        }
    }

    #[test]
    fn hashing_reader_covers_all_bytes_across_short_reads() {
        let data = b"short-read hashing coverage".to_vec();
        let mut reader = HashingReader::new(Chunked {
            data: data.clone(),
            offset: 0,
            chunk: 3,
        });
        let mut output = Vec::new();
        reader.read_to_end(&mut output).expect("read all bytes");
        assert_eq!(output, data);
        assert_eq!(reader.finalize_hex(), hex(&Sha256::digest(data.as_slice())));
    }

    #[test]
    fn dataset_digest_covers_the_entire_file() {
        let content = csv_rows(PREFIX + 1, true);
        let path = write_temp("digest", &content);
        let (rows, digest) = read_dataset_with_hash(path.to_str().unwrap()).expect("valid dataset");
        assert_eq!(rows.len(), PREFIX + 1);
        assert_eq!(digest, hex(&Sha256::digest(content.as_bytes())));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn malformed_record_precedes_a_later_read_error() {
        let mut content = String::from(
            "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov\n",
        );
        content.push_str("736389,1.0,10,20,30\n");
        while content.len() < 8192 {
            content.push_str("736390,1.0,10,20,30,40,50\n");
        }
        let mut reader = BytesThenError {
            data: content.into_bytes(),
            offset: 0,
            error: Some(io::Error::other("injected read failure")),
        };
        let error = parse_reader(&mut reader).expect_err("malformed record must win");
        assert!(
            error.to_string().contains("malformed CSV record"),
            "{error}"
        );
        assert!(
            !error.to_string().contains("injected read failure"),
            "{error}"
        );
    }

    #[test]
    fn read_error_after_headers_surfaces() {
        let content =
            b"ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov\n"
                .to_vec();
        let mut reader = BytesThenError {
            data: content,
            offset: 0,
            error: Some(io::Error::other("injected read failure")),
        };
        let error = parse_reader(&mut reader).expect_err("read failure must surface");
        assert!(
            error.to_string().contains("injected read failure"),
            "{error}"
        );
    }

    #[test]
    fn missing_file_reports_open_error_before_hash() {
        let path = temp_path("missing");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&path);
        let baseline = std::fs::File::open(&path).expect_err("missing-file baseline must fail");
        let error =
            read_dataset_with_hash(path.to_str().unwrap()).expect_err("missing file must fail");
        let io_error = error
            .downcast_ref::<io::Error>()
            .expect("missing file must surface an io::Error");
        assert_eq!(io_error.kind(), io::ErrorKind::NotFound);
        assert_eq!(io_error.kind(), baseline.kind());
        assert_eq!(io_error.to_string(), baseline.to_string());
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_file_reports_permission_error_before_hash() {
        use std::os::unix::fs::PermissionsExt;

        let path = write_temp("unreadable", &csv_rows(PREFIX + 1, false));
        let mut permissions = std::fs::metadata(&path)
            .expect("unreadable temp metadata")
            .permissions();
        permissions.set_mode(0o000);
        std::fs::set_permissions(&path, permissions).expect("remove read permission");
        let baseline = std::fs::File::open(&path)
            .expect_err("permission-denied control requires an unreadable real file (non-root)");
        let result = read_dataset_with_hash(path.to_str().unwrap());
        let mut permissions = std::fs::metadata(&path)
            .expect("unreadable temp metadata")
            .permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(&path, permissions).expect("restore read permission");
        let _ = std::fs::remove_file(&path);

        let error = result.expect_err("unreadable file must fail");
        let io_error = error
            .downcast_ref::<io::Error>()
            .expect("unreadable file must surface an io::Error");
        assert_eq!(io_error.kind(), baseline.kind());
        assert_eq!(io_error.to_string(), baseline.to_string());
        assert_eq!(io_error.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn truncated_record_reports_parse_error_before_hash() {
        let mut content = csv_rows(PREFIX, false);
        content.push_str(&format!(
            "{},{},{},{},{},{}\n",
            736_389 + PREFIX as i64,
            1.0,
            10.0,
            20.0,
            30.0,
            40.0
        ));
        let path = write_temp("truncated", &content);
        let error = read_dataset_with_hash(path.to_str().unwrap())
            .expect_err("a truncated record must fail");
        let _ = std::fs::remove_file(path);
        assert!(
            error.downcast_ref::<io::Error>().is_none(),
            "truncated input must be a parse error, not an OS I/O error: {error}"
        );
        assert!(
            error.to_string().contains("missing field for you_gov")
                || error.to_string().contains("malformed CSV record"),
            "{error}"
        );
    }

    #[test]
    fn trailing_data_controls_are_explicit() {
        let content = csv_rows(PREFIX + 1, false);
        let with_blank_lines = format!("{content}\n\n");
        let path = write_temp("trailing_blank_lines", &with_blank_lines);
        let (rows, digest) =
            read_dataset_with_hash(path.to_str().unwrap()).expect("trailing blank lines are valid");
        assert_eq!(rows.len(), PREFIX + 1);
        assert_eq!(digest, hex(&Sha256::digest(with_blank_lines.as_bytes())));
        let _ = std::fs::remove_file(path);

        let without_final_newline = content.trim_end_matches('\n').to_string();
        let path = write_temp("trailing_no_newline", &without_final_newline);
        let (rows, digest) = read_dataset_with_hash(path.to_str().unwrap())
            .expect("a final record without a newline is valid");
        assert_eq!(rows.len(), PREFIX + 1);
        assert_eq!(
            digest,
            hex(&Sha256::digest(without_final_newline.as_bytes()))
        );
        let _ = std::fs::remove_file(path);

        let with_trailing_garbage = format!("{content}trailing-garbage");
        let path = write_temp("trailing_garbage", &with_trailing_garbage);
        let error = read_dataset_with_hash(path.to_str().unwrap())
            .expect_err("trailing non-record bytes must fail");
        let _ = std::fs::remove_file(path);
        assert!(
            error.downcast_ref::<io::Error>().is_none(),
            "trailing non-record bytes must be a parse error, not an OS I/O error: {error}"
        );
        assert!(
            error.to_string().contains("missing field")
                || error.to_string().contains("malformed CSV record"),
            "{error}"
        );
    }

    #[test]
    fn drain_failure_component_after_successful_parse_withholds_hash() {
        struct EofThenError {
            data: Vec<u8>,
            offset: usize,
            eof_returned: bool,
            error: Option<io::Error>,
        }

        impl Read for EofThenError {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if self.offset < self.data.len() {
                    let count = (self.data.len() - self.offset).min(buffer.len());
                    buffer[..count].copy_from_slice(&self.data[self.offset..self.offset + count]);
                    self.offset += count;
                    Ok(count)
                } else if !self.eof_returned {
                    self.eof_returned = true;
                    Ok(0)
                } else if let Some(error) = self.error.take() {
                    Err(error)
                } else {
                    Ok(0)
                }
            }
        }

        let content = csv_rows(PREFIX + 1, false);
        let mut reader = HashingReader::new(EofThenError {
            data: content.into_bytes(),
            offset: 0,
            eof_returned: false,
            error: Some(io::Error::other("injected drain failure")),
        });
        let rows = parse_reader(&mut reader).expect("records must parse before the drain");
        assert_eq!(rows.len(), PREFIX + 1);
        assert_eq!(rows[0].ordinal_date, 736_389);
        assert_eq!(rows[PREFIX].ordinal_date, 736_389 + PREFIX as i64);
        let error = io::copy(&mut reader, &mut io::sink())
            .expect_err("drain failure must surface after successful parse");
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(
            error.to_string().contains("injected drain failure"),
            "{error}"
        );
    }

    #[test]
    fn standardizer_applies_frozen_affine_map_and_intercept() {
        let scaler = Standardizer {
            mean: [10.0; 5],
            scale: [2.0; 5],
        };
        let row = Row {
            ordinal_date: 1,
            features: [12.0, 8.0, 10.0, 14.0, 6.0],
            target: 0.0,
        };
        let standardized = scaler.apply(&row, 0).expect("finite standardization");
        assert_eq!(standardized.as_slice(), &[1.0, -1.0, 0.0, 2.0, -2.0, 1.0]);
    }

    #[test]
    fn standardizer_rejects_non_finite_standardized_values() {
        let scaler = Standardizer {
            mean: [0.0; 5],
            scale: [0.0; 5],
        };
        let row = Row {
            ordinal_date: 1,
            features: [12.0, 8.0, 10.0, 14.0, 6.0],
            target: 0.0,
        };
        let error = scaler.apply(&row, 7).expect_err("zero scale must fail");
        assert!(
            error.to_string().contains("non-finite standardized value"),
            "{error}"
        );
    }

    #[test]
    fn baseline_metrics_match_manual_reference() {
        let mut rows = Vec::with_capacity(PREFIX + 2);
        for i in 0..PREFIX {
            rows.push(Row {
                ordinal_date: i as i64,
                features: [10.0; 5],
                target: 2.0,
            });
        }
        rows.push(Row {
            ordinal_date: PREFIX as i64,
            features: [10.0; 5],
            target: 4.0,
        });
        rows.push(Row {
            ordinal_date: PREFIX as i64 + 1,
            features: [10.0; 5],
            target: 6.0,
        });

        let result = baseline_metrics(&rows).expect("finite baselines");
        let second_expanding = (PREFIX as f64 * 2.0 + 4.0) / (PREFIX as f64 + 1.0);
        let expanding_errors = [4.0 - 2.0, 6.0 - second_expanding];
        let expected_expanding_mae = (expanding_errors[0].abs() + expanding_errors[1].abs()) / 2.0;
        let expected_expanding_rmse =
            ((expanding_errors[0].powi(2) + expanding_errors[1].powi(2)) / 2.0).sqrt();
        assert_eq!(result[0].0, "expanding target mean");
        assert!((result[0].1.mae - expected_expanding_mae).abs() < 1e-12);
        assert!((result[0].1.rmse - expected_expanding_rmse).abs() < 1e-12);

        assert_eq!(result[1].0, "previous target");
        assert!((result[1].1.mae - 2.0).abs() < 1e-12);
        assert!((result[1].1.rmse - 2.0).abs() < 1e-12);

        assert_eq!(result[2].0, "raw five-feature mean");
        assert!((result[2].1.mae - 5.0).abs() < 1e-12);
        assert!((result[2].1.rmse - 26.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn weight_formatting_is_stable() {
        assert_eq!(
            format_weights(&[1.5, -0.0, 0.25]),
            "1.500000, -0.000000, 0.250000"
        );
    }
}
