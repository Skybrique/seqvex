//! Reproducible CSV-to-frozen-serving walkthrough for OLS A/B/C.
//!
//! The embedded CSV is authored illustrative data, not a measurement. An
//! optional path accepts the same narrow two-column schema.

use std::process::ExitCode;

#[path = "fit_profiles/reporting.rs"]
mod reporting;
#[path = "fit_profiles/support.rs"]
mod support;
#[path = "fit_profiles/workflow.rs"]
mod workflow;

fn main() -> ExitCode {
    match workflow::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fit_profiles: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::support::*;

    static TEMP_ID: AtomicUsize = AtomicUsize::new(0);

    fn temp_path(label: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "seqvex-fit-profiles-{}-{}-{label}.csv",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn valid_csv(rows: usize) -> Vec<u8> {
        let mut csv = String::from("DisplacementMeters,TimeOffsetSeconds\n");
        for row in 0..rows {
            csv.push_str(&format!("{row},{row}\n"));
        }
        csv.into_bytes()
    }

    #[test]
    fn embedded_fixture_has_one_input_feature_and_declared_split() {
        let (bytes, source) = read_input(&[]).unwrap();
        let data = parse_dataset(bytes, source).unwrap();
        assert!(data.canonical);
        assert_eq!(data.features64.len(), 8);
        assert_eq!(data.targets64.len(), 8);
        assert_eq!(data.features64[0].len(), 1);
        assert_eq!(data.features32[0].len(), 1);
        assert_eq!(TRAIN_ROWS, 5);
        assert_eq!(data.features64.len() - TRAIN_ROWS, 3);
    }

    #[test]
    fn optional_path_reads_identical_canonical_bytes() {
        let path = temp_path("canonical");
        fs::write(&path, CANONICAL_CSV).unwrap();
        let args = [OsString::from(path.as_os_str())];
        let (bytes, source) = read_input(&args).unwrap();
        let data = parse_dataset(bytes, source).unwrap();
        assert!(data.canonical);
        assert_eq!(data.features64.len(), 8);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn valid_external_csv_does_not_receive_canonical_assertions() {
        let mut bytes = valid_csv(6);
        bytes.extend_from_slice(b"\n");
        let data = parse_dataset(bytes, "external fixture".to_owned()).unwrap();
        assert!(!data.canonical);
        assert_eq!(data.features64.len(), 6);
    }

    #[test]
    fn headers_and_record_shapes_are_strict() {
        for csv in [
            b"DisplacementMeters\n0,1\n1,2\n2,3\n3,4\n4,5\n5,6\n".as_slice(),
            b"DisplacementMeters,DisplacementMeters\n0,1\n1,2\n2,3\n3,4\n4,5\n5,6\n",
            b"TimeOffsetSeconds,DisplacementMeters\n0,1\n1,2\n2,3\n3,4\n4,5\n5,6\n",
            b"DisplacementMeters,TimeOffsetSeconds,Extra\n0,1,2\n1,2,3\n2,3,4\n3,4,5\n4,5,6\n5,6,7\n",
            b"DisplacementMeters,TimeOffsetSeconds\n0,1\n1,2\n2,3\n3,4\n4,5\n5\n",
        ] {
            assert!(parse_dataset(csv.to_vec(), "invalid fixture".to_owned()).is_err());
        }
    }

    #[test]
    fn empty_malformed_nonfinite_and_too_few_inputs_are_rejected() {
        assert!(parse_dataset(Vec::new(), "empty".to_owned()).is_err());
        for value in ["", "bad", "NaN", "inf"] {
            let csv =
                format!("{FEATURE_HEADER},{TARGET_HEADER}\n{value},1\n1,2\n2,3\n3,4\n4,5\n5,6\n");
            assert!(parse_dataset(csv.into_bytes(), "invalid value".to_owned()).is_err());
        }
        assert!(parse_dataset(valid_csv(5), "too few".to_owned()).is_err());
    }

    #[test]
    fn mantissa_only_underflow_check_preserves_zero_sign_and_subnormals() {
        assert_eq!(
            parse_f64("0e10", 0, FEATURE_HEADER).unwrap().to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            parse_f32("0e10", 0, FEATURE_HEADER).unwrap().to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            parse_f64("-0e-10", 0, FEATURE_HEADER).unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            parse_f32("-0e-10", 0, FEATURE_HEADER).unwrap().to_bits(),
            (-0.0f32).to_bits()
        );
        assert!(parse_f64("1e-400", 0, FEATURE_HEADER).is_err());
        assert!(parse_f32("1e-50", 0, FEATURE_HEADER).is_err());
        assert_ne!(parse_f64("1e-310", 0, FEATURE_HEADER).unwrap(), 0.0);
        assert_ne!(parse_f32("1e-40", 0, FEATURE_HEADER).unwrap(), 0.0);
    }

    #[test]
    fn b_serving_input_narrowing_is_checked_and_preserves_signed_zero() {
        assert_eq!(
            narrow_b_input(-0.0, 0).unwrap().to_bits(),
            (-0.0f32).to_bits()
        );
        assert_ne!(narrow_b_input(1.0e-40, 0).unwrap(), 0.0);
        assert!(narrow_b_input(1.0e-50, 0).is_err());
        assert!(narrow_b_input(f64::MAX, 0).is_err());
    }

    #[test]
    fn numeric_tokens_are_parsed_directly_at_both_precisions() {
        let value64 = parse_f64("0.1", 0, FEATURE_HEADER).unwrap();
        let value32 = parse_f32("0.1", 0, FEATURE_HEADER).unwrap();
        assert_eq!(value64.to_bits(), 0x3fb9_9999_9999_999a);
        assert_eq!(value32.to_bits(), 0x3dcc_cccd);
        assert_eq!(f64::from(value32).to_bits(), 0x3fb9_9999_a000_0000);
    }

    #[test]
    fn input_byte_and_row_limits_are_enforced() {
        assert!(parse_dataset(vec![b' '; MAX_INPUT_BYTES + 1], "large".to_owned()).is_err());
        assert!(parse_dataset(valid_csv(MAX_DATA_ROWS + 1), "many rows".to_owned()).is_err());

        let path = temp_path("over-limit");
        fs::write(&path, vec![b' '; MAX_INPUT_BYTES + 1]).unwrap();
        assert!(read_input(&[OsString::from(path.as_os_str())]).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn arguments_are_bounded_and_unreadable_paths_are_reported() {
        assert!(read_input(&[OsString::from("one"), OsString::from("two")]).is_err());
        let path = temp_path("missing");
        assert!(read_input(&[OsString::from(path.as_os_str())]).is_err());
    }
}
