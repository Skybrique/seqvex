//! Sharing A/B controls.

use seqvex::foundation::state::process_one;

use crate::diagnostics::failure::diag_check;
use crate::fixtures::{fingerprint32, sample32};
use crate::measurement::{SharingFailure, measure_sharing};
use crate::mode::has_unexpected_failure;

use super::StopDiagnosticContext;

pub(super) fn check_sharing_controls(ctx: &StopDiagnosticContext) -> Option<bool> {
    let sharing_valid = sample32(&[1.0], 1.0);
    let sharing_invalid = sample32(&[f32::NAN], 0.0);
    let initial_fingerprint = fingerprint32(&ctx.initial);

    let mut stream_a = ctx.initial.clone();
    let stream_b = ctx.initial.clone();
    let mut sharing_raws = Vec::new();
    let mut sharing_failure = None;
    let mut sharing_record = None;
    let failure_before = has_unexpected_failure();
    let sharing_summary = measure_sharing(
        "F-Sharing-A",
        "f32",
        1,
        1,
        0,
        false,
        true,
        &mut sharing_failure,
        &mut sharing_record,
        |_| match process_one(&ctx.model, &stream_a, &sharing_invalid) {
            Ok(next) => {
                stream_a = next;
                Ok((1, 1))
            }
            Err(error) => Err(SharingFailure {
                stream: "A",
                a_attempted: 1,
                a_committed: 0,
                b_attempted: 0,
                b_committed: 0,
                retained_a: fingerprint32(&stream_a),
                retained_b: fingerprint32(&stream_b),
                message: format!("{error:?}"),
            }),
        },
        &mut sharing_raws,
    );
    let a_failure_ok = sharing_summary.is_none()
        && sharing_failure.as_ref().is_some_and(|failure| {
            failure.stream == "A"
                && failure.a_attempted == 1
                && failure.a_committed == 0
                && failure.b_attempted == 0
                && failure.b_committed == 0
                && failure.message.contains("NonFiniteInput")
                && failure.retained_a == initial_fingerprint
                && failure.retained_b == initial_fingerprint
        })
        && sharing_record.as_ref().is_some_and(|record| {
            record.status == "invalid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && record.failure_record_id.is_some()
                && record.scheduled_units == 2
                && record.attempted_units == 1
                && record.successful_units == 0
                && record.failed_units == 1
                && record.unattempted_units == 1
                && record.completed_updates == 0
                && record.stream_a_attempted == Some(1)
                && record.stream_a_committed == Some(0)
                && record.stream_b_attempted == Some(0)
                && record.stream_b_committed == Some(0)
        })
        && !failure_before
        && !has_unexpected_failure();
    let mut ok = diag_check(
        "F-Sharing-A",
        a_failure_ok,
        "expected stream A runner failure with invalid record and no summary",
    );

    let mut stream_a = ctx.initial.clone();
    let mut stream_b = ctx.initial.clone();
    let mut sharing_raws = Vec::new();
    let mut sharing_failure = None;
    let mut sharing_record = None;
    let failure_before = has_unexpected_failure();
    let sharing_summary = measure_sharing(
        "F-Sharing-B",
        "f32",
        1,
        1,
        0,
        false,
        true,
        &mut sharing_failure,
        &mut sharing_record,
        |_| match process_one(&ctx.model, &stream_a, &sharing_valid) {
            Ok(next) => {
                stream_a = next;
                match process_one(&ctx.model, &stream_b, &sharing_invalid) {
                    Ok(next_b) => {
                        stream_b = next_b;
                        Ok((1, 1))
                    }
                    Err(error) => Err(SharingFailure {
                        stream: "B",
                        a_attempted: 1,
                        a_committed: 1,
                        b_attempted: 1,
                        b_committed: 0,
                        retained_a: fingerprint32(&stream_a),
                        retained_b: fingerprint32(&stream_b),
                        message: format!("{error:?}"),
                    }),
                }
            }
            Err(error) => Err(SharingFailure {
                stream: "A",
                a_attempted: 1,
                a_committed: 0,
                b_attempted: 0,
                b_committed: 0,
                retained_a: fingerprint32(&stream_a),
                retained_b: fingerprint32(&stream_b),
                message: format!("{error:?}"),
            }),
        },
        &mut sharing_raws,
    );
    let b_failure_ok = sharing_summary.is_none()
        && sharing_failure.as_ref().is_some_and(|failure| {
            failure.stream == "B"
                && failure.a_attempted == 1
                && failure.a_committed == 1
                && failure.b_attempted == 1
                && failure.b_committed == 0
                && failure.message.contains("NonFiniteInput")
                && failure.retained_a != initial_fingerprint
                && failure.retained_b == initial_fingerprint
        })
        && sharing_record.as_ref().is_some_and(|record| {
            record.status == "invalid"
                && !record.timed
                && record.elapsed_ns.is_none()
                && record.allocation_requests.is_none()
                && record.requested_bytes.is_none()
                && record.failure_record_id.is_some()
                && record.scheduled_units == 2
                && record.attempted_units == 2
                && record.successful_units == 1
                && record.failed_units == 1
                && record.unattempted_units == 0
                && record.completed_updates == 1
                && record.stream_a_attempted == Some(1)
                && record.stream_a_committed == Some(1)
                && record.stream_b_attempted == Some(1)
                && record.stream_b_committed == Some(0)
        })
        && !failure_before
        && !has_unexpected_failure();
    ok &= diag_check(
        "F-Sharing-B",
        b_failure_ok,
        "expected stream B runner failure with A committed, invalid record, no summary",
    );
    Some(ok)
}
