//! Comparison-evidence record control and emission.

use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::rls::RlsState;

use crate::diagnostics::failure::diag_check;
use crate::evidence::{failure_id, sweep_comparison_failure, sweep_failure_record};
use crate::fixtures::{fingerprint32, first_divergence32};
use crate::mode::has_unexpected_failure;

use super::StopDiagnosticContext;

pub(super) fn check_comparison_evidence_control(ctx: &StopDiagnosticContext) -> Option<bool> {
    let mut altered_weights = ctx.prefix_state.w.as_slice().to_vec();
    if let Some(first) = altered_weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    let altered = RlsState {
        w: Vector::from_slice(&altered_weights),
        p: ctx.prefix_state.p.clone(),
    };
    let left_fingerprint = fingerprint32(&ctx.prefix_state);
    let right_fingerprint = fingerprint32(&altered);
    let difference = first_divergence32(&ctx.prefix_state, &altered)
        .unwrap_or_else(|| "states are bitwise equal".to_owned());
    // prefix_a and prefix_b both committed, so the compared trajectory's plan
    // and completed work are both 2 units.
    let comparison_failure = sweep_comparison_failure(
        "left-controlled",
        &ctx.prefix_state,
        "right-controlled",
        &altered,
        "diagnostic one-bit mismatch",
        2,
    );
    let message = comparison_failure.message();
    let evidence_ok = comparison_failure.left_path == "left-controlled"
        && comparison_failure.right_path == "right-controlled"
        && comparison_failure.left_fingerprint == left_fingerprint
        && comparison_failure.right_fingerprint == right_fingerprint
        && comparison_failure.scheduled == 2
        && comparison_failure.committed == 2
        && comparison_failure.first_difference == difference
        && message.contains(&format!("left-controlled fp=0x{left_fingerprint:016x}"))
        && message.contains(&format!("right-controlled fp=0x{right_fingerprint:016x}"))
        && message.contains("diagnostic one-bit mismatch")
        && message.contains(&difference)
        && difference != "states are bitwise equal";
    let failure_before = has_unexpected_failure();
    let record = sweep_failure_record("F-ComparisonEvidence", 1, comparison_failure.into());
    let expected_failure_id = failure_id("F-ComparisonEvidence", "precheck", None, None);
    let record_ok = record.kind == "comparison_failure"
        && record.phase == "precheck"
        && record.dimension == 1
        && record.retained_fingerprint.is_none()
        && record.committed_progress == 2
        && record.scheduled_units == 2
        && record.attempted_units == 2
        && record.successful_units == 2
        && record.failed_units == 0
        && record.unattempted_units == 0
        && record.validation_failure
        && !record.postcondition_failure
        && record.first_divergence.as_deref() == Some(difference.as_str())
        && record.attempted_units <= record.scheduled_units
        && record.successful_units + record.failed_units == record.attempted_units
        && record.unattempted_units == record.scheduled_units - record.attempted_units
        && record
            .error
            .as_deref()
            .is_some_and(|error| error.contains("left-controlled fp=0x"))
        && record
            .error
            .as_deref()
            .is_some_and(|error| error.contains("right-controlled fp=0x"))
        && record
            .error
            .as_deref()
            .is_some_and(|error| error.contains(&difference))
        && failure_id(
            &record.row,
            record.phase,
            record.window,
            record.observation_global,
        ) == expected_failure_id;
    record.emit_observed();
    Some(diag_check(
        "F-ComparisonEvidence",
        evidence_ok && record_ok && !failure_before && !has_unexpected_failure(),
        "expected comparison_failure classification, both fingerprints, exact divergence, consistent counts and linkage",
    ))
}
