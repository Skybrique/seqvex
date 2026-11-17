//! Decision-bearing operation-count diagnostics.

use crate::DELTA;
use crate::common;
use crate::diagnostics::failure::diag_check;
use crate::evidence::{failure_id, sweep_failure_record};
use crate::fixtures::{fingerprint32, make_model32, sample32};
use crate::mode::set_unexpected_failure;
use crate::validation::{legacy_horizon, sweep_state_after_outer_calls};
use seqvex::foundation::state::StateModel;

pub(super) fn run_operation_count_diagnostics() -> bool {
    let Some(model) = make_model32(4, common::LAMBDA_PE, DELTA) else {
        return false;
    };
    // Fixed plan: 4 outer calls x 4 observations. Index 11 fails: groups 0 and
    // 1 commit (8), group 2 commits its first 3 observations, the 4th fails,
    // and 4 observations remain unattempted.
    let make_observation =
        |offset: usize| sample32(&[0.5 + offset as f32 * 0.01, 0.25, 0.75, 0.125], 0.5);
    let mut observations: Vec<_> = (0..16).map(make_observation).collect();
    observations[11] = sample32(&[f32::NAN, 0.0, 0.0, 0.0], 0.0);

    let mut expected_prefix = model.initial_state();
    for observation in &observations[..11] {
        match model.update(&expected_prefix, observation) {
            Ok(next) => expected_prefix = next,
            Err(error) => {
                println!("DIAGNOSTIC F-OperationCounts setup FAIL {error:?}");
                set_unexpected_failure();
                return false;
            }
        }
    }
    let expected_fingerprint = fingerprint32(&expected_prefix);

    let mut ok = true;
    for &batch in &[2_usize, 3, 4] {
        let label = format!("F-OperationCounts-batch{batch}");
        let evidence =
            match sweep_state_after_outer_calls(&model, &observations, 4, batch, 4, false) {
                Ok(_) => {
                    ok &= diag_check(
                        &label,
                        false,
                        "expected a bounded-reference precheck failure at observation 11",
                    );
                    continue;
                }
                Err(evidence) => evidence,
            };
        ok &= diag_check(
            &label,
            evidence.scheduled == 16
                && evidence.committed == 11
                && evidence.attempted == 12
                && evidence.retained_fingerprint == Some(expected_fingerprint)
                && evidence.message.contains("observation=11")
                && evidence.message.contains("NonFiniteInput"),
            "expected scheduled=16 committed=11 attempted=12 and the index-11 prefix fingerprint",
        );
    }

    for &batch in &[2_usize, 3] {
        let label = format!("F-OperationCounts-foundation-batch{batch}");
        let evidence = match sweep_state_after_outer_calls(&model, &observations, 4, batch, 4, true)
        {
            Ok(_) => {
                ok &= diag_check(
                    &label,
                    false,
                    "expected a foundation-fold precheck failure at observation 11",
                );
                continue;
            }
            Err(evidence) => evidence,
        };
        ok &= diag_check(
            &label,
            evidence.scheduled == 16
                && evidence.committed == 11
                && evidence.attempted == 12
                && evidence.retained_fingerprint == Some(expected_fingerprint)
                && evidence.message.contains("observation=11")
                && evidence.message.contains("NonFiniteInput"),
            "expected the foundation fold to report the same committed prefix",
        );
    }

    let horizon_evidence = match legacy_horizon(&model, &observations, 4, 4) {
        Ok(()) => {
            ok &= diag_check(
                "F-OperationCounts-horizon",
                false,
                "expected a horizon precheck failure at observation 11",
            );
            return ok;
        }
        Err(evidence) => evidence,
    };
    let record = sweep_failure_record("F-OperationCounts", 4, horizon_evidence.clone().into());
    let expected_failure_id = failure_id("F-OperationCounts", "precheck", None, None);
    ok &= diag_check(
        "F-OperationCounts-horizon",
        horizon_evidence.scheduled == 16
            && horizon_evidence.committed == 11
            && horizon_evidence.attempted == 12
            && horizon_evidence.retained_fingerprint == Some(expected_fingerprint)
            && horizon_evidence.message.contains("observation=11")
            && horizon_evidence.message.contains("NonFiniteInput")
            && record.kind == "operation_error"
            && record.committed_progress == 11
            && record.scheduled_units == 16
            && record.attempted_units == 12
            && record.successful_units == 11
            && record.failed_units == 1
            && record.unattempted_units == 4
            && record.retained_fingerprint == Some(expected_fingerprint)
            && record.attempted_units <= record.scheduled_units
            && record.successful_units + record.failed_units == record.attempted_units
            && record.unattempted_units == record.scheduled_units - record.attempted_units
            && failure_id(
                &record.row,
                record.phase,
                record.window,
                record.observation_global,
            ) == expected_failure_id,
        "expected scheduled=16 committed=11 attempted=12 failed=1 unattempted=4 and linkage",
    );
    record.emit_observed();
    ok
}
