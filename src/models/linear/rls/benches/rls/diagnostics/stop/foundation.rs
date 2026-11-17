//! Foundation stop/continue and precheck controls.

use seqvex::foundation::state::{StateModel, process_one, process_stream_continuing};
use seqvex::models::linear::rls::RlsError;

use crate::candidates::{Candidate, precheck32};
use crate::diagnostics::failure::diag_check;
use crate::fixtures::state_eq32;
use crate::mode::set_unexpected_failure;

use super::StopDiagnosticContext;

pub(super) fn check_foundation_and_continue_controls(ctx: &StopDiagnosticContext) -> Option<bool> {
    let foundation_before = ctx.prefix_state.clone();
    let foundation_failure = process_one(&ctx.model, &ctx.prefix_state, &ctx.invalid);
    let mut ok = diag_check(
        "F-FoundationStop",
        matches!(foundation_failure, Err(RlsError::NonFiniteInput))
            && state_eq32(&ctx.prefix_state, &foundation_before),
        "expected foundation process_one Err with unchanged retained state",
    );

    let mut continue_failures = 0_u64;
    let continue_summary = process_stream_continuing(
        &ctx.model,
        ctx.initial.clone(),
        vec![
            ctx.prefix_a.clone(),
            ctx.prefix_b.clone(),
            ctx.invalid.clone(),
            ctx.suffix_a.clone(),
            ctx.group_b.clone(),
        ],
        |position, error| {
            if position == 2 && matches!(*error, RlsError::NonFiniteInput) {
                continue_failures += 1;
            }
        },
    )
    .unwrap();
    let continue_state = continue_summary.state;
    let mut expected = ctx.prefix_state.clone();
    for observation in [&ctx.suffix_a, &ctx.group_b] {
        match ctx.model.update(&expected, observation) {
            Ok(next) => expected = next,
            Err(error) => {
                println!("DIAGNOSTIC F-ContinueContrast setup FAIL {error:?}");
                set_unexpected_failure();
                return None;
            }
        }
    }
    ok &= diag_check(
        "F-ContinueContrast",
        continue_failures == 1 && state_eq32(&continue_state, &expected),
        "expected process_stream to continue from the last valid state",
    );
    Some(ok)
}

pub(super) fn check_precheck_controls(ctx: &StopDiagnosticContext) -> Option<bool> {
    let comparison_observations = vec![ctx.prefix_a.clone(), ctx.prefix_b.clone()];
    let comparison = precheck32(
        &ctx.model,
        &comparison_observations,
        1,
        1,
        Candidate::Mismatch,
    );
    let comparison_ok = matches!(
        comparison,
        Err(ref failure)
            if failure.kind == "comparison_failure"
                && failure.validation
                && failure.index == 0
                && failure.first_divergence.is_some()
    );
    Some(diag_check(
        "F-ComparisonOnly",
        comparison_ok,
        "expected the real precheck comparison path to localize a boundary mismatch",
    ))
}
