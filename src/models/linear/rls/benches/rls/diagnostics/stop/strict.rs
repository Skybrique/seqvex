//! Strict-stop, retry, batch/bounded/executor and symmetric contrasts.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::error::FailureCause;
use seqvex::foundation::state::{BatchFailure, StateModel, process_batch};
use seqvex::models::linear::rls::{RlsBatchError, RlsError};

use crate::candidates::update_sym32;
use crate::diagnostics::failure::diag_check;
use crate::diagnostics::grouped::run_grouped32;
use crate::fixtures::{fingerprint32, state_eq32};
use crate::mode::set_unexpected_failure;

use super::StopDiagnosticContext;

pub(super) fn check_strict_stop_and_retry(ctx: &StopDiagnosticContext) -> Option<bool> {
    let group_a_direct = vec![
        ctx.prefix_a.clone(),
        ctx.prefix_b.clone(),
        ctx.invalid.clone(),
        ctx.suffix_a.clone(),
    ];
    let direct = run_grouped32(
        &ctx.model,
        ctx.initial.clone(),
        &group_a_direct,
        1,
        std::slice::from_ref(&ctx.group_b),
    );
    let direct_scheduled = 5_u64;
    let direct_unattempted = direct_scheduled - direct.attempted;
    let stop_ok = direct.attempted == 3
        && direct.successful == 2
        && direct.failed == 1
        && direct_unattempted == 2
        && direct.suffix_a_attempted == 0
        && direct.group_b_calls == 0
        && matches!(direct.error, Some(RlsError::NonFiniteInput))
        && direct
            .retained
            .as_ref()
            .is_some_and(|state| state_eq32(state, &ctx.prefix_state))
        && direct
            .retained
            .as_ref()
            .is_some_and(|state| fingerprint32(state) == ctx.prefix_fingerprint);
    let mut ok = diag_check(
        "F-StrictStop-direct",
        stop_ok,
        "expected failed observation, preserved prefix, unattempted suffix_a/group_b",
    );

    let retry = run_grouped32(
        &ctx.model,
        ctx.prefix_state.clone(),
        &[ctx.corrected.clone(), ctx.suffix_a.clone()],
        1,
        std::slice::from_ref(&ctx.group_b),
    );
    let retry_ok = retry.attempted == 2
        && retry.successful == 2
        && retry.failed == 0
        && retry.suffix_a_attempted == 1
        && retry.group_b_calls == 1
        && retry.attempted + retry.group_b_calls == 3
        && retry.retained.is_none();
    ok &= diag_check(
        "F-Retry",
        retry_ok,
        "expected caller-driven corrected retry from preserved prefix without replay",
    );
    Some(ok)
}

pub(super) fn check_batch_executor_and_symmetric_contrasts(
    ctx: &StopDiagnosticContext,
) -> Option<bool> {
    let mut ok = true;
    let group_a = vec![
        ctx.prefix_a.clone(),
        ctx.prefix_b.clone(),
        ctx.invalid.clone(),
        ctx.suffix_a.clone(),
    ];
    let batch = process_batch(&ctx.model, ctx.initial.clone(), group_a.iter().cloned());
    let batch_group_b_calls = match &batch {
        Ok(state) => {
            let _ = ctx.model.update(state, &ctx.group_b);
            1
        }
        Err(_) => 0,
    };
    let batch_ok = match batch {
        Err(BatchFailure { state, cause, .. }) => {
            state_eq32(&state, &ctx.prefix_state)
                && matches!(cause, FailureCause::Model(RlsError::NonFiniteInput))
                && batch_group_b_calls == 0
        }
        Ok(_) => false,
    };
    ok &= diag_check(
        "F-BatchContrast",
        batch_ok,
        "expected BatchFailure with prefix state and no later-group call",
    );

    let bounded = ctx
        .model
        .process_batch_reference(ctx.initial.clone(), &group_a, group_a.len());
    let bounded_group_b_calls = match &bounded {
        Ok(state) => {
            let _ = ctx.model.update(state, &ctx.group_b);
            1
        }
        Err(_) => 0,
    };
    let bounded_ok = match bounded {
        Err(RlsBatchError::Transition {
            failed_index,
            state,
            error,
        }) => {
            failed_index == 2
                && state_eq32(&state, &ctx.prefix_state)
                && matches!(error, RlsError::NonFiniteInput)
                && bounded_group_b_calls == 0
        }
        Ok(_) | Err(_) => false,
    };
    ok &= diag_check(
        "F-BoundedContrast",
        bounded_ok,
        "expected Transition with failed_index 2 and prefix state",
    );

    let mut executor = StreamingExecutor::new(&ctx.model, ctx.initial.clone());
    let mut executor_attempts = 0_u64;
    for observation in [&ctx.prefix_a, &ctx.prefix_b] {
        executor_attempts += 1;
        if executor.process_one(observation).is_err() {
            println!("DIAGNOSTIC F-ExecutorContrast setup FAIL");
            set_unexpected_failure();
            return None;
        }
    }
    executor_attempts += 1;
    let executor_before = executor.state().clone();
    let executor_failure = executor.process_one(&ctx.invalid);
    let executor_failed = executor_failure.is_err();
    let executor_state_ok = state_eq32(&executor_before, executor.state());
    let executor_group_b_calls = if executor_failed {
        0
    } else {
        let _ = executor.process_one(&ctx.group_b);
        1
    };
    ok &= diag_check(
        "F-ExecutorContrast",
        executor_failed
            && executor_state_ok
            && executor_attempts == 3
            && executor_group_b_calls == 0,
        "expected executor Err, retained committed state, and no later-group call",
    );

    let prefix_before_symmetric = ctx.prefix_state.clone();
    let symmetric_failure = update_sym32(&ctx.model, &ctx.prefix_state, ctx.invalid.value());
    ok &= diag_check(
        "F-SymStop",
        matches!(symmetric_failure, Err(RlsError::NonFiniteInput))
            && state_eq32(&ctx.prefix_state, &prefix_before_symmetric),
        "expected corrected symmetric control Err and unchanged prefix",
    );
    Some(ok)
}
