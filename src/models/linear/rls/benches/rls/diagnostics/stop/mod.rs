//! Ordered stop-diagnostic coordinator and shared context.
//!
//! The pre-split `run_stop_diagnostics` performed shared setup, then a sequence
//! of independent diagnostic groups. Each group still returns its own result;
//! `None` is reserved for an original setup/precondition early return, while
//! `Some(false)` is an ordinary failed diagnostic that must not skip later
//! groups.

mod evidence;
mod foundation;
mod postcondition;
mod sharing;
mod strict;

use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, RlsSample, RlsState};

use crate::fixtures::{fingerprint32, make_model32, sample32};
use crate::mode::set_unexpected_failure;

struct StopDiagnosticContext {
    model: Rls,
    initial: RlsState,
    prefix_state: RlsState,
    prefix_fingerprint: u64,
    prefix_a: Observation<RlsSample>,
    prefix_b: Observation<RlsSample>,
    invalid: Observation<RlsSample>,
    suffix_a: Observation<RlsSample>,
    corrected: Observation<RlsSample>,
    group_b: Observation<RlsSample>,
}

impl StopDiagnosticContext {
    fn new() -> Option<Self> {
        let model = make_model32(1, 1.0, 1.0)?;
        let prefix_a = sample32(&[1.0], 1.0);
        let prefix_b = sample32(&[0.5], 0.5);
        let invalid = sample32(&[f32::NAN], 0.0);
        let suffix_a = sample32(&[0.25], 0.25);
        let corrected = sample32(&[0.75], 0.75);
        let group_b = sample32(&[0.125], 0.125);
        let initial = model.initial_state();

        let mut prefix_state = initial.clone();
        for observation in [&prefix_a, &prefix_b] {
            match model.update(&prefix_state, observation) {
                Ok(next) => prefix_state = next,
                Err(error) => {
                    println!("DIAGNOSTIC F-StrictStop setup FAIL {error:?}");
                    set_unexpected_failure();
                    return None;
                }
            }
        }
        let prefix_fingerprint = fingerprint32(&prefix_state);

        Some(Self {
            model,
            initial,
            prefix_state,
            prefix_fingerprint,
            prefix_a,
            prefix_b,
            invalid,
            suffix_a,
            corrected,
            group_b,
        })
    }
}

pub(super) fn run_stop_diagnostics() -> bool {
    let Some(context) = StopDiagnosticContext::new() else {
        return false;
    };
    let mut ok = true;
    match strict::check_strict_stop_and_retry(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match strict::check_batch_executor_and_symmetric_contrasts(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match sharing::check_sharing_controls(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match foundation::check_foundation_and_continue_controls(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match foundation::check_precheck_controls(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match postcondition::check_f32_postcondition_controls(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match postcondition::check_postcondition64_controls(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    match evidence::check_comparison_evidence_control(&context) {
        Some(value) => ok &= value,
        None => return false,
    }
    ok
}
