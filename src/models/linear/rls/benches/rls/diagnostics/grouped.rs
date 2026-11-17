//! Ordered grouped-update model used by the strict-stop diagnostics.

use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, RlsError, RlsSample, RlsState};

pub(super) struct GroupedRun {
    pub(super) attempted: u64,
    pub(super) successful: u64,
    pub(super) failed: u64,
    pub(super) suffix_a_attempted: u64,
    pub(super) group_b_calls: u64,
    pub(super) retained: Option<RlsState>,
    pub(super) error: Option<RlsError>,
}

pub(super) fn run_grouped32(
    model: &Rls,
    initial: RlsState,
    group_a: &[Observation<RlsSample>],
    suffix_a_len: usize,
    group_b: &[Observation<RlsSample>],
) -> GroupedRun {
    let mut result = GroupedRun {
        attempted: 0,
        successful: 0,
        failed: 0,
        suffix_a_attempted: 0,
        group_b_calls: 0,
        retained: None,
        error: None,
    };
    let mut state = initial;
    for observation in group_a {
        result.attempted += 1;
        match model.update(&state, observation) {
            Ok(next) => {
                state = next;
                result.successful += 1;
            }
            Err(error) => {
                result.failed += 1;
                result.retained = Some(state.clone());
                result.error = Some(error);
                break;
            }
        }
    }
    let prefix_and_failure_len = group_a.len().saturating_sub(suffix_a_len) as u64;
    result.suffix_a_attempted = if result.failed == 0 {
        suffix_a_len as u64
    } else {
        result
            .attempted
            .saturating_sub(prefix_and_failure_len)
            .min(suffix_a_len as u64)
    };
    if result.failed == 0 {
        for observation in group_b {
            result.group_b_calls += 1;
            match model.update(&state, observation) {
                Ok(next) => state = next,
                Err(error) => {
                    result.failed += 1;
                    result.retained = Some(state.clone());
                    result.error = Some(error);
                    break;
                }
            }
        }
    }
    result
}
