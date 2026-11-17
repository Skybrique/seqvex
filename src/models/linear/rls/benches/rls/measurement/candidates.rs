//! Timed candidate-row runners built on the shared precheck path.

use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64, RlsState, RlsState64};

use crate::WINDOWS;
use crate::candidates::{Candidate, precheck32, precheck64, record_precheck_failure};
use crate::evidence::RawWindow;
use crate::measurement::update::{UpdateRunStatus, measure_update32, measure_update64};

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_candidate32(
    row: &str,
    model: &Rls,
    warm_state: &RlsState,
    expected: &RlsState,
    observations: &[Observation<RlsSample>],
    candidate: Candidate,
    warm: usize,
    horizon: usize,
    raws: &mut Vec<RawWindow>,
) {
    match precheck32(model, observations, warm, horizon, candidate) {
        Ok(_) => {
            let mut ignored_failure = None;
            let mut ignored_raw = None;
            let mut ignored_status = UpdateRunStatus::default();
            let _ = measure_update32(
                row,
                model,
                warm_state,
                expected,
                observations,
                candidate,
                raws,
                WINDOWS,
                true,
                true,
                &mut ignored_failure,
                &mut ignored_raw,
                &mut ignored_status,
            );
        }
        Err(failure) => {
            record_precheck_failure(row, "f32", model.dimension(), warm, horizon, failure)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_candidate64(
    row: &str,
    model: &Rls64,
    warm_state: &RlsState64,
    expected: &RlsState64,
    observations: &[Observation<RlsSample64>],
    candidate: Candidate,
    warm: usize,
    horizon: usize,
    raws: &mut Vec<RawWindow>,
) {
    match precheck64(model, observations, warm, horizon, candidate) {
        Ok(_) => {
            let mut ignored_failure = None;
            let mut ignored_raw = None;
            let mut ignored_status = UpdateRunStatus::default();
            let _ = measure_update64(
                row,
                model,
                warm_state,
                expected,
                observations,
                candidate,
                raws,
                WINDOWS,
                true,
                true,
                &mut ignored_failure,
                &mut ignored_raw,
                &mut ignored_status,
            );
        }
        Err(failure) => {
            record_precheck_failure(row, "f64", model.dimension(), warm, horizon, failure)
        }
    }
}
