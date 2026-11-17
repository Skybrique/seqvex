//! RLS benchmark measurement entry point.
//!
//! Owns the measurement coordinator and re-exports the cross-subtree helpers
//! consumed by diagnostics and validation. Child modules own their runners.

mod candidates;
mod components;
mod control;
mod legacy;
mod memory;
mod update;

pub(crate) use components::{RowIdentity, catalog_report};
pub(crate) use control::{SharingFailure, measure_control, measure_sharing};
pub(crate) use legacy::{LegacyFailure, measure_legacy, measure_legacy_impl};
pub(crate) use update::{UpdateRunStatus, measure_update32, measure_update64};

use self::candidates::{measure_candidate32, measure_candidate64};
use self::components::{check_catalog, run_components32, run_components64};
use self::memory::{run_memory_probes32, run_memory_probes64};
use crate::candidates::{
    Candidate, candidate_row, precheck32, precheck64, record_precheck_failure,
};
use crate::common::comparison::{baseline_first, capture_from_env};
use crate::evidence::emit_retained_records;
use crate::fixtures::{make_model32, make_model64, observations32, observations64};
use crate::mode::{horizon, warmup};
use crate::{DELTA_F32, DELTA_F64, DIMS, LAMBDA_F32, LAMBDA_F64, WINDOWS};

pub(crate) fn run_measurement() {
    let warm = warmup();
    let h = horizon();
    let capture = capture_from_env();
    let order = baseline_first(
        capture,
        &[
            Candidate::Production,
            Candidate::Fused,
            Candidate::Symmetric,
        ],
    );
    let u_order: Vec<Candidate> = order
        .into_iter()
        .filter(|candidate| *candidate != Candidate::Production)
        .collect();
    println!("capture={} candidate_order={u_order:?}", capture.number());
    println!(
        "UNITS allocation-requests/<unit>=wrapper alloc+realloc requests; requested-bytes/<unit>=requested payload bytes; MEMORY retained-requested-bytes and peak-live-requested-bytes-above-baseline are live requested payload, not allocator overhead"
    );
    let mut raws = Vec::new();
    for d in DIMS {
        let Some(model) = make_model32(d, LAMBDA_F32, DELTA_F32) else {
            continue;
        };
        let observations = observations32(d);
        let production = match precheck32(&model, &observations, warm, h, Candidate::Production) {
            Ok(value) => value,
            Err(failure) => {
                record_precheck_failure("S-C", "f32", d, warm, h, failure);
                continue;
            }
        };
        let (warm_state, expected) = production;
        let mut ignored_failure = None;
        let mut ignored_raw = None;
        let mut ignored_status = UpdateRunStatus::default();
        let _ = measure_update32(
            "S-C",
            &model,
            &warm_state,
            &expected,
            &observations,
            Candidate::Production,
            &mut raws,
            WINDOWS,
            true,
            true,
            &mut ignored_failure,
            &mut ignored_raw,
            &mut ignored_status,
        );
        for candidate in &u_order {
            let row = candidate_row("f32", *candidate);
            measure_candidate32(
                row,
                &model,
                &warm_state,
                &expected,
                &observations,
                *candidate,
                warm,
                h,
                &mut raws,
            );
        }
        let last = &observations[warm + h - 1];
        run_components32(&model, &expected, last, &mut raws);
        run_memory_probes32(&model, &warm_state, last, &mut raws);

        let Some(model64) = make_model64(d, LAMBDA_F64, DELTA_F64) else {
            continue;
        };
        let observations64 = observations64(d);
        let production64 =
            match precheck64(&model64, &observations64, warm, h, Candidate::Production) {
                Ok(value) => value,
                Err(failure) => {
                    record_precheck_failure("S-A", "f64", d, warm, h, failure);
                    continue;
                }
            };
        let (warm_state64, expected64) = production64;
        let mut ignored_failure = None;
        let mut ignored_raw = None;
        let mut ignored_status = UpdateRunStatus::default();
        let _ = measure_update64(
            "S-A",
            &model64,
            &warm_state64,
            &expected64,
            &observations64,
            Candidate::Production,
            &mut raws,
            WINDOWS,
            true,
            true,
            &mut ignored_failure,
            &mut ignored_raw,
            &mut ignored_status,
        );
        for candidate in &u_order {
            let row = candidate_row("f64", *candidate);
            measure_candidate64(
                row,
                &model64,
                &warm_state64,
                &expected64,
                &observations64,
                *candidate,
                warm,
                h,
                &mut raws,
            );
        }
        let last64 = &observations64[warm + h - 1];
        run_components64(&model64, &expected64, last64, &mut raws);
        run_memory_probes64(&model64, &warm_state64, last64, &mut raws);
    }
    check_catalog(&raws);
    let emitted = emit_retained_records(&mut raws);
    println!(
        "RETAINED records_emitted={emitted} records_total={}",
        raws.len()
    );
}
