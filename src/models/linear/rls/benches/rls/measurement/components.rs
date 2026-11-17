//! Component controls, row catalog and retained-record verification.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;

use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64, RlsState, RlsState64};

use crate::candidates::{
    p_full_into32, p_full_into64, p_sym_into32, p_sym_into64, vcur_alloc32, vcur_alloc64,
    vcur_pre32, vcur_pre64, vfused_alloc32, vfused_alloc64, vfused_pre32, vfused_pre64,
};
use crate::evidence::{FailureRecord, RawWindow};
use crate::fixtures::{fingerprint32, fingerprint64};
use crate::measurement::control::measure_control;
use crate::mode::{horizon, set_unexpected_failure, warmup};
use crate::{CONTROL_ITERS, DIMS, WINDOWS};

pub(super) fn run_components32(
    model: &Rls,
    expected: &RlsState,
    last: &Observation<RlsSample>,
    raws: &mut Vec<RawWindow>,
) {
    let d = model.dimension();
    let sample = last.value();
    let v = match expected.p.mul_vector(&sample.features) {
        Ok(value) => value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f32",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint32(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    let denom = match sample.features.dot(&v) {
        Ok(value) => model.lambda() + value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f32",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint32(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    if !denom.is_finite() || denom <= 0.0 {
        FailureRecord {
            kind: "operation_error",
            phase: "control_setup",
            row: "C-Pfull".to_owned(),
            precision: "f32",
            dimension: d,
            window: None,
            observation_global: Some(warmup() + horizon() - 1),
            observation_window: Some(horizon() - 1),
            committed_progress: 0,
            retained_fingerprint: Some(fingerprint32(expected)),
            scheduled_units: 0,
            attempted_units: 0,
            successful_units: 0,
            failed_units: 0,
            unattempted_units: 0,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            error: Some("invalid control denominator".to_owned()),
        }
        .emit();
        return;
    }
    let prediction = match expected.w.dot(&sample.features) {
        Ok(value) => value,
        Err(error) => {
            control_setup_failure("C-Pfull", "f32", d, format!("{error:?}"));
            return;
        }
    };
    let error = sample.target - prediction;
    let lambda = model.lambda();
    let mut full = vec![0.0_f32; d * d];
    let _ = measure_control(
        "C-Pfull",
        "f32",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_full_into32(
                &mut full,
                expected.p.as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&full);
            Ok(())
        },
        raws,
    );
    let mut sym = vec![0.0_f32; d * d];
    let _ = measure_control(
        "C-Psym",
        "f32",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_sym_into32(
                &mut sym,
                expected.p.as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&sym);
            Ok(())
        },
        raws,
    );
    let _ = measure_control(
        "C-Vcur-alloc",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vcur_alloc32(&expected.w, &v, error, denom),
        raws,
    );
    let mut gain = vec![0.0_f32; d];
    let mut scaled = vec![0.0_f32; d];
    let mut next = vec![0.0_f32; d];
    let _ = measure_control(
        "C-Vcur-pre",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || {
            vcur_pre32(
                &expected.w,
                &v,
                error,
                denom,
                &mut gain,
                &mut scaled,
                &mut next,
            )
        },
        raws,
    );
    let _ = measure_control(
        "C-Vfused-alloc",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_alloc32(&expected.w, &v, error, denom),
        raws,
    );
    let mut fused = vec![0.0_f32; d];
    let _ = measure_control(
        "C-Vfused-pre",
        "f32",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_pre32(&expected.w, &v, error, denom, &mut fused),
        raws,
    );
    let _ = measure_control(
        "C-Alloc",
        "f32",
        d,
        "allocation_set",
        CONTROL_ITERS,
        || {
            let a = Vec::<f32>::with_capacity(d);
            let b = Vec::<f32>::with_capacity(d);
            let c = Vec::<f32>::with_capacity(d);
            let e = Vec::<f32>::with_capacity(d);
            let p = Vec::<f32>::with_capacity(d * d);
            black_box((&a, &b, &c, &e, &p));
            Ok(())
        },
        raws,
    );
}

pub(super) fn run_components64(
    model: &Rls64,
    expected: &RlsState64,
    last: &Observation<RlsSample64>,
    raws: &mut Vec<RawWindow>,
) {
    let d = model.dimension();
    let sample = last.value();
    let v = match expected.covariance().mul_vector(&sample.features) {
        Ok(value) => value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f64",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint64(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    let denom = match sample.features.dot(&v) {
        Ok(value) => model.lambda() + value,
        Err(error) => {
            FailureRecord {
                kind: "operation_error",
                phase: "control_setup",
                row: "C-Pfull".to_owned(),
                precision: "f64",
                dimension: d,
                window: None,
                observation_global: Some(warmup() + horizon() - 1),
                observation_window: Some(horizon() - 1),
                committed_progress: 0,
                retained_fingerprint: Some(fingerprint64(expected)),
                scheduled_units: 0,
                attempted_units: 0,
                successful_units: 0,
                failed_units: 0,
                unattempted_units: 0,
                validation_failure: false,
                postcondition_failure: false,
                first_divergence: None,
                error: Some(format!("{error:?}")),
            }
            .emit();
            return;
        }
    };
    if !denom.is_finite() || denom <= 0.0 {
        FailureRecord {
            kind: "operation_error",
            phase: "control_setup",
            row: "C-Pfull".to_owned(),
            precision: "f64",
            dimension: d,
            window: None,
            observation_global: Some(warmup() + horizon() - 1),
            observation_window: Some(horizon() - 1),
            committed_progress: 0,
            retained_fingerprint: Some(fingerprint64(expected)),
            scheduled_units: 0,
            attempted_units: 0,
            successful_units: 0,
            failed_units: 0,
            unattempted_units: 0,
            validation_failure: false,
            postcondition_failure: false,
            first_divergence: None,
            error: Some("invalid control denominator".to_owned()),
        }
        .emit();
        return;
    }
    let error = sample.target
        - match expected.weights().dot(&sample.features) {
            Ok(value) => value,
            Err(error) => {
                FailureRecord {
                    kind: "operation_error",
                    phase: "control_setup",
                    row: "C-Pfull".to_owned(),
                    precision: "f64",
                    dimension: d,
                    window: None,
                    observation_global: Some(warmup() + horizon() - 1),
                    observation_window: Some(horizon() - 1),
                    committed_progress: 0,
                    retained_fingerprint: Some(fingerprint64(expected)),
                    scheduled_units: 0,
                    attempted_units: 0,
                    successful_units: 0,
                    failed_units: 0,
                    unattempted_units: 0,
                    validation_failure: false,
                    postcondition_failure: false,
                    first_divergence: None,
                    error: Some(format!("{error:?}")),
                }
                .emit();
                return;
            }
        };
    let lambda = model.lambda();
    let mut full = vec![0.0_f64; d * d];
    let _ = measure_control(
        "C-Pfull",
        "f64",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_full_into64(
                &mut full,
                expected.covariance().as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&full);
            Ok(())
        },
        raws,
    );
    let mut sym = vec![0.0_f64; d * d];
    let _ = measure_control(
        "C-Psym",
        "f64",
        d,
        "buffer_computation",
        CONTROL_ITERS,
        || {
            p_sym_into64(
                &mut sym,
                expected.covariance().as_slice(),
                v.as_slice(),
                d,
                lambda,
                denom,
            );
            black_box(&sym);
            Ok(())
        },
        raws,
    );
    let _ = measure_control(
        "C-Vcur-alloc",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vcur_alloc64(expected.weights(), &v, error, denom),
        raws,
    );
    let mut gain = vec![0.0_f64; d];
    let mut scaled = vec![0.0_f64; d];
    let mut next = vec![0.0_f64; d];
    let _ = measure_control(
        "C-Vcur-pre",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || {
            vcur_pre64(
                expected.weights(),
                &v,
                error,
                denom,
                &mut gain,
                &mut scaled,
                &mut next,
            )
        },
        raws,
    );
    let _ = measure_control(
        "C-Vfused-alloc",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_alloc64(expected.weights(), &v, error, denom),
        raws,
    );
    let mut fused = vec![0.0_f64; d];
    let _ = measure_control(
        "C-Vfused-pre",
        "f64",
        d,
        "vector_step",
        CONTROL_ITERS,
        || vfused_pre64(expected.weights(), &v, error, denom, &mut fused),
        raws,
    );
    let _ = measure_control(
        "C-Alloc",
        "f64",
        d,
        "allocation_set",
        CONTROL_ITERS,
        || {
            let a = Vec::<f64>::with_capacity(d);
            let b = Vec::<f64>::with_capacity(d);
            let c = Vec::<f64>::with_capacity(d);
            let e = Vec::<f64>::with_capacity(d);
            let p = Vec::<f64>::with_capacity(d * d);
            black_box((&a, &b, &c, &e, &p));
            Ok(())
        },
        raws,
    );
}

fn expected_identities() -> BTreeSet<(String, &'static str, usize, usize)> {
    let mut set = BTreeSet::new();
    for (precision, s_row, d) in [
        ("f32", "S-C", DIMS[0]),
        ("f32", "S-C", DIMS[1]),
        ("f64", "S-A", DIMS[0]),
        ("f64", "S-A", DIMS[1]),
    ] {
        for window in 0..WINDOWS {
            set.insert((s_row.to_owned(), precision, d, window));
        }
    }
    for precision in ["f32", "f64"] {
        for d in DIMS {
            for row in [
                "U-Fused",
                "U-Sym",
                "C-Pfull",
                "C-Psym",
                "C-Vcur-alloc",
                "C-Vcur-pre",
                "C-Vfused-alloc",
                "C-Vfused-pre",
                "C-Alloc",
            ] {
                for window in 0..WINDOWS {
                    set.insert((row.to_owned(), precision, d, window));
                }
            }
        }
    }
    set
}

pub(crate) type RowIdentity = (String, &'static str, usize, usize);

pub(crate) fn catalog_report(
    expected: &BTreeSet<RowIdentity>,
    raws: &[RawWindow],
) -> (Vec<RowIdentity>, Vec<RowIdentity>, Vec<(RowIdentity, u64)>) {
    let mut counts: BTreeMap<RowIdentity, u64> = BTreeMap::new();
    for record in raws {
        if record.status != "warmup" && matches!(record.category, "S" | "U" | "C") {
            if let Some(window) = record.window_index {
                *counts
                    .entry((
                        record.row_id.clone(),
                        record.precision,
                        record.dimension,
                        window,
                    ))
                    .or_insert(0) += 1;
            }
        }
    }
    let missing = expected
        .iter()
        .filter(|identity| !counts.contains_key(*identity))
        .cloned()
        .collect();
    let extra = counts
        .keys()
        .filter(|identity| !expected.contains(*identity))
        .cloned()
        .collect();
    let duplicates = counts
        .iter()
        .filter(|(_, count)| **count != 1)
        .map(|(identity, count)| (identity.clone(), *count))
        .collect();
    (missing, extra, duplicates)
}

pub(super) fn check_catalog(raws: &[RawWindow]) -> bool {
    let expected = expected_identities();
    let (missing, extra, duplicates) = catalog_report(&expected, raws);
    if !missing.is_empty() || !extra.is_empty() || !duplicates.is_empty() {
        println!("CATALOG missing={missing:?} extra={extra:?} duplicates={duplicates:?}");
        set_unexpected_failure();
        false
    } else {
        println!(
            "CATALOG expected={} observed={} status=ok",
            expected.len(),
            expected.len()
        );
        true
    }
}

fn control_setup_failure(row: &str, precision: &'static str, dimension: usize, error: String) {
    FailureRecord {
        kind: "operation_error",
        phase: "control_setup",
        row: row.to_owned(),
        precision,
        dimension,
        window: None,
        observation_global: Some(warmup() + horizon() - 1),
        observation_window: Some(horizon() - 1),
        committed_progress: 0,
        retained_fingerprint: None,
        scheduled_units: 0,
        attempted_units: 0,
        successful_units: 0,
        failed_units: 0,
        unattempted_units: 0,
        validation_failure: false,
        postcondition_failure: false,
        first_divergence: None,
        error: Some(error),
    }
    .emit();
}
