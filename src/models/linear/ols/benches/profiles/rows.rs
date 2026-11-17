use std::hint::black_box;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    CheckedLinearRegression, CheckedLinearRegression64, FitConfig, FitConfig32, FitConfig64,
    FitResult64, LinearRegression, LinearRegression64, UnderflowPolicy, fit, fit_f32, fit_f64,
};

use super::fixture::{Fixture, PROBE_COUNT, fixture_key};
use super::guards::{
    check_b_conversion, check_fit_a, check_fit_c, check_fit_legacy, check_serving32,
    check_serving64, fit_a, fit_c, fit_legacy, model_bits32, model_bits64, source_bits64,
};
use super::harness::{Work, measure_windows, probe_memory, retained_model_heap};
use super::reporting::{
    report, report_common_diagnostics, report_cross_profile_differences, report_fit, row_identity,
};

pub(super) fn run_fit_rows(fixture: &Fixture, row_ids: &mut Vec<String>) {
    let key = fixture_key(fixture);
    let work = Work {
        calls: 1,
        observations: fixture.n,
    };

    let legacy = fit_legacy(fixture);
    check_fit_legacy(&legacy, fixture);
    let legacy_diagnostics = legacy.diagnostics();
    let legacy_rss = format!(
        "rss_f64_reference={:.17e} rss_f32_predictor={:.17e}",
        legacy_diagnostics.rss_f64_reference, legacy_diagnostics.rss_f32_predictor
    );
    let legacy_rank = format!(
        "rank={} required_rank={} rank_threshold={:.17e}",
        legacy_diagnostics.rank,
        legacy_diagnostics.required_rank,
        legacy_diagnostics.rank_threshold
    );
    let requested = legacy_diagnostics.requested_workspace_bytes;
    drop(legacy);
    let config = FitConfig::default();
    let mut legacy_timed = || {
        let features = black_box(&fixture.features32);
        let targets = black_box(&fixture.targets32);
        let selection = black_box(&fixture.selection);
        drop(black_box(
            fit(features, targets, selection, black_box(&config)).expect("legacy fit succeeds"),
        ));
    };
    let mut legacy_memory = || {
        fit(
            black_box(&fixture.features32),
            black_box(&fixture.targets32),
            black_box(&fixture.selection),
            black_box(&config),
        )
        .expect("legacy fit succeeds")
    };
    let legacy_measured = measure_windows(work, &mut legacy_timed);
    let legacy_probe = probe_memory(&mut legacy_memory);
    let legacy_id = row_identity("legacy_hybrid", "fit", &key, None);
    report_fit(
        &legacy_id,
        &legacy_rank,
        requested,
        &legacy_rss,
        &legacy_measured,
        &legacy_probe,
        row_ids,
    );

    let a = fit_a(fixture);
    check_fit_a(&a, fixture);
    let a_rss = format!("rss_f64={:.17e}", a.diagnostics().rss_f64);
    let a_rank = format!(
        "rank={} required_rank={} rank_threshold={:.17e}",
        a.diagnostics().rank,
        a.diagnostics().required_rank,
        a.diagnostics().rank_threshold
    );
    let requested = a.diagnostics().requested_workspace_bytes;
    drop(a);
    let config = FitConfig64::default();
    let mut a_timed = || {
        let features = black_box(&fixture.features64);
        let targets = black_box(&fixture.targets64);
        let selection = black_box(&fixture.selection);
        drop(black_box(
            fit_f64(features, targets, selection, black_box(&config)).expect("A fit succeeds"),
        ));
    };
    let mut a_memory = || {
        fit_f64(
            black_box(&fixture.features64),
            black_box(&fixture.targets64),
            black_box(&fixture.selection),
            black_box(&config),
        )
        .expect("A fit succeeds")
    };
    let a_measured = measure_windows(work, &mut a_timed);
    let a_probe = probe_memory(&mut a_memory);
    let a_id = row_identity("A", "fit", &key, None);
    report_fit(
        &a_id,
        &a_rank,
        requested,
        &a_rss,
        &a_measured,
        &a_probe,
        row_ids,
    );

    let c = fit_c(fixture);
    check_fit_c(&c, fixture);
    let c_rss = format!("rss_f32={:.9e}", c.diagnostics().rss_f32);
    let c_rank = format!(
        "rank={} required_rank={} rank_threshold={:.9e}",
        c.diagnostics().rank,
        c.diagnostics().required_rank,
        c.diagnostics().rank_threshold
    );
    let requested = c.diagnostics().requested_workspace_bytes;
    drop(c);
    let config = FitConfig32::default();
    let mut c_timed = || {
        let features = black_box(&fixture.features32);
        let targets = black_box(&fixture.targets32);
        let selection = black_box(&fixture.selection);
        drop(black_box(
            fit_f32(features, targets, selection, black_box(&config)).expect("C fit succeeds"),
        ));
    };
    let mut c_memory = || {
        fit_f32(
            black_box(&fixture.features32),
            black_box(&fixture.targets32),
            black_box(&fixture.selection),
            black_box(&config),
        )
        .expect("C fit succeeds")
    };
    let c_measured = measure_windows(work, &mut c_timed);
    let c_probe = probe_memory(&mut c_memory);
    let c_id = row_identity("C", "fit", &key, None);
    report_fit(
        &c_id,
        &c_rank,
        requested,
        &c_rss,
        &c_measured,
        &c_probe,
        row_ids,
    );
}

fn build_group_cycle<T: Clone>(observations: &[T], batch: usize) -> Vec<Vec<T>> {
    (0..(PROBE_COUNT / batch))
        .map(|group| {
            (0..batch)
                .map(|index| observations[group * batch + index].clone())
                .collect()
        })
        .collect()
}

pub(super) fn run_single32(
    model: &LinearRegression,
    probes: &[Vector],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let mut index = 0;
    let mut timed = || {
        let probe = &probes[index % probes.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_checked(black_box(probe))
                .expect("single succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_checked(black_box(&probes[0]))
            .expect("single memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

pub(super) fn run_single64(
    model: &LinearRegression64,
    probes: &[Vector64],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let mut index = 0;
    let mut timed = || {
        let probe = &probes[index % probes.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_checked(black_box(probe))
                .expect("single succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_checked(black_box(&probes[0]))
            .expect("single memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

pub(super) fn run_stream32(
    model: &LinearRegression,
    observations: &[Observation<Vector>],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let checked = CheckedLinearRegression::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f32);
    let memory_checked = CheckedLinearRegression::new(model);
    let mut memory_executor = StreamingExecutor::new(&memory_checked, 0.0f32);
    let mut index = 0;
    let mut timed = || {
        let observation = &observations[index % observations.len()];
        index += 1;
        black_box(
            *executor
                .process_one(black_box(observation))
                .expect("stream succeeds"),
        );
    };
    let mut memory = || {
        *memory_executor
            .process_one(black_box(&observations[0]))
            .expect("stream memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

pub(super) fn run_stream64(
    model: &LinearRegression64,
    observations: &[Observation<Vector64>],
    label: &str,
    row_ids: &mut Vec<String>,
) {
    let checked = CheckedLinearRegression64::new(model);
    let mut executor = StreamingExecutor::new(&checked, 0.0f64);
    let memory_checked = CheckedLinearRegression64::new(model);
    let mut memory_executor = StreamingExecutor::new(&memory_checked, 0.0f64);
    let mut index = 0;
    let mut timed = || {
        let observation = &observations[index % observations.len()];
        index += 1;
        black_box(
            *executor
                .process_one(black_box(observation))
                .expect("stream succeeds"),
        );
    };
    let mut memory = || {
        *memory_executor
            .process_one(black_box(&observations[0]))
            .expect("stream memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(label, &measured, &memory, row_ids);
}

pub(super) fn run_groups32(
    model: &LinearRegression,
    observations: &[Observation<Vector>],
    batch: usize,
    fixture: &str,
    profile: &str,
    row_ids: &mut Vec<String>,
) {
    let groups = build_group_cycle(observations, batch);
    let mut index = 0;
    let mut timed = || {
        let group = &groups[index % groups.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_batch_checked(black_box(group))
                .expect("group succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_batch_checked(black_box(&groups[0]))
            .expect("group memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096 / batch,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(
        &row_identity(profile, "checked_group", fixture, Some(batch)),
        &measured,
        &memory,
        row_ids,
    );
}

pub(super) fn run_groups64(
    model: &LinearRegression64,
    observations: &[Observation<Vector64>],
    batch: usize,
    fixture: &str,
    profile: &str,
    row_ids: &mut Vec<String>,
) {
    let groups = build_group_cycle(observations, batch);
    let mut index = 0;
    let mut timed = || {
        let group = &groups[index % groups.len()];
        index += 1;
        black_box(
            black_box(model)
                .predict_batch_checked(black_box(group))
                .expect("group succeeds"),
        );
    };
    let mut memory = || {
        black_box(model)
            .predict_batch_checked(black_box(&groups[0]))
            .expect("group memory succeeds")
    };
    let measured = measure_windows(
        Work {
            calls: 4096 / batch,
            observations: 4096,
        },
        &mut timed,
    );
    let memory = probe_memory(&mut memory);
    report(
        &row_identity(profile, "checked_group", fixture, Some(batch)),
        &measured,
        &memory,
        row_ids,
    );
}

pub(super) fn run_conversion_rows(
    fixture: &Fixture,
    row_ids: &mut Vec<String>,
) -> (FitResult64, i64) {
    let key = fixture_key(fixture);
    let (a_result, a_heap) = retained_model_heap(|| fit_a(fixture));
    check_fit_a(&a_result, fixture);
    let source_before = source_bits64(&a_result);
    let b_check = a_result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion check succeeds");
    check_b_conversion(&a_result, &b_check, &fixture.probes32);
    drop(b_check);
    assert_eq!(source_bits64(&a_result), source_before);

    let mut conversion_timed = || {
        let source = black_box(&a_result);
        drop(black_box(
            source
                .to_serving_f32(black_box(UnderflowPolicy::Reject))
                .expect("B conversion succeeds"),
        ));
    };
    let mut conversion_memory = || {
        black_box(&a_result)
            .to_serving_f32(black_box(UnderflowPolicy::Reject))
            .expect("B conversion memory succeeds")
    };
    let conversion = measure_windows(
        Work {
            calls: 256,
            observations: 0,
        },
        &mut conversion_timed,
    );
    let conversion_probe = probe_memory(&mut conversion_memory);
    assert_eq!(source_bits64(&a_result), source_before);
    report(
        &row_identity("B", "conversion", &key, None),
        &conversion,
        &conversion_probe,
        row_ids,
    );
    (a_result, a_heap)
}

pub(super) fn run_serving_rows(
    fixture: &Fixture,
    a_result: FitResult64,
    a_heap: i64,
    row_ids: &mut Vec<String>,
) {
    let key = fixture_key(fixture);
    let source_before = source_bits64(&a_result);
    let (b_model, b_heap) = retained_model_heap(|| {
        a_result
            .to_serving_f32(UnderflowPolicy::Reject)
            .expect("B serving conversion succeeds")
    });
    check_b_conversion(&a_result, &b_model, &fixture.probes32);
    assert_eq!(source_bits64(&a_result), source_before);
    let a_model = a_result.into_predictor();
    let (c_result, c_heap) = retained_model_heap(|| fit_c(fixture));
    check_fit_c(&c_result, fixture);
    let c_model = c_result.into_predictor();
    let legacy_result = fit_legacy(fixture);
    check_fit_legacy(&legacy_result, fixture);
    let legacy_model = legacy_result.into_predictor();

    println!(
        "retained_serving_model_heap profile=A fixture={key} heap_add_B={a_heap} object_size_B={} logical_payload_B={}",
        std::mem::size_of_val(&a_model),
        std::mem::size_of_val(a_model.weights().as_slice())
    );
    println!(
        "retained_serving_model_heap profile=B fixture={key} heap_add_B={b_heap} object_size_B={} logical_payload_B={}",
        std::mem::size_of_val(&b_model),
        std::mem::size_of_val(b_model.weights().as_slice())
    );
    println!(
        "retained_serving_model_heap profile=C fixture={key} heap_add_B={c_heap} object_size_B={} logical_payload_B={}",
        std::mem::size_of_val(&c_model),
        std::mem::size_of_val(c_model.weights().as_slice())
    );

    let a_predictions: Vec<f64> = fixture
        .probes64
        .iter()
        .map(|probe| a_model.predict_checked(probe).expect("A diagnostic"))
        .collect();
    let b_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(b_model.predict_checked(probe).expect("B diagnostic")))
        .collect();
    let c_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(c_model.predict_checked(probe).expect("C diagnostic")))
        .collect();
    let legacy_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| {
            f64::from(
                legacy_model
                    .predict_checked(probe)
                    .expect("legacy diagnostic"),
            )
        })
        .collect();
    report_common_diagnostics("A", fixture, &a_predictions);
    report_common_diagnostics("B", fixture, &b_predictions);
    report_common_diagnostics("C", fixture, &c_predictions);
    report_common_diagnostics("legacy_hybrid", fixture, &legacy_predictions);
    report_cross_profile_differences(
        fixture,
        &[
            ("A", &a_predictions),
            ("B", &b_predictions),
            ("C", &c_predictions),
            ("legacy_hybrid", &legacy_predictions),
        ],
    );

    let aa = fixture
        .probes32
        .iter()
        .cloned()
        .map(Observation::new)
        .collect::<Vec<_>>();
    let aa64 = fixture
        .probes64
        .iter()
        .cloned()
        .map(Observation::new)
        .collect::<Vec<_>>();

    check_serving64(&a_model, &fixture.probes64);
    check_serving32(&b_model, &fixture.probes32);
    check_serving32(&c_model, &fixture.probes32);
    let a_before = model_bits64(&a_model);
    let b_before = model_bits32(&b_model);
    let c_before = model_bits32(&c_model);

    run_single64(
        &a_model,
        &fixture.probes64,
        &row_identity("A", "checked_single", &key, None),
        row_ids,
    );
    run_single32(
        &b_model,
        &fixture.probes32,
        &row_identity("B", "checked_single", &key, None),
        row_ids,
    );
    run_single32(
        &c_model,
        &fixture.probes32,
        &row_identity("C", "checked_single", &key, None),
        row_ids,
    );
    run_stream64(
        &a_model,
        &aa64,
        &row_identity("A", "checked_stream", &key, None),
        row_ids,
    );
    run_stream32(
        &b_model,
        &aa,
        &row_identity("B", "checked_stream", &key, None),
        row_ids,
    );
    run_stream32(
        &c_model,
        &aa,
        &row_identity("C", "checked_stream", &key, None),
        row_ids,
    );
    for batch in [1usize, 32, 128] {
        run_groups64(&a_model, &aa64, batch, &key, "A", row_ids);
        run_groups32(&b_model, &aa, batch, &key, "B", row_ids);
        run_groups32(&c_model, &aa, batch, &key, "C", row_ids);
    }
    assert_eq!(model_bits64(&a_model), a_before, "A model preserved");
    assert_eq!(model_bits32(&b_model), b_before, "B model preserved");
    assert_eq!(model_bits32(&c_model), c_before, "C model preserved");
}

pub(super) fn report_anchor_diagnostics(fixture: &Fixture) {
    let a_result = fit_a(fixture);
    check_fit_a(&a_result, fixture);
    let source_before = source_bits64(&a_result);
    let b_model = a_result
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("anchor B serving conversion");
    check_b_conversion(&a_result, &b_model, &fixture.probes32);
    assert_eq!(source_bits64(&a_result), source_before);
    let a_model = a_result.into_predictor();
    let c_result = fit_c(fixture);
    check_fit_c(&c_result, fixture);
    let c_model = c_result.into_predictor();
    let legacy_result = fit_legacy(fixture);
    check_fit_legacy(&legacy_result, fixture);
    let legacy_model = legacy_result.into_predictor();

    let a_predictions: Vec<f64> = fixture
        .probes64
        .iter()
        .map(|probe| a_model.predict_checked(probe).expect("anchor A diagnostic"))
        .collect();
    let b_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(b_model.predict_checked(probe).expect("anchor B diagnostic")))
        .collect();
    let c_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| f64::from(c_model.predict_checked(probe).expect("anchor C diagnostic")))
        .collect();
    let legacy_predictions: Vec<f64> = fixture
        .probes32
        .iter()
        .map(|probe| {
            f64::from(
                legacy_model
                    .predict_checked(probe)
                    .expect("anchor legacy diagnostic"),
            )
        })
        .collect();
    report_common_diagnostics("A", fixture, &a_predictions);
    report_common_diagnostics("B", fixture, &b_predictions);
    report_common_diagnostics("C", fixture, &c_predictions);
    report_common_diagnostics("legacy_hybrid", fixture, &legacy_predictions);
    report_cross_profile_differences(
        fixture,
        &[
            ("A", &a_predictions),
            ("B", &b_predictions),
            ("C", &c_predictions),
            ("legacy_hybrid", &legacy_predictions),
        ],
    );
}
