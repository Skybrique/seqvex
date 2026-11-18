use super::fixture::{Fixture, fixture_key};
use super::harness::{MEASURED_WINDOWS, Measured, MemoryProbe, quantile};

pub(super) fn row_identity(
    profile: &str,
    operation: &str,
    fixture: &str,
    batch: Option<usize>,
) -> String {
    match batch {
        Some(batch) => {
            format!("profile={profile} operation={operation} fixture={fixture} batch={batch}")
        }
        None => format!("profile={profile} operation={operation} fixture={fixture} batch=none"),
    }
}

pub(super) fn report(
    label: &str,
    measured: &Measured,
    memory: &MemoryProbe,
    row_ids: &mut Vec<String>,
) {
    row_ids.push(label.to_string());
    let raw_elapsed_ns: Vec<u64> = measured
        .windows
        .iter()
        .map(|window| window.elapsed_ns)
        .collect();
    let raw_allocs: Vec<u64> = measured
        .windows
        .iter()
        .map(|window| window.allocs)
        .collect();
    let raw_bytes: Vec<u64> = measured.windows.iter().map(|window| window.bytes).collect();
    let mut call_values: Vec<f64> = raw_elapsed_ns
        .iter()
        .map(|elapsed| *elapsed as f64 / measured.work.calls as f64)
        .collect();
    let obs_values: Vec<f64> = if measured.work.observations == 0 {
        Vec::new()
    } else {
        raw_elapsed_ns
            .iter()
            .map(|elapsed| *elapsed as f64 / measured.work.observations as f64)
            .collect()
    };
    call_values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let obs_summary = if obs_values.is_empty() {
        "n/a".to_string()
    } else {
        let mut values = obs_values.clone();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        format!(
            "median={:.1} min={:.1} max={:.1} iqr={:.1}",
            quantile(&values, 0.5),
            values[0],
            values[values.len() - 1],
            quantile(&values, 0.75) - quantile(&values, 0.25)
        )
    };
    let allocs: u64 = measured.windows.iter().map(|window| window.allocs).sum();
    let bytes: u64 = measured.windows.iter().map(|window| window.bytes).sum();
    let total_calls = (measured.work.calls * MEASURED_WINDOWS) as f64;
    println!(
        "{label}: calls/window={} observations/window={} raw_elapsed_ns={raw_elapsed_ns:?} \
         ns/call median={:.1} min={:.1} max={:.1} iqr={:.1} ns/observation {obs_summary} \
         allocs/call={:.4} bytes/call={:.1} raw_allocs={raw_allocs:?} raw_bytes={raw_bytes:?} \
         peak_add_B={} retained_B={} after_drop_B={} mem_allocs={} mem_bytes={}",
        measured.work.calls,
        measured.work.observations,
        quantile(&call_values, 0.5),
        call_values[0],
        call_values[call_values.len() - 1],
        quantile(&call_values, 0.75) - quantile(&call_values, 0.25),
        allocs as f64 / total_calls,
        bytes as f64 / total_calls,
        memory.peak_add,
        memory.retained,
        memory.after_drop,
        memory.allocs,
        memory.bytes,
    );
}

pub(super) fn report_fit(
    label: &str,
    rank_record: &str,
    requested: usize,
    native_rss: &str,
    measured: &Measured,
    memory: &MemoryProbe,
    row_ids: &mut Vec<String>,
) {
    println!(
        "{label}: metadata_source=untimed_guard_fit_same_fixture_config {rank_record} requested_workspace_B={requested} native_{native_rss}"
    );
    report(label, measured, memory, row_ids);
}

pub(super) fn report_common_diagnostics(profile: &str, fixture: &Fixture, predictions: &[f64]) {
    let targets = &fixture.all_targets[fixture.n..];
    assert_eq!(predictions.len(), targets.len());
    let mut rss = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&prediction, &target) in predictions.iter().zip(targets) {
        let target = f64::from(target);
        let residual = prediction - target;
        rss += residual * residual;
        max_abs = max_abs.max(residual.abs());
    }
    let rmse = (rss / predictions.len() as f64).sqrt();
    println!(
        "common_f64_diagnostics profile={profile} fixture={} probes={} rss_f64={:.17e} rmse_f64={:.17e} max_abs_f64={:.17e}",
        fixture_key(fixture),
        predictions.len(),
        rss,
        rmse,
        max_abs
    );
}

pub(super) fn report_cross_profile_differences(fixture: &Fixture, profiles: &[(&str, &[f64])]) {
    for left in 0..profiles.len() {
        for right in (left + 1)..profiles.len() {
            let (left_name, left_values) = profiles[left];
            let (right_name, right_values) = profiles[right];
            assert_eq!(left_values.len(), right_values.len());
            let mut squared = 0.0f64;
            let mut max_abs = 0.0f64;
            for (&left_value, &right_value) in left_values.iter().zip(right_values) {
                let difference = left_value - right_value;
                squared += difference * difference;
                max_abs = max_abs.max(difference.abs());
            }
            let rmse = (squared / left_values.len() as f64).sqrt();
            println!(
                "cross_profile_prediction_difference fixture={} pair={left_name},{right_name} probes={} rmse_f64={:.17e} max_abs_f64={:.17e}",
                fixture_key(fixture),
                left_values.len(),
                rmse,
                max_abs
            );
        }
    }
}
