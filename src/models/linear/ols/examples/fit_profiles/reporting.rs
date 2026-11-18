use super::support::Metrics;

pub(super) fn report_a_fit(fit: &seqvex::models::linear::ols::FitResult64) {
    let diagnostics = fit.diagnostics();
    println!(
        "A learned: weights={:?}, intercept={:.15}, rank={}/{}, threshold={:.6e}, training_rss_f64={:.15}, requested_workspace_bytes={}",
        fit.weights().as_slice(),
        fit.intercept(),
        diagnostics.rank,
        diagnostics.required_rank,
        diagnostics.rank_threshold,
        diagnostics.rss_f64,
        diagnostics.requested_workspace_bytes,
    );
}

pub(super) fn report_c_fit(fit: &seqvex::models::linear::ols::FitResult32) {
    let diagnostics = fit.diagnostics();
    println!(
        "C learned: weights={:?}, intercept={:.9}, rank={}/{}, threshold={:.6e}, training_rss_f32={:.9}, requested_workspace_bytes={}",
        fit.weights().as_slice(),
        fit.intercept(),
        diagnostics.rank,
        diagnostics.required_rank,
        diagnostics.rank_threshold,
        diagnostics.rss_f32,
        diagnostics.requested_workspace_bytes,
    );
}

pub(super) fn report_metrics(name: &str, metrics: Metrics) {
    println!(
        "{name}: held_out_rss={:.12} held_out_mae={:.12} held_out_rmse={:.12}",
        metrics.rss, metrics.mae, metrics.rmse
    );
}
