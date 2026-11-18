//! Bounded CSV ingestion comparison for the reviewed component-design plan.
//!
//! Debug execution runs equivalence guards only. Release execution runs the
//! pre-registered rows; see the implementation-plan row registry.

use common::comparison::capture_from_env;

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

#[allow(dead_code, unused_imports)]
#[path = "../examples/fit_profiles/support.rs"]
mod fit_profiles_support;

#[path = "ingestion/baseline.rs"]
mod baseline;
#[path = "ingestion/candidates.rs"]
mod candidates;
#[path = "ingestion/fixtures.rs"]
mod fixtures;
#[path = "ingestion/measurement.rs"]
mod measurement;
#[path = "ingestion/reporting.rs"]
mod reporting;

fn main() {
    println!(
        "CSV ingestion comparison (benchmark-only); capture={}",
        capture_from_env().number()
    );
    println!("registered: 45 timing rows, 27 memory probes");
    println!("{}", common::env_summary());
    if cfg!(debug_assertions) {
        if common::comparison::run_failure_case_if_requested() {
            return;
        }
        reporting::verify_registrations();
        common::comparison::comparison_self_check();
        fixtures::guard_fixtures();
        println!("debug: ingestion guards complete; release rows skipped");
        return;
    }
    reporting::verify_registrations();
    fixtures::guard_fixtures();
    let timing = reporting::timing_catalog();
    let probes = reporting::probe_catalog();
    let mut evidence = Vec::with_capacity(45);
    let mut probe_ids = Vec::with_capacity(27);
    measurement::run_ingestion_rows(&timing, &probes, &mut evidence, &mut probe_ids);
    measurement::run_model_rows(&timing, &mut evidence);
    measurement::run_workflow_rows(&timing, &mut evidence);
    reporting::verify_actual_evidence(&evidence, &probe_ids);
}
