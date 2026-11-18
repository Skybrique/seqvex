//! OLS profile-specific performance/resource evidence (benchmark-only).
//!
//! Debug builds run bounded correctness guards only. Release execution emits a
//! fixed 41-row matrix over native-f64 A, strict-f32 C, explicit B conversion,
//! A/B/C checked serving, and matched legacy hybrid fitting. Window summaries
//! are not individual-request tails. No production code or optimization is
//! changed by this target.

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

#[path = "profiles/fixture.rs"]
mod fixture;
#[path = "profiles/guards.rs"]
mod guards;
#[path = "profiles/harness.rs"]
mod harness;
#[path = "profiles/reporting.rs"]
mod reporting;
#[path = "profiles/rows.rs"]
mod rows;

fn main() {
    if cfg!(debug_assertions) {
        println!("ols_profiles: bounded correctness guards only; no performance evidence");
        guards::verify_profiles();
        println!("ols_profiles: correctness guards passed; no measurement windows run");
        return;
    }
    harness::run_release_matrix();
}
