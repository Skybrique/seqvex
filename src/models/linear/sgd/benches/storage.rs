//! Experimental SGD temporary/storage controls for the reviewed design plan.
//! Private to this benchmark; production learners are unchanged.

use common::comparison::capture_from_env;

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

#[path = "storage/candidates.rs"]
mod candidates;
#[path = "storage/fixtures.rs"]
mod fixtures;
#[path = "storage/measurement.rs"]
mod measurement;
#[path = "storage/metadata.rs"]
mod metadata;

use self::measurement::{guard_storage, run_storage_rows};
use self::metadata::{
    probe_catalog, timing_catalog, verify_actual_evidence, verify_detached_metadata_adversarial,
    verify_registrations,
};

const FIXTURE_LEN: usize = 2048;
const WARM_START: usize = 512;
const ETA32: f32 = 0.05;
const ETA64: f64 = 0.05;
const SEED32: u64 = 0x5eed_55d0_0000_0001;
const SEED64: u64 = 0x5eed_55d0_0000_0002;
const DIMS: [usize; 7] = [3, 6, 7, 8, 9, 32, 128];
const DETACHED_DIMS: [usize; 3] = [6, 8, 32];

fn main() {
    println!(
        "SGD temporary/storage comparison (benchmark-only); capture={}",
        capture_from_env().number()
    );
    println!("registered: 156 timing rows, 64 memory probes");
    println!("{}", common::env_summary());
    if cfg!(debug_assertions) {
        if common::comparison::run_failure_case_if_requested() {
            return;
        }
        verify_registrations();
        verify_detached_metadata_adversarial();
        common::comparison::comparison_self_check();
        guard_storage();
        println!("debug: storage guards complete; release rows skipped");
        return;
    }
    verify_registrations();
    guard_storage();
    let timing = timing_catalog();
    let probes = probe_catalog();
    let mut evidence = Vec::with_capacity(156);
    let mut probe_ids = Vec::with_capacity(64);
    run_storage_rows(&timing, &probes, &mut evidence, &mut probe_ids);
    verify_actual_evidence(&evidence, &probe_ids);
}
