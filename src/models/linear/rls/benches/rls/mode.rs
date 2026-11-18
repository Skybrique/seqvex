//! Benchmark mode resolution, debug/release identity and global failure flags.
//!
//! Owns the single process-wide unexpected-failure and legacy-validation flags.
//! Child modules call these helpers; no module defines a second copy.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::common;
use crate::common::comparison::capture_from_env;
use crate::{DEBUG_HORIZON, DEBUG_WARMUP, HORIZON, WARMUP};

static UNEXPECTED_FAILURE: AtomicBool = AtomicBool::new(false);

pub(crate) static LEGACY_VALIDATE_ONLY: AtomicBool = AtomicBool::new(false);

pub(crate) fn legacy_validating() -> bool {
    LEGACY_VALIDATE_ONLY.load(Ordering::Relaxed)
}

pub(crate) fn set_unexpected_failure() {
    UNEXPECTED_FAILURE.store(true, Ordering::Relaxed);
}

pub(crate) fn has_unexpected_failure() -> bool {
    UNEXPECTED_FAILURE.load(Ordering::Relaxed)
}

pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    DebugValidate,
    ReleaseValidate,
    LegacyEvidence,
    MeasureUpdateOnly,
    MeasureDefault,
}

impl Mode {
    pub(crate) fn allows_timing(self) -> bool {
        matches!(
            self,
            Self::MeasureUpdateOnly | Self::MeasureDefault | Self::LegacyEvidence
        )
    }
}

pub(crate) fn print_run_identity(mode: Mode) {
    let capture = capture_from_env();
    println!(
        "RUN_IDENTITY mode={mode:?} capture={} update_only={} validate_only={} evidence_only={} legacy_diagnostics={} debug={} profile={} env={}",
        capture.number(),
        env_flag("SEQVEX_RLS_UPDATE_ONLY"),
        env_flag("SEQVEX_RLS_VALIDATE_ONLY"),
        env_flag("SEQVEX_EVIDENCE_ONLY"),
        env_flag("SEQVEX_LEGACY_DIAGNOSTICS"),
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        common::env_summary()
    );
}

pub(crate) fn resolve_mode() -> Mode {
    if cfg!(debug_assertions) {
        Mode::DebugValidate
    } else if env_flag("SEQVEX_RLS_VALIDATE_ONLY") {
        Mode::ReleaseValidate
    } else if env_flag("SEQVEX_EVIDENCE_ONLY") {
        Mode::LegacyEvidence
    } else if env_flag("SEQVEX_RLS_UPDATE_ONLY") {
        Mode::MeasureUpdateOnly
    } else {
        Mode::MeasureDefault
    }
}

pub(crate) fn warmup() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_WARMUP
    } else {
        WARMUP
    }
}

pub(crate) fn horizon() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_HORIZON
    } else {
        HORIZON
    }
}
