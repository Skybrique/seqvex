//! Untimed diagnostics entry point and shared diagnostic check helpers.

mod failure;
mod grouped;
mod operation_count;
mod retention;
mod stop;

pub(crate) use failure::run_failure_diagnostics;
