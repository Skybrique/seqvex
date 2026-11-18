//! RLS behavior tests.
//!
//! Two references are used, for two different purposes:
//!
//! - `ref_step` is a scalar `Vec<f32>` implementation of the production
//!   recurrence, accumulating in the same order as the library. Every state
//!   comparison against it is bitwise, because both paths use the same
//!   operations in the same order.
//! - the literal-objective oracle (`objective_normal_equations` plus an `f64`
//!   direct solve) evaluates the final weighted least-squares objective without
//!   using the information-matrix recurrence. Comparisons against it are
//!   tolerance-based, because the production `f32` recursion and the `f64`
//!   direct solve differ in precision and reduction order.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::error::FailureCause;
use seqvex::foundation::numerical::{Matrix, Matrix64, Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{
    StateModel, process_batch, process_one, process_stream, process_stream_continuing,
};
use seqvex::models::linear::rls::{
    Rls, Rls64, RlsBatchError, RlsBatchError64, RlsError, RlsError64, RlsSample, RlsSample64,
    RlsServing32, RlsServingConversionError, RlsState, RlsState64, RlsStateAdmissionError,
};

#[path = "rls/attribution.rs"]
mod attribution;
#[path = "rls/batch.rs"]
mod batch;
#[path = "rls/excitation.rs"]
mod excitation;
#[path = "rls/reference.rs"]
mod reference;
#[path = "rls/regime.rs"]
mod regime;
#[path = "rls/scaling.rs"]
mod scaling;
#[path = "rls/support.rs"]
mod support;
#[path = "rls/validation.rs"]
mod validation;

#[path = "rls/precision/mod.rs"]
mod precision;
