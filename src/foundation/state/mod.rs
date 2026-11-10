//! State, transitions, execution progress, and failure-preserving execution.

pub mod atomicity;
pub mod transition;

pub use crate::foundation::error::{
    BatchFailure, ExecutorStreamFailure, FailureCause, StreamFailure, StreamProgress,
};
pub use atomicity::{StreamSummary, process_batch, process_stream, process_stream_continuing};
pub use transition::{StateModel, process_one};
