//! Ordinary least-squares (OLS) regression.
//!
//! The intended user-facing precision profiles are A (`fit_f64` and native-f64
//! serving), B (the same A fit converted to independently owned checked f32
//! serving), and C (`fit_f32` and native-f32 serving). OLS learning is an
//! explicit selected-row batch fit/refit, not an incremental update. Checked
//! single, grouped, and streaming prediction are separate delivery paths.
//!
//! The original mixed-precision [`fit`] path remains for compatibility. Its
//! [`FitResult`] alone exposes the legacy admission check and construction path
//! into [`TrainedModel`] and binary v1 [`artifact`] persistence. Typed A/B/C
//! results do not currently have that artifact construction path. See the local
//! `README.md` for the workflow, precision boundaries, and limitations.

pub mod artifact;
mod fit;
mod predict;

pub use artifact::{
    ArtifactError, ArtifactExpectation, ArtifactLimits, ArtifactMetadata, Cutoff, Expect,
    LoadedArtifact, ModelId, PublicationId, SchemaId, TrainedModel, TrainingMeta, TransformId,
};
pub use fit::{
    AcceptanceCheckError, AcceptanceCheckReport, AllocationSource, ConversionCause,
    DEFAULT_MAX_WORKSPACE_BYTES, FitConfig, FitConfig32, FitConfig64, FitDiagnostics,
    FitDiagnostics32, FitDiagnostics64, FitError, FitError32, FitError64, FitResult, FitResult32,
    FitResult64, SelectedField, ServingConversionError, ServingField, StoredDiagnostic,
    UnderflowPolicy, fit, fit_f32, fit_f64,
};
pub use predict::{
    CheckedLinearRegression, CheckedLinearRegression64, LinearRegression, LinearRegression64,
    RegressionError,
};
