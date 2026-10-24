//! Ordinary least-squares (OLS) regression.
//!
//! Supplied-parameter prediction ([`LinearRegression`]) and the batch `fit`
//! workflow are implemented. Fitted results can be turned into an immutable
//! [`TrainedModel`] carrying [`ArtifactMetadata`], and exported/imported as the
//! binary v1 [`artifact`] format.
//!
//! The deferred cross-precision numerical-certification project and the broader
//! fitting lifecycle remain follow-up work under
//! [#23](https://github.com/Skybrique/seqvex/issues/23). See `README.md` in this
//! directory for the algorithm contract.

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
