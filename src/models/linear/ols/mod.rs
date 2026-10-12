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
    AllocationSource, ConversionCause, DEFAULT_MAX_WORKSPACE_BYTES, FitConfig, FitDiagnostics,
    FitError, FitResult, UnderflowPolicy, fit,
};
pub use predict::{CheckedLinearRegression, LinearRegression, RegressionError};
