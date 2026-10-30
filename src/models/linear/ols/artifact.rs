//! Immutable trained-model metadata and the binary artifact v1 format.
//!
//! A [`TrainedModel`] pairs an immutable predictor with [`ArtifactMetadata`]
//! derived from the fit. `ModelId` is a content identity over the parameters
//! and training linkage; it **excludes** publication identity. Publication is
//! supplied to [`TrainedModel::export_to`] and returned by
//! [`TrainedModel::import_from`]; it is never retained as mutable state.
//!
//! The artifact format is little-endian binary v1. See the module constants and
//! [`TrainedModel::export_to`] for the exact field order. Import validates a
//! complete candidate or returns an error; it never replaces an active model.
//! Public fitted-model construction currently belongs to the legacy `FitResult`
//! path; typed A/B/C results do not have a v1 construction or persistence path.
//! An import reader must contain exactly one complete artifact through EOF:
//! import consumes it, rejects trailing bytes, and provides no reader-position
//! or retry guarantee. The byte limit bounds accepted artifact size, not total
//! memory use.
//!
//! The library provides **no** filesystem atomicity or durability.

use super::predict::{LinearRegression, RegressionError};
use crate::foundation::numerical::Vector;

use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{Read, Write};

/// 16-byte caller-supplied schema identifier (linkage only).
pub type SchemaId = [u8; 16];
/// 16-byte caller-supplied transform identifier (linkage only).
pub type TransformId = [u8; 16];
/// 16-byte application-owned publication identifier (excluded from identity).
pub type PublicationId = [u8; 16];
/// 32-byte content identity.
pub type ModelId = [u8; 32];

const MAGIC: [u8; 5] = *b"SXOLS";
const FORMAT_VERSION: u16 = 1;

const MODEL_ID_DOMAIN: &[u8] = b"sx.ols.model.v1\0";
const PARAM_HASH_DOMAIN: &[u8] = b"sx.ols.params.v1\0";
const ARTIFACT_DOMAIN: &[u8] = b"sx.ols.artifact.v1\0";

const FLAG_INTERCEPT: u8 = 1 << 0;
const FLAG_TRANSFORM: u8 = 1 << 1;
const FLAG_CUTOFF: u8 = 1 << 2;
const FLAG_PUBLICATION: u8 = 1 << 3;
const KNOWN_FLAGS: u8 = FLAG_INTERCEPT | FLAG_TRANSFORM | FLAG_CUTOFF | FLAG_PUBLICATION;

/// A caller-supplied cutoff marker: an application-defined kind and value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cutoff {
    /// Application-defined kind.
    pub kind: u8,
    /// Application-defined value.
    pub value: u64,
}

/// Caller-supplied **actual** training linkage, distinct from import
/// expectations. Identifiers are linkage only; they do not prove correctness.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingMeta {
    /// Schema identifier.
    pub schema_id: SchemaId,
    /// Optional transform identifier.
    pub transform_link: Option<TransformId>,
    /// Optional cutoff marker.
    pub cutoff: Option<Cutoff>,
    /// Opaque 8-byte fit revision.
    pub fit_revision: [u8; 8],
}

/// Metadata derived from an actual fit. `intercept` records explicit presence,
/// independent of whether the bias happens to be zero.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactMetadata {
    /// Schema identifier (copied from [`TrainingMeta`]).
    pub schema_id: SchemaId,
    /// Optional transform identifier.
    pub transform_link: Option<TransformId>,
    /// Optional cutoff marker.
    pub cutoff: Option<Cutoff>,
    /// Opaque fit revision.
    pub fit_revision: [u8; 8],
    /// Training row count (checked `u32`).
    pub rows: u32,
    /// Feature count / `d` (checked `u32`).
    pub features: u32,
    /// Whether an intercept column was fitted (presence, not value).
    pub intercept: bool,
    /// Computed content identity.
    pub model_id: ModelId,
    /// Hash over parameters only (coefficients plus optional intercept).
    pub param_hash: [u8; 32],
}

/// Import-time compatibility expectation for a single field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expect<T> {
    /// Accept any value.
    Any,
    /// Require exactly this value.
    Required(T),
    /// Require the field to be absent.
    Absent,
}

/// Import-time compatibility checks. Missing expectations default to
/// [`Expect::Any`]; use [`ArtifactExpectation::any`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactExpectation {
    /// Expected feature count.
    pub feature_count: Expect<u32>,
    /// Expected intercept presence.
    pub intercept: Expect<bool>,
    /// Expected schema identifier.
    pub schema_id: Expect<SchemaId>,
    /// Expected transform identifier (`Absent` means no transform).
    pub transform_link: Expect<TransformId>,
    /// Expected cutoff marker (`Absent` means no cutoff).
    pub cutoff: Expect<Cutoff>,
}

impl ArtifactExpectation {
    /// An expectation that accepts any valid artifact.
    #[must_use]
    pub fn any() -> Self {
        Self {
            feature_count: Expect::Any,
            intercept: Expect::Any,
            schema_id: Expect::Any,
            transform_link: Expect::Any,
            cutoff: Expect::Any,
        }
    }
}

/// Import resource limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactLimits {
    /// Maximum accepted feature count `d`.
    pub max_features: usize,
    /// Maximum accepted total artifact bytes.
    pub max_bytes: usize,
}

/// Artifact construction, export and import failures.
#[derive(Debug)]
pub enum ArtifactError {
    /// Underlying reader/writer failure.
    Io {
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Input ended before a required field was read.
    Truncated {
        /// Bytes required for the field.
        needed: usize,
        /// Bytes remaining.
        available: usize,
    },
    /// Stored checksum does not match the recomputed checksum.
    ChecksumMismatch {
        /// Stored checksum.
        expected: [u8; 32],
        /// Recomputed checksum.
        actual: [u8; 32],
    },
    /// The format version is not supported.
    UnsupportedVersion {
        /// Version found in the artifact.
        found: u16,
        /// Supported version.
        supported: u16,
    },
    /// Schema identifier is inconsistent or unsupported.
    SchemaMismatch,
    /// Feature dimension is inconsistent (for example `d != features`, or
    /// zero features).
    ShapeMismatch,
    /// A required field does not match the expectation.
    FieldMismatch {
        /// Field name.
        field: &'static str,
    },
    /// A field that should be absent is present.
    UnexpectedField {
        /// Field name.
        field: &'static str,
    },
    /// A parameter is not finite.
    NonFiniteParameter {
        /// Parameter index (intercept uses index `d`).
        index: usize,
    },
    /// Recomputed model identity does not match the stored identity.
    ModelIdMismatch {
        /// Recomputed identity.
        expected: [u8; 32],
        /// Stored identity.
        actual: [u8; 32],
    },
    /// A resource limit was exceeded.
    Capacity {
        /// The limit that was exceeded.
        limit: usize,
    },
    /// Unexpected trailing bytes after the artifact.
    TrailingBytes {
        /// Number of trailing bytes.
        count: usize,
    },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { .. } => f.write_str("artifact I/O failed"),
            Self::Truncated { needed, available } => write!(
                f,
                "artifact truncated: needed {needed} bytes, {available} available"
            ),
            Self::ChecksumMismatch { expected, actual } => {
                f.write_str("artifact checksum mismatch: expected ")?;
                write_hex(f, expected)?;
                f.write_str(", actual ")?;
                write_hex(f, actual)
            }
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "unsupported artifact version {found}; supported version is {supported}"
            ),
            Self::SchemaMismatch => {
                f.write_str("artifact schema identifier is inconsistent or unsupported")
            }
            Self::ShapeMismatch => f.write_str("artifact shape is inconsistent"),
            Self::FieldMismatch { field } => {
                write!(f, "artifact field {field} does not match expectation")
            }
            Self::UnexpectedField { field } => {
                write!(f, "unexpected artifact field {field}; it should be absent")
            }
            Self::NonFiniteParameter { index } => {
                write!(f, "artifact parameter at index {index} is not finite")
            }
            Self::ModelIdMismatch { expected, actual } => {
                f.write_str("artifact model identity mismatch: expected ")?;
                write_hex(f, expected)?;
                f.write_str(", actual ")?;
                write_hex(f, actual)
            }
            Self::Capacity { limit } => {
                write!(f, "artifact capacity limit {limit} exceeded")
            }
            Self::TrailingBytes { count } => {
                write!(f, "artifact contains {count} trailing bytes")
            }
        }
    }
}

fn write_hex(formatter: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    for byte in bytes {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}

impl std::error::Error for ArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ArtifactError::Io { source } => Some(source),
            _ => None,
        }
    }
}

/// An immutable trained model: predictor plus derived metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainedModel {
    predictor: LinearRegression,
    metadata: ArtifactMetadata,
}

/// A successfully imported model plus any publication identity from the file.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedArtifact {
    /// The imported model.
    pub model: TrainedModel,
    /// Publication identity carried by the artifact, if any.
    pub publication: Option<PublicationId>,
}

impl TrainedModel {
    /// The immutable predictor.
    #[must_use]
    pub fn predictor(&self) -> &LinearRegression {
        &self.predictor
    }

    /// The derived metadata.
    #[must_use]
    pub fn metadata(&self) -> &ArtifactMetadata {
        &self.metadata
    }

    /// The content identity.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.metadata.model_id
    }

    /// Serializes the model as a binary v1 artifact.
    ///
    /// `publication` is written to the artifact only; it is not retained on
    /// `self` and does not affect the model identity. On a writer failure the
    /// stream may be left partial; the library provides no atomicity/durability.
    pub fn export_to<W: Write>(
        &self,
        writer: &mut W,
        limits: ArtifactLimits,
        publication: Option<PublicationId>,
    ) -> Result<(), ArtifactError> {
        let metadata = &self.metadata;
        let weights = self.predictor.weights().as_slice();
        let d = usize::try_from(metadata.features).map_err(|_| ArtifactError::ShapeMismatch)?;
        if d != weights.len() {
            return Err(ArtifactError::ShapeMismatch);
        }
        if d > limits.max_features {
            return Err(ArtifactError::Capacity {
                limit: limits.max_features,
            });
        }

        let coefficient_bytes = d.checked_mul(4).ok_or(ArtifactError::Capacity {
            limit: limits.max_bytes,
        })?;
        let optional = usize::from(metadata.transform_link.is_some()) * 16
            + usize::from(metadata.cutoff.is_some()) * 9
            + usize::from(metadata.intercept) * 4
            + usize::from(publication.is_some()) * 16;
        let total = 5 + 2 + 1 + 4 + 16 + 8 + 4 + 4 + 32 + 32 + coefficient_bytes + optional;
        if total > limits.max_bytes {
            return Err(ArtifactError::Capacity {
                limit: limits.max_bytes,
            });
        }

        let mut body: Vec<u8> = Vec::with_capacity(total - 32);
        body.extend_from_slice(&MAGIC);
        body.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        let mut flags = 0u8;
        if metadata.intercept {
            flags |= FLAG_INTERCEPT;
        }
        if metadata.transform_link.is_some() {
            flags |= FLAG_TRANSFORM;
        }
        if metadata.cutoff.is_some() {
            flags |= FLAG_CUTOFF;
        }
        if publication.is_some() {
            flags |= FLAG_PUBLICATION;
        }
        body.push(flags);
        body.extend_from_slice(&metadata.features.to_le_bytes());
        body.extend_from_slice(&metadata.schema_id);
        if let Some(transform) = metadata.transform_link {
            body.extend_from_slice(&transform);
        }
        if let Some(cutoff) = metadata.cutoff {
            body.push(cutoff.kind);
            body.extend_from_slice(&cutoff.value.to_le_bytes());
        }
        body.extend_from_slice(&metadata.fit_revision);
        body.extend_from_slice(&metadata.rows.to_le_bytes());
        body.extend_from_slice(&metadata.features.to_le_bytes());
        if let Some(publication) = publication {
            body.extend_from_slice(&publication);
        }
        for &weight in weights {
            body.extend_from_slice(&weight.to_le_bytes());
        }
        if metadata.intercept {
            body.extend_from_slice(&self.predictor.bias().to_le_bytes());
        }
        body.extend_from_slice(&metadata.model_id);

        let checksum = artifact_checksum(&body);
        writer
            .write_all(&body)
            .map_err(|source| ArtifactError::Io { source })?;
        writer
            .write_all(&checksum)
            .map_err(|source| ArtifactError::Io { source })?;
        Ok(())
    }

    /// Reads and validates a complete binary v1 artifact.
    pub fn import_from<R: Read>(
        reader: &mut R,
        expected: &ArtifactExpectation,
        limits: ArtifactLimits,
    ) -> Result<LoadedArtifact, ArtifactError> {
        let bytes = read_bounded(reader, limits.max_bytes)?;
        let mut cursor = Cursor::new(&bytes);

        if cursor.take(5)? != MAGIC.as_slice() {
            return Err(ArtifactError::FieldMismatch { field: "magic" });
        }
        let version = u16::from_le_bytes(cursor.array::<2>()?);
        if version != FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let flags = cursor.byte()?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(ArtifactError::UnexpectedField {
                field: "file_flags",
            });
        }
        let dimensions = u32::from_le_bytes(cursor.array::<4>()?);
        let d = usize::try_from(dimensions).map_err(|_| ArtifactError::ShapeMismatch)?;
        if d == 0 {
            return Err(ArtifactError::ShapeMismatch);
        }
        if d > limits.max_features {
            return Err(ArtifactError::Capacity {
                limit: limits.max_features,
            });
        }
        let schema_id = cursor.array::<16>()?;
        let transform_link = if flags & FLAG_TRANSFORM != 0 {
            Some(cursor.array::<16>()?)
        } else {
            None
        };
        let cutoff = if flags & FLAG_CUTOFF != 0 {
            let kind = cursor.byte()?;
            let value = u64::from_le_bytes(cursor.array::<8>()?);
            Some(Cutoff { kind, value })
        } else {
            None
        };
        let fit_revision = cursor.array::<8>()?;
        let rows = u32::from_le_bytes(cursor.array::<4>()?);
        let features = u32::from_le_bytes(cursor.array::<4>()?);
        let publication = if flags & FLAG_PUBLICATION != 0 {
            Some(cursor.array::<16>()?)
        } else {
            None
        };
        if usize::try_from(features).map_err(|_| ArtifactError::ShapeMismatch)? != d {
            return Err(ArtifactError::ShapeMismatch);
        }

        // Validate the implied payload before any dimension-driven allocation.
        // The payload is the coefficient block plus the optional intercept and
        // the mandatory model_id + checksum trailers. A declared size beyond the
        // configured limit is `Capacity`; an input too short to hold an otherwise
        // admissible payload is `Truncated`.
        let coefficient_bytes = d.checked_mul(4).ok_or(ArtifactError::Capacity {
            limit: limits.max_bytes,
        })?;
        let intercept_bytes = if flags & FLAG_INTERCEPT != 0 { 4 } else { 0 };
        let payload_bytes = coefficient_bytes
            .checked_add(intercept_bytes)
            .and_then(|value| value.checked_add(64))
            .ok_or(ArtifactError::Capacity {
                limit: limits.max_bytes,
            })?;
        let implied_total =
            cursor
                .position()
                .checked_add(payload_bytes)
                .ok_or(ArtifactError::Capacity {
                    limit: limits.max_bytes,
                })?;
        if implied_total > limits.max_bytes {
            return Err(ArtifactError::Capacity {
                limit: limits.max_bytes,
            });
        }
        if payload_bytes > cursor.remaining() {
            return Err(ArtifactError::Truncated {
                needed: payload_bytes,
                available: cursor.remaining(),
            });
        }

        let mut weights: Vec<f32> = Vec::new();
        weights
            .try_reserve_exact(d)
            .map_err(|_| ArtifactError::Capacity {
                limit: limits.max_bytes,
            })?;
        for index in 0..d {
            let value = f32::from_le_bytes(cursor.array::<4>()?);
            if !value.is_finite() {
                return Err(ArtifactError::NonFiniteParameter { index });
            }
            weights.push(value);
        }
        let intercept_present = flags & FLAG_INTERCEPT != 0;
        let intercept = if intercept_present {
            let value = f32::from_le_bytes(cursor.array::<4>()?);
            if !value.is_finite() {
                return Err(ArtifactError::NonFiniteParameter { index: d });
            }
            value
        } else {
            0.0
        };
        let stored_model_id = cursor.array::<32>()?;
        let body_len = cursor.position();
        let stored_checksum = cursor.array::<32>()?;

        if cursor.remaining() > 0 {
            return Err(ArtifactError::TrailingBytes {
                count: cursor.remaining(),
            });
        }

        let recomputed_checksum = artifact_checksum(&bytes[..body_len]);
        if stored_checksum != recomputed_checksum {
            return Err(ArtifactError::ChecksumMismatch {
                expected: stored_checksum,
                actual: recomputed_checksum,
            });
        }

        let model_id = model_identity(
            &schema_id,
            transform_link.as_ref(),
            cutoff,
            &fit_revision,
            rows,
            features,
            intercept_present,
            &weights,
            intercept,
        );
        if stored_model_id != model_id {
            return Err(ArtifactError::ModelIdMismatch {
                expected: model_id,
                actual: stored_model_id,
            });
        }

        check_expectations(
            expected,
            features,
            intercept_present,
            &schema_id,
            transform_link.as_ref(),
            cutoff,
        )?;

        let param_hash = parameter_hash(&weights, intercept_present, intercept);
        let predictor = LinearRegression::new(Vector::from_slice(&weights), intercept)
            .map_err(|_| ArtifactError::ShapeMismatch)?;
        Ok(LoadedArtifact {
            model: TrainedModel {
                predictor,
                metadata: ArtifactMetadata {
                    schema_id,
                    transform_link,
                    cutoff,
                    fit_revision,
                    rows,
                    features,
                    intercept: intercept_present,
                    model_id,
                    param_hash,
                },
            },
            publication,
        })
    }
}

/// Builds a [`TrainedModel`] from fitted parameters and caller metadata.
///
/// Called by `FitResult::into_trained`; parameters were validated by `fit`.
pub(crate) fn build_trained(
    meta: &TrainingMeta,
    coefficients: Vector,
    bias: f32,
    rows: u32,
    features: u32,
    intercept_present: bool,
) -> Result<TrainedModel, ArtifactError> {
    let weights = coefficients.as_slice();
    if weights.len() != usize::try_from(features).map_err(|_| ArtifactError::ShapeMismatch)? {
        return Err(ArtifactError::ShapeMismatch);
    }
    let param_hash = parameter_hash(weights, intercept_present, bias);
    let model_id = model_identity(
        &meta.schema_id,
        meta.transform_link.as_ref(),
        meta.cutoff,
        &meta.fit_revision,
        rows,
        features,
        intercept_present,
        weights,
        bias,
    );
    let predictor = LinearRegression::new(coefficients, bias)
        .map_err(|_: RegressionError| ArtifactError::ShapeMismatch)?;
    Ok(TrainedModel {
        predictor,
        metadata: ArtifactMetadata {
            schema_id: meta.schema_id,
            transform_link: meta.transform_link,
            cutoff: meta.cutoff,
            fit_revision: meta.fit_revision,
            rows,
            features,
            intercept: intercept_present,
            model_id,
            param_hash,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn model_identity(
    schema_id: &SchemaId,
    transform_link: Option<&TransformId>,
    cutoff: Option<Cutoff>,
    fit_revision: &[u8; 8],
    rows: u32,
    features: u32,
    intercept_present: bool,
    weights: &[f32],
    intercept: f32,
) -> ModelId {
    let mut hasher = Sha256::new();
    hasher.update(MODEL_ID_DOMAIN);
    hasher.update(FORMAT_VERSION.to_le_bytes());
    let mut flags = 0u8;
    if intercept_present {
        flags |= FLAG_INTERCEPT;
    }
    if transform_link.is_some() {
        flags |= FLAG_TRANSFORM;
    }
    if cutoff.is_some() {
        flags |= FLAG_CUTOFF;
    }
    hasher.update([flags]);
    hasher.update(features.to_le_bytes());
    hasher.update(schema_id);
    if let Some(transform) = transform_link {
        hasher.update(transform);
    }
    if let Some(cutoff) = cutoff {
        hasher.update([cutoff.kind]);
        hasher.update(cutoff.value.to_le_bytes());
    }
    hasher.update(fit_revision);
    hasher.update(rows.to_le_bytes());
    hasher.update(features.to_le_bytes());
    for weight in weights {
        hasher.update(weight.to_le_bytes());
    }
    if intercept_present {
        hasher.update(intercept.to_le_bytes());
    }
    finalize32(hasher)
}

/// Parameters only: coefficients and the optional intercept, with a presence
/// flag so an absent intercept is distinct from a present zero intercept.
fn parameter_hash(weights: &[f32], intercept_present: bool, intercept: f32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PARAM_HASH_DOMAIN);
    hasher.update((weights.len() as u32).to_le_bytes());
    for weight in weights {
        hasher.update(weight.to_le_bytes());
    }
    hasher.update([u8::from(intercept_present)]);
    if intercept_present {
        hasher.update(intercept.to_le_bytes());
    }
    finalize32(hasher)
}

fn artifact_checksum(body: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(ARTIFACT_DOMAIN);
    hasher.update(body);
    finalize32(hasher)
}

fn finalize32(hasher: Sha256) -> [u8; 32] {
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn check_expectations(
    expected: &ArtifactExpectation,
    features: u32,
    intercept_present: bool,
    schema_id: &SchemaId,
    transform_link: Option<&TransformId>,
    cutoff: Option<Cutoff>,
) -> Result<(), ArtifactError> {
    match expected.feature_count {
        Expect::Any => {}
        Expect::Required(value) if value == features => {}
        Expect::Required(_) => {
            return Err(ArtifactError::FieldMismatch { field: "features" });
        }
        Expect::Absent => {
            return Err(ArtifactError::UnexpectedField { field: "features" });
        }
    }
    match expected.intercept {
        Expect::Any => {}
        Expect::Required(value) if value == intercept_present => {}
        Expect::Required(_) => {
            return Err(ArtifactError::FieldMismatch { field: "intercept" });
        }
        Expect::Absent if !intercept_present => {}
        Expect::Absent => {
            return Err(ArtifactError::UnexpectedField { field: "intercept" });
        }
    }
    match &expected.schema_id {
        Expect::Any => {}
        Expect::Required(value) if value == schema_id => {}
        Expect::Required(_) => {
            return Err(ArtifactError::FieldMismatch { field: "schema_id" });
        }
        Expect::Absent => {
            return Err(ArtifactError::UnexpectedField { field: "schema_id" });
        }
    }
    match &expected.transform_link {
        Expect::Any => {}
        Expect::Required(value) if transform_link == Some(value) => {}
        Expect::Required(_) => {
            return Err(ArtifactError::FieldMismatch { field: "transform" });
        }
        Expect::Absent if transform_link.is_none() => {}
        Expect::Absent => {
            return Err(ArtifactError::UnexpectedField { field: "transform" });
        }
    }
    match &expected.cutoff {
        Expect::Any => {}
        Expect::Required(value) if cutoff == Some(*value) => {}
        Expect::Required(_) => {
            return Err(ArtifactError::FieldMismatch { field: "cutoff" });
        }
        Expect::Absent if cutoff.is_none() => {}
        Expect::Absent => {
            return Err(ArtifactError::UnexpectedField { field: "cutoff" });
        }
    }
    Ok(())
}

/// Retains at most `max_bytes + 1` bytes while detecting an over-limit stream.
/// Each `Read` may consume a full 4096-byte chunk before the limit is checked,
/// so the reader can advance beyond the first over-limit byte. Import provides
/// no reader-position or retry guarantee; no allocation is driven by an
/// unchecked encoded length.
fn read_bounded<R: Read>(reader: &mut R, max_bytes: usize) -> Result<Vec<u8>, ArtifactError> {
    let ceiling = max_bytes.saturating_add(1);
    let mut bytes: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = reader
            .read(&mut chunk)
            .map_err(|source| ArtifactError::Io { source })?;
        if read == 0 {
            break;
        }
        if bytes.len() + read > ceiling {
            return Err(ArtifactError::Capacity { limit: max_bytes });
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    if bytes.len() > max_bytes {
        return Err(ArtifactError::Capacity { limit: max_bytes });
    }
    Ok(bytes)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ArtifactError> {
        if self.remaining() < count {
            return Err(ArtifactError::Truncated {
                needed: count,
                available: self.remaining(),
            });
        }
        let slice = &self.bytes[self.position..self.position + count];
        self.position += count;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, ArtifactError> {
        Ok(self.take(1)?[0])
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ArtifactError> {
        let slice = self.take(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }
}
