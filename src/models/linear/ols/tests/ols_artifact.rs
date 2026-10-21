//! Artifact v1 tests: metadata, content identity, encoding, and validation.
//!
//! Includes an independently assembled byte-layout fixture (fields, flags and
//! checksum written by hand from the specification) rather than only exporting
//! and importing through the same implementation.

use seqvex::foundation::numerical::Vector;
use seqvex::models::linear::ols::{
    ArtifactError, ArtifactExpectation, ArtifactLimits, Cutoff, Expect, FitConfig, PublicationId,
    TrainedModel, TrainingMeta, TransformId, fit,
};

use sha2::{Digest, Sha256};
use std::io::{Read, Write};

fn limits() -> ArtifactLimits {
    ArtifactLimits {
        max_features: 1024,
        max_bytes: 1 << 20,
    }
}

fn meta(schema: u8, transform: Option<TransformId>, cutoff: Option<Cutoff>) -> TrainingMeta {
    TrainingMeta {
        schema_id: [schema; 16],
        transform_link: transform,
        cutoff,
        fit_revision: [1, 2, 3, 4, 5, 6, 7, 8],
    }
}

fn simple_meta() -> TrainingMeta {
    meta(7, None, None)
}

/// Fits `y = 2x + 1` (intercept) or `y = 2x` (no intercept).
fn model(intercept: bool, meta: &TrainingMeta) -> TrainedModel {
    let features = vec![
        Vector::from_slice(&[0.0]),
        Vector::from_slice(&[1.0]),
        Vector::from_slice(&[2.0]),
    ];
    let targets = if intercept {
        [1.0f32, 3.0, 5.0]
    } else {
        [0.0f32, 2.0, 4.0]
    };
    let config = FitConfig {
        intercept,
        ..FitConfig::default()
    };
    let result = fit(&features, &targets, &[0, 1, 2], &config).expect("fit");
    result.into_trained(meta).expect("into_trained")
}

fn round_trip(model: &TrainedModel, publication: Option<PublicationId>) -> Vec<u8> {
    let mut buffer = Vec::new();
    model.export_to(&mut buffer, limits(), publication).unwrap();
    buffer
}

fn import(bytes: &[u8]) -> Result<seqvex::models::linear::ols::LoadedArtifact, ArtifactError> {
    let mut reader = bytes;
    TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), limits())
}

fn assert_same_params(a: &TrainedModel, b: &TrainedModel) {
    let aw = a.predictor().weights().as_slice();
    let bw = b.predictor().weights().as_slice();
    assert_eq!(aw.len(), bw.len());
    for (x, y) in aw.iter().zip(bw) {
        assert_eq!(x.to_bits(), y.to_bits(), "weight bits differ");
    }
    assert_eq!(
        a.predictor().bias().to_bits(),
        b.predictor().bias().to_bits(),
        "intercept bits differ"
    );
    assert_eq!(a.metadata(), b.metadata());
}

#[test]
fn round_trip_with_intercept() {
    let original = model(true, &simple_meta());
    assert!(original.metadata().intercept);
    let bytes = round_trip(&original, None);
    let loaded = import(&bytes).expect("import");
    assert!(loaded.publication.is_none());
    assert_same_params(&original, &loaded.model);
}

#[test]
fn round_trip_without_intercept() {
    let original = model(false, &simple_meta());
    assert!(!original.metadata().intercept);
    let bytes = round_trip(&original, None);
    let loaded = import(&bytes).expect("import");
    assert!(!loaded.model.metadata().intercept);
    assert_same_params(&original, &loaded.model);
}

#[test]
fn zero_valued_but_present_intercept_is_preserved() {
    // y = 3x: the fitted intercept is exactly zero but must remain present.
    let features = vec![
        Vector::from_slice(&[1.0]),
        Vector::from_slice(&[2.0]),
        Vector::from_slice(&[3.0]),
    ];
    let targets = [3.0f32, 6.0, 9.0];
    let result = fit(&features, &targets, &[0, 1, 2], &FitConfig::default()).unwrap();
    let original = result.into_trained(&simple_meta()).unwrap();
    assert!(original.metadata().intercept);
    assert!(original.predictor().bias().abs() < 1e-6);
    let loaded = import(&round_trip(&original, None)).expect("import");
    assert!(loaded.model.metadata().intercept);
    assert_eq!(
        loaded.model.predictor().bias().to_bits(),
        original.predictor().bias().to_bits()
    );
}

#[test]
fn optional_metadata_and_publication_round_trip() {
    let transform: TransformId = [9; 16];
    let cutoff = Cutoff {
        kind: 3,
        value: 1_700_000_000,
    };
    let original = model(true, &meta(11, Some(transform), Some(cutoff)));
    let publication: PublicationId = [42; 16];
    let bytes = round_trip(&original, Some(publication));
    let loaded = import(&bytes).expect("import");
    assert_eq!(loaded.model.metadata().transform_link, Some(transform));
    assert_eq!(loaded.model.metadata().cutoff, Some(cutoff));
    assert_eq!(loaded.publication, Some(publication));
    assert_same_params(&original, &loaded.model);
}

#[test]
fn model_identity_is_stable_across_repeated_construction() {
    let a = model(true, &simple_meta());
    let b = model(true, &simple_meta());
    assert_eq!(a.model_id(), b.model_id());
    assert_eq!(a.metadata().param_hash, b.metadata().param_hash);
}

#[test]
fn publication_changes_checksum_but_not_model_id() {
    let original = model(true, &simple_meta());
    let a = round_trip(&original, Some([1; 16]));
    let b = round_trip(&original, Some([2; 16]));
    assert_ne!(a, b, "publication must alter the artifact bytes/checksum");
    let la = import(&a).unwrap();
    let lb = import(&b).unwrap();
    assert_eq!(la.model.model_id(), lb.model.model_id());
    assert_eq!(la.model.model_id(), original.model_id());
    assert_eq!(la.publication, Some([1; 16]));
    assert_eq!(lb.publication, Some([2; 16]));
}

#[test]
fn parameter_changes_affect_model_id() {
    let a = model(true, &simple_meta());
    // Different targets -> different parameters -> different identity.
    let features = vec![
        Vector::from_slice(&[0.0]),
        Vector::from_slice(&[1.0]),
        Vector::from_slice(&[2.0]),
    ];
    let targets = [2.0f32, 4.0, 6.0];
    let other = fit(&features, &targets, &[0, 1, 2], &FitConfig::default())
        .unwrap()
        .into_trained(&simple_meta())
        .unwrap();
    assert_ne!(a.model_id(), other.model_id());
}

#[test]
fn training_linkage_changes_affect_model_id() {
    let a = model(true, &meta(7, None, None));
    let b = model(true, &meta(8, None, None));
    let c = model(true, &meta(7, Some([1; 16]), None));
    assert_ne!(a.model_id(), b.model_id());
    assert_ne!(a.model_id(), c.model_id());
    // Same parameters, different linkage -> same parameter hash.
    assert_eq!(a.metadata().param_hash, b.metadata().param_hash);
    assert_eq!(a.metadata().param_hash, c.metadata().param_hash);
}

#[test]
fn parameter_hash_ignores_fit_revision_and_cutoff() {
    let a = model(true, &meta(7, None, None));
    let mut meta_b = meta(7, None, Some(Cutoff { kind: 1, value: 5 }));
    meta_b.fit_revision = [9; 8];
    let b = model(true, &meta_b);
    assert_eq!(a.metadata().param_hash, b.metadata().param_hash);
    assert_ne!(a.model_id(), b.model_id());
}

#[test]
fn expectations_match_and_mismatch() {
    let original = model(true, &meta(7, Some([3; 16]), None));
    let bytes = round_trip(&original, None);

    let matching = ArtifactExpectation {
        feature_count: Expect::Required(1),
        intercept: Expect::Required(true),
        schema_id: Expect::Required([7; 16]),
        transform_link: Expect::Required([3; 16]),
        cutoff: Expect::Absent,
    };
    let mut reader = &bytes[..];
    TrainedModel::import_from(&mut reader, &matching, limits()).expect("matching");

    let cases: [ArtifactExpectation; 5] = [
        ArtifactExpectation {
            feature_count: Expect::Required(2),
            ..ArtifactExpectation::any()
        },
        ArtifactExpectation {
            intercept: Expect::Required(false),
            ..ArtifactExpectation::any()
        },
        ArtifactExpectation {
            schema_id: Expect::Required([1; 16]),
            ..ArtifactExpectation::any()
        },
        ArtifactExpectation {
            transform_link: Expect::Absent,
            ..ArtifactExpectation::any()
        },
        ArtifactExpectation {
            cutoff: Expect::Required(Cutoff { kind: 0, value: 0 }),
            ..ArtifactExpectation::any()
        },
    ];
    for expected in cases {
        let mut reader = &bytes[..];
        assert!(
            TrainedModel::import_from(&mut reader, &expected, limits()).is_err(),
            "expected rejection for {expected:?}"
        );
    }
}

#[test]
fn rejects_bad_magic_version_and_flags() {
    let bytes = round_trip(&model(true, &simple_meta()), None);

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        import(&bad_magic),
        Err(ArtifactError::FieldMismatch { field: "magic" })
    ));

    let mut bad_version = bytes.clone();
    bad_version[5..7].copy_from_slice(&2u16.to_le_bytes());
    assert!(matches!(
        import(&bad_version),
        Err(ArtifactError::UnsupportedVersion { found: 2, .. })
    ));

    let mut bad_flags = bytes.clone();
    bad_flags[7] |= 0x80;
    assert!(matches!(
        import(&bad_flags),
        Err(ArtifactError::UnexpectedField {
            field: "file_flags"
        })
    ));
}

#[test]
fn rejects_truncated_corrupted_and_trailing() {
    let bytes = round_trip(&model(true, &simple_meta()), None);

    let truncated = &bytes[..bytes.len() - 1];
    assert!(matches!(
        import(truncated),
        Err(ArtifactError::Truncated { .. })
    ));

    // Flip the first coefficient byte (offset 44 for d=1, no optional fields):
    // the checksum no longer matches.
    let mut corrupted = bytes.clone();
    corrupted[44] ^= 0xff;
    assert!(matches!(
        import(&corrupted),
        Err(ArtifactError::ChecksumMismatch { .. })
    ));

    let mut trailing = bytes.clone();
    trailing.push(0xAB);
    assert!(matches!(
        import(&trailing),
        Err(ArtifactError::TrailingBytes { count: 1 })
    ));
}

#[test]
fn rejects_non_finite_parameter() {
    let bytes = round_trip(&model(true, &simple_meta()), None);
    let mut corrupted = bytes.clone();
    // Overwrite the first coefficient (offset 44) with NaN; finiteness is
    // checked before the checksum.
    corrupted[44..48].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(matches!(
        import(&corrupted),
        Err(ArtifactError::NonFiniteParameter { index: 0 })
    ));
}

#[test]
fn rejects_model_id_mismatch_after_checksum_is_repaired() {
    let mut bytes = round_trip(&model(true, &simple_meta()), None);
    let body_len = bytes.len() - 32;
    // Zero the stored model_id (last 32 bytes of the body) and repair checksum.
    for byte in &mut bytes[body_len - 32..body_len] {
        *byte = 0;
    }
    let repaired = checksum(&bytes[..body_len]);
    bytes[body_len..].copy_from_slice(&repaired);
    assert!(matches!(
        import(&bytes),
        Err(ArtifactError::ModelIdMismatch { .. })
    ));
}

#[test]
fn rejects_over_limit_artifacts() {
    let bytes = round_trip(&model(true, &simple_meta()), None);

    let too_small_bytes = ArtifactLimits {
        max_features: 1024,
        max_bytes: 8,
    };
    let mut reader = &bytes[..];
    assert!(matches!(
        TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), too_small_bytes),
        Err(ArtifactError::Capacity { .. })
    ));

    let too_few_features = ArtifactLimits {
        max_features: 0,
        max_bytes: 1 << 20,
    };
    let mut reader = &bytes[..];
    assert!(matches!(
        TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), too_few_features),
        Err(ArtifactError::Capacity { .. })
    ));
}

#[test]
fn writer_and_reader_failures_are_reported() {
    let original = model(true, &simple_meta());

    let mut writer = FailWriter { remaining: 10 };
    assert!(matches!(
        original.export_to(&mut writer, limits(), None),
        Err(ArtifactError::Io { .. })
    ));

    let mut reader = FailReader { remaining: 4 };
    assert!(matches!(
        TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), limits()),
        Err(ArtifactError::Io { .. })
    ));
}

#[test]
fn artifact_error_displays_designated_fields_and_hex_values() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<ArtifactError>();

    let cases = [
        (
            ArtifactError::Truncated {
                needed: 17,
                available: 3,
            },
            &["needed 17 bytes", "3 available"][..],
        ),
        (
            ArtifactError::UnsupportedVersion {
                found: 17,
                supported: 3,
            },
            &["version 17", "supported version is 3"][..],
        ),
        (
            ArtifactError::FieldMismatch { field: "schema_id" },
            &["schema_id"][..],
        ),
        (
            ArtifactError::UnexpectedField { field: "cutoff" },
            &["cutoff", "absent"][..],
        ),
        (
            ArtifactError::NonFiniteParameter { index: 0 },
            &["index 0"][..],
        ),
        (ArtifactError::Capacity { limit: 17 }, &["limit 17"][..]),
        (
            ArtifactError::TrailingBytes { count: 3 },
            &["3 trailing bytes"][..],
        ),
    ];
    for (error, designated) in cases {
        let message = error.to_string();
        for field in designated {
            assert!(message.contains(field), "{message:?} omitted {field:?}");
        }
        assert!(std::error::Error::source(&error).is_none());
    }

    for (error, expected) in [
        (
            ArtifactError::SchemaMismatch,
            "artifact schema identifier is inconsistent or unsupported",
        ),
        (
            ArtifactError::ShapeMismatch,
            "artifact shape is inconsistent",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }

    let expected = [
        0x00, 0x01, 0x0a, 0x0f, 0x10, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x80, 0x9a, 0xab,
        0xbc, 0xcd, 0xde, 0xef, 0xf0, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
        0xaa, 0xff,
    ];
    let actual = [
        0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0x00, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4, 0xc3, 0xd2,
        0xe1, 0x00,
    ];
    let expected_hex = "00010a0f101a2b3c4d5e6f70809aabbccddeeff000112233445566778899aaff";
    let actual_hex = "ffeeddccbbaa998877665544332211000f1e2d3c4b5a69788796a5b4c3d2e100";
    assert_eq!(expected_hex.len(), 64);
    assert_eq!(actual_hex.len(), 64);

    let checksum = ArtifactError::ChecksumMismatch { expected, actual }.to_string();
    let checksum_error = ArtifactError::ChecksumMismatch { expected, actual };
    assert!(std::error::Error::source(&checksum_error).is_none());
    assert!(checksum.contains(&format!("expected {expected_hex}")));
    assert!(checksum.contains(&format!("actual {actual_hex}")));
    assert!(!checksum.contains(&format!("expected {actual_hex}")));
    assert!(!checksum.contains(&format!("actual {expected_hex}")));

    let model_id_error = ArtifactError::ModelIdMismatch { expected, actual };
    assert!(std::error::Error::source(&model_id_error).is_none());
    let model_id = model_id_error.to_string();
    assert!(model_id.contains(&format!("expected {expected_hex}")));
    assert!(model_id.contains(&format!("actual {actual_hex}")));
    assert!(!model_id.contains(&format!("expected {actual_hex}")));
    assert!(!model_id.contains(&format!("actual {expected_hex}")));

    let capacity = ArtifactError::Capacity { limit: 17 }.to_string();
    assert!(!capacity.contains("bytes"));
}

#[test]
fn artifact_io_display_chains_the_stored_cause_without_duplicating_it() {
    assert_eq!(
        format!("{:?}", ArtifactError::SchemaMismatch),
        "SchemaMismatch"
    );
    let artifact_error = ArtifactError::Io {
        source: std::io::Error::other("artifact I/O cause sentinel"),
    };
    let stored = match &artifact_error {
        ArtifactError::Io { source } => source,
        _ => unreachable!(),
    };
    let message = artifact_error.to_string();
    assert_eq!(message, "artifact I/O failed");
    assert!(!message.contains("cause sentinel"));

    let source = std::error::Error::source(&artifact_error).unwrap();
    let source = source.downcast_ref::<std::io::Error>().unwrap();
    assert!(std::ptr::eq(source, stored));
    assert_eq!(source.kind(), std::io::ErrorKind::Other);
}

#[test]
fn independently_assembled_artifact_matches_export_and_imports() {
    let original = model(
        true,
        &meta(7, Some([3; 16]), Some(Cutoff { kind: 2, value: 77 })),
    );
    let publication: PublicationId = [5; 16];

    let independent = assemble(&original, Some(publication));
    let exported = round_trip(&original, Some(publication));
    assert_eq!(
        independent, exported,
        "export must match the specified layout"
    );

    let loaded = import(&independent).expect("independent artifact imports");
    assert_same_params(&original, &loaded.model);
    assert_eq!(loaded.publication, Some(publication));
}

#[test]
fn rejects_enormous_dimension_before_coefficient_allocation() {
    // A structurally sufficient header declares a huge dimension, but the input
    // is tiny. With permissive max_features and a small max_bytes, the implied
    // artifact size exceeds the byte limit and must be rejected *before* any
    // coefficient allocation.
    let dimension: u32 = 1_000_000_000;
    let mut header: Vec<u8> = Vec::new();
    header.extend_from_slice(b"SXOLS");
    header.extend_from_slice(&1u16.to_le_bytes());
    header.push(0u8); // no intercept/transform/cutoff/publication
    header.extend_from_slice(&dimension.to_le_bytes());
    header.extend_from_slice(&[0u8; 16]); // schema
    header.extend_from_slice(&[0u8; 8]); // fit_revision
    header.extend_from_slice(&0u32.to_le_bytes()); // rows
    header.extend_from_slice(&dimension.to_le_bytes()); // features == d
    assert_eq!(header.len(), 44);

    let permissive = ArtifactLimits {
        max_features: 2_000_000_000,
        max_bytes: 64,
    };
    let mut reader = &header[..];
    let error = TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), permissive)
        .expect_err("must reject before allocating");
    assert!(
        matches!(error, ArtifactError::Capacity { .. }),
        "expected Capacity, got {error:?}"
    );
}

#[test]
fn truncated_payload_reports_truncated_not_capacity() {
    // d=1, intercept present, no transform/cutoff/publication: header ends at 44,
    // payload is 4 (coefficient) + 4 (intercept) + 64 (model_id + checksum) = 72.
    let bytes = round_trip(&model(true, &simple_meta()), None);
    assert_eq!(bytes.len(), 116);
    for cut in [44usize, 48, 52, bytes.len() - 1] {
        let mut reader = &bytes[..cut];
        let error = TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), limits())
            .expect_err("short payload must be rejected");
        assert!(
            matches!(error, ArtifactError::Truncated { .. }),
            "cut {cut}: expected Truncated, got {error:?}"
        );
    }
}

#[test]
fn valid_artifact_at_exact_byte_limit() {
    let bytes = round_trip(&model(true, &simple_meta()), None);
    let exact = ArtifactLimits {
        max_features: 1024,
        max_bytes: bytes.len(),
    };
    let mut reader = &bytes[..];
    TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), exact)
        .expect("artifact exactly at the byte limit must import");
}

#[test]
fn valid_artifact_rejected_when_max_bytes_too_small() {
    let bytes = round_trip(&model(true, &simple_meta()), None);
    let too_small = ArtifactLimits {
        max_features: 1024,
        max_bytes: bytes.len() - 1,
    };
    let mut reader = &bytes[..];
    let error = TrainedModel::import_from(&mut reader, &ArtifactExpectation::any(), too_small)
        .expect_err("oversized input must be rejected");
    assert!(
        matches!(error, ArtifactError::Capacity { .. }),
        "expected Capacity, got {error:?}"
    );
}

// ------------------------- test-only helpers -------------------------

fn checksum(body: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"sx.ols.artifact.v1\0");
    hasher.update(body);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Independent implementation of the documented `ModelId` encoding.
fn independent_model_id(
    m: &seqvex::models::linear::ols::ArtifactMetadata,
    weights: &[f32],
    bias: f32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"sx.ols.model.v1\0");
    hasher.update(1u16.to_le_bytes());
    let mut flags = 0u8;
    if m.intercept {
        flags |= 1;
    }
    if m.transform_link.is_some() {
        flags |= 2;
    }
    if m.cutoff.is_some() {
        flags |= 4;
    }
    hasher.update([flags]);
    hasher.update(m.features.to_le_bytes());
    hasher.update(m.schema_id);
    if let Some(t) = m.transform_link {
        hasher.update(t);
    }
    if let Some(c) = m.cutoff {
        hasher.update([c.kind]);
        hasher.update(c.value.to_le_bytes());
    }
    hasher.update(m.fit_revision);
    hasher.update(m.rows.to_le_bytes());
    hasher.update(m.features.to_le_bytes());
    for w in weights {
        hasher.update(w.to_le_bytes());
    }
    if m.intercept {
        hasher.update(bias.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Byte-for-byte assembly from the specification (not via `export_to`).
fn assemble(model: &TrainedModel, publication: Option<PublicationId>) -> Vec<u8> {
    let m = model.metadata();
    let weights = model.predictor().weights().as_slice();
    let mut flags = 0u8;
    if m.intercept {
        flags |= 1;
    }
    if m.transform_link.is_some() {
        flags |= 2;
    }
    if m.cutoff.is_some() {
        flags |= 4;
    }
    if publication.is_some() {
        flags |= 8;
    }

    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(b"SXOLS");
    body.extend_from_slice(&1u16.to_le_bytes());
    body.push(flags);
    body.extend_from_slice(&m.features.to_le_bytes());
    body.extend_from_slice(&m.schema_id);
    if let Some(t) = m.transform_link {
        body.extend_from_slice(&t);
    }
    if let Some(c) = m.cutoff {
        body.push(c.kind);
        body.extend_from_slice(&c.value.to_le_bytes());
    }
    body.extend_from_slice(&m.fit_revision);
    body.extend_from_slice(&m.rows.to_le_bytes());
    body.extend_from_slice(&m.features.to_le_bytes());
    if let Some(p) = publication {
        body.extend_from_slice(&p);
    }
    for w in weights {
        body.extend_from_slice(&w.to_le_bytes());
    }
    if m.intercept {
        body.extend_from_slice(&model.predictor().bias().to_le_bytes());
    }
    body.extend_from_slice(&independent_model_id(m, weights, model.predictor().bias()));

    let mut artifact = body.clone();
    artifact.extend_from_slice(&checksum(&body));
    artifact
}

struct FailWriter {
    remaining: usize,
}

impl Write for FailWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Err(std::io::Error::other("writer failed"));
        }
        let written = buf.len().min(self.remaining);
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct FailReader {
    remaining: usize,
}

impl Read for FailReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Err(std::io::Error::other("reader failed"));
        }
        let read = buf.len().min(self.remaining);
        for slot in &mut buf[..read] {
            *slot = 0;
        }
        self.remaining -= read;
        Ok(read)
    }
}
