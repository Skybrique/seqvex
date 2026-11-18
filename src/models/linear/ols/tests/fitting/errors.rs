use seqvex::models::linear::ols::{AllocationSource, ConversionCause, FitError};

#[test]
fn fit_error_displays_designated_context_and_no_fabricated_causes() {
    fn assert_error_traits<T: std::error::Error + Clone + PartialEq + std::fmt::Debug>() {}
    assert_error_traits::<FitError>();

    let cases = [
        (
            FitError::RowTargetMismatch {
                features: 17,
                targets: 3,
            },
            &["feature row count 17", "target count 3"][..],
        ),
        (
            FitError::IndexOutOfRange { index: 0, rows: 19 },
            &["index 0", "19 rows"][..],
        ),
        (
            FitError::TooManyFeatures {
                features: 17,
                max: 3,
            },
            &["feature count 17", "maximum 3"][..],
        ),
        (
            FitError::RaggedFeatures {
                row: 0,
                expected: 17,
                actual: 3,
            },
            &["row 0", "width 3", "expected 17"][..],
        ),
        (
            FitError::InvalidRankThreshold { value: -0.25 },
            &["-0.25"][..],
        ),
        (
            FitError::SizeOverflow {
                rows: 19,
                features: 3,
            },
            &["19 rows", "3 design-matrix columns"][..],
        ),
        (
            FitError::Capacity {
                requested: 17,
                limit: 3,
            },
            &["17 bytes", "3 bytes"][..],
        ),
        (
            FitError::AllocationFailed {
                buffer: "test-buffer",
                requested_bytes: 17,
                source: AllocationSource::Faer,
            },
            &["test-buffer", "17 bytes", "faer allocator"][..],
        ),
        (
            FitError::AllocationFailed {
                buffer: "stack-buffer",
                requested_bytes: 19,
                source: AllocationSource::DynStack,
            },
            &["stack-buffer", "19 bytes", "dynamic stack allocator"][..],
        ),
        (
            FitError::AllocationFailed {
                buffer: "std-buffer",
                requested_bytes: 23,
                source: AllocationSource::Std,
            },
            &["std-buffer", "23 bytes", "standard allocator"][..],
        ),
        (FitError::DuplicateIndex { index: 0 }, &["index 0"][..]),
        (
            FitError::NonFiniteTrainingValue { row: 0, column: 0 },
            &["row 0", "column 0"][..],
        ),
        (
            FitError::RankDeficient {
                rank: 3,
                required: 17,
            },
            &["rank 3", "required rank 17"][..],
        ),
        (
            FitError::ConversionRejected(ConversionCause::BiasUnderflow),
            &["bias underflow"][..],
        ),
        (
            FitError::ConversionRejected(ConversionCause::CoefficientUnderflow),
            &["coefficient underflow"][..],
        ),
        (
            FitError::ConversionRejected(ConversionCause::Overflow),
            &["overflow"][..],
        ),
    ];
    for (error, designated) in cases {
        let message = error.to_string();
        for field in designated {
            assert!(message.contains(field), "{message:?} omitted {field:?}");
        }
        assert!(
            std::error::Error::source(&error).is_none(),
            "classification-only FitError must not expose a cause: {message}"
        );
    }

    for (error, expected) in [
        (FitError::TooFewRows, "no training rows were selected"),
        (
            FitError::ZeroFeatures,
            "selected training rows have zero feature columns",
        ),
        (FitError::ScratchLayout, "scratch layout is invalid"),
    ] {
        assert!(error.to_string().contains(expected));
        assert!(std::error::Error::source(&error).is_none());
    }

    let numerical = FitError::NumericalFailure;
    assert_eq!(numerical.to_string(), "OLS numerical calculation failed");
    assert!(std::error::Error::source(&numerical).is_none());
}
