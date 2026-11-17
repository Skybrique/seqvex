//! Public grouped-error presentation evidence.

use std::error::Error;

use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdPredictionError, SgdPredictionError64, SgdUpdateBatchError,
    SgdUpdateBatchError64,
};

use super::support::{features, features64, sample, sample64};

#[test]
fn error_display_and_source_32() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let too_large = model
        .update_batch(
            initial.clone(),
            &[sample(&[0.0, 0.0], 0.0), sample(&[0.0, 0.0], 0.0)],
            1,
        )
        .unwrap_err();
    let text = too_large.to_string();
    assert!(text.contains("max 1"));
    assert!(text.contains("got 2"));
    assert!(text.contains("supplied state was returned unchanged"));
    assert!(too_large.source().is_none());

    let transition = model
        .update_batch(
            initial.clone(),
            &[sample(&[0.0, 0.0], 0.0), sample(&[1.0], 1.0)],
            2,
        )
        .unwrap_err();
    let text = transition.to_string();
    let inner = match &transition {
        SgdUpdateBatchError::Transition { error, .. } => error.to_string(),
        other => panic!("expected Transition, got {other:?}"),
    };
    assert!(text.contains("index 1"));
    assert!(text.contains("1 committed rows"));
    assert!(!text.contains(&inner));
    assert_eq!(
        transition.source().expect("inner source").to_string(),
        inner
    );

    let prediction_too_large = model
        .predict_batch(&initial, &[features(&[0.0, 0.0]), features(&[0.0, 0.0])], 1)
        .unwrap_err();
    let text = prediction_too_large.to_string();
    assert!(text.contains("max 1"));
    assert!(text.contains("got 2"));
    assert!(text.contains("no output was returned"));
    assert!(prediction_too_large.source().is_none());

    let prediction_at = model
        .predict_batch(
            &initial,
            &[features(&[0.0, 0.0]), features(&[f32::NAN, 0.0])],
            2,
        )
        .unwrap_err();
    let text = prediction_at.to_string();
    let inner = match &prediction_at {
        SgdPredictionError::At { error, .. } => error.to_string(),
        other => panic!("expected At, got {other:?}"),
    };
    assert!(text.contains("index 1"));
    assert!(text.contains("no partial prediction vector was returned"));
    assert!(!text.contains(&inner));
    assert_eq!(
        prediction_at.source().expect("inner source").to_string(),
        inner
    );
}

#[test]
fn error_display_and_source_64() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let too_large = model
        .update_batch(
            initial.clone(),
            &[sample64(&[0.0, 0.0], 0.0), sample64(&[0.0, 0.0], 0.0)],
            1,
        )
        .unwrap_err();
    let text = too_large.to_string();
    assert!(text.contains("max 1"));
    assert!(text.contains("got 2"));
    assert!(too_large.source().is_none());

    let transition = model
        .update_batch(
            initial.clone(),
            &[sample64(&[0.0, 0.0], 0.0), sample64(&[1.0], 1.0)],
            2,
        )
        .unwrap_err();
    let text = transition.to_string();
    let inner = match &transition {
        SgdUpdateBatchError64::Transition { error, .. } => error.to_string(),
        other => panic!("expected f64 Transition, got {other:?}"),
    };
    assert!(text.contains("index 1"));
    assert!(!text.contains(&inner));
    assert_eq!(
        transition.source().expect("inner source").to_string(),
        inner
    );

    let prediction_too_large = model
        .predict_batch(
            &initial,
            &[features64(&[0.0, 0.0]), features64(&[0.0, 0.0])],
            1,
        )
        .unwrap_err();
    assert!(matches!(
        prediction_too_large,
        SgdPredictionError64::TooLarge { max: 1, actual: 2 }
    ));
    assert!(prediction_too_large.source().is_none());

    let prediction_at = model
        .predict_batch(
            &initial,
            &[features64(&[0.0, 0.0]), features64(&[f64::NAN, 0.0])],
            2,
        )
        .unwrap_err();
    let text = prediction_at.to_string();
    let inner = match &prediction_at {
        SgdPredictionError64::At { error, .. } => error.to_string(),
        other => panic!("expected f64 At, got {other:?}"),
    };
    assert!(text.contains("index 1"));
    assert!(!text.contains(&inner));
    assert_eq!(
        prediction_at.source().expect("inner source").to_string(),
        inner
    );
}
