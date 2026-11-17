//! Shared grouped-operation bounds, empty inputs and returned-state retention.

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdPredictionError, SgdPredictionError64, SgdSample, SgdSample64,
    SgdUpdateBatchError, SgdUpdateBatchError64,
};

use super::support::{
    bits, bits32_vec, bits64, bits64_vec, features, features64, sample, sample64, state32, state64,
};

#[test]
fn bound_first_rejects_before_any_row_validation() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let rows = [sample(&[0.5, 0.5], 1.0), sample(&[f32::NAN, 0.0], 1.0)];
    match model.update_batch(initial.clone(), &rows, 1).unwrap_err() {
        SgdUpdateBatchError::TooLarge { max, actual, state } => {
            assert_eq!((max, actual), (1, 2));
            assert_eq!(bits(&state.w), bits(&initial.w));
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
    assert!(matches!(
        model.update_batch(initial.clone(), &rows, 0),
        Err(SgdUpdateBatchError::TooLarge {
            max: 0,
            actual: 2,
            ..
        })
    ));

    let prediction_rows = [features(&[0.5, 0.5]), features(&[f32::NAN, 0.0])];
    assert!(matches!(
        model.predict_batch(&initial, &prediction_rows, 1),
        Err(SgdPredictionError::TooLarge { max: 1, actual: 2 })
    ));

    let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial64 = model64.initial_state();
    let rows64 = [sample64(&[0.5, 0.5], 1.0), sample64(&[f64::NAN, 0.0], 1.0)];
    assert!(matches!(
        model64.update_batch(initial64.clone(), &rows64, 1),
        Err(SgdUpdateBatchError64::TooLarge {
            max: 1,
            actual: 2,
            ..
        })
    ));
    let prediction_rows64 = [features64(&[0.5, 0.5]), features64(&[f64::NAN, 0.0])];
    assert!(matches!(
        model64.predict_batch(&initial64, &prediction_rows64, 1),
        Err(SgdPredictionError64::TooLarge { max: 1, actual: 2 })
    ));
}

#[test]
fn empty_and_zero_preserve_state_bits() {
    let model = Sgd::new(3, 0.1).expect("valid configuration");
    let state = state32(&[f32::NAN, -0.0, 0.0]);
    let state_bits = bits(&state.w);
    let empty: [Observation<SgdSample>; 0] = [];
    let returned = model
        .update_batch(state.clone(), &empty, 0)
        .expect("empty update");
    assert_eq!(bits(&returned.w), state_bits);
    let returned = model
        .update_batch(state.clone(), &empty, usize::MAX)
        .expect("empty update with max");
    assert_eq!(bits(&returned.w), state_bits);
    assert!(matches!(
        model.update_batch(state.clone(), &[sample(&[0.0, 0.0, 0.0], 0.0)], 0),
        Err(SgdUpdateBatchError::TooLarge {
            max: 0,
            actual: 1,
            ..
        })
    ));

    let empty_predictions: [Observation<Vector>; 0] = [];
    let predicted = model
        .predict_batch(&state, &empty_predictions, 0)
        .expect("empty prediction");
    assert!(predicted.is_empty());
    let predicted = model
        .predict_batch(&state, &empty_predictions, usize::MAX)
        .expect("empty prediction with max");
    assert!(predicted.is_empty());
    assert!(matches!(
        model.predict_batch(&state, &[features(&[0.0, 0.0, 0.0])], 0),
        Err(SgdPredictionError::TooLarge { max: 0, actual: 1 })
    ));

    let model64 = Sgd64::new(3, 0.1).expect("valid configuration");
    let state64 = state64(&[f64::NAN, -0.0, 0.0]);
    let state64_bits = bits64(&state64.w);
    let empty64: [Observation<SgdSample64>; 0] = [];
    let returned = model64
        .update_batch(state64.clone(), &empty64, 0)
        .expect("empty f64 update");
    assert_eq!(bits64(&returned.w), state64_bits);
    assert!(matches!(
        model64.update_batch(state64.clone(), &[sample64(&[0.0, 0.0, 0.0], 0.0)], 0),
        Err(SgdUpdateBatchError64::TooLarge {
            max: 0,
            actual: 1,
            ..
        })
    ));
}

#[test]
fn exact_limit_and_usize_max_accept() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let initial = model.initial_state();
    let rows = [sample(&[0.5, 0.5], 1.0), sample(&[-0.5, 0.5], 0.0)];
    let expected = rows.iter().fold(initial.clone(), |state, row| {
        model.update(&state, row).expect("update")
    });
    let exact = model
        .update_batch(initial.clone(), &rows, rows.len())
        .expect("exact limit");
    assert_eq!(bits(&exact.w), bits(&expected.w));
    let unlimited = model
        .update_batch(initial.clone(), &rows, usize::MAX)
        .expect("usize::MAX");
    assert_eq!(bits(&unlimited.w), bits(&expected.w));

    let invalid = [sample(&[1.0, 1.0], 1.0), sample(&[1.0], 1.0)];
    assert!(matches!(
        model.update_batch(initial.clone(), &invalid, usize::MAX),
        Err(SgdUpdateBatchError::Transition {
            failed_index: 1,
            committed: 1,
            ..
        })
    ));

    let prediction_rows = [features(&[0.5, 0.5]), features(&[-0.5, 0.5])];
    let expected_predictions: Vec<f32> = prediction_rows
        .iter()
        .map(|row| model.predict(&initial, row.value()).expect("prediction"))
        .collect();
    let exact_predictions = model
        .predict_batch(&initial, &prediction_rows, prediction_rows.len())
        .expect("exact prediction limit");
    assert_eq!(
        bits32_vec(&exact_predictions),
        bits32_vec(&expected_predictions)
    );
    let unlimited_predictions = model
        .predict_batch(&initial, &prediction_rows, usize::MAX)
        .expect("usize::MAX prediction");
    assert_eq!(
        bits32_vec(&unlimited_predictions),
        bits32_vec(&expected_predictions)
    );

    let invalid_predictions = [features(&[0.5, 0.5]), features(&[f32::NAN, 0.0])];
    assert!(matches!(
        model.predict_batch(&initial, &invalid_predictions, usize::MAX),
        Err(SgdPredictionError::At { index: 1, .. })
    ));

    let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
    let initial64 = model64.initial_state();
    let rows64 = [sample64(&[0.5, 0.5], 1.0), sample64(&[-0.5, 0.5], 0.0)];
    let expected64 = rows64.iter().fold(initial64.clone(), |state, row| {
        model64.update(&state, row).expect("update")
    });
    let unlimited64 = model64
        .update_batch(initial64.clone(), &rows64, usize::MAX)
        .expect("usize::MAX f64");
    assert_eq!(bits64(&unlimited64.w), bits64(&expected64.w));

    let invalid64 = [sample64(&[0.5, 0.5], 1.0), sample64(&[1.0], 1.0)];
    assert!(matches!(
        model64.update_batch(initial64.clone(), &invalid64, usize::MAX),
        Err(SgdUpdateBatchError64::Transition {
            failed_index: 1,
            committed: 1,
            ..
        })
    ));

    let prediction_rows64 = [features64(&[0.5, 0.5]), features64(&[-0.5, 0.5])];
    let expected_predictions64: Vec<f64> = prediction_rows64
        .iter()
        .map(|row| {
            model64
                .predict(&initial64, row.value())
                .expect("prediction")
        })
        .collect();
    let exact64 = model64
        .predict_batch(&initial64, &prediction_rows64, prediction_rows64.len())
        .expect("exact f64 prediction limit");
    assert_eq!(bits64_vec(&exact64), bits64_vec(&expected_predictions64));
    let invalid_predictions64 = [features64(&[0.5, 0.5]), features64(&[f64::NAN, 0.0])];
    assert!(matches!(
        model64.predict_batch(&initial64, &invalid_predictions64, usize::MAX),
        Err(SgdPredictionError64::At { index: 1, .. })
    ));
}

#[test]
fn bitwise_retention_with_nan_and_signed_zero() {
    let model = Sgd::new(3, 0.1).expect("valid configuration");
    let state = state32(&[f32::NAN, -0.0, 0.0]);
    let expected = bits(&state.w);
    let empty: [Observation<SgdSample>; 0] = [];
    assert_eq!(
        bits(
            &model
                .update_batch(state.clone(), &empty, 0)
                .expect("empty update")
                .w
        ),
        expected
    );
    assert_eq!(
        bits(
            &model
                .update_batch(state.clone(), &empty, usize::MAX)
                .expect("empty update with max")
                .w
        ),
        expected
    );
    let oversize = [sample(&[0.0, 0.0, 0.0], 0.0), sample(&[0.0, 0.0, 0.0], 0.0)];
    match model.update_batch(state.clone(), &oversize, 1).unwrap_err() {
        SgdUpdateBatchError::TooLarge {
            state: returned, ..
        } => {
            assert_eq!(bits(&returned.w), expected);
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }

    let model64 = Sgd64::new(3, 0.1).expect("valid configuration");
    let state64 = state64(&[f64::NAN, -0.0, 0.0]);
    let expected64 = bits64(&state64.w);
    let empty64: [Observation<SgdSample64>; 0] = [];
    assert_eq!(
        bits64(
            &model64
                .update_batch(state64.clone(), &empty64, 0)
                .expect("empty f64 update")
                .w
        ),
        expected64
    );
    let oversize64 = [
        sample64(&[0.0, 0.0, 0.0], 0.0),
        sample64(&[0.0, 0.0, 0.0], 0.0),
    ];
    match model64
        .update_batch(state64.clone(), &oversize64, 1)
        .unwrap_err()
    {
        SgdUpdateBatchError64::TooLarge {
            state: returned, ..
        } => {
            assert_eq!(bits64(&returned.w), expected64);
        }
        other => panic!("expected f64 TooLarge, got {other:?}"),
    }
}
