//! Precision-profile construction, failure and state-preservation contracts.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdServingConversionError, SgdState, SgdState64,
};

use super::support::{bits64, reference_update32_sum, reference_update64, sample64};
use crate::{bits, sample};

fn assert_error_traits<T>()
where
    T: std::error::Error + Clone + Copy + PartialEq + Eq + std::fmt::Debug,
{
}

#[test]
fn a_construction_and_error_contract() {
    assert_eq!(Sgd64::new(0, 0.1).unwrap_err(), SgdError64::ZeroDimension);
    for rate in [0.0f64, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            Sgd64::new(3, rate).unwrap_err(),
            SgdError64::InvalidLearningRate
        );
    }
    for rate in [f64::MIN_POSITIVE, 1.0e-300, 1.0, 1.0e300] {
        let model = Sgd64::new(3, rate).expect("finite positive rate");
        assert_eq!(model.dimension(), 3);
        assert_eq!(model.learning_rate().to_bits(), rate.to_bits());
    }

    assert_error_traits::<SgdError64>();
    assert_error_traits::<SgdServingConversionError>();

    let mismatch = SgdError64::DimensionMismatch {
        expected: 4,
        actual: 1,
    };
    assert_eq!(
        mismatch.to_string(),
        "dimension mismatch: expected 4, got 1"
    );
    assert!(std::error::Error::source(&mismatch).is_none());

    let conversion = SgdServingConversionError::Underflow { index: 2 };
    assert!(conversion.to_string().contains("underflows"));
    assert!(std::error::Error::source(&conversion).is_none());

    let state = Sgd64::new(2, 0.1)
        .expect("valid configuration")
        .initial_state();
    assert!(state.w.as_slice().iter().all(|value| *value == 0.0));
}

#[test]
fn learned_state_failures_preserve_and_continue() {
    let model = Sgd64::new(1, 1.0e308).expect("valid configuration");
    let learned = model
        .update(&model.initial_state(), &sample64(&[1.0], 1.0))
        .expect("learned finite state");
    let learned_bits = bits64(&learned.w);
    assert_eq!(
        model
            .update(&learned, &sample64(&[1.0e308], 1.0e308))
            .unwrap_err(),
        SgdError64::NonFiniteCandidate
    );
    assert_eq!(bits64(&learned.w), learned_bits);
    let next = model
        .update(&learned, &sample64(&[0.0], 1.0))
        .expect("valid continuation");
    let expected = reference_update64(1.0e308, learned.w.as_slice(), &[0.0], 1.0);
    assert_eq!(
        bits64(&next.w),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let model = Sgd::new(1, 1.0e38).expect("valid configuration");
    let learned = model
        .update(&model.initial_state(), &sample(&[1.0], 1.0))
        .expect("learned finite state");
    let learned_bits = bits(&learned.w);
    assert_eq!(
        model
            .update(&learned, &sample(&[3.0e38], 3.0e38))
            .unwrap_err(),
        SgdError::NonFiniteCandidate
    );
    assert_eq!(bits(&learned.w), learned_bits);
    let next = model
        .update(&learned, &sample(&[0.0], 1.0))
        .expect("valid continuation");
    let expected = reference_update32_sum(1.0e38, learned.w.as_slice(), &[0.0], 1.0);
    assert_eq!(
        bits(&next.w),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
    let learned = model64
        .update(&model64.initial_state(), &sample64(&[1.0, -1.0], 0.5))
        .expect("learned state");
    let learned_bits = bits64(&learned.w);
    assert_eq!(
        model64
            .predict(&learned, &Vector64::from_slice(&[1.0]))
            .unwrap_err(),
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        model64
            .predict(&learned, &Vector64::from_slice(&[f64::NAN, 0.0]))
            .unwrap_err(),
        SgdError64::NonFiniteInput
    );
    let short = SgdState64 {
        w: Vector64::zeros(1),
    };
    assert_eq!(
        model64
            .update(&short, &sample64(&[1.0, 0.0], 0.0))
            .unwrap_err(),
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        model64
            .update(&learned, &sample64(&[1.0, 0.0], f64::INFINITY))
            .unwrap_err(),
        SgdError64::NonFiniteInput
    );
    assert_eq!(
        model64
            .update(&learned, &sample64(&[f64::NAN, 0.0], 0.5))
            .unwrap_err(),
        SgdError64::NonFiniteInput
    );
    assert_eq!(bits64(&learned.w), learned_bits);
    let continued = model64
        .update(&learned, &sample64(&[0.5, 1.0], 0.0))
        .expect("valid continuation after failures");
    let expected = reference_update64(0.1, learned.w.as_slice(), &[0.5, 1.0], 0.0);
    assert_eq!(
        bits64(&continued.w),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let model32 = Sgd::new(2, 0.1).expect("valid configuration");
    let learned32 = model32
        .update(&model32.initial_state(), &sample(&[1.0, -1.0], 0.5))
        .expect("learned state");
    let learned32_bits = bits(&learned32.w);
    assert_eq!(
        model32
            .predict(&learned32, &Vector::from_slice(&[1.0]))
            .unwrap_err(),
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        model32
            .update(&learned32, &sample(&[1.0, 0.0], f32::INFINITY))
            .unwrap_err(),
        SgdError::NonFiniteInput
    );
    assert_eq!(
        model32
            .update(&learned32, &sample(&[f32::NAN, 0.0], 0.5))
            .unwrap_err(),
        SgdError::NonFiniteInput
    );
    // Deliberately constructed wrong-dimension state; the learned state above
    // is checked separately for feature/target failures and continuation.
    let short32 = SgdState {
        w: Vector::zeros(1),
    };
    assert_eq!(
        model32
            .update(&short32, &sample(&[1.0, 0.0], 0.0))
            .unwrap_err(),
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(bits(&learned32.w), learned32_bits);
    let continued32 = model32
        .update(&learned32, &sample(&[0.5, 1.0], 0.0))
        .expect("valid continuation after failures");
    let expected32 = reference_update32_sum(0.1, learned32.w.as_slice(), &[0.5, 1.0], 0.0);
    assert_eq!(
        bits(&continued32.w),
        expected32.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn a_output_and_precedence_failures_preserve_state() {
    let model = Sgd64::new(1, 0.1).expect("valid configuration");
    // Deliberately constructed extreme state for the output-overflow boundary;
    // it is not a learned state.
    let extreme = SgdState64 {
        w: Vector64::from_slice(&[f64::MAX]),
    };
    let extreme_bits = bits64(&extreme.w);
    assert_eq!(
        model
            .predict(&extreme, &Vector64::from_slice(&[2.0]))
            .unwrap_err(),
        SgdError64::NonFiniteOutput
    );
    assert_eq!(bits64(&extreme.w), extreme_bits);

    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let state = model.initial_state();
    let state_bits = bits64(&state.w);
    assert_eq!(
        model
            .predict(&state, &Vector64::from_slice(&[f64::NAN]))
            .unwrap_err(),
        SgdError64::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(bits64(&state.w), state_bits);

    let snapshot = model.to_serving_f32(&state).expect("snapshot");
    let snapshot_bits = bits(snapshot.weights());
    assert_eq!(
        snapshot
            .predict(&Vector::from_slice(&[f32::NAN]))
            .unwrap_err(),
        SgdError::DimensionMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(bits(snapshot.weights()), snapshot_bits);
}
