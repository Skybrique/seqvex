//! Supplied-parameter linear-regression prediction.
//!
//! Demonstrates the **implemented** predictor only: construction from
//! caller-supplied coefficients and intercept, multi-feature single-value
//! prediction, ordered streaming through the existing execution API,
//! caller-bounded grouped prediction, and the existing construction/input
//! errors.
//!
//! The coefficients below are supplied by the caller. They are **not** an OLS
//! fit. Fitting, diagnostics and refitting are planned under #23/#43 and are
//! not implemented.
//!
//! Run with:
//!
//! ```text
//! cargo run --example ols_predict
//! ```
//!
//! Feature order is coefficient order: for `w = [w0, w1, w2]` and
//! `x = [x0, x1, x2]`, the prediction is
//! `y = w0*x0 + w1*x1 + w2*x2 + b`, accumulated in feature-index order in `f32`.
//!
//! Expected output (exact values; `f32` arithmetic on representable inputs):
//!
//! ```text
//! weights = [2.0, -1.0, 0.5]
//! bias    = 1.25
//! predict([1, 3, 4])      = 2.25
//! StateModel::update      = 2.25
//! stream  = [2.25, 1.25, -2.75]
//! grouped = [2.25, 1.25, -2.75]
//! empty group             = 0 predictions
//! empty weights           -> ZeroDimension
//! non-finite weight       -> NonFiniteParameter
//! wrong width             -> DimensionMismatch { expected: 3, actual: 1 }
//! non-finite feature      -> NonFiniteInput
//! overflow envelope       = inf (documented f32 behaviour; output is not checked)
//! ```

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::ols::{LinearRegression, RegressionError};

fn observation(features: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(features))
}

fn main() -> Result<(), RegressionError> {
    // 1. Construction from caller-supplied coefficients and intercept.
    //    weights = [2.0, -1.0, 0.5], intercept = 1.25.
    let model = LinearRegression::new(Vector::from_slice(&[2.0, -1.0, 0.5]), 1.25)?;
    println!("weights = {:?}", model.weights().as_slice());
    println!("bias    = {}", model.bias());

    // 2. Multiple input features -> one scalar prediction.
    //    2*1 + (-1)*3 + 0.5*4 + 1.25 = 2.25
    let prediction = model.predict(&Vector::from_slice(&[1.0, 3.0, 4.0]))?;
    println!("predict([1, 3, 4])      = {prediction}");

    // 3. Single-observation prediction through the existing StateModel.
    let single = model.update(&0.0, &observation(&[1.0, 3.0, 4.0]))?;
    println!("StateModel::update      = {single}");

    // 4. Ordered streaming through the existing execution API.
    //    The executor carries the most recent prediction as its state; it
    //    does not learn or refit coefficients.
    let stream_model = LinearRegression::new(Vector::from_slice(&[2.0, -1.0, 0.5]), 1.25)?;
    let mut executor = StreamingExecutor::new(&stream_model, 0.0);
    let streamed = [[1.0_f32, 3.0, 4.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 2.0]]
        .map(|features| *executor.process_one(&observation(&features)).unwrap());
    println!("stream  = {streamed:?}");

    // 5. Caller-bounded grouped prediction through the existing model API.
    //    Outputs are returned in input order; the caller bounds the group.
    let group = [
        observation(&[1.0, 3.0, 4.0]),
        observation(&[0.0, 0.0, 0.0]),
        observation(&[-2.0, 1.0, 2.0]),
    ];
    let grouped = model.predict_batch(&group)?;
    println!("grouped = {grouped:?}");

    // An empty group returns an empty output vector.
    let empty = model.predict_batch(&[])?;
    println!("empty group             = {} predictions", empty.len());

    // 6. Existing construction and input errors, in the documented order.
    match LinearRegression::new(Vector::zeros(0), 0.0) {
        Err(RegressionError::ZeroDimension) => println!("empty weights           -> ZeroDimension"),
        other => println!("empty weights           -> unexpected {other:?}"),
    }
    match LinearRegression::new(Vector::from_slice(&[1.0, f32::NAN]), 0.0) {
        Err(RegressionError::NonFiniteParameter) => {
            println!("non-finite weight       -> NonFiniteParameter");
        }
        other => println!("non-finite weight       -> unexpected {other:?}"),
    }
    match model.predict(&Vector::from_slice(&[1.0])) {
        Err(RegressionError::DimensionMismatch { expected, actual }) => {
            println!(
                "wrong width             -> DimensionMismatch {{ expected: {expected}, actual: {actual} }}"
            );
        }
        other => println!("wrong width             -> unexpected {other:?}"),
    }
    match model.predict(&Vector::from_slice(&[1.0, f32::NAN, 4.0])) {
        Err(RegressionError::NonFiniteInput) => {
            println!("non-finite feature      -> NonFiniteInput")
        }
        other => println!("non-finite feature      -> unexpected {other:?}"),
    }

    // The documented f32 envelope: finite coefficients and finite inputs do not
    // guarantee a finite output, and prediction does not check its output.
    let overflow_model = LinearRegression::new(Vector::from_slice(&[1.0e20]), 0.0)?;
    let overflow = overflow_model.predict(&Vector::from_slice(&[1.0e20]))?;
    println!(
        "overflow envelope       = {overflow} (documented f32 behaviour; output is not checked)"
    );

    Ok(())
}
