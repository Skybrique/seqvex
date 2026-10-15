//! RLS learning example (self-contained, synthetic, deterministic).
//!
//! Run with `cargo run --example rls_learn`.
//!
//! Learns from an explicitly synthetic prefix through public `update` calls,
//! then scores subsequent rows prequentially with `Rls::predict_checked`
//! (predict from the committed state before revealing the target and updating).
//! The known coefficients only generate the synthetic targets; they are never
//! installed into the model as learned weights. No external files or network.

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{Rls, RlsSample};

/// Known coefficients, used only to generate synthetic targets. The targets
/// switch from regime A to regime B at zero-based index `SPLIT`.
const W_A: [f32; 2] = [0.7, -1.3];
const W_B: [f32; 2] = [-0.5, 1.1];
const DIMENSION: usize = W_A.len();
/// Zero-based row `SPLIT` is the first scored row (learn rows are `0..SPLIT`).
const SPLIT: usize = 400;
const TOTAL: usize = 1400;
const DELTA: f32 = 1.0e3;
const LAMBDA: f32 = 0.95;

fn synthetic_row(index: usize) -> ([f32; 2], f32) {
    let features = [
        ((index + 1) as f32 * 0.31).sin(),
        ((index + 1) as f32 * 0.53 + 0.7).sin(),
    ];
    let weights = if index < SPLIT { W_A } else { W_B };
    let target = features[0] * weights[0] + features[1] * weights[1];
    (features, target)
}

fn sample(features: [f32; 2], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(&features),
        target,
    })
}

fn main() {
    let model = Rls::new(DIMENSION, LAMBDA, DELTA).expect("valid configuration");
    let mut state = model.initial_state();

    // Learn the synthetic prefix through exactly `SPLIT` successful updates.
    for index in 0..SPLIT {
        let (features, target) = synthetic_row(index);
        state = model
            .update(&state, &sample(features, target))
            .unwrap_or_else(|error| panic!("learn update failed at row {index}: {error:?}"));
    }
    let learned = state.w.as_slice().to_vec();

    // Prequential scoring: predict (checked) from the committed state, record
    // the error, reveal the target, then update.
    let scored = TOTAL - SPLIT;
    let mut errors: Vec<f64> = Vec::with_capacity(scored);
    let mut tail: Vec<(usize, f32, f32)> = Vec::new();
    for index in SPLIT..TOTAL {
        let (features, target) = synthetic_row(index);
        let prediction = model
            .predict_checked(&state, &Vector::from_slice(&features))
            .unwrap_or_else(|error| panic!("checked prediction failed at row {index}: {error:?}"));
        errors.push((f64::from(target) - f64::from(prediction)).abs());
        state = model
            .update(&state, &sample(features, target))
            .unwrap_or_else(|error| panic!("scoring update failed at row {index}: {error:?}"));
        if index + 3 > TOTAL {
            tail.push((index, target, prediction));
        }
    }

    let mae = |window: &[f64]| window.iter().sum::<f64>() / window.len() as f64;
    let mae_all = mae(&errors);
    let mae_early = mae(&errors[..20]);
    let mae_late = mae(&errors[scored - 100..]);

    println!("RLS learning example (synthetic, deterministic)");
    println!("dimension={DIMENSION} lambda={LAMBDA} delta={DELTA}");
    println!(
        "regime switch at index {SPLIT}: A {W_A:?} -> B {W_B:?} (coefficients generate targets only)"
    );
    println!("learn rows 0..{SPLIT} ({SPLIT} updates); scored rows {SPLIT}..{TOTAL}");
    println!("learned weights: {learned:?}");
    println!("scored rows: {scored}");
    println!("MAE all scored rows: {mae_all:.6e}");
    println!("MAE first 20 scored rows: {mae_early:.6e}");
    println!("MAE last 100 scored rows: {mae_late:.6e}");
    for (index, target, prediction) in tail {
        println!("row {index}: target={target:.6} checked_prediction={prediction:.6}");
    }
}
