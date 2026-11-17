//! Shared precision-profile helpers.

use seqvex::foundation::numerical::Vector64;
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::sgd::SgdSample64;

pub(super) fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
    Observation::new(SgdSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

pub(super) fn bits64(vector: &Vector64) -> Vec<u64> {
    vector
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

pub(super) fn reference_update64(eta: f64, w: &[f64], x: &[f64], y: f64) -> Vec<f64> {
    let dot: f64 = w.iter().zip(x).map(|(w, x)| w * x).sum();
    let error = y - dot;
    let scale = eta * error;
    x.iter().zip(w).map(|(x, w)| w + x * scale).collect()
}

pub(super) fn reference_update32_sum(eta: f32, w: &[f32], x: &[f32], y: f32) -> Vec<f32> {
    let dot: f32 = w.iter().zip(x).map(|(w, x)| w * x).sum();
    let error = y - dot;
    let scale = eta * error;
    x.iter().zip(w).map(|(x, w)| w + x * scale).collect()
}
