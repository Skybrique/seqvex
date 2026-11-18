//! Shared precision-profile helpers.

pub(super) use crate::support::{bits64, sample64};

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
