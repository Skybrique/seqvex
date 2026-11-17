//! Deterministic benchmark fixtures, state helpers and observation builders.
//!
//! Fixture arithmetic, seeds and draw order are preserved from the pre-split
//! benchmark. These helpers are shared by measurement, diagnostics and
//! validation; no production code is involved.

use seqvex::foundation::numerical::{Matrix, Matrix64, Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64, RlsState, RlsState64};

use crate::DELTA;
use crate::common;
use crate::mode::{horizon, set_unexpected_failure, warmup};

pub(crate) fn observations32(d: usize) -> Vec<Observation<RlsSample>> {
    let n = warmup() + horizon();
    let xs = common::persistent_excitation_features(d, n);
    let weights = common::fixed_weights(d);
    let ys = common::linear_targets(&xs, &weights);
    xs.iter()
        .zip(ys)
        .map(|(x, target)| {
            Observation::new(RlsSample {
                features: Vector::from_slice(x),
                target,
            })
        })
        .collect()
}

pub(crate) fn observations64(d: usize) -> Vec<Observation<RlsSample64>> {
    let n = warmup() + horizon();
    let mut lcg = common::Lcg::new(common::EXCITATION_SEED);
    let weights: Vec<f64> = (0..d)
        .map(|index| (index as f64 * 0.9).sin() + 0.3)
        .collect();
    (0..n)
        .map(|_| {
            let features: Vec<f64> = (0..d).map(|_| 2.0 * lcg.next_f64()).collect();
            let target = features
                .iter()
                .zip(&weights)
                .map(|(value, weight)| value * weight)
                .sum();
            Observation::new(RlsSample64 {
                features: Vector64::from_slice(&features),
                target,
            })
        })
        .collect()
}

pub(crate) fn fingerprint32(state: &RlsState) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in state.w.as_slice().iter().chain(state.p.as_slice().iter()) {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(crate) fn fingerprint64(state: &RlsState64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in state
        .weights()
        .as_slice()
        .iter()
        .chain(state.covariance().as_slice().iter())
    {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(crate) fn state_eq32(left: &RlsState, right: &RlsState) -> bool {
    left.w.as_slice().len() == right.w.as_slice().len()
        && left
            .w
            .as_slice()
            .iter()
            .zip(right.w.as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left.p.as_slice().len() == right.p.as_slice().len()
        && left
            .p
            .as_slice()
            .iter()
            .zip(right.p.as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

pub(crate) fn state_eq64(left: &RlsState64, right: &RlsState64) -> bool {
    left.weights().as_slice().len() == right.weights().as_slice().len()
        && left
            .weights()
            .as_slice()
            .iter()
            .zip(right.weights().as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left.covariance().as_slice().len() == right.covariance().as_slice().len()
        && left
            .covariance()
            .as_slice()
            .iter()
            .zip(right.covariance().as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

pub(crate) fn symmetric32(matrix: &Matrix) -> bool {
    let d = matrix.rows();
    matrix.cols() == d
        && (0..d).all(|row| {
            (0..d).all(|col| {
                matrix.as_slice()[row * d + col].to_bits()
                    == matrix.as_slice()[col * d + row].to_bits()
            })
        })
}

pub(crate) fn symmetric64(matrix: &Matrix64) -> bool {
    let d = matrix.rows();
    matrix.cols() == d
        && (0..d).all(|row| {
            (0..d).all(|col| {
                matrix.as_slice()[row * d + col].to_bits()
                    == matrix.as_slice()[col * d + row].to_bits()
            })
        })
}

pub(crate) fn first_divergence32(left: &RlsState, right: &RlsState) -> Option<String> {
    for (index, (a, b)) in left.w.as_slice().iter().zip(right.w.as_slice()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "w[{index}] left={:#010x} right={:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    for (index, (a, b)) in left.p.as_slice().iter().zip(right.p.as_slice()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "p[{index}] left={:#010x} right={:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    None
}

pub(crate) fn first_divergence64(left: &RlsState64, right: &RlsState64) -> Option<String> {
    for (index, (a, b)) in left
        .weights()
        .as_slice()
        .iter()
        .zip(right.weights().as_slice())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "w[{index}] left={:#018x} right={:#018x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    for (index, (a, b)) in left
        .covariance()
        .as_slice()
        .iter()
        .zip(right.covariance().as_slice())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Some(format!(
                "p[{index}] left={:#018x} right={:#018x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    None
}

pub(crate) fn postcondition_divergence32(state: &RlsState, expected: &RlsState) -> Option<String> {
    if state_eq32(state, expected) {
        None
    } else {
        first_divergence32(state, expected)
    }
}

pub(crate) fn postcondition_divergence64(
    state: &RlsState64,
    expected: &RlsState64,
) -> Option<String> {
    if state_eq64(state, expected) {
        None
    } else {
        first_divergence64(state, expected)
    }
}

/// Diagnostic-only controlled mismatch: flips the lowest weight bit so the
/// bitwise comparison fails while both trajectories remain finite.
pub(crate) fn one_bit_mismatch(state: &RlsState) -> RlsState {
    let mut weights = state.w.as_slice().to_vec();
    if let Some(first) = weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    RlsState {
        w: Vector::from_slice(&weights),
        p: state.p.clone(),
    }
}

pub(crate) fn make_model32(d: usize, lambda: f32, delta: f32) -> Option<Rls> {
    match Rls::new(d, lambda, delta) {
        Ok(model) => Some(model),
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL Rls::new {error:?}");
            set_unexpected_failure();
            None
        }
    }
}

pub(crate) fn make_model64(d: usize, lambda: f64, delta: f64) -> Option<Rls64> {
    match Rls64::new(d, lambda, delta) {
        Ok(model) => Some(model),
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL Rls64::new {error:?}");
            set_unexpected_failure();
            None
        }
    }
}

pub(crate) fn sample32(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

pub(crate) fn sample64(features: &[f64], target: f64) -> Observation<RlsSample64> {
    Observation::new(RlsSample64 {
        features: Vector64::from_slice(features),
        target,
    })
}

pub(crate) fn invalid_state32() -> RlsState {
    RlsState {
        w: Vector::zeros(2),
        p: Matrix::from_fn(2, 2, |row, col| {
            if row == col {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            }
        }),
    }
}

pub(crate) fn invalid_state64() -> Option<RlsState64> {
    match RlsState64::admit(
        Vector64::zeros(2),
        Matrix64::from_fn(2, 2, |row, col| {
            if row == col {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            }
        }),
    ) {
        Ok(state) => Some(state),
        Err(error) => {
            println!("DIAGNOSTIC setup FAIL invalid_state64 {error:?}");
            set_unexpected_failure();
            None
        }
    }
}

pub(crate) fn legacy_model(features: usize, lambda: f32) -> Rls {
    match Rls::new(features, lambda, DELTA) {
        Ok(model) => model,
        Err(error) => {
            println!("FAILURE legacy model construction {error:?}");
            set_unexpected_failure();
            std::process::exit(1);
        }
    }
}

pub(crate) fn control_observation(features: usize) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_fn(features, |index| (index as f32 * 0.31).sin()),
        target: 0.25,
    })
}

pub(crate) fn workload_observations(xs: &[Vec<f32>], ys: &[f32]) -> Vec<Observation<RlsSample>> {
    xs.iter()
        .zip(ys)
        .map(|(x, &target)| {
            Observation::new(RlsSample {
                features: Vector::from_slice(x),
                target,
            })
        })
        .collect()
}
