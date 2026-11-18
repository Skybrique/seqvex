//! Shared f32 scaled benchmark fixture generation for the two SGD bench targets.
//!
//! The seed, LCG advance, reciprocal-sqrt conversion and f32 grouping are frozen
//! by the existing expected manifests.

use super::common::Lcg;

pub(crate) fn scaled_features32(d: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let inv = (1.0f64 / (d as f64).sqrt()) as f32;
    let mut lcg = Lcg::new(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    let raw = lcg.next_f32();
                    (2.0f32 * raw) * inv
                })
                .collect()
        })
        .collect()
}
