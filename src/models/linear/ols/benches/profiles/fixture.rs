use sha2::{Digest, Sha256};

use crate::common;
use seqvex::foundation::numerical::{Vector, Vector64};

pub(super) const PROBE_COUNT: usize = 128;
pub(super) const FIXTURE_LABEL: &str = "dense_q32_residual_v1";
const FIXTURE_MAGIC: &[u8] = b"SEQVEX-OLS-PROFILE-M1\0";
pub(super) const CELLS: [(usize, usize); 3] = [(1024, 8), (16384, 8), (1024, 128)];

pub(super) struct Fixture {
    pub(super) n: usize,
    pub(super) d: usize,
    pub(super) all_features: Vec<Vec<f32>>,
    pub(super) all_targets: Vec<f32>,
    pub(super) features32: Vec<Vector>,
    pub(super) targets32: Vec<f32>,
    pub(super) features64: Vec<Vector64>,
    pub(super) targets64: Vec<f64>,
    pub(super) probes32: Vec<Vector>,
    pub(super) probes64: Vec<Vector64>,
    pub(super) selection: Vec<usize>,
}

pub(super) fn widen(row: &[f32]) -> Vector64 {
    let values: Vec<f64> = row.iter().map(|&value| f64::from(value)).collect();
    Vector64::from_slice(&values)
}

pub(super) fn build_fixture(n: usize, d: usize) -> Fixture {
    let all_features = common::persistent_excitation_features(d, n + PROBE_COUNT);
    let weights = common::fixed_weights(d);
    let mut all_targets = common::linear_targets(&all_features, &weights);
    for (index, target) in all_targets.iter_mut().enumerate() {
        *target += 1.0e-3 * ((index as f32) * 0.01).sin();
    }

    let features32: Vec<Vector> = all_features[..n]
        .iter()
        .map(|row| Vector::from_slice(row))
        .collect();
    let targets32 = all_targets[..n].to_vec();
    let features64: Vec<Vector64> = all_features[..n].iter().map(|row| widen(row)).collect();
    let targets64 = all_targets[..n]
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    let probes32: Vec<Vector> = all_features[n..]
        .iter()
        .map(|row| Vector::from_slice(row))
        .collect();
    let probes64: Vec<Vector64> = all_features[n..].iter().map(|row| widen(row)).collect();

    Fixture {
        n,
        d,
        all_features,
        all_targets,
        features32,
        targets32,
        features64,
        targets64,
        probes32,
        probes64,
        selection: (0..n).collect(),
    }
}

pub(super) fn fixture_key(fixture: &Fixture) -> String {
    format!("(N={},d={})", fixture.n, fixture.d)
}

pub(super) fn fixture_digest(fixture: &Fixture) -> String {
    let mut hasher = Sha256::new();
    hasher.update(FIXTURE_MAGIC);
    hasher.update(1_u16.to_le_bytes());
    hasher.update((fixture.n as u32).to_le_bytes());
    hasher.update((fixture.d as u32).to_le_bytes());
    hasher.update((PROBE_COUNT as u32).to_le_bytes());
    hasher.update(((fixture.n + PROBE_COUNT) as u32).to_le_bytes());
    hasher.update(common::EXCITATION_SEED.to_le_bytes());
    hasher.update((fixture.selection.len() as u32).to_le_bytes());
    for &index in &fixture.selection {
        hasher.update((index as u32).to_le_bytes());
    }
    hasher.update((PROBE_COUNT as u32).to_le_bytes());
    for index in fixture.n..(fixture.n + PROBE_COUNT) {
        hasher.update((index as u32).to_le_bytes());
    }
    for index in 0..(fixture.n + PROBE_COUNT) {
        hasher.update((index as u32).to_le_bytes());
        for &value in &fixture.all_features[index] {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(fixture.all_targets[index].to_bits().to_le_bytes());
        for &value in &fixture.all_features[index] {
            hasher.update(f64::from(value).to_bits().to_le_bytes());
        }
        hasher.update(
            f64::from(fixture.all_targets[index])
                .to_bits()
                .to_le_bytes(),
        );
    }
    format!("{:x}", hasher.finalize())
}
