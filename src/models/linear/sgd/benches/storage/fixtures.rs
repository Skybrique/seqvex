//! Deterministic storage fixtures and manifests.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdSample, SgdSample64, SgdState, SgdState64};
use sha2::{Digest, Sha256};

use super::{FIXTURE_LEN, SEED32, SEED64, WARM_START, common};

fn scaled_features32(d: usize, n: usize) -> Vec<Vec<f32>> {
    let inv = (1.0f64 / (d as f64).sqrt()) as f32;
    let mut lcg = common::Lcg::new(SEED32);
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

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn features64(d: usize, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let inv = 1.0 / (d as f64).sqrt();
    let mut rng = SplitMix64::new(SEED64);
    let weights: Vec<f64> = (0..d).map(|j| (j as f64) * 0.03125 - 0.5).collect();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..d).map(|_| (2.0 * rng.unit() - 1.0) * inv).collect();
        let mut y = 0.0f64;
        for j in 0..d {
            y += weights[j] * row[j];
        }
        xs.push(row);
        ys.push(y);
    }
    (xs, ys)
}

pub(super) fn observations32(d: usize) -> Vec<Observation<SgdSample>> {
    let xs = scaled_features32(d, FIXTURE_LEN);
    let weights = common::fixed_weights(d);
    let ys = common::linear_targets(&xs, &weights);
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            Observation::new(SgdSample {
                features: Vector::from_slice(x),
                target: y,
            })
        })
        .collect()
}

pub(super) fn observations64(d: usize) -> Vec<Observation<SgdSample64>> {
    let (xs, ys) = features64(d, FIXTURE_LEN);
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            Observation::new(SgdSample64 {
                features: Vector64::from_slice(x),
                target: y,
            })
        })
        .collect()
}

pub(super) fn warm32(model: &Sgd, observations: &[Observation<SgdSample>]) -> SgdState {
    let mut state = model.initial_state();
    for observation in observations.iter().take(WARM_START) {
        state = model.update(&state, observation).expect("f32 warm state");
    }
    state
}

pub(super) fn warm64(model: &Sgd64, observations: &[Observation<SgdSample64>]) -> SgdState64 {
    let mut state = model.initial_state();
    for observation in observations.iter().take(WARM_START) {
        state = model.update(&state, observation).expect("f64 warm state");
    }
    state
}

pub(super) fn expected_manifest32(d: usize) -> &'static str {
    match d {
        3 => "b60241349e4712850ed90afd8e76d591bd182a5ecd265250993d689bc5b5b1be",
        6 => "d8315835e6f4e62a8c297ff78b4bd3c8f5978126f66f5a4ebbe684a3b57c0c05",
        7 => "bccd1c1d923f40ba1a395f72289b765761eddb2da526c6bd7b7fa940f9cced04",
        8 => "4e9049b1b4eaa13b2784d43ad417ebbb2dd7920926e3ee214d196bcbf524978c",
        9 => "11f29b4d1c194f93ebbed33647f77c3c5ae9bf23a21a6721498701d1ef8d6c93",
        32 => "c95e54ff42b6bc28cf6dceffd19c516d9f0ed111d31e273d4c78542b38f0a35b",
        128 => "b17f1f906bad1f8b52b6ffcdc07923844b51c2ef5153fc3f583718193a07e640",
        _ => panic!("unregistered dimension"),
    }
}

pub(super) fn expected_manifest64(d: usize) -> &'static str {
    match d {
        3 => "ad844b1b15df7bce16d10d8f8bfd24e39a360fd1a5f28535b28cb14caa64341a",
        6 => "c928a207edfbda8d9073d7bf2ffcfb5f55098956ac926e4c33becd55ecdff52f",
        7 => "ae612213c9a50444f4a8c05cf1bce6c7705f165098039a55c3f029321f807f45",
        8 => "954000ba6d6c7043f9aafb72d58e5daf63f62c8c71e7e77007c4b3e24ecb5134",
        9 => "09a96be70231208895f0350ef856d37dff199cce533143e64245a8c3fa7f0b57",
        32 => "91501f0049d400d41733c1275103b8f271c38ded816760020c4580090f28ef3c",
        128 => "c8396692a6671cab0f29d913f70a3cf16e8626ab43c9e9d00db7c9ad32dabc05",
        _ => panic!("unregistered dimension"),
    }
}

pub(super) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String");
    }
    out
}

pub(super) fn manifest32(
    d: usize,
    observations: &[Observation<SgdSample>],
    warm: &SgdState,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"f32");
    hasher.update((d as u64).to_le_bytes());
    hasher.update((warm.w.len() as u64).to_le_bytes());
    for value in warm.w.as_slice() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for (index, observation) in observations.iter().enumerate() {
        hasher.update((index as u64).to_le_bytes());
        for value in observation.value().features.as_slice() {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(observation.value().target.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

pub(super) fn manifest64(
    d: usize,
    observations: &[Observation<SgdSample64>],
    warm: &SgdState64,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"f64");
    hasher.update((d as u64).to_le_bytes());
    hasher.update((warm.w.len() as u64).to_le_bytes());
    for value in warm.w.as_slice() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for (index, observation) in observations.iter().enumerate() {
        hasher.update((index as u64).to_le_bytes());
        for value in observation.value().features.as_slice() {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(observation.value().target.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}
