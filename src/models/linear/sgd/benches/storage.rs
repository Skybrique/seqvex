//! Experimental SGD temporary/storage controls for the reviewed design plan.
//! Private to this benchmark; production learners are unchanged.

use std::hint::black_box;
use std::time::Instant;

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdSample, SgdSample64, SgdState, SgdState64,
};
use sha2::{Digest, Sha256};
use tinyvec::TinyVec;

#[allow(dead_code)]
#[path = "../../../../../benches/common/mod.rs"]
mod common;

const FIXTURE_LEN: usize = 2048;
const WARM_START: usize = 512;
const ETA32: f32 = 0.05;
const ETA64: f64 = 0.05;
const SEED32: u64 = 0x5eed_55d0_0000_0001;
const SEED64: u64 = 0x5eed_55d0_0000_0002;
const DIMS: [usize; 7] = [3, 6, 7, 8, 9, 32, 128];
const DETACHED_DIMS: [usize; 3] = [6, 8, 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Baseline,
    Direct,
    Array,
    Reuse,
    Tinyvec,
}

impl Candidate {
    const ALL: [Self; 5] = [
        Self::Baseline,
        Self::Direct,
        Self::Array,
        Self::Reuse,
        Self::Tinyvec,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Direct => "direct",
            Self::Array => "array",
            Self::Reuse => "reuse",
            Self::Tinyvec => "tinyvec",
        }
    }

    fn applicable(self, d: usize) -> bool {
        self != Self::Array || d <= 8
    }
}

fn capture_index() -> usize {
    let capture = std::env::var("SEQVEX_CAPTURE")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("SEQVEX_CAPTURE must be 1, 2, or 3")
        })
        .unwrap_or(1);
    assert!(
        (1..=3).contains(&capture),
        "SEQVEX_CAPTURE must be 1, 2, or 3"
    );
    capture
}

fn ordered_candidates() -> Vec<Candidate> {
    let mut rest: Vec<Candidate> = Candidate::ALL[1..].to_vec();
    if capture_index() == 2 {
        rest.reverse();
    }
    let mut ordered = Vec::with_capacity(Candidate::ALL.len());
    ordered.push(Candidate::Baseline);
    ordered.extend(rest);
    ordered
}

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

fn observations32(d: usize) -> Vec<Observation<SgdSample>> {
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

fn observations64(d: usize) -> Vec<Observation<SgdSample64>> {
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

fn warm32(model: &Sgd, observations: &[Observation<SgdSample>]) -> SgdState {
    let mut state = model.initial_state();
    for observation in observations.iter().take(WARM_START) {
        state = model.update(&state, observation).expect("f32 warm state");
    }
    state
}

fn warm64(model: &Sgd64, observations: &[Observation<SgdSample64>]) -> SgdState64 {
    let mut state = model.initial_state();
    for observation in observations.iter().take(WARM_START) {
        state = model.update(&state, observation).expect("f64 warm state");
    }
    state
}

fn check_shape32(model: &Sgd, state: &SgdState, features: &Vector) -> Result<(), SgdError> {
    if features.len() != model.dimension() {
        return Err(SgdError::DimensionMismatch {
            expected: model.dimension(),
            actual: features.len(),
        });
    }
    if state.w.len() != model.dimension() {
        return Err(SgdError::DimensionMismatch {
            expected: model.dimension(),
            actual: state.w.len(),
        });
    }
    Ok(())
}

fn update32(
    candidate: Candidate,
    model: &Sgd,
    state: &SgdState,
    observation: &Observation<SgdSample>,
    scratch: &mut Vec<f32>,
) -> Result<SgdState, SgdError> {
    if candidate == Candidate::Baseline {
        return model.update(state, observation);
    }
    let sample = observation.value();
    check_shape32(model, state, &sample.features)?;
    if !sample.features.as_slice().iter().all(|v| v.is_finite()) || !sample.target.is_finite() {
        return Err(SgdError::NonFiniteInput);
    }
    let error = sample.target - state.w.dot(&sample.features)?;
    let scale = model.learning_rate() * error;
    let next_w = match candidate {
        Candidate::Direct => Vector::from_fn(model.dimension(), |j| {
            state.w.as_slice()[j] + sample.features.as_slice()[j] * scale
        }),
        Candidate::Array => {
            let mut temp = [0.0f32; 8];
            for (j, cell) in temp.iter_mut().enumerate().take(model.dimension()) {
                *cell = sample.features.as_slice()[j] * scale;
            }
            Vector::from_fn(model.dimension(), |j| state.w.as_slice()[j] + temp[j])
        }
        Candidate::Reuse => {
            scratch.clear();
            scratch.extend(sample.features.as_slice().iter().map(|v| *v * scale));
            Vector::from_fn(model.dimension(), |j| state.w.as_slice()[j] + scratch[j])
        }
        Candidate::Tinyvec => {
            let mut temp: TinyVec<[f32; 8]> = TinyVec::new();
            for value in sample.features.as_slice() {
                temp.push(*value * scale);
            }
            Vector::from_fn(model.dimension(), |j| state.w.as_slice()[j] + temp[j])
        }
        Candidate::Baseline => unreachable!(),
    };
    if !next_w.as_slice().iter().all(|v| v.is_finite()) {
        return Err(SgdError::NonFiniteCandidate);
    }
    Ok(SgdState { w: next_w })
}

fn check_shape64(model: &Sgd64, state: &SgdState64, features: &Vector64) -> Result<(), SgdError64> {
    if features.len() != model.dimension() {
        return Err(SgdError64::DimensionMismatch {
            expected: model.dimension(),
            actual: features.len(),
        });
    }
    if state.w.len() != model.dimension() {
        return Err(SgdError64::DimensionMismatch {
            expected: model.dimension(),
            actual: state.w.len(),
        });
    }
    Ok(())
}

fn update64(
    candidate: Candidate,
    model: &Sgd64,
    state: &SgdState64,
    observation: &Observation<SgdSample64>,
    scratch: &mut Vec<f64>,
) -> Result<SgdState64, SgdError64> {
    if candidate == Candidate::Baseline {
        return model.update(state, observation);
    }
    let sample = observation.value();
    check_shape64(model, state, &sample.features)?;
    if !sample.features.as_slice().iter().all(|v| v.is_finite()) || !sample.target.is_finite() {
        return Err(SgdError64::NonFiniteInput);
    }
    let error = sample.target - state.w.dot(&sample.features)?;
    let scale = model.learning_rate() * error;
    let next_w = match candidate {
        Candidate::Direct => Vector64::from_fn(model.dimension(), |j| {
            state.w.as_slice()[j] + sample.features.as_slice()[j] * scale
        }),
        Candidate::Array => {
            let mut temp = [0.0f64; 8];
            for (j, cell) in temp.iter_mut().enumerate().take(model.dimension()) {
                *cell = sample.features.as_slice()[j] * scale;
            }
            Vector64::from_fn(model.dimension(), |j| state.w.as_slice()[j] + temp[j])
        }
        Candidate::Reuse => {
            scratch.clear();
            scratch.extend(sample.features.as_slice().iter().map(|v| *v * scale));
            Vector64::from_fn(model.dimension(), |j| state.w.as_slice()[j] + scratch[j])
        }
        Candidate::Tinyvec => {
            let mut temp: TinyVec<[f64; 8]> = TinyVec::new();
            for value in sample.features.as_slice() {
                temp.push(*value * scale);
            }
            Vector64::from_fn(model.dimension(), |j| state.w.as_slice()[j] + temp[j])
        }
        Candidate::Baseline => unreachable!(),
    };
    if !next_w.as_slice().iter().all(|v| v.is_finite()) {
        return Err(SgdError64::NonFiniteCandidate);
    }
    Ok(SgdState64 { w: next_w })
}

fn states_equal32(a: &SgdState, b: &SgdState) -> bool {
    a.w.len() == b.w.len()
        && a.w
            .as_slice()
            .iter()
            .zip(b.w.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn states_equal64(a: &SgdState64, b: &SgdState64) -> bool {
    a.w.len() == b.w.len()
        && a.w
            .as_slice()
            .iter()
            .zip(b.w.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn guard_storage() {
    for d in DIMS {
        let observations = observations32(d);
        let model = Sgd::new(d, ETA32).unwrap();
        let warm = warm32(&model, &observations);
        let manifest = hex(&manifest32(d, &observations, &warm));
        assert_eq!(
            manifest,
            expected_manifest32(d),
            "f32 manifest mismatch D={d}"
        );
        println!("manifest f32 D={d} {manifest}");
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            for sequence in [32usize, 256] {
                let mut reference = warm.clone();
                let mut state = warm.clone();
                for observation in observations.iter().skip(WARM_START).take(sequence) {
                    reference = model.update(&reference, observation).unwrap();
                    state = update32(candidate, &model, &state, observation, &mut scratch).unwrap();
                    assert!(
                        states_equal32(&reference, &state),
                        "f32 {} D={d} sequence={sequence} bit mismatch",
                        candidate.name()
                    );
                }
                let final_index = WARM_START + sequence;
                let final_features = &observations[final_index].value().features;
                let reference_prediction = model.predict(&reference, final_features).unwrap();
                let candidate_prediction = model.predict(&state, final_features).unwrap();
                assert!(candidate_prediction.is_finite());
                assert_eq!(
                    candidate_prediction.to_bits(),
                    reference_prediction.to_bits(),
                    "f32 {} D={d} sequence={sequence} final prediction mismatch",
                    candidate.name()
                );
            }
        }
    }
    for d in DIMS {
        let observations = observations64(d);
        let model = Sgd64::new(d, ETA64).unwrap();
        let warm = warm64(&model, &observations);
        let manifest = hex(&manifest64(d, &observations, &warm));
        assert_eq!(
            manifest,
            expected_manifest64(d),
            "f64 manifest mismatch D={d}"
        );
        println!("manifest f64 D={d} {manifest}");
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            for sequence in [32usize, 256] {
                let mut reference = warm.clone();
                let mut state = warm.clone();
                for observation in observations.iter().skip(WARM_START).take(sequence) {
                    reference = model.update(&reference, observation).unwrap();
                    state = update64(candidate, &model, &state, observation, &mut scratch).unwrap();
                    assert!(
                        states_equal64(&reference, &state),
                        "f64 {} D={d} sequence={sequence} bit mismatch",
                        candidate.name()
                    );
                }
                let final_index = WARM_START + sequence;
                let final_features = &observations[final_index].value().features;
                let reference_prediction = model.predict(&reference, final_features).unwrap();
                let candidate_prediction = model.predict(&state, final_features).unwrap();
                assert!(candidate_prediction.is_finite());
                assert_eq!(
                    candidate_prediction.to_bits(),
                    reference_prediction.to_bits(),
                    "f64 {} D={d} sequence={sequence} final prediction mismatch",
                    candidate.name()
                );
            }
        }
    }

    for d in [3usize, 6, 8, 9] {
        let observations = observations32(d);
        let model = Sgd::new(d, ETA32).unwrap();
        let warm = warm32(&model, &observations);
        let sample = observations[WARM_START].value();
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            let state = update32(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut scratch,
            )
            .expect("f32 lifecycle update");
            assert_eq!(state.w.len(), d);
            let cloned = state.clone();
            drop(state);
            assert!(states_equal32(&cloned, &cloned));
            let moved = cloned;
            assert_eq!(moved.w.len(), d);
            drop(moved);
            let _ = sample;
            let empty: TinyVec<[f32; 8]> = TinyVec::new();
            assert!(empty.is_empty());
            let mut rejected_scratch = vec![0.0f32; d];
            let bad = Observation::new(SgdSample {
                features: Vector::from_fn(d, |j| if j == 0 { f32::NAN } else { 1.0 }),
                target: 1.0,
            });
            assert!(update32(candidate, &model, &warm, &bad, &mut rejected_scratch).is_err());
            let retried = update32(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut rejected_scratch,
            )
            .expect("f32 retry");
            let mut fresh = Vec::new();
            let fresh_state = update32(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut fresh,
            )
            .expect("f32 fresh");
            assert!(states_equal32(&retried, &fresh_state));
        }
    }
    for d in [3usize, 6, 8, 9] {
        let observations = observations64(d);
        let model = Sgd64::new(d, ETA64).unwrap();
        let warm = warm64(&model, &observations);
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            let state = update64(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut scratch,
            )
            .expect("f64 lifecycle update");
            assert_eq!(state.w.len(), d);
            let cloned = state.clone();
            drop(state);
            let moved = cloned;
            assert_eq!(moved.w.len(), d);
            drop(moved);
            let empty: TinyVec<[f64; 8]> = TinyVec::new();
            assert!(empty.is_empty());
            let bad = Observation::new(SgdSample64 {
                features: Vector64::from_fn(d, |j| if j == 0 { f64::NAN } else { 1.0 }),
                target: 1.0,
            });
            let mut rejected_scratch = vec![0.0f64; d];
            assert!(update64(candidate, &model, &warm, &bad, &mut rejected_scratch).is_err());
            let retried = update64(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut rejected_scratch,
            )
            .expect("f64 retry");
            let mut fresh = Vec::new();
            let fresh_state = update64(
                candidate,
                &model,
                &warm,
                &observations[WARM_START],
                &mut fresh,
            )
            .expect("f64 fresh");
            assert!(states_equal64(&retried, &fresh_state));
        }
    }

    let mut tiny: TinyVec<[f32; 8]> = TinyVec::new();
    assert!(tiny.is_empty());
    for (index, value) in [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
        tiny.push(*value);
        assert_eq!(tiny.len(), index + 1);
        assert_eq!(tiny[index].to_bits(), value.to_bits());
    }
    tiny.push(7.0);
    assert!(tiny.is_inline());
    tiny.push(8.0);
    assert!(!tiny.is_inline());
    let moved = tiny;
    assert_eq!(moved[8].to_bits(), 8.0f32.to_bits());
    let cloned = moved.clone();
    assert_eq!(cloned.as_slice()[0].to_bits(), 0.0f32.to_bits());
    let mut shrunk = moved;
    shrunk.truncate(3);
    assert!(!shrunk.is_inline());
    shrunk.push(10.0);
    assert_eq!(shrunk[3].to_bits(), 10.0f32.to_bits());
    drop(shrunk);
    assert!(
        DIMS.iter()
            .all(|&d| !Candidate::Array.applicable(d) || d <= 8)
    );
    assert!(!Candidate::Array.applicable(9));
    assert!(!Candidate::Array.applicable(32));
    assert!(!Candidate::Array.applicable(128));

    let model = Sgd::new(2, ETA32).unwrap();
    let state = model.initial_state();
    let sample = Observation::new(SgdSample {
        features: Vector::from_slice(&[1.0, 2.0]),
        target: 3.0,
    });
    let mut scratch = Vec::new();
    let before = state.clone();
    assert!(update32(Candidate::Direct, &model, &state, &sample, &mut scratch).is_ok());
    assert!(states_equal32(&state, &before));
    let bad = Observation::new(SgdSample {
        features: Vector::from_slice(&[f32::NAN, 0.0]),
        target: 1.0,
    });
    assert_eq!(
        update32(Candidate::Reuse, &model, &state, &bad, &mut scratch),
        Err(SgdError::NonFiniteInput)
    );
    assert!(states_equal32(&state, &before));
}

fn expected_manifest32(d: usize) -> &'static str {
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

fn expected_manifest64(d: usize) -> &'static str {
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

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String");
    }
    out
}

fn manifest32(d: usize, observations: &[Observation<SgdSample>], warm: &SgdState) -> [u8; 32] {
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

fn manifest64(d: usize, observations: &[Observation<SgdSample64>], warm: &SgdState64) -> [u8; 32] {
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

fn run_row(id: &str, calls: u32, units_per_call: u64, mut body: impl FnMut()) {
    for _ in 0..2 {
        for _ in 0..calls {
            body();
        }
    }
    let mut raw = Vec::with_capacity(20);
    for window in 0..20 {
        let before = common::allocation_totals();
        let start = Instant::now();
        for _ in 0..calls {
            body();
        }
        let elapsed_ns = start.elapsed().as_nanos();
        let after = common::allocation_totals();
        raw.push((window, elapsed_ns, after.0 - before.0, after.1 - before.1));
    }
    for (window, elapsed_ns, requests, requested_bytes) in &raw {
        println!(
            "{id} window={window} elapsed_ns={elapsed_ns} requests={requests} requested_bytes={requested_bytes} units={units_per_call}"
        );
    }
    let mut elapsed: Vec<f64> = raw
        .iter()
        .map(|(_, elapsed, _, _)| *elapsed as f64 / units_per_call as f64)
        .collect();
    let mut requests: Vec<f64> = raw
        .iter()
        .map(|(_, _, requests, _)| *requests as f64 / units_per_call as f64)
        .collect();
    let mut bytes: Vec<f64> = raw
        .iter()
        .map(|(_, _, _, bytes)| *bytes as f64 / units_per_call as f64)
        .collect();
    elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    requests.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bytes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = |v: &[f64]| v[v.len() / 2];
    let iqr = |v: &[f64]| v[v.len() * 3 / 4] - v[v.len() / 4];
    println!(
        "{id} median={:.1}ns/update iqr={:.1} requests/update={:.3} bytes/update={:.1}",
        median(&elapsed),
        iqr(&elapsed),
        median(&requests),
        median(&bytes)
    );
}

fn memory_probe<T>(id: &str, mut body: impl FnMut() -> T) {
    let before = common::live_bytes();
    common::reset_peak();
    let value = body();
    let live = common::live_bytes();
    let peak = common::peak_bytes();
    black_box(&value);
    drop(value);
    let after = common::live_bytes();
    println!("{id}/mem before={before} live={live} peak={peak} after={after}");
}

fn run_storage_rows() {
    for d in DIMS {
        let observations = observations32(d);
        let model = Sgd::new(d, ETA32).unwrap();
        let warm = warm32(&model, &observations);
        println!(
            "manifest f32 D={d} {:02x?}",
            manifest32(d, &observations, &warm)
        );
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            if DETACHED_DIMS.contains(&d) {
                let mut index = 0usize;
                run_row(
                    &format!("storage/detached/f32/{}/dim={d}", candidate.name()),
                    4096,
                    4096,
                    || {
                        black_box(candidate);
                        black_box(&model);
                        black_box(&warm);
                        let sample_index = WARM_START + (index % (FIXTURE_LEN - WARM_START));
                        index += 1;
                        let observation = black_box(&observations[sample_index]);
                        black_box(
                            update32(candidate, &model, &warm, observation, &mut scratch).unwrap(),
                        );
                    },
                );
            }
            for sequence in [32usize, 256] {
                let calls = if sequence == 32 { 32 } else { 8 };
                let units = u64::from(calls) * sequence as u64;
                run_row(
                    &format!("storage/seq{sequence}/f32/{}/dim={d}", candidate.name()),
                    calls,
                    units,
                    || {
                        black_box(candidate);
                        black_box(&model);
                        black_box(&warm);
                        let observations = black_box(observations.as_slice());
                        let mut state = warm.clone();
                        for observation in observations.iter().skip(WARM_START).take(sequence) {
                            state = update32(candidate, &model, &state, observation, &mut scratch)
                                .unwrap();
                        }
                        // Complete sequence operation includes one final prediction, not another update.
                        let final_index = WARM_START + sequence;
                        let final_observation = black_box(&observations[final_index]);
                        let prediction = model
                            .predict(&state, &final_observation.value().features)
                            .unwrap();
                        black_box(prediction);
                        black_box(&state);
                    },
                );
            }
            memory_probe(
                &format!("storage/seq256/f32/{}/dim={d}", candidate.name()),
                || {
                    let mut state = warm.clone();
                    for observation in observations.iter().skip(WARM_START).take(256) {
                        state =
                            update32(candidate, &model, &state, observation, &mut scratch).unwrap();
                    }
                    state
                },
            );
        }
    }
    for d in DIMS {
        let observations = observations64(d);
        let model = Sgd64::new(d, ETA64).unwrap();
        let warm = warm64(&model, &observations);
        println!(
            "manifest f64 D={d} {:02x?}",
            manifest64(d, &observations, &warm)
        );
        for candidate in ordered_candidates() {
            if !candidate.applicable(d) {
                continue;
            }
            let mut scratch = Vec::new();
            if DETACHED_DIMS.contains(&d) {
                let mut index = 0usize;
                run_row(
                    &format!("storage/detached/f64/{}/dim={d}", candidate.name()),
                    4096,
                    4096,
                    || {
                        black_box(candidate);
                        black_box(&model);
                        black_box(&warm);
                        let sample_index = WARM_START + (index % (FIXTURE_LEN - WARM_START));
                        index += 1;
                        let observation = black_box(&observations[sample_index]);
                        black_box(
                            update64(candidate, &model, &warm, observation, &mut scratch).unwrap(),
                        );
                    },
                );
            }
            for sequence in [32usize, 256] {
                let calls = if sequence == 32 { 32 } else { 8 };
                let units = u64::from(calls) * sequence as u64;
                run_row(
                    &format!("storage/seq{sequence}/f64/{}/dim={d}", candidate.name()),
                    calls,
                    units,
                    || {
                        black_box(candidate);
                        black_box(&model);
                        black_box(&warm);
                        let observations = black_box(observations.as_slice());
                        let mut state = warm.clone();
                        for observation in observations.iter().skip(WARM_START).take(sequence) {
                            state = update64(candidate, &model, &state, observation, &mut scratch)
                                .unwrap();
                        }
                        // Complete sequence operation includes one final prediction, not another update.
                        let final_index = WARM_START + sequence;
                        let final_observation = black_box(&observations[final_index]);
                        let prediction = model
                            .predict(&state, &final_observation.value().features)
                            .unwrap();
                        black_box(prediction);
                        black_box(&state);
                    },
                );
            }
            memory_probe(
                &format!("storage/seq256/f64/{}/dim={d}", candidate.name()),
                || {
                    let mut state = warm.clone();
                    for observation in observations.iter().skip(WARM_START).take(256) {
                        state =
                            update64(candidate, &model, &state, observation, &mut scratch).unwrap();
                    }
                    state
                },
            );
        }
    }
}

fn main() {
    println!(
        "SGD temporary/storage comparison (benchmark-only); capture={}",
        capture_index()
    );
    println!("registered: 156 timing rows, 64 memory probes");
    println!("{}", common::env_summary());
    guard_storage();
    if cfg!(debug_assertions) {
        println!("debug: storage guards complete; release rows skipped");
        return;
    }
    run_storage_rows();
}
