//! Candidate update implementations and bitwise state comparisons.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdSample, SgdSample64, SgdState, SgdState64,
};
use tinyvec::TinyVec;

use super::common::comparison::{baseline_first, capture_from_env};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Candidate {
    Baseline,
    Direct,
    Array,
    Reuse,
    Tinyvec,
}

impl Candidate {
    pub(super) const ALL: [Self; 5] = [
        Self::Baseline,
        Self::Direct,
        Self::Array,
        Self::Reuse,
        Self::Tinyvec,
    ];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Direct => "direct",
            Self::Array => "array",
            Self::Reuse => "reuse",
            Self::Tinyvec => "tinyvec",
        }
    }

    pub(super) fn applicable(self, d: usize) -> bool {
        self != Self::Array || d <= 8
    }
}

pub(super) fn ordered_candidates() -> Vec<Candidate> {
    baseline_first(capture_from_env(), &Candidate::ALL)
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

pub(super) fn update32(
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

pub(super) fn update64(
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

pub(super) fn states_equal32(a: &SgdState, b: &SgdState) -> bool {
    a.w.len() == b.w.len()
        && a.w
            .as_slice()
            .iter()
            .zip(b.w.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

pub(super) fn states_equal64(a: &SgdState64, b: &SgdState64) -> bool {
    a.w.len() == b.w.len()
        && a.w
            .as_slice()
            .iter()
            .zip(b.w.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}
