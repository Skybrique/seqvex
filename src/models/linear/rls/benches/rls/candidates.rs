//! Update candidates and precheck machinery for the timed measurement paths.
//!
//! Candidate arithmetic and validation order are preserved from the pre-split
//! benchmark. Includes the fused/symmetric update controls and the shared
//! precheck/update-mismatch paths used by measurement and diagnostics.

use std::hint::black_box;

use seqvex::foundation::numerical::{Matrix, Matrix64, Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::rls::{
    Rls, Rls64, RlsError, RlsError64, RlsSample, RlsSample64, RlsState, RlsState64,
};

use crate::evidence::FailureRecord;
use crate::fixtures::{
    fingerprint32, fingerprint64, first_divergence32, first_divergence64, state_eq32, state_eq64,
    symmetric32, symmetric64,
};

pub(crate) fn p_full_into32(
    out: &mut [f32],
    p: &[f32],
    v: &[f32],
    d: usize,
    lambda: f32,
    denom: f32,
) {
    for row in 0..d {
        for col in 0..d {
            out[row * d + col] = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
        }
    }
}

pub(crate) fn p_sym_into32(
    out: &mut [f32],
    p: &[f32],
    v: &[f32],
    d: usize,
    lambda: f32,
    denom: f32,
) {
    for row in 0..d {
        for col in row..d {
            let value = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
            out[row * d + col] = value;
            out[col * d + row] = value;
        }
    }
}

pub(crate) fn p_full_into64(
    out: &mut [f64],
    p: &[f64],
    v: &[f64],
    d: usize,
    lambda: f64,
    denom: f64,
) {
    for row in 0..d {
        for col in 0..d {
            out[row * d + col] = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
        }
    }
}

pub(crate) fn p_sym_into64(
    out: &mut [f64],
    p: &[f64],
    v: &[f64],
    d: usize,
    lambda: f64,
    denom: f64,
) {
    for row in 0..d {
        for col in row..d {
            let value = (p[row * d + col] - v[row] * v[col] / denom) / lambda;
            out[row * d + col] = value;
            out[col * d + row] = value;
        }
    }
}

pub(crate) fn vcur_alloc32(w: &Vector, v: &Vector, error: f32, denom: f32) -> Result<(), String> {
    let gain = v.map(|value| value / denom);
    let scaled = gain.scale(error);
    let next = w.add(&scaled).map_err(|error| format!("{error:?}"))?;
    black_box(&next);
    Ok(())
}

pub(crate) fn vcur_pre32(
    w: &Vector,
    v: &Vector,
    error: f32,
    denom: f32,
    gain: &mut [f32],
    scaled: &mut [f32],
    next: &mut [f32],
) -> Result<(), String> {
    for index in 0..w.len() {
        gain[index] = v.as_slice()[index] / denom;
        scaled[index] = gain[index] * error;
        next[index] = w.as_slice()[index] + scaled[index];
    }
    black_box(&next);
    Ok(())
}

pub(crate) fn vfused_alloc32(w: &Vector, v: &Vector, error: f32, denom: f32) -> Result<(), String> {
    let next = Vector::from_fn(w.len(), |index| {
        w.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    black_box(&next);
    Ok(())
}

pub(crate) fn vfused_pre32(
    w: &Vector,
    v: &Vector,
    error: f32,
    denom: f32,
    next: &mut [f32],
) -> Result<(), String> {
    for (index, slot) in next.iter_mut().enumerate().take(w.len()) {
        *slot = w.as_slice()[index] + (v.as_slice()[index] / denom) * error;
    }
    black_box(&next);
    Ok(())
}

pub(crate) fn vcur_alloc64(
    w: &Vector64,
    v: &Vector64,
    error: f64,
    denom: f64,
) -> Result<(), String> {
    let gain = v.map(|value| value / denom);
    let scaled = gain.scale(error);
    let next = w.add(&scaled).map_err(|error| format!("{error:?}"))?;
    black_box(&next);
    Ok(())
}

pub(crate) fn vcur_pre64(
    w: &Vector64,
    v: &Vector64,
    error: f64,
    denom: f64,
    gain: &mut [f64],
    scaled: &mut [f64],
    next: &mut [f64],
) -> Result<(), String> {
    for index in 0..w.len() {
        gain[index] = v.as_slice()[index] / denom;
        scaled[index] = gain[index] * error;
        next[index] = w.as_slice()[index] + scaled[index];
    }
    black_box(&next);
    Ok(())
}

pub(crate) fn vfused_alloc64(
    w: &Vector64,
    v: &Vector64,
    error: f64,
    denom: f64,
) -> Result<(), String> {
    let next = Vector64::from_fn(w.len(), |index| {
        w.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    black_box(&next);
    Ok(())
}

pub(crate) fn vfused_pre64(
    w: &Vector64,
    v: &Vector64,
    error: f64,
    denom: f64,
    next: &mut [f64],
) -> Result<(), String> {
    for (index, slot) in next.iter_mut().enumerate().take(w.len()) {
        *slot = w.as_slice()[index] + (v.as_slice()[index] / denom) * error;
    }
    black_box(&next);
    Ok(())
}

fn validate32(model: &Rls, state: &RlsState, sample: &RlsSample) -> Result<(), RlsError> {
    let d = model.dimension();
    let features = &sample.features;
    if features.len() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: features.len(),
        });
    }
    if state.w.len() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: state.w.len(),
        });
    }
    if state.p.rows() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: state.p.rows(),
        });
    }
    if state.p.cols() != d {
        return Err(RlsError::DimensionMismatch {
            expected: d,
            actual: state.p.cols(),
        });
    }
    if !features.as_slice().iter().all(|value| value.is_finite()) || !sample.target.is_finite() {
        return Err(RlsError::NonFiniteInput);
    }
    Ok(())
}

fn validate64(model: &Rls64, state: &RlsState64, sample: &RlsSample64) -> Result<(), RlsError64> {
    let d = model.dimension();
    let features = &sample.features;
    let weights = state.weights();
    let covariance = state.covariance();
    if features.len() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: features.len(),
        });
    }
    if weights.len() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: weights.len(),
        });
    }
    if covariance.rows() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: covariance.rows(),
        });
    }
    if covariance.cols() != d {
        return Err(RlsError64::DimensionMismatch {
            expected: d,
            actual: covariance.cols(),
        });
    }
    if !features.as_slice().iter().all(|value| value.is_finite()) || !sample.target.is_finite() {
        return Err(RlsError64::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn update_fused32(
    model: &Rls,
    state: &RlsState,
    sample: &RlsSample,
) -> Result<RlsState, RlsError> {
    validate32(model, state, sample)?;
    let features = &sample.features;
    let v = state.p.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError::InvalidDenominator);
    }
    let error = sample.target - state.w.dot(features)?;
    let d = model.dimension();
    let next_w = Vector::from_fn(d, |index| {
        state.w.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    let p_data = state.p.as_slice();
    let v_data = v.as_slice();
    let lambda = model.lambda();
    let next_p = Matrix::from_fn(d, d, |row, col| {
        (p_data[row * d + col] - v_data[row] * v_data[col] / denom) / lambda
    });
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError::NonFiniteCandidate);
    }
    Ok(RlsState {
        w: next_w,
        p: next_p,
    })
}

pub(crate) fn update_sym32(
    model: &Rls,
    state: &RlsState,
    sample: &RlsSample,
) -> Result<RlsState, RlsError> {
    validate32(model, state, sample)?;
    let features = &sample.features;
    let v = state.p.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError::InvalidDenominator);
    }
    let error = sample.target - state.w.dot(features)?;
    let gain = v.map(|value| value / denom);
    let next_w = state.w.add(&gain.scale(error))?;
    let d = model.dimension();
    let mut buffer = vec![0.0_f32; d * d];
    p_sym_into32(
        &mut buffer,
        state.p.as_slice(),
        v.as_slice(),
        d,
        model.lambda(),
        denom,
    );
    let next_p = Matrix::from_fn(d, d, |row, col| buffer[row * d + col]);
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError::NonFiniteCandidate);
    }
    Ok(RlsState {
        w: next_w,
        p: next_p,
    })
}

fn admit64(weights: Vector64, covariance: Matrix64) -> Result<RlsState64, RlsError64> {
    RlsState64::admit(weights, covariance).map_err(|_| RlsError64::NonFiniteCandidate)
}

pub(crate) fn update_fused64(
    model: &Rls64,
    state: &RlsState64,
    sample: &RlsSample64,
) -> Result<RlsState64, RlsError64> {
    validate64(model, state, sample)?;
    let features = &sample.features;
    let weights = state.weights();
    let covariance = state.covariance();
    let v = covariance.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError64::InvalidDenominator);
    }
    let error = sample.target - weights.dot(features)?;
    let d = model.dimension();
    let next_w = Vector64::from_fn(d, |index| {
        weights.as_slice()[index] + (v.as_slice()[index] / denom) * error
    });
    let p_data = covariance.as_slice();
    let v_data = v.as_slice();
    let lambda = model.lambda();
    let next_p = Matrix64::from_fn(d, d, |row, col| {
        (p_data[row * d + col] - v_data[row] * v_data[col] / denom) / lambda
    });
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError64::NonFiniteCandidate);
    }
    admit64(next_w, next_p)
}

pub(crate) fn update_sym64(
    model: &Rls64,
    state: &RlsState64,
    sample: &RlsSample64,
) -> Result<RlsState64, RlsError64> {
    validate64(model, state, sample)?;
    let features = &sample.features;
    let weights = state.weights();
    let covariance = state.covariance();
    let v = covariance.mul_vector(features)?;
    let denom = model.lambda() + features.dot(&v)?;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(RlsError64::InvalidDenominator);
    }
    let error = sample.target - weights.dot(features)?;
    let gain = v.map(|value| value / denom);
    let next_w = weights.add(&gain.scale(error))?;
    let d = model.dimension();
    let mut buffer = vec![0.0_f64; d * d];
    p_sym_into64(
        &mut buffer,
        covariance.as_slice(),
        v.as_slice(),
        d,
        model.lambda(),
        denom,
    );
    let next_p = Matrix64::from_fn(d, d, |row, col| buffer[row * d + col]);
    if !next_w.as_slice().iter().all(|value| value.is_finite())
        || !next_p.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(RlsError64::NonFiniteCandidate);
    }
    admit64(next_w, next_p)
}

#[derive(Debug)]
pub(crate) struct PrecheckFailure {
    pub(crate) index: usize,
    pub(crate) kind: &'static str,
    pub(crate) error: String,
    pub(crate) fingerprint: u64,
    pub(crate) validation: bool,
    pub(crate) first_divergence: Option<String>,
}

pub(crate) fn precheck32(
    model: &Rls,
    observations: &[Observation<RlsSample>],
    warm: usize,
    horizon: usize,
    candidate: Candidate,
) -> Result<(RlsState, RlsState), PrecheckFailure> {
    let total = warm + horizon;
    let mut production = model.initial_state();
    let mut candidate_state = production.clone();
    let mut warm_state = None;
    for (index, observation) in observations.iter().enumerate().take(total) {
        let production_result = model.update(&production, observation);
        let candidate_result = match candidate {
            Candidate::Production => production_result.clone(),
            Candidate::Fused => update_fused32(model, &candidate_state, observation.value()),
            Candidate::Symmetric => update_sym32(model, &candidate_state, observation.value()),
            Candidate::Mismatch => update_mismatch32(model, &candidate_state, observation),
        };
        if let Err(error) = &production_result {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: format!("{error:?}"),
                fingerprint: fingerprint32(&production),
                validation: false,
                first_divergence: None,
            });
        }
        let production_next = match production_result {
            Ok(next) => next,
            Err(_) => unreachable!("production error already handled"),
        };
        let candidate_next = match candidate_result {
            Ok(next) => next,
            Err(error) => {
                return Err(PrecheckFailure {
                    index,
                    kind: "comparison_failure",
                    error: format!("candidate returned {error:?}"),
                    fingerprint: fingerprint32(&production),
                    validation: true,
                    first_divergence: Some(format!("candidate Err at observation {index}")),
                });
            }
        };
        if !state_eq32(&production_next, &candidate_next) {
            return Err(PrecheckFailure {
                index,
                kind: "comparison_failure",
                error: "coupled state mismatch".to_owned(),
                fingerprint: fingerprint32(&production),
                validation: true,
                first_divergence: first_divergence32(&production_next, &candidate_next),
            });
        }
        if !production_next
            .w
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
            || !production_next
                .p
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "non-finite committed state".to_owned(),
                fingerprint: fingerprint32(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        if index % 1_000 == 0 && !symmetric32(&production_next.p) {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "covariance is not bitwise symmetric".to_owned(),
                fingerprint: fingerprint32(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        production = production_next;
        candidate_state = candidate_next;
        if index + 1 == warm {
            warm_state = Some(production.clone());
        }
    }
    let warm_state = match warm_state {
        Some(state) => state,
        None => {
            return Err(PrecheckFailure {
                index: total,
                kind: "operation_error",
                error: "warm state was not captured".to_owned(),
                fingerprint: fingerprint32(&production),
                validation: false,
                first_divergence: None,
            });
        }
    };
    Ok((warm_state, production))
}

pub(crate) fn precheck64(
    model: &Rls64,
    observations: &[Observation<RlsSample64>],
    warm: usize,
    horizon: usize,
    candidate: Candidate,
) -> Result<(RlsState64, RlsState64), PrecheckFailure> {
    let total = warm + horizon;
    let mut production = model.initial_state();
    let mut candidate_state = production.clone();
    let mut warm_state = None;
    for (index, observation) in observations.iter().enumerate().take(total) {
        let production_result = model.update(&production, observation);
        let candidate_result = match candidate {
            Candidate::Production => production_result.clone(),
            Candidate::Fused => update_fused64(model, &candidate_state, observation.value()),
            Candidate::Symmetric => update_sym64(model, &candidate_state, observation.value()),
            Candidate::Mismatch => production_result.clone(),
        };
        if let Err(error) = &production_result {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: format!("{error:?}"),
                fingerprint: fingerprint64(&production),
                validation: false,
                first_divergence: None,
            });
        }
        let production_next = match production_result {
            Ok(next) => next,
            Err(_) => unreachable!("production error already handled"),
        };
        let candidate_next = match candidate_result {
            Ok(next) => next,
            Err(error) => {
                return Err(PrecheckFailure {
                    index,
                    kind: "comparison_failure",
                    error: format!("candidate returned {error:?}"),
                    fingerprint: fingerprint64(&production),
                    validation: true,
                    first_divergence: Some(format!("candidate Err at observation {index}")),
                });
            }
        };
        if !state_eq64(&production_next, &candidate_next) {
            return Err(PrecheckFailure {
                index,
                kind: "comparison_failure",
                error: "coupled state mismatch".to_owned(),
                fingerprint: fingerprint64(&production),
                validation: true,
                first_divergence: first_divergence64(&production_next, &candidate_next),
            });
        }
        if !production_next
            .weights()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
            || !production_next
                .covariance()
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "non-finite committed state".to_owned(),
                fingerprint: fingerprint64(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        if index % 1_000 == 0 && !symmetric64(production_next.covariance()) {
            return Err(PrecheckFailure {
                index,
                kind: "operation_error",
                error: "covariance is not bitwise symmetric".to_owned(),
                fingerprint: fingerprint64(&production_next),
                validation: false,
                first_divergence: None,
            });
        }
        production = production_next;
        candidate_state = candidate_next;
        if index + 1 == warm {
            warm_state = Some(production.clone());
        }
    }
    let warm_state = match warm_state {
        Some(state) => state,
        None => {
            return Err(PrecheckFailure {
                index: total,
                kind: "operation_error",
                error: "warm state was not captured".to_owned(),
                fingerprint: fingerprint64(&production),
                validation: false,
                first_divergence: None,
            });
        }
    };
    Ok((warm_state, production))
}

pub(crate) fn record_precheck_failure(
    row: &str,
    precision: &'static str,
    dimension: usize,
    warm: usize,
    horizon: usize,
    failure: PrecheckFailure,
) {
    let attempted = (failure.index + 1) as u64;
    let successful = if failure.validation {
        attempted
    } else {
        failure.index as u64
    };
    let failed = if failure.validation { 0 } else { 1 };
    let scheduled = (warm + horizon) as u64;
    FailureRecord {
        kind: failure.kind,
        phase: "precheck",
        row: row.to_owned(),
        precision,
        dimension,
        window: None,
        observation_global: Some(failure.index),
        observation_window: None,
        committed_progress: failure.index as u64,
        retained_fingerprint: Some(failure.fingerprint),
        scheduled_units: scheduled,
        attempted_units: attempted,
        successful_units: successful,
        failed_units: failed,
        unattempted_units: scheduled - attempted,
        validation_failure: failure.validation,
        postcondition_failure: false,
        first_divergence: failure.first_divergence,
        error: Some(failure.error),
    }
    .emit();
}

pub(crate) fn candidate_update32(
    model: &Rls,
    state: &RlsState,
    observation: &Observation<RlsSample>,
    candidate: Candidate,
) -> Result<RlsState, RlsError> {
    match candidate {
        Candidate::Production => model.update(state, observation),
        Candidate::Fused => update_fused32(model, state, observation.value()),
        Candidate::Symmetric => update_sym32(model, state, observation.value()),
        Candidate::Mismatch => update_mismatch32(model, state, observation),
    }
}

pub(crate) fn candidate_update64(
    model: &Rls64,
    state: &RlsState64,
    observation: &Observation<RlsSample64>,
    candidate: Candidate,
) -> Result<RlsState64, RlsError64> {
    match candidate {
        Candidate::Production => model.update(state, observation),
        Candidate::Fused => update_fused64(model, state, observation.value()),
        Candidate::Symmetric => update_sym64(model, state, observation.value()),
        Candidate::Mismatch => model.update(state, observation),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Candidate {
    Production,
    Fused,
    Symmetric,
    Mismatch,
}

impl Candidate {
    pub(crate) fn category(self) -> &'static str {
        match self {
            Self::Production => "S",
            Self::Fused | Self::Symmetric => "U",
            Self::Mismatch => "F",
        }
    }
}

fn update_mismatch32(
    model: &Rls,
    state: &RlsState,
    observation: &Observation<RlsSample>,
) -> Result<RlsState, RlsError> {
    let next = model.update(state, observation)?;
    let mut weights = next.w.as_slice().to_vec();
    if let Some(first) = weights.first_mut() {
        *first = f32::from_bits(first.to_bits() ^ 1);
    }
    Ok(RlsState {
        w: Vector::from_slice(&weights),
        p: next.p,
    })
}

pub(crate) fn candidate_row(precision: &'static str, candidate: Candidate) -> &'static str {
    match (precision, candidate) {
        ("f32", Candidate::Production) => "S-C",
        ("f64", Candidate::Production) => "S-A",
        (_, Candidate::Fused) => "U-Fused",
        (_, Candidate::Symmetric) => "U-Sym",
        (_, Candidate::Mismatch) => "F-ComparisonOnly",
        (_, Candidate::Production) => "S-C",
    }
}
