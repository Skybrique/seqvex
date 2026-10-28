#![allow(dead_code)] // shared by test/example consumers that use different subsets.

//! Shared caller-local protocol support for the linear-family execution tests
//! and the runnable example. This is not production infrastructure.
//!
//! The helpers encode the LF-EXEC-D3 caller contract: completed-prefix counts,
//! opaque identities, bounded ordered learning, all-or-error calculation,
//! caller-owned publication, pending-head inspection/commit, and caller-owned
//! OLS acceptance/pairing.

use std::collections::{BTreeMap, VecDeque};

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdSample, SgdSample64, SgdState, SgdState64,
};

pub type PrefixCount = usize;
pub type ObservationIndex = usize;
pub type Identity = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowMode {
    HistoricalReplay,
    LiveAdaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessPolicy {
    FrozenSelected {
        serving_identity: Identity,
        served_through: PrefixCount,
    },
    LatestEligible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    pub learning: PrefixCount,
    pub selected: PrefixCount,
    pub served: PrefixCount,
    pub calculated: PrefixCount,
    pub published: PrefixCount,
}

impl Progress {
    pub fn commit_update<S, E>(&mut self, state: &mut S, update: Result<S, E>) -> Result<(), E> {
        let next = update?;
        *state = next;
        self.learning += 1;
        Ok(())
    }

    pub fn select<S: Clone>(&mut self, state: &S, identity: Identity) -> (S, SelectedRecord) {
        self.selected = self.learning;
        (
            state.clone(),
            SelectedRecord {
                identity,
                selected_through: self.selected,
            },
        )
    }

    pub fn install_serving<S, E>(
        &mut self,
        active: &mut (S, ServingRecord),
        conversion_and_check: Result<S, E>,
        identity: Identity,
    ) -> Result<(), E> {
        let snapshot = conversion_and_check?;
        let record = ServingRecord {
            identity,
            served_through: self.learning,
        };
        *active = (snapshot, record);
        self.served = self.learning;
        Ok(())
    }

    pub fn retain_calculation<T, E>(
        &mut self,
        calculation: Result<RetainedBatch<T>, E>,
    ) -> Result<RetainedBatch<T>, E> {
        let mut batch = calculation?;
        assert_eq!(batch.observation_indices.len(), batch.values.len());
        self.calculated += batch.values.len();
        batch.calculated_count = self.calculated;
        Ok(batch)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedRecord {
    pub identity: Identity,
    pub selected_through: PrefixCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingRecord {
    pub identity: Identity,
    pub served_through: PrefixCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OlsProgress {
    pub candidate_identity: Option<Identity>,
    pub installed_identity: Identity,
    pub calculated: PrefixCount,
    pub published: PrefixCount,
}

impl OlsProgress {
    pub fn record_candidate<P, T, E>(
        &mut self,
        candidate: Result<OlsCandidate<P, T>, E>,
    ) -> Result<OlsCandidate<P, T>, E> {
        let candidate = candidate?;
        self.candidate_identity = Some(candidate.candidate_identity);
        Ok(candidate)
    }

    pub fn record_install<E>(&mut self, installation: Result<Identity, E>) -> Result<Identity, E> {
        let identity = installation?;
        self.installed_identity = identity;
        self.candidate_identity = None;
        Ok(identity)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetainedBatch<T> {
    pub observation_indices: Vec<ObservationIndex>,
    pub values: Vec<T>,
    pub model_identity: Identity,
    pub calculated_count: PrefixCount,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSnapshot<F, P, S> {
    pub observation_index: ObservationIndex,
    pub model_identity: Identity,
    pub preprocessing_identity: Identity,
    pub features: F,
    pub preprocessing: P,
    pub selected_state: S,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictionRecord<T> {
    pub observation_index: ObservationIndex,
    pub model_identity: Identity,
    pub output: T,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionResolutionFailure<L, E> {
    MissingPrediction {
        item: (ObservationIndex, L),
    },
    PredictionFailed {
        item: (ObservationIndex, L),
        error: E,
    },
}

pub fn recover_missing_prediction<L, Ftr, Prep, State, Output, E, F>(
    item: (ObservationIndex, L),
    eligible_model_identity: Identity,
    eligible_preprocessing_identity: Identity,
    existing: Option<PredictionRecord<Output>>,
    snapshot: Option<&FeatureSnapshot<Ftr, Prep, State>>,
    mut predict: F,
) -> Result<PredictionRecord<Output>, PredictionResolutionFailure<L, E>>
where
    F: FnMut(&State, &Ftr, &Prep) -> Result<Output, E>,
{
    if let Some(record) = existing {
        if record.observation_index == item.0 && record.model_identity == eligible_model_identity {
            return Ok(record);
        }
        return Err(PredictionResolutionFailure::MissingPrediction { item });
    }
    let Some(snapshot) = snapshot.filter(|snapshot| {
        snapshot.observation_index == item.0
            && snapshot.model_identity == eligible_model_identity
            && snapshot.preprocessing_identity == eligible_preprocessing_identity
    }) else {
        return Err(PredictionResolutionFailure::MissingPrediction { item });
    };
    let output = predict(
        &snapshot.selected_state,
        &snapshot.features,
        &snapshot.preprocessing,
    )
    .map_err(|error| PredictionResolutionFailure::PredictionFailed { item, error })?;
    Ok(PredictionRecord {
        observation_index: snapshot.observation_index,
        model_identity: snapshot.model_identity,
        output,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderedSource<T> {
    events: VecDeque<(ObservationIndex, T)>,
    consumed: usize,
}

impl<T> OrderedSource<T> {
    pub fn new(events: Vec<(ObservationIndex, T)>) -> Self {
        Self {
            events: events.into(),
            consumed: 0,
        }
    }

    pub fn next_expected(
        &mut self,
        expected: ObservationIndex,
    ) -> Result<Option<(ObservationIndex, T)>, TemporalFailure> {
        match self.events.front() {
            Some((index, _)) if *index != expected => Err(TemporalFailure::SourceCannotDeliverGap),
            Some(_) => {
                self.consumed += 1;
                Ok(self.events.pop_front())
            }
            None => Ok(None),
        }
    }

    pub fn consumed(&self) -> usize {
        self.consumed
    }

    pub fn remaining(&self) -> usize {
        self.events.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationOutcome<E> {
    FullyAcknowledged { acknowledged_count: PrefixCount },
    NoWrite,
    AcknowledgedPrefixUnwrittenSuffix { acknowledged_count: PrefixCount },
    AcknowledgedPrefixUncertainSuffix { acknowledged_count: PrefixCount },
    Unclassified { error: E },
}

pub trait CallerSink<T> {
    type Error;
    fn publish(&mut self, batch: &RetainedBatch<T>) -> PublicationOutcome<Self::Error>;
}

#[derive(Debug, Clone)]
pub struct ObservingSink<T, E> {
    outcomes: VecDeque<PublicationOutcome<E>>,
    submitted: Vec<RetainedBatch<T>>,
}

pub type ScriptedSink<T, E> = ObservingSink<T, E>;

impl<T, E> ObservingSink<T, E> {
    pub fn new(outcomes: Vec<PublicationOutcome<E>>) -> Self {
        Self {
            outcomes: outcomes.into(),
            submitted: Vec::new(),
        }
    }

    pub fn calls(&self) -> usize {
        self.submitted.len()
    }

    pub fn submitted(&self) -> &[RetainedBatch<T>] {
        &self.submitted
    }
}

impl<T: Clone, E> CallerSink<T> for ObservingSink<T, E> {
    type Error = E;

    fn publish(&mut self, batch: &RetainedBatch<T>) -> PublicationOutcome<E> {
        self.submitted.push(batch.clone());
        self.outcomes
            .pop_front()
            .expect("scripted sink has a declared outcome")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationAction<E> {
    Complete,
    Retry,
    Reconcile,
    Stop(E),
    InvalidAcknowledgement { expected_max: usize, actual: usize },
}

pub fn publish_retained<T, E, S: CallerSink<T, Error = E>>(
    retained: &mut Option<RetainedBatch<T>>,
    progress: &mut Progress,
    sink: &mut S,
) -> PublicationAction<E> {
    let Some(batch) = retained.as_mut() else {
        return PublicationAction::Complete;
    };
    let actual = batch.values.len();
    let outcome = sink.publish(batch);
    match outcome {
        PublicationOutcome::FullyAcknowledged { acknowledged_count } => {
            if acknowledged_count != actual {
                return PublicationAction::InvalidAcknowledgement {
                    expected_max: actual,
                    actual: acknowledged_count,
                };
            }
            progress.published += acknowledged_count;
            *retained = None;
            PublicationAction::Complete
        }
        PublicationOutcome::NoWrite => PublicationAction::Retry,
        PublicationOutcome::AcknowledgedPrefixUnwrittenSuffix { acknowledged_count } => {
            if acknowledged_count > actual {
                return PublicationAction::InvalidAcknowledgement {
                    expected_max: actual,
                    actual: acknowledged_count,
                };
            }
            progress.published += acknowledged_count;
            batch.observation_indices.drain(..acknowledged_count);
            batch.values.drain(..acknowledged_count);
            if batch.values.is_empty() {
                *retained = None;
                PublicationAction::Complete
            } else {
                PublicationAction::Retry
            }
        }
        PublicationOutcome::AcknowledgedPrefixUncertainSuffix { acknowledged_count } => {
            if acknowledged_count > actual {
                return PublicationAction::InvalidAcknowledgement {
                    expected_max: actual,
                    actual: acknowledged_count,
                };
            }
            progress.published += acknowledged_count;
            batch.observation_indices.drain(..acknowledged_count);
            batch.values.drain(..acknowledged_count);
            if batch.values.is_empty() {
                *retained = None;
                PublicationAction::Complete
            } else {
                PublicationAction::Reconcile
            }
        }
        PublicationOutcome::Unclassified { error } => PublicationAction::Stop(error),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchFailure<S, E> {
    TooLarge {
        max: usize,
        actual: usize,
        state: S,
    },
    Transition {
        failed_index: ObservationIndex,
        committed: PrefixCount,
        state: S,
        error: E,
    },
}

pub fn replay_sgd32(
    model: &Sgd,
    state: SgdState,
    rows: &[Observation<SgdSample>],
    max_batch: usize,
) -> Result<SgdState, BatchFailure<SgdState, SgdError>> {
    replay_sgd32_observed(model, state, rows, max_batch, |_| {})
}

pub fn replay_sgd32_observed<F>(
    model: &Sgd,
    mut state: SgdState,
    rows: &[Observation<SgdSample>],
    max_batch: usize,
    mut attempted: F,
) -> Result<SgdState, BatchFailure<SgdState, SgdError>>
where
    F: FnMut(usize),
{
    if rows.len() > max_batch {
        return Err(BatchFailure::TooLarge {
            max: max_batch,
            actual: rows.len(),
            state,
        });
    }
    for (failed_index, row) in rows.iter().enumerate() {
        attempted(failed_index);
        match model.update(&state, row) {
            Ok(next) => state = next,
            Err(error) => {
                return Err(BatchFailure::Transition {
                    failed_index,
                    committed: failed_index,
                    state,
                    error,
                });
            }
        }
    }
    Ok(state)
}

pub fn replay_sgd64(
    model: &Sgd64,
    state: SgdState64,
    rows: &[Observation<SgdSample64>],
    max_batch: usize,
) -> Result<SgdState64, BatchFailure<SgdState64, SgdError64>> {
    replay_sgd64_observed(model, state, rows, max_batch, |_| {})
}

pub fn replay_sgd64_observed<F>(
    model: &Sgd64,
    mut state: SgdState64,
    rows: &[Observation<SgdSample64>],
    max_batch: usize,
    mut attempted: F,
) -> Result<SgdState64, BatchFailure<SgdState64, SgdError64>>
where
    F: FnMut(usize),
{
    if rows.len() > max_batch {
        return Err(BatchFailure::TooLarge {
            max: max_batch,
            actual: rows.len(),
            state,
        });
    }
    for (failed_index, row) in rows.iter().enumerate() {
        attempted(failed_index);
        match model.update(&state, row) {
            Ok(next) => state = next,
            Err(error) => {
                return Err(BatchFailure::Transition {
                    failed_index,
                    committed: failed_index,
                    state,
                    error,
                });
            }
        }
    }
    Ok(state)
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionFailure<E> {
    TooLarge { max: usize, actual: usize },
    Failed { index: ObservationIndex, error: E },
}

pub fn collect32<E, F>(
    max_predictions: usize,
    rows: &[Observation<Vector>],
    mut predict: F,
) -> Result<Vec<f32>, PredictionFailure<E>>
where
    F: FnMut(&Vector) -> Result<f32, E>,
{
    if rows.len() > max_predictions {
        return Err(PredictionFailure::TooLarge {
            max: max_predictions,
            actual: rows.len(),
        });
    }
    let mut outputs = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        match predict(row.value()) {
            Ok(value) => outputs.push(value),
            Err(error) => return Err(PredictionFailure::Failed { index, error }),
        }
    }
    Ok(outputs)
}

pub fn collect64<E, F>(
    max_predictions: usize,
    rows: &[Observation<Vector64>],
    mut predict: F,
) -> Result<Vec<f64>, PredictionFailure<E>>
where
    F: FnMut(&Vector64) -> Result<f64, E>,
{
    if rows.len() > max_predictions {
        return Err(PredictionFailure::TooLarge {
            max: max_predictions,
            actual: rows.len(),
        });
    }
    let mut outputs = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        match predict(row.value()) {
            Ok(value) => outputs.push(value),
            Err(error) => return Err(PredictionFailure::Failed { index, error }),
        }
    }
    Ok(outputs)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalFailure {
    MissingPrediction { observation_index: ObservationIndex },
    CapacityRefused,
    SourceCannotDeliverGap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Refused<T> {
    pub failure: TemporalFailure,
    pub item: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingBudget<T> {
    ordinary_capacity: usize,
    ordinary: BTreeMap<ObservationIndex, T>,
    head: Option<(ObservationIndex, T)>,
    next_required: Option<ObservationIndex>,
}

impl<T> PendingBudget<T> {
    pub fn new(ordinary_capacity: usize) -> Self {
        Self::new_at(ordinary_capacity, 0)
    }

    pub fn new_at(ordinary_capacity: usize, first_required: ObservationIndex) -> Self {
        Self {
            ordinary_capacity,
            ordinary: BTreeMap::new(),
            head: None,
            next_required: Some(first_required),
        }
    }

    pub fn admit_later(
        &mut self,
        observation_index: ObservationIndex,
        value: T,
    ) -> Result<(), Refused<(ObservationIndex, T)>> {
        let Some(next_required) = self.next_required else {
            return Err(Refused {
                failure: TemporalFailure::CapacityRefused,
                item: (observation_index, value),
            });
        };
        if observation_index <= next_required
            || self.ordinary.contains_key(&observation_index)
            || self
                .head
                .as_ref()
                .is_some_and(|(index, _)| *index == observation_index)
            || self.ordinary.len() >= self.ordinary_capacity
        {
            return Err(Refused {
                failure: TemporalFailure::CapacityRefused,
                item: (observation_index, value),
            });
        }
        self.ordinary.insert(observation_index, value);
        Ok(())
    }

    pub fn admit_head(
        &mut self,
        observation_index: ObservationIndex,
        value: T,
    ) -> Result<(), Refused<(ObservationIndex, T)>> {
        if self.head.is_some() || self.next_required != Some(observation_index) {
            return Err(Refused {
                failure: TemporalFailure::CapacityRefused,
                item: (observation_index, value),
            });
        }
        self.head = Some((observation_index, value));
        self.promote_required();
        Ok(())
    }

    fn promote_required(&mut self) {
        let Some(index) = self.next_required else {
            return;
        };
        if self.head.is_none() {
            if let Some(value) = self.ordinary.remove(&index) {
                self.head = Some((index, value));
            }
        }
    }

    pub fn inspect_head(&self) -> Option<(ObservationIndex, &T)> {
        self.head.as_ref().map(|(index, value)| (*index, value))
    }

    pub fn replace_head(
        &mut self,
        observation_index: ObservationIndex,
        value: T,
    ) -> Result<T, Refused<(ObservationIndex, T)>> {
        match self.head.as_mut() {
            Some((index, current)) if *index == observation_index => {
                Ok(std::mem::replace(current, value))
            }
            _ => Err(Refused {
                failure: TemporalFailure::CapacityRefused,
                item: (observation_index, value),
            }),
        }
    }

    pub fn commit_head(&mut self) -> Option<(ObservationIndex, T)> {
        let item = self.head.take()?;
        self.next_required = item.0.checked_add(1);
        self.promote_required();
        Some(item)
    }

    pub fn discard_head(&mut self) -> Option<(ObservationIndex, T)> {
        let item = self.head.take()?;
        self.next_required = item.0.checked_add(1);
        self.promote_required();
        Some(item)
    }

    pub fn pending_len(&self) -> usize {
        self.ordinary.len() + usize::from(self.head.is_some())
    }

    pub fn ordinary_len(&self) -> usize {
        self.ordinary.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingUpdateError<E> {
    pub observation_index: ObservationIndex,
    pub error: E,
}

pub fn apply_pending_prefix<S, T, E, F>(
    pending: &mut PendingBudget<T>,
    state: &mut S,
    progress: &mut Progress,
    mut update: F,
) -> Result<Vec<S>, PendingUpdateError<E>>
where
    S: Clone,
    F: FnMut(&S, &T) -> Result<S, E>,
{
    let mut committed = Vec::new();
    while let Some((index, value)) = pending.inspect_head() {
        let next = update(state, value).map_err(|error| PendingUpdateError {
            observation_index: index,
            error,
        })?;
        progress
            .commit_update::<S, ()>(state, Ok(next))
            .expect("successful candidate commits");
        let (committed_index, _) = pending
            .commit_head()
            .expect("inspected pending head remains available");
        assert_eq!(committed_index, index);
        committed.push(state.clone());
    }
    Ok(committed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessCheck {
    Ready,
    Blocked {
        learned: PrefixCount,
        served: PrefixCount,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessPolicyError {
    MissingPolicy,
    FrozenSnapshotMismatch,
}

pub fn check_freshness(
    policy: Option<FreshnessPolicy>,
    serving: &ServingRecord,
    eligible_learning: PrefixCount,
) -> Result<FreshnessCheck, FreshnessPolicyError> {
    let policy = policy.ok_or(FreshnessPolicyError::MissingPolicy)?;
    if let FreshnessPolicy::FrozenSelected {
        serving_identity,
        served_through,
    } = policy
    {
        if serving.identity != serving_identity || serving.served_through != served_through {
            return Err(FreshnessPolicyError::FrozenSnapshotMismatch);
        }
        return Ok(FreshnessCheck::Ready);
    }
    Ok(latest_freshness(serving.served_through, eligible_learning))
}

pub fn latest_freshness(served: PrefixCount, learned: PrefixCount) -> FreshnessCheck {
    if served == learned {
        FreshnessCheck::Ready
    } else {
        FreshnessCheck::Blocked { learned, served }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceDecision<E> {
    Accepted,
    Rejected,
    Error(E),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OlsCandidate<P, T> {
    pub predictor: P,
    pub preprocessing: T,
    pub candidate_identity: Identity,
    pub preprocessing_identity: Identity,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActiveOls<P, T> {
    pub predictor: P,
    pub preprocessing: T,
    pub installed_identity: Identity,
    pub preprocessing_identity: Identity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InstallError<E> {
    Rejected,
    Decision(E),
    Pairing {
        expected: Identity,
        actual: Identity,
    },
}

pub fn try_install_ols<P, T, E>(
    active: &mut ActiveOls<P, T>,
    candidate: OlsCandidate<P, T>,
    decision: AcceptanceDecision<E>,
    expected_preprocessing_identity: Identity,
) -> Result<Identity, InstallError<E>> {
    match decision {
        AcceptanceDecision::Rejected => Err(InstallError::Rejected),
        AcceptanceDecision::Error(error) => Err(InstallError::Decision(error)),
        AcceptanceDecision::Accepted => {
            if candidate.preprocessing_identity != expected_preprocessing_identity {
                return Err(InstallError::Pairing {
                    expected: expected_preprocessing_identity,
                    actual: candidate.preprocessing_identity,
                });
            }
            let OlsCandidate {
                predictor,
                preprocessing,
                candidate_identity,
                preprocessing_identity,
            } = candidate;
            active.predictor = predictor;
            active.preprocessing = preprocessing;
            active.installed_identity = candidate_identity;
            active.preprocessing_identity = preprocessing_identity;
            Ok(candidate_identity)
        }
    }
}
