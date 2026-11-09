//! State transition contract and single-observation execution.
//!
//! The transition mechanism is deliberately provisional. A candidate next state
//! is represented as a returned value rather than by mutating the current state,
//! which makes failure atomicity structural.

/// A stateful computation.
///
/// Given the current valid state and one observation, an implementation
/// produces a candidate next state or a classified failure. It must not mutate
/// the current state, so a failed update cannot commit partial state.
pub trait StateModel {
    /// The committed state carried between observations.
    type State;
    /// The input accepted by this model.
    type Observation;
    /// The failure reported by this model.
    type Error;

    /// Computes the candidate next state for `observation`.
    fn update(
        &self,
        state: &Self::State,
        observation: &Self::Observation,
    ) -> Result<Self::State, Self::Error>;
}

/// Single-observation execution: the smallest semantic execution unit.
pub fn process_one<M: StateModel>(
    model: &M,
    state: &M::State,
    observation: &M::Observation,
) -> Result<M::State, M::Error> {
    model.update(state, observation)
}
