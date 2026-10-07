# Seqvex — Foundation Scaffold

> **Historical reference — superseded execution instructions:** retain this record for its original scope and evidence; do not execute it as current implementation authorization. Any earlier prediction-only scope, preset-parameter demonstration or completion claim must be assessed under the [mandatory training-to-inference contract](DEVELOPMENT.md#model-training-and-valid-inference--mandatory) and current Architect → Planner → Coder gates. Raise unresolved failure decisions to the maintainer.

## Mission

This is the **execution specification** for KiloCode after reading `docs/KILOCODE_CONTEXT.md` and the existing Seqvex documentation.

Create the smallest useful **foundation structure and executable behavioral specification**.

Do not build Seqvex.

The human developer will personally implement production functionality from the tests.

## 1. Inspect First

Read:

- `README.md`
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `docs/DEVELOPMENT.md`
- `docs/FAILURE_AND_RECOVERY.md`
- `docs/KILOCODE_CONTEXT.md`
- current `Cargo.toml`
- current `src/`
- current `tests/`

Inspect the repository before changing it.

## 2. Modular Structure

Seqvex is modular and hierarchical.

Use this rule:

> **Decompose a functional domain into sub-functional components until each lowest-level component represents a coherent responsibility that can own focused tests and implementation.**

However:

> **Do not create a README.md below the functional-module level by default.**

Functional modules receive `README.md`.

Sub-functional components receive:

- focused tests;
- meaningful inline code documentation;
- clear names;
- small implementation boundaries.

For example:

```text
foundation/
├── observation/
│   ├── README.md
│   ├── representation/
│   ├── ordering/
│   └── ...
│
└── state/
    ├── README.md
    ├── transition/
    ├── validation/
    └── atomicity/
```

Do not generate a large empty hierarchy merely for completeness.

Instantiate the hierarchy deeply only where the foundation requires it.

## 3. Functional-Level README

For each foundation functional module that is actually created, add a concise `README.md`.

A functional README should explain:

1. purpose;
2. responsibility;
3. relationship to Seqvex;
4. what is inside the module;
5. what is explicitly outside;
6. current tests/specification;
7. major deferred decisions.

Do not duplicate `ARCHITECTURE.md`.

Do not create README files for every sub-function.

## 4. Foundation Scope

Create tests for:

1. observation semantics;
2. sequence ordering;
3. streaming semantics;
4. single-observation execution;
5. bounded micro-batch semantics;
6. persistent state;
7. state transitions;
8. failed-update atomicity;
9. invalid-input handling;
10. numerical failure handling;
11. state-invariant protection;
12. sequential continuity after recoverable failure;
13. deterministic replay where meaningful;
14. minimal numerical primitives;
15. streaming numerical semantics.

Stop at this foundation.

## 5. No Production ML/RL Implementation

Do NOT implement:

- neural networks;
- CNNs;
- RNNs;
- LSTMs;
- GRUs;
- transformers;
- attention;
- state-space models;
- regression algorithms;
- trees/random forests;
- clustering algorithms;
- RL algorithms;
- optimizers;
- automatic differentiation;
- GPU kernels;
- CUDA/ROCm;
- tensor engines;
- production allocators;
- schedulers;
- computational graphs;
- serialization;
- distributed execution;
- dataframes;
- ingestion;
- ETL;
- databases;
- visualization.

## 6. No Fake Completeness

Do not generate large quantities of:

- TODO-heavy code;
- `unimplemented!()`;
- `panic!("not implemented")`;
- dummy values;
- fake implementations.

A tiny fixture or minimal boundary type is acceptable when genuinely necessary.

The objective is an executable specification, not apparent completeness.

## 7. Do Not Freeze Speculative APIs

Tests should specify behavior and invariants.

Do not unnecessarily dictate:

- final method names;
- final trait hierarchies;
- tensor APIs;
- device APIs;
- buffer APIs;
- storage APIs;
- scheduler APIs.

Use the simplest interface necessary to express a stable requirement.

## 8. Observation

Create tests establishing that:

- an observation can be represented;
- required input information is preserved;
- sequence/order context can be represented when required;
- observations participate in sequential processing.

Suggested names:

```text
observation_can_be_represented
observation_preserves_required_input_information
observation_can_carry_sequence_context_when_required
```

Do not require every observation to contain a timestamp.

## 9. Ordering

Test:

```text
x1, x2, x3, x4
```

is logically processed as:

```text
x1 → x2 → x3 → x4
```

Suggested names:

```text
processing_preserves_observation_order
sequence_replay_preserves_order
```

Do not dictate the internal collection.

## 10. Streaming

Establish:

```text
initial state
    ↓
x1 → state1
    ↓
x2 → state2
    ↓
x3 → state3
```

Suggested names:

```text
streaming_processing_preserves_state
sequential_observations_update_state_in_order
```

Do not implement a production streaming engine.

## 11. Single Observation

A single observation is the smallest semantic execution unit.

Test:

```text
S0 + x1 → S1
S1 + x2 → S2
```

Suggested names:

```text
single_observation_is_smallest_execution_unit
single_observation_updates_current_state
```

This does not require all physical execution to be one-at-a-time.

## 12. Micro-Batch

Seqvex is streaming-first, not batchless.

Create one deliberately simple fixture where equivalence is clearly defined.

Test that:

```text
process(x1)
process(x2)
process(x3)
```

and:

```text
process_batch(x1, x2, x3)
```

preserve the same defined state semantics.

Suggested name:

```text
micro_batch_preserves_defined_sequence_semantics
```

Do not assume universal equivalence.

## 13. State

Test that state:

- persists across observations;
- is distinct from an observation;
- is used by subsequent observations;
- becomes the next valid state after successful processing.

Suggested names:

```text
state_persists_across_observations
next_observation_uses_previous_valid_state
state_transition_produces_next_state
```

## 14. State Atomicity

Test:

```text
S0 + x → invalid candidate
               ↓
             reject
               ↓
              S0
```

Required invariant:

> A failed update must not silently commit partial state.

Suggested names:

```text
failed_update_preserves_previous_valid_state
failed_update_does_not_commit_partial_state
```

Do not prescribe the implementation mechanism.

## 15. Invalid Input

Test:

```text
S0 + invalid_input
        ↓
      reject
        ↓
       S0
```

Suggested name:

```text
invalid_observation_does_not_mutate_state
```

## 16. Numerical Failure

Create a minimal test-only failure fixture.

Expected behavior:

```text
numerical failure
      ↓
update rejected
      ↓
previous valid state preserved
```

Suggested name:

```text
numerical_failure_preserves_previous_valid_state
```

## 17. State Invariants

Use a simple test-only invariant, for example:

```text
state_value >= 0
```

Test:

```text
candidate violates invariant
        ↓
candidate rejected
        ↓
previous valid state preserved
```

Suggested name:

```text
invalid_candidate_state_is_not_committed
```

Do not create a generalized invariant engine.

## 18. Sequential Continuity After Failure

Test:

```text
x1 → S1
x2 → FAILURE
x3 → S3
```

If processing continues, `x3` must use `S1`.

Suggested name:

```text
recoverable_failure_preserves_sequential_continuity
```

## 19. Determinism

Where deterministic behavior is defined, test:

```text
same initial state
+
same ordered observations
+
same configuration
=
same resulting state
```

Suggested name:

```text
deterministic_replay_produces_same_result
```

Do not claim cross-platform floating-point determinism.

## 20. Minimal Numerical Tests

Create tests only for a small foundation:

```text
sum
mean
variance
dot product
vector norm
```

Test ordinary values and meaningful edge cases.

Do not implement the numerical algorithms during this task.

Do not add a large numerical dependency.

## 21. Streaming Numerical Tests

Where contracts are clear, specify:

```text
online mean
online variance
rolling statistic semantics
```

Do not optimize or implement them now.

## 22. Rust Coding Principles

The scaffold and any minimal supporting code must follow the coding principles in `docs/DEVELOPMENT.md`.

In particular:

### DRY

Avoid duplicated knowledge and logic, but do not introduce abstractions merely because code looks similar.

### Composition over inheritance

Prefer composition of focused structs and small behavior-oriented traits.

Example:

```rust
struct Model<S, E> {
    state: S,
    executor: E,
}
```

Avoid deep inheritance-like hierarchies.

### Clippy

Maintain a clean Clippy baseline.

Where supported, use:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Do not suppress warnings casually.

### Extension traits

Use extension traits only for coherent domain-specific behavior on existing types.

Do not use them to hide trivial methods.

### Builder pattern

Use builders when construction involves meaningful configuration, optional parameters, or validation.

Do not use builders for trivial structures.

### Typestate

Use typestate only when compile-time state distinctions provide a real correctness guarantee.

This may become useful as Seqvex evolves toward asynchronous/stateful execution.

Do not build a speculative typestate framework now.

### Newtype

Use newtypes when semantic distinction matters.

For example:

```rust
struct ObservationId(u64);
struct SequenceNumber(u64);
```

Do not wrap every primitive.

### Error enums

Prefer explicit domain error enums and `Result<T, E>` for recoverable failures.

Do not use stringly typed errors for core domain behavior.

Avoid unnecessary `unwrap()` and `expect()` in production paths.

### Iterators / functional style

Prefer iterator composition where it is clear and appropriate.

Example:

```rust
let sum: f64 = values.iter().copied().sum();
```

Do not force iterator style where an explicit loop is clearer or materially better for performance.

Benchmark performance-sensitive choices rather than relying on stylistic assumptions.

## 23. Test Organization

Keep testing simple.

Use:

- unit tests for local behavior;
- integration tests for public boundaries;
- failure tests for state integrity;
- invariant/property tests where useful.

Do not introduce elaborate testing infrastructure.

BDD is optional.

## 24. Dependencies

Keep dependencies minimal.

Do not add large ML, numerical, dataframe, GPU, CUDA, or ROCm dependencies.

If a dependency appears necessary, explain why before adding it.

## 25. Crates

Do not create all future crates.

Create a crate only when an actual responsibility/dependency boundary requires it.

Potential future names are not current requirements.

## 26. Verification

Run as appropriate:

```bash
cargo fmt --check
cargo check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Do not make the repository green at all costs.

Do not fake implementations just to satisfy tests.

Clearly distinguish:

- passing tests;
- tests specifying behavior awaiting human implementation.

## 27. Stop Condition

Stop after completing the foundation scaffold.

Do not proceed automatically into:

- classical ML;
- online-learning algorithms;
- temporal validation;
- neural networks;
- RL;
- runtime;
- GPU;
- hardware optimization.

## 28. Final Report

Report exactly:

### Files created
Every new file.

### Files modified
Every modified file.

### Functional modules created
List them and explain their responsibility briefly.

### Sub-functional components created
List them.

### Functional-level README files created
List them.

### Tests created
Group by:

- observation;
- ordering;
- streaming;
- state;
- failure;
- determinism;
- numerical.

### Tests passing
List them.

### Tests awaiting implementation
List them.

### Dependencies added
List them or `None`.

### Architectural decisions introduced
Only genuinely new decisions.

### Architectural decisions deliberately deferred
List them.

### Issues discovered
Report conflicts, ambiguities, or repository inconsistencies requiring human review.

Do not make changes outside this task.

## Final Constraint

> **Do not build Seqvex for the developer. Build the executable specification from which the developer can build Seqvex.**
