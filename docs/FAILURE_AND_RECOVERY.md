# Seqvex Failure and Recovery

## Mandatory escalation and valid outputs

Apply the [failure decision rule](DEVELOPMENT.md#calculation-failures-and-error-handling--mandatory). Bring any missing, ambiguous or contradictory failure policy to the maintainer with the exact failing operation, valid committed state/progress, consequences and candidate responses before affected work proceeds. Do not silently infer retry, skip, reset, continuation, rollback, numeric repair or default-output behavior. Existing behavior described below is source/history evidence; it does not authorize new unresolved policies.

A failed fit or prediction must not be disguised as a valid result. Preserve the defined previously valid state without labelling it as a prediction for the failed observation. Training readiness and the numerical/resource envelope must be explicit in the model design. No model is complete until the required data → train → validate → infer path and its failure evidence exist.

> **Status:** Early-stage failure-handling architecture / living technical document

## 1. Purpose

Seqvex is a stateful ML/RL framework. In streaming and online workloads, a model update can change the state that will be used by subsequent observations.

Therefore, failure handling must protect the validity of model state and preserve sequential semantics.

The fundamental requirement is:

> **A failed model update must never silently commit invalid or partially updated model state.**

This document describes the intended failure paths and recovery principles. It does not freeze a specific implementation mechanism.

---

## 2. Normal State Transition

A successful update follows the conceptual path:

```text
observation_t
     │
     ▼
current valid state_t
     │
     ▼
compute update
     │
     ▼
candidate state_(t+1)
     │
     ▼
validate
     │
     ▼
   COMMIT
     │
     ▼
valid state_(t+1)
     │
     ▼
next observation
```

The important distinction is between:

- **current state** — the last committed valid state;
- **candidate state** — the result of the current update before acceptance;
- **committed state** — the valid state exposed to subsequent computation.

The implementation does not have to literally copy the complete state to create a candidate. The architectural requirement is that invalid or incomplete updates must not silently become committed state.

---

## 3. General Failure Path

If an update fails:

```text
observation_t
     │
     ▼
current valid state_t
     │
     ▼
attempt update
     │
     ├───────────────┐
     │               │
   success         failure
     │               │
     ▼               ▼
candidate         reject update
state                │
     │               ▼
     ▼          preserve state_t
  validate            │
     │                ▼
 ┌───┴───┐       report failure
 │       │            │
valid   invalid       ▼
 │       │       apply failure policy
 ▼       ▼
commit  reject
 │       │
 ▼       ▼
state   state_t
_(t+1)
```

The default expectation is:

> **After a failed update, the last committed valid state remains the active model state unless an explicit recovery mechanism changes it.**

---

## 4. Failure Categories

Seqvex should distinguish different failure classes because the correct action may differ.

### 4.1 Invalid Input

Examples:

- invalid dimensionality;
- incompatible representation;
- invalid value;
- missing required input;
- invalid temporal ordering where ordering is part of the contract.

Conceptual path:

```text
invalid observation
       │
       ▼
reject observation
       │
       ▼
preserve current model state
       │
       ▼
report failure
       │
       ▼
continue or stop according to runtime policy
```

An invalid observation should not silently alter model state.

---

### 4.2 Numerical Failure

Examples:

- NaN;
- infinite values;
- numerical overflow;
- invalid matrix operation;
- failure of an algorithm-specific numerical condition;
- non-finite model parameters.

Conceptual path:

```text
update
  │
  ▼
numerical validation
  │
  ├── valid ──→ commit
  │
  └── invalid
          │
          ▼
       reject update
          │
          ▼
   preserve last valid state
          │
          ▼
      report failure
```

The framework should not silently commit a numerically invalid model state.

---

### 4.3 Model or State Invariant Failure

Examples:

- violated dimensional invariant;
- invalid internal state;
- violated algorithm-specific constraint;
- invalid state transition.

Conceptual path:

```text
state transition
       │
       ▼
invariant validation
       │
       └── failure
              │
              ▼
         do not commit
              │
              ▼
      preserve/recover valid state
              │
              ▼
        report / escalate failure
```

If the model state itself cannot be trusted, the runtime should not blindly continue.

---

### 4.4 Memory or Allocation Failure

Examples:

- allocation failure;
- memory exhaustion;
- inability to obtain required device memory.

Conceptual path:

```text
allocation request
       │
       ├── success ──→ continue update
       │
       └── failure
              │
              ▼
          stop update
              │
              ▼
       preserve valid state
       where possible
              │
              ▼
        report explicit error
```

Whether execution can continue depends on the runtime and deployment environment.

---

### 4.5 Runtime or Hardware Failure

Examples:

- accelerator execution failure;
- device failure;
- synchronization failure;
- host/device communication failure;
- unrecoverable runtime error.

Conceptual path:

```text
runtime / hardware failure
          │
          ▼
       stop update
          │
          ▼
 preserve valid state if possible
          │
          ▼
     report explicit error
          │
          ▼
 continue, recover, isolate,
 or terminate according to
 deployment requirements
```

Hardware recovery must not be assumed to be equivalent to model rollback.

---

## 5. Failure Actions

A failure should result in an explicit action.

Possible actions are:

```text
failure
   │
   ├── invalid input
   │      └── reject input
   │
   ├── invalid update
   │      └── reject update
   │
   ├── transient / retryable
   │      └── retry where appropriate
   │
   ├── recoverable state failure
   │      └── restore / rollback / checkpoint
   │
   └── unrecoverable failure
          └── stop or isolate execution
```

These actions are conceptual categories, not a frozen API.

---

## 6. Rollback Is Not the Universal Response

Rollback is one possible recovery mechanism.

It should not automatically be applied to every failure.

For example:

```text
Invalid input
    → reject input

Invalid model update
    → reject update

Transient runtime failure
    → retry where safe

Recoverable state corruption
    → restore / rollback / checkpoint

Unrecoverable hardware or runtime failure
    → isolate or terminate
```

The correct action depends on:

- failure type;
- algorithm;
- state representation;
- execution environment;
- latency requirements;
- memory constraints;
- deployment requirements.

---

## 7. State Integrity

For a sequential stream:

```text
x₁ → successful update → state₁
x₂ → failed update    → state₁
x₃ → update           → state based on state₁
```

The framework must not silently produce:

```text
x₁ → state₁
x₂ → partial update → invalid state
x₃ → update from invalid state
```

This requirement applies particularly to:

- online learning;
- reinforcement learning;
- adaptive estimators;
- recurrent models;
- state-space models;
- continuous inference;
- continual learning;
- latency-sensitive deployments.

---

## 8. Update Atomicity

Seqvex should preserve the concept of an update boundary.

```text
        ┌──────────── Update boundary ────────────┐
        │                                          │
state_t ──→ computation ──→ validation ──→ commit │
        │                                          │
        └──────────────────────────────────────────┘
```

Before the commit point, the update is provisional.

After the commit point:

```text
state_t
   │
   ▼
 COMMIT
   │
   ▼
state_(t+1)
```

The committed state becomes the state visible to subsequent operations.

The implementation may use different mechanisms to achieve this property.

Possible mechanisms include:

- transactional state construction;
- copy-on-write;
- double buffering;
- checkpoint-and-restore;
- in-place mutation with rollback information;
- algorithm-specific recovery mechanisms.

No particular mechanism is currently mandated.

---

## 9. Streaming Semantics After Failure

Failure handling must preserve temporal and sequential semantics.

Consider:

```text
x₁ → update → state₁
x₂ → update → FAILURE
x₃ → update
```

The runtime must have a defined state for `x₃`.

The default conceptual behavior is:

```text
x₁ → state₁
x₂ → FAILURE
       │
       ▼
    state₁
       │
       ▼
x₃ → update from state₁
```

The framework should not silently advance the model to an unknown or partially updated state.

Whether `x₂` is retried, discarded, reprocessed, or causes execution to stop is a separate failure-policy decision.

---

## 10. Checkpointing

Long-running or higher-value workloads may eventually require checkpoints.

Conceptually:

```text
valid model state
       │
       ▼
   checkpoint
       │
       ▼
 continuous execution
       │
   ┌───┴────┐
   │        │
success   failure
   │        │
   ▼        ▼
continue   recover from
           valid checkpoint
```

Checkpointing should not necessarily occur on every observation.

The appropriate frequency depends on:

- model state size;
- update frequency;
- recovery requirements;
- latency requirements;
- storage cost;
- device placement;
- durability requirements.

The checkpoint format and persistence mechanism remain undecided.

---

## 11. Failure Reporting

A failed operation should be explicitly observable.

The eventual system should be able to determine:

```text
Did the operation succeed?
Did model state change?
Was the update rejected?
Was recovery performed?
Can execution continue?
What caused the failure?
```

Failure information may eventually include:

- failure category;
- affected operation;
- observation/update context;
- numerical diagnostics;
- device/runtime information;
- whether state was committed;
- whether recovery occurred;
- whether execution may continue.

The exact error hierarchy and API remain undecided.

---

## 12. Recovery and Determinism

Recovery may interact with deterministic execution.

Potential sources of differences include:

- retry order;
- parallel execution order;
- floating-point reduction order;
- random-number generation;
- asynchronous accelerator execution;
- checkpoint timing.

If deterministic replay is required, the recovery mechanism must eventually define how failed observations, retries, state restoration, and random state are handled.

This is an architectural consideration, not yet a frozen implementation requirement.

---

## 13. Failure Testing

Failure paths should be tested as deliberately as successful paths.

At minimum, tests should eventually cover:

```text
valid input
    → successful update
    → committed state

invalid input
    → rejected input
    → unchanged state

numerical failure
    → rejected update
    → unchanged state

state invariant failure
    → rejected update
    → valid state preserved

runtime failure
    → explicit error
    → defined continuation/recovery behavior
```

A particularly important invariant is:

```text
failed update
     ↓
committed state before failure
     =
committed state after failure
```

unless an explicit recovery mechanism intentionally restores a different known-valid state.

Failure testing should become part of the integration test suite as the corresponding runtime behavior becomes stable.

---

## 14. Performance and Recovery

Recovery mechanisms have costs.

Potential costs include:

- additional memory;
- state copying;
- checkpointing;
- synchronization;
- allocation;
- persistence;
- latency;
- device transfers.

Therefore:

```text
recovery requirement
       ↓
candidate mechanism
       ↓
measure cost
       ↓
compare against workload requirements
       ↓
select mechanism
```

Rollback or checkpointing should not be introduced solely because it is conceptually simple.

The implementation must eventually be evaluated against actual Seqvex workloads.

---

## 15. Architectural Requirements

The following requirements are established:

1. A failed update must not silently commit invalid state.
2. The last committed valid state must remain identifiable.
3. State-update boundaries must support safe failure handling.
4. Failure handling must preserve sequential and temporal semantics.
5. Recovery must be explicit rather than accidental.
6. Rollback is one possible recovery mechanism, not a universal requirement.
7. Failure must be observable by the caller or runtime.
8. Recovery mechanisms must eventually be evaluated against latency, memory, allocation, and correctness requirements.

---

## 16. Deliberately Deferred Decisions

The following remain open:

- exact state-update mechanism;
- transactional versus in-place updates;
- rollback implementation;
- checkpoint implementation;
- checkpoint frequency;
- checkpoint persistence;
- state validation strategy;
- retry policy;
- failure-policy API;
- recovery granularity;
- device-specific recovery;
- distributed recovery;
- deterministic replay after failure;
- interaction between recovery and model/optimizer state.

These decisions should be informed by actual Seqvex implementations, tests, benchmarks, profiling, and observed failure modes.

---

## 17. Guiding Principle

The overall failure-handling principle is:

```text
Detect
  ↓
Stop unsafe state transition
  ↓
Preserve last valid state where possible
  ↓
Classify failure
  ↓
Apply appropriate recovery/failure action
  ↓
Report explicitly
  ↓
Continue, recover, isolate, or terminate
```

> **Protect valid state first. Recover deliberately. Never silently continue from an unknown state.**
