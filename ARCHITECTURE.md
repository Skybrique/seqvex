# Seqvex Architecture

> **Status:** Early-stage architecture / living technical document  
> **Project:** Seqvex  
> **Language:** Rust  
> **License:** Apache-2.0  
> **Current release:** `0.1.0`

<p align="center">
  <img src="assets/architecture-overview.png" alt="Seqvex architecture" width="850">
</p>

## 1. Architectural motivation

Seqvex is being built from a specific problem rather than from a generic desire to create another ML library.

Many existing ML workflows are naturally expressed around:

```text
static dataset
     ↓
batch / mini-batch
     ↓
offline training
     ↓
model
     ↓
separate inference
```

That model is appropriate for many workloads.

Seqvex addresses a different class of systems:

```text
evolving observations
        ↓
sequential / temporal / non-IID
        ↓
continuous stream
        ↓
persistent model state
        ↓
learning and/or inference
        ↓
new observations
        ↺
```

The architectural question is:

> **What should an ML/RL framework look like when sequential, temporal, non-IID, stateful, continuously arriving data is a primary design assumption rather than an edge case?**

A second question follows:

> **How can that execution model retain enough control over memory, computation, and hardware placement to support efficient and predictable execution from conventional CPUs through accelerators and constrained deployments?**

Seqvex answers these questions incrementally through implementation, testing, benchmarking, and profiling.

## 2. Core architectural stance

The architecture currently rests on four principles:

1. **Sequential, temporal, and non-IID data is foundational.**
2. **Streaming / online execution is the primary computational context.**
3. **Single-observation execution is the fundamental semantic unit; bounded micro-batching is an optimization/capability, and larger batch execution is a secondary capability.**
4. **Execution semantics are separated from compute placement.**

The hierarchy is intentional:

```text
semantic unit
    ↓
single observation
    ↓
primary execution context
    ↓
streaming / online
    ↓
optional execution optimizations
    ↓
bounded micro-batch
    ↓
secondary capability
    ↓
larger batch operations
```

This does **not** mean Seqvex cannot process large datasets or high-dimensional observations. It means that dataset size is not the organizing semantic principle. Sequential state, temporal context, ordering, and evolving computation are.

Batching is therefore not forbidden.

> **Batch capability is retained; batch-first semantics are not the architectural foundation.**

## 3. Data semantics

Seqvex treats the following as important properties of the computational problem:

- temporal ordering;
- sequential dependence;
- non-IID observations;
- evolving distributions;
- persistent state;
- continuous observations;
- feedback between system outputs and future observations.

The framework should not silently discard those properties simply to fit a static dataset abstraction.

## 4. Scope by process responsibility

Seqvex differentiates its scope by **process responsibility rather than by individual function**.

It owns the computational stages from:

- ML/RL preprocessing and representation;
- model computation;
- training and learning;
- validation;
- inference;
- numerical computation required by ML/RL;
- execution/runtime mechanisms.

General-purpose ingestion, storage, databases, ETL, generic data manipulation/cleaning, exploratory analysis, visualization, and domain/business logic remain outside the framework.

The boundary is based on the **role of a computation in the ML/RL pipeline**, not on its name. For example, one-hot encoding, normalization/standardization, PCA, rolling statistics, and online statistics may be implemented by Seqvex when they form part of an ML/RL computational method or pipeline.

Conceptually:

```text
External data ecosystem
        │
        ├─ ingestion
        ├─ storage / databases
        ├─ general data manipulation
        ├─ general-purpose cleaning
        └─ exploratory analysis / visualization
        │
        ▼
Seqvex ML/RL pipeline
        │
        ├─ preprocessing / representation
        ├─ model computation
        ├─ training / learning
        ├─ validation
        └─ inference / execution
```

## 5. Streaming-first execution

<p align="center">
  <img src="assets/seqvex-high-level-flow.png" alt="Seqvex high-level flow" width="680">
</p>

Streaming is the primary execution context.

Conceptually:

```text
observation_t
      ↓
model / state
      ↓
prediction / action
      ↓
state update
      ↓
observation_(t+1)
      ↓
...
```

A stream can originate from live sensors, network events, application events, historical data replay, databases, files, or other external systems.

Therefore:

> **Streaming describes execution semantics, not necessarily the physical origin of the data.**

A historical dataset can be replayed as a stream. A live system can produce an effectively unbounded stream.

## 6. Execution strategies

<p align="center">
  <img src="assets/execution-modes.png" alt="Seqvex execution modes" width="820">
</p>

### 6.1 Single observation

One observation is processed immediately.

Relevant to:

- low-latency inference;
- online learning;
- adaptive estimators;
- reinforcement learning;
- event-driven systems;
- constrained edge systems.

The framework should not require an artificial batch merely to make an observation executable.

### 6.2 Streaming / online

The model maintains state while observations arrive:

```text
x₁ → state₁
x₂ → state₂
x₃ → state₃
...
```

The state may represent model parameters, hidden state, running statistics, optimizer state, policy/value state, or other algorithm-specific information.

### 6.3 Optional bounded micro-batching

Micro-batching accumulates a bounded number of observations:

```text
x₁
x₂
x₃
x₄
 ↓
[x₁ x₂ x₃ x₄]
 ↓
vectorized / parallel computation
```

It may improve:

- SIMD utilization;
- cache efficiency;
- CPU parallelism;
- accelerator utilization;
- arithmetic intensity;
- throughput.

Micro-batching is an **optimization mechanism**, not the conceptual foundation of Seqvex.

It must not silently violate:

- temporal ordering;
- causality;
- state-transition semantics;
- algorithmic online-learning semantics.

The buffering policy, batch size, scheduler, and automatic batching heuristics remain undecided.

The foundation `process_batch` in `src/foundation/state` remains an **unbounded,
stop-on-first-failure ordered fold** — not a bounded micro-batch executor. The
vertical slices also provide algorithm-local bounded batching: LR, Decision Tree,
and KNN evaluate independent predictions over a bounded group (classification and
training remain out of scope), while GRU and RLS expose bounded ordered reference
APIs (`process_batch_reference`) that preserve sequential state-transition
semantics over borrowed observations. Grouping dependent updates does not
authorize independent or parallel updates. There is still no generic micro-batch
executor/runtime, buffering policy, scheduler, or automatic execution selection,
and none is added until the vertical slices demonstrate a recurring need
(`docs/ML_VERTICAL_SLICES.md`).

### 6.4 Larger batch operations

Some algorithms legitimately need large-scale batch computation.

Seqvex should support those operations where useful, including historical/offline computation when aggregation materially benefits an algorithm or hardware target.

However:

> **Larger batch execution is a capability, not the semantic foundation of Seqvex.**

A historical dataset may be processed sequentially, with bounded micro-batches, or with larger batches depending on the algorithm and measured execution trade-offs.

## 7. Training and inference

Training and inference can both exist inside the streaming computational context.

### 7.1 Historical training

Historical data can be replayed:

```text
historical source
      ↓
sequential replay
      ↓
x₁ → update
x₂ → update
...
xₙ → update
```

Alternatively, the same historical source can provide bounded or larger batches where the algorithm benefits from them.

Offline training is therefore an available execution strategy, not a requirement that production systems operate with prolonged downtime.

### 7.2 Online / continual training

```text
live stream
    ↓
observation
    ↓
prediction
    ↓
learning update
    ↓
next observation
```

There does not need to be a fixed dataset boundary or conventional epoch structure.

Future online/continual learning, replay, bounded-update, and model-deployment/model-swap strategies remain open implementation questions.

### 7.3 Inference

Inference can operate continuously:

```text
observation → prediction → next observation → prediction → ...
```

or use bounded batching where throughput and hardware utilization justify it.

### 7.4 Continual operation

A deployment may combine historical initialization with live adaptation:

```text
historical data
      ↓
initial model
      ↓
deployment
      ↓
live stream
      ↓
inference + optional online updates
      ↺
```

The architecture does not assume that production retraining must be performed as a conventional offline batch job.

## 8. Compute placement

Compute placement answers:

> **Where does the computation execute?**

It does not answer:

> **What does the computation mean?**

Potential placement targets include:

- CPU;
- GPU;
- accelerator;
- heterogeneous resources.

The same execution semantics should be capable of using different implementations.

## 9. Single-device residency and heterogeneous execution

"Heterogeneous" does not mean constant CPU↔GPU movement.

The preferred principle is:

> **Keep model state and frequently used data resident on one device when possible. Move data only when the measured computational benefit exceeds transfer and synchronization costs.**

The architecture must therefore preserve the ability to reason about:

- locality;
- memory hierarchy;
- layout;
- alignment;
- SIMD/vectorization;
- bandwidth;
- synchronization;
- host/device movement;
- accelerator residency.

## 10. Hardware architecture

<p align="center">
  <img src="assets/hardware-abstraction.png" alt="Seqvex hardware and data-movement model" width="850">
</p>

Hardware optimization is a major capability of Seqvex.

Relevant optimization surfaces include:

- SIMD/vectorization;
- cache locality;
- memory layout;
- alignment;
- allocation;
- memory bandwidth;
- branch behavior;
- synchronization;
- CPU affinity;
- NUMA locality;
- accelerator kernels;
- host/device transfers;
- compiler/LLVM behavior;
- hardware-specific optimization.

These are **optimization surfaces**, not guarantees that every implementation will use every technique.

## 11. Memory hierarchy

A simplified CPU hierarchy is:

```text
registers
   ↓
L1
   ↓
L2
   ↓
L3 / LLC
   ↓
RAM
```

Accelerators introduce their own local memory hierarchy and device memory.

The architectural implication is:

> **Memory access is not uniformly expensive.**

Future implementations must remain capable of optimizing for locality, layout, bandwidth, and data movement.

## 12. Constrained and bare-metal direction

Seqvex should preserve a path toward:

- embedded systems;
- low-power devices;
- constrained edge systems;
- specialized appliances;
- minimal-runtime deployments;
- potentially bare-metal environments.

This is a **future architectural direction**, not a claim that the current `0.1.0` release provides certified bare-metal or hard-real-time support.

Seqvex is currently **std-first** rather than universally `no_std`.

## 13. Memory and storage architecture

Storage is deliberately not frozen.

Still undecided:

- tensor representation;
- buffer representation;
- ownership model;
- allocator;
- memory pools;
- arena allocation;
- device-memory ownership;
- pinned memory;
- zero-copy;
- unified memory;
- contiguous/strided layout;
- row/column-major policy;
- alignment guarantees;
- sparse representation.

Storage is a high-reversal-cost decision.

The correct sequence is:

```text
implement real workloads
        ↓
observe access patterns
        ↓
measure allocations / locality / transfers
        ↓
identify recurring constraints
        ↓
design abstraction
```

## 14. Layered architecture

The conceptual stack is:

```text
┌──────────────────────────────────────────────┐
│ Applications / User Systems                  │
│ Sensors • services • robotics • analytics    │
│ Data ingestion / storage / EDA               │
│ Outside Seqvex core                          │
└──────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────┐
│ High-Level Seqvex Interfaces                 │
│ Models • Training • Inference • Streaming • RL│
└──────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────┐
│ ML / RL Layer                                │
│ Classical • Online • Sequential • RL         │
└──────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────┐
│ Numerical + Execution Layer                  │
│ Math • state • streaming • runtime           │
└──────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────┐
│ Device / Hardware Layer                      │
│ CPU • SIMD • GPU • accelerators              │
└──────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────┐
│ Physical Resources                           │
│ Cache • RAM • VRAM • interconnect • edge     │
└──────────────────────────────────────────────┘
```

These layers describe responsibility and dependency direction conceptually. They do not yet define exact Rust traits or crates.

## 15. Module structure

<p align="center">
  <img src="assets/module-structure.png" alt="Seqvex module structure" width="850">
</p>

A possible future workspace may contain focused components for:

- core types/traits/errors;
- numerical computation;
- online learning;
- sequential/temporal models;
- supervised learning;
- unsupervised learning;
- reinforcement learning;
- validation;
- execution/runtime;
- device backends;
- narrowly scoped utilities.

These names are illustrative.

A component/crate should be created only when a meaningful boundary has emerged.

## 16. Numerical computation for ML/RL

Numerical computation in Seqvex is scoped to what is required by its ML/RL methods, preprocessing and representation, validation procedures, and execution path.

The numerical foundation should be developed through small, understandable implementations.

Potential foundations include:

- statistics;
- linear algebra;
- online numerical computation;
- ML/RL-specific numerical operations.

Mature low-level Rust crates may later be used where they provide a better engineering trade-off.

## 17. ML and model families

Potential algorithm families include:

### Classical ML

- linear regression;
- ridge regression;
- logistic regression;
- k-nearest neighbors;
- k-means;
- Naive Bayes;
- trees;
- random forests.

### Online ML

- online linear/logistic learning;
- online gradient methods;
- recursive least squares;
- adaptive estimators.

### Sequential/deep learning

- 1D convolution;
- temporal convolution;
- RNN;
- GRU;
- LSTM;
- attention;
- transformers;
- efficient attention;
- state-space sequence models.

### Reinforcement learning

RL naturally reinforces the need for sequential state, action, reward, environment interaction, and continuous execution.

Exact policy, value-function, environment, replay, and training abstractions remain open.

## 18. Temporal validation

Temporal workloads require validation that respects time.

Potential capabilities:

- chronological train/test separation;
- rolling-window evaluation;
- expanding-window evaluation;
- walk-forward evaluation;
- purged cross-validation;
- embargo;
- group-aware temporal splitting.

The framework should make temporal leakage and accidental IID assumptions easier to detect.

## 19. Automatic execution selection and explicit user control

Seqvex may eventually provide two execution-control modes:

```text
Explicit
    user selects execution policy
        ↓
    Seqvex executes that policy

Auto
    user permits automatic selection
        ↓
    Seqvex evaluates relevant workload / workflow /
    execution / hardware characteristics
        ↓
    selects an available execution strategy
```

The intended design principle is:

> **Automatic selection should assist execution choice, not remove user control.**

A user must be able to override automatic selection through an explicit, simple policy.

Potential signals may include:

- observation frequency;
- statefulness and temporal dependence;
- workload size and dimensionality;
- latency requirements;
- throughput requirements;
- computational intensity;
- memory behavior;
- available CPU/GPU/accelerator resources;
- transfer/synchronization cost;
- whether online learning or historical replay is occurring;
- whether bounded micro-batching preserves the algorithm's semantics.

Automatic selection is **future architecture only**.

It is not currently implemented, and the scheduler, policy representation, heuristics, explainability, fallback behavior, and hardware-selection mechanism remain open.

No automatic execution abstraction should be implemented until multiple real execution strategies and workloads provide enough evidence to define a stable contract.

This future feature must preserve:

- explicit user override;
- temporal ordering;
- state atomicity;
- deterministic behavior where promised;
- hardware-aware placement freedom.

## 20. Performance methodology

Performance should be treated empirically.

Measure:

- per-observation latency;
- tail latency;
- throughput;
- update cost;
- allocation count;
- peak memory;
- cache behavior;
- memory bandwidth;
- SIMD utilization;
- device utilization;
- transfer time;
- synchronization overhead.

Profile before generalizing.

The useful question is:

> **What resource limits this workload, and does the proposed change address that limit?**

## 21. Predictability and real-time considerations

Seqvex may eventually serve latency-sensitive and potentially safety-relevant environments.

The architecture should preserve the possibility of bounded and predictable execution.

However, do not convert future goals into present guarantees.

Distinguish clearly between:

```text
architectural direction
implementation capability
measured property
formal guarantee
```

## 22. Reference and production implementations

Seqvex may intentionally contain both:

```text
reference implementation
        ↓
mathematical clarity / semantic oracle
        ↓
production implementation
        ↓
measured optimization
```

A production path must preserve the reference path's observable semantics.

The GRU allocation-free execution path is an example of this principle. Its reusable workspace belongs to an algorithm-local execution context (`GruExecutor`), not to the model: `Gru` holds only immutable parameters and configuration, and the generic `StreamingExecutor` borrows the model immutably (`&M`). One immutable model may therefore drive several independent execution contexts, each owning its own State and (for GRU) its own private per-execution workspace. The former model-owned workspace and `&mut M` borrow were removed. Do not propagate the GRU-specific workspace to other models until a recurring requirement is demonstrated.

## 23. Architectural principles

Current principles:

1. Sequential/temporal/non-IID computation is foundational.
2. Single observation is the fundamental semantic unit.
3. Streaming/online execution is the primary computational context.
4. Micro-batching is an optimization/capability, not the semantic foundation.
5. Larger batch execution is a secondary capability where useful.
6. Execution semantics are separate from compute placement.
7. State transitions must preserve valid committed state across failures.
8. Heterogeneous memory/device locality must remain possible.
9. Hardware-specific optimization must be evidence-driven.
10. Storage, ownership, allocator, scheduler, backend, and device abstractions remain deliberately open until demonstrated requirements justify them.
11. Automatic execution selection is a future capability with explicit user override; it is not a current implementation commitment.

## 24. Deferred decisions

The following remain deliberately open until implementation experience provides evidence:

- exact tensor/storage representation;
- memory ownership model;
- allocator architecture;
- device abstraction;
- backend abstraction;
- synchronization model;
- execution scheduler;
- automatic execution-selection mechanism;
- graph/operator representation;
- exact crate boundaries;
- exact model trait hierarchy;
- serialization format;
- plugin architecture;
- GPU backend strategy;
- unified memory strategy;
- zero-copy strategy;
- sparse representation;
- distributed execution;
- multi-node execution;
- exact RL abstractions;
- exact deep-learning abstractions.

> **Deferred means intentionally deferred.**

## 25. Dependency and abstraction discipline

Every abstraction introduces conceptual complexity, maintenance cost, API surface, compile-time consequences, potential performance implications, and future compatibility constraints.

> **Prefer the smallest abstraction that accurately represents a demonstrated recurring requirement.**

Do not abstract for hypothetical reuse or optimize for theoretical elegance at the expense of understanding.

## 26. Documentation discipline

Documentation should record:

- what was decided;
- why it was decided;
- supporting evidence;
- uncertainty;
- material rejected alternatives;
- what would cause a decision to be revisited.

Use labels where appropriate:

- **Current**
- **Tentative**
- **Deferred**
- **Experimental**
- **Measured**
- **Not yet demonstrated**

## 27. Integration discipline

Before integrating a meaningful change:

```text
Understand
   ↓
Implement
   ↓
Format
   ↓
Test
   ↓
Clippy
   ↓
Benchmark/profile if relevant
   ↓
Review architecture impact
   ↓
Update documentation
   ↓
Integrate
```

A change is not complete merely because it compiles.

## 28. Current development priority

Build Seqvex from its foundations outward:

1. Observation
2. Ordering / sequence
3. Streaming semantics
4. State
5. State transition
6. Failure atomicity
7. Minimal numerical primitives
8. Single-observation semantics
9. Bounded micro-batch semantics where demonstrated useful

Do not begin by implementing the full ML/RL framework.

## 29. Success criteria

Early success is demonstrated by clear semantics, understandable Rust, explicit invariants, strong tests, controlled failure behavior, measured performance, justified abstractions, useful modular boundaries, preserved hardware flexibility, and a credible path toward streaming and constrained execution.

It is not measured by number of crates, algorithms, abstractions, generated code, dependencies, premature GPU support, or theoretical performance claims.

## 30. Guiding rules

1. **Understand before abstracting.**
2. **Streaming-first does not mean batchless.**
3. **Single observation is the fundamental semantic unit.**
4. **Micro-batching is an optimization, not the semantic foundation.**
5. **Separate computation semantics from hardware placement.**
6. **Treat state transitions as explicit operations with failure boundaries.**
7. **Preserve valid state when an update fails.**
8. **Keep heterogeneous memory and hardware locality possible.**
9. **Remain std-first while preserving a practical path toward constrained/no_std foundations.**
10. **Define scope by process responsibility, not function names.**
11. **Create modules and crates only when meaningful boundaries emerge.**
12. **Measure performance before optimizing.**
13. **Treat hardware-specific optimization as evidence-driven.**
14. **Do not turn future goals into present guarantees.**
15. **Do not silently resolve deferred architectural decisions.**
16. **Any change that could materially constrain hardware-aware execution, heterogeneous memory/device placement, accelerator offload, or Rust-native low-level optimization requires CRITICAL ARCHITECTURE REVIEW before implementation.**
17. **Prefer experiments when architecture is uncertain.**
18. **Let implementation evidence shape the architecture.**

## 31. Relationship to other documentation

| Document | Purpose |
|---|---|
| `README.md` | Project identity, vision, scope, and public orientation |
| `ARCHITECTURE.md` | Architectural reasoning, system structure, and design decisions |
| `docs/DEVELOPMENT.md` | Development rules for humans and AI |
| `FEATURE_DEVELOPMENT.md` | Operational feature-development runbook (lifecycle, decomposition, traceability, templates) |
| `docs/ROADMAP.md` | Current development direction and phase planning |
| `docs/FAILURE_AND_RECOVERY.md` | Failure paths, state integrity, and recovery principles |
| `CONTRIBUTING.md` | Contributor participation, classification taxonomy, and contribution process |

If another document conflicts with this architecture, determine whether the conflict is an outdated document, a deliberate architectural change, or a missing clarification. Do not silently choose one interpretation.

## Final principle

> **Seqvex should be built by learning the problem deeply, implementing the smallest understandable mechanism, measuring its behavior, and allowing evidence—not speculation—to determine the architecture.**
