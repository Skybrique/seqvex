# Streaming Execution

The `execution` module defines **when and how sequential computation is executed** in Seqvex.

It sits between the caller and the underlying stateful model.

The execution layer does not define the mathematical behavior of an ML algorithm. Instead, it coordinates the application of a model's state-transition function over observations.

The fundamental relationship is:

$$
\text{observation}
\rightarrow
\text{model transition}
\rightarrow
\text{new committed state}
$$

For a sequential model such as a GRU:

$$
(x_t,h_{t-1})
\rightarrow
h_t
$$

The execution layer repeatedly drives this transition while preserving ordering, state semantics, and failure atomicity.

---

# 1. Purpose

Seqvex is designed around sequential, temporal, non-IID, and streaming workloads.

The execution hierarchy is:

```text
single observation
        ↓
streaming / online execution
        ↓
optional bounded micro-batching
        ↓
optional larger batch execution
```

The first two are foundational to Seqvex's semantics. Micro-batching is an optimization/capability. Larger batch execution is a secondary capability for workloads where aggregation is useful.

The execution layer provides mechanisms for:

- driving sequential model execution;
- preserving observation order;
- maintaining committed execution state;
- exposing single-observation execution;
- processing ordered streams;
- supporting state reset;
- preserving state atomicity when transitions fail.

---

# 2. Execution vs. Model Semantics

A central architectural distinction is:

> **The model defines what computation means. The execution layer defines when and how that computation is driven.**

For example, a GRU defines:

$$
(x_t,h_{t-1})
\rightarrow
h_t
$$

The execution layer does not redefine the GRU equations.

It invokes the model transition for each observation, preserving the state relationship between calls.

This separation allows different sequential models to use the same foundational execution semantics.

---

# 3. The `StateModel` Boundary

The execution layer operates against the `StateModel` abstraction defined in the foundation layer.

Conceptually:

$$
(\text{state},\text{observation})
\rightarrow
\text{candidate new state}
$$

The model is responsible for calculating the transition.

The execution layer is responsible for driving it.

```text
┌─────────────────────────────┐
│        Execution Layer      │
│                             │
│  When/how observations run  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        StateModel            │
│                             │
│  What the transition means  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Model State            │
│                             │
│  Current committed state    │
└─────────────────────────────┘
```

The execution layer therefore does not need to know whether the model is:

- a GRU;
- another recurrent model;
- an online statistical model;
- an anomaly detector;
- an RL state transition;
- or another future sequential model.

---

# 4. Streaming as the Primary Execution Context

A stream is an ordered sequence:

$$
x_1,x_2,x_3,\ldots,x_n
$$

The execution layer processes observations in order:

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
\cdots
\rightarrow
x_n
$$

with state transitions:

$$
h_0
\rightarrow
h_1
\rightarrow
h_2
\rightarrow
\cdots
\rightarrow
h_n
$$

The state produced by one successful transition becomes the state supplied to the next transition.

Therefore, execution order is semantically significant.

---

# 5. Single-Observation Execution

The fundamental operation is processing one observation:

$$
(\text{current state},x_t)
\rightarrow
\text{next state}
$$

In the current implementation, `StreamingExecutor::process_one` performs the reference transition path.

A simplified sequence is:

```text
current committed state
        │
        ▼
    observation
        │
        ▼
   StateModel::update
        │
        ▼
 candidate state
        │
        ▼
    validation
        │
        ▼
 committed state
```

The second call begins from the state successfully produced by the first call.

---

# 6. Stream Execution

A sequence is processed as a sequential fold:

$$
h_0
\xrightarrow{x_1}
h_1
\xrightarrow{x_2}
h_2
\xrightarrow{x_3}
\cdots
\xrightarrow{x_n}
h_n
$$

The observations are not assumed to be independent.

For a stateful model:

$$
h_t=f(h_{t-1},x_t)
$$

Therefore:

$$
h_t
$$

depends on both the current observation and the state produced by previous observations.

---

# 7. Observation Ordering

Observation order is part of the execution semantics.

For observations `A` and `B`:

$$
A\rightarrow B
$$

is generally different from:

$$
B\rightarrow A
$$

because the state entering the second transition is different.

The execution layer must preserve observation order unless an explicit execution mode defines another semantic while preserving the model's required temporal contract.

---

# 8. State Ownership

## 8.1 Reference topology

The `StreamingExecutor` owns the **committed execution state** for one execution
context and borrows the model **immutably**:

```text
StreamingExecutor
├── immutable model borrow (&Model)
└── committed State (one per execution)
```

Because the model is only read during execution, one immutable model may be
shared by several independent execution contexts, each owning its own State.
The execution layer itself holds no reusable scratch: `StateModel` does not
model a workspace.

## 8.2 Reference vs optimized execution

Where an algorithm needs reusable scratch, that scratch belongs to an
algorithm-local execution context, not to the model:

```text
reference path
    ↓
StateModel::update (value-returning)
    ↓
semantic reference

optimized path (GRU only, currently)
    ↓
GruExecutor: &Gru + State + private reusable workspace
    ↓
allocation-free steady state
```

The optimized path is local to the GRU and is not a general `StateModel`
redesign. Per-execution State and per-execution Workspace are never shared
across live executions.

The former model-owned workspace and the resulting `&mut M` executor borrow were
removed; the executor no longer requires exclusive access to the model.

---

# 9. State Transition and Atomicity

A stateful execution path must not silently corrupt the committed state.

The intended transition is:

$$
\text{previous valid state}
\rightarrow
\text{candidate state}
\rightarrow
\text{validation}
\rightarrow
\text{commit}
$$

If validation fails:

$$
\text{previous valid state}
\rightarrow
\text{failure}
$$

The previous committed state remains active.

This applies to both the reference and optimized GRU paths.

Scratch/workspace state must remain separate from committed model state so a failed computation cannot partially commit invalid results.

---

# 10. Failure Behavior

Failure handling is part of execution semantics.

Examples include:

- invalid observation dimensions;
- non-finite input;
- invalid model parameters;
- invalid candidate state;
- numerical failure;
- future device/runtime failures.

The execution layer should make the failure boundary explicit.

A failed transition must not silently become the next committed state.

---

# 11. Reset

Reset defines an explicit boundary in the execution history.

Conceptually:

```text
state₀
  ↓
x₁
  ↓
state₁
  ↓
x₂
  ↓
state₂
  ↓
reset
  ↓
state₀
```

For a GRU, reset currently returns the hidden state to its defined initial state.

Reset semantics are separate from model computation and must remain explicit.

---

# 12. GRU Integration

The GRU demonstrates the execution architecture.

Reference path:

```text
StreamingExecutor (&Gru)
        ↓
StateModel::update
        ↓
Gru::compute
        ↓
new hidden state
```

Optimized path (algorithm-local):

```text
GruExecutor (&Gru + State + workspace)
        ↓
process_one_optimized
        ↓
compute_in_place
        ↓
reusable GRU workspace (private, per execution)
        ↓
validated State commit
```

The two paths are semantically equivalent and bitwise-tested.

The optimized path eliminates steady-state allocations by reusing three
hidden-dimension workspace buffers owned by the `GruExecutor`, not by `Gru`.
This is a **measured local optimization**, not a decision that Seqvex will use
model-owned or framework-wide workspace abstractions.

---

# 13. Execution vs. Compute Placement

Execution semantics answer:

> **What temporal/execution operation is being performed?**

Compute placement answers:

> **Where does that operation execute?**

Potential placement includes:

```text
CPU
GPU
accelerator
heterogeneous resources
```

The architecture must preserve this separation.

A CPU implementation and an accelerator implementation should be able to express the same sequential/stateful semantics without forcing the execution layer to become a device-specific abstraction prematurely.

---

# 14. Streaming, Micro-Batching, and Batch

Seqvex distinguishes semantic execution from computational aggregation.

### Single observation

The fundamental semantic unit.

### Streaming / online

The primary execution context for sequential/stateful workloads.

### Bounded micro-batch

An optimization/capability that may improve:

- SIMD/vectorization;
- cache locality;
- CPU parallelism;
- accelerator utilization;
- throughput.

Micro-batching must preserve temporal and state semantics.

There is currently **no generic micro-batch executor**, and one is not
introduced until several algorithms demonstrate a recurring need. The
foundation's `process_batch` is **reference ordered-fold semantics** — an
unbounded, stop-on-first-failure fold — not a bounded micro-batch
implementation. A bounded micro-batch is introduced per algorithm, only where
that algorithm's semantics permit it (`docs/ML_VERTICAL_SLICES.md`).

### Larger batch

A secondary capability for algorithms or workloads that benefit from larger aggregation, including historical/offline computation.

It is not the conceptual foundation of Seqvex.

---

# 15. Automatic Execution Selection

Seqvex may eventually support:

```text
Explicit
    user selects execution policy
        ↓
    execute that policy

Auto
    user permits automatic selection
        ↓
    inspect relevant workload/workflow/hardware signals
        ↓
    select an available execution strategy
```

A user must always be able to override Auto with an explicit policy.

Potential selection signals include:

- observation rate;
- workload dimensionality and size;
- temporal/state dependence;
- latency target;
- throughput target;
- computational intensity;
- memory behavior;
- CPU/GPU/accelerator availability;
- transfer and synchronization costs;
- online learning versus historical replay;
- whether micro-batching preserves the algorithm's semantics.

Automatic selection is **not implemented**.

The scheduler, policy contract, heuristics, explainability, fallback behavior, and device-selection mechanism remain deferred.

No generic automatic scheduler should be introduced until multiple real execution strategies and workloads provide enough evidence to define a stable abstraction.

---

# 16. Current Implementation

The current execution module provides:

- stateful single-observation execution;
- ordered stream execution;
- committed State ownership (one per execution);
- reset;
- failure-aware progression;
- the generic `StateModel` reference path;
- an algorithm-local GRU optimized path (`GruExecutor`).

The `StreamingExecutor` borrows the model immutably (`&M`), so independent
execution contexts can share one model. Reusable scratch, where an algorithm
requires it, belongs to that algorithm's execution context (GRU today) and is
never model-owned.

---

# 17. Performance Considerations

Execution performance may be affected by:

- allocations;
- state copies;
- data layout;
- cache locality;
- SIMD;
- synchronization;
- dispatch;
- memory bandwidth;
- CPU/GPU transfers;
- accelerator launch overhead;
- batching strategy.

The correct optimization sequence is:

```text
correctness
    ↓
benchmark
    ↓
profile
    ↓
identify bottleneck
    ↓
optimize
    ↓
benchmark again
```

The current GRU workspace optimization demonstrates allocation elimination and a measured small-dimension streaming benefit. It should not be generalized beyond the evidence.

---

# 18. Testing

Execution tests should protect:

- observation ordering;
- state progression;
- reset semantics;
- failure atomicity;
- repeated streaming behavior;
- reference/production equivalence;
- deterministic behavior where promised;
- bounded micro-batch semantics when implemented.

Performance tests should separately report:

- latency;
- throughput;
- allocations;
- bytes allocated;
- memory footprint;
- relevant variance/noise.

A performance optimization is not considered universally beneficial merely because one benchmark improves.

---

# 19. Current Scope

Current scope includes:

- stateful single-observation execution;
- sequential stream execution;
- explicit state ownership;
- reset;
- failure-aware progression;
- `StateModel`;
- CPU execution;
- a provisional GRU production hot path.

Current scope does **not** imply a final generic workspace, scheduler, GPU backend, automatic execution policy, or universal zero-allocation guarantee.

---

# 20. Deferred Capabilities

The following remain deferred until demonstrated requirements justify them:

- bounded micro-batch execution as a generalized API;
- general batch execution;
- asynchronous execution;
- execution scheduling;
- automatic execution selection;
- GPU/accelerator scheduling;
- graph execution;
- pipeline parallelism;
- automatic synchronization;
- zero-copy policies;
- generalized heterogeneous-memory policies;
- generic reusable workspace infrastructure.

---

# 21. Architectural Guard

Any implementation, optimization, or API change that could materially constrain:

- heterogeneous memory/device placement;
- accelerator offload;
- hardware-aware execution;
- per-stream versus model-owned state/scratch;
- Rust-native low-level optimization;

must undergo **CRITICAL ARCHITECTURE REVIEW before implementation**.

The former `StreamingExecutor<'_, M>` mutable model borrow was an example of this
guard; it was removed in favour of an immutable model borrow with
execution-owned State and algorithm-local Workspace.

---

# 22. Relationship to Other Documentation

- `ARCHITECTURE.md` — system-wide architectural reasoning.
- `docs/DEVELOPMENT.md` — canonical human/AI development contract.
- `src/foundation/state/README.md` — state-transition and failure semantics.
- `src/foundation/observation/README.md` — observation and ordering semantics.
- `src/models/recurrent/README.md` — GRU mathematics, implementation, and reference/production distinction.

> **The execution layer coordinates stateful computation; it should not silently become the owner of every future runtime abstraction.**
