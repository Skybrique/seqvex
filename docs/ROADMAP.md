# Seqvex Development Roadmap

> **This roadmap describes current development direction, not a fixed
> architecture or delivery commitment.**

The roadmap separates **what Seqvex is currently trying to learn/build**
from the more durable engineering principles in `docs/DEVELOPMENT.md`.

Future phases may change as implementation experience, benchmarks,
failure analysis, and architectural evidence invalidate earlier assumptions.

## How to read this roadmap

- **Current** — actively being implemented or investigated.
- **Next** — likely follow-on work, but not implementation-locked.
- **Later** — directional only.
- **Deferred** — intentionally not decided.

A phase describes the **problem and capability to establish**, not a
mandatory crate structure or implementation technology.

> **Plan the problem space broadly; commit to implementation narrowly.**

## Phase 1 — Foundations

**Status: Current**

Phase 1 establishes the smallest semantic foundation on which later ML/RL
functionality can depend.

### 1. Observation

Establish what Seqvex means by an observation and how observations can be
represented without prematurely fixing a universal tensor/storage model.

Focus on representation, type safety, ownership/borrowing behavior,
representation boundaries, and clear invalid-input behavior.

### 2. Ordering and sequence

Establish temporal ordering as an explicit concept.

Focus on sequence position/order, ordering invariants, repeated observations,
and preservation of sequence semantics.

### 3. Streaming semantics

Establish streaming as the semantic foundation.

Focus on single-observation processing, stateful sequential processing,
bounded micro-batch semantics where useful, ordering/causality, and the
distinction between execution semantics and compute placement.

Do not prematurely implement a general scheduler or backend architecture.

### 4. State

Treat model/runtime state as a first-class concept.

Focus on valid state, state transitions, state ownership, candidate next
state, and preservation of valid state after failed updates.

The exact storage/ownership abstraction remains open until implementation
experience justifies one.

### 5. Failure atomicity

Establish the minimum failure contract for stateful operations.

Focus on invalid transitions, preserving prior valid state, structured
error propagation, deterministic behavior where promised, and failure-path
tests.

Do not build a universal recovery framework before concrete failure modes
require one.

### 6. Numerical foundation

Implement only numerical primitives justified by current requirements.

Initial candidates include:

- sum;
- mean;
- variance;
- dot product;
- norm;
- online statistics.

Before optimization, establish contracts for empty inputs, invalid
dimensions, NaN/Infinity behavior, population/sample definitions where
applicable, numerical stability, and error-versus-panic semantics.

Use understandable reference implementations first.

### 7. Testing and development discipline

Phase 1 should demonstrate:

- focused local tests;
- global integration tests under `tests/`;
- failure-path tests;
- deterministic behavior where promised;
- formatting and Clippy baseline;
- benchmark infrastructure only where a meaningful measurement question exists.

### Phase 1 exit evidence

Phase 1 should not be considered complete because a list of files exists.

Useful evidence includes:

- observation semantics are understandable;
- ordering is explicit;
- streaming semantics are testable;
- state transitions preserve valid state on failure;
- numerical contracts are explicit and tested;
- unnecessary global helpers have not been introduced;
- performance claims are backed by measurements where made;
- the foundation supports at least one small end-to-end experiment.

## Phase 2 — Streaming and Online Learning

**Status: Next / high-level**

Establish a usable streaming learning layer built on Phase 1 semantics.

Likely areas include online estimators, incremental learning,
rolling/expanding statistics, temporal validation, stateful training loops,
bounded micro-batch learning, and model update lifecycle.

### Planned concrete slice

The first concrete online estimator is **Recursive Least Squares (RLS)** —
ordered updates of adaptive `(w, P)` state with atomic commit and
numerical-stability handling (Issue #26). RLS is deliberately scheduled last:
it is the strongest evidence source for per-stream adaptive state and for
committing two coupled state components together.

Exact algorithms, traits, crate boundaries, and runtime architecture remain
open. The slice sequence and the evidence each algorithm is expected to supply
are recorded in `docs/ML_VERTICAL_SLICES.md`.

## Phase 3 — Core ML Algorithms

**Status: Later / directional**

Introduce a first useful set of ML algorithms whose semantics fit the
streaming-first model.

Potential areas include supervised learning, unsupervised learning,
sequential/temporal models, representation learning, and incremental
variants where justified.

### Planned concrete slices

The first classic-ML slices are **Linear Regression** (closed-form prediction;
Issue #23), **Decision Tree** (read-only traversal; Issue #24), and
**K-Nearest Neighbors** (brute-force query; Issue #25). They are intended to
establish the simplest ownership topology — immutable, shareable model data
with no model-owned scratch — and the first micro-batch pattern for independent
observations. Bounded micro-batch over the existing GRU **reference** path
(Issue #34) follows.

Algorithm selection should follow actual Seqvex use cases and experiments,
not an attempt to reproduce every conventional ML library. The development
sequence and per-slice evidence are recorded in `docs/ML_VERTICAL_SLICES.md`.

## Phase 4 — Reinforcement Learning

**Status: Later / directional**

Develop RL abstractions after state, streaming, and execution foundations
are sufficiently understood.

Potential areas include observations/actions, rewards, environment
interaction, policy/value state, online updates, and episodic/continuing
tasks.

Exact RL abstractions remain deferred until experiments expose the required
contracts.

## Phase 5 — Compute and Hardware Specialization

**Status: Later / evidence-driven**

Investigate performance specialization only where benchmarks establish a
meaningful need.

Possible areas include compiler-assisted vectorization, explicit SIMD,
CPU parallelism, accelerator execution, optimized numerical kernels,
backend integration, device-local state, and data-movement strategies.

CUDA, ROCm, Rayon, explicit SIMD, NUMA, zero-copy, unified memory, and other
mechanisms are **candidate implementation techniques**, not Phase 5
requirements.

The optimization path is:

```text
correctness
→ reference implementation
→ benchmark
→ profile
→ data/layout optimization
→ compiler optimization
→ specialization
```

Not every workload needs to reach the final step.

## Phase 6 — Constrained and Edge Deployment

**Status: Later / directional**

Investigate deployment characteristics relevant to embedded systems,
low-power devices, edge inference, constrained memory environments, and
potentially bare-metal targets.

The current project direction is std-first with a practical path toward
no_std-compatible foundational components. Concrete deployment guarantees
must be demonstrated rather than assumed.

## What is deliberately not planned yet

The following should not become implementation commitments merely because
they appear in future discussions:

- a fixed multi-crate architecture;
- a universal tensor/storage abstraction;
- a universal global error type;
- a universal execution scheduler;
- a mandatory graph/operator representation;
- mandatory GPU support;
- mandatory CUDA or ROCm support;
- mandatory Rayon usage;
- distributed execution;
- multi-node execution;
- a plugin architecture;
- a complete dataframe/data-engineering layer;
- reproduction of every feature in existing ML frameworks.

## Roadmap rule

> **A future phase identifies a problem worth investigating. It does not
> authorize speculative infrastructure today.**

Evidence from experiments, tests, benchmarks, profiling, and real
implementation constraints should be used to revise this roadmap.
