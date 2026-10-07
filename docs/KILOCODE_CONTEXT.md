# Seqvex — KiloCode Architectural Context

> **Current model requirement:** every ML model must ingest data, train/learn, validate and calculate inference from the learned result under the [mandatory training-to-inference contract](DEVELOPMENT.md#model-training-and-valid-inference--mandatory). A reference slice, imported parameters or initialization alone does not establish complete model acceptance. Read this as architectural context, not an implementation authorization or replacement for separate Architect design and Planner plan. Raise any unknown failure/recovery policy to the maintainer.

## Purpose

This is a **context and comprehension brief** for KiloCode. Read it before any implementation work.

Its purpose is to make KiloCode understand Seqvex's architectural intent, scope, constraints, coding philosophy, and deliberately deferred decisions before modifying the repository.

This document is not an implementation specification.

## Required Reading

Before making changes, read:

- `README.md`
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `docs/DEVELOPMENT.md`
- `docs/FAILURE_AND_RECOVERY.md`
- `Cargo.toml`
- current `src/`
- current `tests/`

Inspect the actual repository state. Do not assume future components already exist.

If a conflict exists, identify it rather than silently resolving it.

## 1. What Seqvex Is

Seqvex is intended to become a Rust-native machine-learning and reinforcement-learning framework designed for:

- sequential workloads;
- temporal workloads;
- non-IID data;
- streaming workloads;
- online/incremental learning;
- stateful model execution;
- predictable and low-latency execution where feasible;
- hardware-aware computation.

Seqvex is general-purpose ML/RL infrastructure, not merely a time-series library.

Potential domains include cyber defense, autonomous systems, industrial IoT, edge computing, low-power sensor networks, and streaming predictive analytics.

Quantitative research is an external potential consumer, not the framework's defining scope.

## 2. Central Architectural Idea

> **Seqvex is streaming-first, not batchless.**

Streaming is the semantic foundation and a single observation is the smallest semantic execution unit.

Batching remains valid because many algorithms and hardware architectures benefit from vectorization and parallelism.

Seqvex should therefore eventually support:

- single-observation execution;
- bounded micro-batches;
- larger batches where appropriate.

Batching must not silently violate ordering, temporal semantics, causality, state transitions, or model-state consistency.

## 3. Execution Semantics vs Compute Placement

These are separate concerns.

Execution semantics answer:

> What does the computation mean?

They include streaming, single-observation processing, micro-batching, ordering, and state transitions.

Compute placement answers:

> Where does the computation execute?

It may eventually include CPU, GPU, accelerators, or heterogeneous execution.

Do not interpret heterogeneous execution as moving every observation between devices.

The intended principle is:

> **Keep model state and data resident on one device when possible; move data only when measured computational benefit exceeds transfer and synchronization cost.**

## 4. Stateful Computation

Persistent model/runtime state is fundamental.

Conceptually:

```text
observation 1 → state 1
observation 2 → state 2
observation 3 → state 3
```

This matters for online learning, continual learning, recurrent models, adaptive estimators, reinforcement learning, and streaming inference.

The architecture must distinguish input observation/data from persistent model/runtime state.

## 5. State Transition Integrity

A foundational invariant is:

```text
current valid state
        ↓
observation
        ↓
computation
        ↓
candidate next state
        ↓
validation
     ┌──┴──┐
   valid invalid
     │      │
   commit  reject
     │      │
     ▼      ▼
 next     original
 valid     valid
 state      state
```

A failed update must not silently leave the model partially modified or invalid.

The implementation mechanism is deliberately undecided.

## 6. Failure and Recovery

Failure behavior is architectural. See `docs/FAILURE_AND_RECOVERY.md`.

Core principle:

> **Detect → stop unsafe transition → preserve valid state → classify → act → report → continue/recover/isolate/terminate.**

If processing continues after a recoverable failure:

```text
x1 → S1
x2 → FAILURE → S1 remains valid
x3 → S3
```

The failed observation must not accidentally create an implicit state transition.

## 7. Scope Boundary

Seqvex differentiates its scope by **process responsibility**, not by the name of an individual function.

Seqvex owns the computational stages from:

- ML/RL preprocessing and representation;
- model execution;
- training;
- validation;
- inference;
- learning;
- numerical computation required by ML/RL.

General-purpose data ingestion, manipulation, cleaning, EDA, visualization, databases, storage, ETL, and domain/business logic remain outside.

An operation's membership therefore depends on its role.

## 8. Numerical Foundation

Seqvex requires numerical computation for ML/RL but is not intended to replace NumPy, SciPy, Pandas, Polars, dataframe engines, or general-purpose statistical software.

Introduce numerical functionality incrementally.

## 9. Hardware and Memory Awareness

The architecture must preserve the ability to reason about:

```text
CPU registers
L1/L2/L3/LLC cache
RAM
interconnect
GPU registers/cache
GPU VRAM
NUMA
memory locality
alignment
SIMD
memory bandwidth
data movement
synchronization
```

Do not implement these concerns in the foundation scaffold.

## 10. Real-Time / Constrained Direction

Seqvex may eventually serve constrained or safety-sensitive environments.

It does not currently claim hard real-time guarantees, safety certification, bare-metal support, zero-allocation execution, or cross-platform deterministic floating-point behavior.

These must be demonstrated later.

## 11. Modular Architecture

Seqvex is intended to be modular and hierarchical.

The architectural hierarchy should be decomposed by functional responsibility:

```text
framework
  ↓
functional domain
  ↓
functional module
  ↓
sub-functional component
```

A **functional module** is a coherent responsibility large enough to warrant its own `README.md`.

Sub-functional components normally do not receive individual README files.

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

At the sub-functional level, use:

- meaningful inline code documentation;
- focused tests;
- clear names;
- small implementation boundaries.

Do not create many README files merely for documentation symmetry.

The principle is:

> **Document at the highest level where shared context is useful; keep detailed implementation knowledge close to the code and tests.**

## 12. Crate Architecture

Use focused Cargo crates only as meaningful boundaries emerge.

Potential future directions include:

```text
seqvex-core
seqvex-math
seqvex-runtime
seqvex-validation
seqvex-supervised
seqvex-unsupervised
seqvex-rl
seqvex-compute
seqvex
```

These are possibilities, not current mandatory crates.

Do not create empty future crates.

> **Do not create an abstraction until you have experienced the problem it solves.**

## 13. Coding Principles

Seqvex should use idiomatic, maintainable Rust.

The following principles are important design guidance, not excuses for premature abstraction.

### DRY — Don't Repeat Yourself

Avoid duplicating knowledge, invariants, or logic.

Do not interpret DRY as "every similar-looking line needs a generic abstraction."

Prefer a small helper when one rule genuinely has one source of truth.

```rust
fn is_finite(x: f64) -> bool {
    x.is_finite()
}
```

Do not abstract merely because two pieces of code currently look similar.

### Composition Over Inheritance

Rust does not use classical inheritance as its primary abstraction mechanism.

Prefer composing structs from focused components and using traits for behavior.

```rust
struct Model<S, E> {
    state: S,
    executor: E,
}
```

rather than attempting to create deep type hierarchies.

Keep traits small and behavior-focused.

### Enforce Clippy

Seqvex should maintain a clean Clippy baseline.

Use:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

where compatible with the current project configuration.

Do not silence Clippy warnings casually.

If a lint must be allowed, document why the exception is justified.

### Extension Traits

Use extension traits when they provide a coherent domain-specific operation on an existing type without forcing ownership of that type.

Example:

```rust
trait IteratorExt: Iterator {
    fn finite_only(self) -> impl Iterator<Item = Self::Item>;
}
```

Do not create extension traits merely to rename ordinary methods or hide trivial code.

### Builder Pattern

Use builders when constructing a type involves:

- multiple optional parameters;
- meaningful configuration;
- validation before construction;
- readability benefits.

Example:

```rust
let model = ModelBuilder::new()
    .learning_rate(0.01)
    .build()?;
```

Do not use builders for trivial types with one or two obvious fields.

### Typestate Pattern

Use typestate where the **state of a value at compile time prevents invalid transitions**.

This may become particularly useful in Seqvex because the framework is stateful and may evolve toward asynchronous execution.

Conceptually:

```rust
struct Uninitialized;
struct Ready;

struct Runtime<S> {
    state: S,
}
```

The important principle is to use the type system when it provides a real correctness guarantee.

Do not create elaborate typestate machines simply because they are possible.

### Newtype Pattern

Use newtypes when two values share an underlying representation but have different semantic meaning.

Example:

```rust
struct ObservationId(u64);
struct SequenceNumber(u64);
```

This prevents accidental interchange of semantically different values.

Do not wrap every primitive automatically.

### Error Handling via Enums

Prefer explicit domain errors represented by enums.

Example:

```rust
enum UpdateError {
    InvalidInput,
    NumericalFailure,
    InvariantViolation,
}
```

Use `Result<T, E>` for recoverable failures.

Avoid stringly typed errors for core domain behavior.

Do not use `unwrap()` or `expect()` in production paths unless the invariant is genuinely impossible to violate and the reasoning is documented.

### Iterators / Functional Style

Prefer Rust iterators and functional composition when they make the code clearer and avoid unnecessary intermediate allocations.

Example:

```rust
let sum: f64 = values.iter().copied().sum();
```

Prefer iterator-based transformations over unnecessary indexing loops when readability and performance are not harmed.

Do not force functional style where an explicit loop is clearer or materially faster.

Performance-sensitive code should eventually be benchmarked rather than optimized by stylistic assumption.

## 14. Testing Philosophy

Keep testing simple:

- unit tests for local behavior;
- integration tests for meaningful boundaries;
- failure tests for invalid transitions and state integrity;
- invariant/property tests where genuinely useful.

Tests should define behavior before implementation wherever practical.

The progression is:

```text
define behavior
    ↓
write test cases
    ↓
implement
    ↓
pass tests
    ↓
integrate
    ↓
CI
```

TDD is a useful technique, not a project-wide doctrine.

BDD is optional.

## 15. Development Philosophy

The intended process is:

```text
learn
  ↓
implement smallest useful experiment
  ↓
test
  ↓
break
  ↓
understand
  ↓
benchmark
  ↓
profile
  ↓
optimize
  ↓
document
  ↓
integrate
  ↓
automate with CI
```

The developer wants to personally understand and implement the system.

KiloCode should therefore act as an engineering assistant, not an autonomous framework generator.

## 16. Deferred Decisions

Keep these open:

- exact crate boundaries;
- tensor/buffer/storage abstractions;
- ownership/allocation strategy;
- device abstraction;
- GPU backend;
- CUDA/ROCm;
- SIMD architecture;
- scheduler;
- computational graph;
- automatic differentiation;
- serialization;
- multi-device scheduling;
- distributed execution;
- exact public API.

Do not resolve these speculatively.

## 17. Comprehension Gate

Before executing `KILO_SCAFFOLD.md`, KiloCode should be able to explain:

1. what Seqvex is;
2. why it is streaming-first but not batchless;
3. why state is fundamental;
4. why failed updates preserve valid state;
5. why execution semantics and compute placement are separate;
6. what the scope boundary is;
7. why numerical computation is included but general data processing is not;
8. why hardware locality matters;
9. how functional/sub-functional modularity is organized;
10. which architectural decisions remain deferred;
11. why tests are being established before implementation;
12. which Rust coding principles guide implementation.

Do not modify the repository merely because this document was read.

The scaffold task is specified separately in `KILO_SCAFFOLD.md`.
