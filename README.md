# Seqvex

> **Model delivery requirement:** every ML model must ingest data, train/learn, validate the learned result and calculate predictions from it. Preset-parameter demonstrations and prediction-only components are not complete model deliveries. Current component evidence does not imply a full training workflow exists. See the [mandatory training-to-inference contract](docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory).

<p align="center">
  <img src="assets/seqvex-official-display.png" alt="Seqvex" width="360">
</p>

> **Streaming-first ML/RL for sequential, temporal, and non-IID data — with continuous state, online learning, and hardware-aware execution.**

Seqvex is an open-source, Rust-native machine learning and reinforcement
learning framework designed from the ground up for **sequential, temporal,
non-IID, and streaming workloads**.

Many real systems do not behave like a static IID dataset followed by a
separate offline training stage. Observations arrive over time,
distributions can evolve, models maintain state, and the system may need
to learn and infer continuously.

Seqvex therefore treats **evolving data, stateful computation, and streaming
execution as first-class concerns**.

<p align="center">
  <img src="assets/seqvex-at-a-glance.png" alt="Seqvex at a glance" width="820">
</p>

## Why Seqvex?

Conventional batch-oriented ML remains highly useful. Large datasets and
vectorized batch computation are often the right computational strategy.

Seqvex does not reject batching. Instead, it changes the **semantic starting
point**:

```text
sequential / temporal / non-IID
              ↓
        streaming-first
              ↓
       execution context
              ↓
     ┌────────┴────────┐
     │                 │
single observation   optional
                     micro-batch
     │                 │
     └────────┬────────┘
              ↓
         model / state
              ↓
     ┌────────┴────────┐
     │                 │
  learning          inference
```

Compute placement is a separate architectural dimension rather than another
semantic stage:

```text
compute placement
       │
 ┌─────┼──────────────────┐
 │     │                  │
CPU   GPU            accelerator /
                     heterogeneous
       │
       ↓
locality / layout / memory /
SIMD / synchronization / data movement
```

Larger batch operations remain available when an algorithm or hardware
workload benefits from them, but **batch-first execution is not the
architectural foundation**.

**Streaming is the primary computational context.** Single-observation
execution is fundamental. Bounded micro-batching is available when it
improves computational or hardware efficiency without violating ordering,
causality, state, or online-learning semantics.

Larger batch operations remain available where an algorithm benefits from
them, particularly for historical/offline computation. They are a
**computational capability, not the semantic foundation of Seqvex**.

The architecture also separates **execution semantics** from **compute
placement**. Streaming, single-observation execution, and micro-batching
describe how computation is expressed; CPU, GPU, accelerators, and
heterogeneous execution describe where and how an implementation runs.

## Core characteristics

- **Sequential / temporal / non-IID first** — evolving observations and
  changing distributions are central design concerns.
- **Streaming-first** — continuous, stateful computation is the primary
  computational context.
- **Single-observation capable** — an observation can be processed
  immediately when the workload requires it.
- **Optional micro-batching** — bounded aggregation can improve throughput,
  vectorization, or accelerator utilization without becoming the conceptual
  model.
- **Batch-capable where justified** — larger batch computation remains
  available when algorithms or hardware benefit from it.
- **Continuous state** — models and runtimes can maintain state across
  observations.
- **Learning and inference can be continuous** — online learning, inference,
  or both may operate within an evolving stream.
- **Hardware-aware** — CPU, GPU, and other accelerators are placement
  options rather than mandatory dependencies.
- **Performance measured** — locality, cache behavior, allocation, SIMD,
  bandwidth, synchronization, and data movement are optimization surfaces,
  not assumed performance guarantees.
- **Rust-native and type-safe** — safe Rust is the default, with lower-level
  control where justified.
- **Cloud-to-edge direction** — the architecture preserves a path toward
  constrained and potentially bare-metal deployment.

## High-level architecture

<p align="center">
  <img src="assets/seqvex-high-level-flow.png" alt="Seqvex high-level architecture" width="820">
</p>

The high-level architecture describes how Seqvex treats the computational
problem:

```text
sequential / temporal / non-IID
              ↓
        streaming-first
              ↓
       execution context
              ↓
     ┌────────┴────────┐
     │                 │
single observation   optional
                     micro-batch
     │                 │
     └────────┬────────┘
              ↓
         model / state
              ↓
     ┌────────┴────────┐
     │                 │
  learning          inference
     │                 │
     └────────┬────────┘
              ↓
       compute placement
              ↓
     CPU / GPU / accelerator
              ↓
 locality / memory / SIMD /
 synchronization / data movement
              ↓
      outputs / environment
              ↺
 new observations / rewards /
     environment changes
```

This is a conceptual model, not a frozen implementation or crate layout.

A historical dataset can be replayed as a stream. A live deployment can
produce an effectively unbounded stream. The physical source of data does
not define the execution semantics.

## Streaming workflow

<p align="center">
  <img src="assets/streaming-workflow.png" alt="Seqvex streaming workflow" width="820">
</p>

A typical end-to-end system may look like:

```text
External data sources
        ↓
application-side ingestion / preparation
        ↓
Seqvex ML/RL preprocessing and representation
        ↓
Seqvex streaming context
        ↓
single observation
        OR
optional bounded micro-batch
        ↓
model / state
        ↓
learning and/or inference
        ↓
hardware-aware execution
        ↓
application outputs
        ↺
new observations / rewards /
environment changes
```

The first stages may belong to the surrounding application or data
ecosystem. General-purpose ingestion, databases, ETL, data cleaning,
exploratory analysis, visualization, and storage are not part of Seqvex's
core scope.

Seqvex begins where computation becomes part of the ML/RL pipeline and
continues through model computation, learning, validation, inference, and
execution.

## Training and inference

Seqvex does not require training and inference to be architecturally
disconnected.

### Historical / offline training

Historical data can be replayed sequentially:

```text
historical source
      ↓
sequential replay
      ↓
x₁ → update
x₂ → update
x₃ → update
...
xₙ → update
```

The same source may also provide larger batches when the algorithm benefits
from aggregation.

### Online / continual learning

After actual data-driven learning has established the approved prediction-ready state, a live stream can continuously update model state:

```text
xₜ → predict → update → xₜ₊₁ → predict → update → ...
```

There does not need to be a fixed dataset boundary for continuous learning.

### Inference

Inference can process observations individually or use bounded batching
when throughput or hardware utilization justifies it.

The important property is that **the stream remains the computational
context even when an implementation uses bounded or larger batches for
efficiency**.

## Scope

Seqvex differentiates its scope by **process responsibility rather than by
individual function**.

Seqvex owns the computational stages from:

- ML/RL preprocessing and representation;
- model computation;
- training and learning;
- validation;
- inference;
- numerical computation required by ML/RL;
- execution/runtime mechanisms.

General-purpose:

- data ingestion;
- data manipulation;
- data cleaning;
- exploratory data analysis;
- visualization;
- databases and storage;
- ETL;
- domain/business logic

remain outside the framework.

The boundary is determined by the **role a computation plays in the ML/RL
pipeline**, not by its name. For example, one-hot encoding,
normalization/standardization, PCA, rolling statistics, and online
statistics may belong inside Seqvex when they are part of an ML/RL method
or computational pipeline, while the same operation may remain outside
when used for general data analysis or manipulation.

Seqvex does not aim to become:

- a dataframe library;
- a general ETL framework;
- a database or storage system;
- a general-purpose data-cleaning platform;
- an exploratory data-analysis environment;
- a visualization system;
- a domain/business-logic framework;
- a general data-ingestion platform.

Users can bring data from files, databases, sensors, message systems,
Arrow/Polars workflows, or custom pipelines.

## Development

Seqvex is developed experimentally and incrementally. The development
cycle is intentionally closed: integration produces new knowledge and
returns the project to learning.

<p align="center">
  <img src="assets/development-process.png" alt="Seqvex development cycle" width="880">
</p>

```text
Learn
  ↓
Implement the smallest useful experiment
  ↓
Test
  ↓
Break / find edge cases
  ↓
Understand
  ↓
Benchmark
  ↓
Profile
  ↓
Optimize
  ↓
Document
  ↓
Integrate
  ↺
Learn
```

### Why the stages are separate

- **Learn** — research, explore ideas, and understand the problem space.
- **Implement** — build the smallest useful experiment.
- **Test** — validate correctness, invariants, and expected behavior.
- **Break / find edge cases** — deliberately exercise failure paths and
  expose assumptions.
- **Understand** — analyze behavior, failures, and evidence.
- **Benchmark** — measure behavior under representative workloads.
- **Profile** — locate the actual bottleneck.
- **Optimize** — improve the bottleneck, then measure again.
- **Document** — record knowledge, decisions, and evidence.
- **Integrate** — incorporate validated results into Seqvex and return to
  the next learning cycle.

The sequence is a guiding development loop rather than a mandatory
ten-step checklist for every change. Small documentation or bug-fix
changes may not require every stage; performance-sensitive changes should
use the measurement stages when relevant.

> **Do not create an abstraction until you have experienced the problem it solves.**

> **Do not optimize until you can measure the problem.**

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architectural reasoning,
  execution model, scope, hardware direction, current decisions, and
  deferred design questions.
- [`DEVELOPMENT.md`](docs/DEVELOPMENT.md) — development principles for
  humans and AI-assisted work (the canonical development contract).
- [`FEATURE_DEVELOPMENT.md`](FEATURE_DEVELOPMENT.md) — the operational
  runbook for taking a feature, algorithm, or function from issue to merge.
- [`ROADMAP.md`](docs/ROADMAP.md) — current development direction and phase
  planning; intentionally non-contractual.
- [`FAILURE_AND_RECOVERY.md`](docs/FAILURE_AND_RECOVERY.md) — failure
  paths, state integrity, and recovery principles.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution principles and
  participation.
- [`LICENSE`](LICENSE) — Apache License 2.0.

### Visual guide

The repository graphics intentionally answer different questions:

| Asset | Purpose |
|---|---|
| `seqvex-at-a-glance.png` | What Seqvex is and its core positioning |
| `seqvex-high-level-flow.png` | The conceptual Seqvex architecture and execution model |
| `streaming-workflow.png` | An end-to-end workflow from external data through Seqvex and back to evolving observations |
| `development-process.png` | How Seqvex is developed and how evidence feeds back into the next cycle |

Do not treat these diagrams as interchangeable. The architecture diagram
describes the system; the streaming workflow describes how a surrounding
application can use it; the development-cycle diagram describes how the
project itself is developed.

## Repository structure

Seqvex is currently a **single Cargo package**. Additional crates and a
Cargo workspace may emerge later when demonstrated responsibility
boundaries justify them.

Possible future areas include core types, numerical computation,
streaming/online learning, sequential models, validation, execution/runtime,
device backends, and algorithm families.

These are architectural directions, not a requirement to create every
component or crate immediately.

## Installation

Seqvex is published on crates.io.

```bash
cargo add seqvex
```

Or:

```toml
[dependencies]
seqvex = "0.1"
```

Then:

```bash
cargo build
```

> **Current status:** Seqvex `0.1.0` is an early foundational release.
> Public APIs and architecture are expected to evolve as implementation
> experience and measurements accumulate.

## Contributing

Contributions, experiments, benchmarks, documentation, bug reports, and
architectural discussion are welcome.

Before making a substantial change, read:

1. [`ARCHITECTURE.md`](ARCHITECTURE.md)
2. [`DEVELOPMENT.md`](docs/DEVELOPMENT.md)
3. [`FEATURE_DEVELOPMENT.md`](FEATURE_DEVELOPMENT.md)
4. [`FAILURE_AND_RECOVERY.md`](docs/FAILURE_AND_RECOVERY.md)
5. [`ROADMAP.md`](docs/ROADMAP.md), where the change affects current phase work
6. [`CONTRIBUTING.md`](CONTRIBUTING.md)

Architecture changes should be supported by a concrete problem, evidence,
and an understanding of reversal cost.

## License

Seqvex is licensed under the [Apache License 2.0](LICENSE).
