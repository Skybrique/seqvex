# Seqvex — Connectome Vertical Slice Sprint

> **Historical reference — superseded execution instructions:** retain this record for its original scope and evidence; do not execute it as current implementation authorization. Any earlier prediction-only scope, preset-parameter demonstration or completion claim must be assessed under the [mandatory training-to-inference contract](DEVELOPMENT.md#model-training-and-valid-inference--mandatory) and current Architect → Planner → Coder gates. Raise unresolved failure decisions to the maintainer.
## Technical Scaffold, Execution Specification, and Sprint Record

> **Status:** Sprint documentation  
> **Primary use:** KiloCode implementation guidance + permanent repository record  
> **Recommended location:** `docs/KILO_CONNECTOME_SPRINT.md`
>
> This document records a deliberately bounded Seqvex development sprint. It is both an execution specification for the sprint and a historical record of what Seqvex temporarily built before returning to the broader framework roadmap.
>
> This document supplements, and does not override, `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `AGENTS.md`, `docs/DEVELOPMENT.md`, `docs/KILOCODE_CONTEXT.md`, `docs/KILO_SCAFFOLD.md`, `docs/FAILURE_AND_RECOVERY.md`, and `docs/ROADMAP.md`.

---

# 1. Why this sprint exists

The purpose of this sprint is to take Seqvex beyond its initial foundation scaffold and establish its **first real end-to-end sequential ML vertical slice**.

The motivating workload is the Wunder Fund Connectome competition. Connectome is the workload that gives this sprint a concrete computational target; it is **not** the definition of Seqvex.

The sprint is therefore intentionally narrow:

```text
existing foundation
       ↓
usable numerical substrate
       ↓
CPU reference implementation
       ↓
measure + profile
       ↓
SIMD/vectorization where justified
       ↓
stateful GRU
       ↓
streaming execution
       ↓
end-to-end benchmark
       ↓
Connectome adapter
```

The sprint is successful when this path works coherently.

It is **not** successful merely because the repository contains the corresponding directories or files.

---

# 2. Permanent Seqvex architecture vs. sprint scope

Seqvex remains:

> **A streaming-first ML/RL framework for sequential, temporal, and non-IID data, with continuous state and hardware-aware execution.**

The sprint must preserve these principles:

- sequential / temporal / non-IID data is foundational;
- streaming is the primary computational context;
- single-observation execution is fundamental;
- bounded micro-batching may be used when it provides computational or hardware efficiency;
- larger batch operations remain valid where appropriate;
- execution semantics are separate from compute placement;
- persistent state is first-class;
- failed updates must not silently commit invalid or partial state;
- hardware awareness includes layout, locality, allocation, synchronization, SIMD, and data movement;
- CPU optimization is a **sprint workload target**, not a permanent CPU-only architecture;
- GRU is the first temporal model for this vertical slice, not Seqvex's identity;
- Connectome is one application/workload, not a core domain;
- quant trading is not a core Seqvex domain;
- unresolved architecture decisions remain unresolved unless this sprint produces sufficient evidence to decide them.

The permanent conceptual direction remains:

```text
sequential / temporal / non-IID
            ↓
      streaming-first
            ↓
          ML / RL
            ↓
 CPU / GPU / accelerator
            ↓
   hardware-aware execution
```

The sprint temporarily instantiates only a small portion of this space.

---

# 3. Relationship to the earlier foundation scaffold

`docs/KILO_SCAFFOLD.md` defined the original foundation stage and intentionally prohibited production ML/RL implementation at that stage.

That document was correct for its original purpose.

This sprint is the next bounded experiment. It therefore **extends the repository rather than invalidating the foundation design**.

The old scaffold's prohibitions still apply to anything that this sprint does not actually require:

- no universal tensor engine;
- no general autodiff system;
- no universal optimizer framework;
- no GPU backend;
- no distributed runtime;
- no general scheduler;
- no production allocator;
- no dataframe/ETL/ingestion framework;
- no speculative multi-crate architecture;
- no collection of future model families merely because they may eventually be useful.

The rule is:

> Build the smallest concrete implementation required to cross the next measurable boundary.

---

# 4. Sprint boundaries

## In scope

1. Complete or reconcile the existing foundation sufficiently for the vertical slice.
2. Make the existing numerical foundation usable.
3. Add only the numerical representations and operations required by the first GRU.
4. Establish a scalar/reference CPU implementation.
5. Benchmark and profile the numerical hot path.
6. Evaluate compiler auto-vectorization and explicit SIMD where evidence supports it.
7. Implement a stateful GRU for single-observation sequential execution.
8. Connect the GRU to streaming execution semantics.
9. Demonstrate state continuity and explicit reset.
10. Benchmark the complete streaming path.
11. Build a thin Connectome adapter outside Seqvex core.
12. Record measurements, architectural observations, and unresolved questions.

## Explicitly out of scope

- complete Seqvex;
- general-purpose tensor infrastructure;
- GPU/accelerator implementation;
- universal device abstraction;
- universal memory allocator;
- distributed training/execution;
- full RL implementation;
- LSTM/Transformer/SSM families;
- complete autodiff;
- general training orchestration;
- dataframes;
- data ingestion/ETL;
- trading infrastructure;
- order-book abstractions;
- portfolio/exchange/broker APIs;
- competition-specific concepts inside `src/`.

---

# 5. Current repository reconciliation

Before modifying anything, KiloCode must inspect the actual working tree.

Do not assume that the public `main` branch is identical to the developer workspace.

Minimum procedure:

```text
inspect working tree
        ↓
inspect current branch/main
        ↓
identify local changes
        ↓
read source + tests
        ↓
map existing issues to actual implementation
        ↓
run baseline validation
```

Baseline:

```bash
cargo test
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

KiloCode must not overwrite uncommitted work or refactor unrelated foundation code merely to make the repository match a target tree.

---

# 6. Dependency path

The sprint should normally progress through this dependency chain:

```text
Foundation contracts
        ↓
Numerical reference path
        ↓
CPU baseline
        ↓
Benchmark
        ↓
Profile
        ↓
Targeted optimization
        ↓
GRU
        ↓
Streaming execution
        ↓
End-to-end benchmark
        ↓
Connectome adapter
```

This is an execution dependency, **not a new permanent roadmap phase structure**.

A later component may be prototyped early if needed to answer a concrete question, but speculative architecture should not be frozen before the dependency is understood.

---

# 7. Target repository structure

The following is the intended shape **if the corresponding responsibilities are actually reached**:

```text
seqvex/
├── src/
│   ├── foundation/
│   │   ├── numerical/
│   │   │   ├── mod.rs
│   │   │   ├── README.md
│   │   │   ├── statistics.rs
│   │   │   └── additional numerical modules only as required
│   │   ├── observation/
│   │   │   ├── mod.rs
│   │   │   ├── ordering.rs
│   │   │   ├── README.md
│   │   │   └── representation.rs
│   │   └── state/
│   │       ├── atomicity.rs
│   │       ├── mod.rs
│   │       ├── README.md
│   │       └── transition.rs
│   │
│   ├── execution/
│   │   ├── mod.rs
│   │   └── streaming.rs
│   │
│   ├── models/
│   │   ├── mod.rs
│   │   └── recurrent/
│   │       ├── mod.rs
│   │       └── gru.rs
│   │
│   └── lib.rs
│
├── tests/
├── benches/
│   ├── numerical.rs
│   └── gru.rs
│
├── examples/
│   ├── streaming_gru.rs
│   └── state_reset.rs
│
├── competition/
│   └── connectome/
│       ├── solution.py
│       └── adapter/application files as required
│
└── docs/
    └── KILO_CONNECTOME_SPRINT.md
```

This is **not** a request to create empty files.

Only create a module when there is a concrete responsibility and an implementation or test that justifies it.

Do not split the current single Cargo package into multiple crates unless implementation evidence demonstrates a meaningful crate boundary.

---

# 8. Foundation work

The existing foundation concerns:

```text
observation
ordering
streaming
state
transition
failure atomicity
failure continuity
numerical semantics
determinism
```

The existing issue set should be inspected rather than recreated:

- #1 Minimal observation semantics
- #2 Sequence ordering semantics
- #3 Single-observation streaming semantics
- #4 Bounded micro-batch semantics
- #5 Minimal state-transition semantics
- #6 Failed-update atomicity
- #7 Sequential continuity after failure
- #8 Minimal numerical foundation
- #9 Phase 1 integration experiment

The foundation is sufficient for this sprint when a small executable path can demonstrate:

```text
observation
    ↓
ordered processing
    ↓
state transition
    ↓
candidate update
    ↓
validation
    ↓
commit
```

and, on failure:

```text
valid committed state
        ↓
failed update
        ↓
invalid candidate rejected
        ↓
previous committed state preserved
        ↓
next valid observation continues correctly
```

Do not introduce a second unrelated state model merely because the GRU needs state.

Reuse existing semantics where they genuinely fit; otherwise keep the boundary explicit rather than forcing an artificial abstraction.

---

# 9. Technical specification: numerical foundation

The numerical foundation is the computational substrate beneath the first temporal model.

It is **not** a tensor framework.

## 9.1 Existing `statistics.rs`

### Purpose

Provide the already-established basic statistical/numerical contracts needed by the foundation.

### Functional responsibility

At minimum, verify and cleanly implement the existing behavior for operations such as:

- sum;
- mean;
- variance;
- online/incremental statistics where already specified.

### Inputs

Numeric observations/slices or the input types already established by the repository.

### Outputs

Numerical results or explicit errors where the existing contract requires failure.

### Required semantic questions

The implementation must make behavior explicit for:

- empty input;
- singleton input;
- NaN;
- positive/negative infinity;
- numerical overflow/underflow where relevant;
- sample vs. population variance;
- numerical stability;
- deterministic accumulation.

Do not silently change existing semantics merely to accommodate the GRU.

### Performance

This is foundation code. Correctness and clear semantics take priority.

Benchmark it only when it becomes a demonstrated hot path.

### Must not implement

- tensor operations;
- neural-network layers;
- model state;
- automatic differentiation;
- GPU execution.

---

# 10. Technical specification: numerical representation

The first GRU requires a small number of low-level numerical operations. The exact module split should be driven by the implementation.

A likely minimum set is:

```text
contiguous f32 vector
elementwise add
elementwise multiply
scalar operations
dot product
matrix-vector multiplication
sigmoid
tanh
```

These are **requirements at the behavioral level**, not a demand for a particular public API.

## 10.1 Vector representation

### Purpose

Represent the contiguous one-dimensional data used by observations, hidden state, biases, and intermediate GRU vectors.

### Functional responsibility

The representation should support:

- known length;
- contiguous storage;
- indexed access where needed;
- iteration;
- construction from existing data;
- elementwise operations required by the GRU;
- predictable ownership/lifetime behavior.

### Performance requirements

The implementation should make the following properties visible and measurable:

- contiguous access;
- minimal unnecessary allocation;
- predictable memory footprint;
- sequential iteration;
- suitability for compiler vectorization;
- clear boundaries for later SIMD.

### Must not implement

- arbitrary-rank tensors;
- broadcasting engine;
- device memory;
- GPU allocation;
- automatic differentiation.

If the existing code can provide these semantics without introducing a new vector abstraction, prefer the existing representation.

---

# 11. Technical specification: matrix/vector computation

The GRU requires repeated affine transformations.

A minimal implementation may therefore require:

```text
y = W x
```

or an equivalent fused representation.

## Functional responsibility

Provide only the matrix-vector computation needed by the GRU.

The implementation should make dimensions explicit enough to catch mismatches.

## Required correctness properties

- dimension mismatch is detected;
- output dimensions are deterministic;
- no silent truncation;
- numerical behavior is reproducible for the same inputs and parameters;
- the scalar/reference implementation is easy to inspect.

## Performance considerations

The hot loop should use a layout that makes its access pattern understandable.

Measure before introducing:

- transposed storage;
- blocking;
- packing;
- explicit SIMD;
- unsafe pointer arithmetic;
- architecture-specific kernels.

Do not create general matrix-matrix multiplication unless another concrete sprint requirement requires it.

---

# 12. Technical specification: elementwise operations

If the GRU implementation requires reusable elementwise kernels, isolate them only where that improves clarity or measurement.

Likely operations:

```text
a + b
a * b
scalar * a
1 - a
```

## Requirements

- equal-length operands must be handled deterministically;
- mismatched lengths must fail explicitly;
- loops should be straightforward enough for compiler optimization;
- allocation behavior must be observable;
- scalar/reference behavior must remain testable.

These operations are likely SIMD candidates, but no SIMD implementation is justified until profiling/benchmarking demonstrates value.

---

# 13. Technical specification: activation functions

The GRU requires nonlinearities:

```text
sigmoid(x)
tanh(x)
```

## Functional responsibility

Provide numerically defined activation behavior for the range of values exercised by the model.

## Correctness

Test:

- zero;
- positive values;
- negative values;
- large magnitudes;
- symmetry/known mathematical properties where applicable;
- vectorized results against the scalar/reference result.

## Performance

Activations may become hot-path operations.

The optimization sequence remains:

```text
correct scalar implementation
        ↓
compiler optimization
        ↓
benchmark
        ↓
profile
        ↓
explicit specialization only if justified
```

Do not introduce a general activation framework for two functions.

---

# 14. Technical specification: stateful GRU

The first temporal model is a **GRU** because it provides a concrete recurrent state mechanism for testing Seqvex's streaming semantics.

The implementation should be the smallest useful stateful GRU, not a general neural-network framework.

## 14.1 Required state

A GRU instance must own or otherwise explicitly reference:

- input dimension;
- hidden dimension;
- model parameters;
- current hidden state.

Parameters include the weights and biases required by the selected GRU formulation.

## 14.2 Required operations

At minimum:

```text
construct/configure
step(one observation)
read/output current state as needed
reset state
```

Training infrastructure is not implied by these operations.

## 14.3 GRU recurrence

Use one clearly documented gate convention and implement it consistently.

A standard formulation is:

```text
z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)

r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)

h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)

h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

where:

- `x_t` is the current observation;
- `h_{t-1}` is the committed previous hidden state;
- `z_t` is the update gate;
- `r_t` is the reset gate;
- `h̃_t` is the candidate hidden state;
- `h_t` is the new committed hidden state;
- `⊙` denotes elementwise multiplication.

The exact convention must be documented in `gru.rs` or its module documentation. Do not mix gate conventions from different references.

## 14.4 Step semantics

Conceptually:

```text
previous committed state
          +
       x_t
          ↓
    GRU computation
          ↓
   candidate h_t
          ↓
 validation / failure boundary
          ↓
    committed h_t
```

A failed step must not leave partially updated hidden state.

If the numerical/model path cannot complete successfully, the previously committed valid state must remain intact.

## 14.5 Sequential behavior

The implementation must support:

```text
step(x1) → h1
step(x2) → h2
step(x3) → h3
```

where each step observes the state produced by the previous successful step.

It must also support:

```text
reset()

step(y1) → h1'
```

where `h1'` is computed from the reset state rather than from the previous sequence.

## 14.6 GRU tests

At minimum test:

- dimension validation;
- deterministic initialization/parameter application;
- one-step computation;
- repeated sequential steps;
- reset;
- long sequence continuity;
- invalid input;
- numerical failure if the chosen API can represent it;
- failure atomicity;
- reference calculation against independently calculated expected values for small dimensions.

The tests should prove behavior, not merely code coverage.

## 14.7 GRU must not implement

- LSTM;
- Transformer;
- SSM;
- generic layer registry;
- optimizer hierarchy;
- autodiff;
- GPU kernels;
- Connectome logic;
- trading logic.

---

# 15. Technical specification: streaming execution

`src/execution/streaming.rs` should exist only when there is a real execution responsibility distinct from the model itself.

## Purpose

Connect the established observation semantics to a stateful model without embedding model-specific mathematics into the execution layer.

Conceptually:

```text
observation
    ↓
ordering / stream semantics
    ↓
stateful model
    ↓
output
```

## Functional responsibility

The execution layer may own:

- sequential processing;
- forwarding observations to the model;
- sequence/reset boundaries;
- output propagation;
- execution-level failure handling;
- single-observation semantics.

Micro-batching can be added when a concrete use case justifies it.

## Must not own

- GRU equations;
- matrix multiplication kernels;
- activation implementations;
- SIMD;
- Connectome callbacks;
- trading logic;
- data ingestion;
- scheduling;
- asynchronous runtime.

The execution layer should express **when/how a model is stepped**, not **how the model computes**.

---

# 16. Technical specification: module namespaces

## `src/models/mod.rs`

Purpose:

- expose the model namespace;
- provide a clean location for the first model family.

Do not invent a universal model trait unless the actual implementation demonstrates a meaningful need for one.

## `src/models/recurrent/mod.rs`

Purpose:

- define the recurrent-model namespace;
- expose the GRU implementation.

Do not scaffold:

```text
rnn.rs
lstm.rs
gru_variants.rs
transformer.rs
```

just because the directory is named `recurrent`.

---

# 17. Training vs. inference

The sprint does not require a complete general-purpose training framework.

The practical requirement is to determine whether a trainable model can be produced without contaminating the inference path with unnecessary training machinery.

A possible boundary is:

```text
training environment
       ↓
trained parameters
       ↓
model artifact / parameter transfer
       ↓
Seqvex inference
```

The sprint may use external or temporary training tooling if that is the smallest path to obtaining valid GRU parameters.

Do not build a complete autodiff/optimizer/training scheduler stack merely because a model must eventually be trained.

If training becomes necessary inside Seqvex, implement only the smallest concrete capability required and document why.

---

# 18. CPU baseline and optimization

CPU work is an explicit sprint workstream because it provides a concrete optimization target for the Connectome workload.

It must **not** become a permanent CPU-only architectural assumption.

The required optimization methodology is:

```text
correct reference implementation
          ↓
simple contiguous loops
          ↓
compile with normal Rust optimizations
          ↓
benchmark
          ↓
profile
          ↓
identify dominant hot path
          ↓
optimize the measured bottleneck
          ↓
benchmark again
          ↓
retain only improvements supported by evidence
```

Likely candidates:

- dot product;
- matrix-vector multiplication;
- elementwise operations;
- sigmoid/tanh;
- GRU gate computation.

Do not optimize every operation.

Do not use explicit SIMD merely because it is technically possible.

---

# 19. SIMD/vectorization rules

The sprint should distinguish:

1. compiler auto-vectorization;
2. explicit portable SIMD where supported;
3. architecture-specific intrinsics.

The default order is:

```text
scalar correctness
      ↓
compiler optimization
      ↓
measure
      ↓
explicit SIMD experiment
      ↓
measure
      ↓
keep only if materially beneficial
```

If explicit SIMD introduces `unsafe`, isolate it behind a narrow boundary and document its invariants.

A SIMD implementation must be compared against the scalar/reference path for:

- correctness;
- numerical tolerance;
- edge cases;
- performance;
- allocation behavior.

Do not make SIMD part of the public conceptual model.

---

# 20. Benchmark specification

Benchmarks should answer actual engineering questions.

## `benches/numerical.rs`

Measure the operations that actually appear in the GRU hot path.

Potential measurements:

- dot product throughput/latency;
- matrix-vector multiplication;
- elementwise operations;
- activation functions;
- input sizes relevant to the workload;
- allocations where meaningful.

Questions:

```text
Does contiguous layout matter?
Does compiler optimization vectorize the loop?
Does explicit SIMD improve it?
At what sizes?
Does the optimization remain useful in the complete GRU?
```

## `benches/gru.rs`

Measure:

- per-step latency;
- observations/second;
- sustained processing over long sequences;
- hidden-state footprint;
- allocation behavior;
- performance as input/hidden dimensions change.

The benchmark must not be presented as a production performance claim without workload-representative evidence.

Benchmark results should be retained in the sprint record when they materially influenced an architectural or optimization decision.

---

# 21. Profiling

Profiling is required before making strong performance claims.

The goal is to identify the actual bottleneck:

```text
GRU step
   ↓
profile
   ↓
hot operation
   ↓
optimize
   ↓
re-profile
```

Possible outcomes include:

- numerical kernel dominates;
- allocation dominates;
- memory movement dominates;
- activation functions dominate;
- model-state handling dominates;
- execution-layer overhead dominates.

The implementation must follow the evidence rather than assuming that matrix multiplication or SIMD is automatically the bottleneck.

---

# 22. Examples

Examples should demonstrate semantics, not become a second framework.

## `examples/streaming_gru.rs`

Demonstrate:

```text
construct model
      ↓
create/receive observation
      ↓
stream observation
      ↓
GRU step
      ↓
observe output/state
      ↓
repeat
```

The example should make persistent state visible.

## `examples/state_reset.rs`

Demonstrate:

```text
x1 → state1
x2 → state2
x3 → state3

reset

y1 → state1'
```

The example should make the sequence boundary explicit.

---

# 23. Connectome adapter

The competition-specific layer belongs outside Seqvex core.

Conceptually:

```text
Connectome callback/input
          ↓
      adapter
          ↓
Seqvex observation representation
          ↓
streaming execution
          ↓
GRU
          ↓
prediction/output
          ↓
      adapter
          ↓
Connectome
```

The adapter owns:

- competition callback/API translation;
- competition-specific input/output formats;
- competition-specific orchestration;
- model parameter loading if appropriate;
- application-level data handling.

Seqvex core owns:

- numerical computation;
- model computation;
- persistent model state;
- streaming execution semantics.

No Connectome-specific types should enter:

```text
src/foundation/
src/execution/
src/models/
```

---

# 24. Competition scope boundary

Do not allow the competition to pull unrelated domain infrastructure into Seqvex.

Do not add to core:

- dataframe engines;
- data ingestion;
- ETL;
- Parquet infrastructure solely for the competition;
- trading instruments;
- order books;
- exchange APIs;
- broker APIs;
- portfolio logic.

If the application needs these things, the application layer owns them.

The architectural test is:

> Could this component be useful to a sequential ML/RL workload that has nothing to do with Connectome?

If the answer is no, it should normally remain outside Seqvex core.

---

# 25. Failure and state-integrity requirements

The sprint must preserve the architectural rule from `FAILURE_AND_RECOVERY.md`:

> A failed update must not silently commit invalid or partially updated state.

For the GRU, this means that state mutation should conceptually behave as:

```text
committed h_(t-1)
       ↓
compute candidate
       ↓
validate
       ↓
commit h_t
```

not:

```text
mutate h
  ↓
compute
  ↓
error
  ↓
partially corrupted h
```

The exact implementation technique is intentionally open.

Possible implementation techniques include:

- compute-then-commit;
- temporary buffers;
- transactional state transition;
- copy-on-write for small state;
- carefully controlled in-place update with rollback.

Choose based on measurement and actual state size rather than introducing a universal rollback framework.

---

# 26. Memory and allocation discipline

The sprint should make allocation behavior visible because streaming workloads can magnify per-observation overhead.

Measure where meaningful:

- allocations per observation;
- temporary buffers;
- state storage;
- parameter storage;
- data movement.

Do not prematurely build:

- custom allocators;
- arena systems;
- memory pools;
- universal buffer managers;
- device memory managers.

If repeated allocation becomes a measured bottleneck, implement the smallest local solution and record the evidence.

---

# 27. API design discipline

The sprint should define **behavior before abstraction**.

Prefer:

```text
required behavior
      ↓
small implementation
      ↓
tests
      ↓
repeated use
      ↓
abstraction if justified
```

Avoid:

```text
possible future use cases
      ↓
generic trait hierarchy
      ↓
many implementations
      ↓
little evidence
```

Do not create a universal model interface, tensor interface, device interface, scheduler interface, or memory interface solely to make the architecture look complete.

A concrete, narrow API that survives the sprint is more valuable than a speculative general API.

---

# 28. Git / `.gitignore` discipline

Generated Rust build output should remain ignored:

```text
target/
```

Before changing `.gitignore`, inspect tool-specific directories such as `.kilo/`.

Do not ignore a directory simply because it belongs to a development tool.

The question is:

```text
version-controlled project knowledge?
        OR
local/generated development state?
```

Keep the former tracked.

Ignore the latter.

Do not remove useful existing ignore rules without evidence.

---

# 29. KiloCode implementation behavior

For every implementation boundary:

### Before coding

```text
read relevant existing code
        ↓
identify current invariant
        ↓
check whether capability already exists
        ↓
identify smallest required change
        ↓
write/extend tests
```

### During implementation

```text
implement
   ↓
test
   ↓
format
   ↓
clippy
```

### For numerical/performance work

```text
reference implementation
        ↓
benchmark
        ↓
profile
        ↓
optimize
        ↓
benchmark again
        ↓
verify correctness
```

### For architectural uncertainty

```text
uncertain
   ↓
small experiment
   ↓
evidence
   ↓
decision or explicit deferral
```

Do not guess.

---

# 30. What KiloCode must not do

Do not:

- replace the existing architecture with a new one;
- split into multiple crates without evidence;
- create a universal tensor framework;
- create a universal device abstraction;
- create a universal allocator;
- create a scheduler;
- add GPU support for completeness;
- implement RL merely because Seqvex eventually supports RL;
- implement LSTM/Transformer/SSM families;
- add Connectome concepts to core;
- create broad `common`/`utils` abstractions to hide small duplication;
- rewrite foundation semantics without a concrete reason;
- claim SIMD benefit without measurement;
- claim low latency without measurement;
- treat CPU-first implementation as permanent architecture;
- create empty/speculative modules;
- create fake implementations just to complete a target tree;
- optimize before establishing a correct reference path.

---

# 31. Required test layers

The sprint should retain multiple validation levels.

## Foundation tests

Verify:

- observation validity;
- ordering;
- single-observation streaming;
- bounded micro-batch behavior where already defined;
- state transitions;
- failure atomicity;
- sequential continuity;
- determinism.

## Numerical tests

Verify:

- dimensions;
- normal values;
- edge values;
- numerical stability where relevant;
- deterministic results;
- reference behavior.

## GRU tests

Verify:

- parameter dimensions;
- one-step computation;
- repeated steps;
- reset;
- long sequences;
- failure behavior;
- state preservation after failed updates.

## Integration tests

Verify:

```text
observation
   ↓
streaming execution
   ↓
GRU
   ↓
output
```

over multiple sequential observations.

## End-to-end application test

Verify that the Connectome adapter can translate a representative external sequence through the complete Seqvex path without adding competition concepts to core.

---

# 32. End-of-sprint definition of done

The sprint is complete when the following path is real, tested, and measurable:

```text
foundation
    ↓
numerical substrate
    ↓
CPU reference implementation
    ↓
measured optimization
    ↓
stateful GRU
    ↓
streaming execution
    ↓
end-to-end benchmark
    ↓
Connectome adapter
```

Minimum acceptance checklist:

- [ ] foundation semantics remain coherent;
- [ ] existing foundation tests pass;
- [ ] numerical operations required by the GRU are implemented;
- [ ] scalar/reference numerical path exists;
- [ ] numerical correctness is tested;
- [ ] CPU baseline is benchmarked;
- [ ] actual bottlenecks are profiled;
- [ ] compiler vectorization is evaluated;
- [ ] explicit SIMD is used only where evidence justifies it;
- [ ] stateful GRU exists;
- [ ] GRU gate convention is documented;
- [ ] repeated single-observation stepping works;
- [ ] GRU reset works;
- [ ] failed model updates preserve committed state;
- [ ] streaming execution drives the GRU without embedding GRU mathematics in the execution layer;
- [ ] long-sequence behavior is benchmarked;
- [ ] allocation/per-step overhead is understood sufficiently for the sprint;
- [ ] Connectome adapter works outside Seqvex core;
- [ ] no speculative future architecture has been frozen;
- [ ] `cargo test` passes;
- [ ] `cargo fmt --check` passes;
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes;
- [ ] documentation reflects the implementation rather than the intended tree alone.

---

# 33. Sprint closeout: what this document must preserve

This document should remain in `docs/` after the sprint.

It serves as a historical boundary marker:

```text
SEQVEX FOUNDATION
       ↓
CONNECTOME VERTICAL SLICE SPRINT
       ↓
RETURN TO BROADER SEQVEX DEVELOPMENT
```

After the sprint, the repository should **not** be interpreted as if:

- Seqvex became a Connectome framework;
- Seqvex became a quant framework;
- GRU became the canonical Seqvex model;
- CPU became the canonical execution device;
- SIMD became a permanent architectural requirement.

Instead, the sprint should leave behind:

1. a concrete end-to-end sequential ML implementation;
2. empirical evidence about the computational path;
3. validated streaming/state semantics;
4. performance measurements;
5. lessons about numerical representation and memory behavior;
6. a clearer understanding of which abstractions deserve to become permanent;
7. explicit unresolved questions that should remain deferred.

At closeout, update this document with a short **Sprint Results** section containing:

```text
Implemented
-----------
What actually shipped.

Measured
--------
Important benchmark/profile results.

Learned
-------
What the implementation demonstrated.

Changed
-------
Any architectural decision that changed because of evidence.

Deferred
--------
Questions intentionally left unresolved.

Next
----
Which normal Seqvex roadmap work resumes after this sprint.
```

Do not rewrite history to make the sprint appear cleaner than it was. The value of this document is that future development can distinguish **evidence-backed architecture** from **temporary sprint implementation choices**.

---

# 34. Final KiloCode instruction

> **Build the capability, not the directory tree.**

The correct target is:

```text
existing foundation
       ↓
minimal numerical computation
       ↓
CPU reference path
       ↓
measure + profile
       ↓
targeted optimization
       ↓
stateful GRU
       ↓
streaming execution
       ↓
real workload adapter
       ↓
Connectome
```

The correct permanent identity remains:

```text
sequential / temporal / non-IID
            ↓
      streaming-first
            ↓
          ML / RL
            ↓
 CPU / GPU / accelerator
            ↓
   hardware-aware execution
```

Connectome is one workload.

GRU is one model.

CPU optimization is one sprint target.

None of these redefine Seqvex.

> **Do the smallest correct thing that gets the project to the next measurable boundary.**

---

# 35. Sprint Results

> Recorded at the end of the vertical-slice implementation. Kept as an honest
> boundary marker, including work that is blocked or deliberately deferred.

## Implemented

- **Foundation reconciliation.** `src/foundation/numerical/statistics.rs` no
  longer carries the stray imports (`core::error`, `std::arch::x86_64`,
  `std::collections::btree_map::Values`, `std::fmt::Alignment::Left`), the stale
  `#![allow(unused_variables)]`, or the "not implemented" module docs. The
  developer's algorithms are preserved: two-pass population variance and
  Welford-style online variance. `tests/numerical.rs` is un-ignored and extended
  to 28 tests covering empty/singleton/NaN/±Inf, population-vs-sample variance,
  online-vs-batch agreement, and the documented panic contract.
- **`f32` numerical substrate** (`src/foundation/numerical/`):
  - `vector.rs` — contiguous `Vector` with `zeros`, `from_slice`, `from_fn`,
    indexed access, iteration, `map`, `add`, `multiply`, `scale`, `complement`,
    `dot`; shape failures return `DimensionMismatch`.
  - `linalg.rs` — dense row-major `Matrix` and `y = W x` (`mul_vector`).
  - `activations.rs` — scalar `sigmoid` and `tanh`.
  - `f64` statistics and the `f32` substrate remain separate.
- **Numerical benchmark** `benches/numerical.rs` (`harness = false`, std timing).
- **Execution layer** `src/execution/streaming.rs` — `StreamingExecutor`
  composes the foundation `process_one`/`process_stream` rather than duplicating
  the fold; it adds committed-state retention and an explicit `reset` boundary.
- **Stateful GRU** `src/models/recurrent/gru.rs` implementing the foundation
  `StateModel` (State = hidden `Vector`), so atomicity is structural. One
  documented gate convention; `step`/`reset`/`hidden` plus candidate validation
  and failure classes.
- **Tests and examples:** `tests/gru.rs` (14 tests, including an independent
  scalar reference for small dimensions and a 500-step reference fold),
  `tests/streaming_execution.rs` (5 tests, first against a trivial `StateModel`
  fixture, then the full GRU path), `examples/streaming_gru.rs`,
  `examples/state_reset.rs`, `benches/gru.rs`.

Not implemented: `competition/connectome/**`. The competition interface is not
in the repository, and the sprint forbids guessing its shape. Only the boundary
(§23–24) is fixed; no competition concept entered `src/`.

## Measured

`cargo bench --bench numerical` (scalar reference, release):

```text
size 512      dot 235 ns     matvec 104,024 ns     map sigmoid 1,215 ns
size 128      dot  43 ns     matvec   6,113 ns     map sigmoid   287 ns
size  32      dot   8 ns     matvec     297 ns     map sigmoid    73 ns
scalar sigmoid 2.4 ns; scalar tanh 7.4 ns (size-independent)
```

`cargo bench --bench gru` (100,000 steps/measurement):

```text
input=8   hidden=16     868 ns/step   1,151,589 obs/s   26.00 allocs/step   1,664 bytes/step
input=32  hidden=64   5,860 ns/step     170,649 obs/s   26.00 allocs/step   6,656 bytes/step
input=128 hidden=256 104,492 ns/step      9,570 obs/s   26.00 allocs/step  26,624 bytes/step
```

`cargo test`, `cargo fmt --check`, and
`cargo clippy --all-targets --all-features -- -D warnings` all pass.

## Learned

- The GRU step allocates **26 vectors per step, independent of dimensions**. Cost
  scales with state size, but allocation count is constant, so allocation is a
  distinct axis from mat-vec work. This is a per-observation overhead the
  streaming workload will magnify (§26), and it is the strongest measured
  candidate for the next optimization — ahead of SIMD.
- `matvec` dominates at larger sizes (104 µs at 512×512 vs 235 ns for a 512-dot),
  which is expected for the reference scalar loop. No profiling tool was run, so
  this is a benchmark observation, not a profile-confirmed bottleneck.
- Scalar activations are cheap (2–7 ns) relative to a step at realistic sizes;
  the per-element `map` cost is dominated by allocation.

## Changed

- **Error semantics.** The existing `f64` statistics keep their documented panic
  contract; the new `f32` substrate and GRU return `Result`
  (`DimensionMismatch`, `GruError`). No existing semantics were silently changed.
- **Execution vs foundation duplication.** The execution layer composes
  `process_one`/`process_stream`; the fold exists in exactly one place. No
  duplication and no foundation rewrite.
- **`forbid(unsafe_code)`.** Unchanged. No measurement justified SIMD, so the
  narrow-unsafe escape hatch was not opened.
- No architectural abstraction was frozen: no universal model/tensor/device/
  optimizer interface was introduced.

## Deferred

- **Sprint record filename.** The document recommends
  `docs/KILO_CONNECTOME_SPRINT.md`, but the file on disk is
  `docs/KILO_CONNECTOME_SPRINT_REVISED.md`. §33 says to update *this* document,
  and moving/renaming it would rewrite repository history, so this section was
  appended to the existing file. **Confirm whether to rename it.**
- **Connectome adapter.** Blocked on the external competition interface; the
  boundary is specified, the files are not written.
- **Allocation reduction.** 26 allocs/step is measured but not yet optimized;
  a scratch-buffer or in-place strategy needs its own benchmark comparison.
- **SIMD / transposed or packed layouts.** Deferred until a measured hot path
  justifies them (§19).
- **`.gitignore` ignores `AGENTS.md`** (§28): raised, not changed. Confirm whether
  the Ponytail instruction file should be version-controlled.

## Next

The sprint path `foundation → substrate → GRU → streaming execution →
benchmark` is real and tested. The remaining work is the measured optimization
loop (budgeted by the allocation evidence) and the externally blocked Connectome
adapter. Absent that interface, normal Seqvex roadmap work resumes; GRU, CPU,
and SIMD remain sprint artifacts, not permanent architecture.

## Reconciliation audit

A reconciliation audit (governed by `docs/KILO_SPRINT_RECONCILIATION.md`) compared
the repository against this specification and the architecture docs. It found **no
behavioral or correctness defects**: the GRU recurrence matches its documented
convention against an independent scalar reference, failure atomicity is structural
via `StateModel`, and the execution layer composes the foundation fold without
duplication. No competing state model, tensor/device/optimizer abstraction, or
speculative directory was introduced.

One **gate defect** was found and repaired minimally.
`src/foundation/numerical/statistics.rs` had reverted to its unimplemented stub
form — stray imports, a stale `#![allow(unused_variables)]`, stale "not
implemented" module docs, and Clippy lints — which failed
`cargo clippy --all-targets --all-features -- -D warnings`. The hand-written
algorithms were preserved (two-pass population variance, Welford-style online
variance); only the surrounding defects were removed.

Status distinctions used by this record:

- **Implemented** — foundation, `f32` substrate, `f64` statistics, streaming
  execution, stateful GRU, tests, benches, examples.
- **Measured** — the benchmark numbers above. These are **measurements, not
  profile-confirmed bottlenecks**.
- **Profiled** — nothing. No profiler has been run.
- **Deferred** — profiling, auto-vectorization/SIMD evaluation, and allocation
  reduction, pending evidence.
- **Blocked** — the Connectome adapter, pending the external competition interface.
- **Future** — everything in the broader Seqvex roadmap (§33).

The only changes during reconciliation were the `statistics.rs` repair and the
two documentation corrections above (the closeout no longer depends on the
implementation plan's private decision labels, and the statuses are stated
directly). The sprint closes here.

