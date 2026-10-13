# Seqvex Connectome Sprint — Audit & Correction Instruction

> **Historical reference — superseded execution instructions:** retain this record for its original scope and evidence; do not execute it as current implementation authorization. Any earlier prediction-only scope, preset-parameter demonstration or completion claim must be assessed under the [mandatory training-to-inference contract](DEVELOPMENT.md#model-training-and-valid-inference--mandatory) and current Architect → Planner → Coder gates. Raise unresolved failure decisions to the maintainer.

## 1. Purpose

This document is an **audit and correction instruction**, not a new implementation plan.

The Connectome vertical slice has already been substantially implemented in the repository. The task now is to compare the **actual repository state** against:

1. the agreed Connectome sprint specification;
2. the existing Seqvex architecture and development documents;
3. the KiloCode-generated Connectome plan;
4. the intended scope and boundaries of the sprint.

Do **not** restart the implementation.

Do **not** redesign Seqvex.

Do **not** create speculative infrastructure.

Do **not** assume that every difference from the sprint document is a defect. First classify each difference.

---

# 2. Authority and Precedence

Use the following precedence when reconciling conflicts:

1. Existing Seqvex architectural constraints and invariants
   - `ARCHITECTURE.md`
   - `docs/FAILURE_AND_RECOVERY.md`
   - `docs/DEVELOPMENT.md`
   - `docs/KILOCODE_CONTEXT.md`
   - `docs/ROADMAP.md`

2. The agreed sprint specification:
   - `docs/KILO_CONNECTOME_SPRINT_REVISED.md`

3. Existing implementation and tests in the repository.

4. The KiloCode-generated Connectome plan.

The Kilo-generated plan is an **implementation aid**, not a new architectural authority.

If the Kilo plan conflicts with the established architecture or sprint specification, do not preserve the Kilo plan merely because it already exists.

---

# 3. Critical Development Boundary

The intended development workflow requires human ownership of production implementation.

For this reconciliation task:

- Inspect existing code.
- Review existing tests.
- Identify defects and deviations.
- Propose the required correction.
- Implement only corrections that are clearly justified by the audit and explicitly within this task's scope.
- Do not expand the architecture.
- Do not introduce new abstractions merely because they appear convenient.
- Do not implement future roadmap infrastructure.
- Do not turn a benchmark observation into an architectural commitment.

If a correction requires a design decision that is not already supported by the existing architecture or sprint specification, **stop and report the decision required instead of inventing one**.

The goal is to bring the existing implementation into alignment, not to maximize the amount of code produced.

---

# 4. First Task: Reconcile the Repository

Before modifying anything:

1. Inspect the working tree.
2. Inspect the current branch and repository status.
3. Read the relevant architecture/development documents.
4. Read `docs/KILO_CONNECTOME_SPRINT_REVISED.md`.
5. Read the Kilo-generated Connectome plan.
6. Inspect all files added or materially changed for the vertical slice.
7. Inspect the tests, examples, and benchmarks.
8. Run the existing validation commands before making changes.

Do not overwrite unrelated local work.

Create an internal audit table with these categories:

| Category | Meaning |
|---|---|
| Correct | Implementation agrees with the specification and architecture. |
| Acceptable variation | Different implementation, but behavior and architectural intent are preserved. |
| Defect | Behavior, correctness, safety, or integration is wrong. |
| Incomplete | Required sprint functionality is missing. |
| Documentation mismatch | Code is acceptable but documentation is inaccurate or stale. |
| Performance finding | Measured behavior requiring investigation, not automatically a defect. |
| Deferred | Explicitly outside the current available evidence/scope. |
| Speculative | Work exists that should not have been introduced without architectural justification. |

Do not label something a defect merely because it differs from the plan.

---

# 5. Required Audit Areas

## 5.1 Foundation Semantics

Verify that the existing foundation semantics remain intact.

Check:

- observation representation;
- ordering semantics;
- single-observation processing;
- bounded micro-batch semantics;
- state transitions;
- failed-update atomicity;
- sequential continuity after failure;
- deterministic behavior;
- invalid-input behavior;
- numerical failure behavior.

The new vertical slice must build on these semantics rather than bypassing them.

Pay particular attention to whether GRU and streaming execution use the existing state/failure contracts rather than silently introducing a competing state model.

---

# 6. Numerical Substrate Audit

Inspect:

- `src/foundation/numerical/vector.rs`
- `src/foundation/numerical/linalg.rs`
- `src/foundation/numerical/activations.rs`
- `src/foundation/numerical/statistics.rs`
- `src/foundation/numerical/mod.rs`

Verify:

### Vector

- dimension invariants;
- indexing/access behavior;
- elementwise operations;
- dimension mismatch handling;
- allocation behavior where relevant;
- deterministic results.

### Linear algebra

- matrix/vector dimensions;
- matrix-vector multiplication;
- numerical correctness against simple reference calculations;
- error handling;
- no hidden architectural dependency on a future tensor system.

### Activations

- sigmoid;
- tanh;
- expected numerical behavior;
- sensible handling of values at the tested numerical extremes.

### Existing statistics

Do not silently rewrite established `f64` statistics contracts merely to make the new `f32` substrate look uniform.

The current architecture intentionally permits the existing statistics layer and the GRU substrate to use different numerical types where justified.

Any change to this boundary requires explicit architectural justification.

---

# 7. GRU Mathematical Audit

Inspect:

`src/models/recurrent/gru.rs`

Verify the implementation mathematically rather than judging it by naming or structure.

Confirm:

- input dimension;
- hidden dimension;
- parameter dimensions;
- recurrent state;
- update/reset gate convention;
- candidate hidden-state calculation;
- final hidden-state update;
- output/state semantics;
- deterministic behavior.

Use the independent scalar reference implementation in the tests, where available.

The multi-step reference-fold test is especially important: verify that repeated execution matches the independently calculated recurrence rather than merely checking that the implementation is self-consistent.

Do not change the GRU equations simply because another GRU convention exists. Preserve the documented convention unless the implementation is demonstrably inconsistent with its own specification/tests.

---

# 8. State and Failure Atomicity Audit

Verify that a failed GRU update cannot silently commit partial or invalid state.

Test at least conceptually:

```text
valid state
    ↓
attempt update
    ↓
failure
    ↓
previous committed state remains intact
```

Also verify reset semantics.

The implementation must not claim rollback semantics beyond what the existing state architecture supports.

Do not introduce a general transaction system, checkpoint framework, rollback framework, or persistence layer for this sprint.

---

# 9. Streaming Execution Audit

Inspect:

`src/execution/streaming.rs`

Verify that the execution layer:

- processes one observation correctly;
- supports streaming sequences;
- preserves sequential state;
- handles reset explicitly;
- composes existing foundation semantics;
- does not duplicate model-specific mathematical logic;
- does not become a general-purpose scheduler;
- does not introduce speculative execution infrastructure.

The execution layer should provide execution semantics.

The GRU should provide GRU computation.

Keep those responsibilities distinct.

---

# 10. Public Crate Integration

Inspect:

`src/lib.rs`

and all relevant module declarations.

Verify that the implemented vertical slice is actually reachable through the intended public crate surface.

In particular, check whether newly implemented modules such as:

- `execution`
- `models`

are correctly exposed where the public API requires them.

Do not expose internal implementation details merely for convenience.

Examples and integration tests should exercise the same public surface intended for users wherever practical.

---

# 11. Tests

Audit existing tests rather than merely counting them.

Required coverage should include, as applicable:

### Numerical

- correctness;
- dimension mismatch;
- edge cases;
- deterministic behavior.

### GRU

- construction;
- dimensions;
- reset;
- one-step behavior;
- multi-step behavior;
- independent reference comparison;
- failure behavior;
- state continuity.

### Streaming

- one observation;
- multiple observations;
- state continuity;
- reset;
- model integration.

### Foundation

Ensure the new implementation has not broken the existing foundation test suite.

A test that only reproduces the implementation's own logic is weaker evidence than an independent reference calculation.

Do not add large numbers of tests merely to increase test count.

---

# 12. Benchmark Audit

Inspect:

- `benches/numerical.rs`
- `benches/gru.rs`

The current benchmark measurements are engineering measurements, not proof of an optimal implementation.

Distinguish:

```text
benchmark observation
        ≠
profile-confirmed bottleneck
        ≠
architectural conclusion
```

The current reported GRU allocation behavior must be treated as a measured optimization target:

- approximately 26 allocations per step were observed;
- allocation count appeared independent of model dimension in the recorded measurements;
- this is important;
- but it does not by itself determine the correct optimization.

Do not immediately introduce:

- custom allocators;
- arena allocators;
- unsafe code;
- tensor engines;
- global memory pools;
- object pools;
- scheduler infrastructure.

First determine where the allocations originate and whether reducing them materially improves the workload.

---

# 13. CPU Optimization / SIMD

CPU optimization is a **sprint workstream**, not a permanent architectural constraint.

Follow this sequence:

```text
correctness
→ benchmark
→ profile
→ identify bottleneck
→ optimize
→ benchmark again
→ verify correctness again
```

Prefer:

1. clear scalar/reference implementation;
2. compiler optimization;
3. data/layout improvements;
4. allocation reduction;
5. measured specialization;
6. explicit SIMD only where profiling and evidence justify it.

Do not introduce unsafe SIMD merely because the sprint mentions CPU optimization.

The repository currently uses:

```rust
#![forbid(unsafe_code)]
```

Do not remove that constraint as part of this reconciliation unless an explicit architectural decision is made outside this task.

---

# 14. Memory and Allocation Audit

Because Seqvex is intended for sequential/streaming workloads, inspect allocation behavior carefully.

For each important hot-path allocation, determine:

- where it occurs;
- whether it is required for correctness;
- whether it is caused by temporary vectors;
- whether ownership/borrowing can eliminate it;
- whether the compiler can already optimize it;
- whether a reusable state buffer would preserve the existing architecture;
- whether the optimization is measurable.

Do not optimize allocation count in isolation.

The relevant question is:

> Does the allocation behavior materially constrain the intended sequential workload, and is there a simpler correction that preserves the existing semantics?

---

# 15. Examples

Inspect:

- `examples/streaming_gru.rs`
- `examples/state_reset.rs`

Examples should demonstrate actual intended usage rather than internal implementation details.

Verify:

- they compile;
- they exercise the public API;
- they demonstrate streaming state;
- reset behavior is understandable;
- no example implies that Seqvex is a Connectome-specific framework;
- no example implies that GRU is the framework's permanent model abstraction.

---

# 16. Connectome Boundary

Connectome is the **workload/application motivating this vertical slice**.

It is not Seqvex's architectural identity.

Therefore:

- Connectome-specific code must remain outside core Seqvex;
- do not add Connectome domain semantics to `src/`;
- do not add competition-specific data ingestion to Seqvex core;
- do not invent an external competition API;
- do not fabricate an adapter contract if the actual external interface is unavailable.

If the external Connectome interface remains unavailable, document the integration point as blocked/deferred.

That is preferable to guessing.

---

# 17. Repository Structure Audit

Check whether newly created directories/files are justified by current responsibilities.

Do not create empty future directories merely because they appear in a target tree.

Do not create:

- GPU backends;
- device abstraction layers;
- tensor frameworks;
- autodiff;
- optimizers;
- RL infrastructure;
- schedulers;
- distributed execution;
- serialization frameworks;
- universal allocators;
- generic plugin systems;
- generic data pipelines.

unless already required by the actual current implementation and architecture.

The target tree is a destination for justified work, not a checklist requiring every directory to exist immediately.

---

# 18. Documentation Reconciliation

After code auditing, check:

- `README.md`
- `ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/KILOCODE_CONTEXT.md`
- `docs/ROADMAP.md`
- `docs/FAILURE_AND_RECOVERY.md`
- `docs/KILO_CONNECTOME_SPRINT_REVISED.md`

Documentation must accurately distinguish:

### Implemented

What actually exists and is tested.

### Measured

What has been benchmarked.

### Profiled

What has been confirmed by profiling.

### Deferred

What remains intentionally unfinished.

### Blocked

What cannot proceed because required external information is unavailable.

### Future

What belongs to a later architectural phase.

Do not upgrade a benchmark observation into a performance claim.

Do not upgrade a prototype into a permanent architecture.

Do not claim Connectome integration if only the Seqvex vertical slice exists.

---

# 19. Kilo-Generated Plan

Treat the Kilo-generated Connectome plan as an implementation-history artifact.

Do not allow it to become a competing source of architectural truth.

After the audit:

- retain it temporarily if it provides useful historical context;
- extract any durable architectural decisions into the appropriate authoritative documentation;
- otherwise remove it from the repository rather than maintaining duplicate planning documents.

Do not rewrite the entire sprint specification to conform to Kilo's plan.

---

# 20. Correction Rules

When a discrepancy is found:

### If it is correct

Leave it alone.

### If it is an acceptable variation

Leave it alone and document why it is acceptable if future confusion is likely.

### If it is a real defect

Fix the smallest thing necessary.

### If it is incomplete but required

Implement only the missing required capability.

### If it is a performance issue

Measure/profile before redesigning.

### If it is a documentation issue

Correct the documentation.

### If it is speculative architecture

Do not expand it further. Remove it only if doing so is safe and clearly within the current task.

### If the correct solution requires a new architectural decision

Stop and report the decision rather than inventing one.

---

# 21. Validation After Correction

After corrections:

Run, as applicable:

```text
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo bench
cargo run --example streaming_gru
cargo run --example state_reset
```

Use the repository's actual supported commands if any differ.

Confirm:

- existing foundation tests pass;
- new vertical-slice tests pass;
- examples compile/run;
- public crate integration works;
- no unrelated functionality was broken.

If a command cannot be run, report exactly why.

Do not claim validation that was not actually performed.

---

# 22. Final Audit Report

At the end, produce a concise report with:

## A. Repository state

What is currently implemented.

## B. Confirmed correct

What was audited and found sound.

## C. Corrections made

Exact files and behavioral corrections.

## D. Remaining issues

Only concrete unresolved issues.

## E. Performance findings

Separate benchmark observations from profile-confirmed findings.

## F. Deferred/blocked

Especially Connectome integration if the external interface remains unavailable.

## G. Documentation changes

What was updated and why.

## H. Kilo plan disposition

State whether the Kilo-generated plan should be:

- retained temporarily;
- retained as historical documentation;
- consolidated into another document;
- deleted.

## I. Stop condition

Once the audit, required corrections, validation, and documentation reconciliation are complete:

**STOP.**

Do not continue into the next roadmap phase.

Do not add speculative abstractions.

Do not begin another architecture/planning cycle.

The sprint should close with the smallest validated implementation that satisfies the agreed vertical-slice objective.

---

# 23. Core Principle

The purpose of this task is not to make Seqvex look more complete.

The purpose is to determine whether the implementation is:

```text
correct
+ coherent
+ tested
+ measurable
+ consistent with Seqvex's architecture
+ appropriately scoped
```

Anything beyond that belongs to a future evidence-driven development cycle.
