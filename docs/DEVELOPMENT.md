# Seqvex Development Contract

> **Status:** Canonical development contract  
> **Audience:** Human contributors and AI-assisted development agents  
> **Project:** Seqvex

This document defines how Seqvex should be developed.

It is intentionally stricter than a conventional contribution guide because the project is still establishing its architecture. The goal is to prevent premature abstraction, accidental architectural commitments, and optimization decisions that later constrain hardware-aware execution.

---

# 1. Core Development Philosophy

Seqvex should be developed through a closed learning loop:

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

The loop is not a mandatory checklist for every tiny change. Documentation edits and straightforward bug fixes may require only part of it.

The principle is:

> **Implementation produces evidence; evidence shapes architecture.**

---

# 2. Architecture Before Implementation

Before implementing a non-trivial change:

1. Read the relevant documentation.
2. Inspect the existing implementation.
3. Identify the invariants being protected.
4. Classify the proposed change.
5. Determine whether the architecture is settled, tentative, experimental, or deferred.
6. Choose the smallest change that tests the requirement.
7. Identify any architecture-level consequences before coding.

Do not silently settle a deferred decision merely because an implementation needs an answer.

When architecture is uncertain:

> **Build an experiment instead of prematurely committing to an abstraction.**

---

# 3. Critical Architecture Review Guard

## 3.1 Mandatory rule

**Any implementation, optimization, or API change that could materially constrain, complicate, or prematurely freeze:**

- hardware-aware execution;
- heterogeneous memory/device placement;
- accelerator offload;
- CPU/GPU/device residency;
- per-stream versus model-owned state or scratch;
- synchronization strategy;
- Rust-native low-level optimization;
- low-level allocation/layout control;

**must undergo CRITICAL ARCHITECTURE REVIEW before implementation.**

This is a permanent development rule.

It applies even when the proposed change appears locally small.

## 3.2 Why this guard exists

Seqvex is deliberately leaving several high-reversal-cost decisions open.

A locally efficient implementation can accidentally make a later design much harder if it assumes:

```text
model owns all mutable execution resources
```

when the eventual architecture may require:

```text
model parameters
    +
per-stream state
    +
per-stream workspace
    +
device-specific execution
```

or another topology.

The purpose of the review is not to prevent optimization.

It is to prevent an optimization from silently becoming the architecture.

## 3.3 Examples that require review

Examples include:

- changing a public API from `&T` to `&mut T` because an implementation needs mutable scratch;
- making a model own a reusable workspace;
- introducing a generic allocator or memory pool;
- introducing a device abstraction;
- introducing a scheduler;
- making a tensor representation globally canonical;
- adding a synchronization model;
- adding GPU/accelerator-specific ownership assumptions;
- introducing unsafe low-level primitives that constrain later backends;
- changing state ownership to make a benchmark faster.

The former GRU model-owned workspace and `StreamingExecutor` `&mut M` API were
active examples of this trigger; both were removed by the execution-ownership
refactor (model borrowed immutably, per-execution State and algorithm-local
Workspace). Any future model-owned scratch or exclusive model borrow is a new
example.

## 3.4 Review output

A CRITICAL ARCHITECTURE REVIEW should state:

- the concrete requirement;
- current evidence;
- affected invariants;
- alternatives considered;
- reversal cost;
- what is being intentionally left undecided;
- whether the implementation should proceed;
- what measurements or workloads are still required.

A review does not automatically authorize implementation.

---

# 4. Seqvex Semantic Hierarchy

Implementation must preserve the distinction between:

```text
single observation
        ↓
streaming / online
        ↓
bounded micro-batch
        ↓
larger batch
```

The intended hierarchy is:

- **Single observation** — fundamental semantic unit.
- **Streaming / online** — primary computational context.
- **Bounded micro-batching** — optimization/capability.
- **Larger batch execution** — secondary capability where useful.

This does not mean Seqvex cannot process large datasets or high-dimensional observations.

It means sequential/stateful semantics, temporal ordering, and evolving computation—not dataset size—are the organizing principles.

---

# 5. Execution Semantics vs Hardware Placement

Keep these dimensions separate.

Execution semantics describe:

```text
single observation
streaming / online
micro-batch
batch
```

Placement describes:

```text
CPU
GPU
accelerator
heterogeneous resources
```

A hardware optimization must not silently redefine the temporal or state semantics of an algorithm.

---

# 6. State and Failure Semantics

For stateful operations, reason explicitly about:

```text
current valid state
        ↓
candidate state
        ↓
validation
        ↓
commit
```

A failure must not silently commit an invalid or partial state.

Before implementing a stateful operation, determine:

- what can fail;
- whether valid state can be preserved;
- whether retry is safe;
- whether rollback is meaningful;
- whether processing should continue;
- what must be reported;
- whether recovery preserves required determinism.

Rollback is not universal. Do not add rollback machinery merely because failure exists.

See `docs/FAILURE_AND_RECOVERY.md` for detailed failure policy.

---

# 7. Scope Boundary

Seqvex owns computational responsibility from:

- ML/RL preprocessing and representation;
- model computation;
- training and learning;
- validation;
- inference;
- numerical computation required by ML/RL;
- execution/runtime mechanisms.

General-purpose ingestion, storage, databases, ETL, generic cleaning/manipulation, exploratory analysis, visualization, and domain/business logic remain outside the framework.

The boundary is based on the role of a computation in the ML/RL pipeline.

---

# 8. Build From First Principles

Seqvex should be understandable from its foundations.

Existing libraries may be used when they provide clear value, but dependencies should not determine Seqvex's architecture unnecessarily.

Before adopting a dependency, consider:

- problem solved;
- ownership/licensing;
- compile-time impact;
- runtime overhead;
- allocation behavior;
- portability;
- `std`/`no_std` implications;
- hardware/backend constraints;
- whether it introduces an abstraction Seqvex would otherwise need to understand itself.

Do not reject dependencies merely because they are external. Do not add dependencies merely because they are convenient.

---

# 9. Modularity

A module should represent a coherent responsibility.

Do not create a README, crate, abstraction, or helper solely because a folder or function exists.

A shared abstraction should emerge when:

- responsibility is genuinely cross-functional;
- a stable semantic contract exists;
- local ownership creates meaningful duplication of domain knowledge;
- or sharing is required for a concrete architectural property.

> **DRY should remove duplicated knowledge, not erase useful ownership boundaries.**

---

# 10. Shared Infrastructure and Global Helpers

Do not introduce global `common`, `utils`, or helper modules merely to remove local duplication.

The same applies to:

- shared error types;
- state helpers;
- numerical utilities;
- execution contexts;
- memory/storage abstractions;
- test support.

Similar implementation is not sufficient evidence for a shared abstraction.

Prefer local ownership first, then refactor upward when repeated responsibility and dependency relationships demonstrate that the abstraction is genuinely shared.

---

# 11. Crate and Workspace Boundaries

Seqvex is currently a **single Cargo package**.

A workspace and additional crates may emerge later when demonstrated responsibility boundaries justify them.

Possible future areas include:

- core types/errors;
- numerical computation;
- streaming/online learning;
- sequential models;
- supervised/unsupervised learning;
- reinforcement learning;
- validation;
- execution/runtime;
- device backends.

These are directions, not a frozen crate list.

---

# 12. Rust Engineering Principles

These principles are design tools, not patterns to apply mechanically.

- **DRY:** avoid duplicated knowledge and invariants.
- **Composition over inheritance:** prefer focused structures.
- **Traits:** introduce for real semantic contracts or meaningful implementations.
- **Extension traits:** use for coherent behavior on types Seqvex does not own.
- **Builder pattern:** use for meaningful configuration and validation.
- **Typestate:** use only when compile-time distinctions provide real correctness.
- **Newtype:** use for distinct semantic values.
- **Error enums:** prefer domain-specific errors and `Result<T,E>`.
- **Iterators:** use where they improve clarity without material performance cost; explicit loops are appropriate when clearer or measurably better.

## Model training and valid inference — mandatory

Every Seqvex ML model must support actual data ingestion and its declared training/learning procedure, then calculate inference outputs from the learned result. Training must precede user-facing inference or prediction. This is a project-wide completion and delivery requirement with no prediction-only model exception.

- A delivered model must provide a usable data → train/learn → validate → infer/predict path, with mathematically correct calculations, independent numerical checks and task/regime-appropriate prediction-quality evidence. There is no completion waiver for a basic/reference implementation. The Architect defines the algorithm-specific learning and validation contract; the Planner specifies its implementation and evidence.
- Preset coefficients, authored outputs, dummy models and partial prediction primitives must not stand in for a working trained model in user demonstrations or feature acceptance. A demonstration must execute the declared training procedure on data and use its learned result; printing predetermined expectations is not learning.
- Imported parameters alone do not establish this capability. The model must support training, and any separate inference deployment must have a validated link to an actual trained result.
- Existing prediction-only slices are incomplete development components, not exceptions, complete ML deliveries or acceptable training-to-prediction demonstrations. Preserve their history and report the missing capability; do not imply it has been implemented.
- Training does not itself establish accuracy on unseen observations. Mathematical, numerical, statistical and operational evidence must support the claims being made.

## Calculation failures and error handling — mandatory

Model operations must implement their accepted mathematics within explicit input, numerical and resource limits. Report failures through defined errors and preserve the declared committed-state boundary.

- Never silently turn a failed calculation into a successful prediction by substituting zero, a default, a cached value, fabricated output or repaired input.
- Propagate an error or handle it explicitly at the application boundary. Reject invalid observations/candidates; preserve previously valid state where the contract requires it. Retry only under a defined, meaningful recovery policy. Do not label a failed observation's previous state as its prediction.
- Error enums identify failure classes; they do not justify panicking on recoverable input failures. User demonstrations must teach the approved handling, and incorrect expected results/errors must fail verification.
- Declare and validate the supported numerical envelope. An invalid/non-finite result must not be passed off as a valid model prediction. Existing primitive overflow semantics require review and an explicit implementation decision before change; documentation of a limitation is not model-completion evidence.
- No silent fallback is authorized. If an error/failure behavior, numerical boundary, continuation, retry, rejection, reset or resource policy is missing, ambiguous or contradicted, bring the exact case, impact and available options to the maintainer before affected work proceeds. Do not infer a policy or choose a substitute output. Existing documented behavior must be reported as fact, not treated as permission for an unreviewed new policy. Required numerical/API/recovery changes follow Architect design, Planner plan and scope approval before Coder implementation.

## Unsafe Rust

`unsafe` may be justified for:

- SIMD;
- specialized memory access;
- FFI;
- accelerator interfaces;
- custom allocators;
- low-level kernels.

Unsafe boundaries require documented invariants and appropriate tests.

An unsafe optimization that could materially constrain later hardware-aware architecture requires CRITICAL ARCHITECTURE REVIEW first.

---

# 13. Numerical Foundation

Seqvex requires numerical computation for ML/RL but is not intended to replace general dataframe, scientific-computing, or statistical ecosystems.

Introduce numerical primitives incrementally when they support actual workloads.

Do not build a complete numerical ecosystem before real Seqvex workloads justify it.

---

# 14. Execution and State Semantics

Implementation should preserve the distinction between:

```text
single observation
streaming / online
micro-batch
batch
```

and:

```text
CPU
GPU / accelerator
heterogeneous
```

A backend optimization must not accidentally redefine the algorithm's temporal or state semantics.

The precise state-transition implementation mechanism remains deliberately open until actual state models and failure characteristics justify a stable abstraction.

---

# 15. Automatic Execution Selection

Seqvex may eventually support both:

```text
Explicit
    user selects execution policy

Auto
    user permits Seqvex to select among available strategies
```

Automatic selection is intended to assist execution choice, not remove user control.

A user must be able to override Auto explicitly.

Potential signals include:

- workload dimensions;
- observation frequency;
- statefulness;
- latency/throughput requirements;
- computational intensity;
- memory behavior;
- available hardware;
- transfer/synchronization costs;
- historical replay versus live operation;
- online learning requirements;
- whether micro-batching preserves semantics.

This is a **future architecture/design direction**, not a current implementation.

Do not create a generic scheduler or automatic execution framework until multiple real execution strategies and workloads demonstrate a stable requirement.

---

# 16. Failure, Recovery, and Determinism

Failure classes may include:

- invalid input;
- numerical failure;
- state/model invariant failure;
- memory/allocation failure;
- runtime failure;
- hardware/device failure.

For each stateful operation, consider:

- what can fail;
- state preservation;
- retry safety;
- rollback semantics;
- continuation policy;
- error reporting;
- deterministic recovery where promised.

Determinism is a capability, not a universal requirement.

Do not promise deterministic floating-point behavior across arbitrary hardware unless measured and intentionally specified.

---

# 17. Testing

Testing should establish behavior and invariants, not merely increase coverage numbers.

For foundational components, test:

- valid behavior;
- invalid inputs;
- boundaries;
- ordering;
- state transitions;
- state preservation after failure;
- repeated streaming updates;
- deterministic behavior where promised;
- numerical edge cases;
- single-observation semantics;
- bounded micro-batch semantics where implemented.

Ask:

> **What invariant does this test protect?**

---

# 18. Formatting and Clippy

The development baseline is:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Warnings should not be suppressed casually.

Compilation success is not sufficient evidence of correctness.

---

# 19. Performance Engineering

Performance must be measured.

```text
Implement
   ↓
Verify correctness
   ↓
Benchmark
   ↓
Profile
   ↓
Identify bottleneck
   ↓
Optimize
   ↓
Benchmark again
   ↓
Keep or revert based on evidence
```

Potential surfaces include:

- allocations;
- cache locality/misses;
- data layout;
- alignment;
- SIMD/vectorization;
- branch behavior;
- memory bandwidth;
- synchronization;
- contention;
- copying;
- CPU/accelerator transfer;
- kernel launch overhead;
- device utilization;
- compiler/LLVM behavior.

> **Zero allocation is not the definition of performance.**

> **GPU execution is not automatically faster than CPU execution.**

Do not make performance claims without appropriate benchmarks or profiling.

---

# 20. Hardware-Aware Design and Optimization

Seqvex should remain capable of exploiting appropriate hardware without turning every optimization mechanism into an architectural commitment.

Preserve the properties needed for efficient implementations:

- sensible data layout;
- ownership;
- locality;
- vectorization;
- parallel execution;
- accelerator placement;
- controlled data movement.

Specific mechanisms such as explicit SIMD, Rayon, CUDA, ROCm, NUMA-aware allocation, pinned memory, or zero-copy paths are implementation choices, not requirements for every component.

Use this optimization ladder where relevant:

```text
correctness
    ↓
reference implementation
    ↓
benchmark
    ↓
profile
    ↓
data / layout optimization
    ↓
compiler / auto-vectorization
    ↓
explicit SIMD
    ↓
CPU parallelism
    ↓
accelerator / specialized backend
```

Not every workload should progress through every step.

> **Hardware-aware design is an architectural constraint; hardware-specific optimization is an evidence-driven implementation decision.**

---

# 21. Predictability and Resource Behavior

Remain aware of:

- bounded/unbounded work;
- allocation behavior;
- synchronization;
- memory footprint;
- state growth;
- data movement;
- deterministic behavior;
- failure containment.

Distinguish:

```text
architectural direction
implementation capability
measured property
formal guarantee
```

Do not convert future goals into present guarantees.

---

# 22. AI Development Contract

AI-assisted development is expected, but AI must operate within this contract.

Before modifying code, AI should:

1. read relevant documentation;
2. inspect the existing implementation;
3. identify invariants;
4. classify the change;
5. distinguish settled/tentative/deferred decisions;
6. choose the smallest useful change;
7. explain consequential assumptions;
8. run appropriate validation;
9. document evidence for material decisions.

AI must not:

- invent architecture without approval;
- silently settle deferred decisions;
- create speculative abstractions;
- add convenience dependencies without justification;
- generate opaque implementations the developer cannot understand;
- claim performance without evidence;
- redesign unrelated components;
- bypass CRITICAL ARCHITECTURE REVIEW.

Prefer:

```text
small change
   ↓
explain
   ↓
test
   ↓
measure
   ↓
review
   ↓
next change
```

The human developer retains responsibility for consequential design decisions.

---

# 23. Change Classification

## Local implementation change

Bug fixes, focused improvements, tests, or implementation of an already decided mechanism may proceed within existing architecture.

## Architectural change

Changes to:

- state semantics;
- execution semantics;
- abstraction boundaries;
- ownership/storage strategy;
- backend models;
- crate responsibilities;
- public API topology that affects hardware-aware execution;

require an explicit architectural decision before implementation.

If the change falls under the CRITICAL ARCHITECTURE REVIEW guard, review must occur **before** implementation.

## Experimental change

When architecture is uncertain:

> **Build an experiment instead of prematurely committing to an abstraction.**

Experiments should be:

- small;
- isolated;
- measurable;
- disposable;
- clearly separated from production code.

---

# 24. Issue and Planning Discipline

The step-by-step operational procedure that implements this discipline — issue decomposition, branch/commit/PR lifecycle, traceability, validation, and templates — is in [`FEATURE_DEVELOPMENT.md`](../FEATURE_DEVELOPMENT.md). This section remains authoritative for the rules; the runbook does not replace them.

GitHub issues are part of the project's architectural paper trail.

Before creating, reopening, updating, or closing an issue:

1. inspect the relevant repository evidence;
2. confirm the issue is actually required;
3. consult the owner before making consequential issue-state changes;
4. use the repository's existing labels, issue types, and milestones;
5. do not create duplicate labels, milestones, or issues;
6. preserve the issue's relationship to the relevant architecture and evidence.

For development-agent or AI-assisted reviews, record the review/audit outcome in the relevant GitHub issue comment even when the issue remains open or is later closed.

Implementation issues should not be created merely because an idea exists.

Prefer:

```text
evidence
   ↓
architecture/review
   ↓
decision
   ↓
issue
   ↓
implementation
```

when the work is architecture-sensitive.

### 24.1 Issue Architecture: Parent Issues and Categorical Sub-Issues

A **parent Issue** represents a meaningful feature, function, defect, or engineering objective.

When that objective contains distinct categories of work with their own scope or evidence, use GitHub **sub-issues** to organize those workstreams.

```text
                    PARENT ISSUE
              Feature / Function / Objective
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   Implementation      Audit        Benchmark
     sub-issue       sub-issue       sub-issue
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
              Documentation / Integration
                   when justified
```

The categories are **workstreams, not individual actions**.

Repeated test runs remain under the same Test/Audit sub-issue. Repeated benchmark runs remain under the same Benchmark sub-issue. Do not create a new Issue or sub-issue merely because the same category of work must be rerun, adjusted, or repeated to satisfy the parent objective.

Use a new sub-issue when the work becomes a materially different category or independently meaningful objective.

### 24.2 Audit and Benchmark Separation

Correctness/audit work and performance/benchmark work are separate categories even when they concern the same implementation.

```text
Parent Feature
├── Implementation
├── Correctness / Audit
├── Benchmark / Performance
└── Documentation / Integration, when justified
```

An audit establishes whether behavior and invariants are correct. A benchmark establishes measured performance/resource behavior. Neither substitutes for the other.

### 24.3 Issue References and Development Traceability

Every meaningful development action must reference the **Issue that owns the work**.

Use the Issue number as the authoritative reference. Do not create a second internal numbering system for comments, test runs, benchmark runs, or development updates.

When a sub-issue exists, reference the **sub-issue number** for work performed against that category. Reference the parent Issue when discussing the overall feature or objective.

Issue comments, commits, Pull Requests, benchmark reports, audit findings, and development-agent reports should reference the owning Issue number when they represent meaningful work or evidence.

### 24.4 Parent, Related, and Dependency Relationships

These relationships must not be conflated.

**Parent / sub-issue** — the child is a category required to complete the parent objective.

**Dependency** — one Issue or sub-issue must be completed before another can proceed, even if it is not conceptually a child.

```text
#28 Shared Benchmark Harness
          │
          └──── dependency ────► #33 GRU Benchmark
```

A shared infrastructure Issue may therefore be a dependency of multiple feature or benchmark Issues without becoming a sub-issue of each one.

**Related Issue** — two Issues concern the same component or evidence area, but neither is a parent/child relationship nor a prerequisite.

This distinction prevents shared infrastructure, feature work, audits, and benchmarks from being forced into one hierarchy.

### 24.5 Issue Scope and Reuse

Same objective and same work category should remain under the existing Issue/sub-issue.

Do not create a new sub-issue merely because:

- a test must be rerun;
- a benchmark must be rerun;
- a benchmark configuration changes within the same objective;
- an audit must repeat after a correction;
- additional evidence is required for the same acceptance criteria;
- implementation needs another iteration within the approved scope.

Create a new Issue/sub-issue when the work becomes materially different in objective, category, acceptance criteria, API contract, architectural scope, or independently meaningful optimization.

If a closed Issue's original objective remains incomplete, reopen it rather than creating a duplicate.

If a completed Issue is followed by a genuinely new defect or objective, create the appropriate new Issue and reference the earlier work.

### 24.6 Mandatory Issue-Driven Git Workflow

Issue-specific development must follow:

```text
GitHub Issue / Sub-Issue
  ↓
Maintainer confirms scope
  ↓
Dedicated issue branch
  ↓
Design / Planning review as required
  ↓
Implementation
  ↓
Validation / Audit / Benchmark as applicable
  ↓
One logical issue update
  ↓
Commit with issue reference (#XX)
  ↓
Push issue branch
  ↓
Pull Request
  ↓
Review + validation
  ↓
Merge into main
  ↓
Issue / sub-issue closes when appropriate
  ↓
User / maintainer deletes feature branch
```

`main` is the integration branch. Issue-specific implementation must not be performed directly on `main`.

Preferred branch naming:

```text
issue-XX-short-description
```

The branch should correspond to the Issue or sub-issue whose work is being implemented.

The user/maintainer deletes the feature branch after merge. Development agents must not delete it unless explicitly authorized.

### 24.7 One Issue Update = One Logical Commit = One Push

For Seqvex issue work, treat each authorized **issue update** as one coherent delivery unit:

```text
Issue / Sub-Issue update
        ↓
one logical commit
        ↓
one push
```

The commit must use `-m` and reference the owning Issue:

```bash
git commit -m "<type>: <concise description> — #XX"
git push origin issue-XX-short-description
```

The commit should contain only the work belonging to that Issue/Sub-Issue update.

This rule does not authorize combining unrelated work into one commit. If a proposed update spans multiple independent Issues or categories, stop and resolve the scope before committing.

### 24.8 Separate Authorization Gates

Authorization for one development action must not be inferred from another. The following are separate gates:

1. issue/sub-issue creation, reopening, or update;
2. branch creation;
3. implementation;
4. commit;
5. push;
6. Pull Request creation;
7. merge;
8. issue/sub-issue closure or other issue-state mutation;
9. branch deletion.

Implementation authorization alone does not authorize commit, push, PR creation, merge, issue mutation, or branch deletion.

### 24.9 Working-Tree Protection

Before issue-specific work and before committing:

```bash
git status
git diff
```

Preserve unrelated pre-existing changes. Do not reset, stash, clean, overwrite, discard, or commit unrelated changes merely to make issue work easier.

Before committing, stage only files belonging to the authorized Issue/Sub-Issue and inspect the staged diff. Avoid `git add .` when unrelated changes may be present.

Do not force-push or rewrite published history without explicit authorization.

### 24.10 Issue-Linked Commits and Pull Requests

Use:

```text
<type>: <concise description> — #XX
```

Recommended types include `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, and `chore`.

Use `perf` only when supported by measurement. Use `Closes #XX`, `Fixes #XX`, or `Resolves #XX` only when the Issue/Sub-Issue objective is genuinely complete.

A Pull Request should identify the Issue/Sub-Issue, parent Issue when applicable, scope, implementation, validation, architectural implications, and known limitations. Merge requires the applicable review, validation, architecture, and authorization gates.

### 24.11 Conflict-First Rule

If a conflict, contradiction, error, inconsistency, or architectural ambiguity is identified during planning, implementation, testing, audit, or review:

> **Resolve the identified conflict before extending the feature or continuing implementation.**

Do not silently work around a highlighted conflict. Defer or ignore it only when the user/maintainer explicitly instructs that it should be deferred or ignored.

### 24.12 Architecture Escalation During Issue Work

If implementation reveals that the approved Issue/Sub-Issue requires changes to state/model ownership, `StateModel`, `StreamingExecutor`, workspace or scratch ownership, execution topology, memory representation, device placement, heterogeneous memory, concurrency, scheduler, micro-batch semantics, workspace/context abstractions, public API boundaries, or another deferred architectural decision, stop at the architectural boundary and perform the appropriate CRITICAL ARCHITECTURE REVIEW before continuing.

An Issue/Sub-Issue branch does not authorize an architectural commitment merely because the implementation appears to require it.

### 24.14 Parent Feature Branch and Sub-Issue Commit Traceability

When a parent Issue contains multiple native sub-issues, the parent objective should normally use **one feature branch**.

```text
Parent Issue
    │
    ├── Native Sub-Issue A
    ├── Native Sub-Issue B
    └── Native Sub-Issue C
             │
             ▼
      Parent feature branch
             │
             ├── commit → #A
             ├── commit → #A
             ├── commit → #B
             └── commit → #C
```

The branch represents the parent objective. Each commit should reference the specific sub-issue whose work it primarily addresses:

```bash
git commit -m "<type>: <concise description> — #XX"
```

For example:

```bash
git commit -m "perf: implement reusable GRU workspace — #33"
git commit -m "test: validate workspace allocation behavior — #33"
git commit -m "feat: establish bounded GRU micro-batch execution — #34"
```

A commit may reference multiple Issues/sub-issues when one logical change genuinely spans them. Do not manufacture separate commits solely to create separate Issue references.

The parent feature branch therefore provides parent-level implementation context, while commit references provide sub-issue-level traceability.

### 24.15 Parent Completion and Sub-Issue Closure

Sub-issues represent workstreams required by the parent objective. Completing all sub-issues is strong evidence that the parent work is approaching completion, but it does not automatically close the parent.

After the relevant sub-issues are completed:

1. verify the parent's own objective;
2. verify the parent's acceptance criteria;
3. verify applicable validation, review, and integration requirements;
4. only then close the parent Issue.

This prevents the parent from being treated as complete merely because its child workstreams have been closed.

### 24.16 Native Sub-Issue Relationship

A native GitHub parent/sub-issue relationship is distinct from a textual reference such as `Parent #XX` in an Issue body.

Before treating an Issue as a child:

1. inspect the proposed parent;
2. verify that the candidate is genuinely a workstream/category required by the parent;
3. establish the native GitHub sub-issue relationship;
4. preserve the Issue's scope and history.

An existing standalone Issue may be attached to the parent through GitHub's native **Sub-issues → Add existing issue** mechanism. When this is possible, prefer attaching the existing Issue rather than creating another replacement Issue.

### 24.17 Duplicate Replacement and Historical Traceability

When an existing standalone Issue is determined to be a child by nature:

1. verify the supposed parent;
2. create or attach the replacement as a native sub-issue;
3. transfer the necessary scope, acceptance criteria, dependencies, evidence, and relevant history;
4. explicitly identify the replacement relationship;
5. add the `duplicate` label to the original Issue;
6. add a closing comment identifying the replacement;
7. close the original Issue as a duplicate.

The original Issue should not be closed before the replacement native sub-issue exists.

Historical duplicate Issues remain closed for traceability. Do not bulk-close Issues merely because their work appears related.


### 24.13 Issue Planning Approval

Before creating, reopening, or substantially updating an Issue or sub-issue as part of planned work, the proposed action should be presented for review first.

The proposal should clearly state:

1. **Issue action** — create, reopen, update, or continue an existing Issue/sub-issue.
2. **Issue** — the Issue number and title, when an existing Issue is involved.
3. **Parent relationship** — parent Issue, when applicable.
4. **Objective** — what engineering outcome the work addresses.
5. **Category** — implementation, audit, benchmark, documentation/integration, or another meaningful work category.
6. **Planned change** — what will be investigated, modified, tested, or benchmarked.
7. **Classification** — proposed Issue Type, Area label, Nature label(s), and Milestone where applicable.
8. **Relationship** — why this belongs to the selected Issue/sub-issue rather than a different or new Issue.

Do not create, reopen, or substantially update the Issue until the user has given explicit approval in the chat.

This approval applies especially when planning or carrying out feature implementation, new algorithms, modifications, optimization audits, debugging, benchmarking, or architectural changes.

The purpose is to keep Issue scope deliberate and prevent duplicate, overly broad, or prematurely created Issues.

# 25. Deferred Decisions

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

A blank or undecided area is not an invitation to invent an answer.

---

# 26. Dependency and Abstraction Discipline

Every abstraction introduces:

- conceptual complexity;
- maintenance cost;
- API surface;
- compile-time consequences;
- performance implications;
- future compatibility constraints.

> **Prefer the smallest abstraction that accurately represents a demonstrated recurring requirement.**

Do not abstract for hypothetical reuse.

Do not optimize for theoretical elegance at the expense of understanding.

---

# 27. Documentation Discipline

Documentation should record:

- what was decided;
- why;
- supporting evidence;
- uncertainty;
- material rejected alternatives;
- what would cause the decision to be revisited.

Use labels where appropriate:

- **Current**
- **Tentative**
- **Deferred**
- **Experimental**
- **Measured**
- **Not yet demonstrated**

Do not document speculation as fact.

When a public API or implementation topology is provisional, say so explicitly.

---

# 28. Integration Discipline

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
Update documentation if the decision changed
   ↓
Integrate
```

A change is not complete merely because it compiles.

For architecture-sensitive changes, the integration record should identify:

- the evidence;
- the decision;
- the affected issue;
- the remaining uncertainty.

---

# 29. Current Development Priority

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

---

# 30. What Success Looks Like

Early success is demonstrated by:

- clear semantics;
- understandable Rust;
- explicit invariants;
- strong tests;
- controlled failure behavior;
- measured performance;
- justified abstractions;
- useful modular boundaries;
- preserved hardware flexibility;
- credible streaming execution;
- a path toward constrained execution.

It is not measured by:

- number of crates;
- number of algorithms;
- number of abstractions;
- generated code volume;
- dependency count;
- premature GPU support;
- theoretical performance claims.

---

# 31. Guiding Rules

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
12. **Keep functional responsibilities locally owned until shared responsibility is demonstrated.**
13. **Use Rust patterns because they solve problems, not because they are fashionable.**
14. **Measure performance before optimizing.**
15. **Treat hardware-specific optimization as evidence-driven.**
16. **Do not turn future goals into present guarantees.**
17. **Do not silently resolve deferred architectural decisions.**
18. **Keep AI-generated changes small enough to understand and review.**
19. **Use CRITICAL ARCHITECTURE REVIEW before changes that could constrain hardware-aware execution, heterogeneous memory/device placement, accelerator offload, or Rust-native low-level optimization.**
20. **Prefer experiments when architecture is uncertain.**
21. **Let implementation evidence shape the architecture.**

---

# 32. Relationship to Other Documentation

`DEVELOPMENT.md` is the **canonical development contract**.

| Document | Purpose |
|---|---|
| `README.md` | Project identity, vision, scope, and public orientation |
| `ARCHITECTURE.md` | Architectural reasoning, system structure, and design decisions |
| `DEVELOPMENT.md` | Development rules for humans and development agents |
| `ROADMAP.md` | Current development direction and phase planning |
| `FAILURE_AND_RECOVERY.md` | Failure paths, state integrity, and recovery principles |
| `CONTRIBUTING.md` | Contributor participation, classification taxonomy, and contribution process |
| `FEATURE_DEVELOPMENT.md` | Operational feature-development runbook (lifecycle, decomposition, traceability, templates) that implements this contract |
| AI-assisted development guidance | Tool-neutral guidance for development agents and AI-assisted repository work |

If another document conflicts with this development contract, determine whether the conflict is an outdated document, a deliberate architectural change, or a missing clarification.

Do not silently choose one interpretation.

---

# Final Principle

> **Seqvex should be built by learning the problem deeply, implementing the smallest understandable mechanism, measuring its behavior, and allowing evidence—not speculation—to determine the architecture.**

# 33. Algorithm Development Lifecycle

Each substantive ML/RL algorithm should progress through a complete lifecycle:

```text
Research / mathematical specification
  ↓
Reference implementation
  ↓
Independent correctness audit
  ↓
Sequential / non-IID audit
  ↓
Numerical / statistical audit
  ↓
Performance measurement
  ↓
Local optimization when justified
  ↓
Documentation / integration
```

Optimization is not opened merely because an implementation exists. It requires correctness evidence, a measured bottleneck, and a local optimization that does not silently establish a deferred architectural decision.

## 33.1 Five-Algorithm Architecture Gate

After the fifth algorithm completes its lifecycle, perform a **post-five Pre-Hardware Optimization Audit** across the algorithms and shared execution infrastructure.

This is distinct from an earlier or partial audit performed before the five-algorithm set is complete.

The post-five audit should examine:

- execution semantics;
- state and transition semantics;
- failure atomicity;
- ordering and causality where applicable;
- numerical behavior;
- statistical assumptions;
- long-horizon behavior;
- reset and stream isolation;
- performance baselines;
- allocation/resource behavior;
- repeated implementation patterns;
- evidence for shared abstractions;
- hardware-aware execution implications.

Only after this evidence is reviewed should Seqvex settle the execution architecture required for the next optimization phase.

## 33.2 Common Audit Core and Algorithm-Specific Audit

Seqvex may use a **Common Audit Core + Algorithm-Specific Audit** model.

Common categories may include reference correctness, validation, state/transition behavior, failure atomicity, reset/isolation, ordering/causality where applicable, numerical stability, statistical assumptions, long-horizon behavior, performance baseline, allocation/resource behavior, and API/integration behavior.

A common category does not imply an identical test for every algorithm. Do not implement a generic audit framework merely to formalize this concept before repeated evidence demonstrates that it is useful.

## 33.3 Optimization Gate

A local optimization should satisfy all of the following:

1. correctness has already been established;
2. the bottleneck is measured;
3. the optimization is local and understandable;
4. mathematical and sequential semantics are preserved;
5. the change does not silently settle a deferred architecture decision;
6. the resulting behavior is benchmarked again.

Allocation reduction is not, by itself, evidence of lower latency or higher throughput.

## 33.4 Evidence Classification

Development decisions should distinguish:

```text
FACT
MEASURED EVIDENCE
INFERENCE
ASSUMPTION
PROVISIONAL DECISION
ARCHITECTURAL DECISION
OPEN QUESTION
```

Do not present an inference as a measurement, an assumption as a fact, or a provisional decision as a settled architectural decision.

The repository-evidence boundary — verifying what actually exists before planning or implementing, and distinguishing `FACT` / `APPROVED DESIGN` / `PLANNED` / `HISTORICAL` / `INFERENCE` / `UNKNOWN` — is operationalized in [`FEATURE_DEVELOPMENT.md`](../FEATURE_DEVELOPMENT.md) §8.
