# Seqvex Feature Development Runbook

> **Status:** Operational runbook
> **Audience:** Human contributors and AI-assisted development agents
> **Scope:** How to take a feature, algorithm, or function from idea to merged change
> **Nature:** This is a procedural runbook. It does **not** define new rules. It links to the authoritative documents and tells you, in order, what to do.

`FEATURE_DEVELOPMENT.md` answers one question:

> *"I want to add or modify a feature, model, or function in Seqvex. What exactly do I do?"*

It is deliberately operational. Architecture, engineering rules, and contribution taxonomy live in the authoritative documents below. When this runbook and an authoritative document disagree, the authoritative document wins — report the conflict rather than choosing silently.

---

## 1. Required reading

Read the authoritative source for the question you have; do not rely on this runbook as a substitute.

| Question | Authoritative document |
|---|---|
| What is Seqvex's architecture, scope, and execution model? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| What are the engineering rules and gates? | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| How are issues classified (type, area, nature, milestone, status)? | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| What algorithms exist and what evidence do they have? | [`docs/ML_VERTICAL_SLICES.md`](docs/ML_VERTICAL_SLICES.md) |
| What are the state/failure rules? | [`docs/FAILURE_AND_RECOVERY.md`](docs/FAILURE_AND_RECOVERY.md) |
| What does the execution layer guarantee? | [`src/execution/README.md`](src/execution/README.md) |
| What numerical contracts exist? | [`src/foundation/numerical/README.md`](src/foundation/numerical/README.md) |
| What is the CI baseline? | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

---

## 2. Feature characterization (metadata, not directories)

Before designing, characterize the work along five independent axes. These are **metadata and behavior dimensions**; they must not become directory levels.

| Axis | Values |
|---|---|
| **T** — Temporal dependence | `Pointwise` / `Windowed` / `Recurrent` |
| **R** — Data regime | `iid` / `non-iid` / `serially-correlated` / `time-series` / `distribution-shift` |
| **L** — Learning semantics | `batch-fit` / `incremental` / `online` / `adaptive` / `frozen` / `recurrent-transition` |
| **X** — Execution semantics | `single` / `streaming` / `micro-batch` / `batch` |
| **V** — Validation methodology | `iid-cv` / `temporal-split` / `walk-forward` / `purged-embargoed-cv` / `prequential` |

Governing rule:

> **Mathematical sequence independence does not imply statistical IID validity.**

Do not collapse these into a single "supports non-IID" flag. State exactly what changes: sampling, labelling, weighting/loss, features/representation, validation, learning regime, or drift handling.

### Execution-delivery modes (X)

| Mode | Definition |
|---|---|
| **Single observation** | One observation is processed per invocation. |
| **Streaming** | Observations arrive incrementally over time and are processed individually or through a streaming path. |
| **Micro-batch** | A bounded group of observations processed together as one execution unit; the boundary may be count, time, availability, or another explicitly documented policy. |
| **Batch** | A finite collection of observations processed as one execution unit. |

These modes are **independent** of learning regime, temporal dependence, data regime, and concurrency:

```text
execution delivery mode
≠ learning regime
≠ temporal dependence
≠ IID / non-IID property
≠ concurrency model
```

A single observation is not online learning, temporal dependence, or an IID/non-IID declaration. Streaming is an execution/data-delivery characteristic, not a claim that the algorithm learns online. Micro-batch boundaries are a per-algorithm policy that must be explicitly documented; this runbook does **not** prescribe a universal micro-batch size, time window, queue length, or flush policy. These are definitions only: they introduce **no** executor, public trait, state type, or batching API.

### State, stream, concurrency, and temporal dependence

- **State** is information retained across execution events. It may belong to model parameters, learned statistics, reference observations, optimizer state, recurrent state, execution/workspace state, or another explicitly defined ownership domain. Not every model has temporal state.
- **Stream** is an ordered or continuously delivered sequence of observations/events. A stream is **not itself state**; state may be associated with processing a stream, but the two are distinct.
- **Concurrency** is multiple activities in progress during overlapping execution. A **thread** is one possible implementation mechanism for concurrency; concurrency does not imply multithreading.
- **Temporal dependence** describes whether the mathematical/data-generating process depends on temporal ordering or prior observations. It must **not** be inferred merely because data arrives through a stream.

A classical algorithm can be executed on streaming data while remaining mathematically sequence-agnostic when its mathematical formulation does not depend on temporal ordering.

---

## 3. Issue model

Parent/child/task rules, scope/reuse, and the decision tree are authoritative in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24. This section is the quick operational summary.

```text
Parent Feature — algorithm / feature / meaningful objective
├── Implementation                        (conditional child)
│   ├── independently meaningful implementation concern A
│   ├── independently meaningful implementation concern B
│   ├── streaming / micro-batching        [if applicable]
│   ├── RNG / randomness                  [if applicable]
│   └── other independently meaningful implementation concerns
├── Mathematical correctness              (conditional child)
├── Statistical / non-IID validation      (conditional child)
├── Benchmark / allocation / latency audit   (usually a child)
├── Integration / API review              (conditional child)
├── Architecture / planning               (conditional child — see below)
└── Documentation                         (completion requirement)
```

**A child issue requires all three:**

1. independently meaningful evidence;
2. a distinct verification/review category;
3. the ability to fail or block the parent independently.

Otherwise it is a **task** (a checklist item in the parent/child body).

**Do not create a child issue for:** an individual test; a single benchmark run; a documentation correction; a rerun; formatting; a component that is merely "step N"; a file or function with no independent evidence.

### Scope and reuse decision tree

```text
Does an owning Issue already exist?
        │
   ┌────┴────┐
  Yes        No
   │          │
Same       Is this a
objective? meaningful objective?
   │          │
 ┌─┴─┐      Yes → CREATE new Issue
Yes  No
 │    └─ New Issue
 └─ Is this a new work category?
         │
    ┌────┴────┐
   No         Yes
    │          └─ Create a sub-issue
    └─ Continue existing category

If the Issue is closed but the same objective remains incomplete → REOPEN.
```

An optimization is **not** automatically a new Issue. Keep it under the existing Issue/sub-issue when it optimizes the same identified bottleneck or objective; track it separately only when it introduces a materially different objective, API contract, acceptance criteria, or architectural change.

KNN #25 is a **retrospective worked example**, not a universal template. Derive the decomposition per algorithm.

### Architecture / planning category

Architecture/planning work receives its own traceability **only when it is an independently meaningful engineering concern**, for example:

- an architectural decision;
- abstraction design;
- repository-wide restructuring;
- API contract design;
- execution architecture;
- ownership model;
- workspace design;
- common-infrastructure design.

Do not create an architecture issue for every ordinary implementation decision, and do not turn individual test cases into project-management objects. Traceability corresponds to independently meaningful work only.

### Repository organization (governing direction)

Algorithms should be substantially self-contained where practical: a model-family/algorithm implementation may locally contain its implementation, tests, benchmarks, audit material, model-specific documentation, and examples where appropriate. Common infrastructure remains centralized only when it is genuinely reusable.

This is a governing direction for the future restructuring stage. **It is not performed here:** this runbook does not move algorithms, rename modules, add compatibility aliases, or restructure directories.

---

## 4. Roles and boundaries

Roles are defined by responsibility, not by actor type. The same boundaries apply to human-only, AI-assisted, and mixed teams. Role ≠ actor; one person may hold several roles, but the **Code Reviewer must be independent from the Coder for non-trivial work**.

| Role | Answers | Owns | Must not |
|---|---|---|---|
| **Architect** | What/why, contract, evidence | Design, alternatives, workstream boundaries, pre-/post-implementation audit, decision records | Implement for convenience; silently settle deferred decisions |
| **Planner** | How it is organized and executed | Decomposition, dependencies, sequencing, file/module map, acceptance criteria | Redesign architecture; invent scope |
| **Coder** | Can it be built correctly and reproducibly? | Implementation, tests, benchmarks, documentation, issue-referenced commits | Expand scope; redesign; commit/push without authorization |
| **Code Reviewer** | Does it satisfy the approved contract and evidence requirements? | Independent review, evidence verification, findings | Silently fix; review own non-trivial work |
| **Maintainer** | Is it authorized? | Scope, architecture, branch, commit, push, PR/merge, issue closure | Delegate gates implicitly; infer one gate from another |

The full gate rules are authoritative in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24.8, §24.13, §24.15 and §3 (CRITICAL ARCHITECTURE REVIEW).

### Architect abstraction inspection

Before proposing an abstraction, the Architect/Planner must inspect **both** the actual repository implementation **and** the actual/planned model families, then determine the **minimum** abstraction that accommodates the model families that genuinely require it. Do not derive architecture from one algorithm in isolation, and do not generalize prematurely.

Identify explicitly:

- common behavior;
- model-specific behavior;
- the required abstraction;
- unnecessary abstraction;
- unresolved evidence.

Architecture decisions remain evidence-driven.

---

## 5. Lifecycle

Each step names its actor, precondition, and evidence. Gate rules are authoritative in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24.

| # | Step | Actor | Precondition | Evidence |
|---|---|---|---|---|
| 1 | Understand existing architecture | All | — | — |
| 2 | Identify or update the GitHub issue | Architect/Maintainer | evidence exists | issue record |
| 3 | Decide parent vs child vs task | Planner | issue exists | decomposition |
| 4 | Architect technical design (contract, alternatives, workstreams, risks, deferred) | Architect | parent issue open | design record in issue |
| 5 | Architect pre-implementation audit / CRITICAL ARCHITECTURE REVIEW | Architect | review triggers | review outcome |
| 6 | Planner decomposition and sequencing | Planner | design approved | child issues/tasks, deps, repo map |
| 7 | Maintainer authorization | Maintainer | plan presented | explicit scope + branch authorization |
| 8 | Checkout / sync / base verification | Coder | authorized | clean base recorded |
| 9 | Create dedicated issue branch | Coder | base verified | `issue-XX-short-description` |
| 10 | Implement approved scope | Coder | branch created | scoped edits |
| 11 | Test | Coder | implementation | applicable tests, errors and edge cases; model deliveries also require data ingestion → actual training → validated inference, mathematics and quality evidence |
| 12 | Benchmark (conditional) | Coder | tests pass | measured evidence |
| 13 | Update documentation | Coder | implementation | model README / rustdoc / evidence entry |
| 14 | Code Review | Code Reviewer | implementation + evidence | findings recorded |
| 15 | Corrections | Coder | findings | resolved or deferral recorded |
| 16 | Post-implementation audit (conditional) | Architect | trigger | audit record |
| 17 | Final verification | Coder + CI | all gates | fmt/clippy/test; diff inspected |
| 18 | Commit | Coder | commit authorization | issue-referenced commit |
| 19 | Push | Coder | push authorization | branch pushed |
| 20 | Open Pull Request | Coder | PR authorization | PR with traceability block |
| 21 | Review + merge | Reviewer/Maintainer | all gates pass | merge |
| 22 | Update and close issue | Maintainer | parent completion rule | closure |
| 23 | Delete branch | Maintainer | post-merge | branch removed |

Small, clearly scoped changes may compress steps 2–7 with maintainer acknowledgment; checkout (8/9) and verification (17) still apply. See [`CONTRIBUTING.md`](CONTRIBUTING.md) "Before Contributing".

---

## 6. Architecture gate

Before designing or changing behavior, check the CRITICAL ARCHITECTURE REVIEW triggers in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §3. If the work could constrain hardware-aware execution, per-stream vs model-owned state, device placement, ownership, or low-level allocation, run the review **before** implementation.

Deferred decisions must not be silently settled ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §25). If implementation forces a deferred decision, escalate to Architect.

### Workspace / ownership evidence gate

A reusable **Workspace** is an evidence-gated architectural concept, not an implementation requirement. This runbook does not implement a Workspace type, change `StateModel`, or redesign executor ownership.

Before standardizing a reusable workspace abstraction, evidence should establish its relevant benefits and constraints, including:

- ownership;
- lifetime;
- reuse;
- allocation behavior;
- memory footprint;
- latency;
- synchronization;
- concurrency;
- cache/locality behavior;
- API usability;
- model-family applicability.

> If even one model receives significant measurable benefit from an abstraction, that is sufficient reason to retain or investigate the concept. One model benefiting does **not** automatically justify making the abstraction mandatory across the whole project.

---

## 7. Research → implementation gate

Research findings do **not** automatically become implementation requirements.

```text
Research / external evidence
        ↓  (interpret, do not adopt)
Architect: what failure mode does it solve? which model families need it?
        ↓
Architectural decision: part of the algorithm, an optional variant, or infrastructure?
        ↓
Planner decomposition
        ↓
Implementation requirement
        ↓
Coder → Code Reviewer → tests / benchmarks / evidence
        ↓
Decision record / documentation
```

Mandatory questions before any research-derived capability enters Seqvex:

1. What specific failure mode does it solve?
2. Which model families actually require it?
3. Is it mathematically intrinsic to an algorithm, or an optional variant/methodology?
4. Do more than one real consumer need it — **or** is one model's measured benefit large enough to justify it?
5. Can the benefit be measured?
6. What are the ownership/lifetime semantics?

"Research says it may help" is never sufficient. Financial/time-series methodology must not silently become the definition of an underlying ML algorithm.

---

## 8. Checkout and baseline discipline

Before any code change:

```bash
git status --short
git branch --show-current
git log -1 --oneline
git fetch --all --prune
```

- Verify the current branch and the intended base.
- Verify a clean (or fully understood) working tree.
- Preserve unrelated pre-existing changes; never reset, stash, clean, discard, or overwrite another person's work without explicit authorization.
- If the base is stale or the tree is not clean, **stop and report**.

Full rules: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24.9.

### Repository-evidence boundary

The **current repository is the authoritative source for what actually exists.** Plans, memory, previous reports, and intended architecture establish *intent and rationale*; they are **not** proof of current repository state. Inspect the repository and Git state before planning or implementing, and classify every material conclusion:

| Classification | Meaning |
|---|---|
| `FACT` | Directly verified from the current repository or Git state (source, tests, benchmarks, `Cargo.toml`, docs, branches, commits). |
| `APPROVED DESIGN` | Established by an accepted architectural decision or explicitly approved design — intended design, not necessarily implemented. |
| `PLANNED` | Described in an approved plan but not yet implemented. |
| `HISTORICAL` | Established by a previous implementation, report, or retrospective; may be stale. |
| `INFERENCE` | Reasoned from available evidence but not directly stated by the repository. |
| `UNKNOWN` | Insufficient repository evidence to determine the answer. |

Rules:

- Never silently convert an `UNKNOWN`, `PLANNED`, or `INFERENCE` into a `FACT`.
- Never present memory as `FACT`; a plan as implementation; an inference as repository state; or historical behavior as current behavior. When uncertain, use `UNKNOWN`.
- Never invent a directory, API, abstraction, dependency, test, benchmark, or rule the repository does not contain.
- If repository evidence is insufficient, say so and identify what evidence is missing.

Material contradictions involving **any** of the following must be surfaced before affected work proceeds: repository implementation; architecture; API semantics; mathematical derivation; statistical assumptions; non-IID/time-series assumptions; tests; benchmark evidence; documentation; plans; previous reports; remembered/projected design intent. If two credible sources conflict, do not silently choose one — apply the conflict-first rule in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24.11.

> **If the actual repository contradicts a plan or previous report, report the discrepancy and do not silently invent a resolution.**

```text
Repository fact:
Approved plan/design:
Conflict:
Impact:
Recommended escalation:  (documentation correction / implementation correction / Architect / Planner / Maintainer)
```

Resolve the discovery separately from the step that found it unless the correction is explicitly in scope.

---

## 9. Branch governance

- Never implement directly on `main`.
- Branch naming: `issue-XX-short-description`.
- One parent objective normally uses one feature branch; commits reference the specific child/sub-issue they address.
- The user/maintainer deletes the branch after merge; a development agent must not delete it without authorization.
- Do not force-push or rewrite published history without explicit authorization.

---

## 10. Commit and issue-reference governance

```text
<type>: <concise description> — #XX
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`. Use `perf` only when supported by measurement.

- Every meaningful development commit references the owning issue (child if one exists, otherwise the parent).
- Commit granularity follows meaningful work — not one commit per function or file.
- Use `Closes #XX`, `Fixes #XX`, or `Resolves #XX` only when the issue/sub-issue objective is genuinely complete; never close the parent from a child commit.
- Push is a separate authorization and action from commit.

Operational examples:

```bash
git commit -m "<type>: <concise description> — #XX"
git push origin <branch-name>
```

Full Git governance remains authoritative in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24.6–§24.10; this runbook does not restate it.

Full rules: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §24.6, §24.7, §24.8, §24.10.

---

## 11. Testing philosophy

> Test every behavior that can materially affect correctness, safety, reproducibility, usability, or real-world performance, using the strongest practical evidence for that behavior.

Do not optimize for test count. For each workstream:

1. identify what can materially fail;
2. identify the evidence required;
3. select the strongest practical test/oracle/benchmark;
4. document what was actually tested;
5. document meaningful limitations.

Prefer independent evidence over self-consistency: a mathematical oracle, an analytical result, a reference implementation, a property/invariant, or an equivalence check between execution paths (e.g. streaming vs repeated single-observation, micro-batch vs repeated single).

A passing test does not by itself establish mathematical correctness, statistical validity, experimental validity, or operational performance.

---

## 12. Mathematical, statistical, and non-IID validation

```text
mathematical contract
      ↓
implementation
      ↓
independent oracle / control
      ↓
edge cases
      ↓
regime-appropriate validation
      ↓
performance validation
      ↓
code review
      ↓
post-implementation audit
```

- **Mathematical correctness:** the implementation matches the defined mathematics (independent oracle, hand-derived cases, edge/boundary cases, deterministic ordering, invariants).
- **Statistical validity:** determined by the claims being made and the data regime — **not** merely by whether the algorithm is deterministic or stochastic. A deterministic KNN can still require temporal/non-IID validation; deterministic RLS still requires sequential evaluation.
- **Stochastic algorithms:** add reproducibility controls (seed contract, no hidden global RNG) in addition to statistical validation. Do not impose statistical validation on algorithms and regimes where it provides no useful evidence.
- **Non-IID/time-series:** state what was validated and what was not. Do not claim time-series or non-IID validity because an algorithm is online or recurrent.

Full evidence-classification vocabulary: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §33.4.

---

## 13. Benchmark and performance evidence

Benchmarking is multidimensional; allocation count alone is insufficient. Where relevant, measure:

- **Operational latency:** cold start, warm-up, steady state, p50, p95, p99.
- **Throughput:** observations/sec, batches/sec.
- **Memory:** model footprint, working memory, workspace footprint.
- **Allocation:** allocations and bytes per operation, transient allocations.
- **Scaling:** the factors that materially affect the algorithm (dimension, sample count, model size, batch size, state size, reference count, sequence length).
- **Workload regime:** IID / non-IID / serial correlation / drift, cold vs warm, streaming, micro-batch, batch.

Rules:

- Report the distribution, not only averages.
- Label each benchmark as **compute** or **regime/statistical** behavior.
- A justified cold-path slowdown is acceptable only if steady-state behavior is measured and documented.
- A temporary maintenance/adaptation cost is acceptable if it does not materially damage normal operation.
- Do not benchmark every combination blindly; benchmark what materially affects the algorithm.
- No optimization without measurement; a local measurement is not a universal guarantee. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §19 and §33.3.

Existing harness: [`benches/common/mod.rs`](benches/common/mod.rs). Decision numbers are release-only; debug runs are not evidence.

---

## 14. Documentation requirements

### Training-to-inference and failure gate

Apply the mandatory model-training and calculation-failure rules in [the development contract](docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory). Every model delivery and user demonstration must ingest data, execute its declared training/learning procedure, validate the learned result and perform inference from that result. Prediction-only slices, preset-parameter demonstrations and authored outputs do not satisfy completion.

If any error/failure case or policy is unresolved, report it to the maintainer before affected work proceeds; do not assume a default, continuation or recovery action.

The Architect defines the learning, numerical/error and recovery contracts; the Planner includes the end-to-end data/training/inference evidence and user examples. The Coder implements only the accepted design and plan. Independent review verifies that training actually occurs, outputs use learned parameters and failures are reported without fabricated/default predictions. Existing incomplete components remain incomplete until the requirements are met.

Documentation is part of the feature, not an afterthought. Each model `README.md` should eventually answer:

1. What the algorithm is.
2. What problem it solves.
3. Mathematical formulation.
4. Assumptions.
5. Supported data regimes.
6. Learning semantics.
7. Execution semantics.
8. State semantics (if applicable).
9. Streaming behavior.
10. Micro-batch behavior.
11. Validation methodology.
12. Performance characteristics.
13. Memory/allocation characteristics where relevant.
14. API usage.
15. Practical examples.
16. Limitations.
17. Edge cases.

Public APIs must be documented with rustdoc (meaning, arguments, errors, execution semantics, and a realistic example where practical). Do not duplicate authoritative project documentation; link to it. Documentation must be sufficient for a competent user without prior Seqvex knowledge.

Documentation is part of feature completion: a feature that is technically correct but unusable or unexplained is **not** fully complete. A user unfamiliar with Seqvex internals should be able to understand what the algorithm does; its mathematical basis; its assumptions; supported execution semantics; input/output expectations; state semantics; examples; API usage; limitations; validation status; and relevant performance characteristics — with an API-oriented explanation where appropriate. Do not invent API behavior that does not exist.

---

## 15. Code review and audits

The Code Reviewer independently evaluates the dimensions that are **materially applicable** to the work; a dimension that does not apply is recorded as not-applicable with a reason, not treated as automatically required.

| Dimension | What it covers |
|---|---|
| **Implementation correctness** | The code does what the contract says; errors, edge cases, failure behavior, scope, and regressions. |
| **Architectural correctness** | Approved architecture and contracts preserved; no silently settled deferred decision; no unauthorized abstraction or ownership/execution change. |
| **Mathematical correctness** | Independent oracle / hand-derived evidence; tie and ordering determinism; numerical envelope; no overfitting to the implementation. |
| **Statistical / data-regime correctness** | Applied where the claims or regime require it: non-IID/time-series validity, sampling, seed/RNG reproducibility, and what is claimed vs inferred. |
| **Operational correctness** | Real-world performance characteristics, not memory alone: latency, throughput, allocations, memory, cold-start, warm-start, steady-state, and relevant tail latency. |

Governing principle:

> Test every behavior that can materially affect correctness, safety, reproducibility, usability, or real-world performance, using the strongest practical evidence for that behavior.

Findings are recorded in the owning issue/PR; the Reviewer does not silently implement fixes. Post-implementation audit triggers include CRITICAL-review items, mathematical-contract changes, stochastic algorithms, and state/execution-semantics changes.

---

## 16. Validation matrix

| Area | Required evidence |
|---|---|
| Compilation | `cargo build --all-targets --all-features` |
| Tests | `cargo test --all-targets --all-features` |
| Formatting | `cargo fmt --all -- --check` |
| Clippy | `cargo clippy --all-targets --all-features -- -D warnings` |
| Benchmarks (if affected) | `cargo bench --no-run` at minimum; release run for decision numbers |
| Mathematical | independent oracle / hand-derived / invariant tests |
| Documentation | rustdoc + README updated; links resolve |
| Git scope | only issue-owned files changed; `git diff --check` clean |

Report only what was actually run: `FACT: passed`, `FACT: failed`, `NOT RUN`, or `NOT APPLICABLE`.

---

## 17. Deferred-work register

Use explicit status vocabulary so attractive ideas are not mistaken for requirements:

```text
IMPLEMENT NOW
CANDIDATE
DEFERRED
EXPERIMENTAL
REQUIRES EVIDENCE
```

Examples of intentionally deferred work: exact `StateModel` trait design; automatic batching policy; workspace pooling; generalized drift detection; advanced RNG architecture; device/GPU abstractions; temporal-validation/sampling utilities; Graphify integration. Record deferred items in the repository (issue/decision record/README), not only in chat.

---

## 18. Templates

### Parent issue (feature/algorithm)

```text
Objective:
Parent of: (children when created)
Mathematical/technical contract:
Alternatives considered:
Workstreams and repository map:
Risks (performance / allocation / numerical / architectural):
Deferred decisions:
Acceptance criteria:
```

### Child issue (workstream)

```text
Parent: #XX
Category: Implementation | Mathematical correctness | Statistical / non-IID validation |
          Benchmark / allocation / latency audit | Integration / API |
          Architecture / planning | Documentation
Scope:
Evidence required:
Acceptance criteria:
Dependencies:
```

### Pull request traceability block

```text
Parent feature issue: #XX
Child implementation issue(s): #AA, #BB
Audit / benchmark issue(s): #CC
Mathematical / statistical validation issue(s): #DD
Architecture / planning issue(s): #EE   (only when an independently meaningful concern exists)
Branch:
Scope:
Implementation:
Tests: (commands + result)
Mathematical validation:
Statistical / non-IID validation:
Benchmark evidence:
Latency evidence:
Allocation / memory evidence:
Documentation:
Audit / review status:
Deferred / out-of-scope work:
Merge / closure semantics: (which issues this PR actually completes)
```

Closing keywords (`Fixes #`, `Closes #`, `Resolves #`) are used **only** when this PR actually completes the referenced issue. A child PR must not close its parent feature issue. Individual test cases must not become GitHub issues merely for traceability. Architecture / planning issue(s) are recorded only when an independently meaningful architecture/planning concern exists (see §3 above); do not create one for an ordinary implementation decision.

---

## 19. Worked examples

- **KNN #25 (retrospective):** pointwise inference, independent `f64` oracle, deterministic ordering/tie-break, streaming and micro-batch equivalence, allocation and scan-vs-sort controls, scaling, documentation, review. It is an example of a well-evidenced pointwise model, **not** a mandatory template.
- **Random Forest (forward):** distinguish standard Random Forest mathematics from sampling/statistical methodology (sequential bootstrap, sample uniqueness) and from validation methodology (purged/embargoed CV). Financial variants are opt-in; they do not silently redefine the algorithm.

---

## 20. Final principle

```text
UNDERSTAND → VERIFY REPOSITORY → CHECKOUT/SYNC → BRANCH
→ IMPLEMENT APPROVED SCOPE → TEST → AUDIT → BENCHMARK → DOCUMENT
→ VERIFY DIFF → COMMIT → PUSH → PR → CODE REVIEW → MERGE
```

The goal is a change whose mathematics, correctness, execution semantics, ownership, reproducibility, statistical validity, operational latency, memory behavior, API, documentation, Git history, and issue traceability can all be independently audited from the repository.
