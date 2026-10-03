# Seqvex ML Vertical Slices

> **This document describes the current vertical-slice development sequence and
> algorithm matrix. It is a development direction, not a fixed architecture or
> delivery commitment.**

The roadmap (`docs/ROADMAP.md`) identifies *what Seqvex is trying to learn*.
This document records *how each ML algorithm is brought into the framework* and
*what evidence each slice is expected to produce*.

It exists because the algorithm matrix is otherwise only visible on GitHub, and
each slice is an experiment about ownership, state, and execution strategy — not
merely an algorithm implementation.

## How to develop a slice

Every algorithm follows the same sequence. Do not skip a stage, and do not begin
optimizing before the reference behavior is correct and measured.

```text
reference implementation
        ↓
streaming execution
        ↓
bounded micro-batch
        ↓
equivalence tests
        ↓
benchmark
        ↓
profile
        ↓
targeted optimization
```

1. **Reference implementation.** The simplest readable version, written for
   mathematical transparency. It is the semantic oracle for every later stage.
2. **Streaming execution.** Per-observation execution that preserves the
   framework's ordering, state, and failure semantics (`ARCHITECTURE.md` §5–6).
3. **Bounded micro-batch.** A bounded aggregation strategy, only where the
   algorithm's semantics permit it. It must not change observable results.
4. **Equivalence tests.** Prove the streaming and micro-batch paths agree with
   the reference, and that failures preserve committed state.
5. **Benchmark.** Measure honestly, in release mode, with repeated runs. Do not
   assume batching or a lower-level implementation is faster.
6. **Profile.** Only after correctness and the baseline benchmark.
7. **Targeted optimization.** Only if the profile identifies a material
   bottleneck. If no optimization is justified, leave the straightforward
   implementation in place and record that result.

The reference implementation must remain present and tested. A production or
optimized path must never replace or bypass the oracle (`ARCHITECTURE.md` §22).

## Algorithm matrix

Each slice is one engineering objective. The linked Issue is the unit of scope;
this matrix does not create or close Issues.

| Algorithm | Issue | Goal | State profile | Workspace / scratch | Execution strategy | Status |
|---|---|---|---|---|---|---|
| **Linear Regression** | #23 | Closed-form prediction `ŷ = w · x + b`; prediction only | None during prediction (immutable weights) | None | Streaming per-observation; independent observations may micro-batch | Implemented (reference/streaming/micro-batch) |
| **Decision Tree** | #24 | Read-only traversal from a trained tree | None (immutable at inference) | None, or a small traversal path | Streaming per-observation traversal; independent observations may micro-batch | Implemented (reference/streaming/micro-batch; regression only, classification deferred) |
| **K-Nearest Neighbors** | #25 | Brute-force distance + neighbor selection | Stored reference observations + `k` | Per-query distance scratch `O(N)`; neighbor set `O(k)` | Streaming per-query; independent queries may micro-batch | Implemented (reference/streaming/micro-batch; regression only, classification deferred) |
| **GRU bounded micro-batch** | #34 | Bounded, ordered micro-batch over existing GRU semantics | Hidden state `h` per stream | Reference path only; do not touch the executor-owned workspace | Ordered fold; **not** independent; unchanged failure semantics | Implemented (reference streaming + bounded fold); measurement pending |
| **Recursive Least Squares** | #26 | Ordered online adaptation of `(w, P)` | Adaptive `w` and covariance `P` per stream | `d`-vectors and `d×d` rank-1 update scratch | Ordered, state-dependent; **not** independent | Implemented (reference/streaming/bounded fold) |

Two families emerge from the matrix and are the point of the exercise:

- **Ordered / stateful** — GRU and RLS. A micro-batch remains an ordered fold;
  observations are not independent and must not be parallelized as though they
  were.
- **Independent at inference** — Linear Regression prediction, Decision Tree
  prediction, and KNN query. Each observation is computed from immutable model
  data, so a micro-batch may aggregate or vectorize across observations.

This spread is why these five algorithms were selected: together they supply
evidence about state, scratch, and ownership before any of those choices is
frozen.

## Micro-batch semantics

Bounded micro-batching is an optimization/capability, not the semantic
foundation (`ARCHITECTURE.md` §6.3). Across all slices:

- A micro-batch must produce the **same committed state and outputs** as the
  equivalent repeated single-observation streaming sequence, or the difference
  must be an explicitly documented and validated algorithmic change.
- Micro-batching must **not reorder observations** unless the algorithm is
  demonstrably order-independent and the resulting computation is unchanged.
- Whether a failed observation **stops** the batch or is skipped while the fold
  continues from the last valid state is a per-algorithm policy decision. It
  must be explicit, not assumed.
- A micro-batch is not automatically parallel. Independent algorithms may
  vectorize only if the observable result is preserved; stateful algorithms stay
  ordered.

`process_batch` (`src/foundation/state`) is the **reference ordered fold**
(unbounded, stop-on-first-failure). It is not a bounded micro-batch executor and
must not be re-documented as one. There is currently no generic micro-batch
executor, and none should be introduced until the slices below demonstrate a
recurring need.

## Ownership hypothesis (evidence, not decision)

The slices are expected to provide evidence about where mutable resources
belong. The current hypothesis, to be confirmed across slices rather than
assumed, is:

```text
trained / immutable parameters   → shared, possibly device-resident
per-stream state                 → owned by the execution session
per-stream scratch / workspace   → owned by the execution session
```

The GRU slice initially placed its reusable scratch in the model to reach
allocation-free stepping, forcing `&mut M` on the generic executor. That
arrangement was resolved: the workspace now belongs to the algorithm-local
`GruExecutor` (an execution context), `Gru` holds only immutable parameters and
configuration, and the generic executor borrows the model immutably (`&M`).
One immutable model may therefore drive several independent execution contexts,
each owning its own State and (for GRU) its own private workspace. Vertical
slices must not reintroduce model-owned scratch.

RLS is the first **stateful** slice to confirm the hypothesis rather than
contradict it: its adaptive `(w, P)` is per-stream execution state, the model
stays immutable and shareable, and no model-owned scratch is needed.

KNN adds the observation-backed case: its reference set is immutable, shareable
model data, and its distance/selection scratch stays local and transient, so it
also confirms the hypothesis rather than extending the GRU's model-owned
workspace arrangement.

## Critical architecture review requirement

Any work in a slice that could materially constrain, complicate, or prematurely
freeze hardware-aware execution, heterogeneous memory/device placement, device
residency, per-stream versus model-owned state/scratch, synchronization, or
low-level allocation/layout control requires **CRITICAL ARCHITECTURE REVIEW
before implementation** (`docs/DEVELOPMENT.md` §3).

Active examples are none at present: the former `StreamingExecutor<'m, M>`
holding `&'m mut M` and the model-owned GRU workspace were both removed by the
execution-ownership refactor. Treat any future model-owned state/scratch or
exclusive model borrow as a trigger for this review.

Stop and escalate rather than resolving such a question locally. In particular,
do not add a generic workspace, micro-batch executor, scheduler, or
storage/tensor abstraction to serve one slice.

## Measured evidence

### Linear Regression (#23)

Reference and streaming prediction are allocation-free (`0.000` allocations and
`0` bytes per observation) and consume only immutable weights through `&self` —
the intended control case for the ownership hypothesis. The bounded micro-batch
adds exactly one output allocation per batch (amortized `1/batch_size` per
observation) and is **not** faster than single-observation prediction; at the
largest tested feature dimension it is materially slower (median +4.8 ns/obs,
≈ 16× the single-observation IQR). No optimization was justified. The
micro-batch is retained as a semantically-valid capability and as a pattern for
future independent algorithms, not as a performance claim.

### Decision Tree (#24)

Read-only regression traversal; classification and training remain deferred.
Reference and streaming inference allocate nothing (`0.000` allocations, `0`
bytes per observation). The measured streaming median differed from the direct
reference by +0.6 ns/obs at a 127-node tree (15.9 vs 15.3 ns/obs) and by
-1.7 ns/obs at the largest tested 2047-node tree (106.7 vs 108.4 ns/obs); the
benchmark applies no materiality test, so these are reported measured
differences, not a statistical-equivalence claim. The bounded micro-batch adds
exactly one output `Vec` per batch (amortized `1/batch_size` per observation) and
is not faster than single-observation prediction; a first-element failure makes
the entire `predict_batch` call return `Err`, with no partial output vector
returned. Construction builds one growing node `Vec` (11, 14, and 16 heap
allocations at the measured tree sizes), distinct from the allocation-free
per-observation inference and the per-batch output allocation. No optimization
was implemented; there is no profile-identified bottleneck.

### K-Nearest Neighbors (#25)

Read-only regression inference over stored references; classification and
training remain out of scope. Unlike LR/DT prediction, the current **full-sort reference implementation** is
not allocation-free: each query materialises all `N` candidates in one
transient `O(N)` buffer (16 bytes per reference, so `bytes/query = 16·N`), the
scratch this slice's profile anticipates; a distance-scan-only control allocates
nothing. Streaming matches the direct reference (same `predict`). The scan
control localises the baseline cost: at `N = 262144, d = 8` the full-sort
selection dominates (9.80 ms/query vs a 1.48 ms scan), while at
`N = 8192, d = 256` distance computation dominates (1.14 ms vs 0.97 ms).
Latency is flat across `k ∈ {1, 8, 32, N}` at `N = 8192, d = 32` (~230 µs)
because selection is the sort, not `k`. Construction is `N + 1` allocations.
The bounded micro-batch preserves input order but is not allocation-free either:
it adds one output `Vec` that grows through reallocations, so the measured
allocator calls per observation are B-dependent — 2.000, 1.250, 1.125, and 1.047
for `B = 1, 8, 32, 128` (one output allocation at `B = 1`, with +1, +2, +4, +6
further output reallocations at `B = 8, 32, 128`); it is not faster than
single-query prediction (33.7 µs vs 49.4 µs at `B = 128`). This allocation
behaviour is a property of the current reference implementation, not an intrinsic
requirement of all KNN implementations. No optimization was implemented;
full-sort selection is the obvious future target but requires the existing
correctness evidence to remain and a measured decision, per the slice lifecycle.

### Recursive Least Squares (#26)

The first stateful model whose parameters change every observation. `Rls` holds
only immutable configuration (`D`, `λ`, `δ`) and implements `StateModel` with a
`&self` contract; the adaptive `(w, P)` lives entirely in `StateModel::State`,
owned by the execution session. No model-owned workspace is used.

Unlike linear regression, the reference is **not** allocation-free: the
value-returning contract produces a new `D×D` candidate `P'` every observation,
so the steady cost is 5 allocations/observation dominated by `P'`, and
bytes/observation scale with `D²` (`384` at `D = 8`, `266240` at `D = 256`).
Streaming and the bounded foundation fold are **not** faster than the direct
reference; the fold additionally costs one extra allocation/observation because
it consumes owned observations. The `O(D²)` candidate allocation is the
RLS-specific evidence, reported rather than optimized — a double-buffered `P`
would raise an unresolved workspace-ownership question.

RLS also confirms the ownership rule: adaptive state belongs to the stream, not
the model. One immutable `&Rls` can now drive independent per-stream states both
through the foundation `process_one`/`process_stream` functions and through the
generic `StreamingExecutor`, which borrows the model immutably (`&M`) and owns
the per-stream State. The former `&mut M` limitation was removed by the
execution-ownership refactor.

## Matrix growth rule

> **The matrix is not expanded without evidence.**

Do not add algorithms, labels, traits, or abstractions because they seem useful.
A new slice is justified by an actual Seqvex use case or by evidence from an
existing slice. Issue scope is the boundary; the absence of a convenient
abstraction is not a reason to create one.

## Relationship to other documentation

- `docs/ROADMAP.md` — phase-level direction.
- `docs/DEVELOPMENT.md` — the canonical development contract, including the
  CRITICAL ARCHITECTURE REVIEW guard (§3) and issue discipline (§24).
- `ARCHITECTURE.md` — system-wide architectural reasoning.
- `src/foundation/state/README.md` — state-transition and failure semantics.
- `src/execution/README.md` — execution modes and their semantics.
