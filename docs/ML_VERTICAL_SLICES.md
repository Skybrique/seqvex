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
| **GRU bounded micro-batch** | #34 | Bounded, ordered micro-batch over existing GRU semantics | Hidden state `h` per stream | Reference path only; do not touch the executor-owned workspace | Ordered fold; **not** independent; unchanged failure semantics | Implemented (reference streaming + bounded fold); release bounded batch-size sweep recorded |
| **Recursive Least Squares** | #26 | Ordered online adaptation of `(w, P)` | Adaptive `w` and covariance `P` per stream | `d`-vectors and `d×d` rank-1 update scratch | Ordered, state-dependent; **not** independent | Implemented (reference/streaming + RLS-local bounded reference batching); release batch-size sweep recorded |

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

**[historical]** Reference and streaming inference allocate nothing (`0.000`
allocations, `0` bytes per observation). The measured streaming median differed
from the direct reference by +0.6 ns/obs at a 127-node tree (15.9 vs 15.3 ns/obs)
and by -1.7 ns/obs at the largest tested 2047-node tree (106.7 vs 108.4 ns/obs);
the benchmark applies no materiality test, so these are reported measured
differences, not a statistical-equivalence claim. The bounded micro-batch adds
exactly one output `Vec` per batch (amortized `1/batch_size` per observation) and
is not faster than single-observation prediction; a first-element failure makes
the entire `predict_batch` call return `Err`, with no partial output vector
returned. Construction builds one growing node `Vec` (11, 14, and 16 heap
allocations at the measured tree sizes), distinct from the allocation-free
per-observation inference and the per-batch output allocation. No optimization
was implemented.

**[new run]** A release batch-size sweep at the **associated inspected revision**
`b63136f5` (`cargo bench --bench decision_tree -j 2`; `os=linux arch=x86_64
profile=release parallelism=20`; 20 normalized measurement windows; single
machine, 12th Gen Intel Core i7-12700H, 20 logical CPUs). The actual run-time
revision, compiler toolchain, and machine-contention status are **unverified**
(the supplied output contains no revision or toolchain evidence and no run-time
contention evidence). It recorded all 16 `B ∈ {1, 8, 32, 128}` rows across the
four tree configurations. The full table, together with the measured,
harness-derived, and calculated quantities separated, is in
`src/models/classic/README.md` ("Release batch-size sweep"). The fixture repeats
one deterministic observation, so it measures **repeated-input** behavior, not
input-distribution performance. Under the harness materiality rule
(`difference > 2·max(IQR)` AND `difference > 5%·max(median)`): micro-batch is
materially slower than the direct reference only at 8 features (+5.9 ns/obs);
streaming is not materially different at 32, 128, or 256 features, and at 8
features the materiality classification is unresolved at displayed precision
because the reported difference equals the reported `2·max(IQR)` threshold;
within the repeated-input fixture, per-observation cost at `B = 128` is
materially lower than at `B = 1` in all four configurations (≈ 20.5–22.5 ns/obs),
with `allocs/obs` and allocator `bytes/obs` falling from `1.000`/`16.0` to
`0.047`/`7.9`. The run does not decompose traversal from per-batch output
handling, so the cause of the per-observation reduction is not isolated here.
These comparative outcomes are single-run **directional** evidence, not
acceptance-grade comparative performance conclusions; the run establishes that
representative batch sizes were measured.

The reference path is retained; optimization and profiling are deferred. No CPU
profiling, bottleneck identification, universal speedup, or production readiness
is established by this run.

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
Streaming and the foundation (unbounded) ordered fold are **not** faster than
the direct reference; the fold additionally costs one extra
allocation/observation because it consumes owned observations. These are
**historical** measurements of the foundation **unbounded** ordered fold, not
the RLS-local bounded API measured below.

**[new run]** A release batch-size sweep at the recorded run-time revision
`b63da4f` (clean worktree; rustc/cargo `1.97.1`;
Linux WSL2 x86_64; i7-12700H, 20 logical CPUs; `cargo bench --bench rls -j 2`)
executed the reviewed matrix `λ = 0.999`, `D ∈ {8, 32}`, `B ∈ {1, 8, 32, 128}`,
`outer = 128` (`work_per_call = OUTER`), comparing grouped
`StreamingExecutor::process_one` with
`Rls::process_batch_reference` and a separately labelled foundation-fold cloning
control. Full-horizon numerical validation passed **before** timing (`D = 8`
obs `209920`; `D = 32` obs `168960`). Within this capture, bounded-reference
per-observation cost does not change materially across `B` at either dimension;
grouped streaming and the bounded reference do not differ materially
(`D = 32, B = 1` is borderline/noisy); the foundation-fold control is clearly
measurable only at `D = 8, B = 128` (247.0 vs 227.3 ns/obs) and always costs
`6.000` allocs/obs and `+32` bytes/obs. No clearly measurable grouping benefit
is observed. The fixture is a fixed-target cyclic replay (period `2048`) on one
machine over 20 windows within the `D ∈ {8, 32}` `λ = 0.999` numerical envelope
(the full-horizon validator passed with no guard failure; larger `D` remain
subject to the `λ < 1` covariance-inflation envelope, and `D = 128` is deferred);
these are single-run directional outcomes, not acceptance-grade comparative
performance conclusions. The full table, the
separated measured/harness-derived/calculated quantities, and the provenance are
in `src/models/online/README.md` ("Release batch-size sweep").

The RLS-local bounded reference API (`Rls::process_batch_reference`) preserves
the same ordered transition semantics through direct `Rls::update` calls over
borrowed observations — not by calling the foundation fold. The `O(D²)`
candidate allocation is the RLS-specific evidence, reported rather than
optimized — a double-buffered `P` would raise an unresolved workspace-ownership
question.

The reference path is retained; optimization and profiling are deferred. No CPU
profiling, bottleneck identification, universal speedup, or production readiness
is established by this run, and the RLS production-readiness gates are
unchanged.

RLS also confirms the ownership rule: adaptive state belongs to the stream, not
the model. One immutable `&Rls` can now drive independent per-stream states both
through the foundation `process_one`/`process_stream` functions and through the
generic `StreamingExecutor`, which borrows the model immutably (`&M`) and owns
the per-stream State. The former `&mut M` limitation was removed by the
execution-ownership refactor.

### GRU production hot path (#17 / #18)

Measured evidence for the GRU production hot path. The GRU slice's matrix row
above is the bounded micro-batch work (#34), which is separate; this entry
records the production-path comparison.

**Compared paths and source version.** Historical measurements of the generic
reference `StreamingExecutor::process_one` (via the GRU's `StateModel`) against
the algorithm-local optimized `GruExecutor::process_one_optimized`. Both used
the same immutable `&Gru`, one deterministic observation per size, equivalent
zero initial hidden state, `work_per_call = 1`, and a 16-step bitwise
committed-state pre-check (`verify_paths_agree`) outside the timed region. The
benchmark source is `benches/gru.rs`, merged on `main` as commit `2346179`; the
runs were recorded on branch `issue-17-executor-ownership` (HEAD `f6ca217`).
These are **historical** measurements: they are recorded here from preserved
development records and were not reproduced for this entry.

**Provenance and methodology.** Linux x86_64; release `[profile.bench]`; Intel
Core i7-12700H; shared `benches/common` harness (counting allocator +
`Instant`). 20 measurement windows per run; warm-up `max(steps/10, 1000)`; step
counts 100,000 / 50,000 / 5,000 / 2,000 for 8×16 / 32×64 / 128×256 / 256×512.
The reported `parallelism=20` is the machine's available parallelism, not
benchmark threading; the benchmark executes single-threaded.

**Run A (`MEASURED EVIDENCE`, historical).** Each cell is median / IQR (ns)
— derived throughput (obs/s):

| size (steps) | ref `StreamingExecutor` | opt `GruExecutor` | opt allocs/obs |
|---|---|---|---|
| 8×16 (100k) | 786.8 / 12.0 — 1,270,932 | 483.3 / 16.7 — 2,069,093 | 0.000 (ref 20.000) |
| 32×64 (50k) | 5,934.8 / 221.9 — 168,498 | 5,401.9 / 238.6 — 185,119 | 0.000 |
| 128×256 (5k) | 96,243.6 / 2,033.8 — 10,390 | 95,333.5 / 1,003.6 — 10,489 | 0.000 |
| 256×512 (2k) | 432,477.3 / 5,963.5 — 2,312 | 429,823.5 / 8,836.8 — 2,327 | 0.000 |

The throughput figures are **derived** by the harness as `1e9 / median` from
its unrounded median before display formatting; they are retained as recorded,
and re-deriving from the displayed one-decimal median is subject to rounding.
Reference allocation count is recorded as 20.000 at 8×16 and stated for the
comparison as a whole; per-size reference allocation counts were not retained.

**Run B (`MEASURED EVIDENCE`, historical repeat).** An independent repeat of
the same comparison, medians only (ns):

| size | ref `StreamingExecutor` | opt `GruExecutor` |
|---|---|---|
| 8×16 | 858.4 | 520.2 |
| 32×64 | 6,014.5 | 5,536.0 |
| 128×256 | 107,396.1 | 106,612.6 |
| 256×512 | 483,218.0 | 475,052.6 |

Run B's dispersion was not retained; Run A's IQR and throughput must not be
attached to Run B's medians. The two runs are reported separately.

**Run C (`MEASURED EVIDENCE`, new release capture; supplies the previously
missing p95 and allocated bytes/observation).** A distinct release capture at
recorded run-time revision `6a38fec` (branch `rust-development`, clean working
tree, upstream `+0 -0`), command `cargo bench --bench gru -j 2`,
`[profile.bench]`; rustc/cargo `1.97.1` (LLVM `22.1.6`); Linux WSL2 x86_64;
Intel Core i7-12700H, 20 logical CPUs; 20 measurement windows. It is **not**
merged with Run A or Run B. It exercises the same per-step comparison and the
same `benches/gru.rs` per-step block; the benchmark source is unchanged between
`6a38fec` and the inspected revision `04e7d1e` (the only intervening commit is
documentation). Each cell is median / p95 / IQR (ns/obs) — derived obs/s —
allocs/obs — bytes/obs:

| size (steps) | ref `StreamingExecutor` | opt `GruExecutor` | ref allocs/obs; bytes/obs | opt allocs/obs; bytes/obs |
|---|---|---|---|---|
| 8×16 (100k) | 945.2 / 971.9 / 21.3 — 1,057,965 | 555.7 / 569.8 / 16.5 — 1,799,633 | 20.000; 1,280.0 | 0.000; 0.0 |
| 32×64 (50k) | 6,609.1 / 7,223.2 / 314.7 — 151,306 | 6,013.9 / 6,588.8 / 365.6 — 166,282 | 20.000; 5,120.0 | 0.000; 0.0 |
| 128×256 (5k) | 106,833.4 / 109,734.6 / 1,387.1 — 9,360 | 104,096.5 / 109,959.7 / 2,346.0 — 9,606 | 20.000; 20,480.0 | 0.000; 0.0 |
| 256×512 (2k) | 503,533.7 / 577,850.0 / 19,121.9 — 1,986 | 532,487.1 / 731,483.2 / 48,584.6 — 1,878 | 20.000; 40,960.0 | 0.000; 0.0 |

`allocs/obs` and `bytes/obs` are the harness's counting-allocator measurements;
`bytes/obs` is allocation/reallocation traffic, **not** live/peak memory. The
optimized path's `0.000`/`0.0` reflect its allocation-free steady state.
`p95`/`IQR` are variability across the 20 normalized measurement windows, not
per-call latency percentiles.

**Materiality (Run C; same rule).** 8×16 diff 389.5 ns (`2·max(IQR)` 42.6,
`5%·max(median)` 47.3) → **clearly measurable**; 32×64 diff 595.2 (731.2, 330.5)
→ **borderline/noisy**; 128×256 diff 2,736.9 (4,692.0, 5,341.7) → **not
materially different**; 256×512 diff 28,953.4 (97,169.2, 26,624.4) →
**borderline/noisy**. As in Run A, the latency benefit is clear only at the
smallest size; **no universal speedup** is claimed.

**Materiality (Run A; harness rule `difference > 2·max(IQR)` AND
`difference > 5%·max(median)`).** 8×16: difference 303.5 ns (2·IQR 33.4, 5%
39.3) → **clearly measurable** (~39%). 32×64: difference 532.9 ns (477.2,
296.7) → **clearly measurable** (~9%), with a narrow margin over the 2×IQR
proxy (~12%). 128×256: difference 910.1 ns (4,067.6, 4,812.2) → **not
materially different** (~0.9%). 256×512: difference 2,653.8 ns (17,673.6,
21,623.9) → **not materially different** (~0.6%) and additionally noisy across
runs. Allocation elimination (20 → 0 allocations/observation) is confirmed; the
latency benefit is clear only at small sizes. **No universal speedup is
claimed.**

**Startup (separate from steady state).** Startup figures are average ns/init
across the configured repetitions (`startup_reps()`); the measured closure
includes destruction of the value it constructs, so these are not median/IQR
timings and not pure construction latency. The retained record lists
allocations/init but not bytes/init, so bytes per initialization are omitted.

| size | `params-clone` | `params-clone+model-init` | `streaming-init` | `gru-executor-init` |
|---|---|---|---|---|
| 8×16 | 154.1 ns / 9 | 528.0 ns / 9 | 16.7 ns / 1 | 60.9 ns / 4 |
| 32×64 | 1,689.7 / 9 | 5,771.2 / 9 | 16.6 / 1 | 67.5 / 4 |
| 128×256 | 266,255.6 / 9 | 313,733.5 / 9 | 18.7 / 1 | 76.6 / 4 |
| 256×512 | 1,108,244.6 / 9 | 1,401,386.1 / 9 | 37.7 / 1 | 149.8 / 4 |

`Gru::new` adds no allocations (9 → 9): the 9 `params-clone` allocations are
the parameter tensors. `streaming-init` includes the caller's `Vector::zeros`
initial state (1 allocation); `gru-executor-init` includes the State and the
private workspace (4 allocations). All are setup, not per-observation.

**Consolidated trade-offs** (see the ownership hypothesis above and
[`src/models/recurrent/README.md`](../src/models/recurrent/README.md) §9.2-§9.3).
The immutable `&Gru` is shared by independent execution contexts; each execution
owns one authoritative State; the GRU workspace is private to the `GruExecutor`
and lives for that execution. The workspace payload is a **calculated** `3 × H`
`f32` scratch — 192 / 768 / 3,072 / 6,144 bytes at `H` = 16 / 64 / 256 / 512 —
distinct from the State payload (`H × 4` = 64 / 256 / 1,024 / 2,048 bytes) and
from container/allocator overhead, which is not measured here. The dual
reference/optimized paths carry maintenance cost, and no automatic execution
selection is implemented (#21 is design-only). Sharing an immutable model is
not evidence of tested parallel execution.

**Limitations.** Run A and Run B are historical measurements, not reproduced for
this entry, from a single machine and 20 windows; the 256×512 result varies
across runs. Run A's corrected-comparison **p95** and **allocated
bytes/observation** were not retained; **Run C records both** for the corrected
comparison and is reported separately. `GruParameters::deterministic` is a
test/benchmark fixture, not production initialization (#20). Run C is a single
capture on a shared machine and is not merged with Run A or Run B. The
allocated-bytes/observation gap previously recorded for #18 is addressed by Run
C; whether #18 is complete is a separate decision. #17 remains open.

### GRU bounded micro-batch (#34)

**Compared paths and source version.** A release capture at recorded run-time
revision `6a38fec` (branch `rust-development`, clean working tree, upstream
`+0 -0`), command `cargo bench --bench gru -j 2`, `[profile.bench]` (release).
Toolchain rustc/cargo `1.97.1` (LLVM `22.1.6`); Linux
`6.18.40.1-microsoft-standard-WSL2` x86_64; Intel Core i7-12700H, 20 logical
CPUs. `parallelism=20` is available parallelism, not benchmark threading. 20
measurement windows per cell. Three paths process the **same** prepared ordered
prefix `obs[..B]` (smaller `B` are prefixes of larger `B`), each starting from
`Vector::zeros(H)` and carrying State across calls: `bounded grouped exec` (B ×
`StreamingExecutor::process_one`), `bounded ref batch` (one
`Gru::process_batch_reference(state, &obs[..B], B)`), and `foundation
process_batch` (a **labelled control** that consumes owned observations, so its
timed body includes `iter().cloned()`). The primary comparison is grouped vs
bounded reference batch; both borrow observations and use no workspace. Cross-path
bitwise prechecks (`precheck_calls = 4`) run outside timing and passed (the run
exited `0`). `(256,512)` is excluded from this block as the optional dimension
(the existing per-step block above still covers it).

**Methodology.** `common::measure(label, steps, work_per_call = B, …)` normalises
by `steps × B`, the actual observations processed; `steps = window_budget / B`
keeps the observation count equal across paths and `B` within a dimension
(8×16 / 32×64 / 128×256 budgets 8,192 / 2,048 / 256). Reported `ns/obs`
(median, p95, IQR), `obs/s`, `allocs/obs`, and `bytes/obs` are **measured**;
`ns/batch = ns/obs × B` and `batches/s` are **harness-derived** from the same
window (amortized batch-call duration, not an individual-call latency
percentile). `p95`/`IQR` describe variability across the normalized measurement
windows, not per-call latency. `bytes/obs` is allocator allocation/reallocation
traffic, not live/peak memory.

**Measured results** (`median ns/obs`; p95 / IQR; `obs/s`; `allocs/obs`;
`bytes/obs`; derived `ns/batch`):

| dim | B | path | ns/obs | p95 | IQR | obs/s | allocs/obs | bytes/obs | ns/batch |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8×16 | 1 | grouped | 899.1 | 1012.0 | 48.7 | 1112167 | 20.000 | 1280.0 | 899.1 |
| 8×16 | 1 | bounded-ref | 948.3 | 974.3 | 28.1 | 1054542 | 20.000 | 1280.0 | 948.3 |
| 8×16 | 1 | foundation | 975.0 | 1094.1 | 74.6 | 1025667 | 21.000 | 1312.0 | 975.0 |
| 8×16 | 8 | grouped | 958.6 | 1206.3 | 89.9 | 1043208 | 20.000 | 1280.0 | 7668.7 |
| 8×16 | 8 | bounded-ref | 995.2 | 1130.9 | 100.2 | 1004821 | 20.000 | 1280.0 | 7961.6 |
| 8×16 | 8 | foundation | 975.3 | 1040.1 | 55.5 | 1025372 | 21.000 | 1312.0 | 7802.0 |
| 8×16 | 32 | grouped | 984.5 | 1039.8 | 51.2 | 1015697 | 20.000 | 1280.0 | 31505.5 |
| 8×16 | 32 | bounded-ref | 955.5 | 982.9 | 22.2 | 1046553 | 20.000 | 1280.0 | 30576.6 |
| 8×16 | 32 | foundation | 1046.3 | 1101.6 | 24.0 | 955792 | 21.000 | 1312.0 | 33480.1 |
| 8×16 | 128 | grouped | 955.6 | 973.9 | 10.4 | 1046473 | 20.000 | 1280.0 | 122315.6 |
| 8×16 | 128 | bounded-ref | 1028.6 | 1410.7 | 76.7 | 972148 | 20.000 | 1280.0 | 131667.2 |
| 8×16 | 128 | foundation | 986.8 | 1003.9 | 49.2 | 1013410 | 21.000 | 1312.0 | 126306.2 |
| 32×64 | 1 | grouped | 6836.2 | 7171.8 | 564.6 | 146280 | 20.000 | 5120.0 | 6836.2 |
| 32×64 | 1 | bounded-ref | 6268.6 | 6445.1 | 226.2 | 159526 | 20.000 | 5120.0 | 6268.6 |
| 32×64 | 1 | foundation | 6192.5 | 6881.3 | 207.6 | 161485 | 21.000 | 5248.0 | 6192.5 |
| 32×64 | 8 | grouped | 6340.4 | 6582.6 | 143.0 | 157718 | 20.000 | 5120.0 | 50723.5 |
| 32×64 | 8 | bounded-ref | 6140.0 | 6642.2 | 244.0 | 162867 | 20.000 | 5120.0 | 49119.9 |
| 32×64 | 8 | foundation | 6348.6 | 6549.7 | 416.5 | 157514 | 21.000 | 5248.0 | 50789.1 |
| 32×64 | 32 | grouped | 6166.6 | 6363.8 | 103.5 | 162164 | 20.000 | 5120.0 | 197331.3 |
| 32×64 | 32 | bounded-ref | 6229.7 | 6606.9 | 298.9 | 160522 | 20.000 | 5120.0 | 199350.1 |
| 32×64 | 32 | foundation | 6160.2 | 6547.5 | 388.6 | 162332 | 21.000 | 5248.0 | 197126.6 |
| 32×64 | 128 | grouped | 6518.2 | 7409.1 | 793.6 | 153417 | 20.000 | 5120.0 | 834325.3 |
| 32×64 | 128 | bounded-ref | 6293.0 | 6695.8 | 273.6 | 158908 | 20.000 | 5120.0 | 805499.9 |
| 32×64 | 128 | foundation | 6475.8 | 6761.1 | 141.6 | 154422 | 21.000 | 5248.0 | 828899.9 |
| 128×256 | 1 | grouped | 108407.0 | 114628.1 | 7741.8 | 9224 | 20.000 | 20480.0 | 108407.0 |
| 128×256 | 1 | bounded-ref | 108035.9 | 112999.2 | 4373.8 | 9256 | 20.000 | 20480.0 | 108035.9 |
| 128×256 | 1 | foundation | 104586.3 | 110072.2 | 4634.8 | 9561 | 21.000 | 20992.0 | 104586.3 |
| 128×256 | 8 | grouped | 108277.4 | 115883.2 | 3739.8 | 9236 | 20.000 | 20480.0 | 866218.9 |
| 128×256 | 8 | bounded-ref | 104538.7 | 112510.6 | 4873.0 | 9566 | 20.000 | 20480.0 | 836309.5 |
| 128×256 | 8 | foundation | 109281.3 | 116963.3 | 5554.3 | 9151 | 21.000 | 20992.0 | 874250.2 |
| 128×256 | 32 | grouped | 108190.3 | 111623.9 | 4628.5 | 9243 | 20.000 | 20480.0 | 3462088.1 |
| 128×256 | 32 | bounded-ref | 106441.0 | 115118.8 | 1775.4 | 9395 | 20.000 | 20480.0 | 3406113.0 |
| 128×256 | 32 | foundation | 109931.3 | 114271.1 | 5432.4 | 9097 | 21.000 | 20992.0 | 3517800.5 |
| 128×256 | 128 | grouped | 111275.4 | 117372.6 | 9976.6 | 8987 | 20.000 | 20480.0 | 14243248.0 |
| 128×256 | 128 | bounded-ref | 110645.7 | 127501.6 | 8048.4 | 9038 | 20.000 | 20480.0 | 14162648.0 |
| 128×256 | 128 | foundation | 107568.0 | 111574.6 | 3758.6 | 9296 | 21.000 | 20992.0 | 13768698.0 |

**Materiality** (harness rule `difference > 2·max(IQR)` AND
`difference > 5%·max(median)`). Primary comparison grouped vs bounded reference
batch: **no clearly measurable difference at any (dimension, B)**; differences
are *not materially different* except **borderline/noisy** at 8×16 B=1
(49.2 ns against 2·IQR 97.4 and 5% 47.4), 8×16 B=128 (73.0 vs 153.4, 51.4), and
32×64 B=1 (567.6 vs 1,129.2, 341.8). No grouping benefit or slowdown is
established. Against the foundation control, the only clearly measurable case is
8×16 B=32 (bounded-ref 955.5 vs foundation 1,046.3; 90.8 vs 48.0 and 52.3). The
foundation control, which includes observation cloning, is slower under the
materiality rule at (8,16), B=32 in this capture. Allocation counts are
B-invariant: grouped and bounded-ref are `20.000` allocs/obs and the foundation
control `21.000`; the control's extra allocator traffic is the cloned
observation (`input_dim × 4` bytes/obs = 32 / 128 / 512 at input dimension
8 / 32 / 128).

**Limitations.** Single deterministic `Lcg` fixture (128 observations per
dimension, prefix `obs[..B]`); single machine; one capture with 20 windows and no
independent repeat; the machine is shared (editor/service processes present,
lightly loaded; the benchmark itself drove the load). These are single-run
**directional** outcomes, not acceptance-grade comparative performance
conclusions. No general speedup is claimed.

The reference path is retained; optimization and profiling are deferred. This
run does not establish a CPU profile, a bottleneck identification, or production
readiness.

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
