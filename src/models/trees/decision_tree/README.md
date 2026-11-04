# Decision Tree

> **Completeness:** this document covers the implemented read-only Decision Tree
> inference component. It does not satisfy complete-model acceptance by itself:
> training/fitting from data remains required. Every model must ingest data,
> fit/learn, validate and predict from the learned result under the
> [mandatory training-to-inference contract](../../../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory).
> Classification is deferred. Report missing capabilities before proceeding.

## Purpose

The Decision Tree model provides read-only traversal of a caller-supplied
trained regression tree, its reference prediction, and a bounded micro-batch
prediction over independent observations. The canonical public path is
`models::trees::decision_tree`.

## Ownership and state

Nodes are immutable once constructed. Prediction needs no mutable model state
and no reusable scratch, so the model implements `StateModel` with a `&self`
contract and is freely shareable. Traversal is iterative with a step bound, so a
malformed graph cannot recurse without bound. The `StateModel::State` is the
most recent prediction, matching the other prediction-only models.

## Contract

The current primitive traverses a tree supplied by the caller; it does not
implement fitting from data. The first task variant is **regression** (finite
`f32` leaves); classification is deferred. A split routes **left** when
`feature <= threshold` and right otherwise. Construction validates the graph
(empty/zero-dimension, feature index, child range, finite threshold and leaf,
cycle, shared child, unreachable node), and traversal is iterative with a step
bound. Because a prediction returns a validated leaf value with no arithmetic,
its output is always finite. `predict_batch` preserves input order; the first
element failure makes the entire call return `Err` and no partial output vector
is returned.

## Micro-batch

Prediction may be micro-batched because each output depends only on its own
observation and the immutable tree. The batch must preserve input order and
must produce bitwise-identical results to repeated single-observation
prediction. No micro-batch path may silently change the traversal or output
order.

## Numerical envelope

Feature vectors use the existing `f32` substrate. Construction rejects
non-finite thresholds and leaves, an out-of-range feature index, out-of-range
children, cycles, shared children, and unreachable nodes. Prediction validates
the feature dimension first, then feature finiteness, and does not re-validate
the output. Because prediction returns a validated leaf directly, it performs no
arithmetic and is always finite.

## Missing capabilities

- Decision Tree training/fitting is missing from the read-only component and is
  required for complete model delivery. Classification is a separate task;
  neither this note nor historical evidence approves implementation.
- A generic tree/arena/storage abstraction or a micro-batch executor is out of
  scope.
- This namespace migration changes no training, precision, optimization or
  performance behavior.

## Tests and benchmark

- `tests/decision_tree.rs` — public-contract integration tests.
- `benches/decision_tree.rs` — prediction benchmark, Cargo target
  `decision_tree`.
- No benchmark measurement is performed by the namespace migration; benchmark
  verification is compilation-only.

## Measured evidence (#24)

Two separate runs are recorded here and are not merged: a historical pre-sweep
baseline ([historical]) and the release batch-size sweep at the **associated
inspected revision** `b63136f5` ([new run]). The actual run-time revision,
compiler toolchain, and machine-contention status are **unverified**: the
supplied output contains no revision or toolchain evidence and no run-time
contention evidence.

### Historical baseline ([historical])

Release mode (`cargo bench --bench decision_tree`), median of 20 runs. Balanced
trees: depth 6 (127 nodes) at 8 features, depth 8 (511 nodes) at 32 features,
depth 10 (2047 nodes) at 128 and 256 features.

| features | depth | nodes | reference | streaming | micro-batch |
|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 127 | 15.3 | 15.9 | 20.5 |
| 32 | 8 | 511 | 31.9 | 32.3 | 35.7 |
| 128 | 10 | 2047 | 79.5 | 78.1 | 83.7 |
| 256 | 10 | 2047 | 108.4 | 106.7 | 114.5 |

ns per observation. Reference and streaming allocate `0.000` allocations and `0`
bytes per observation; the bounded micro-batch allocates exactly one output
`Vec` per batch (amortized `1/batch_size` per observation). Streaming's measured
median is within 1.7 ns/obs of the direct reference (see table); the benchmark
applies no materiality test, so this is a reported measured difference, not a
statistical-equivalence claim. Micro-batch is not faster than single-observation
prediction. Construction builds the growing node `Vec` (11–16 heap allocations at
these tree sizes), separate from the allocation-free per-observation inference
and the per-batch output allocation. No optimization was implemented.

### Release batch-size sweep ([new run], associated inspected revision `b63136f5`)

Release run `cargo bench --bench decision_tree -j 2` at the **associated
inspected revision** `b63136f5b5aa19f03a4c34ab7b4c4b863195ed40` (branch
`rust-development`), `env os=linux arch=x86_64 profile=release parallelism=20`,
20 normalized measurement windows per cell, single machine (12th Gen Intel Core
i7-12700H, 20 logical CPUs). The actual run-time revision, compiler toolchain,
and machine-contention status are **unverified** (the supplied output contains no
revision or toolchain evidence and no run-time contention evidence). Step
budgets: 200000 / 200000 / 50000 / 20000 for 8 / 32 / 128 / 256 features.

Fixture limitation: the sweep repeats one deterministic observation to a
constant length (`128`) for every batch size, so it measures **repeated-input**
behavior, not input-distribution performance; the traversal workload is held
constant across `B`.

Control paths for this run, ns per observation (median):

| features | reference | streaming | micro-batch |
|---:|---:|---:|---:|
| 8 | 16.4 | 18.8 | 22.3 |
| 32 | 33.6 | 33.6 | 34.8 |
| 128 | 81.0 | 81.2 | 84.5 |
| 256 | 113.9 | 112.8 | 115.2 |

Initialization for this run: `1778.8` ns/init (`11` allocs, `8655` bytes) at 8
features; `5734.4` (`14`, `34127`) at 32; `21052.9` (`16`, `133967`) at 128;
`21105.5` (`16`, `133967`) at 256.

Micro-batch sweep, all 16 rows. `median`, `p95`, `IQR`, `obs/s`, `allocs/obs`,
and `bytes/obs` are the harness's measured values; `p95`/`IQR` describe
variability across the normalized measurement windows, **not** per-call or event
latency. `bytes/obs` is allocator-counted allocation/reallocation traffic, **not**
live/peak memory. The per-batch duration and traffic (`derived ns/batch`,
`derived bytes/batch`) are **harness-derived** from the unrounded median and
allocation values (approximately the displayed `median × B` and
`bytes/obs × B`; small differences are display rounding). `capacity` is the
output `Vec` capacity **observed during the precheck** (outside timing), and
`payload bytes` is the **calculated** logical payload `B × size_of::<f32>()`.

| features | B | median ns/obs | p95 | IQR | obs/s | allocs/obs | bytes/obs | capacity | payload bytes | derived ns/batch | derived bytes/batch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 41.7 | 53.5 | 1.9 | 23981089 | 1.000 | 16.0 | 4 | 4 | 41.7 | 16.0 |
| 8 | 8 | 24.9 | 28.8 | 2.2 | 40163642 | 0.250 | 6.0 | 8 | 32 | 199.2 | 48.0 |
| 8 | 32 | 23.8 | 26.4 | 1.3 | 42090393 | 0.125 | 7.5 | 32 | 128 | 760.3 | 240.0 |
| 8 | 128 | 21.2 | 30.2 | 1.1 | 47176487 | 0.047 | 7.9 | 128 | 512 | 2713.2 | 1008.0 |
| 32 | 1 | 54.1 | 65.3 | 1.9 | 18479746 | 1.000 | 16.0 | 4 | 4 | 54.1 | 16.0 |
| 32 | 8 | 37.6 | 40.2 | 0.8 | 26594186 | 0.250 | 6.0 | 8 | 32 | 300.8 | 48.0 |
| 32 | 32 | 33.7 | 34.8 | 0.3 | 29653043 | 0.125 | 7.5 | 32 | 128 | 1079.1 | 240.0 |
| 32 | 128 | 31.8 | 32.2 | 0.5 | 31436933 | 0.047 | 7.9 | 128 | 512 | 4071.6 | 1008.0 |
| 128 | 1 | 104.8 | 110.1 | 3.5 | 9545797 | 1.000 | 16.0 | 4 | 4 | 104.8 | 16.0 |
| 128 | 8 | 84.4 | 86.3 | 1.5 | 11847454 | 0.250 | 6.0 | 8 | 32 | 675.3 | 48.0 |
| 128 | 32 | 83.0 | 86.2 | 1.1 | 12044361 | 0.125 | 7.5 | 32 | 128 | 2656.8 | 240.0 |
| 128 | 128 | 82.3 | 89.6 | 2.6 | 12143855 | 0.047 | 7.9 | 128 | 512 | 10540.3 | 1008.0 |
| 256 | 1 | 131.8 | 134.5 | 1.0 | 7589092 | 1.000 | 16.0 | 4 | 4 | 131.8 | 16.0 |
| 256 | 8 | 118.5 | 124.2 | 4.7 | 8436286 | 0.250 | 6.0 | 8 | 32 | 948.3 | 48.0 |
| 256 | 32 | 115.4 | 119.3 | 3.8 | 8665361 | 0.125 | 7.5 | 32 | 128 | 3692.9 | 240.0 |
| 256 | 128 | 109.9 | 117.2 | 1.2 | 9101509 | 0.047 | 7.9 | 128 | 512 | 14063.6 | 1008.0 |

Applying the harness materiality rule (`difference > 2·max(IQR)` AND
`difference > 5%·max(median)`):

- Against the direct reference, `micro-batch` is materially slower only at 8
  features (`+5.9` ns/obs; thresholds `3.8` and `1.1`) and is not materially
  different at 32, 128, or 256 features.
- `streaming` versus the direct reference: the difference is not material at 32,
  128, and 256 features. At 8 features, the materiality classification is
  unresolved at displayed precision: the reported difference (`+2.4` ns/obs)
  equals the reported `2·max(IQR)` threshold (`2.4` ns/obs).
- Within the repeated-input fixture, the per-observation median at `B = 128` is
  materially lower than at `B = 1` in all four configurations (differences
  ≈ `20.5`–`22.5` ns/obs against `2.4`–`7.0` ns/obs and `2.1`–`6.6` ns/obs
  thresholds). `allocs/obs` and allocator `bytes/obs` also fall with `B`
  (`1.000 → 0.047` and `16.0 → 7.9`). This run does not decompose traversal from
  per-batch output handling, so the cause of the per-observation reduction is not
  isolated here.

These comparative outcomes are single-run **directional** evidence; they are not
acceptance-grade comparative performance conclusions. The run does establish that
representative batch sizes were measured.

The reference path is retained; optimization and profiling are deferred. This
run does not establish a CPU profile, a bottleneck identification, a universal
speedup, or production readiness.

## Deferred decisions

- The numerical policy (`f32` substrate versus the `f64` statistics layer): the
  model uses the existing `f32` substrate because observations are already
  `Observation<Vector>`; a switch requires a measured requirement.
- Training/fitting and classification remain separate work.
- No optimization or production-readiness claim is made.

## References

- [Model-family index](../../README.md)
- [OLS canonical document](../../linear/ols/README.md)
- [Vertical-slice evidence](../../../../docs/ML_VERTICAL_SLICES.md)
- [Historical algorithm correctness audit](../../../../docs/ALGORITHM_CORRECTNESS_AUDIT.md)
