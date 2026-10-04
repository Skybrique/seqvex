# Classic ML

## Purpose

Holds the classic (non-recurrent) ML vertical slices. The first is linear
regression prediction; the second is decision tree prediction; the third is
K-nearest-neighbors prediction.

## Responsibility

- `linear_regression` — the immutable model `ŷ = w · x + b`, its reference
  prediction, and a bounded micro-batch prediction over independent
  observations.
- `decision_tree` — read-only traversal of a caller-supplied trained regression
  tree, its reference prediction, and a bounded micro-batch prediction over
  independent observations.
- `knn` — read-only prediction over a caller-supplied reference set
  (squared-Euclidean neighbors, unweighted mean of the `k` nearest targets), its
  reference prediction, and a bounded micro-batch prediction over independent
  queries.

## Relationship to Seqvex

This module is the architectural **control case** for the current ownership
review. Unlike the GRU slice, linear-regression prediction:

- needs no mutable model state;
- needs no reusable scratch/workspace;
- is therefore shareable behind `&self`, with no `&mut M` requirement.

Its characteristics are evidence for the hypothesis that immutable, shareable
model data and per-stream state/scratch should be separate concerns. It must not
be made to conform to a model-owned-workspace topology.

The former generic `StreamingExecutor` `&mut M` requirement was removed by the
execution-ownership refactor: the executor now borrows `&Model`, so one
immutable model can drive several independent execution contexts. Linear
regression, which needs no mutable model state at all, was direct evidence that
the `&mut M` requirement was broader than any individual model's needs and
belonged to the executor/workspace review, not to this slice.

## Inside

- `LinearRegression` and `RegressionError`.
- Reference prediction (`predict`).
- Streaming prediction via the foundation `StateModel` contract (the carried
  state is the most recent prediction).
- Bounded micro-batch prediction (`predict_batch`), valid because prediction is
  order-independent across observations.
- `DecisionTree`, `DecisionTreeNode`, and `DecisionTreeError` — read-only
  traversal of a model-local `Vec` of nodes (provisional layout), immutable
  `&self`, no scratch, with the same `State = f32` "most recent prediction"
  convention and model-local micro-batch.
- `Knn`, `Reference`, and `KnnError` — immutable model-owned reference data
  (features + targets) with a construction-validated `k`, squared-Euclidean
  distance, deterministic `(distance, index)` neighbor ordering, an unweighted
  mean over `f32` targets, and the same `State = f32` convention and model-local
  micro-batch. The `O(N)` candidate buffer is transient per-query scratch, not
  model-owned.

## Decision Tree (read-only inference)

Inference traverses a trained tree supplied by the caller; training/fitting is
deliberately out of scope. The first task variant is **regression** (finite `f32`
leaves); classification is deferred. A split routes **left** when
`feature <= threshold` and right otherwise. Construction validates the graph
(empty/zero-dimension, feature index, child range, finite threshold and leaf,
cycle, shared child, unreachable node), and traversal is iterative with a step
bound. Because a prediction returns a validated leaf value with no arithmetic,
its output is always finite. `predict_batch` preserves input order; the first
element failure makes the entire call return `Err` and no partial output vector
is returned.

## K-Nearest Neighbors (stored-reference inference)

Prediction uses a caller-supplied reference set; training/fitting is deliberately
out of scope and the first task variant is **regression** (classification is
deferred). With `N` references, `k` neighbors, and query `x`, the model selects
the first `k` references under `(squared_distance, reference_index)` ascending
and returns the unweighted mean of their targets. `sqrt` is strictly increasing
on `[0, ∞)`, so squared distance orders neighbors identically to Euclidean
distance; it is used because only rank matters for an unweighted mean (a future
distance-weighted rule would need `sqrt(D)`). Construction validates the
dimension (`ZeroDimension`), a non-empty reference set (`NoReferences`),
`1 <= k <= N` (`InvalidK`), reference dimensions (`DimensionMismatch`), and
reference finiteness (`NonFiniteReference`). Query validation is dimension first,
then finiteness (`NonFiniteInput`); the output is not re-checked. Equal
distances and duplicate references are deterministic via the reference-index
tiebreak and are never collapsed. `predict_batch` preserves input order; the
first element failure returns `Err` for the whole call with no partial output.

## Outside

- Online parameter updating, training, optimizers, and autodiff. Prediction is
  deliberately separate from learning; the online update is a later, ordered
  concern (RLS, Issue #26).
- Decision Tree training/fitting and classification; the slice is read-only
  regression traversal until those are explicitly approved.
- KNN classification, distance-weighted prediction, configurable metrics, and
  any approximate or indexed neighbor search (ANN, KD-tree, ball tree, HNSW); the
  slice is brute-force regression search only.
- A generic linear-model trait, a generalized `predict` trait, a tree/arena/
  storage abstraction, a generalized nearest-neighbor container, or a micro-batch
  executor.
- Storage/tensor abstractions.

## Current tests / specification

- `tests/linear_regression.rs`
- `tests/decision_tree.rs`
- `tests/knn.rs`

## Micro-batch policy

Prediction may be micro-batched because each output depends only on its own
observation and the immutable weights. The batch must preserve input order and
must produce bitwise-identical results to repeated single-observation
prediction. No micro-batch path may silently change the reduction order of the
per-observation dot product.

## Numerical envelope

Prediction runs on the framework's `f32` substrate. Construction rejects
non-finite weights and bias (`NonFiniteParameter`), and prediction rejects a
dimension mismatch and non-finite features (`NonFiniteInput`). Prediction does
**not** validate its output.

Because the operands are `f32`, finite operands can still overflow during
multiplication or summation and produce `±inf`; a subsequent `inf + (-inf)` can
produce `NaN`. Finite parameters and finite inputs therefore do **not** guarantee
a finite prediction. Callers are responsible for keeping the products and sums
representable, for example by scaling features and coefficients. No numeric
threshold is imposed or implied: this is expected IEEE-754 `f32` behaviour, not a
defect, and Seqvex does not invent a magnitude cutoff.

The analogous pure-prediction path `Rls::predict` computes `wᵀx` with the same
no-output-validation behaviour; each model documents its own envelope rather than
sharing a rule or abstraction.

## Linear regression applicability

Linear regression in this slice is fixed-coefficient prediction: `ŷ = w·x + b`
is evaluated against caller-supplied immutable coefficients. It performs no
fitting, adaptation, or parameter update. Evaluating the prediction equation
does not require inputs to be IID, stationary, or exchangeable; each prediction
depends only on its own observation and the fixed coefficients. How the
coefficients were estimated, predictive validity, uncertainty, and
distribution-shift behaviour are separate questions outside this slice. The
f32 numerical envelope above is unchanged: finite parameters and finite inputs
do not guarantee a finite output, and callers keep products and sums
representable.

## Measured evidence (#23)

Release-mode benchmark (`cargo bench --bench linear_regression`), median of 20
runs, ns per observation:

| features | micro-batch | reference | streaming | micro-batch |
|---|---:|---:|---:|---:|
| 8 | 32 | 7.6 | 6.6 | 6.9 |
| 32 | 32 | 20.9 | 15.0 | 18.5 |
| 128 | 16 | 87.1 | 88.0 | 92.8 |

Reference and streaming allocate nothing; micro-batch allocates one output
`Vec` per batch (amortized `1/batch_size` per observation). Micro-batch is not
faster than single-observation prediction and is materially slower at
`features = 128` (effect ≈ 16× the single-observation IQR). No optimization was
implemented, per the slice sequence (no profile-identified bottleneck beyond the
inherent output allocation).

## Measured evidence (#24)

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
and the per-batch output allocation. No optimization was implemented: there is no
profile-identified bottleneck.

## Measured evidence (#25)

Release mode (`cargo bench --bench knn`), median of 20 runs. One factor is varied
at a time; `scan-only` is a distance-scan control that does not select or sort.

`N` sweep (`d = 8`, `k = 4`):

| N | reference | scan-only | bytes/query |
|---:|---:|---:|---:|
| 1 024 | 16.0 µs | 2.4 µs | 16 384 |
| 16 384 | 425.2 µs | 40.2 µs | 262 144 |
| 65 536 | 2 032.9 µs | 163.4 µs | 1 048 576 |
| 262 144 | 9 795.0 µs | 1 475.5 µs | 4 194 304 |

`d` sweep (`N = 8 192`, `k = 4`):

| d | reference | scan-only |
|---:|---:|---:|
| 8 | 176.2 µs | 20.0 µs |
| 32 | 228.4 µs | 73.2 µs |
| 128 | 538.3 µs | 378.0 µs |
| 256 | 1 136.2 µs | 965.0 µs |

`k` sweep (`N = 8 192`, `d = 32`): reference is essentially flat across
`k ∈ {1, 8, 32, N}` (229–232 µs), because selection cost is the full sort, not
`k`.

`B` sweep (`N = 2 048`, `d = 8`, `k = 4`): bounded micro-batch, in input order.

| B | micro-batch | allocs/obs |
|---:|---:|---:|
| 1 | 33.7 µs | 2.000 |
| 8 | 46.4 µs | 1.250 |
| 32 | 49.0 µs | 1.125 |
| 128 | 49.4 µs | 1.047 |

Unlike LR/DT, the current **full-sort reference implementation** is not
allocation-free: `predict` materialises all `N` candidates in one transient
`Vec<(f32, usize)>` (`O(N)` scratch, 16 bytes per reference per query —
`bytes/query = 16·N`), which is the scratch the slice profile anticipates; the
`scan-only` control allocates nothing, so the allocation is the candidate buffer,
not the distance scan. Streaming matches the direct reference (same `predict`
path). The `scan-only` difference localises the baseline cost: at
`N = 262 144, d = 8` the full sort dominates (9.80 ms vs 1.48 ms scan), while at
`N = 8 192, d = 256` distance computation dominates (1.14 ms vs 0.97 ms scan).
Construction is `N + 1` allocations (`N` reference `Vector`s plus the outer
`Vec`); at the measured sizes construction is 31.8 µs (`N = 1 024`) to 17.0 ms
(`N = 262 144`). The bounded micro-batch preserves input order but is not
allocation-free either: it adds one output `Vec` that grows through
reallocations, so the measured allocator calls per observation are B-dependent —
2.000, 1.250, 1.125, and 1.047 for `B = 1, 8, 32, 128` (one output allocation at
`B = 1`, with +1, +2, +4, +6 further output reallocations at `B = 8, 32, 128`).
It is not faster than single-query prediction (33.7 µs vs 49.4 µs at `B = 128`).
This allocation behaviour is a property of the current reference implementation,
not an intrinsic requirement of all KNN implementations. No optimization was
implemented:
the full-sort selection is the obvious future target, but it requires the
existing correctness evidence to remain and a measured decision, per the slice
lifecycle.

## Major deferred decisions

- The numerical policy for classic ML (`f32` substrate vs the `f64` statistics
  layer). This slice uses the existing `f32` substrate because observations are
  already `Observation<Vector>`; a switch needs a measured requirement.
- Whether online/adaptive parameter updates belong in this module or the online
  ML namespace.
