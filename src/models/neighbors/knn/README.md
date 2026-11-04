# K-Nearest Neighbors (KNN)

> **Completeness:** this document covers the implemented read-only KNN
> prediction component. Retaining labelled reference observations can constitute
> KNN fitting, but a complete model must still demonstrate data ingestion,
> reference-set fitting and validated predictions under the
> [mandatory training-to-inference contract](../../../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory).
> Classification, distance weighting, metric configuration and approximate
> neighbor search are deferred. Report missing capabilities before proceeding.

## Purpose

The KNN model provides read-only prediction over a caller-supplied reference
set using squared-Euclidean neighbors and an unweighted mean of the `k` nearest
targets. It exposes reference prediction and a bounded micro-batch prediction
over independent queries. The canonical public path is
`models::neighbors::knn`.

## Ownership and state

`Knn` owns its immutable reference data (features and targets), `dimension`, and
construction-validated `k`. Inference takes `&self`, never mutates the model,
and needs no reusable scratch or workspace: the `O(N)` candidate buffer is a
transient local allocation inside `Knn::predict`. The `StateModel::State` is the
most recent prediction, matching the other prediction-only models. No temporal
state is introduced merely because Seqvex is streaming-first.

## Contract

Construction validates and stores labelled reference observations. The first
task variant is **regression** (classification is deferred). With `N`
references, `k` neighbors, and query `x`, the model selects the first `k`
references under `(squared_distance, reference_index)` ascending and returns the
unweighted mean of their targets. `sqrt` is strictly increasing on `[0, ∞)`, so
squared distance orders neighbors identically to Euclidean distance; it is used
because only rank matters for an unweighted mean (a future distance-weighted
rule would need `sqrt(D)`). Construction validates the dimension
(`ZeroDimension`), a non-empty reference set (`NoReferences`), `1 <= k <= N`
(`InvalidK`), reference dimensions (`DimensionMismatch`), and reference
finiteness (`NonFiniteReference`). Query validation is dimension first, then
finiteness (`NonFiniteInput`); the output is not re-checked. Equal distances and
duplicate references are deterministic via the reference-index tiebreak and are
never collapsed. `predict_batch` preserves input order; the first element
failure returns `Err` for the whole call with no partial output.

## Micro-batch

Queries are independent, so the bounded micro-batch is a model-local,
order-preserving capability over a caller-bounded slice, not a generic
micro-batch executor. The first element failure fails the whole call with no
partial output.

## Numerical envelope

The `f32` substrate is reused; no `f64` distance accumulation is introduced.
Finite references are required at construction; a query is validated for
dimension then finiteness; the output is not validated. Squared distances are
non-negative and can overflow to `+inf` for large finite inputs (multiple `+inf`
distances then tie and are ordered by reference index); distances can underflow
to `0.0` for very small values, which likewise tie deterministically. These are
expected IEEE-754 `f32` behaviours, and the caller is responsible for keeping
magnitudes representable.

## Missing capabilities

- KNN classification, distance-weighted prediction, configurable metrics, and
  any approximate or indexed neighbor search (ANN, KD-tree, ball tree, HNSW);
  the slice is brute-force regression search only.
- A generalized nearest-neighbor container, storage/tensor abstraction or
  micro-batch executor is out of scope.
- This namespace migration changes no training, precision, optimization or
  performance behavior.

## Tests and benchmark

- `tests/knn.rs` — public-contract integration tests.
- `benches/knn.rs` — prediction benchmark, Cargo target `knn`.
- No benchmark measurement is performed by the namespace migration; benchmark
  verification is compilation-only.

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

## Deferred decisions

- The numerical policy (`f32` substrate versus the `f64` statistics layer): the
  model uses the existing `f32` substrate because observations are already
  `Observation<Vector>`; a switch requires a measured requirement.
- Classification, distance weighting, configurable metrics and approximate
  neighbor search remain separate work.
- No optimization or production-readiness claim is made.

## References

- [Model-family index](../../README.md)
- [OLS canonical document](../../linear/ols/README.md)
- [Decision Tree canonical document](../../trees/decision_tree/README.md)
- [Vertical-slice evidence](../../../../docs/ML_VERTICAL_SLICES.md)
- [Historical algorithm correctness audit](../../../../docs/ALGORITHM_CORRECTNESS_AUDIT.md)
