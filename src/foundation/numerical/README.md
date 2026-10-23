# Numerical

> **Failure boundary:** the documented primitive panic/IEEE behavior below must not be assumed to be an acceptable model-level recovery policy. Model designs must declare validated numerical limits and raise unresolved failure cases to the maintainer under the [failure decision rule](../../../docs/DEVELOPMENT.md#calculation-failures-and-error-handling--mandatory). This documentation update does not change primitive APIs or authorize silent numeric substitution.

## Purpose

Owns the minimal numerical computation required by Seqvex's own ML/RL methods,
not a general-purpose numerical or scientific-computing ecosystem.

## Responsibility

- `statistics` — the established `f64` primitives: `sum`, `mean`, `variance`,
  `dot`, `norm`, `online_mean`, `online_variance`.
- `vector` — concrete contiguous `f32` and `f64` vectors with elementwise
  `add`, `multiply`, `scale`, `complement`, `map`, and ordered `dot`.
- `linalg` — dense row-major `f32` and `f64` matrices and the transform `y = W x`.
- `kernels` — private loop helpers shared only by equivalent f32/f64 operations.
- `activations` — scalar `sigmoid` and `tanh`.

## Error semantics

Two numeric layers with deliberately different contracts:

- `statistics` is the existing `f64` contract and fails by **panic** on
  undefined input (empty mean/variance/norm, mismatched `dot`, inconsistent online
  statistics). `sum`, `mean`, and `dot` propagate IEEE-754 `NaN`/`±Inf`.
  `variance` asserts a non-negative result, so any non-finite input yields a
  `NaN` result and panics. `norm` also asserts non-negativity: a `NaN` input
  panics, while a `±Inf` input returns `+Inf`. `sum(&[]) == 0.0`. Variance is
  the **population** variance (divide by `N`), computed in two passes.
- Both typed container families return `Result` with `DimensionMismatch` for
  shape errors. Model-specific failure classes remain model-local; the GRU adds
  `GruError` for its own failure classes.

The `f32`/`f64` container types remain concrete and separate. Private shared
kernels do not merge their public APIs, the statistics contract, or algorithm
mathematics. Adding f64 containers does not implement any model precision path,
training method, preprocessing pipeline, or profile selector.

## Inside

- `statistics`, `vector`, `linalg`, private `kernels`, `activations`.

## Outside

- General linear algebra, matrix decompositions, tensors, broadcasting,
  dataframes, ETL, EDA, autodiff, GPU kernels.
- Optimized, vectorized, or hardware-accelerated implementations until a
  benchmark justifies them.

## Current tests

`tests/numerical.rs` covers established statistics and f32 compatibility
controls. `tests/numerical_f64.rs` covers the additive f64 container APIs with
analytical values and native-precision discrimination.

## Major deferred decisions

- Rolling-window semantics (window boundary, update order) are not yet clear
  enough to fix a contract.
- Whether the primitives remain free functions, gain a trait, or delegate to a
  mature numerical crate.
- SIMD / transposed or packed matrix layouts are deferred until measured.
