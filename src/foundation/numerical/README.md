# Numerical

> **Failure boundary:** the documented primitive panic/IEEE behavior below must not be assumed to be an acceptable model-level recovery policy. Model designs must declare validated numerical limits and raise unresolved failure cases to the maintainer under the [failure decision rule](../../../docs/DEVELOPMENT.md#calculation-failures-and-error-handling--mandatory). This documentation update does not change primitive APIs or authorize silent numeric substitution.

## Purpose

Owns the minimal numerical computation required by Seqvex's own ML/RL methods,
not a general-purpose numerical or scientific-computing ecosystem.

## Responsibility

- `statistics` — the established `f64` primitives: `sum`, `mean`, `variance`,
  `dot`, `norm`, `online_mean`, `online_variance`.
- `vector` — contiguous `f32` vectors with elementwise `add`, `multiply`,
  `scale`, `complement`, `map`.
- `linalg` — dense row-major `f32` matrices and the affine transform `y = W x`.
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
- The `f32` substrate returns `Result` with `DimensionMismatch` for shape
  errors. The GRU adds `GruError` for its own failure classes.

The `f32`/`f64` split is intentional and must not be silently unified.

## Inside

- `statistics`, `vector`, `linalg`, `activations`.

## Outside

- General linear algebra, matrix decompositions, tensors, broadcasting,
  dataframes, ETL, EDA, autodiff, GPU kernels.
- Optimized, vectorized, or hardware-accelerated implementations until a
  benchmark justifies them.

## Current tests

`tests/numerical.rs` covers normal, edge, dimension, and reference behavior for
both layers.

## Major deferred decisions

- Rolling-window semantics (window boundary, update order) are not yet clear
  enough to fix a contract.
- Whether the primitives remain free functions, gain a trait, or delegate to a
  mature numerical crate.
- SIMD / transposed or packed matrix layouts are deferred until measured.
