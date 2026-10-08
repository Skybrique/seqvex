# Linear models

> **Completeness:** the existing parameter-based predictor is an incomplete component. Complete linear models must ingest data, fit/learn, validate and predict using the learned result. Preset-parameter examples do not satisfy the [mandatory training-to-inference contract](../../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory); no working end-to-end OLS model is claimed before fitting and evidence are delivered.

## Purpose

Holds the linear-model family. The first slice is ordinary least-squares (OLS)
regression, relocated here from `models::classic` as a behavior-preserving move;
the classic import paths remain as compatibility re-exports.

## Responsibility

- [`ols`](ols/README.md) — the canonical OLS algorithm document: requirements,
  issue owners and pending design questions; separate Architect design and
  Planner implementation plan remain required.

## Implemented versus planned

Implemented (relocated unchanged from
`src/models/classic/linear_regression.rs` to `ols/predict.rs`):

- The immutable supplied-parameter predictor `ŷ = w · x + b`, with reference
  `predict`, bounded `predict_batch`, and streaming via the foundation
  `StateModel` contract (state = most recent prediction).
- A runnable supplied-parameter example:
  [`ols/examples/predict.rs`](ols/examples/predict.rs)
  (`cargo run --example ols_predict`).

Planned under [#23](https://github.com/Skybrique/seqvex/issues/23), not yet
present:

- OLS fitting and rank/numerical diagnostics, the minimal versioned artifact
  contract, and candidate acceptance/recovery. See the canonical
  [OLS README](ols/README.md).

Out of scope until separately approved: ridge, lasso, logistic classification,
weighted/sparse/multi-output fitting, and any generic linear-model trait.

## Status

Prediction behavior is preserved bitwise: the reduction order, validation order,
error values and documented `f32` overflow envelope are unchanged. The fitting
design is a **proposal** requiring Maintainer review before implementation.
